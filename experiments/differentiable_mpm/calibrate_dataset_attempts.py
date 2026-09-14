"""Run a sequential Cartesian grid of shared-material dataset calibrations."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time
import types
from typing import Callable, TextIO, cast
import uuid

import numpy as np

from .results import RUN_ROOT, SCHEMA as RUN_SCHEMA, RunStore, canonical_hash, source_identity


MATERIAL_NAMES = ("youngs_modulus", "poisson_ratio", "viscosity", "plastic_min", "plastic_max")
CONTACT_NAMES = ("floor_retention", "tool_friction_coefficient", "tool_stickiness")
DATASET_V2 = "taichidough/differentiable-dataset/v2"


SCHEMA = "taichidough/dataset-calibration-attempts/v1"
SUMMARY_SCHEMA = "taichidough/dataset-calibration-attempt-summary/v1"
ATTEMPT_SCHEMA = "taichidough/dataset-calibration-attempt/v1"
INVOCATION_SCHEMA = "taichidough/dataset-calibration-invocation/v1"
BEST_SCHEMA = "taichidough/dataset-calibration-best-attempt/v1"
SUCCESS_STATUSES = {
    "converged_gradient",
    "converged_parameters",
    "stalled_step",
    "evaluation_budget_exhausted",
    "budget_exhausted",
}


@dataclass(frozen=True)
class _HostLossConfig:
    """Validation-only copy of the partial-view loss settings, without Taichi."""
    footprint_radius: int = 2
    visibility_temperature_m: float = 0.005
    opacity_gain: float = 8.0
    visible_cutoff_temperatures: float = 2.0
    depth_scale_m: float = 0.005
    distance_scale_m: float = 0.005
    huber_delta: float = 1.0
    depth_weight: float = 1.0
    coverage_weight: float = 0.5
    distance_weight: float = 0.5
    min_observed_pixels: int = 4
    min_depth_pixels: int = 1
    min_predicted_pixels: int = 1
    max_observed_points: int = 1024
    missing_depth_residual: float = 4.0
    version: str = "partial-visible-splats-v2"

    def __post_init__(self):
        if self.version not in ("partial-visible-splats-v1", "partial-visible-splats-v2"):
            raise ValueError(f"Unsupported training loss version: {self.version}")
        positive = ("visibility_temperature_m", "opacity_gain", "visible_cutoff_temperatures",
                    "depth_scale_m", "distance_scale_m", "huber_delta", "missing_depth_residual")
        if any(not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0 for name in positive):
            raise ValueError("Loss scales and temperatures must be finite and positive")
        weights = (self.depth_weight, self.coverage_weight, self.distance_weight)
        if not np.isfinite(weights).all() or min(weights) < 0 or sum(weights) <= 0:
            raise ValueError("Loss weights must be finite, nonnegative and not all zero")
        for name in ("min_observed_pixels", "min_depth_pixels", "min_predicted_pixels", "max_observed_points"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.footprint_radius, int) or not 1 <= self.footprint_radius <= 8:
            raise ValueError("footprint_radius must be an integer in [1,8]")

    def as_dict(self):
        return asdict(self)


def _load_dataset_host_only(path, **options):
    """Load the existing dataset validator without importing its Taichi loss implementation."""
    module_name = "experiments.differentiable_mpm.loss"
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)
        setattr(module, "LOSS_VERSION", "partial-visible-splats-v2")
        setattr(module, "SUPPORTED_LOSS_VERSIONS", ("partial-visible-splats-v1", "partial-visible-splats-v2"))
        setattr(module, "LossConfig", _HostLossConfig)
        sys.modules[module_name] = module
    from .dataset_config import load_dataset
    return load_dataset(path, **options)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _unique_values(values, name):
    if not values:
        raise ValueError(f"At least one {name} is required")
    result = []
    for raw in values:
        value = _finite(raw, name)
        if value in result:
            raise ValueError(f"Duplicate {name}: {value}")
        result.append(value)
    return result


def _resolve_python(value):
    candidate = shutil.which(str(value))
    if candidate is None:
        expanded = Path(value).expanduser()
        candidate = os.path.abspath(expanded) if expanded.exists() else None
    if candidate is None:
        raise ValueError(f"Python executable was not found: {value}")
    # Preserve an explicitly selected virtual-environment path instead of
    # resolving its symlink to the system interpreter.
    path = Path(os.path.abspath(candidate))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Python executable is not an executable file: {path}")
    return path


def _path_options(items):
    nested = {}
    normalized = []
    for text in items:
        try:
            key, raw_path = text.split("=", 1)
            episode, name = key.rsplit(".", 1)
        except ValueError as error:
            raise ValueError("Path overrides require EPISODE.INPUT=PATH") from error
        if not episode or not name or not raw_path or name in nested.get(episode, {}):
            raise ValueError("Empty or duplicate episode path override")
        path = Path(raw_path).expanduser().resolve()
        nested.setdefault(episode, {})[name] = str(path)
        normalized.append(f"{episode}.{name}={path}")
    return nested, normalized


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--initial-youngs-modulus", type=float, action="append", required=True)
    parser.add_argument("--initial-viscosity", type=float, action="append", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--path", action="append", default=[], metavar="EPISODE.INPUT=PATH")
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"))
    parser.add_argument("--precision", choices=("f32", "f64"))
    parser.add_argument("--physics-version")
    parser.add_argument("--p2g-mode")
    parser.add_argument("--segment-length", type=int)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--reference-policy", choices=("strict", "frozen"), default="strict")
    parser.add_argument("--ignore-recompute-mismatch", action="store_true")
    parser.add_argument("--worker-timeout-s", type=float)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--learning-rate-policy", choices=("persistent-v1", "recover-v1"))
    parser.add_argument("--learning-rate-growth", type=float)
    parser.add_argument("--max-backtracks", type=int)
    parser.add_argument("--max-evaluations", type=int)
    parser.add_argument("--parameter-stability-updates", type=int)
    parser.add_argument("--parameter-stability-rtol", type=float)
    parser.add_argument("--parameter-stability-atol", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--fit-log", choices=("concise", "detailed", "quiet"), default="concise")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--interrupt-grace-s", type=float, default=15.0)
    parser.add_argument("--terminate-grace-s", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir")
    if args.iterations < 0 or args.cpu_threads < 1:
        parser.error("iterations must be nonnegative and cpu-threads positive")
    if args.segment_length is not None and args.segment_length < 1:
        parser.error("segment-length must be positive")
    for name in ("worker_timeout_s", "interrupt_grace_s", "terminate_grace_s"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(name.replace("_", "-") + " must be positive and finite")
    return args


def _common_child_options(args, normalized_paths):
    options = []
    for value in normalized_paths:
        options.extend(("--path", value))
    for name in ("backend", "precision", "physics_version", "p2g_mode", "segment_length"):
        value = getattr(args, name)
        if value is not None:
            options.extend(("--" + name.replace("_", "-"), str(value)))
    options.extend(("--cpu-threads", str(args.cpu_threads),
                    "--reference-policy", args.reference_policy,
                    "--iterations", str(args.iterations),
                    "--fit-log", args.fit_log))
    if args.ignore_recompute_mismatch:
        options.append("--ignore-recompute-mismatch")
    if args.worker_timeout_s is not None:
        options.extend(("--worker-timeout-s", str(args.worker_timeout_s)))
    for name in ("learning_rate", "learning_rate_policy", "learning_rate_growth",
                 "max_backtracks", "max_evaluations", "parameter_stability_updates",
                 "parameter_stability_rtol"):
        value = getattr(args, name)
        if value is not None:
            options.extend(("--" + name.replace("_", "-"), str(value)))
    for value in args.parameter_stability_atol:
        options.extend(("--parameter-stability-atol", value))
    return options


def build_plan(args, *, source_provider=source_identity):
    dataset_path = args.dataset.expanduser().resolve()
    python = _resolve_python(args.python)
    youngs = _unique_values(args.initial_youngs_modulus, "initial Young's modulus")
    viscosities = _unique_values(args.initial_viscosity, "initial viscosity")
    path_overrides, normalized_paths = _path_options(args.path)
    load_options = {key: getattr(args, key) for key in
                    ("backend", "precision", "p2g_mode", "physics_version", "segment_length")
                    if getattr(args, key) is not None}
    if path_overrides:
        load_options["path_overrides"] = path_overrides
    attempts = []
    ordinal = 0
    source_hash = None
    for youngs_modulus in youngs:
        for viscosity in viscosities:
            ordinal += 1
            dataset = _load_dataset_host_only(dataset_path, initial_overrides={
                "youngs_modulus": youngs_modulus,
                "viscosity": viscosity,
            }, **load_options)
            if source_hash is None:
                source_hash = dataset.source_document_sha256
            elif source_hash != dataset.source_document_sha256:
                raise ValueError("Resolved attempts do not use the same dataset document")
            key = canonical_hash({"ordinal": ordinal, "dataset_fingerprint": dataset.fingerprint,
                                  "youngs_modulus": youngs_modulus, "viscosity": viscosity})
            attempts.append({
                "id": f"attempt_{ordinal:03d}_{key[:12]}",
                "ordinal": ordinal,
                "youngs_modulus": youngs_modulus,
                "viscosity": viscosity,
                "dataset_fingerprint": dataset.fingerprint,
                "dataset": dataset.as_dict(),
            })
    child_options = _common_child_options(args, normalized_paths)
    settings = {
        "dataset": str(dataset_path),
        "python": str(python),
        "path_overrides": normalized_paths,
        "runtime_and_optimizer_argv": child_options,
        "interrupt_grace_s": args.interrupt_grace_s,
        "terminate_grace_s": args.terminate_grace_s,
    }
    identity = {
        "schema": SCHEMA,
        "action": "dataset-calibration-attempts",
        "dataset_source_document_sha256": source_hash,
        "settings": settings,
        "attempts": [{key: row[key] for key in ("id", "ordinal", "youngs_modulus",
                     "viscosity", "dataset_fingerprint")} for row in attempts],
        "source": source_provider(),
    }
    return {"identity": identity, "identity_sha256": canonical_hash(identity),
            "dataset": dataset_path, "python": python, "child_options": child_options,
            "attempts": attempts}


def child_command(plan, attempt, calibration_directory, *, resume=False):
    command = [str(plan["python"]), "-m", "experiments.differentiable_mpm.calibrate_dataset", "fit",
               "--dataset", str(plan["dataset"]), "--output-dir", str(calibration_directory),
               "--initial-youngs-modulus", str(attempt["youngs_modulus"]),
               "--initial-viscosity", str(attempt["viscosity"]), *plan["child_options"]]
    if resume:
        command.append("--resume")
    return command


def _read_json(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read verified JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_completed(calibration_directory, attempt, exit_code=0):
    """Verify the child run and return fields used to rank completed attempts."""
    if exit_code != 0:
        raise ValueError(f"Child exit code is {exit_code}, not zero")
    manifest_path = calibration_directory / "run_manifest.json"
    result_path = calibration_directory / "result.json"
    selected_path = calibration_directory / "selected_parameters.json"
    manifest = _read_json(manifest_path)
    result = _read_json(result_path)
    selected = _read_json(selected_path)
    if manifest.get("schema") != RUN_SCHEMA:
        raise ValueError("Child run manifest has the wrong schema")
    identity = manifest.get("identity")
    identity_hash = manifest.get("identity_sha256")
    if not isinstance(identity, dict) or canonical_hash(identity) != identity_hash:
        raise ValueError("Child run manifest identity is corrupt")
    dataset_identity = identity.get("dataset_identity")
    if identity.get("action") != "dataset-fit" or not isinstance(dataset_identity, dict):
        raise ValueError("Child is not a dataset fit")
    if dataset_identity.get("dataset_fingerprint") != attempt["dataset_fingerprint"]:
        raise ValueError("Child dataset fingerprint differs from the planned attempt")
    if dataset_identity.get("dataset") != attempt["dataset"]:
        raise ValueError("Child resolved dataset differs from the planned attempt")
    dataset_schema = attempt["dataset"].get("schema")
    selection_schema = ("taichidough/dataset-physical-selection/v2" if dataset_schema == DATASET_V2
                        else "taichidough/dataset-material-selection/v1")
    if selected.get("schema") != selection_schema:
        raise ValueError("Child selected parameters have the wrong schema")
    if selected.get("identity_sha256") != identity_hash or selected.get("dataset_identity") != dataset_identity:
        raise ValueError("Selected parameters do not belong to the child run")
    if selected.get("dataset_fingerprint") != attempt["dataset_fingerprint"]:
        raise ValueError("Selected parameters use another dataset fingerprint")
    names = MATERIAL_NAMES + CONTACT_NAMES if dataset_schema == DATASET_V2 else MATERIAL_NAMES
    parameter_key = "shared_physical_parameters" if dataset_schema == DATASET_V2 else "shared_material_parameters"
    values = selected.get(parameter_key)
    if not isinstance(values, dict) or set(values) != set(names):
        raise ValueError("Selected parameters do not contain exactly the dataset shared values")
    values = {name: _finite(values[name], f"selected {name}") for name in names}
    training_value = _finite(selected.get("training_value"), "training value")
    status = result.get("status")
    if status not in SUCCESS_STATUSES:
        raise ValueError(f"Child result status is not complete: {status}")
    optimization = result.get("optimization")
    if not isinstance(optimization, dict) or optimization.get("status") != status:
        raise ValueError("Child optimization status does not match its result")
    if _finite(optimization.get("best_value"), "optimizer best value") != training_value:
        raise ValueError("Selected training value differs from the optimizer best value")
    if result.get(parameter_key) != values:
        raise ValueError("Child result and selected physical values differ")
    if result.get("independent_evaluation_skipped") is not False:
        raise ValueError("Required final strict evaluation was skipped")
    evaluations = result.get("evaluations")
    episode_ids = {row["id"] for row in attempt["dataset"]["episodes"]}
    if not isinstance(evaluations, dict) or set(evaluations) != episode_ids:
        raise ValueError("Final strict evaluation does not cover every dataset episode")
    if not all(isinstance(row, dict) and row.get("strict_evaluation", {}).get("valid") is True
               for row in evaluations.values()):
        raise ValueError("Final strict evaluation failed")
    return {"training_value": training_value, "parameters": values, "status": status,
            "selected_parameters": str(selected_path.resolve()),
            "selected_parameters_sha256": _sha256(selected_path),
            "child_identity_sha256": identity_hash}


def _write_new_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_text(path, text):
    temporary = path.parent / ("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_csv(path, rows):
    from io import StringIO
    stream = StringIO(newline="")
    fields = ("ordinal", "id", "youngs_modulus", "viscosity", "status", "invocations",
              "last_exit_code", "training_value", "error")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fields})
    _atomic_text(path, stream.getvalue())


def _winner(rows):
    completed = [row for row in rows if row.get("status") == "completed"
                 and isinstance(row.get("training_value"), (int, float))
                 and math.isfinite(row["training_value"])]
    return min(completed, key=lambda row: (row["training_value"], row["ordinal"])) if completed else None


def _write_summary(store, summary):
    store.write_json("summary.json", summary)
    _write_csv(store.path / "summary.csv", summary["attempts"])
    winner = _winner(summary["attempts"])
    if winner is not None:
        store.write_json("best_attempt.json", {
            "schema": BEST_SCHEMA,
            "attempt_id": winner["id"],
            "ordinal": winner["ordinal"],
            "training_value": winner["training_value"],
            "selected_parameters": winner["selected_parameters"],
            "selected_parameters_sha256": winner["selected_parameters_sha256"],
            "note": "Minimum verified training loss; this is not held-out model selection.",
        })


@dataclass
class ProcessOutcome:
    exit_code: int
    interrupted: bool = False
    signals: tuple[str, ...] = ()


class Interruption:
    def __init__(self):
        self.requested = False
        self.signum = None
        self.controller = None

    def handler(self, signum, _frame):
        self.requested = True
        self.signum = signum
        if self.controller is not None:
            self.controller.interrupt()


class _RunningProcess:
    def __init__(self, process):
        self.process = process
        self.interrupted_at = None
        self.sent = []

    def interrupt(self):
        if self.process.poll() is None and self.interrupted_at is None:
            self.process.send_signal(signal.SIGINT)
            self.interrupted_at = time.monotonic()
            self.sent.append("SIGINT")


def run_child(command, *, cwd, log_path, prefix, interruption, interrupt_grace_s, terminate_grace_s):
    """Stream one coordinator process and stop it in stages after interruption."""
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        controller = _RunningProcess(process)
        interruption.controller = controller
        selector = selectors.DefaultSelector()
        assert process.stdout is not None
        selector.register(process.stdout, selectors.EVENT_READ)
        terminated_at = None
        try:
            while process.poll() is None:
                for key, _ in selector.select(timeout=0.2):
                    line = cast(TextIO, key.fileobj).readline()
                    if line:
                        log.write(line)
                        log.flush()
                        print(f"[{prefix}] {line}", end="", flush=True)
                now = time.monotonic()
                if interruption.requested:
                    controller.interrupt()
                if controller.interrupted_at is not None and terminated_at is None \
                        and now - controller.interrupted_at >= interrupt_grace_s and process.poll() is None:
                    process.terminate()
                    controller.sent.append("SIGTERM")
                    terminated_at = now
                if terminated_at is not None and now - terminated_at >= terminate_grace_s \
                        and process.poll() is None:
                    process.kill()
                    controller.sent.append("SIGKILL")
            for line in process.stdout:
                log.write(line)
                print(f"[{prefix}] {line}", end="", flush=True)
            log.flush()
            code = process.wait()
        finally:
            selector.close()
            interruption.controller = None
            if process.poll() is None:
                process.kill()
                process.wait()
        return ProcessOutcome(code, controller.interrupted_at is not None, tuple(controller.sent))


def _initial_summary(plan):
    return {
        "schema": SUMMARY_SCHEMA,
        "identity_sha256": plan["identity_sha256"],
        "status": "running",
        "updated_at": utc_now(),
        "attempts": [{
            "id": row["id"], "ordinal": row["ordinal"],
            "youngs_modulus": row["youngs_modulus"], "viscosity": row["viscosity"],
            "dataset_fingerprint": row["dataset_fingerprint"], "status": "pending",
            "invocations": 0,
        } for row in plan["attempts"]],
    }


def _load_summary(store, plan):
    summary = store.read_json("summary.json")
    if summary.get("schema") != SUMMARY_SCHEMA or summary.get("identity_sha256") != plan["identity_sha256"]:
        raise ValueError("Attempt summary does not belong to this launcher identity")
    expected = [row["id"] for row in plan["attempts"]]
    rows = summary.get("attempts")
    if not isinstance(rows, list) or [row.get("id") for row in rows] != expected:
        raise ValueError("Attempt summary list was modified or is corrupt")
    return summary


def _attempt_metadata(attempt):
    return {"schema": ATTEMPT_SCHEMA, **attempt}


def _prepare_attempt(root, attempt, *, resume):
    directory = root / "attempts" / attempt["id"]
    metadata_path = directory / "attempt.json"
    if resume and directory.exists():
        if not metadata_path.is_file() or _read_json(metadata_path) != _attempt_metadata(attempt):
            raise ValueError(f"Attempt metadata differs or is corrupt: {attempt['id']}")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        _write_new_json(metadata_path, _attempt_metadata(attempt))
    return directory, directory / "calibration"


def execute(args, *, process_runner=run_child, source_provider=source_identity, repository_root=None,
            store_factory: Callable[..., RunStore] = RunStore):
    plan = build_plan(args, source_provider=source_provider)
    if args.output_dir is None:
        name = datetime.now(timezone.utc).strftime("dataset_attempts_%Y%m%dT%H%M%S_") + plan["identity_sha256"][:8]
        output = RUN_ROOT / name
    else:
        output = args.output_dir.expanduser().resolve()
    preview = []
    for attempt in plan["attempts"]:
        calibration = output / "attempts" / attempt["id"] / "calibration"
        preview.append({**{key: attempt[key] for key in ("id", "ordinal", "youngs_modulus", "viscosity",
                                                         "dataset_fingerprint")},
                        "argv": child_command(plan, attempt, calibration)})
    if args.validate_only:
        print(json.dumps({"schema": SCHEMA, "identity_sha256": plan["identity_sha256"],
                          "output_dir": str(output), "attempts": preview}, indent=2, sort_keys=True))
        return 0
    interruption = Interruption()
    old_handlers = {}
    root = Path(__file__).resolve().parents[2] if repository_root is None else Path(repository_root)
    with store_factory(output, plan["identity"], resume=args.resume) as store:
        summary = _load_summary(store, plan) if args.resume else _initial_summary(plan)
        if not args.resume:
            (store.path / "attempts").mkdir()
        _write_summary(store, summary)
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, interruption.handler)
        try:
            for attempt, row in zip(plan["attempts"], summary["attempts"], strict=True):
                if interruption.requested:
                    break
                attempt_dir, default_calibration = _prepare_attempt(store.path, attempt, resume=args.resume)
                saved_calibration = row.get("calibration_directory")
                calibration = (Path(saved_calibration) if isinstance(saved_calibration, str)
                               else default_calibration)
                if row.get("status") == "completed":
                    try:
                        row.update(verify_completed(calibration, attempt, 0))
                        continue
                    except ValueError as error:
                        row.update(status="failed", error=f"Completed artifact verification failed: {error}")
                if row.get("status") == "failed" and not args.retry_failed:
                    continue
                invocation_number = int(row.get("invocations", 0)) + 1
                retrying_failed = row.get("status") == "failed"
                if retrying_failed:
                    calibration = attempt_dir / f"calibration_retry_{invocation_number:04d}"
                resume_child = not retrying_failed and (calibration / "run_manifest.json").is_file()
                invocation_dir = attempt_dir / "invocations" / f"{invocation_number:04d}"
                invocation_dir.mkdir(parents=True, exist_ok=False)
                command = child_command(plan, attempt, calibration, resume=resume_child)
                started = {"schema": INVOCATION_SCHEMA, "attempt_id": attempt["id"],
                           "invocation": invocation_number, "started_at": utc_now(),
                           "argv": command, "cwd": str(root.resolve()), "resume_child": resume_child}
                _write_new_json(invocation_dir / "started.json", started)
                row.update(status="running", invocations=invocation_number, error="",
                           calibration_directory=str(calibration.resolve()))
                summary["updated_at"] = utc_now()
                _write_summary(store, summary)
                store.append_event({"event": "attempt_started", "attempt_id": attempt["id"],
                                    "invocation": invocation_number, "argv": command})
                try:
                    outcome = process_runner(command, cwd=root, log_path=invocation_dir / "console.log",
                                             prefix=attempt["id"], interruption=interruption,
                                             interrupt_grace_s=args.interrupt_grace_s,
                                             terminate_grace_s=args.terminate_grace_s)
                except BaseException as error:
                    outcome = ProcessOutcome(-1, interruption.requested)
                    runner_error = f"{type(error).__name__}: {error}"
                else:
                    runner_error = ""
                finished = {"schema": INVOCATION_SCHEMA, "attempt_id": attempt["id"],
                            "invocation": invocation_number, "finished_at": utc_now(),
                            "exit_code": outcome.exit_code, "interrupted": outcome.interrupted,
                            "signals": list(outcome.signals), "runner_error": runner_error}
                _write_new_json(invocation_dir / "finished.json", finished)
                row["last_exit_code"] = outcome.exit_code
                if interruption.requested or outcome.interrupted:
                    row.update(status="interrupted", error=runner_error or "Launcher interruption")
                elif runner_error:
                    row.update(status="failed", error=runner_error)
                else:
                    try:
                        row.update(verify_completed(calibration, attempt, outcome.exit_code), status="completed", error="")
                    except ValueError as error:
                        row.update(status="failed", error=str(error))
                summary["updated_at"] = utc_now()
                _write_summary(store, summary)
                store.append_event({"event": "attempt_finished", "attempt_id": attempt["id"],
                                    "invocation": invocation_number, "status": row["status"],
                                    "exit_code": outcome.exit_code, "error": row.get("error", "")})
                if interruption.requested:
                    break
            statuses = [row["status"] for row in summary["attempts"]]
            if interruption.requested:
                summary["status"] = "interrupted"
            elif all(status == "completed" for status in statuses):
                summary["status"] = "completed"
            elif any(status == "completed" for status in statuses):
                summary["status"] = "completed_with_failures"
            else:
                summary["status"] = "failed"
            summary["updated_at"] = utc_now()
            _write_summary(store, summary)
            store.append_event({"event": "launcher_finished", "status": summary["status"]})
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    if summary["status"] == "completed":
        return 0
    if summary["status"] == "interrupted":
        return 128 + int(interruption.signum or signal.SIGINT)
    return 2


def main(argv=None):
    try:
        return execute(parse_args(argv))
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
