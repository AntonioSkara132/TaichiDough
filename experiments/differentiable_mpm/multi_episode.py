"""Weighted episode objectives and one-at-a-time numerical subprocesses."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Mapping

import numpy as np

from .config import REPOSITORY_ROOT
from .dataset_config import MATERIAL_NAMES
from .optimize import ObjectiveValue
from .results import RUN_ROOT, canonical_hash, json_value, source_identity
from .state import InvalidStateError

REQUEST_SCHEMA = "taichidough/episode-request/v1"
RESULT_SCHEMA = "taichidough/episode-result/v1"


class EpisodeExecutionError(RuntimeError):
    """Input, protocol, or process failure, not an optimizer candidate rejection."""


def _finite_number(value, name):
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)):
        raise EpisodeExecutionError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise EpisodeExecutionError(f"{name} must be finite")
    return number


def normalized_weights(episodes):
    weights = np.asarray([_finite_number(ep.weight, "episode weight") for ep in episodes], dtype=np.float64)
    if not len(weights) or np.any(weights <= 0):
        raise ValueError("Require at least one episode and positive weights")
    weights /= weights.max()
    weights /= weights.sum(dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Episode weights cannot be represented as positive normalized float64 values")
    return weights


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EpisodeExecutionError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def invalid_constant(value):
        raise EpisodeExecutionError(f"Nonfinite JSON constant: {value}")
    try:
        return json.loads(Path(path).read_text(), object_pairs_hook=_unique_object,
                          parse_constant=invalid_constant)
    except (OSError, ValueError) as error:
        raise EpisodeExecutionError(f"Cannot read JSON result {path}: {error}") from error


def write_new_json(path, record):
    with Path(path).open("x") as stream:
        json.dump(json_value(record), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def stable_runtime(record):
    """Exclude execution counters while keeping requested and actual runtime identity."""
    keys = ("backend", "precision", "cpu_threads", "debug", "seed", "actual_arch",
            "taichi_version", "host", "platform", "machine", "CUDA_VISIBLE_DEVICES", "device_identity")
    return {key: record.get(key) for key in keys}


class DatasetObjective:
    """Evaluate every training episode at the same candidate; never skip failures.

    evaluator(episode, shared_parameters, compute_grad) returns the existing
    evaluation_record mapping. Its loss and gradient are already episode means.
    """

    def __init__(self, dataset, evaluator, event=None):
        self.dataset = dataset
        self.episodes = tuple(ep for ep in dataset.episodes if ep.membership == "training")
        self.weights = normalized_weights(self.episodes)
        self.evaluator = evaluator
        self.event = event or (lambda record: None)
        self.calls = 0
        self.successful_calls = 0
        self.calls_with_mismatches = 0

    def __call__(self, parameters):
        return self.value_and_gradient(parameters, compute_grad=True)

    def value_and_gradient(self, parameters, *, compute_grad=True):
        if type(compute_grad) is not bool:
            raise ValueError("compute_grad must be boolean")
        shared = {name: _finite_number(parameters[name], name) for name in MATERIAL_NAMES}
        self.calls += 1
        started = time.monotonic()
        self.event({"event": "dataset_objective_started", "call": self.calls,
                    "shared_parameters": shared, "compute_grad": compute_grad})
        contributions = []
        derivatives = {name: [] for name in self.dataset.fit_parameters}
        rows = []
        mismatch_counts = {}
        replay_consistent = True
        try:
            for ep, weight in zip(self.episodes, self.weights):
                begin = time.monotonic()
                self.event({"event": "episode_started", "call": self.calls, "episode_id": ep.id,
                            "normalized_weight": float(weight)})
                record = self.evaluator(ep, dict(shared), compute_grad)
                if not isinstance(record, Mapping):
                    raise EpisodeExecutionError("Episode evaluation must be a mapping")
                value = _finite_number(record.get("value"), f"{ep.id} loss")
                diagnostics = record.get("diagnostics", {})
                if not isinstance(diagnostics, Mapping):
                    raise EpisodeExecutionError("Episode diagnostics must be a mapping")
                gradient = {}
                if compute_grad:
                    supplied = record.get("gradient")
                    if not isinstance(supplied, Mapping):
                        raise EpisodeExecutionError(f"Missing gradient for {ep.id}")
                    for name in self.dataset.fit_parameters:
                        if name not in supplied:
                            raise EpisodeExecutionError(f"Missing {name} gradient for {ep.id}")
                        gradient[name] = _finite_number(supplied[name], f"{ep.id} {name} gradient")
                        derivatives[name].append(float(weight) * gradient[name])
                consistent = diagnostics.get("replay_consistent", True)
                if type(consistent) is not bool:
                    raise EpisodeExecutionError("replay_consistent must be boolean")
                replay_consistent = replay_consistent and consistent
                counts = diagnostics.get("recompute_mismatch_counts", {})
                if not isinstance(counts, Mapping):
                    raise EpisodeExecutionError("Mismatch counts must be a mapping")
                for name, count in counts.items():
                    if type(count) is not int or count < 0:
                        raise EpisodeExecutionError("Mismatch counts must be nonnegative integers")
                    mismatch_counts[name] = mismatch_counts.get(name, 0) + count
                weighted = float(weight) * value
                contributions.append(weighted)
                row = {"episode_id": ep.id, "value": value, "normalized_weight": float(weight),
                       "weighted_value": weighted, "gradient": gradient,
                       "elapsed_s": time.monotonic() - begin, "diagnostics": dict(diagnostics)}
                rows.append(row)
                self.event({"event": "episode_finished", "call": self.calls, **row})
            value = math.fsum(contributions)
            gradient = {name: math.fsum(values) for name, values in derivatives.items()} if compute_grad else {}
            if not math.isfinite(value) or not all(math.isfinite(v) for v in gradient.values()):
                raise InvalidStateError("Combined dataset loss or gradient is nonfinite")
        except BaseException as error:
            self.event({"event": "dataset_objective_failed", "call": self.calls,
                        "completed_episodes": [row["episode_id"] for row in rows],
                        "error_type": type(error).__name__, "error": str(error)})
            raise
        self.successful_calls += 1
        self.calls_with_mismatches += int(not replay_consistent)
        diagnostics = {"objective_version": "weighted-episode-means-v1", "episodes": rows,
                       "training_episode_count": len(rows), "elapsed_s": time.monotonic() - started,
                       "replay_consistency_checked": compute_grad, "replay_consistent": replay_consistent,
                       "recompute_mismatch_counts": mismatch_counts,
                       "recompute_mismatch_count": sum(mismatch_counts.values())}
        result = ObjectiveValue(value, gradient, diagnostics)
        self.event({"event": "dataset_objective_finished", "call": self.calls, "value": value,
                    "gradient": gradient, "replay_consistent": replay_consistent,
                    "elapsed_s": diagnostics["elapsed_s"]})
        return result


class EpisodeProcessEvaluator:
    """Own at most one child process, releasing its device allocations on exit."""

    def __init__(self, dataset, dataset_path, dataset_options, output_root, *,
                 reference_policy="strict", cpu_threads=1, ignore_recompute_mismatch=False,
                 expected_prepared=None, expected_runtimes=None, expected_source=None,
                 timeout_s=None, event=None, launcher=None):
        self.dataset = dataset
        self.dataset_path = str(Path(dataset_path).resolve())
        self.dataset_options = json_value(dataset_options)
        self.output_root = Path(output_root).resolve()
        if self.output_root == RUN_ROOT.resolve() or not self.output_root.is_relative_to(RUN_ROOT.resolve()):
            raise ValueError("Episode outputs must be below the experiment runs directory")
        if reference_policy not in {"strict", "frozen"}:
            raise ValueError("Unknown reference policy")
        if type(cpu_threads) is not int or cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        if type(ignore_recompute_mismatch) is not bool:
            raise ValueError("ignore_recompute_mismatch must be boolean")
        if timeout_s is not None and (isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("timeout_s must be positive and finite")
        self.reference_policy = reference_policy
        self.cpu_threads = cpu_threads
        self.ignore_recompute_mismatch = ignore_recompute_mismatch
        self.expected_prepared = dict(expected_prepared or {})
        self.expected_runtimes = dict(expected_runtimes or {})
        self.expected_source = expected_source if expected_source is not None else source_identity()
        self.timeout_s = timeout_s
        self.event = event or (lambda record: None)
        self.launcher = launcher or self._launch
        self._active = False
        self._lock = threading.Lock()

    @staticmethod
    def _stop(process):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        else:
            process.wait()

    def _launch(self, request_path, worker_path, request_dir):
        command = [sys.executable, "-m", "experiments.differentiable_mpm.episode_worker",
                   "--request", str(request_path), "--output-dir", str(worker_path)]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPOSITORY_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        if env.get("CLAUDE_JOB_DIR"):
            env["TMPDIR"] = str(Path(env["CLAUDE_JOB_DIR"]) / "tmp")
        process = None
        with (request_dir / "stdout.log").open("x") as stdout, (request_dir / "stderr.log").open("x") as stderr:
            try:
                process = subprocess.Popen(command, cwd=REPOSITORY_ROOT, env=env,
                                           stdout=stdout, stderr=stderr, start_new_session=True)
                return process.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired as error:
                raise EpisodeExecutionError(f"Episode worker timed out; logs: {request_dir}") from error
            finally:
                if process is not None:
                    self._stop(process)

    def __call__(self, episode, parameters, compute_grad=True):
        return self.run(episode, "objective", parameters, compute_grad=compute_grad)["evaluation"]

    def run(self, episode, action, parameters, *, compute_grad=False, no_runtime=False):
        if not self._lock.acquire(blocking=False):
            raise EpisodeExecutionError("An episode worker is already active")
        try:
            return self._run_episode(episode, action, parameters, compute_grad=compute_grad, no_runtime=no_runtime)
        finally:
            self._lock.release()

    def _run_episode(self, episode, action, parameters, *, compute_grad=False, no_runtime=False):
        if self._active:
            raise EpisodeExecutionError("An episode worker is already active")
        if action not in {"validate", "objective", "replay", "evaluate"}:
            raise ValueError("Unknown episode action")
        if type(compute_grad) is not bool or type(no_runtime) is not bool:
            raise ValueError("Worker switches must be boolean")
        if no_runtime and action != "validate":
            raise ValueError("no_runtime is only valid for input validation")
        if action != "objective" and compute_grad:
            raise ValueError("Only objective requests compute gradients")
        shared = {name: _finite_number(parameters[name], name) for name in MATERIAL_NAMES}
        request = {"schema": REQUEST_SCHEMA, "request_id": uuid.uuid4().hex,
                   "episode_id": episode.id, "action": action,
                   "dataset_path": self.dataset_path, "dataset_options": self.dataset_options,
                   "parameters": shared, "compute_grad": compute_grad, "no_runtime": no_runtime,
                   "reference_policy": self.reference_policy, "cpu_threads": self.cpu_threads,
                   "ignore_recompute_mismatch": self.ignore_recompute_mismatch,
                   "expected_source": self.expected_source,
                   "expected_dataset_fingerprint": self.dataset.fingerprint,
                   "expected_prepared_fingerprint": self.expected_prepared.get(episode.id)}
        request_dir = self.output_root / request["request_id"]
        request_dir.mkdir(parents=True, exist_ok=False)
        request_path = request_dir / "request.json"
        worker_path = request_dir / "worker"
        write_new_json(request_path, request)
        self._active = True
        started = time.monotonic()
        try:
            self.event({"event": "worker_started", "episode_id": episode.id, "action": action,
                        "request_id": request["request_id"], "directory": str(request_dir)})
            code = self.launcher(request_path, worker_path, request_dir)
            result = read_json(worker_path / "result.json")
            self._check_result(result, request, episode, code)
            self.event({"event": "worker_finished", "episode_id": episode.id, "action": action,
                        "request_id": request["request_id"], "elapsed_s": time.monotonic() - started,
                        "directory": str(request_dir)})
            return result
        except BaseException as error:
            self.event({"event": "worker_failed", "episode_id": episode.id, "action": action,
                        "request_id": request["request_id"], "error_type": type(error).__name__,
                        "error": str(error), "directory": str(request_dir)})
            raise
        finally:
            self._active = False

    def _check_result(self, result, request, episode, code):
        if not isinstance(result, dict):
            raise EpisodeExecutionError("Episode result must be an object")
        expected = {"schema": RESULT_SCHEMA, "request_id": request["request_id"],
                    "request_sha256": canonical_hash(request), "episode_id": episode.id,
                    "source": self.expected_source}
        for key, value in expected.items():
            if result.get(key) != value:
                raise EpisodeExecutionError(f"Episode result {key} does not match the request")
        if source_identity() != self.expected_source:
            raise EpisodeExecutionError("Experiment sources changed during episode execution")
        status = result.get("status")
        if status == "error":
            raise EpisodeExecutionError(f"Episode {episode.id} failed: {result.get('error', 'unspecified error')}")
        if result.get("dataset_fingerprint") != self.dataset.fingerprint:
            raise EpisodeExecutionError("Episode result dataset_fingerprint does not match the request")
        if status not in {"ok", "invalid"} or code != {"ok": 0, "invalid": 2}[status]:
            raise EpisodeExecutionError(f"Abnormal episode status/exit: {status!r}/{code}")
        if result.get("source_after") != self.expected_source:
            raise EpisodeExecutionError("Worker sources changed before completion")
        if result.get("parameters") != episode.parameters_for(request["parameters"]):
            raise EpisodeExecutionError("Worker evaluated different effective parameters")
        if result.get("compute_grad") is not request["compute_grad"]:
            raise EpisodeExecutionError("Worker gradient mode does not match")
        fingerprint = result.get("prepared_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise EpisodeExecutionError("Missing prepared input fingerprint")
        if request["expected_prepared_fingerprint"] is not None and fingerprint != request["expected_prepared_fingerprint"]:
            raise EpisodeExecutionError("Prepared episode inputs changed")
        runtime = result.get("runtime")
        if not isinstance(runtime, dict):
            raise EpisodeExecutionError("Missing episode runtime record")
        if not request["no_runtime"]:
            config = episode.config
            backend = config.backend
            architectures = {"cpu": {"Arch.x64", "Arch.arm64"}, "cuda": {"Arch.cuda"}, "vulkan": {"Arch.vulkan"}}
            if (runtime.get("backend") != backend or runtime.get("precision") != config.simulation.get("precision", "f32")
                    or runtime.get("actual_arch") not in architectures[backend]
                    or runtime.get("initialization_verified") is not True):
                raise EpisodeExecutionError("Actual episode runtime does not match requested backend/precision")
            if episode.id in self.expected_runtimes and stable_runtime(runtime) != self.expected_runtimes[episode.id]:
                raise EpisodeExecutionError("Episode runtime identity changed since validation")
        if status == "invalid":
            if request["action"] != "objective":
                raise EpisodeExecutionError(f"Episode {episode.id} failed outside candidate evaluation: {result.get('error', '')}")
            raise InvalidStateError(f"Episode {episode.id} invalidates candidate: {result.get('error', 'invalid state')}")
        if request["action"] == "objective":
            record = result.get("evaluation")
            if not isinstance(record, dict):
                raise EpisodeExecutionError("Missing episode evaluation")
            _finite_number(record.get("value"), "episode loss")
            if request["compute_grad"]:
                gradient = record.get("gradient")
                if not isinstance(gradient, dict) or any(name not in gradient for name in self.dataset.fit_parameters):
                    raise EpisodeExecutionError("Missing fitted episode gradients")
                for name, value in gradient.items():
                    _finite_number(value, f"{name} gradient")
