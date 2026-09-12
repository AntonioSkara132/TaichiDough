"""Qualify actual simulator and observation-loss execution on an explicit backend.

Run from the repository root, for example:
    python3 -m experiments.differentiable_mpm.backend_check \
        --backend vulkan --precision f32 --compare-cpu

Each backend executes in a fresh process. Evidence, initialization logs and
numerical comparisons are kept in a new directory under this experiment's runs/.
This checks a small contact/plasticity fixture, not a full Episode 18 replay.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np

from .state import P2G_MODES


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
EVIDENCE_ROOT = EXPERIMENT_ROOT / "runs"
SCHEMA_VERSION = 1
FIXTURE_VERSION = "backend-contact-plasticity-loss-v1"
SOURCE_NAMES = ("backend_check.py", "runtime.py", "solver.py", "spectral.py", "state.py", "renderer.py", "loss.py")


def source_hashes():
    return {name: hashlib.sha256((EXPERIMENT_ROOT / name).read_bytes()).hexdigest() for name in SOURCE_NAMES}


def device_inventory():
    """Record local devices without claiming which device the runtime selected."""
    devices = []
    for path in sorted(Path("/sys/class/drm").glob("renderD*/device")):
        record = {"render_node": str(Path("/dev/dri") / path.parent.name),
                  "pci_path": str(path.resolve()), "driver": path.joinpath("driver").resolve().name}
        for name in ("vendor", "device", "subsystem_vendor", "subsystem_device"):
            try:
                record[name] = path.joinpath(name).read_text().strip()
            except OSError:
                pass
        devices.append(record)
    return {"drm_devices": devices, "platform": platform.platform(),
            "selection_note": "Inventory is not proof of the selected device; inspect initialization logs.",
            "environment": {name: os.environ.get(name) for name in (
                "CUDA_VISIBLE_DEVICES", "TI_VISIBLE_DEVICE", "VK_ICD_FILENAMES", "VK_DRIVER_FILES")}}


def drm_clients():
    """Read this process's DRM clients; duplicate descriptors share one counter."""
    clients = {}
    try:
        descriptors = list(Path("/proc/self/fdinfo").iterdir())
    except OSError:
        return []
    for descriptor in descriptors:
        try:
            values = dict(line.split(":", 1) for line in descriptor.read_text().splitlines() if line.startswith("drm-") and ":" in line)
        except OSError:
            continue
        values = {key: value.strip() for key, value in values.items()}
        if "drm-driver" not in values:
            continue
        identity = (values.get("drm-driver"), values.get("drm-pdev"), values.get("drm-client-id"))
        record = clients.setdefault(identity, {"driver": identity[0], "pci_device": identity[1],
                                               "client_id": identity[2], "engine_time_ns": {}})
        for name, value in values.items():
            if name.startswith("drm-engine-") and value.endswith(" ns"):
                try:
                    elapsed = int(value[:-3].strip())
                except ValueError:
                    continue
                engine = name[len("drm-engine-"):]
                record["engine_time_ns"][engine] = max(record["engine_time_ns"].get(engine, 0), elapsed)
    return list(clients.values())


def drm_activity(before, after):
    earlier = {(r["driver"], r["pci_device"], r["client_id"]): r["engine_time_ns"] for r in before}
    changes = []
    for record in after:
        previous = earlier.get((record["driver"], record["pci_device"], record["client_id"]), {})
        delta = {name: count - previous.get(name, 0) for name, count in record["engine_time_ns"].items()}
        if any(count > 0 for count in delta.values()):
            changes.append({**record, "engine_delta_ns": delta})
    return {"verified": bool(changes), "active_clients": changes,
            "basis": "Increased per-process DRM engine time between runtime initialization and synchronized solver/loss execution.",
            "limitation": "Absent counters mean hardware identity is unverified, not that the backend ran on CPU."}


def _write_json(path, value):
    temporary = path.with_suffix(".next.json")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def create_evidence_directory(requested=None):
    path = Path(requested).expanduser().resolve() if requested else EVIDENCE_ROOT / (
        "backend_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    if not path.is_relative_to(EVIDENCE_ROOT.resolve()) or path == EVIDENCE_ROOT.resolve():
        raise ValueError(f"Backend evidence must be in a new directory under {EVIDENCE_ROOT}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _fixture(precision, p2g_mode="atomic"):
    from .state import DEFAULT_PARAMETERS, ParticleState, SDFData, SimulationConfig, ToolControl

    dtype = np.float32 if precision == "f32" else np.float64
    config = SimulationConfig(n_particles=4, grid=12, precision=precision, dt=2e-4,
                              particle_mass=0.001, particle_volume=1e-6, gravity=-2.0,
                              plasticity="stretch-clamp", use_jp=True, jp_hardening=2.0,
                              floor_y=0.412, floor_plastic_damping_band=0.01,
                              plastic_affine_damping=0.87, tool_collision="sdf",
                              tool_contact_padding=0.004, p2g_mode=p2g_mode)
    state = ParticleState.initial([[.417, .40, .405], [.441, .415, .409],
                                   [.456, .428, .432], [.431, .421, .445]], dtype)
    state.v[:] = [[.1, -.2, .04], [-.05, -.1, .08], [.02, -.3, -.03], [.01, -.1, .07]]
    state.C[:] = [[.2, .3, .02], [-.08, -.1, .07], [.03, -.01, .15]]
    state.F[:] = [[.84, .03, .01], [.0, 1.04, -.02], [.01, .015, 1.17]]
    state.Jp[:] = [1.01, .97, 1.04, .93]
    parameters = dict(DEFAULT_PARAMETERS, youngs_modulus=4200.0, viscosity=8.0,
                      plastic_min=.91, plastic_max=1.10, tool_retention=.28, floor_retention=.43)
    control = ToolControl.stationary()
    control.poses[0, :3] = [.43, .40, .37]
    control.poses[1, :3] = [.42, .42, .52]
    control.poses[0, 3:] = [0, np.sin(.27 / 2), 0, np.cos(.27 / 2)]
    control.velocities[:] = [[.015, -.012, .05, .2, -.3, .4], [-.01, .012, -.035, -.1, .25, .35]]
    resolution = 13
    axis = np.linspace(-.18, .18, resolution)
    coordinates = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    norm = np.linalg.norm(coordinates, axis=-1)
    distance = norm - .08
    normals = coordinates / np.maximum(norm[..., None], 1e-12)
    sdf = SDFData(np.stack([distance, distance]), np.stack([normals, normals]),
                  np.full((2, 3), -.18), np.full((2, 3), .36 / (resolution - 1)))
    return config, state, parameters, control, sdf


def _observation_objective(initial, precision):
    from .loss import LossConfig, Observation, ObservationLoss
    from .renderer import Camera

    transform = np.eye(4)
    transform[:3, 3] = [-.43, -.42, .595]
    camera = Camera(32, 24, 40.0, 40.0, 16.0, 12.0, transform)
    config = LossConfig()
    objective = ObservationLoss(camera, config, initial.x, precision=precision)
    depth = np.full((24, 32), 1.02, dtype=initial.x.dtype)
    mask = np.zeros((24, 32), dtype=bool)
    mask[10:14, 14:18] = True
    observation = Observation(3, 3, depth + .003, mask, depth, mask, timestamp=.0006)
    return objective, observation


def _worker(backend, precision, report_path, p2g_mode="atomic"):
    import taichi as ti
    from .runtime import init_runtime
    from .solver import Stepper
    from .state import STATE_NAMES

    path = Path(report_path).resolve()
    if not path.is_relative_to(EVIDENCE_ROOT.resolve()) or path.exists():
        raise ValueError("Worker report must be a new file in the experiment runs directory")
    started = time.monotonic()
    report = {"schema_version": SCHEMA_VERSION, "fixture_version": FIXTURE_VERSION,
              "requested_backend": backend, "precision": precision, "p2g_mode": p2g_mode, "status": "running",
              "stage": "initialization", "source_hashes": source_hashes(),
              "device_inventory": device_inventory(), "stages": {}}

    def stage(name, verified):
        report["stage"] = name
        report["stages"][name] = {"verified": verified, "elapsed_s": time.monotonic() - started}
        _write_json(path, report)
        print(json.dumps({"backend": backend, "stage": name, "verified": verified}), flush=True)

    try:
        stage("initialization", False)
        previous_log_level = os.environ.get("TI_LOG_LEVEL")
        os.environ["TI_LOG_LEVEL"] = "debug"
        try:
            report["runtime"] = init_runtime(backend=backend, precision=precision, cpu_threads=1, debug=False)
        finally:
            if previous_log_level is None:
                os.environ.pop("TI_LOG_LEVEL", None)
            else:
                os.environ["TI_LOG_LEVEL"] = previous_log_level
            ti.set_logging_level(ti.INFO)
        report["drm_after_initialization"] = drm_clients()
        stage("initialization", True)
        config, initial, parameters, control, sdf = _fixture(precision, p2g_mode)
        report["configuration"] = asdict(config)
        report["parameters"] = parameters
        report["steps"] = 3
        stage("simulator_forward", False)
        solver = Stepper(config, parameters, capacity=4, sdf=sdf)
        solver.load_state(0, initial)
        counts = []
        for slot in range(3):
            control.time = slot * config.dt
            solver.advance(slot, control)
            counts.append(solver.diagnostics())
        ti.sync()
        final = solver.state(3)
        final.validate()
        report["forward_state"] = {name: getattr(final, name).tolist() for name in STATE_NAMES}
        report["branch_counts"] = counts
        for name in ("yielded", "grid_tool0", "particle_tool0", "particle_tool1", "grid_floor", "particle_floor"):
            if sum(count[name] for count in counts) == 0:
                raise RuntimeError(f"Qualification fixture did not exercise {name}")
        stage("simulator_forward", True)
        stage("observation_loss", False)
        objective, observation = _observation_objective(initial, precision)
        loss = objective.value_and_grad_positions(final.x, observation)
        ti.sync()
        if not np.isfinite(loss.value) or loss.gradient is None or not np.isfinite(loss.gradient).all():
            raise RuntimeError("Observation loss or its position gradient is not finite")
        if np.linalg.norm(loss.gradient) == 0:
            raise RuntimeError("Qualification observation has a zero position gradient")
        report["observation"] = {"value": loss.value, "components": loss.components,
                                 "diagnostics": loss.diagnostics, "position_gradient": loss.gradient.tolist()}
        stage("observation_loss", True)
        stage("simulator_backward", False)
        seed = final.zeros_like()
        rng = np.random.default_rng(527)
        for name, scale in zip(STATE_NAMES, [.3, .7, .2, .3, .2]):
            getattr(seed, name)[:] = rng.normal(size=getattr(seed, name).shape) * scale
        seed.x[:] += loss.gradient
        solver.clear_state_gradients()
        solver.clear_parameter_gradients()
        solver.load_adjoint(3, seed)
        for slot in reversed(range(3)):
            control.time = slot * config.dt
            solver.reverse_step(slot, control)
        ti.sync()
        initial_adjoint = solver.adjoint(0)
        initial_adjoint.validate()
        gradients = solver.parameter_gradients()
        if not np.isfinite(list(gradients.values())).all():
            raise RuntimeError("Nonfinite simulator parameter gradient")
        if any(gradients[name] == 0 for name in ("youngs_modulus", "poisson_ratio", "viscosity", "plastic_min", "plastic_max")):
            raise RuntimeError("Fixture did not exercise all five material parameter derivatives")
        report["initial_state_adjoint"] = {name: getattr(initial_adjoint, name).tolist() for name in STATE_NAMES}
        report["parameter_gradients"] = gradients
        for slot, expected in ((0, initial), (3, final)):
            after = solver.state(slot)
            if any(not np.array_equal(getattr(after, name), getattr(expected, name)) for name in STATE_NAMES):
                raise RuntimeError("Backward execution changed a stored forward state")
        report["drm_after_execution"] = drm_clients()
        report["gpu_execution_evidence"] = drm_activity(report["drm_after_initialization"], report["drm_after_execution"])
        if report["gpu_execution_evidence"]["verified"]:
            report["runtime"]["device_identity"] = "; ".join(
                f"{client['driver']} PCI {client['pci_device']} (DRM engine activity verified)"
                for client in report["gpu_execution_evidence"]["active_clients"])
        stage("simulator_backward", True)
        if source_hashes() != report["source_hashes"]:
            raise RuntimeError("Implementation source changed during backend qualification")
        report["status"] = "execution_passed"
        report["runtime"].update(forward_verified=True, backward_verified=True)
        report["elapsed_s"] = time.monotonic() - started
        _write_json(path, report)
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        report["elapsed_s"] = time.monotonic() - started
        _write_json(path, report)
        print(report["error"]["traceback"], file=sys.stderr, flush=True)
        return 2


def compare_results(reference, candidate):
    """Compare a candidate's complete fixture result with a same-precision CPU run."""
    checks = []

    def arrays(name, expected, actual, atol, rtol):
        a, b = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
        compatible = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        passed = bool(compatible and np.allclose(a, b, atol=atol, rtol=rtol))
        checks.append({"name": name, "passed": passed, "atol": atol, "rtol": rtol,
                       "max_absolute_difference": float(np.max(np.abs(a - b))) if compatible and a.size else None})

    required = (reference.get("status") == "execution_passed" and candidate.get("status") == "execution_passed"
                and reference.get("precision") == candidate.get("precision")
                and reference.get("fixture_version") == candidate.get("fixture_version")
                and reference.get("source_hashes") == candidate.get("source_hashes")
                and reference.get("configuration") == candidate.get("configuration")
                and reference.get("parameters") == candidate.get("parameters"))
    if not required:
        return {"passed": False, "reason": "Both executions must pass with identical precision, source, fixture and parameters", "checks": checks}
    if reference["precision"] == "f64":
        forward_limits = {"x": (1e-9, 1e-7), "v": (1e-8, 1e-6), "C": (1e-7, 1e-6),
                          "F": (1e-8, 1e-7), "Jp": (1e-8, 1e-7)}
        gradient_limits, parameter_limits, loss_limits = (1e-7, 1e-4), (1e-8, 1e-4), (1e-8, 1e-7)
    else:
        forward_limits = {"x": (1e-6, 5e-5), "v": (1e-5, 5e-4), "C": (1e-4, 5e-4),
                          "F": (1e-5, 5e-5), "Jp": (1e-5, 5e-5)}
        gradient_limits, parameter_limits, loss_limits = (1e-4, 1e-2), (1e-5, 1e-2), (1e-4, 1e-4)
    for name, limits in forward_limits.items():
        arrays("forward_" + name, reference["forward_state"][name], candidate["forward_state"][name], *limits)
        arrays("adjoint_" + name, reference["initial_state_adjoint"][name], candidate["initial_state_adjoint"][name], *gradient_limits)
    checks.append({"name": "contact_and_yield_decisions", "passed": reference["branch_counts"] == candidate["branch_counts"]})
    arrays("observation_value", reference["observation"]["value"], candidate["observation"]["value"], *loss_limits)
    arrays("observation_position_gradient", reference["observation"]["position_gradient"],
           candidate["observation"]["position_gradient"], *gradient_limits)
    for name, value in reference["parameter_gradients"].items():
        scale = max(abs(reference["parameters"][name]), .1)
        arrays("scaled_parameter_gradient_" + name, value * scale,
               candidate["parameter_gradients"][name] * scale, *parameter_limits)
    return {"passed": all(check["passed"] for check in checks), "checks": checks,
            "note": f"Tolerances apply to this short {reference['precision']} contact fixture, not long-horizon GPU replay."}


def _launch(backend, precision, directory, timeout_s, p2g_mode="atomic"):
    report_path = directory / "worker_result.json"
    command = [sys.executable, "-m", "experiments.differentiable_mpm.backend_check",
               "--backend", backend, "--precision", precision, "--p2g-mode", p2g_mode,
               "--_worker-report", str(report_path)]
    _write_json(directory / "command.json", {"argv": command})
    stdout_path, stderr_path = directory / "stdout.log", directory / "stderr.log"
    timeout = False
    with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
        try:
            process = subprocess.run(command, cwd=REPOSITORY_ROOT, stdout=stdout, stderr=stderr, timeout=timeout_s)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timeout, returncode = True, None
    try:
        report = json.loads(report_path.read_text())
    except (OSError, ValueError):
        report = {"requested_backend": backend, "precision": precision, "p2g_mode": p2g_mode, "status": "failed",
                  "stage": "worker_process", "error": {"message": "Worker produced no complete JSON record"}}
    if timeout or returncode != 0:
        report["status"] = "failed"
        report["process_failure"] = {"timed_out": timeout, "returncode": returncode}
    report["logs"] = {"stdout": str(stdout_path), "stderr": str(stderr_path)}
    report["initialization_device_log_lines"] = [line for line in (stdout_path.read_text(errors="replace") +
        stderr_path.read_text(errors="replace")).splitlines() if "device" in line.lower()]
    _write_json(directory / "result.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("cpu", "cuda", "vulkan"))
    parser.add_argument("--precision", default="f32", choices=("f32", "f64"))
    parser.add_argument("--p2g-mode", default="atomic", choices=P2G_MODES,
                        help="Transfer mode for both backends; serial can be much slower on GPUs")
    parser.add_argument("--compare-cpu", action="store_true", help="Run the identical fixture on CPU in a separate process")
    parser.add_argument("--output-dir", type=Path, help="A new directory under experiments/differentiable_mpm/runs")
    parser.add_argument("--timeout-s", type=float, default=600.0, help="Maximum time per backend process")
    parser.add_argument("--_worker-report", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker_report:
        return _worker(args.backend, args.precision, args._worker_report, args.p2g_mode)
    if not np.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be finite and positive")
    if args.compare_cpu and args.backend == "cpu":
        parser.error("--compare-cpu requires a non-CPU requested backend")
    directory = create_evidence_directory(args.output_dir)
    print(f"Backend evidence: {directory}", flush=True)
    requested_dir = directory / "requested_backend"
    requested_dir.mkdir()
    print(f"P2G mode: {args.p2g_mode}", flush=True)
    candidate = _launch(args.backend, args.precision, requested_dir, args.timeout_s, args.p2g_mode)
    reference, comparison = None, None
    if args.compare_cpu:
        if candidate.get("status") == "execution_passed":
            cpu_dir = directory / "cpu_reference"
            cpu_dir.mkdir()
            reference = _launch("cpu", args.precision, cpu_dir, args.timeout_s, args.p2g_mode)
            comparison = compare_results(reference, candidate)
        else:
            comparison = {"passed": False, "checks": [],
                          "reason": "Requested backend failed; CPU comparison was not started"}
    passed = candidate.get("status") == "execution_passed" and (comparison is None or comparison["passed"])
    summary = {"schema_version": SCHEMA_VERSION, "fixture_version": FIXTURE_VERSION,
               "status": "passed" if passed else "failed", "requested_backend": args.backend,
               "precision": args.precision, "p2g_mode": args.p2g_mode,
               "execution_verified": candidate.get("status") == "execution_passed",
               "cpu_comparison": comparison, "candidate": candidate, "cpu_reference": reference,
               "scope": "Three-step 4-particle MLS-MPM with active stretch plasticity/Jp, viscosity, moving rotating SDF tools, floor and observation-loss reverse differentiation; not full Episode18 qualification."}
    _write_json(directory / "backend_check.json", summary)
    print(json.dumps({"status": summary["status"], "execution_verified": summary["execution_verified"],
                      "cpu_comparison_passed": comparison["passed"] if comparison else None,
                      "report": str(directory / "backend_check.json")}), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
