"""Recorded-data forward comparison against the frozen simulator, without a loss.

Inputs and candidate source bytes are fixed once per new run. Each solver executes
in a separate single-thread CPU float32 process. No working files are modified.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
from contextlib import ExitStack

from .reference_adapter import current_reference_policy, reference_policy


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
STATE_NAMES = ("x", "v", "C", "F", "Jp")
CONTACT_NAMES = ("grid_tool0", "grid_tool1", "grid_inward0", "grid_inward1",
                 "particle_tool0", "particle_tool1", "particle_inward0", "particle_inward1")
PINNED_FILES = ("__init__.py", "state.py", "spectral.py", "solver.py", "runtime.py", "reference_adapter.py")
ONE_STEP_ATOL = {"x": 1e-6, "v": 1e-5, "C": 1e-4, "F": 1e-5, "Jp": 1e-5}
POSITION_LIMITS = {
    "short": {"rms_m": 1e-5, "max_m": 1e-4},
    "full": {"rms_m": 1e-4, "max_m": 1e-3},
}


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _run_root(value):
    root = Path(value).expanduser().resolve()
    if not root.is_relative_to(EXPERIMENT_ROOT / "runs") or root == EXPERIMENT_ROOT / "runs":
        raise ValueError("Real parity outputs must use a new directory under experiments/differentiable_mpm/runs")
    return root


def prepare_run(config_path, output_dir, short_end_frame, end_frame):
    from .config import load_config
    from .data import prepare_experiment
    from .reference_adapter import verify_reference

    started = time.perf_counter()
    config = load_config(config_path)
    if config.simulation.get("precision", "f32") != "f32" or config.backend != "cpu":
        raise ValueError("Reference comparison requires an explicitly configured CPU float32 experiment")
    if not 1 <= short_end_frame < end_frame <= config.training.end_frame:
        raise ValueError("Require 1 <= short endpoint < full endpoint <= training endpoint")
    root = _run_root(output_dir)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing parity run: {root}")
    prepared = prepare_experiment(config, end_frame=end_frame)
    root.mkdir(parents=True, exist_ok=False)
    source = root / "candidate_source"
    source.mkdir()
    hashes = {}
    for name in PINNED_FILES:
        path = EXPERIMENT_ROOT / name
        content = path.read_bytes()
        if content != path.read_bytes():
            raise RuntimeError(f"Candidate source changed while taking snapshot: {name}")
        (source / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    arrays = {f"state_{name}": getattr(prepared.initial_state, name) for name in STATE_NAMES}
    arrays.update(control_poses=prepared.controls.poses, control_velocities=prepared.controls.velocities)
    if prepared.sdf is not None:
        arrays.update(sdf_distances=prepared.sdf.distances, sdf_gradients=prepared.sdf.gradients,
                      sdf_minimums=prepared.sdf.minimums, sdf_spacings=prepared.sdf.spacings)
    np.savez(root / "inputs.npz", **arrays)
    metadata = {
        "schema": "taichidough/real-forward-parity-inputs/v1",
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": _hash_file(config_path),
        "simulation": asdict(prepared.simulation_config),
        "parameters": prepared.parameters,
        "seed": config.seed,
        "segment_length": config.segment_length,
        "short_end_frame": short_end_frame,
        "end_frame": end_frame,
        "total_steps": prepared.total_steps,
        "frames": [asdict(frame) for frame in prepared.frames],
        "prepared_fingerprint": prepared.fingerprint,
        "prepared_provenance": prepared.provenance,
        "candidate_source_sha256": hashes,
        "inputs_npz_sha256": _hash_file(root / "inputs.npz"),
        "reference_verification": verify_reference(),
        "position_acceptance_limits": POSITION_LIMITS,
        "preparation_seconds": time.perf_counter() - started,
    }
    _write_json_new(root / "inputs.json", metadata)
    print(f"PARITY prepared run={root} particles={prepared.simulation_config.n_particles} "
          f"steps={prepared.total_steps} preparation_s={metadata['preparation_seconds']:.3f}", flush=True)
    return root, metadata


def _verified_metadata(root):
    metadata = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != "taichidough/real-forward-parity-inputs/v1":
        raise ValueError("Unsupported real parity input metadata")
    if _hash_file(root / "inputs.npz") != metadata["inputs_npz_sha256"]:
        raise ValueError("Serialized parity inputs changed")
    if set(metadata["candidate_source_sha256"]) != set(PINNED_FILES):
        raise ValueError("Pinned candidate module set is incomplete")
    for name, expected in metadata["candidate_source_sha256"].items():
        if _hash_file(root / "candidate_source" / name) != expected:
            raise ValueError(f"Pinned candidate source changed: {name}")
    if metadata["position_acceptance_limits"] != POSITION_LIMITS:
        raise ValueError("Saved position criteria differ from the declared comparison criteria")
    return metadata


def _candidate_package(root):
    suffix = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    name = f"_taichidough_parity_candidate_{suffix}"
    directory = root / "candidate_source"
    spec = importlib.util.spec_from_file_location(name, directory / "__init__.py",
                                                  submodule_search_locations=[str(directory)])
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the pinned candidate package")
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return name


def _state_metrics(state, initial, mass):
    x = np.asarray(state.x, dtype=np.float64)
    v = np.asarray(state.v, dtype=np.float64)
    displacement = x - np.asarray(initial.x, dtype=np.float64)
    determinant = np.linalg.det(np.asarray(state.F, dtype=np.float64))
    return {
        "particle_count": len(x),
        "centroid_m": x.mean(axis=0).tolist(),
        "lower_m": x.min(axis=0).tolist(), "upper_m": x.max(axis=0).tolist(),
        "rms_displacement_m": float(np.sqrt(np.mean(np.sum(displacement * displacement, axis=1)))),
        "max_displacement_m": float(np.linalg.norm(displacement, axis=1).max()),
        "rms_speed_m_s": float(np.sqrt(np.mean(np.sum(v * v, axis=1)))),
        "max_speed_m_s": float(np.linalg.norm(v, axis=1).max()),
        "kinetic_energy_J": float(0.5 * mass * np.sum(v * v)),
        "elastic_detF_min": float(determinant.min()),
        "elastic_detF_mean": float(determinant.mean()),
        "elastic_detF_max": float(determinant.max()),
        "Jp_min": float(np.min(state.Jp)), "Jp_max": float(np.max(state.Jp)),
    }


def _contact_counts(reference_diagnostics):
    return [reference_diagnostics[k]["grid_nodes"]["applied_responses"] for k in range(2)] + [
        reference_diagnostics[k]["grid_nodes"]["inward_normal_velocity_removed"] for k in range(2)
    ] + [reference_diagnostics[k]["particles"]["applied_responses"] for k in range(2)] + [
        reference_diagnostics[k]["particles"]["inward_normal_velocity_removed"] for k in range(2)
    ]


def run_worker(root, stage, kind):
    root = _run_root(root)
    metadata = _verified_metadata(root)
    endpoint = metadata["short_end_frame"] if stage == "short" else metadata["end_frame"]
    frames = [row for row in metadata["frames"] if row["source_frame"] <= endpoint]
    target_steps = int(frames[-1]["completed_substeps"])
    output = root / stage / kind
    output.mkdir(parents=True, exist_ok=False)
    status: dict[str, Any] = {"kind": kind, "stage": stage, "end_frame": endpoint,
                              "expected_steps": target_steps, "frames": [], "valid": False}
    started = time.perf_counter()
    policy_contexts = ExitStack()
    try:
        package = _candidate_package(root)
        state_module = importlib.import_module(f"{package}.state")
        runtime_module = importlib.import_module(f"{package}.runtime")
        config = state_module.SimulationConfig(**metadata["simulation"])
        with np.load(root / "inputs.npz", allow_pickle=False) as saved:
            initial = state_module.ParticleState(**{name: saved[f"state_{name}"].copy() for name in STATE_NAMES})
            poses = saved["control_poses"].copy()
            velocities = saved["control_velocities"].copy()
            sdf = None
            if "sdf_distances" in saved.files:
                sdf = state_module.SDFData(saved["sdf_distances"].copy(), saved["sdf_gradients"].copy(),
                                           saved["sdf_minimums"].copy(), saved["sdf_spacings"].copy())
        status["runtime"] = runtime_module.init_runtime("cpu", "f32", cpu_threads=1, debug=False,
                                                        seed=metadata["seed"])
        import taichi as ti
        status["runtime"]["fast_math"] = bool(ti.lang.impl.current_cfg().fast_math)
        if kind == "reference":
            adapter = importlib.import_module(f"{package}.reference_adapter")
            adapter.EXPERIMENT_ROOT = EXPERIMENT_ROOT
            adapter.REPOSITORY_ROOT = REPOSITORY_ROOT
            policy_contexts.enter_context(adapter.reference_policy(current_reference_policy()))
            solver = adapter.ReferenceStepper(config, metadata["parameters"], sdf=sdf)
            solver.load_state(initial)
            read_state = solver.state
        else:
            module = importlib.import_module(f"{package}.solver")
            capacity = max(2, int(metadata["segment_length"]) + 1)
            solver = module.Stepper(config, metadata["parameters"], capacity=capacity, sdf=sdf)
            solver.load_state(0, initial)
            local_slot = 0
            read_state = lambda: solver.state(local_slot)
        status["setup_seconds"] = time.perf_counter() - started
        counts = np.zeros((target_steps, len(CONTACT_NAMES)), dtype=np.int64)
        forward_start = time.perf_counter()
        completed = 0
        first_step_seconds = None
        for frame in frames:
            target = int(frame["completed_substeps"])
            while completed < target:
                control = state_module.ToolControl(poses[completed], velocities[completed], completed * config.dt)
                tick = time.perf_counter() if completed == 0 else None
                if kind == "reference":
                    counts[completed] = _contact_counts(solver.advance(control))
                else:
                    if local_slot == solver.capacity - 1:
                        solver.load_state(0, solver.state(local_slot))
                        local_slot = 0
                    solver.advance(local_slot, control)
                    local_slot += 1
                    diagnostics = solver.diagnostics()
                    counts[completed] = [diagnostics[name] for name in CONTACT_NAMES]
                if tick is not None:
                    first_step_seconds = time.perf_counter() - tick
                completed += 1
            state = read_state()
            state.validate()
            frame_file = output / f"frame_{frame['source_frame']:04d}.npz"
            if frame_file.exists():
                raise FileExistsError(f"Repeated source frame output: {frame_file}")
            np.savez(frame_file, **state.arrays())
            record = {**frame, "file": frame_file.name, "sha256": _hash_file(frame_file),
                      "elapsed_forward_seconds": time.perf_counter() - forward_start}
            if frame["source_frame"] in {0, endpoint}:
                record["physical_state_metrics"] = _state_metrics(state, initial, config.particle_mass)
            status["frames"].append(record)
            if frame["source_frame"] in {0, endpoint} or frame["source_frame"] % 10 == 0:
                print(f"PARITY stage={stage} worker={kind} frame={frame['source_frame']} "
                      f"substeps={completed} elapsed_forward_s={record['elapsed_forward_seconds']:.3f}", flush=True)
        np.save(output / "contact_counts.npy", counts, allow_pickle=False)
        status["runtime"]["forward_verified"] = True
        status.update(valid=True, completed_steps=completed, first_step_seconds=first_step_seconds,
                      forward_seconds=time.perf_counter() - forward_start,
                      total_seconds=time.perf_counter() - started,
                      contact_columns=list(CONTACT_NAMES),
                      contact_counts_sha256=_hash_file(output / "contact_counts.npy"),
                      inputs_npz_sha256=metadata["inputs_npz_sha256"],
                      candidate_source_sha256=metadata["candidate_source_sha256"])
        _verified_metadata(root)
        _write_json_new(output / "result.json", status)
    except BaseException as exc:
        status.update(error_type=type(exc).__name__, error=str(exc),
                      total_seconds=time.perf_counter() - started)
        _write_json_new(output / "failure.json", status)
        raise
    finally:
        policy_contexts.close()


def worker_command(root, stage, kind):
    return [sys.executable, "-m", "experiments.differentiable_mpm.real_parity",
            "--reference-policy", current_reference_policy(),
            "--worker", kind, "--stage", stage, "--run-dir", str(root)]


def _launch_worker(root, stage, kind):
    stage_dir = root / stage
    stage_dir.mkdir(exist_ok=True)
    command = worker_command(root, stage, kind)
    print(f"PARITY starting stage={stage} worker={kind}", flush=True)
    with (stage_dir / f"{kind}.stdout.log").open("x", encoding="utf-8") as stdout, \
            (stage_dir / f"{kind}.stderr.log").open("x", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=REPOSITORY_ROOT, stdout=subprocess.PIPE, stderr=stderr,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                stdout.write(line)
                stdout.flush()
                if line.startswith("PARITY"):
                    print(line.rstrip(), flush=True)
            returncode = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    _write_json_new(stage_dir / f"{kind}.command.json", {"argv": command, "returncode": returncode})
    if returncode != 0:
        raise RuntimeError(f"{stage} {kind} worker failed with exit {returncode}; see {stage_dir / (kind + '.stderr.log')}")
    return json.loads((stage_dir / kind / "result.json").read_text(encoding="utf-8"))


def compare_stage(root, stage):
    metadata = _verified_metadata(root)
    path = root / stage
    reference = json.loads((path / "reference" / "result.json").read_text())
    candidate = json.loads((path / "differentiable" / "result.json").read_text())
    if not reference["valid"] or not candidate["valid"]:
        raise ValueError("A failed worker cannot produce a valid parity comparison")
    endpoint = metadata["short_end_frame"] if stage == "short" else metadata["end_frame"]
    expected = list(range(endpoint + 1))
    for result in (reference, candidate):
        if [row["source_frame"] for row in result["frames"]] != expected:
            raise ValueError("Forward comparison is missing or duplicates expected frames")
        if result["inputs_npz_sha256"] != metadata["inputs_npz_sha256"]:
            raise ValueError("Workers did not use the same serialized physical inputs")
        if result["candidate_source_sha256"] != metadata["candidate_source_sha256"]:
            raise ValueError("Worker candidate source differs from pinned source")
    if reference["runtime"] != candidate["runtime"]:
        raise ValueError("Worker runtime settings differ")
    per_frame = []
    totals = {name: {"sum_squared": 0.0, "count": 0, "max_abs": 0.0} for name in STATE_NAMES}
    worst_rms = 0.0
    worst_max = 0.0
    for ref_frame, candidate_frame in zip(reference["frames"], candidate["frames"]):
        for key in ("source_frame", "original_source_frame", "completed_substeps", "sim_time_s", "target_time_s"):
            if ref_frame[key] != candidate_frame[key]:
                raise ValueError(f"Frame pairing differs for {key}")
        frame = {key: ref_frame[key] for key in ("source_frame", "original_source_frame", "completed_substeps", "sim_time_s")}
        frame["arrays"] = {}
        ref_file = path / "reference" / ref_frame["file"]
        candidate_file = path / "differentiable" / candidate_frame["file"]
        if _hash_file(ref_file) != ref_frame["sha256"] or _hash_file(candidate_file) != candidate_frame["sha256"]:
            raise ValueError("A saved forward state changed after worker completion")
        with np.load(ref_file, allow_pickle=False) as a, np.load(candidate_file, allow_pickle=False) as b:
            for name in STATE_NAMES:
                old = np.asarray(a[name], dtype=np.float64)
                new = np.asarray(b[name], dtype=np.float64)
                if old.shape != new.shape or not np.isfinite(old).all() or not np.isfinite(new).all():
                    raise ValueError(f"Invalid {name} state at frame {ref_frame['source_frame']}")
                difference = new - old
                sum_squared = float(np.sum(difference * difference))
                maximum = float(np.abs(difference).max())
                frame["arrays"][name] = {
                    "max_abs": maximum,
                    "rms_component": float(np.sqrt(sum_squared / difference.size)),
                    "relative_l2": float(np.sqrt(sum_squared) / max(np.linalg.norm(old.ravel()), 1e-30)),
                    "within_one_step_array_tolerance": bool(np.allclose(new, old, atol=ONE_STEP_ATOL[name], rtol=1e-5)),
                }
                totals[name]["sum_squared"] += sum_squared
                totals[name]["count"] += difference.size
                totals[name]["max_abs"] = max(totals[name]["max_abs"], maximum)
                if name == "x":
                    particle_norm = np.linalg.norm(difference, axis=1)
                    frame["position_rms_m"] = float(np.sqrt(np.mean(particle_norm * particle_norm)))
                    frame["position_max_m"] = float(particle_norm.max())
                    frame["worst_particle_index"] = int(particle_norm.argmax())
        worst_rms = max(worst_rms, frame["position_rms_m"])
        worst_max = max(worst_max, frame["position_max_m"])
        per_frame.append(frame)
    for result, kind in ((reference, "reference"), (candidate, "differentiable")):
        if _hash_file(path / kind / "contact_counts.npy") != result["contact_counts_sha256"]:
            raise ValueError("Saved contact counts changed")
    ref_counts = np.load(path / "reference" / "contact_counts.npy", allow_pickle=False)
    new_counts = np.load(path / "differentiable" / "contact_counts.npy", allow_pickle=False)
    if ref_counts.shape != new_counts.shape or ref_counts.shape != (reference["completed_steps"], len(CONTACT_NAMES)):
        raise ValueError("Per-step contact diagnostics have different dimensions")
    differing_steps = np.flatnonzero(np.any(ref_counts != new_counts, axis=1))
    branch = {
        "scope": "Per-tool aggregate SDF grid/particle response and inward-removal counts; equal counts do not prove equal contact IDs.",
        "compared_substeps": len(ref_counts),
        "different_count_steps": len(differing_steps),
        "first_difference": None,
        "reference_totals": dict(zip(CONTACT_NAMES, ref_counts.sum(axis=0).astype(int).tolist())),
        "candidate_totals": dict(zip(CONTACT_NAMES, new_counts.sum(axis=0).astype(int).tolist())),
    }
    if len(differing_steps):
        step = int(differing_steps[0])
        branch["first_difference"] = {"substep_index": step, "time_s": step * metadata["simulation"]["dt"],
                                       "reference": dict(zip(CONTACT_NAMES, ref_counts[step].astype(int).tolist())),
                                       "candidate": dict(zip(CONTACT_NAMES, new_counts[step].astype(int).tolist()))}
    limits = POSITION_LIMITS[stage]
    within = worst_rms <= limits["rms_m"] and worst_max <= limits["max_m"]
    summary = {
        "schema": "taichidough/real-forward-parity-comparison/v1", "stage": stage,
        "status": "passed_position_tolerances" if within else "failed_position_tolerances",
        "within_position_tolerances": within,
        "position_limits": limits,
        "worst_frame_position_rms_m": worst_rms, "worst_frame_position_max_m": worst_max,
        "all_frame_arrays": {name: {"max_abs": value["max_abs"],
                                     "rms_component": float(np.sqrt(value["sum_squared"] / value["count"]))}
                             for name, value in totals.items()},
        "frame_completeness_verified": True,
        "frames": per_frame,
        "contact_count_comparison": branch,
        "runtime": reference["runtime"],
        "timings": {"reference": {k: reference[k] for k in ("setup_seconds", "first_step_seconds", "forward_seconds", "total_seconds")},
                    "candidate": {k: candidate[k] for k in ("setup_seconds", "first_step_seconds", "forward_seconds", "total_seconds")}},
        "endpoint_metrics": {"reference": reference["frames"][-1]["physical_state_metrics"],
                             "candidate": candidate["frames"][-1]["physical_state_metrics"]},
        "tested_candidate_source_sha256": metadata["candidate_source_sha256"],
        "current_candidate_source_sha256": {name: _hash_file(EXPERIMENT_ROOT / name) for name in PINNED_FILES},
        "prepared_fingerprint": metadata["prepared_fingerprint"],
        "notes": ["No observation loss, optimizer, or backward pass was used.",
                  "forward_seconds includes first-step compilation, diagnostic reads, frame-state downloads and exports.",
                  "Long-trajectory acceptance uses declared position limits; one-step all-array tolerances are separately reported diagnostics.",
                  "Candidate source bytes were pinned before either worker; later implementation edits do not change these results.",
                  "SDF response-count differences prove some active decisions differ; matching aggregate counts are not a proof of matching every decision."],
    }
    _write_json_new(path / "comparison.json", summary)
    print(f"PARITY compared stage={stage} status={summary['status']} rms_m={worst_rms:.9g} "
          f"max_m={worst_max:.9g} differing_contact_steps={len(differing_steps)}", flush=True)
    return summary


def execute_stages(root, stages):
    summaries = {}
    for stage in stages:
        if stage == "full":
            previous = summaries.get("short")
            if previous is None:
                previous = json.loads((root / "short" / "comparison.json").read_text())
            if not previous["within_position_tolerances"]:
                raise ValueError("Full replay requires the short replay to meet its declared position criteria")
        _launch_worker(root, stage, "reference")
        _launch_worker(root, stage, "differentiable")
        summaries[stage] = compare_stage(root, stage)
        if not summaries[stage]["within_position_tolerances"]:
            print(f"PARITY stopped after {stage}: position criteria failed", flush=True)
            return 2
    print(f"PARITY finished run={root}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-policy", choices=("strict", "frozen"), default="strict")
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs" / "episode18_viscoelastic.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--short-end-frame", type=int, default=2)
    parser.add_argument("--end-frame", type=int, default=60)
    parser.add_argument("--short-only", action="store_true")
    parser.add_argument("--continue-full", type=Path, help="Execute only the full stage of this verified, already-passing short run")
    parser.add_argument("--worker", choices=("reference", "differentiable"), help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("short", "full"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    with reference_policy(args.reference_policy):
        return _run_cli(args, parser)


def _run_cli(args, parser):
    if args.worker:
        if args.stage is None or args.run_dir is None:
            parser.error("worker requires stage and run-dir")
        run_worker(args.run_dir, args.stage, args.worker)
        return 0
    root = None
    try:
        if args.continue_full:
            if args.output_dir or args.short_only:
                parser.error("continue-full cannot be combined with output-dir or short-only")
            root = _run_root(args.continue_full)
            _verified_metadata(root)
            if (root / "full").exists():
                raise FileExistsError("Full parity stage already exists; it will not be overwritten")
            return execute_stages(root, ["full"])
        if args.output_dir is None:
            parser.error("provide a fresh --output-dir under the experiment runs directory")
        root, _ = prepare_run(args.config, args.output_dir, args.short_end_frame, args.end_frame)
        return execute_stages(root, ["short"] if args.short_only else ["short", "full"])
    except BaseException as exc:
        if root is not None and root.is_dir():
            failure_file = root / ("full_failure.json" if args.continue_full else "failure.json")
            if not failure_file.exists():
                _write_json_new(failure_file, {"valid": False, "error_type": type(exc).__name__, "error": str(exc),
                                               "traceback": traceback.format_exc()})
        print(f"PARITY failed: {type(exc).__name__}: {exc}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
