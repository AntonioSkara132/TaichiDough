#!/usr/bin/env python3
"""Forward-only replay for a derived dataset episode configuration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.differentiable_mpm.config import file_sha256, load_config
from experiments.differentiable_mpm.data import prepare_experiment
from experiments.differentiable_mpm.evaluate import fresh_directory
from experiments.differentiable_mpm.policy_adapter import load_condition_archive
from experiments.differentiable_mpm.reference_adapter import reference_identity, reference_policy
from experiments.differentiable_mpm.runtime import init_runtime
from experiments.differentiable_mpm.solver import Stepper


def write_new_json(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--precision", choices=("f32", "f64"), required=True)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--reference-policy", choices=("strict", "frozen"), default="frozen")
    parser.add_argument("--controls-archive", type=Path)
    parser.add_argument("--condition")
    args = parser.parse_args(argv)
    if (args.controls_archive is None) != (args.condition is None):
        parser.error("--controls-archive and --condition must be supplied together")
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")

    output = fresh_directory(args.output_dir)
    config = load_config(args.config)
    source_paths = [Path(__file__), *[Path(module.__file__) for module in
                    (__import__("experiments.differentiable_mpm.solver", fromlist=["x"]),
                     __import__("experiments.differentiable_mpm.spectral", fromlist=["x"]),
                     __import__("experiments.differentiable_mpm.state", fromlist=["x"]),
                     __import__("experiments.differentiable_mpm.data", fromlist=["x"]))]]
    before = {str(path.resolve()): file_sha256(path.resolve()) for path in source_paths}
    started = time.perf_counter()
    with reference_policy(args.reference_policy):
        reference = reference_identity(config.simulation.get("physics_version", "corrected-v1"))
        runtime = init_runtime(args.backend, args.precision, cpu_threads=args.cpu_threads, seed=config.seed)
        print("Runtime: " + json.dumps(runtime), flush=True)
        prepared = prepare_experiment(config, split="training", build_sdf=True)
        active_controls = prepared.controls
        active_control_provenance = {"kind": "recorded"}
        if args.controls_archive is not None:
            active_controls, active_control_provenance = load_condition_archive(
                args.controls_archive, args.condition,
                expected_dt=prepared.simulation_config.dt,
                expected_steps=prepared.total_steps,
                expected_duration=prepared.total_steps * prepared.simulation_config.dt,
            )
        write_new_json(output / "prepared_inputs.json", prepared.summary())
        frames = {frame.completed_substeps: frame for frame in prepared.frames}

        def active_tool_poses(completed_steps):
            index = min(max(int(completed_steps), 0), len(active_controls) - 1)
            return active_controls[index].poses.tolist()
        snapshots = output / "snapshots"
        snapshots.mkdir()
        records = []
        capacity = min(max(config.segment_length + 1, 2), prepared.total_steps + 1)
        if prepared.sdf is None:
            raise RuntimeError("SDF preparation did not produce collision data")
        stepper = Stepper(prepared.simulation_config, prepared.parameters, capacity=capacity, sdf=prepared.sdf)
        stepper.load_state(0, prepared.initial_state)
        initial_path = snapshots / "particles_000000.npy"
        np.save(initial_path, prepared.initial_state.x)
        records.append({"source_frame": 0, "original_source_frame": prepared.frames[0].original_source_frame,
                        "step": 0, "sim_time_s": 0.0, "particles": str(initial_path.relative_to(output)),
                        "tool_poses": active_tool_poses(0)})
        manifest = {
            "schema": "taichidough/dataset-forward-replay/v2", "created_at": datetime.now(timezone.utc).isoformat(),
            "config": str(args.config.resolve()), "config_sha256": file_sha256(args.config.resolve()),
            "parameters": prepared.parameters, "mass_kg": config.mass_kg,
            "density_kg_m3": config.density_kg_m3, "particle_count": config.simulation["n_particles"],
            "particle_volume_m3": config.mass_kg / config.density_kg_m3 / config.simulation["n_particles"],
            "runtime": runtime, "reference": reference, "target_steps": prepared.total_steps,
            "control_source": active_control_provenance,
            "target_source_frame": prepared.end_frame, "simulation_run": True,
            "calibration_run": False, "backward_run": False, "source_sha256": before,
        }
        write_new_json(output / "run_manifest.json", manifest)
        completed = 0
        slot = 0
        failure = None
        simulation_started = time.perf_counter()
        with (output / "progress.jsonl").open("x") as progress:
            try:
                for step in range(prepared.total_steps):
                    stepper.advance(slot, active_controls[step])
                    completed = step + 1
                    slot += 1
                    if completed in frames:
                        frame = frames[completed]
                        state = stepper.state(slot)
                        path = snapshots / f"particles_{frame.source_frame:06d}.npy"
                        np.save(path, state.x)
                        records.append({"source_frame": frame.source_frame,
                                        "original_source_frame": frame.original_source_frame,
                                        "step": completed, "sim_time_s": frame.sim_time_s,
                                        "particles": str(path.relative_to(output)),
                                        "tool_poses": active_tool_poses(completed),
                                        "bounds_min": state.x.min(axis=0).tolist(),
                                        "bounds_max": state.x.max(axis=0).tolist()})
                    if completed == 1 or completed % 1000 == 0 or completed == prepared.total_steps:
                        elapsed = time.perf_counter() - simulation_started
                        event = {"step": completed, "target_steps": prepared.total_steps,
                                 "sim_time_s": completed * prepared.simulation_config.dt,
                                 "elapsed_s": elapsed, "steps_per_s": completed / elapsed,
                                 "diagnostics": stepper.diagnostics()}
                        progress.write(json.dumps(event, allow_nan=False) + "\n")
                        progress.flush()
                        print(json.dumps(event), flush=True)
                    if slot == stepper.capacity - 1 and completed < prepared.total_steps:
                        stepper.load_state(0, stepper.state(slot))
                        slot = 0
            except Exception as error:
                failure = {"type": type(error).__name__, "message": str(error),
                           "attempted_step": completed + 1, "traceback": traceback.format_exc()}
                print("FORWARD STOPPED: " + json.dumps(failure), flush=True)
        last_state = stepper.state(slot)
        last_state.validate()
        np.savez_compressed(output / "last_valid_state.npz", **last_state.arrays())
        np.save(output / "last_valid_particles.npy", last_state.x)
        after = {path: file_sha256(Path(path)) for path in before}
        result = {
            "schema": manifest["schema"],
            "status": "completed" if failure is None and completed == prepared.total_steps else "stopped",
            "completed_steps": completed, "sim_time_s": completed * prepared.simulation_config.dt,
            "target_steps": prepared.total_steps, "last_saved_source_frame": records[-1]["source_frame"],
            "elapsed_s": time.perf_counter() - started, "failure": failure, "frames": records,
            "runtime": runtime, "last_valid_tool_poses": active_tool_poses(completed),
            "control_source": active_control_provenance,
            "source_unchanged": before == after, "source_sha256_after": after,
        }
        write_new_json(output / "simulation_result.json", result)
        print("RESULT: " + json.dumps({key: result[key] for key in
              ("status", "completed_steps", "sim_time_s", "last_saved_source_frame", "failure")}), flush=True)
        return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
