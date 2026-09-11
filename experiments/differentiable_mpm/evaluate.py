"""Independent exports and scoring through the unchanged strict evaluator."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping

import numpy as np

from .config import EXPERIMENT_ROOT, REPOSITORY_ROOT, canonical_hash, file_sha256
from .reference_adapter import current_reference_policy, get_reference_modules, verify_reference, verified_reference_path
from .state import ParticleState, validate_parameters


STRICT_DEFAULTS = {
    "weights": {"depth_change": 1.0, "mask_iou": 1.0, "observed_to_simulation_distance": 1.0, "real_coverage": 1.0},
    "depth_scale_m": 0.01, "distance_scale_m": 0.01, "huber_delta": 1.0,
    "min_common_pixels": 50, "min_observed_pixels": 50, "min_simulation_pixels": 50,
    "min_observed_points": 50, "min_simulation_points": 50, "nearest_chunk_size": 1024,
    "require_all_frames": True,
}


def require_owned_output(path):
    path = Path(path).expanduser().resolve()
    if EXPERIMENT_ROOT not in path.parents:
        raise ValueError("Generated output must be inside experiments/differentiable_mpm")
    return path


def fresh_directory(path):
    path = require_owned_output(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output directory is not empty; choose a new experiment run: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_complete_frames(records, expected_frames):
    """An evaluator success alone does not prove that all frames were exported."""
    actual = [row.get("source_frame") for row in records]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in actual):
        raise ValueError("Every frame must have an integer source_frame")
    expected = list(expected_frames)
    if actual != expected:
        raise ValueError(f"Exact source-frame set/order mismatch: expected {expected}, got {actual}")


def export_replay(prepared, positions_by_step: Mapping[int, np.ndarray | ParticleState], output_dir,
                  parameters=None, runtime_metadata=None) -> Path:
    """Write initial state and every observation frame at the strict camera resolution."""
    parameters = dict(prepared.parameters if parameters is None else parameters)
    validate_parameters(parameters)
    expected_steps = {frame.completed_substeps for frame in prepared.frames}
    if expected_steps - positions_by_step.keys():
        raise ValueError("Replay export is missing observation states: " + str(sorted(expected_steps - positions_by_step.keys())))
    arrays = {}
    for step in expected_steps:
        value = positions_by_step[step]
        x = value.x if isinstance(value, ParticleState) else np.asarray(value)
        if x.shape != prepared.initial_state.x.shape or not np.isfinite(x).all():
            raise ValueError(f"Invalid replay positions at completed substep {step}")
        arrays[step] = x
    if not np.array_equal(arrays[0], prepared.initial_state.x):
        raise ValueError("Exported state zero must be the unadvanced reconstruction")
    if any(a.sim_time_s >= b.sim_time_s for a, b in zip(prepared.frames, prepared.frames[1:])):
        raise ValueError("Strict evaluator requires a distinct simulation timestamp for each observed frame")
    directory = fresh_directory(output_dir)
    helpers = get_reference_modules()
    camera = prepared.calibration.camera
    sim = prepared.simulation_config
    geometry = helpers.dynamics.tool_geometry_metadata(prepared.tool_geometry)
    reconstruction = prepared.reconstruction_info
    metadata = {
        "schema": "taichidough/differentiable-replay-export/v1", "generator": "experiments/differentiable_mpm/evaluate.py",
        "experiment_fingerprint": prepared.fingerprint,
        "calibration": helpers.topview.calibration_metadata(prepared.calibration),
        "initialization": {"particle_path": str(prepared.config.paths["initial_particles"]),
                           "particle_sha256": reconstruction["particle_sha256"],
                           "metadata_path": reconstruction["metadata_path"], "metadata_sha256": reconstruction["metadata_sha256"],
                           "scene_frame": reconstruction["scene_frame"], "floor_plane_scene": reconstruction["floor_plane_scene"]},
        "mass": prepared.mass_properties, "parameters": {**asdict(sim), **parameters},
        "runtime": runtime_metadata,
        "replay": {"mode": "captured_trajectory_sdf_tools" if sim.tool_collision == "sdf" else "captured_trajectory_no_tools",
                   "tool_collision": sim.tool_collision,
                   "tool_pose_frame": "mesh_tool_link" if sim.tool_collision == "sdf" else "collider",
                   "episode_dir": str(prepared.sequence.episode_dir), "sequence_fingerprint": prepared.sequence.fingerprint,
                   "source_start_frame": 0, "source_end_frame": prepared.end_frame,
                   "source_time_origin_s": float(prepared.sequence.times[0]), "tool_names": list(prepared.sequence.names),
                   "tool_geometry": geometry, "max_interpolation_gap_s": prepared.config.replay_max_gap_s,
                   "tool_contact_padding_scene_m": sim.tool_contact_padding,
                   "solid_collision_assets": prepared.provenance["collision"],
                   "limitations": ["Calibration is supplied, not independently estimated.",
                                   "Depth exports contain dough only; tool occlusion is not modeled.",
                                   "Controls use recorded piecewise-linear translations and SLERP rotations."]},
        "frames": [],
    }
    for index, frame in enumerate(prepared.frames):
        points = np.asarray(arrays[frame.completed_substeps], dtype=np.float32)
        particles_path = directory / f"particles_{index:06d}.npy"
        depth_path = directory / f"deformpath_top_depth_{index:06d}.npy"
        depth, _, *_ = helpers.topview.rasterize_depth(points, camera["width"], camera["height"], camera,
                                                       prepared.config.observation.strict_splat_radius)
        np.save(particles_path, points)
        np.save(depth_path, depth)
        controls = prepared.controls.at_completed_step(frame.completed_substeps)
        row = {"frame": index, "step": frame.completed_substeps, "simulation_step": frame.completed_substeps,
               **asdict(frame), "initial_state": frame.completed_substeps == 0,
               "pairing_error_s": frame.pairing_error_s, "particles": str(particles_path),
               "tool_poses_scene": controls.poses.tolist(), "tool_velocities_scene": controls.velocities.tolist(),
               "tool_validity": prepared.sequence.valid[frame.source_frame].tolist(),
               "views": [{**dict(camera), "name": "deformpath_top", "splat_radius": prepared.config.observation.strict_splat_radius,
                          "depth_array": str(depth_path)}]}
        metadata["frames"].append(row)
    require_complete_frames(metadata["frames"], range(prepared.end_frame + 1))
    metadata_path = directory / "camera_parameters.json"
    with metadata_path.open("x") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return metadata_path


def _strict_loss_module():
    return get_reference_modules().material_calibration


def score_strict_evaluation(evaluation_path, scored_frames, loss_settings=None):
    """Use the unchanged partial_view_loss; additionally require exact scored frames."""
    evaluation_path = Path(evaluation_path).resolve()
    report = json.loads(evaluation_path.read_text())
    if report.get("benchmark") != "dynamic-topview-proxy-replay/v1":
        raise ValueError("Unsupported strict evaluator benchmark")
    rows = report.get("frames", [])
    if not rows or rows[0].get("source_frame") != 0 or rows[0].get("time_s") != 0:
        raise ValueError("Strict evaluation is missing its initial reference frame")
    if not scored_frames:
        raise ValueError("No scored frames requested")
    selected = [row for row in rows if row.get("source_frame") in set(scored_frames)]
    require_complete_frames(selected, scored_frames)
    settings = {**STRICT_DEFAULTS, **(loss_settings or {})}
    if set(settings) - set(STRICT_DEFAULTS):
        raise ValueError("Unknown strict loss settings")
    if settings.pop("require_all_frames") is not True:
        raise ValueError("Strict calibration scoring requires every selected frame")
    helpers = get_reference_modules()
    strict = _strict_loss_module()
    camera = report["camera"]

    def load(row):
        path = helpers.dynamics.resolve_artifact(row["arrays"], evaluation_path)
        with np.load(path, allow_pickle=False) as data:
            return tuple(data[key].copy() for key in ("real_depth", "real_valid", "sim_depth", "sim_valid"))

    real0, real_valid0, sim0, sim_valid0 = load(rows[0])
    initial_simulation_points = strict.visible_optical_points(sim0, sim_valid0, camera)
    values, frozen, records = [], [], []
    for row in selected:
        if row.get("status") != "paired":
            loss = {"valid": False, "weighted_total": None,
                    "failure_reason": f"Strict evaluator frame status is {row.get('status')!r}",
                    "components": {name: None for name in strict.LOSS_COMPONENTS}}
            values.append(loss)
            frozen.append(loss)
            records.append({"source_frame": row["source_frame"], **loss})
            continue
        real, real_valid, sim_depth, sim_valid = load(row)
        real_points = strict.visible_optical_points(real, real_valid, camera)
        sim_points = strict.visible_optical_points(sim_depth, sim_valid, camera)
        loss = strict.partial_view_loss(real, real_valid, sim_depth, sim_valid, real0, real_valid0, sim0, sim_valid0,
                                        real_points, sim_points, **settings)
        initial = strict.partial_view_loss(real, real_valid, sim0, sim_valid0, real0, real_valid0, sim0, sim_valid0,
                                           real_points, initial_simulation_points, **settings)
        values.append(loss)
        frozen.append(initial)
        records.append({"source_frame": row["source_frame"], "time_s": row["time_s"], **loss})
    return {**strict.aggregate_loss_results(values, require_all=True, minimum_valid=1),
            "frames": records, "scored_frames": list(scored_frames),
            "frozen_baseline": strict.aggregate_loss_results(frozen, require_all=True, minimum_valid=1),
            "source_evaluation": str(evaluation_path),
            "loss_source_path": str(Path(strict.__file__).resolve()),
            "loss_source_sha256": file_sha256(Path(strict.__file__))}


def run_strict_evaluation(prepared, simulation_metadata, output_dir, timeout_s=3600):
    """Run the existing CLI into a fresh directory and score only the selected split."""
    simulation_metadata = Path(simulation_metadata).resolve()
    metadata = json.loads(simulation_metadata.read_text())
    require_complete_frames(metadata.get("frames", []), range(prepared.end_frame + 1))
    if metadata.get("experiment_fingerprint") != prepared.fingerprint:
        raise ValueError("Replay export does not match the prepared experiment")
    if not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive and finite")
    verify_reference()
    directory = fresh_directory(output_dir)
    evaluation_dir = directory / "evaluation"
    evaluator_path = verified_reference_path("scripts/evaluate_dynamic_topview_match.py")
    command = [sys.executable, "-m", "experiments.differentiable_mpm.reference_adapter",
               "--reference-policy", current_reference_policy(),
               "--import-report", str(directory / "evaluator_imports.json"), "--",
               "--episode-dir", str(prepared.config.paths["episode"]),
               "--taichi-metadata", str(simulation_metadata), "--calibration", str(prepared.config.paths["calibration"]),
               "--output-dir", str(evaluation_dir), "--view", "deformpath_top", "--frame-stride", "1",
               "--pair-tolerance", repr(prepared.simulation_config.dt), "--cell-size", "0.003",
               "--trim-quantile", repr(prepared.config.observation.trim_quantile)]
    (directory / "evaluator_command.json").write_text(json.dumps({"argv": command,
        "evaluator_source": str(evaluator_path), "evaluator_source_sha256": file_sha256(evaluator_path),
        "reference_policy": current_reference_policy()}, indent=2) + "\n")
    start = time.perf_counter()
    result = {"status": "failed", "valid": False, "split": prepared.split,
              "scored_frames": list(prepared.scored_frames), "command": command,
              "simulation_metadata": str(simulation_metadata), "experiment_fingerprint": prepared.fingerprint}
    with (directory / "evaluator.stdout.log").open("x") as stdout, (directory / "evaluator.stderr.log").open("x") as stderr:
        try:
            completed = subprocess.run(command, cwd=REPOSITORY_ROOT, stdout=stdout, stderr=stderr,
                                       timeout=timeout_s, check=False)
            result["returncode"] = completed.returncode
        except subprocess.TimeoutExpired:
            result["failure_reason"] = f"Strict evaluator exceeded timeout {timeout_s} seconds"
            completed = None
    result["elapsed_s"] = time.perf_counter() - start
    if completed is not None and completed.returncode != 0:
        result["failure_reason"] = f"Strict evaluator returned {completed.returncode}; see evaluator.stderr.log"
    elif completed is not None:
        report_path = evaluation_dir / "dynamic_topview_metrics.json"
        if not report_path.is_file():
            result["failure_reason"] = f"Strict evaluator did not write {report_path.name}"
        else:
            try:
                report = json.loads(report_path.read_text())
                require_complete_frames(report.get("frames", []), range(prepared.end_frame + 1))
                loss = score_strict_evaluation(report_path, prepared.scored_frames, prepared.config.strict_loss)
                result.update(status="completed" if loss["valid"] else "invalid", valid=loss["valid"], loss=loss,
                              evaluation_path=str(report_path), failure_reason=loss.get("failure_reason"))
            except (ValueError, KeyError, OSError) as exc:
                result["failure_reason"] = str(exc)
    (directory / "strict_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
