#!/usr/bin/env python3
"""Compare timestamp-paired visible surfaces without registration or time warping."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from deformpath_topview import apply_calibration, calibration_metadata, depth_to_pointcloud, filter_xyz, fixed_grid_metrics, load_calibration, rasterize_depth
    from deformpath_dynamics import ToolReplay, depth_comparison, filter_scene_points, load_observation_sequence, load_scene_point_filter, paired_frame_indices, resolve_artifact, surface_summary, tool_geometry_from_metadata
except ImportError:
    from .deformpath_topview import apply_calibration, calibration_metadata, depth_to_pointcloud, filter_xyz, fixed_grid_metrics, load_calibration, rasterize_depth
    from .deformpath_dynamics import ToolReplay, depth_comparison, filter_scene_points, load_observation_sequence, load_scene_point_filter, paired_frame_indices, resolve_artifact, surface_summary, tool_geometry_from_metadata


def replay_target_frame(replay_metadata: dict) -> str:
    collision_mode = replay_metadata.get("tool_collision", "box")
    if collision_mode not in {"box", "sdf", "none"}:
        raise ValueError(f"Replay has unsupported tool collision mode: {collision_mode}")
    target_frame = "mesh_tool_link" if collision_mode == "sdf" else "collider"
    declared_frame = replay_metadata.get("tool_pose_frame")
    if declared_frame is not None and declared_frame != target_frame:
        raise ValueError(
            f"Replay tool pose frame {declared_frame!r} is incompatible with tool collision mode {collision_mode!r}"
        )
    return target_frame


def replay_marker_transforms(replay_metadata: dict, geometry) -> np.ndarray:
    return geometry.marker_from_mesh if replay_target_frame(replay_metadata) == "mesh_tool_link" else geometry.marker_from_collider


def validate_frames(metadata: dict, calibration, view_name: str) -> tuple[list[dict], list[dict]]:
    if metadata.get("calibration", {}).get("fingerprint") != calibration.fingerprint:
        raise ValueError("Simulation calibration fingerprint does not match --calibration")
    frames = metadata.get("frames", [])
    if not frames or any("sim_time_s" not in frame for frame in frames):
        raise ValueError("Every simulation frame needs explicit sim_time_s; regenerate legacy exports")
    times = np.asarray([frame["sim_time_s"] for frame in frames], dtype=float)
    paired_frame_indices(times, times, 0)
    if times[0] != 0 or not frames[0].get("initial_state") or frames[0].get("completed_substeps") != 0:
        raise ValueError("First simulation frame must be the unadvanced initialization at time zero")
    dt = float(metadata["parameters"]["dt"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Simulation dt must be finite and positive")
    views = []
    raster_settings = None
    for frame in frames:
        count = frame.get("completed_substeps")
        if not isinstance(count, int) or count < 0 or abs(frame["sim_time_s"] - count * dt) > 1e-8:
            raise ValueError("Simulation timestamp disagrees with completed_substeps * dt")
        matches = [view for view in frame.get("views", []) if view.get("name") == view_name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {view_name!r} view in every simulation frame")
        view = matches[0]
        for key, value in calibration.camera.items():
            if key not in view:
                raise ValueError(f"Exported camera is missing calibration field {key}")
            expected = np.asarray(value)
            actual = np.asarray(view[key])
            if np.issubdtype(expected.dtype, np.number) and np.issubdtype(actual.dtype, np.number):
                matches = expected.shape == actual.shape and np.allclose(actual, expected, atol=1e-7, rtol=0)
            else:
                matches = view[key] == value
            if not matches:
                raise ValueError(f"Exported camera {key} does not match calibration")
        settings = (view["width"], view["height"], view.get("splat_radius", 0))
        if raster_settings is not None and raster_settings != settings:
            raise ValueError("Raster dimensions and splat radius must remain fixed over the sequence")
        raster_settings = settings
        views.append(view)
    return frames, views


def optical_surface(depth: np.ndarray, valid: np.ndarray, camera: dict) -> np.ndarray:
    if all(key in camera for key in ("width", "height", "fx", "fy", "cx", "cy")):
        right = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        forward = np.array([0.0, 0.0, 1.0])
        position = np.zeros(3)
    else:
        try:
            from deformpath_topview import compute_camera_basis
        except ImportError:
            from .deformpath_topview import compute_camera_basis
        right, up, forward = compute_camera_basis(camera["position"], camera["lookAt"], camera.get("up"))
        position = np.asarray(camera["position"])
    return depth_to_pointcloud(
        depth,
        np.where(valid, 0, -1),
        right,
        up,
        forward,
        position,
        camera,
        frame="camera_optical",
    )


def boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    interior = padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return np.argwhere(mask & ~interior).astype(float)


def boundary_distance(real: np.ndarray, sim: np.ndarray) -> float | None:
    a, b = boundary(real), boundary(sim)
    if not len(a) or not len(b):
        return None
    def directed(source, target):
        distances = []
        for offset in range(0, len(source), 128):
            squared = np.sum((source[offset:offset + 128, None] - target[None]) ** 2, axis=2)
            distances.extend(np.sqrt(squared.min(axis=1)).tolist())
        return np.mean(distances)
    return float((directed(a, b) + directed(b, a)) / 2)


def load_depth(view: dict, metadata_path: Path, camera: dict) -> tuple[np.ndarray, np.ndarray]:
    depth = np.load(resolve_artifact(view["depth_array"], metadata_path)).astype(np.float32)
    if depth.shape != (view["height"], view["width"]) or not np.isfinite(depth).all():
        raise ValueError("Exported raw depth has incorrect shape or nonfinite values")
    return depth, (depth > camera["zNear"]) & (depth < camera["zFar"])


def evaluate(args) -> Path:
    calibration = load_calibration(args.calibration)
    sequence = load_observation_sequence(args.episode_dir)
    metadata = json.loads(args.taichi_metadata.read_text())
    frames, views = validate_frames(metadata, calibration, args.view)
    scene_filter = load_scene_point_filter(metadata, args.taichi_metadata, calibration)
    replay_metadata = metadata.get("replay", {})
    if replay_metadata.get("sequence_fingerprint") != sequence.fingerprint:
        raise ValueError("A dynamics comparison requires this episode's timestamped captured-tool replay")
    start = int(replay_metadata["source_start_frame"])
    end = int(replay_metadata["source_end_frame"])
    if not 0 <= start <= end < len(sequence.times):
        raise ValueError("Replay source frame range is invalid")
    if abs(replay_metadata["source_time_origin_s"] - sequence.times[start]) > 1e-6:
        raise ValueError("Replay time origin does not match captured timestamps")
    if args.frame_stride < 1 or args.cell_size <= 0 or not 0 <= args.trim_quantile < .5:
        raise ValueError("Invalid frame stride, metric cell size, or trim quantile")
    indices = list(range(start, end + 1, args.frame_stride))
    if indices[-1] != end:
        indices.append(end)
    target_times = sequence.times[indices] - sequence.times[start]
    sim_times = np.asarray([frame["sim_time_s"] for frame in frames])
    tolerance = float(metadata["parameters"]["dt"]) + 1e-9 if args.pair_tolerance is None else args.pair_tolerance
    pairs = paired_frame_indices(sim_times, target_times, tolerance)
    matched = [index for index in pairs if index is not None]
    if len(matched) != len(set(matched)):
        raise ValueError("Pairing tolerance reuses a simulation frame for multiple observations; reduce the tolerance")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Evaluation output directory must be empty; choose a new run directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = args.output_dir / "arrays"
    arrays_dir.mkdir()
    initial_depth, initial_valid = load_depth(views[0], args.taichi_metadata, calibration.camera)
    initial_sim_summary = surface_summary(optical_surface(initial_depth, initial_valid, calibration.camera), args.cell_size)
    target_frame = replay_target_frame(replay_metadata)
    embedded_tool_geometry = replay_metadata.get("tool_geometry")
    if embedded_tool_geometry is None:
        if target_frame == "mesh_tool_link":
            raise ValueError("SDF replay metadata must embed marker_from_mesh tool geometry")
        marker_offset = replay_metadata.get("tool_marker_offset_source_m")
        marker_from_tool_frames = None
    else:
        geometry = tool_geometry_from_metadata(embedded_tool_geometry, sequence.names)
        marker_offset = None
        marker_from_tool_frames = replay_marker_transforms(replay_metadata, geometry)
    replay = ToolReplay(
        sequence,
        calibration,
        start,
        end,
        marker_offset=marker_offset,
        max_gap_s=replay_metadata.get("max_interpolation_gap_s", .1),
        marker_from_tool_frames=marker_from_tool_frames,
    )
    width, height, radius = views[0]["width"], views[0]["height"], views[0].get("splat_radius", 0)
    rows = []
    initial_real = None
    initial_real_valid = None
    initial_summary = {}
    for source_index, target, pair in zip(indices, target_times, pairs):
        real_points = sequence.points[source_index]
        real_points = real_points[np.isfinite(real_points[:, :3]).all(axis=1)]
        if len(real_points):
            calibrated_points = apply_calibration(filter_xyz(real_points, args.trim_quantile), calibration)
            real_scene, scene_filter_counts = filter_scene_points(calibrated_points, scene_filter)
        else:
            real_scene = np.empty((0, 3), dtype=np.float32)
            scene_filter_counts = {"input": 0, "nonfinite": 0, "kept": 0, "rejected": 0}
        real_depth, nearest, *_ = rasterize_depth(real_scene, width, height, calibration.camera, radius)
        real_valid = nearest >= 0
        real_optical = optical_surface(real_depth, real_valid, calibration.camera)
        summary_real = surface_summary(real_optical, args.cell_size)
        if initial_real is None:
            initial_real, initial_real_valid, initial_summary = real_depth.copy(), real_valid.copy(), summary_real
        row = {
            "source_frame": source_index, "original_source_frame": sequence.original_indices[source_index],
            "source_timestamp_s": float(sequence.times[source_index]), "time_s": float(target),
            "phase": "initialization" if source_index == start else "rollout (contact phase not measured)",
            "status": "unpaired" if pair is None else "paired",
            "sim_time_s": None if pair is None else float(sim_times[pair]),
            "pairing_error_s": None if pair is None else float(sim_times[pair] - target),
            "simulation_step": None if pair is None else frames[pair]["completed_substeps"],
            "tool_validity": sequence.valid[source_index].tolist(),
            "scene_filter_counts": scene_filter_counts,
            "real_summary": summary_real,
            "frozen_baseline": depth_comparison(real_depth, real_valid, initial_depth, initial_valid),
        }
        real_poses, real_velocities = replay.at(float(target))
        row["captured_tools_scene"] = real_poses.tolist()
        row["tool_speed_scene_m_s"] = np.linalg.norm(real_velocities[:, :3], axis=1).tolist()
        row["tool_separation_scene_m"] = float(np.linalg.norm(real_poses[0, :3] - real_poses[1, :3]))
        if pair is None:
            sim_depth = np.full_like(real_depth, calibration.camera["zFar"])
            sim_valid = np.zeros_like(real_valid)
            row.update({"metrics": None, "sim_summary": None, "replayed_tools_scene": None})
        else:
            frame = frames[pair]
            sim_depth, sim_valid = load_depth(views[pair], args.taichi_metadata, calibration.camera)
            sim_optical = optical_surface(sim_depth, sim_valid, calibration.camera)
            metrics = depth_comparison(real_depth, real_valid, sim_depth, sim_valid)
            metrics["boundary_mean_distance_px"] = boundary_distance(real_valid, sim_valid)
            metrics["footprint_iou"] = fixed_grid_metrics(real_optical, sim_optical, args.cell_size)["footprint_iou"] if len(real_optical) and len(sim_optical) else None
            summary_sim = surface_summary(sim_optical, args.cell_size)
            common_change = initial_real_valid & initial_valid & real_valid & sim_valid
            change_error = ((sim_depth - initial_depth) - (real_depth - initial_real))[common_change]
            metrics["depth_change_mae_m"] = float(np.abs(change_error).mean()) if len(change_error) else None
            metrics["change_support_pixels"] = int(common_change.sum())
            for axis in ("x", "y"):
                key = f"centroid_{axis}_m"
                sim0 = initial_sim_summary[key]
                values = (summary_real[key], initial_summary[key], summary_sim[key], sim0)
                metrics[f"centroid_{axis}_change_error_m"] = float((values[2] - values[3]) - (values[0] - values[1])) if all(v is not None for v in values) else None
            row.update({"metrics": metrics, "sim_summary": summary_sim, "replayed_tools_scene": frame.get("tool_poses_scene")})
            if not real_valid.any():
                row["status"] = "observation_unavailable"
                row["metrics"] = None
                row["frozen_baseline"] = {key: None for key in row["frozen_baseline"]}
        arrays_path = arrays_dir / f"frame_{source_index:06d}.npz"
        np.savez_compressed(arrays_path, real_depth=real_depth, real_valid=real_valid,
                            sim_depth=sim_depth, sim_valid=sim_valid)
        row["arrays"] = str(arrays_path.relative_to(args.output_dir))
        rows.append(row)
        print(f"Evaluate source={source_index} t={target:.6f}s status={row['status']}", flush=True)
    successful = [row for row in rows if row["status"] == "paired"]
    post_initial = [row for row in successful if row["time_s"] > 0]
    calibration_distance_note = (
        "All distances use the metric rigid source-to-scene calibration; no fitted scale is applied."
        if getattr(calibration, "is_metric", False)
        else "All distances are calibrated scene metres; source lengths differ by the legacy uniform_scale."
    )
    report = {
        "benchmark": "dynamic-topview-proxy-replay/v1",
        "scope": "initialization_only" if not post_initial else "trajectory_conditioned_proxy_rollout",
        "physical_fidelity_validated": False,
        "limitations": replay_metadata.get("limitations", []) + [
            "Visible-surface comparison only; no observed hidden volume or material-point correspondence.",
            "Unknown/occluded pixels cannot be distinguished from segmentation dropout; neither is zero error.",
            "Depth errors use common visible support; coverage must be read alongside them.",
            calibration_distance_note,
            "No contact labels are inferred from proxy geometry; phase annotations require independent evidence.",
        ],
        "calibration": calibration_metadata(calibration), "replay": replay_metadata,
        "simulation_parameters": metadata["parameters"],
        "pair_tolerance_s": tolerance, "frame_stride": args.frame_stride,
        "camera": dict(calibration.camera, width=width, height=height, splat_radius=radius),
        "metric_cell_size_m": args.cell_size, "trim_quantile": args.trim_quantile,
        "scene_point_filter": None if scene_filter is None else {
            "floor_plane_scene": scene_filter.floor_plane_scene.tolist(),
            "floor_clearance_m": scene_filter.floor_clearance_m,
            "scene_bounds": None if scene_filter.bounds_min is None else {
                "min": scene_filter.bounds_min.tolist(),
                "max": scene_filter.bounds_max.tolist(),
            },
            "reconstruction_metadata_sha256": scene_filter.reconstruction_metadata_sha256,
            "scene_frame": scene_filter.scene_frame,
        },
        "sign_convention": "simulation minus real", "depth_source": "raw metric .npy, never normalized PNG",
        "counts": {"observations": len(rows), "paired": len(successful), "post_initial_paired": len(post_initial)},
        "summary": {}, "frames": rows,
        "inputs": {"episode_dir": str(sequence.episode_dir), "sequence_fingerprint": sequence.fingerprint,
                   "taichi_metadata": str(args.taichi_metadata.resolve()), "calibration": str(args.calibration.resolve())},
    }
    for metric in ("pixel_iou", "pixel_mae_m", "pixel_p95_m", "depth_change_mae_m"):
        values = [(row["source_frame"], row["metrics"][metric]) for row in post_initial if row["metrics"].get(metric) is not None]
        if values:
            worst = min(values, key=lambda item: item[1]) if metric.endswith("iou") else max(values, key=lambda item: item[1])
            report["summary"][metric] = {"mean_post_initial": float(np.mean([v for _, v in values])), "worst_source_frame": worst[0], "worst_value": worst[1]}
    output = args.output_dir / "dynamic_topview_metrics.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    columns = ["source_frame", "original_source_frame", "source_timestamp_s", "time_s", "sim_time_s", "pairing_error_s", "simulation_step", "status", "pixel_iou", "footprint_iou", "pixel_bias_m", "pixel_mae_m", "pixel_p95_m", "real_coverage", "sim_coverage", "common_visible_pixels", "depth_change_mae_m", "boundary_mean_distance_px", "frozen_pixel_iou", "frozen_pixel_p95_m"]
    with (args.output_dir / "paired_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            values = dict(row, **(row.get("metrics") or {}), frozen_pixel_iou=row["frozen_baseline"]["pixel_iou"], frozen_pixel_p95_m=row["frozen_baseline"]["pixel_p95_m"])
            writer.writerow({key: values.get(key) for key in columns})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--taichi-metadata", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view", default="deformpath_top")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--pair-tolerance", type=float, default=None, help="Seconds; default is one integration dt. Frames outside tolerance remain visibly unpaired.")
    parser.add_argument("--cell-size", type=float, default=.003)
    parser.add_argument("--trim-quantile", type=float, default=.005)
    print(f"Wrote {evaluate(parser.parse_args())}")


if __name__ == "__main__":
    main()
