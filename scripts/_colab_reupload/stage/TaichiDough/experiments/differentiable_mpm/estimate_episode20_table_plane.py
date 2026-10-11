"""Independently validate the Episode20 mocap table plane from full-resolution RGB-D frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from experiments.differentiable_mpm.estimate_table_plane import (
    HSV_DEFAULTS,
    PlaneFit,
    aggregate_equal_frame,
    bootstrap_planes,
    exclude_oriented_boxes,
    fit_frame_plane,
    hsv_mask,
    orient_plane_upward,
    quaternion_matrix,
    regression_from_plane,
    signed_distance,
    summarize,
)
from experiments.differentiable_mpm.table_alignment import table_alignment_transform


SCHEMA = "taichidough/episode20-table-plane-validation/v1"
POSE_FRAMES = ("UR5e_spathla", "gen3_spathla")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def torch_load(path: Path, *, mmap: bool = False) -> Any:
    options: dict[str, Any] = {"map_location": "cpu", "weights_only": True}
    if mmap:
        options["mmap"] = True
    try:
        return torch.load(path, **options)
    except TypeError:
        options.pop("weights_only", None)
        options.pop("mmap", None)
        return torch.load(path, **options)


def normalized_plane(value: Any) -> np.ndarray:
    plane = np.asarray(value, dtype=np.float64)
    if plane.shape != (4,) or not np.isfinite(plane).all():
        raise ValueError("Floor plane must contain four finite coefficients")
    length = float(np.linalg.norm(plane[:3]))
    if length <= 1e-12:
        raise ValueError("Floor plane normal must be nonzero")
    return plane / length


def plane_height(plane: np.ndarray, reference_xz: np.ndarray) -> float:
    if abs(float(plane[1])) <= 1e-12:
        raise ValueError("Table plane cannot be vertical")
    return float(-(plane[0] * reference_xz[0] + plane[2] * reference_xz[1] + plane[3]) / plane[1])


def compare_planes(estimated: Any, declared: Any, reference_xz: Any, *,
                   maximum_angle_deg: float = 2.0, maximum_height_difference_m: float = 0.015) -> dict[str, Any]:
    estimated_plane = normalized_plane(estimated)
    declared_plane = normalized_plane(declared)
    if estimated_plane[1] < 0:
        estimated_plane *= -1
    if declared_plane[1] < 0:
        declared_plane *= -1
    reference = np.asarray(reference_xz, dtype=np.float64)
    if reference.shape != (2,) or not np.isfinite(reference).all():
        raise ValueError("reference_xz must contain two finite values")
    angle = float(math.degrees(math.acos(np.clip(np.dot(estimated_plane[:3], declared_plane[:3]), -1, 1))))
    estimated_height = plane_height(estimated_plane, reference)
    declared_height = plane_height(declared_plane, reference)
    difference = estimated_height - declared_height
    return {
        "estimated_plane": estimated_plane.tolist(),
        "declared_plane": declared_plane.tolist(),
        "normal_angle_deg": angle,
        "estimated_height_at_reference_m": estimated_height,
        "declared_height_at_reference_m": declared_height,
        "height_difference_m": difference,
        "maximum_angle_deg": maximum_angle_deg,
        "maximum_abs_height_difference_m": maximum_height_difference_m,
        "passed": angle <= maximum_angle_deg and abs(difference) <= maximum_height_difference_m,
    }


def load_inputs(episode: Path, calibration_path: Path, geometry_path: Path) -> tuple[Any, list[Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    raw_payload = torch_load(episode / "pointclouds.pt", mmap=True)
    filtered_payload = torch_load(episode / "pointclouds_interpolated.pt")
    paths_payload = torch_load(episode / "paths_interpolated.pt")
    sequence = json_object(episode / "sequence_metadata.json")
    calibration = json_object(calibration_path)
    geometry = json_object(geometry_path)
    if not (isinstance(raw_payload, list) and len(raw_payload) == 1 and isinstance(raw_payload[0], list)):
        raise ValueError("pointclouds.pt must contain one episode list")
    if not (isinstance(filtered_payload, list) and len(filtered_payload) == 1 and isinstance(filtered_payload[0], list)):
        raise ValueError("pointclouds_interpolated.pt must contain one episode list")
    if not (isinstance(paths_payload, list) and len(paths_payload) == 1 and isinstance(paths_payload[0], dict)):
        raise ValueError("paths_interpolated.pt must contain one episode dictionary")
    raw_frames = raw_payload[0]
    filtered = filtered_payload[0]
    paths_record = paths_payload[0]
    paths = paths_record.get("path")
    validity = paths_record.get("stream_validity")
    if not isinstance(paths, torch.Tensor) or not isinstance(validity, torch.Tensor):
        raise ValueError("Interpolated path payload is incomplete")
    paths_array = paths.detach().cpu().numpy().astype(np.float64)
    validity_array = validity.detach().cpu().numpy().astype(bool)
    raw_indices = np.asarray(sequence.get("pointcloud_indices"), dtype=np.int64)
    if len(raw_frames) != 250 or len(filtered) != len(paths_array) or len(paths_array) != len(raw_indices):
        raise ValueError("Unexpected Episode20 frame alignment")
    if paths_record.get("pose_frames") != list(POSE_FRAMES) or paths_array.shape[1:] != (2, 14):
        raise ValueError("Episode20 paths must name the two expected streams in [T,2,14] layout")
    if validity_array.shape != (len(paths_array), 2) or not validity_array.all():
        raise ValueError("Both Episode20 tool streams must be valid")
    if calibration.get("source_frame") != "mocap" or calibration.get("scene_frame") != "mocap":
        raise ValueError("Table validation requires a mocap-coordinate Episode20 export")
    transform = np.asarray(calibration.get("scene_from_source"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.allclose(transform, np.eye(4), rtol=0, atol=1e-12):
        raise ValueError("Table validation requires identity scene_from_source")
    tools = {row.get("name"): row for row in geometry.get("tools", []) if isinstance(row, dict)}
    if set(POSE_FRAMES) - set(tools):
        raise ValueError("Tool geometry is missing an Episode20 stream")
    half_extents = np.asarray([tools[name]["half_extents_m"] for name in POSE_FRAMES], dtype=np.float64)
    for name in POSE_FRAMES:
        marker_from_collider = np.asarray(tools[name].get("marker_from_collider"), dtype=np.float64)
        if marker_from_collider.shape != (4, 4) or not np.allclose(
            marker_from_collider, np.eye(4), rtol=0, atol=1e-12
        ):
            raise ValueError("Table validation requires identity marker_from_collider transforms")
    return raw_frames, filtered, paths_array, validity_array, raw_indices, calibration, sequence | {"half_extents": half_extents}


def estimate(episode: Path, calibration_path: Path, geometry_path: Path, *, seed: int = 2020,
             tool_padding_m: float = 0.01, roi_margin_m: float = 0.06,
             sample_limit: int = 4000, sensitivity_sample_limit: int = 2000,
             ransac_iterations: int = 96, threshold_m: float = 0.0035,
             bootstrap_iterations: int = 500, maximum_angle_deg: float = 2.0,
             maximum_height_difference_m: float = 0.015,
             minimum_dough_above_floor_fraction: float = 0.9) -> dict[str, Any]:
    episode, calibration_path, geometry_path = (Path(value).resolve() for value in (episode, calibration_path, geometry_path))
    if min(tool_padding_m, roi_margin_m, threshold_m) < 0 or min(sample_limit, sensitivity_sample_limit, ransac_iterations, bootstrap_iterations) < 1:
        raise ValueError("Sampling, iterations, threshold, and padding must be positive or non-negative as appropriate")
    raw_frames, filtered, paths, validity, raw_indices, calibration, metadata = load_inputs(episode, calibration_path, geometry_path)
    half_extents = np.asarray(metadata.pop("half_extents"), dtype=np.float64)
    recorded_filter = metadata.get("pointcloud_filter", {})
    recorded_thresholds = tuple(recorded_filter.get("hue_range", []) + recorded_filter.get("saturation_range", []) + recorded_filter.get("value_range", []))
    if recorded_thresholds != HSV_DEFAULTS:
        raise ValueError("Episode20 interpolation metadata uses unexpected HSV thresholds")
    frame_stats = recorded_filter.get("frame_stats")
    if not isinstance(frame_stats, list) or len(frame_stats) != len(raw_frames):
        raise ValueError("Episode20 interpolation metadata is missing per-frame HSV counts")
    poses_by_raw = {int(raw): paths[index, :, :7] if validity[index].all() else None for index, raw in enumerate(raw_indices)}
    dough_xz = np.concatenate([np.asarray(frame)[:, [0, 2]] for frame in filtered], axis=0)
    lower = np.quantile(dough_xz, 0.005, axis=0)
    upper = np.quantile(dough_xz, 0.995, axis=0)
    signs = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    tool_corners: list[np.ndarray] = []
    for pose_pair in poses_by_raw.values():
        if pose_pair is None:
            continue
        for tool_index in range(2):
            rotation = quaternion_matrix(pose_pair[tool_index, 3:7])
            corners = pose_pair[tool_index, :3] + (signs * (half_extents[tool_index] + tool_padding_m)) @ rotation.T
            tool_corners.append(corners)
    if not tool_corners:
        raise ValueError("No synchronized tool poses are available for table validation")
    all_tool_corners = np.concatenate(tool_corners, axis=0)
    roi_min = np.minimum(lower, all_tool_corners[:, [0, 2]].min(axis=0)) - roi_margin_m
    roi_max = np.maximum(upper, all_tool_corners[:, [0, 2]].max(axis=0)) + roi_margin_m
    positive_points = np.asarray(filtered[0])[:, :3].astype(np.float64)
    reference_xz = np.median(positive_points[:, [0, 2]], axis=0)

    rows: list[dict[str, Any]] = []
    fits: list[PlaneFit] = []
    sensitivity_fits: list[PlaneFit] = []
    candidates_by_frame: dict[int, np.ndarray] = {}
    for raw_index, tensor in enumerate(raw_frames):
        values = np.asarray(tensor)
        xyz = values[:, :3].astype(np.float64)
        finite = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
        rgb = np.clip(np.rint(values[:, 7:10]), 0, 255).astype(np.uint8)
        dough = finite & hsv_mask(rgb, HSV_DEFAULTS)
        roi = finite & ~dough & np.all((xyz[:, [0, 2]] >= roi_min) & (xyz[:, [0, 2]] <= roi_max), axis=1)
        keep, excluded = exclude_oriented_boxes(xyz, poses_by_raw.get(raw_index), half_extents, tool_padding_m)
        candidate = xyz[roi & keep]
        candidates_by_frame[raw_index] = candidate
        fit = fit_frame_plane(candidate, seed=seed + raw_index * 17, sample_limit=sample_limit,
                              iterations=ransac_iterations, threshold_m=threshold_m)
        sensitivity = fit_frame_plane(candidate, seed=seed + raw_index * 17,
                                      sample_limit=sensitivity_sample_limit,
                                      iterations=ransac_iterations, threshold_m=threshold_m)
        fits.append(fit)
        sensitivity_fits.append(sensitivity)
        expected_hsv = int(frame_stats[raw_index]["hsv_filtered_point_count"])
        rows.append({
            "raw_frame": raw_index,
            "split": "heldout" if raw_index % 5 == 0 else "estimation",
            "candidate_points": int(len(candidate)),
            "hsv_dough_points": int(dough.sum()),
            "expected_hsv_dough_points": expected_hsv,
            "hsv_count_matches_metadata": expected_hsv is None or expected_hsv == int(dough.sum()),
            "tool_pose_available": poses_by_raw.get(raw_index) is not None,
            "tool_points_excluded": excluded,
            "accepted": fit.accepted,
            "accepted_sensitivity": sensitivity.accepted,
            "inlier_count": fit.inlier_count,
            "inlier_fraction": fit.inlier_fraction,
            "median_abs_residual_m": fit.median_abs_residual_m,
            "reason": fit.reason,
        })

    estimation = [fit for fit, row in zip(fits, rows) if row["split"] == "estimation" and fit.accepted]
    heldout = [(fit, row) for fit, row in zip(fits, rows) if row["split"] == "heldout" and fit.accepted]
    sensitivity = [fit for fit, row in zip(sensitivity_fits, rows) if row["split"] == "estimation" and fit.accepted]
    if len(estimation) < 20 or len(heldout) < 5 or len(sensitivity) < 20:
        raise ValueError("Too few Episode20 frames produced accepted table-plane fits")
    _, estimated_plane = aggregate_equal_frame(estimation, positive_points)
    estimated_plane = orient_plane_upward(estimated_plane, positive_points)
    _, sensitivity_plane = aggregate_equal_frame(sensitivity, positive_points)
    sensitivity_plane = orient_plane_upward(sensitivity_plane, positive_points)
    comparison = compare_planes(estimated_plane, calibration["floor_plane_scene"], reference_xz,
                                maximum_angle_deg=maximum_angle_deg,
                                maximum_height_difference_m=maximum_height_difference_m)
    sensitivity_comparison = compare_planes(sensitivity_plane, estimated_plane, reference_xz,
                                            maximum_angle_deg=0.5, maximum_height_difference_m=0.002)
    unplaced_transform = table_alignment_transform(estimated_plane)
    filtered_points = np.concatenate(
        [np.asarray(frame)[:, :3].astype(np.float64) for frame in filtered], axis=0
    )
    combined = np.concatenate([filtered_points, all_tool_corners], axis=0)
    aligned_unplaced = combined @ unplaced_transform[:3, :3].T + unplaced_transform[:3, 3]
    translation_xz = 0.5 - 0.5 * (
        aligned_unplaced.min(axis=0) + aligned_unplaced.max(axis=0)
    )[[0, 2]]
    placed_transform = table_alignment_transform(estimated_plane, translation_xz=translation_xz)
    aligned_placed = combined @ placed_transform[:3, :3].T + placed_transform[:3, 3]
    heldout_rows = []
    for fit, row in heldout:
        points = candidates_by_frame[row["raw_frame"]]
        independently_selected = points[np.abs(signed_distance(points, fit.plane)) <= threshold_m]
        residual = signed_distance(independently_selected, estimated_plane)
        heldout_rows.append({
            "raw_frame": row["raw_frame"],
            "independent_table_points": int(len(independently_selected)),
            "aggregate_inlier_fraction": float(np.mean(np.abs(residual) <= threshold_m)),
            "median_signed_residual_m": float(np.median(residual)),
            "median_abs_residual_m": float(np.median(np.abs(residual))),
            "p95_abs_residual_m": float(np.quantile(np.abs(residual), 0.95)),
        })
    dough_distance = signed_distance(filtered_points, estimated_plane)
    finite_dough = dough_distance[np.isfinite(dough_distance)]
    above_fraction = float(np.mean(finite_dough >= 0.003))
    heldout_median = np.asarray([row["median_abs_residual_m"] for row in heldout_rows])
    heldout_p95 = np.asarray([row["p95_abs_residual_m"] for row in heldout_rows])
    checks = {
        "raw_frame_count_250": len(raw_frames) == 250,
        "synchronized_frames_match_hsv_metadata": all(row["hsv_count_matches_metadata"] for row in rows),
        "estimated_normal_points_upward": bool(estimated_plane[1] > 0),
        "enough_estimation_frames": len(estimation) >= 180,
        "enough_heldout_frames": len(heldout) >= 45,
        "sample_limit_sensitivity_passed": sensitivity_comparison["passed"],
        "heldout_median_abs_residual_below_2mm": bool(float(np.median(heldout_median)) <= 0.002),
        "heldout_p95_abs_residual_below_5mm": bool(float(np.median(heldout_p95)) <= 0.005),
        "dough_above_estimated_floor_fraction": above_fraction >= minimum_dough_above_floor_fraction,
        "placed_geometry_inside_horizontal_unit_domain": bool(
            np.all(aligned_placed[:, [0, 2]].min(axis=0) >= 0.0)
            and np.all(aligned_placed[:, [0, 2]].max(axis=0) < 1.0)
        ),
    }
    inputs = [episode / "pointclouds.pt", episode / "pointclouds_interpolated.pt",
              episode / "paths_interpolated.pt", episode / "sequence_metadata.json",
              calibration_path, geometry_path, Path(__file__).resolve(),
              Path(__file__).with_name("estimate_table_plane.py")]
    return {
        "schema": SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "episode": str(episode),
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
        "settings": {
            "seed": seed, "hsv_opencv": list(HSV_DEFAULTS), "tool_padding_m": tool_padding_m,
            "roi_margin_m": roi_margin_m, "sample_limit": sample_limit,
            "sensitivity_sample_limit": sensitivity_sample_limit, "ransac_iterations": ransac_iterations,
            "threshold_m": threshold_m, "bootstrap_iterations": bootstrap_iterations,
            "maximum_angle_deg": maximum_angle_deg,
            "maximum_height_difference_m": maximum_height_difference_m,
            "minimum_dough_above_floor_fraction": minimum_dough_above_floor_fraction,
            "split": "heldout when raw frame ordinal modulo 5 is zero; estimation otherwise",
        },
        "roi_min_xz_m": roi_min.tolist(),
        "roi_max_xz_m": roi_max.tolist(),
        "reference_xz_m": reference_xz.tolist(),
        "estimated_plane_scene": estimated_plane.tolist(),
        "regression_y_equals_a_x_plus_b_z_plus_c": regression_from_plane(estimated_plane).tolist(),
        "source_mocap_calibration_sha256": sha256_file(calibration_path),
        "declared_mocap_y0_comparison": comparison,
        "declared_mocap_y0_is_measurement_reference_only": True,
        "sample_limit_sensitivity": sensitivity_comparison,
        "recommended_translation_xz_m": translation_xz.tolist(),
        "table_alignment_transform": placed_transform.tolist(),
        "placed_combined_geometry_bounds": {
            "min": aligned_placed.min(axis=0).tolist(),
            "max": aligned_placed.max(axis=0).tolist(),
        },
        "bootstrap_equal_frame_95pct": bootstrap_planes(estimation, positive_points, reference_xz,
                                                         seed=seed + 991, iterations=bootstrap_iterations),
        "frame_counts": {
            "raw": len(raw_frames), "synchronized": len(filtered),
            "accepted_estimation": len(estimation), "accepted_heldout": len(heldout),
        },
        "heldout": {
            "median_abs_residual_m": summarize(heldout_median),
            "p95_abs_residual_m": summarize(heldout_p95),
            "frames": heldout_rows,
        },
        "dough_signed_distance_to_estimated_floor_m": summarize(finite_dough),
        "dough_fraction_at_least_3mm_above_estimated_floor": above_fraction,
        "per_frame": rows,
        "scene_geometry_measurement_run": True,
        "material_calibration_optimization_run": False,
        "fit_run": False,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing table-plane report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.rename(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--tool-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--tool-padding-m", type=float, default=0.01)
    parser.add_argument("--roi-margin-m", type=float, default=0.06)
    parser.add_argument("--sample-limit", type=int, default=4000)
    parser.add_argument("--sensitivity-sample-limit", type=int, default=2000)
    parser.add_argument("--ransac-iterations", type=int, default=96)
    parser.add_argument("--threshold-m", type=float, default=0.0035)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--maximum-angle-deg", type=float, default=2.0)
    parser.add_argument("--maximum-height-difference-m", type=float, default=0.015)
    parser.add_argument("--minimum-dough-above-floor-fraction", type=float, default=0.9)
    args = parser.parse_args(argv)
    report = estimate(
        args.episode, args.calibration, args.tool_geometry, seed=args.seed,
        tool_padding_m=args.tool_padding_m, roi_margin_m=args.roi_margin_m,
        sample_limit=args.sample_limit, sensitivity_sample_limit=args.sensitivity_sample_limit,
        ransac_iterations=args.ransac_iterations, threshold_m=args.threshold_m,
        bootstrap_iterations=args.bootstrap_iterations, maximum_angle_deg=args.maximum_angle_deg,
        maximum_height_difference_m=args.maximum_height_difference_m,
        minimum_dough_above_floor_fraction=args.minimum_dough_above_floor_fraction,
    )
    write_report(args.output, report)
    print(args.output.resolve())
    if not report["passed"]:
        raise ValueError("Episode20 table-plane validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
