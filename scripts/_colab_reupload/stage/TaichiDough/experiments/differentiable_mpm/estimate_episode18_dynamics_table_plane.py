"""Estimate the Episode 18 dynamics table plane from raw RGB-D frames (measurement only).

Reuses the Episode 18 kugla per-frame RANSAC, equal-frame aggregation and bootstrap.
Differences: the raw frames come from the calibrated 566-frame export selected through
``pointcloud_indices``, and tools are removed by distance to the posed closed collision
meshes (``source_from_marker @ marker_from_mesh @ visual @ STL``) instead of box proxies.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree

from .estimate_table_plane import (aggregate_equal_frame, bootstrap_planes, file_sha256, fit_frame_plane,
                                   hsv_mask, json_write, quaternion_matrix, regression_from_plane,
                                   signed_distance, summarize)
from .reference_adapter import get_reference_modules, reference_policy

SCHEMA = "taichidough/table-plane-estimate/v1"
TOOL_HSV = (0.0, 69.0, 96.0, 255.0, 189.0, 255.0)
VISUAL_KEYS = {"UR5e_spathla": ("UR_TOOL_VISUAL_ORIGIN", "UR_TOOL_VISUAL_RPY"),
               "gen3_spathla": ("KINOVA_TOOL_VISUAL_ORIGIN", "KINOVA_TOOL_VISUAL_RPY")}


def surface_samples(vertices, faces_per_mm2=0.25, seed=0):
    """Area-weighted surface samples of an unindexed triangle mesh in metres."""
    triangles = vertices.reshape(-1, 3, 3)
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    count = max(2000, int(areas.sum() * 1e6 * faces_per_mm2))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(triangles), count, p=areas / areas.sum())
    u, v = rng.random((2, count))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = triangles[chosen]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def posed(points_tool, pose, marker_from_mesh):
    transform = np.eye(4)
    transform[:3, :3] = quaternion_matrix(pose[3:7])
    transform[:3, 3] = pose[:3]
    transform = transform @ marker_from_mesh
    return points_tool @ transform[:3, :3].T + transform[:3, 3]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True, help="sequence directory with paths_interpolated.pt")
    parser.add_argument("--raw-pointclouds", type=Path, required=True, help="raw calibrated export pointclouds.pt")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--static-calibration", type=Path, required=True, help="spatula_static_calibration.json")
    parser.add_argument("--ur-mesh", type=Path, required=True)
    parser.add_argument("--kinova-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1818)
    parser.add_argument("--tool-padding-m", type=float, default=.015)
    parser.add_argument("--roi-margin-m", type=float, default=.06)
    parser.add_argument("--sample-limit", type=int, default=4000)
    parser.add_argument("--sensitivity-sample-limit", type=int, default=2000)
    parser.add_argument("--ransac-iterations", type=int, default=96)
    parser.add_argument("--threshold-m", type=float, default=.0035)
    parser.add_argument("--minimum-inliers", type=int, default=300)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--reference-policy", choices=("strict", "frozen"), default="strict")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    for name in ("episode", "raw_pointclouds", "calibration", "static_calibration", "ur_mesh", "kinova_mesh", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    with reference_policy(args.reference_policy):
        simulator = get_reference_modules().simulator
    args.output_dir.mkdir(parents=True)
    source = Path(__file__).resolve()
    source_start = file_sha256(source)
    metadata_path = args.episode / "sequence_metadata.json"
    filtered_path = args.episode / "pointclouds_interpolated.pt"
    paths_path = args.episode / "paths_interpolated.pt"
    inputs = (args.raw_pointclouds, filtered_path, paths_path, metadata_path, args.calibration,
              args.static_calibration, args.ur_mesh, args.kinova_mesh)
    input_hashes = {str(path): file_sha256(path) for path in inputs}
    metadata = json.loads(metadata_path.read_text())
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("scene_frame") != "mocap" or calibration.get("source_frame") != "mocap":
        raise ValueError("Estimator requires points and poses in mocap coordinates")
    if not np.allclose(calibration["scene_from_source"], np.eye(4), atol=1e-9, rtol=0):
        raise ValueError("Raw points are expected to already be in mocap/scene coordinates")
    recorded = metadata["pointcloud_filter"]
    dough_hsv = tuple(recorded["hue_range"] + recorded["saturation_range"] + recorded["value_range"])
    import torch
    raw_frames = torch.load(args.raw_pointclouds, map_location="cpu", weights_only=True, mmap=True)[0]
    filtered = torch.load(filtered_path, map_location="cpu", weights_only=True)[0]
    paths_record = torch.load(paths_path, map_location="cpu", weights_only=True)[0]
    paths = paths_record["path"].numpy()
    validity = paths_record["stream_validity"].numpy().astype(bool)
    raw_indices = np.asarray(metadata["pointcloud_indices"], dtype=int)
    recorded_stats = recorded["frame_stats"]
    if not (len(filtered) == len(paths) == len(raw_indices)) or len(recorded_stats) != len(raw_frames):
        raise ValueError("Unexpected sequence frame alignment")
    if raw_indices.max() >= len(raw_frames):
        raise ValueError("Sequence indices exceed raw export length")
    static = json.loads(args.static_calibration.read_text())["tools"]
    order = list(paths_record["pose_frames"])
    meshes = {"UR5e_spathla": args.ur_mesh, "gen3_spathla": args.kinova_mesh}
    trees, samples_tool, marker_from_mesh = [], [], []
    for index, name in enumerate(order):
        origin_key, rpy_key = VISUAL_KEYS[name]
        vertices, _ = simulator.load_binary_stl(meshes[name], 0.001)
        vertices = simulator.mesh_in_tool_frame(vertices, getattr(simulator, origin_key),
                                                simulator.rpy_to_matrix(getattr(simulator, rpy_key)))
        samples_tool.append(surface_samples(np.asarray(vertices, dtype=np.float64), seed=args.seed + index))
        transform = np.asarray(static[name]["marker_from_mesh"], dtype=np.float64)
        np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-9)
        marker_from_mesh.append(transform)
    # ROI is fixed before plane fitting from DBSCAN-filtered dough and the posed mesh sweep.
    dough_xz = np.concatenate([np.asarray(frame)[:, [0, 2]] for frame in filtered])
    lower, upper = np.quantile(dough_xz, .005, axis=0), np.quantile(dough_xz, .995, axis=0)
    sweep = []
    for i in range(0, len(paths), 5):
        for j in range(2):
            if validity[i, j]:
                sweep.append(posed(samples_tool[j][::50], paths[i, j, :7], marker_from_mesh[j])[:, [0, 2]])
    sweep = np.concatenate(sweep)
    roi_min = np.minimum(lower, sweep.min(axis=0)) - args.roi_margin_m
    roi_max = np.maximum(upper, sweep.max(axis=0)) + args.roi_margin_m
    positive_points = np.asarray(filtered[0])[:, :3].astype(np.float64)
    reference_xz = np.median(positive_points[:, [0, 2]], axis=0)
    settings = {"seed": args.seed, "dough_hsv_opencv": list(dough_hsv), "tool_hsv_opencv": list(TOOL_HSV),
                "split": "heldout when raw_frame % 5 == 0; estimation otherwise",
                "roi_source": "0.5%-99.5% DBSCAN-filtered dough XZ union posed tool-mesh XZ sweep, then fixed margin",
                "roi_min_xz_m": roi_min.tolist(), "roi_max_xz_m": roi_max.tolist(), "roi_margin_m": args.roi_margin_m,
                "tool_exclusion": "points within tool_padding_m of posed closed-mesh surface samples removed; tool HSV removed",
                "tool_padding_m": args.tool_padding_m, "sample_limit": args.sample_limit,
                "sensitivity_sample_limit": args.sensitivity_sample_limit, "ransac_iterations": args.ransac_iterations,
                "threshold_m": args.threshold_m, "maximum_tilt_deg": 15., "minimum_inliers": args.minimum_inliers,
                "minimum_fraction": .25, "bootstrap_iterations": args.bootstrap_iterations, "equal_frame_weighting": True}
    json_write(args.output_dir / "run_manifest.json", {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
               "action": "read-only table geometry measurement", "inputs": input_hashes,
               "source_sha256": source_start, "settings": settings})

    def candidates(k):
        values = raw_frames[int(raw_indices[k])].numpy()
        xyz = values[:, :3].astype(np.float64)
        # Same spatial mask as interpolate_deformpath_pairs.py, so dough counts match its metadata.
        valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
        rgb = np.clip(np.rint(values[:, 7:10]), 0, 255).astype(np.uint8)
        dough = valid & hsv_mask(rgb, dough_hsv)
        tool_colour = valid & hsv_mask(rgb, TOOL_HSV)
        roi = valid & ~dough & ~tool_colour & np.all((xyz[:, [0, 2]] >= roi_min) & (xyz[:, [0, 2]] <= roi_max), axis=1)
        near_tool = np.zeros(len(xyz), dtype=bool)
        for j in range(2):
            if validity[k, j]:
                tree = cKDTree(posed(samples_tool[j], paths[k, j, :7], marker_from_mesh[j]))
                near_tool |= np.isfinite(tree.query(np.where(valid[:, None], xyz, 1e3),
                                                    distance_upper_bound=args.tool_padding_m)[0])
        return xyz[roi & ~near_tool], dict(valid=int(valid.sum()), dough=int(dough.sum()), tool_colour=int(tool_colour.sum()),
                                          roi=int(roi.sum()), near_tool=int((roi & near_tool).sum()), all_valid=bool(validity[k].all()))

    rows, main_fits, sensitivity_fits, cached = [], [], [], {}
    for k, raw in enumerate(raw_indices):
        candidate, counts = candidates(k)
        main_fit = fit_frame_plane(candidate, seed=args.seed + int(raw) * 17, sample_limit=args.sample_limit,
                                   iterations=args.ransac_iterations, threshold_m=args.threshold_m,
                                   minimum_inliers=args.minimum_inliers)
        sensitivity = fit_frame_plane(candidate, seed=args.seed + int(raw) * 17, sample_limit=args.sensitivity_sample_limit,
                                      iterations=args.ransac_iterations, threshold_m=args.threshold_m,
                                      minimum_inliers=args.minimum_inliers)
        main_fits.append(main_fit)
        sensitivity_fits.append(sensitivity)
        split = "heldout" if raw % 5 == 0 else "estimation"
        if split == "heldout":
            cached[k] = candidate
        row = {"frame": k, "raw_frame": int(raw), "split": split, "valid_points": counts["valid"],
               "hsv_dough_points": counts["dough"], "hsv_count_matches_metadata": counts["dough"] == int(recorded_stats[int(raw)]["hsv_filtered_point_count"]),
               "tool_colour_points": counts["tool_colour"], "roi_non_dough_before_tool": counts["roi"],
               "tool_poses_valid": counts["all_valid"], "roi_points_near_tool": counts["near_tool"],
               "candidate_points": len(candidate), "accepted_main": main_fit.accepted, "accepted_sensitivity": sensitivity.accepted,
               "sample_count": main_fit.sample_count, "inlier_count": main_fit.inlier_count,
               "inlier_fraction": main_fit.inlier_fraction, "median_abs_residual_m": main_fit.median_abs_residual_m,
               "reason": main_fit.reason}
        if main_fit.accepted:
            a, b, c = main_fit.coefficients
            row.update(a=a, b=b, c=c, nx=main_fit.plane[0], ny=main_fit.plane[1], nz=main_fit.plane[2], d=main_fit.plane[3],
                       height_at_reference_m=float(a * reference_xz[0] + b * reference_xz[1] + c),
                       tilt_deg=float(math.degrees(math.acos(np.clip(main_fit.plane[1], -1, 1)))))
        else:
            row.update({name: None for name in ("a", "b", "c", "nx", "ny", "nz", "d", "height_at_reference_m", "tilt_deg")})
        rows.append(row)
        if k % 50 == 0:
            print(f"frame {k}/{len(raw_indices) - 1} raw={raw} candidates={len(candidate)} accepted={main_fit.accepted}", flush=True)
    estimation = [fit for fit, row in zip(main_fits, rows) if row["split"] == "estimation" and fit.accepted]
    heldout = [(fit, row) for fit, row in zip(main_fits, rows) if row["split"] == "heldout" and fit.accepted]
    heldout_total = sum(row["split"] == "heldout" for row in rows)
    coefficients, plane = aggregate_equal_frame(estimation, positive_points)
    _, sensitivity_plane = aggregate_equal_frame([fit for fit, row in zip(sensitivity_fits, rows)
                                                  if row["split"] == "estimation" and fit.accepted], positive_points)
    height = float(coefficients[0] * reference_xz[0] + coefficients[1] * reference_xz[1] + coefficients[2])
    # Held-out table points are chosen by each frame's own fit, then scored against the aggregate.
    per_heldout, samples = [], []
    for fit, row in heldout:
        candidate = cached[row["frame"]]
        points = candidate[np.abs(signed_distance(candidate, fit.plane)) <= args.threshold_m]
        residual = signed_distance(points, plane)
        inliers = np.abs(residual) <= args.threshold_m
        per_heldout.append({"frame": row["frame"], "raw_frame": row["raw_frame"], "candidate_count": len(candidate),
                            "independent_table_point_count": len(points), "aggregate_inlier_count": int(inliers.sum()),
                            "aggregate_inlier_fraction": float(inliers.mean()),
                            "median_signed_residual_m": float(np.median(residual)),
                            "median_abs_residual_m": float(np.median(np.abs(residual)))})
        rng = np.random.default_rng(args.seed + row["raw_frame"] * 101)
        ids = rng.choice(len(points), min(1000, len(points)), replace=False)
        samples.append(np.column_stack((points[ids], residual[ids])))
    sample = np.concatenate(samples)
    linear_design = np.column_stack((sample[:, 0], sample[:, 2], np.ones(len(sample))))
    linear = np.linalg.lstsq(linear_design, sample[:, 3], rcond=None)[0]
    linear_rmse = float(np.sqrt(np.mean((sample[:, 3] - linear_design @ linear) ** 2)))
    bootstrap = bootstrap_planes(estimation, positive_points, reference_xz, seed=args.seed + 991,
                                 iterations=args.bootstrap_iterations)
    normals = np.stack([fit.plane[:3] for fit in estimation])
    angular = np.degrees(np.arccos(np.clip(normals @ plane[:3], -1, 1)))
    frame_heights = np.array([fit.coefficients @ [reference_xz[0], reference_xz[1], 1.0] for fit in estimation])
    sensitivity_angle = float(math.degrees(math.acos(np.clip(np.dot(plane[:3], sensitivity_plane[:3]), -1, 1))))
    sensitivity_height = float(regression_from_plane(sensitivity_plane) @ [reference_xz[0], reference_xz[1], 1.0])
    dough_distance = signed_distance(positive_points, plane)
    all_dough_bottom = np.array([np.quantile(signed_distance(np.asarray(f)[:, :3], plane), .02) for f in filtered])
    summary = {"schema": SCHEMA, "episode": str(args.episode), "raw_pointclouds": str(args.raw_pointclouds),
               "sequence_frames": len(rows), "accepted_estimation_frames": len(estimation),
               "accepted_heldout_frames": len(heldout), "heldout_frames": heldout_total,
               "rejected_frames": int(sum(not fit.accepted for fit in main_fits)), "settings": settings,
               "plane_convention": "normalized [nx,ny,nz,d], n dot p + d = 0; positive is dough/free-space side",
               "estimated_plane_scene": plane.tolist(), "regression_y_equals_a_x_plus_b_z_plus_c": coefficients.tolist(),
               "normal_is_upward": bool(plane[1] > 0), "initial_dough_median_signed_distance_m": float(np.median(dough_distance)),
               "dough_2pct_height_above_plane_m": summarize(all_dough_bottom),
               "reference_xz_m": reference_xz.tolist(), "height_at_reference_xz_m": height,
               "tilt_from_scene_y_deg": float(math.degrees(math.acos(np.clip(plane[1], -1, 1)))),
               "bootstrap_equal_frame_95pct": bootstrap, "estimation_frame_angular_deviation_deg": summarize(angular),
               "estimation_frame_height_at_reference_m": summarize(frame_heights),
               "heldout_equal_frame_metrics": {
                   "aggregate_inlier_fraction_on_independently_selected_table_points": summarize(np.array([x["aggregate_inlier_fraction"] for x in per_heldout])),
                   "median_signed_residual_m": summarize(np.array([x["median_signed_residual_m"] for x in per_heldout])),
                   "median_abs_residual_m": summarize(np.array([x["median_abs_residual_m"] for x in per_heldout]))},
               "heldout_sample_point_metrics": {"count": len(sample), "signed_residual_m": summarize(sample[:, 3]),
                   "absolute_residual_m": summarize(np.abs(sample[:, 3])),
                   "linear_residual_bias_coefficients_x_z_intercept": linear.tolist(), "linear_residual_bias_rmse_m": linear_rmse},
               "sensitivity": {"main_sample_limit": args.sample_limit, "secondary_sample_limit": args.sensitivity_sample_limit,
                   "plane_secondary": sensitivity_plane.tolist(), "normal_angle_difference_deg": sensitivity_angle,
                   "height_at_reference_difference_m": sensitivity_height - height},
               "interpretation": {"geometry_measurement_only": True, "derived_calibration_written": False,
                                  "simulation_or_calibration_run": False},
               "input_hashes": input_hashes, "source_sha256": source_start}
    for name, table in (("per_frame.csv", rows), ("heldout.csv", per_heldout)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    json_write(args.output_dir / "summary.json", summary)
    shutil.copyfile(source, args.output_dir / (source.name + ".txt"))
    print(json.dumps({"plane": plane.tolist(), "height_at_reference_m": height, "tilt_deg": summary["tilt_from_scene_y_deg"],
                      "accepted_estimation": len(estimation), "accepted_heldout": f"{len(heldout)}/{heldout_total}",
                      "heldout_median_abs_mm": 1000 * summary["heldout_equal_frame_metrics"]["median_abs_residual_m"]["median"],
                      "sensitivity": summary["sensitivity"]}, indent=2), flush=True)
    input_hashes_end = {str(path): file_sha256(path) for path in inputs}
    outputs = {p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()}
    checks = {"source_unchanged_during_execution": file_sha256(source) == source_start,
              "inputs_unchanged_during_execution": input_hashes_end == input_hashes,
              "all_sequence_frames_processed": len(rows) == len(raw_indices),
              "hsv_counts_match_metadata_all_frames": all(row["hsv_count_matches_metadata"] for row in rows),
              "heldout_acceptance_at_least_90pct": len(heldout) >= 0.9 * heldout_total,
              "upward_normal": bool(plane[1] > 0), "initial_dough_positive": float(np.median(dough_distance)) > 0,
              "heldout_median_abs_residual_below_threshold": summary["heldout_equal_frame_metrics"]["median_abs_residual_m"]["median"] < args.threshold_m / 2,
              "main_vs_secondary_height_difference_below_1mm": abs(sensitivity_height - height) < .001,
              "main_vs_secondary_normal_difference_below_0.25deg": sensitivity_angle < .25}
    json_write(args.output_dir / "verification.json", {"schema": SCHEMA, "passed": all(checks.values()), "checks": checks,
               "input_sha256_end": input_hashes_end, "output_sha256": outputs})
    print(json.dumps(checks, indent=2), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
