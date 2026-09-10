#!/usr/bin/env python3
"""Evaluate a DeformPath frame against a step-zero Taichi virtual top view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from deformpath_topview import (
        apply_calibration,
        calibration_metadata,
        depth_to_pointcloud,
        filter_xyz,
        fixed_grid_metrics,
        load_calibration,
        load_deformpath_frame,
        rasterize_depth,
    )
except ImportError:
    from .deformpath_topview import (
        apply_calibration,
        calibration_metadata,
        depth_to_pointcloud,
        filter_xyz,
        fixed_grid_metrics,
        load_calibration,
        load_deformpath_frame,
        rasterize_depth,
    )

try:
    from deformpath_dynamics import filter_scene_points, load_scene_point_filter
except ImportError:
    from .deformpath_dynamics import filter_scene_points, load_scene_point_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--pointclouds-name", default="pointclouds_interpolated.pt")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--taichi-metadata", type=Path, required=True)
    parser.add_argument("--virtual-pointcloud", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trim-quantile", type=float, default=0.005)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--splat-radius", type=int, default=3)
    parser.add_argument("--metric-cell-size", type=float, default=0.003)
    parser.add_argument("--min-iou", type=float, default=None)
    return parser.parse_args()


def resolve_artifact(path_value: str | Path, metadata_path: Path) -> Path:
    path = Path(path_value)
    candidates = (path,) if path.is_absolute() else (path, metadata_path.parent / path, metadata_path.parent / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve {path_value!r} from {metadata_path}")


def find_initial_view(metadata: dict[str, Any]) -> dict[str, Any]:
    for frame in metadata.get("frames", []):
        if frame.get("simulation_step", frame.get("step")) != 0:
            continue
        for view in frame.get("views", []):
            if view.get("pointcloud_frame") == "camera_optical" and view.get("pointcloud_array"):
                return view
    raise ValueError("No step-zero camera_optical pointcloud was found in Taichi metadata")


def main() -> None:
    args = parse_args()
    if args.metric_cell_size <= 0:
        raise ValueError("--metric-cell-size must be positive")
    calibration = load_calibration(args.calibration)
    metadata = json.loads(args.taichi_metadata.read_text(encoding="utf-8"))
    artifact_calibration = metadata.get("calibration")
    if not artifact_calibration or artifact_calibration.get("fingerprint") != calibration.fingerprint:
        raise ValueError("Taichi artifact does not use the supplied calibration fingerprint")
    scene_filter = load_scene_point_filter(metadata, args.taichi_metadata, calibration)

    view = find_initial_view(metadata)
    virtual_path = resolve_artifact(args.virtual_pointcloud or view["pointcloud_array"], args.taichi_metadata)
    virtual = np.load(virtual_path).astype(np.float32)
    if virtual.ndim != 2 or virtual.shape[1] < 3:
        raise ValueError(f"Expected virtual [N, >=3] pointcloud, got {virtual.shape}")
    virtual = virtual[:, :3]
    if not len(virtual) or not np.isfinite(virtual).all():
        raise ValueError("Virtual pointcloud must be finite and nonempty")

    source = load_deformpath_frame(args.episode_dir, args.pointclouds_name, args.frame)
    real_source = filter_xyz(source, args.trim_quantile)
    calibrated_scene = apply_calibration(real_source, calibration)
    real_scene, scene_filter_counts = filter_scene_points(calibrated_scene, scene_filter)
    if not len(real_scene):
        raise ValueError("No observed dough points remain after applying the reconstruction scene filter")
    if getattr(calibration, "is_metric", False):
        width = int(calibration.camera["width"])
        height = int(calibration.camera["height"])
        if int(view.get("width", width)) != width or int(view.get("height", height)) != height:
            raise ValueError("Taichi view dimensions do not match the v2 calibrated camera")
    else:
        width, height = args.width, args.height
    splat_radius = int(view.get("splat_radius", args.splat_radius))
    depth, nearest, right, up, forward, position = rasterize_depth(
        real_scene,
        width,
        height,
        calibration.camera,
        splat_radius=splat_radius,
    )
    real_optical = depth_to_pointcloud(
        depth, nearest, right, up, forward, position, calibration.camera, frame="camera_optical"
    )
    if not len(real_optical):
        raise ValueError("Calibrated real pointcloud is not visible from the benchmark camera")

    metrics = fixed_grid_metrics(real_optical, virtual, args.metric_cell_size)
    result = {
        "benchmark": "static-topview-initialization/v1",
        "source": {
            "episode_dir": str(args.episode_dir),
            "pointclouds_name": args.pointclouds_name,
            "frame": args.frame,
            "input_shape": list(source.shape),
            "filtered_points": int(len(real_source)),
            "scene_filter_counts": scene_filter_counts,
            "visible_points": int(len(real_optical)),
        },
        "virtual": {
            "pointcloud_path": str(virtual_path),
            "taichi_metadata": str(args.taichi_metadata),
            "point_count": int(len(virtual)),
            "view": view.get("name"),
            "simulation_step": 0,
        },
        "calibration": calibration_metadata(calibration),
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
        "metrics": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "real_visible_camera_optical_xyz.npy", real_optical)
    result_path = args.output_dir / "static_topview_metrics.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {result_path}")
    if args.min_iou is not None and metrics["footprint_iou"] < args.min_iou:
        raise SystemExit(f"Footprint IoU {metrics['footprint_iou']:.4f} is below --min-iou {args.min_iou:.4f}")


if __name__ == "__main__":
    main()
