#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from deformpath_topview import (
        apply_calibration,
        calibration_metadata,
        filter_xyz,
        load_calibration,
        load_deformpath_frame,
    )
except ImportError:
    from .deformpath_topview import (
        apply_calibration,
        calibration_metadata,
        filter_xyz,
        load_calibration,
        load_deformpath_frame,
    )


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a filled voxel dough volume from one DeformPath pointcloud frame."
    )
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--pointclouds-name", default="pointclouds_interpolated.pt")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("data/voxel_reconstruction"))
    parser.add_argument("--num-particles", type=int, default=24000)
    parser.add_argument("--voxel-size", type=float, default=0.003)
    parser.add_argument(
        "--fill-axis",
        choices=tuple(AXIS_TO_INDEX),
        default=None,
        help="Axis along which the visible surface is filled. Defaults to y for floor mode and legacy z otherwise.",
    )
    parser.add_argument(
        "--fill-direction",
        choices=("positive", "negative"),
        default=None,
        help="Fill direction. Defaults to negative for floor mode and legacy positive otherwise.",
    )
    parser.add_argument(
        "--fill-mode",
        choices=("floor", "thickness", "limit"),
        default=None,
        help=(
            "How each visible surface column is filled. When omitted, --fill-limit selects limit mode; "
            "otherwise the legacy 0.045 m thickness is used."
        ),
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=None,
        help="Volume thickness along the fill axis. Thickness mode defaults to the legacy 0.045 m.",
    )
    parser.add_argument(
        "--fill-limit",
        type=float,
        default=None,
        help="Absolute fill-axis coordinate used by limit mode.",
    )
    parser.add_argument(
        "--floor-clearance",
        type=float,
        default=0.003,
        help="Minimum signed distance above the calibrated floor retained as dough in floor mode.",
    )
    parser.add_argument(
        "--floor-min-thickness",
        type=float,
        default=0.001,
        help="Minimum accepted surface-to-floor distance in floor mode.",
    )
    parser.add_argument(
        "--floor-max-thickness",
        type=float,
        default=0.25,
        help="Maximum accepted surface-to-floor distance in floor mode.",
    )
    parser.add_argument(
        "--plane-parallel-epsilon",
        type=float,
        default=1e-8,
        help="Minimum absolute plane/ray dot product accepted for floor intersections.",
    )
    parser.add_argument(
        "--scene-bounds",
        type=float,
        nargs=6,
        default=None,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        help="Optional scene-coordinate bounds applied after calibrated floor segmentation.",
    )
    parser.add_argument(
        "--bbox-padding",
        type=float,
        default=0.006,
        help="Padding added around the input pointcloud bounds before voxelization.",
    )
    parser.add_argument(
        "--trim-quantile",
        type=float,
        default=0.005,
        help="Symmetric XYZ quantile trimming for outlier removal. Use 0 to disable.",
    )
    parser.add_argument(
        "--surface-percentile",
        type=float,
        default=None,
        help="Surface percentile per footprint column. Default uses min for positive fill and max for negative fill.",
    )
    parser.add_argument(
        "--footprint-dilate",
        type=int,
        default=1,
        help="Number of 8-neighbor dilation passes used to close small footprint holes.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview-max-points", type=int, default=12000)
    parser.add_argument("--preview-sampled-points", type=int, default=1024)
    parser.add_argument("--write-preview", action="store_true", help="Write the optional Plotly HTML reconstruction preview.")
    parser.add_argument("--save-pt", action="store_true", help="Also save sampled particles as torch .pt tensors.")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Optional static-top-view calibration JSON recorded in reconstruction metadata.",
    )
    return parser.parse_args()


def load_pointcloud_frame(episode_dir: Path, pointclouds_name: str, frame_index: int) -> np.ndarray:
    return load_deformpath_frame(episode_dir, pointclouds_name, frame_index)


def filter_points(points: np.ndarray, trim_quantile: float) -> np.ndarray:
    return filter_xyz(points, trim_quantile)


def resolve_fill_settings(
    fill_mode: str | None,
    thickness: float | None,
    fill_limit: float | None,
) -> tuple[str, float | None, float | None, bool]:
    """Resolve explicit fill settings while retaining the pre-mode CLI behavior."""
    if fill_mode is None:
        if fill_limit is not None:
            return "limit", None, float(fill_limit), True
        resolved_thickness = 0.045 if thickness is None else float(thickness)
        if not np.isfinite(resolved_thickness) or resolved_thickness <= 0.0:
            raise ValueError("--thickness must be positive and finite")
        return "thickness", resolved_thickness, None, True

    if fill_mode == "floor":
        if thickness is not None or fill_limit is not None:
            raise ValueError("--fill-mode floor cannot be combined with --thickness or --fill-limit")
        return "floor", None, None, False
    if fill_mode == "thickness":
        if fill_limit is not None:
            raise ValueError("--fill-mode thickness cannot be combined with --fill-limit")
        resolved_thickness = 0.045 if thickness is None else float(thickness)
        if not np.isfinite(resolved_thickness) or resolved_thickness <= 0.0:
            raise ValueError("--thickness must be positive and finite")
        return "thickness", resolved_thickness, None, False
    if fill_mode == "limit":
        if thickness is not None:
            raise ValueError("--fill-mode limit cannot be combined with --thickness")
        if fill_limit is None or not np.isfinite(fill_limit):
            raise ValueError("--fill-mode limit requires a finite --fill-limit")
        return "limit", None, float(fill_limit), False
    raise ValueError(f"Unsupported fill mode {fill_mode!r}")


def normalize_plane(plane: Any) -> np.ndarray:
    """Return normalized [nx, ny, nz, d] for the plane n dot p + d = 0."""
    if isinstance(plane, dict):
        if "coefficients" in plane:
            values = plane["coefficients"]
        elif "normal" in plane:
            normal = np.asarray(plane["normal"], dtype=np.float64)
            if "point" in plane:
                point = np.asarray(plane["point"], dtype=np.float64)
                values = [*normal, -float(np.dot(normal, point))]
            else:
                offset = plane.get("offset", plane.get("d", plane.get("distance")))
                if offset is None:
                    raise ValueError("Floor plane with a normal requires offset, d, distance, or point")
                values = [*normal, offset]
        else:
            values = [plane.get(key) for key in ("a", "b", "c", "d")]
    else:
        values = plane
    coefficients = np.asarray(values, dtype=np.float64)
    if coefficients.shape != (4,) or not np.isfinite(coefficients).all():
        raise ValueError("Floor plane must contain four finite coefficients")
    normal_norm = float(np.linalg.norm(coefficients[:3]))
    if normal_norm <= 1e-12:
        raise ValueError("Floor plane normal must be nonzero")
    return coefficients / normal_norm


def _schema_is_v2(schema: Any) -> bool:
    if isinstance(schema, (int, float)):
        return int(schema) == 2
    text = str(schema).strip().lower()
    return text == "v2" or text.endswith("/v2") or text.endswith("_v2") or text.endswith("-v2")


def require_metric_floor_calibration(calibration: Any) -> np.ndarray:
    schema = getattr(calibration, "schema", None)
    if not _schema_is_v2(schema) or getattr(calibration, "is_metric", None) is not True:
        raise ValueError("Floor fill requires a metric v2 calibration")
    plane = getattr(calibration, "floor_plane_scene", None)
    if plane is None:
        raise ValueError("Floor fill requires floor_plane_scene in the calibration")
    scene_frame = getattr(calibration, "scene_frame", None)
    if not scene_frame:
        raise ValueError("Metric v2 calibration must name its scene_frame")
    return normalize_plane(plane)


def transform_and_segment_floor_points(
    source_points: np.ndarray,
    calibration: Any,
    clearance: float,
    scene_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    floor_plane = require_metric_floor_calibration(calibration)
    transformed = np.asarray(apply_calibration(source_points, calibration), dtype=np.float32)
    if transformed.shape != np.asarray(source_points).shape or not np.isfinite(transformed).all():
        raise ValueError("Calibration produced invalid scene-coordinate points")
    segmented, counts = segment_scene_points(transformed, floor_plane, clearance, scene_bounds)
    return transformed, segmented, floor_plane, counts


def parse_scene_bounds(values: list[float] | tuple[float, ...] | None) -> tuple[np.ndarray, np.ndarray] | None:
    if values is None:
        return None
    bounds = np.asarray(values, dtype=np.float64)
    if bounds.shape != (6,) or not np.isfinite(bounds).all():
        raise ValueError("--scene-bounds must contain six finite values")
    lower = bounds[[0, 2, 4]]
    upper = bounds[[1, 3, 5]]
    if np.any(lower >= upper):
        raise ValueError("Each --scene-bounds minimum must be less than its maximum")
    return lower.astype(np.float32), upper.astype(np.float32)


def segment_scene_points(
    scene_points: np.ndarray,
    floor_plane: Any,
    clearance: float,
    scene_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    points = np.asarray(scene_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected scene points with shape [N, 3], got {points.shape}")
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError("Floor clearance must be finite and non-negative")
    plane = normalize_plane(floor_plane)
    finite = np.isfinite(points).all(axis=1)
    signed_distance = points @ plane[:3] + plane[3]
    above_floor = finite & np.isfinite(signed_distance) & (signed_distance >= clearance)
    if scene_bounds is None:
        inside_bounds = np.ones(len(points), dtype=bool)
    else:
        lower, upper = scene_bounds
        inside_bounds = np.all((points >= lower[None, :]) & (points <= upper[None, :]), axis=1)
    kept = above_floor & inside_bounds
    result = points[kept].astype(np.float32, copy=False)
    if len(result) == 0:
        raise ValueError("Floor segmentation removed every point")
    return result, {
        "input": int(len(points)),
        "nonfinite": int((~finite).sum()),
        "at_or_below_clearance": int((finite & ~above_floor).sum()),
        "outside_scene_bounds": int((above_floor & ~inside_bounds).sum()),
        "kept": int(kept.sum()),
    }


def intersect_points_with_plane(
    surface_points: np.ndarray,
    axis_direction: Any,
    floor_plane: Any,
    minimum_thickness: float,
    maximum_thickness: float,
    parallel_epsilon: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Intersect p(t)=p0+t*a with a plane and reject invalid local thicknesses."""
    points = np.asarray(surface_points, dtype=np.float64)
    direction = np.asarray(axis_direction, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected surface points with shape [N, 3], got {points.shape}")
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("Axis direction must contain three finite values")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ValueError("Axis direction must be nonzero")
    direction /= direction_norm
    if (
        not np.isfinite(minimum_thickness)
        or not np.isfinite(maximum_thickness)
        or minimum_thickness < 0.0
        or maximum_thickness <= minimum_thickness
    ):
        raise ValueError("Floor thickness limits must be finite and satisfy 0 <= minimum < maximum")
    if not np.isfinite(parallel_epsilon) or parallel_epsilon <= 0.0:
        raise ValueError("Plane parallel epsilon must be positive and finite")

    plane = normalize_plane(floor_plane)
    denominator = float(np.dot(plane[:3], direction))
    count = len(points)
    thicknesses = np.full(count, np.nan, dtype=np.float64)
    rejection_counts = {
        "nonfinite": 0,
        "parallel": 0,
        "wrong_direction": 0,
        "below_minimum_thickness": 0,
        "above_maximum_thickness": 0,
    }
    finite_points = np.isfinite(points).all(axis=1)
    rejection_counts["nonfinite"] = int((~finite_points).sum())
    if abs(denominator) <= parallel_epsilon:
        rejection_counts["parallel"] = int(finite_points.sum())
    else:
        candidate = -(points @ plane[:3] + plane[3]) / denominator
        finite_candidate = finite_points & np.isfinite(candidate)
        rejection_counts["nonfinite"] += int((finite_points & ~np.isfinite(candidate)).sum())
        wrong_direction = finite_candidate & (candidate <= 0.0)
        below_minimum = finite_candidate & ~wrong_direction & (candidate < minimum_thickness)
        above_maximum = finite_candidate & ~wrong_direction & (candidate > maximum_thickness)
        accepted = finite_candidate & ~wrong_direction & ~below_minimum & ~above_maximum
        rejection_counts["wrong_direction"] = int(wrong_direction.sum())
        rejection_counts["below_minimum_thickness"] = int(below_minimum.sum())
        rejection_counts["above_maximum_thickness"] = int(above_maximum.sum())
        thicknesses[accepted] = candidate[accepted]

    accepted_values = thicknesses[np.isfinite(thicknesses)]
    statistics = None
    if len(accepted_values):
        statistics = {
            "min": float(np.min(accepted_values)),
            "max": float(np.max(accepted_values)),
            "mean": float(np.mean(accepted_values)),
            "median": float(np.median(accepted_values)),
            "p05": float(np.quantile(accepted_values, 0.05)),
            "p95": float(np.quantile(accepted_values, 0.95)),
        }
    return thicknesses.astype(np.float32), {
        "columns": count,
        "accepted": int(len(accepted_values)),
        "rejected": int(count - len(accepted_values)),
        "rejection_counts": rejection_counts,
        "local_thickness_m": statistics,
        "axis_direction": direction.tolist(),
        "plane_denominator": denominator,
    }


def plane_fill_limits_from_surface(
    surface: np.ndarray,
    voxel_size: float,
    fill_axis: int,
    horizontal_axes: tuple[int, int],
    horizontal_origin_indices: tuple[int, int],
    fill_positive: bool,
    floor_plane: Any,
    minimum_thickness: float,
    maximum_thickness: float,
    parallel_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    surface_points = surface_points_from_grid(
        surface, voxel_size, fill_axis, horizontal_axes, horizontal_origin_indices
    )
    direction = np.zeros(3, dtype=np.float32)
    direction[fill_axis] = 1.0 if fill_positive else -1.0
    thicknesses, statistics = intersect_points_with_plane(
        surface_points,
        direction,
        floor_plane,
        minimum_thickness,
        maximum_thickness,
        parallel_epsilon,
    )
    limits = np.full(surface.shape, np.nan, dtype=np.float32)
    accepted_surface = np.full(surface.shape, np.nan, dtype=np.float32)
    for (i, j), surface_point, local_thickness in zip(
        np.argwhere(np.isfinite(surface)), surface_points, thicknesses
    ):
        if not np.isfinite(local_thickness):
            continue
        accepted_surface[i, j] = surface[i, j]
        limits[i, j] = surface_point[fill_axis] + direction[fill_axis] * local_thickness
    return accepted_surface, limits, statistics


def particle_array_sha256(points: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(points, dtype=np.float32))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def build_surface_grid(
    xyz: np.ndarray,
    voxel_size: float,
    fill_axis: int,
    fill_positive: bool,
    bbox_padding: float,
    surface_percentile: float | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    axes = [axis for axis in range(3) if axis != fill_axis]
    horizontal_axes = (axes[0], axes[1])
    bbox_min = xyz.min(axis=0) - bbox_padding
    bbox_max = xyz.max(axis=0) + bbox_padding

    h0_min = int(np.floor(bbox_min[horizontal_axes[0]] / voxel_size))
    h0_max = int(np.ceil(bbox_max[horizontal_axes[0]] / voxel_size))
    h1_min = int(np.floor(bbox_min[horizontal_axes[1]] / voxel_size))
    h1_max = int(np.ceil(bbox_max[horizontal_axes[1]] / voxel_size))
    shape = (h0_max - h0_min + 1, h1_max - h1_min + 1)
    buckets: list[list[list[float]]] = [[[] for _ in range(shape[1])] for _ in range(shape[0])]

    h0 = np.floor(xyz[:, horizontal_axes[0]] / voxel_size).astype(np.int32) - h0_min
    h1 = np.floor(xyz[:, horizontal_axes[1]] / voxel_size).astype(np.int32) - h1_min
    f = xyz[:, fill_axis]
    for i, j, value in zip(h0, h1, f):
        if 0 <= i < shape[0] and 0 <= j < shape[1]:
            buckets[int(i)][int(j)].append(float(value))

    surface = np.full(shape, np.nan, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if not buckets[i][j]:
                continue
            values = np.asarray(buckets[i][j], dtype=np.float32)
            if surface_percentile is not None:
                surface[i, j] = np.percentile(values, surface_percentile)
            elif fill_positive:
                surface[i, j] = np.min(values)
            else:
                surface[i, j] = np.max(values)
    return surface, np.asarray(bbox_min, dtype=np.float32), (h0_min, h1_min), horizontal_axes


def dilate_surface(surface: np.ndarray, iterations: int) -> np.ndarray:
    result = surface.copy()
    for _ in range(max(0, iterations)):
        updated = result.copy()
        missing = np.argwhere(~np.isfinite(result))
        for i, j in missing:
            i0 = max(0, i - 1)
            i1 = min(result.shape[0], i + 2)
            j0 = max(0, j - 1)
            j1 = min(result.shape[1], j + 2)
            neighborhood = result[i0:i1, j0:j1]
            valid = neighborhood[np.isfinite(neighborhood)]
            if valid.size:
                updated[i, j] = float(valid.mean())
        result = updated
    return result


def voxel_centers_from_surface(
    surface: np.ndarray,
    voxel_size: float,
    fill_axis: int,
    horizontal_axes: tuple[int, int],
    horizontal_origin_indices: tuple[int, int],
    fill_positive: bool,
    thickness: float | None,
    fill_limit: float | None,
    fill_limits_by_column: np.ndarray | None = None,
) -> np.ndarray:
    occupied_columns = np.argwhere(np.isfinite(surface))
    centers: list[list[float]] = []
    direction = 1.0 if fill_positive else -1.0
    if fill_limits_by_column is not None and fill_limits_by_column.shape != surface.shape:
        raise ValueError("Per-column fill limits must match the surface grid shape")
    for i, j in occupied_columns:
        surface_value = float(surface[i, j])
        if fill_limits_by_column is not None:
            limit = float(fill_limits_by_column[i, j])
            if not np.isfinite(limit):
                continue
        elif fill_limit is not None:
            limit = float(fill_limit)
        else:
            if thickness is None:
                raise ValueError("A thickness, scalar fill limit, or per-column fill limits are required")
            limit = surface_value + direction * float(thickness)
        lower = min(surface_value, limit)
        upper = max(surface_value, limit)
        k0 = int(np.floor(lower / voxel_size))
        k1 = int(np.ceil(upper / voxel_size))
        h0_center = (horizontal_origin_indices[0] + int(i) + 0.5) * voxel_size
        h1_center = (horizontal_origin_indices[1] + int(j) + 0.5) * voxel_size
        for k in range(k0, k1 + 1):
            fill_center = (k + 0.5) * voxel_size
            if fill_center < lower or fill_center > upper:
                continue
            point = [0.0, 0.0, 0.0]
            point[horizontal_axes[0]] = h0_center
            point[horizontal_axes[1]] = h1_center
            point[fill_axis] = fill_center
            centers.append(point)
    if not centers:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(centers, dtype=np.float32)


def surface_points_from_grid(
    surface: np.ndarray,
    voxel_size: float,
    fill_axis: int,
    horizontal_axes: tuple[int, int],
    horizontal_origin_indices: tuple[int, int],
) -> np.ndarray:
    points: list[list[float]] = []
    for i, j in np.argwhere(np.isfinite(surface)):
        point = [0.0, 0.0, 0.0]
        point[horizontal_axes[0]] = (horizontal_origin_indices[0] + int(i) + 0.5) * voxel_size
        point[horizontal_axes[1]] = (horizontal_origin_indices[1] + int(j) + 0.5) * voxel_size
        point[fill_axis] = float(surface[i, j])
        points.append(point)
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def sample_surface_preserving_particles(
    surface_points: np.ndarray,
    voxel_centers: np.ndarray,
    count: int,
    voxel_size: float,
    seed: int,
) -> np.ndarray:
    if count <= 0:
        raise ValueError("Particle count must be positive")
    if len(surface_points) == 0:
        return sample_particles(voxel_centers, count, voxel_size, seed)
    if count < len(surface_points):
        return farthest_point_sample(surface_points, count).astype(np.float32, copy=False)
    remaining = sample_particles(voxel_centers, count - len(surface_points), voxel_size, seed)
    return np.concatenate([surface_points, remaining], axis=0).astype(np.float32, copy=False)


def sample_particles(voxel_centers: np.ndarray, count: int, voxel_size: float, seed: int) -> np.ndarray:
    if voxel_centers.shape[0] == 0:
        raise ValueError("Voxel reconstruction produced no occupied voxels.")
    rng = np.random.default_rng(seed)
    replace = voxel_centers.shape[0] < count
    indices = rng.choice(voxel_centers.shape[0], size=count, replace=replace)
    jitter = (rng.random((count, 3), dtype=np.float32) - 0.5) * voxel_size
    return (voxel_centers[indices] + jitter).astype(np.float32)


def farthest_point_sample(points: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or points.shape[0] <= count:
        return points
    selected = np.empty(count, dtype=np.int64)
    center = points.mean(axis=0, keepdims=True)
    selected[0] = int(np.argmax(np.sum((points - center) ** 2, axis=1)))
    min_dist = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for i in range(1, count):
        selected[i] = int(np.argmax(min_dist))
        dist = np.sum((points - points[selected[i]]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
    return points[selected]


def preview_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[np.sort(indices)]


def centered_scaled(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return ((points - center[None, :]) / max(scale, 1e-8)).astype(np.float32)


def write_plotly_preview(
    output_path: Path,
    real_xyz: np.ndarray,
    voxel_centers: np.ndarray,
    sampled_particles: np.ndarray,
    preview_max_points: int,
    preview_sampled_points: int,
    seed: int,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("Plotly is required for HTML previews.") from exc

    real_preview = preview_points(real_xyz, preview_max_points, seed)
    voxel_preview = preview_points(voxel_centers, preview_max_points, seed + 1)
    sampled_preview = farthest_point_sample(sampled_particles, preview_sampled_points)

    all_points = np.concatenate([real_preview, voxel_preview, sampled_preview], axis=0)
    center = all_points.mean(axis=0)
    scale = float(np.max(np.linalg.norm(all_points - center[None, :], axis=1)))

    panels = [
        (centered_scaled(real_preview, center, scale), f"Real frame points ({real_xyz.shape[0]:,})", 1, 2),
        (centered_scaled(voxel_preview, center, scale), f"Filled voxel centers ({voxel_centers.shape[0]:,})", 2, 2),
        (centered_scaled(sampled_preview, center, scale), f"FPS sampled particles ({sampled_preview.shape[0]:,})", 3, 3),
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[title for _, title, _, _ in panels],
        horizontal_spacing=0.05,
    )
    for col, (points, title, _, marker_size) in enumerate(panels, start=1):
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={
                    "size": marker_size,
                    "color": points[:, 2],
                    "colorscale": "Viridis",
                    "opacity": 0.78,
                },
                name=title,
            ),
            row=1,
            col=col,
        )

    scene = {
        "xaxis": {"range": [-1, 1], "title": {"text": "x"}},
        "yaxis": {"range": [-1, 1], "title": {"text": "y"}},
        "zaxis": {"range": [-1, 1], "title": {"text": "z"}},
        "camera": {"eye": {"x": 1.4, "y": 1.4, "z": 1.0}},
        "aspectmode": "cube",
    }
    fig.update_layout(
        title="DeformPath frame-0 voxel dough reconstruction | centered/scaled preview",
        height=650,
        width=1500,
        margin={"l": 0, "r": 0, "t": 80, "b": 0},
        showlegend=False,
        scene=scene,
        scene2=scene,
        scene3=scene,
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.voxel_size) or args.voxel_size <= 0.0:
        raise ValueError("--voxel-size must be positive and finite")
    if args.num_particles <= 0:
        raise ValueError("--num-particles must be positive")
    fill_mode, thickness, fill_limit, legacy_fill_selection = resolve_fill_settings(
        args.fill_mode, args.thickness, args.fill_limit
    )
    args.fill_axis = args.fill_axis or ("y" if fill_mode == "floor" else "z")
    args.fill_direction = args.fill_direction or ("negative" if fill_mode == "floor" else "positive")

    frame = load_pointcloud_frame(args.episode_dir, args.pointclouds_name, args.frame)
    source_filtered_xyz = filter_points(frame, args.trim_quantile)
    calibration = load_calibration(args.calibration) if args.calibration is not None else None
    calibration_info = calibration_metadata(calibration) if calibration is not None else None
    source_frame = getattr(calibration, "source_frame", "source") if calibration is not None else "source"
    scene_frame = getattr(calibration, "scene_frame", None) if calibration is not None else None
    floor_plane = None
    segmentation = None
    scene_transformed_xyz = None

    if fill_mode == "floor":
        if calibration is None:
            raise ValueError("--fill-mode floor requires --calibration")
        scene_bounds = parse_scene_bounds(args.scene_bounds)
        scene_transformed_xyz, reconstruction_xyz, floor_plane, segmentation = transform_and_segment_floor_points(
            source_filtered_xyz,
            calibration,
            args.floor_clearance,
            scene_bounds,
        )
        reconstruction_frame = str(scene_frame)
    else:
        if args.scene_bounds is not None:
            raise ValueError("--scene-bounds is only valid with --fill-mode floor")
        reconstruction_xyz = source_filtered_xyz
        reconstruction_frame = str(source_frame)
        scene_bounds = None

    fill_axis = AXIS_TO_INDEX[args.fill_axis]
    fill_positive = args.fill_direction == "positive"
    surface, bbox_min, horizontal_origin_indices, horizontal_axes = build_surface_grid(
        reconstruction_xyz,
        args.voxel_size,
        fill_axis,
        fill_positive,
        args.bbox_padding,
        args.surface_percentile,
    )
    surface = dilate_surface(surface, args.footprint_dilate)
    intersection = None
    fill_limits_by_column = None
    if fill_mode == "floor":
        if (
            not np.isfinite(args.floor_min_thickness)
            or not np.isfinite(args.floor_max_thickness)
            or not np.isfinite(args.plane_parallel_epsilon)
        ):
            raise ValueError("Floor intersection settings must be finite")
        surface, fill_limits_by_column, intersection = plane_fill_limits_from_surface(
            surface,
            args.voxel_size,
            fill_axis,
            horizontal_axes,
            horizontal_origin_indices,
            fill_positive,
            floor_plane,
            args.floor_min_thickness,
            args.floor_max_thickness,
            args.plane_parallel_epsilon,
        )
        if intersection["accepted"] == 0:
            raise ValueError(f"No surface columns reached the floor: {intersection['rejection_counts']}")

    voxel_centers = voxel_centers_from_surface(
        surface,
        args.voxel_size,
        fill_axis,
        horizontal_axes,
        horizontal_origin_indices,
        fill_positive,
        thickness,
        fill_limit,
        fill_limits_by_column,
    )
    if len(voxel_centers) == 0 or not np.isfinite(voxel_centers).all():
        raise ValueError("Voxel reconstruction produced no finite occupied voxels")
    surface_particles = surface_points_from_grid(
        surface,
        args.voxel_size,
        fill_axis,
        horizontal_axes,
        horizontal_origin_indices,
    )
    sampled_particles = sample_surface_preserving_particles(
        surface_particles,
        voxel_centers,
        args.num_particles,
        args.voxel_size,
        args.seed,
    )
    if len(sampled_particles) != args.num_particles or not np.isfinite(sampled_particles).all():
        raise ValueError("Particle sampling produced an invalid reconstruction")

    voxel_count = int(len(voxel_centers))
    object_volume_m3 = float(voxel_count * args.voxel_size**3)
    if not np.isfinite(object_volume_m3) or object_volume_m3 <= 0.0:
        raise ValueError("Reconstructed object volume must be positive and finite")
    particles_sha256 = particle_array_sha256(sampled_particles)

    episode_name = args.episode_dir.name
    output_dir = args.output_dir / episode_name / f"frame_{args.frame:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = output_dir / "source_filtered_points_xyz.npy"
    transformed_path = output_dir / "scene_transformed_points_xyz.npy"
    real_path = output_dir / "real_points_xyz.npy"
    voxel_path = output_dir / "voxel_centers_xyz.npy"
    surface_path = output_dir / "surface_particles_xyz.npy"
    particles_path = output_dir / "sampled_particles_xyz.npy"
    particles_pt_path = output_dir / "sampled_particles_xyz.pt"
    metadata_path = output_dir / "reconstruction_metadata.json"
    html_path = output_dir / "voxel_reconstruction_preview.html"

    np.save(source_path, source_filtered_xyz)
    if scene_transformed_xyz is not None:
        np.save(transformed_path, scene_transformed_xyz)
    np.save(real_path, reconstruction_xyz)
    np.save(voxel_path, voxel_centers)
    np.save(surface_path, surface_particles)
    np.save(particles_path, sampled_particles)

    if args.save_pt:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("--save-pt requires torch.") from exc
        torch.save(torch.Tensor(sampled_particles), particles_pt_path)

    if args.write_preview:
        write_plotly_preview(
            html_path,
            reconstruction_xyz,
            voxel_centers,
            sampled_particles,
            args.preview_max_points,
            args.preview_sampled_points,
            args.seed,
        )

    scene_bounds_metadata = None
    if scene_bounds is not None:
        scene_bounds_metadata = {"min": scene_bounds[0].tolist(), "max": scene_bounds[1].tolist()}
    array_frames = {
        "source_filtered_points_xyz": str(source_frame),
        "real_points_xyz": reconstruction_frame,
        "voxel_centers_xyz": reconstruction_frame,
        "surface_particles_xyz": reconstruction_frame,
        "sampled_particles_xyz": reconstruction_frame,
    }
    if scene_transformed_xyz is not None:
        array_frames["scene_transformed_points_xyz"] = reconstruction_frame
    if args.save_pt:
        array_frames["sampled_particles_xyz_pt"] = reconstruction_frame

    metadata = {
        "schema": "voxel_dough_reconstruction/v2",
        "episode_dir": str(args.episode_dir.resolve()),
        "pointclouds_name": args.pointclouds_name,
        "frame": args.frame,
        "input_shape": list(frame.shape),
        "source_bounds_xyz": {
            "min": source_filtered_xyz.min(axis=0).tolist(),
            "max": source_filtered_xyz.max(axis=0).tolist(),
        },
        "reconstruction_bounds_xyz": {
            "min": reconstruction_xyz.min(axis=0).tolist(),
            "max": reconstruction_xyz.max(axis=0).tolist(),
        },
        "filtered_source_points": int(source_filtered_xyz.shape[0]),
        "filtered_points": int(reconstruction_xyz.shape[0]),
        "reconstruction_points": int(reconstruction_xyz.shape[0]),
        "voxel_count": voxel_count,
        "voxel_centers": voxel_count,
        "surface_particles": int(surface_particles.shape[0]),
        "sampled_particles": int(sampled_particles.shape[0]),
        "surface_particle_reservation": int(min(len(surface_particles), args.num_particles)),
        "voxel_size": args.voxel_size,
        "object_volume_m3": object_volume_m3,
        "sampled_particles_sha256": particles_sha256,
        "array_frames": array_frames,
        "fill_mode": fill_mode,
        "fill_axis": args.fill_axis,
        "fill_direction": args.fill_direction,
        "thickness": thickness,
        "fill_limit": fill_limit,
        "fill": {
            "mode": fill_mode,
            "legacy_implicit_selection": legacy_fill_selection,
            "axis": args.fill_axis,
            "direction": args.fill_direction,
            "thickness_m": thickness,
            "limit": fill_limit,
            "floor_plane_scene": floor_plane.tolist() if floor_plane is not None else None,
            "floor_clearance_m": args.floor_clearance if fill_mode == "floor" else None,
            "floor_min_thickness_m": args.floor_min_thickness if fill_mode == "floor" else None,
            "floor_max_thickness_m": args.floor_max_thickness if fill_mode == "floor" else None,
            "plane_parallel_epsilon": args.plane_parallel_epsilon if fill_mode == "floor" else None,
            "scene_bounds": scene_bounds_metadata,
            "segmentation_counts": segmentation,
            "intersection": intersection,
        },
        "bbox_padding": args.bbox_padding,
        "trim_quantile": args.trim_quantile,
        "surface_percentile": args.surface_percentile,
        "footprint_dilate": args.footprint_dilate,
        "seed": args.seed,
        "bbox_min": bbox_min.tolist(),
        "horizontal_axes": [axis for axis in horizontal_axes],
        "calibration_schema": getattr(calibration, "schema", None) if calibration is not None else None,
        "calibration_fingerprint": getattr(calibration, "fingerprint", None) if calibration is not None else None,
        "calibration": calibration_info,
        "outputs": {
            "source_filtered_points_xyz": str(source_path.resolve()),
            "scene_transformed_points_xyz": str(transformed_path.resolve()) if scene_transformed_xyz is not None else None,
            "real_points_xyz": str(real_path.resolve()),
            "voxel_centers_xyz": str(voxel_path.resolve()),
            "surface_particles_xyz": str(surface_path.resolve()),
            "sampled_particles_xyz": str(particles_path.resolve()),
            "sampled_particles_xyz_pt": str(particles_pt_path.resolve()) if args.save_pt else None,
            "preview_html": str(html_path.resolve()) if args.write_preview else None,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote source points: {source_path} ({source_filtered_xyz.shape[0]} points)")
    if scene_transformed_xyz is not None:
        print(f"Wrote scene-transformed points: {transformed_path} ({scene_transformed_xyz.shape[0]} points)")
    print(f"Wrote reconstruction points: {real_path} ({reconstruction_xyz.shape[0]} points)")
    print(f"Wrote voxel centers: {voxel_path} ({voxel_count} voxels, {object_volume_m3:.9g} m^3)")
    print(f"Wrote surface particles: {surface_path} ({surface_particles.shape[0]} particles)")
    print(f"Wrote sampled particles: {particles_path} ({sampled_particles.shape[0]} particles)")
    if args.write_preview:
        print(f"Wrote preview: {html_path}")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
