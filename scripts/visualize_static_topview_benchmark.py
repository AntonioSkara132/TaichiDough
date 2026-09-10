#!/usr/bin/env python3
"""Create dependency-light visual diagnostics for a static top-view benchmark run.

The report deliberately visualizes artifacts without registering, translating, or
rescaling either cloud for comparison. It is intended to make calibration,
reconstruction, visibility, and metric behavior inspectable before dynamics work.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from deformpath_topview import apply_calibration, load_calibration, rasterize_depth
except ImportError:
    from .deformpath_topview import apply_calibration, load_calibration, rasterize_depth


BACKGROUND = (17, 23, 33)
PANEL = (25, 34, 48)
GRID = (61, 74, 91)
TEXT = (229, 235, 241)
MUTED = (164, 178, 194)
BLUE = (75, 156, 211)
ORANGE = (246, 167, 67)
GREEN = (74, 202, 137)
MAGENTA = (223, 93, 177)
CYAN = (68, 198, 211)
RED = (235, 97, 85)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-metadata", type=Path, required=True)
    parser.add_argument("--taichi-metadata", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-scatter-points", type=int, default=16000)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_artifact(path_value: str | Path, metadata_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    relative_to_metadata = metadata_path.parent / path
    if relative_to_metadata.exists():
        return relative_to_metadata.resolve()
    raise FileNotFoundError(f"Could not resolve artifact {path_value!r} from {metadata_path}")


def primary_view(metadata: dict[str, Any]) -> dict[str, Any]:
    for frame in metadata.get("frames", []):
        if frame.get("simulation_step", frame.get("step")) != 0:
            continue
        views = frame.get("views", [])
        if views:
            return views[0]
    raise ValueError("Taichi metadata has no saved step-zero view")


def sample_rows(points: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0 or len(points) <= maximum:
        return points
    return points[np.linspace(0, len(points) - 1, maximum).round().astype(np.int64)]


def value_colors(values: np.ndarray, low: float | None = None, high: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.tile(np.asarray(MUTED, dtype=np.uint8), (len(values), 1))
    low = float(np.quantile(finite, 0.01) if low is None else low)
    high = float(np.quantile(finite, 0.99) if high is None else high)
    t = np.clip((values - low) / max(high - low, 1e-8), 0.0, 1.0)
    anchors = np.asarray(
        [[205, 226, 251], [134, 182, 239], [57, 135, 229], [28, 92, 171], [13, 54, 107]], dtype=np.float32
    )
    scaled = t * (len(anchors) - 1)
    index = np.minimum(scaled.astype(np.int32), len(anchors) - 2)
    frac = (scaled - index)[:, None]
    return ((1.0 - frac) * anchors[index] + frac * anchors[index + 1]).astype(np.uint8)


def signed_colors(values: np.ndarray, limit: float) -> np.ndarray:
    limit = max(float(limit), 1e-8)
    t = np.clip(values / limit, -1.0, 1.0)
    result = np.empty((len(values), 3), dtype=np.uint8)
    negative = t < 0
    result[negative] = ((1.0 + t[negative, None]) * 255.0 + (-t[negative, None]) * np.asarray(BLUE)).astype(np.uint8)
    result[~negative] = ((1.0 - t[~negative, None]) * 255.0 + t[~negative, None] * np.asarray(RED)).astype(np.uint8)
    return result


def font(size: int = 14) -> Any:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def draw_title(draw: ImageDraw.ImageDraw, title: str, subtitle: str = "") -> None:
    draw.text((18, 14), title, fill=TEXT, font=font(20))
    if subtitle:
        draw.text((18, 41), subtitle, fill=MUTED, font=font(12))


def make_panel(width: int, height: int, title: str, subtitle: str = "") -> Image.Image:
    image = Image.new("RGB", (width, height), PANEL)
    draw = ImageDraw.Draw(image)
    draw_title(draw, title, subtitle)
    draw.rectangle((0, 0, width - 1, height - 1), outline=GRID)
    return image


def pad_bounds(*arrays: np.ndarray, fraction: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    data = np.concatenate([array for array in arrays if len(array)], axis=0)
    lower = data.min(axis=0).astype(np.float32)
    upper = data.max(axis=0).astype(np.float32)
    span = np.maximum(upper - lower, 1e-6)
    return lower - fraction * span, upper + fraction * span


def draw_scatter(
    image: Image.Image,
    points: np.ndarray,
    axes: tuple[int, int],
    colors: np.ndarray | tuple[int, int, int],
    lower: np.ndarray,
    upper: np.ndarray,
    x_label: str,
    y_label: str,
    max_points: int,
    dot_radius: int = 1,
    overlay: bool = False,
) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    left, top, right, bottom = 50, 72, width - 24, height - 45
    for fraction in (0.25, 0.5, 0.75):
        x = left + int((right - left) * fraction)
        y = top + int((bottom - top) * fraction)
        draw.line((x, top, x, bottom), fill=GRID)
        draw.line((left, y, right, y), fill=GRID)
    points = sample_rows(points, max_points)
    if len(points) == 0:
        return
    mapped = (points[:, axes] - lower[None, :]) / np.maximum(upper - lower, 1e-8)
    pixels = np.empty((len(points), 2), dtype=np.int32)
    pixels[:, 0] = left + np.clip(mapped[:, 0], 0, 1) * (right - left)
    pixels[:, 1] = bottom - np.clip(mapped[:, 1], 0, 1) * (bottom - top)
    if isinstance(colors, tuple):
        point_colors = np.tile(np.asarray(colors, dtype=np.uint8), (len(points), 1))
    else:
        colors = np.asarray(colors)
        if len(colors) != len(points):
            colors = sample_rows(colors, max_points)
        point_colors = colors
    for (x, y), color in zip(pixels, point_colors):
        box = (int(x) - dot_radius, int(y) - dot_radius, int(x) + dot_radius, int(y) + dot_radius)
        draw.ellipse(box, fill=tuple(int(value) for value in color))
    draw.line((left, bottom, right, bottom), fill=MUTED)
    draw.line((left, top, left, bottom), fill=MUTED)
    draw.text((left, height - 30), x_label, fill=MUTED, font=font(12))
    draw.text((4, top), y_label, fill=MUTED, font=font(12))
    if not overlay:
        draw.text((left, top - 18), f"range: {lower[0]:.4f}..{upper[0]:.4f} / {lower[1]:.4f}..{upper[1]:.4f}", fill=MUTED, font=font(11))


def _plot_pixels(
    points: np.ndarray,
    axes: tuple[int, int],
    lower: np.ndarray,
    upper: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    left, top, right, bottom = 50, 72, width - 24, height - 45
    mapped = (points[:, axes] - lower[None, :]) / np.maximum(upper - lower, 1e-8)
    pixels = np.empty((len(points), 2), dtype=np.float32)
    pixels[:, 0] = left + mapped[:, 0] * (right - left)
    pixels[:, 1] = bottom - mapped[:, 1] * (bottom - top)
    return pixels


def draw_segments(
    image: Image.Image,
    starts: np.ndarray,
    ends: np.ndarray,
    axes: tuple[int, int],
    colors: np.ndarray | tuple[int, int, int],
    lower: np.ndarray,
    upper: np.ndarray,
    max_segments: int = 500,
    width: int = 1,
) -> None:
    if len(starts) != len(ends):
        raise ValueError("Segment starts and ends must have the same length")
    if not len(starts):
        return
    if len(starts) > max_segments:
        indices = np.linspace(0, len(starts) - 1, max_segments).round().astype(np.int64)
        starts = starts[indices]
        ends = ends[indices]
        if not isinstance(colors, tuple):
            colors = np.asarray(colors)[indices]
    start_pixels = _plot_pixels(starts, axes, lower, upper, image.size)
    end_pixels = _plot_pixels(ends, axes, lower, upper, image.size)
    if isinstance(colors, tuple):
        segment_colors = np.tile(np.asarray(colors, dtype=np.uint8), (len(starts), 1))
    else:
        segment_colors = np.asarray(colors, dtype=np.uint8)
    draw = ImageDraw.Draw(image)
    for start, end, color in zip(start_pixels, end_pixels, segment_colors):
        draw.line(
            (float(start[0]), float(start[1]), float(end[0]), float(end[1])),
            fill=tuple(int(value) for value in color),
            width=width,
        )


def draw_reference_axes(
    image: Image.Image,
    transforms: list[tuple[str, np.ndarray]],
    axes: tuple[int, int],
    lower: np.ndarray,
    upper: np.ndarray,
    length_m: float,
) -> None:
    axis_colors = (RED, GREEN, BLUE)
    draw = ImageDraw.Draw(image)
    for label, transform in transforms:
        origin = transform[:3, 3]
        endpoints = origin[None, :] + transform[:3, :3].T * length_m
        origin_pixel = _plot_pixels(origin[None, :], axes, lower, upper, image.size)[0]
        endpoint_pixels = _plot_pixels(endpoints, axes, lower, upper, image.size)
        for axis_index, (endpoint, color) in enumerate(zip(endpoint_pixels, axis_colors)):
            draw.line(
                (float(origin_pixel[0]), float(origin_pixel[1]), float(endpoint[0]), float(endpoint[1])),
                fill=color,
                width=3,
            )
            draw.text((float(endpoint[0]) + 3, float(endpoint[1]) - 8), "XYZ"[axis_index], fill=color, font=font(10))
        draw.ellipse(
            (origin_pixel[0] - 4, origin_pixel[1] - 4, origin_pixel[0] + 4, origin_pixel[1] + 4),
            fill=TEXT,
        )
        draw.text((float(origin_pixel[0]) + 6, float(origin_pixel[1]) + 4), label, fill=TEXT, font=font(10))


def normalized_plane(plane: Any) -> np.ndarray:
    values = np.asarray(plane, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Floor plane must contain four finite coefficients")
    norm = float(np.linalg.norm(values[:3]))
    if norm <= 1e-12:
        raise ValueError("Floor plane normal must be nonzero")
    return values / norm


def floor_segmentation_diagnostics(points: np.ndarray, fill: dict[str, Any]) -> dict[str, Any]:
    plane = normalized_plane(fill.get("floor_plane_scene"))
    clearance_value = fill.get("floor_clearance_m")
    if clearance_value is None:
        raise ValueError("Floor reconstruction metadata is missing floor_clearance_m")
    clearance = float(clearance_value)
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError("Floor clearance must be finite and non-negative")
    finite = np.isfinite(points).all(axis=1)
    signed_distance = points @ plane[:3] + plane[3]
    above_floor = finite & np.isfinite(signed_distance) & (signed_distance >= clearance)
    bounds = fill.get("scene_bounds")
    if bounds is None:
        inside_bounds = np.ones(len(points), dtype=bool)
    else:
        lower = np.asarray(bounds.get("min"), dtype=np.float64)
        upper = np.asarray(bounds.get("max"), dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower >= upper):
            raise ValueError("Floor scene bounds must contain finite increasing XYZ limits")
        inside_bounds = np.all((points >= lower[None, :]) & (points <= upper[None, :]), axis=1)
    kept = above_floor & inside_bounds
    return {
        "plane": plane,
        "clearance_m": clearance,
        "kept_mask": kept,
        "floor_rejected_mask": finite & ~above_floor,
        "bounds_rejected_mask": above_floor & ~inside_bounds,
        "counts": {
            "input": int(len(points)),
            "nonfinite": int((~finite).sum()),
            "at_or_below_clearance": int((finite & ~above_floor).sum()),
            "outside_scene_bounds": int((above_floor & ~inside_bounds).sum()),
            "kept": int(kept.sum()),
        },
    }


def floor_column_geometry(surface: np.ndarray, fill: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis_name = str(fill.get("axis"))
    if axis_name not in ("x", "y", "z"):
        raise ValueError("Floor fill axis must be x, y, or z")
    axis = "xyz".index(axis_name)
    direction = 1.0 if fill.get("direction") == "positive" else -1.0
    ray = np.zeros(3, dtype=np.float64)
    ray[axis] = direction
    plane = normalized_plane(fill.get("floor_plane_scene"))
    denominator = float(np.dot(plane[:3], ray))
    epsilon = float(fill.get("plane_parallel_epsilon", 1e-8))
    if abs(denominator) <= epsilon:
        raise ValueError("Floor fill direction is parallel to the calibrated floor")
    thickness = -(surface @ plane[:3] + plane[3]) / denominator
    minimum = float(fill.get("floor_min_thickness_m", 0.0))
    maximum = float(fill.get("floor_max_thickness_m", np.inf))
    valid = np.isfinite(surface).all(axis=1) & np.isfinite(thickness) & (thickness > 0) & (thickness >= minimum) & (thickness <= maximum)
    starts = surface[valid].astype(np.float32, copy=False)
    distances = thickness[valid].astype(np.float32, copy=False)
    ends = starts + distances[:, None] * ray[None, :]
    return starts, ends.astype(np.float32), distances


def floor_plane_samples(plane: np.ndarray, lower: np.ndarray, upper: np.ndarray, samples: int = 18) -> np.ndarray:
    plane = normalized_plane(plane)
    solved_axis = int(np.argmax(np.abs(plane[:3])))
    free_axes = [axis for axis in range(3) if axis != solved_axis]
    first = np.linspace(lower[free_axes[0]], upper[free_axes[0]], samples)
    second = np.linspace(lower[free_axes[1]], upper[free_axes[1]], samples)
    grid_first, grid_second = np.meshgrid(first, second, indexing="ij")
    points = np.zeros((samples * samples, 3), dtype=np.float64)
    points[:, free_axes[0]] = grid_first.reshape(-1)
    points[:, free_axes[1]] = grid_second.reshape(-1)
    points[:, solved_axis] = -(
        plane[3]
        + plane[free_axes[0]] * points[:, free_axes[0]]
        + plane[free_axes[1]] * points[:, free_axes[1]]
    ) / plane[solved_axis]
    return points.astype(np.float32)


def calibration_reference_transforms(calibration: Any) -> list[tuple[str, np.ndarray]]:
    transforms: list[tuple[str, np.ndarray]] = [(str(getattr(calibration, "scene_frame", "scene")), np.eye(4))]
    for label, attribute in (
        (str(getattr(calibration, "source_frame", "source")), "scene_from_source"),
        ("camera", "scene_from_camera"),
    ):
        value = getattr(calibration, attribute, None)
        if value is not None:
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError(f"Calibration {attribute} must be a finite 4x4 matrix")
            transforms.append((label, matrix))
    provenance = getattr(calibration, "provenance", {}) or {}
    scene_from_tag = provenance.get("scene_from_tag")
    if scene_from_tag is not None:
        matrix = np.asarray(scene_from_tag, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Calibration scene_from_tag must be a finite 4x4 matrix")
        tag_label = f"tag {provenance.get('tag_family', '')}:{provenance.get('tag_id', '')}".rstrip(":")
        transforms.append((tag_label, matrix))
    return transforms


def resize_for_panel(rgb: np.ndarray, width: int, height: int) -> Image.Image:
    source = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    scale = min(width / source.width, height / source.height)
    source = source.resize((max(1, round(source.width * scale)), max(1, round(source.height * scale))), Image.Resampling.NEAREST)
    output = Image.new("RGB", (width, height), BACKGROUND)
    output.paste(source, ((width - source.width) // 2, (height - source.height) // 2))
    return output


def depth_rgb(depth: np.ndarray, valid: np.ndarray, low: float, high: float) -> np.ndarray:
    output = np.full((*depth.shape, 3), BACKGROUND, dtype=np.uint8)
    if np.any(valid):
        output[valid] = value_colors(depth[valid], low, high)
    return output


def occupancy_grid(points: np.ndarray, lower: np.ndarray, upper: np.ndarray, cell_size: float) -> np.ndarray:
    shape = np.maximum(np.ceil((upper - lower) / cell_size).astype(int), 1)
    grid = np.zeros(tuple(shape), dtype=bool)
    cells = np.floor((points[:, :2] - lower) / cell_size).astype(np.int64)
    valid = np.all((cells >= 0) & (cells < shape), axis=1)
    grid[cells[valid, 0], cells[valid, 1]] = True
    return grid


def occupancy_rgb(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    rgb = np.full((*reference.shape, 3), BACKGROUND, dtype=np.uint8)
    rgb[reference & ~candidate] = BLUE
    rgb[~reference & candidate] = ORANGE
    rgb[reference & candidate] = GREEN
    return np.transpose(rgb, (1, 0, 2))


def paste_panel(canvas: Image.Image, panel: Image.Image, column: int, row: int, margin: int = 16) -> None:
    panel_width, panel_height = panel.size
    x = margin + column * (panel_width + margin)
    y = margin + row * (panel_height + margin)
    canvas.paste(panel, (x, y))


def quad_canvas(panels: list[Image.Image]) -> Image.Image:
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * 2 + 48, height * 2 + 48), BACKGROUND)
    for index, panel in enumerate(panels):
        paste_panel(canvas, panel, index % 2, index // 2)
    return canvas


def save_dashboard(path: Path, panels: list[Image.Image]) -> None:
    dashboard = quad_canvas(panels)
    dashboard.save(path)


def make_source_reconstruction_dashboard(
    observed: np.ndarray,
    voxels: np.ndarray,
    surface: np.ndarray,
    particles: np.ndarray,
    frame_name: str,
    max_points: int,
) -> Image.Image:
    panel_size = (700, 470)
    observed_xy = pad_bounds(observed[:, [0, 1]])
    observed_xz = pad_bounds(observed[:, [0, 2]], voxels[:, [0, 2]])
    reconstruction_xy = pad_bounds(observed[:, [0, 1]], voxels[:, [0, 1]])
    x_label = f"{frame_name} X (m)"
    y_label = f"{frame_name} Y (m)"
    z_label = f"{frame_name} Z (m)"

    observed_top = make_panel(*panel_size, "Observed reconstruction-frame footprint", f"Visible points in {frame_name}; color encodes Z")
    draw_scatter(observed_top, observed, (0, 1), value_colors(observed[:, 2]), *observed_xy, x_label, y_label, max_points)

    observed_side = make_panel(*panel_size, "Observed surface profile", f"Visible {frame_name} X/Z coordinates before volume filling")
    draw_scatter(observed_side, observed, (0, 2), value_colors(observed[:, 1]), *observed_xz, x_label, z_label, max_points)

    reconstruction = make_panel(*panel_size, "Voxel volume and reserved visible surface", "Blue: filled voxels; orange: explicit visible-surface particles")
    draw_scatter(reconstruction, voxels, (0, 1), BLUE, *reconstruction_xy, x_label, y_label, max_points, dot_radius=1)
    draw_scatter(reconstruction, surface, (0, 1), ORANGE, *reconstruction_xy, x_label, y_label, max_points, dot_radius=2, overlay=True)
    draw_scatter(reconstruction, particles, (0, 1), (226, 232, 240), *reconstruction_xy, x_label, y_label, max_points, dot_radius=0, overlay=True)

    profile = make_panel(*panel_size, "Filled-volume profile", "Blue: filled voxel volume; orange: preserved visible surface; white: sampled MPM particles")
    draw_scatter(profile, voxels, (0, 2), BLUE, *observed_xz, x_label, z_label, max_points, dot_radius=1)
    draw_scatter(profile, surface, (0, 2), ORANGE, *observed_xz, x_label, z_label, max_points, dot_radius=2, overlay=True)
    draw_scatter(profile, particles, (0, 2), (226, 232, 240), *observed_xz, x_label, z_label, max_points, dot_radius=0, overlay=True)
    return quad_canvas([observed_top, observed_side, reconstruction, profile])


def camera_position_scene(camera: dict[str, Any]) -> np.ndarray:
    scene_from_camera = getattr(camera, "scene_from_camera", None)
    if scene_from_camera is None and "scene_from_camera" in camera:
        scene_from_camera = camera["scene_from_camera"]
    if scene_from_camera is not None:
        matrix = np.asarray(scene_from_camera, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("camera scene_from_camera must be a finite 4x4 matrix")
        return matrix[:3, 3].astype(np.float32)
    position = np.asarray(camera.get("position"), dtype=np.float32)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Legacy camera position must be a finite XYZ vector")
    return position


def make_calibration_dashboard(
    source: np.ndarray,
    calibrated_source: np.ndarray,
    taichi_particles: np.ndarray,
    calibration: Any,
    max_points: int,
) -> Image.Image:
    panel_size = (700, 470)
    camera = calibration.camera
    source_frame = str(getattr(calibration, "source_frame", "source"))
    scene_frame = str(getattr(calibration, "scene_frame", "scene"))
    optical_bounds = pad_bounds(source[:, [0, 1]])
    scene_top_bounds = pad_bounds(calibrated_source[:, [0, 2]], taichi_particles[:, [0, 2]])
    scene_side_bounds = pad_bounds(calibrated_source[:, [0, 1]], taichi_particles[:, [0, 1]])

    raw = make_panel(*panel_size, "Input coordinate convention", f"Coordinates stored in {source_frame}; color encodes source Z")
    draw_scatter(raw, source, (0, 1), value_colors(source[:, 2]), *optical_bounds, f"{source_frame} X (m)", f"{source_frame} Y (m)", max_points)

    height_low = float(min(calibrated_source[:, 1].min(), taichi_particles[:, 1].min()))
    height_high = float(max(calibrated_source[:, 1].max(), taichi_particles[:, 1].max()))
    scene_top = make_panel(*panel_size, "Calibrated scene top view", f"{scene_frame} Y height: {height_low:.4f}..{height_high:.4f} m (light to dark)")
    draw_scatter(scene_top, calibrated_source, (0, 2), value_colors(calibrated_source[:, 1], height_low, height_high), *scene_top_bounds, f"{scene_frame} X (m)", f"{scene_frame} Z (m)", max_points)
    camera_position = camera_position_scene(camera)[None, :]
    draw_scatter(scene_top, camera_position, (0, 2), RED, *scene_top_bounds, f"{scene_frame} X (m)", f"{scene_frame} Z (m)", 1, dot_radius=5, overlay=True)

    scene_side = make_panel(*panel_size, "Calibrated scene elevation", "Source points mapped by the supplied calibration; no per-frame fit is applied")
    draw_scatter(scene_side, calibrated_source, (0, 1), value_colors(source[:, 2]), *scene_side_bounds, f"{scene_frame} X (m)", f"{scene_frame} Y (m)", max_points)

    initialized = make_panel(*panel_size, "Actual initialized Taichi particle state", "Scene-coordinate particle cloud before any MPM substep")
    draw_scatter(initialized, taichi_particles, (0, 2), value_colors(taichi_particles[:, 1], height_low, height_high), *scene_top_bounds, f"{scene_frame} X (m)", f"{scene_frame} Z (m)", max_points)
    draw_scatter(initialized, camera_position, (0, 2), RED, *scene_top_bounds, f"{scene_frame} X (m)", f"{scene_frame} Z (m)", 1, dot_radius=5, overlay=True)
    return quad_canvas([raw, scene_top, scene_side, initialized])


def draw_key_value_rows(panel: Image.Image, rows: list[tuple[str, str]]) -> None:
    draw = ImageDraw.Draw(panel)
    y = 88
    for label, value in rows:
        draw.rounded_rectangle((32, y, panel.width - 32, y + 52), radius=8, fill=BACKGROUND, outline=GRID)
        draw.text((46, y + 8), label, fill=MUTED, font=font(11))
        draw.text((260, y + 8), value, fill=TEXT, font=font(13))
        y += 61


def make_floor_reconstruction_dashboard(
    transformed: np.ndarray,
    retained: np.ndarray,
    surface: np.ndarray,
    voxels: np.ndarray,
    reconstruction: dict[str, Any],
    calibration: Any,
    max_points: int,
) -> tuple[Image.Image, dict[str, Any]]:
    fill = reconstruction.get("fill", {})
    if fill.get("mode") != "floor":
        raise ValueError("Floor reconstruction dashboard requires floor fill metadata")
    segmentation = floor_segmentation_diagnostics(transformed, fill)
    computed_retained = transformed[segmentation["kept_mask"]]
    if computed_retained.shape != retained.shape or not np.allclose(computed_retained, retained, atol=1e-6, rtol=0):
        raise ValueError("Saved retained points disagree with recorded floor segmentation settings")
    recorded_segmentation = fill.get("segmentation_counts")
    if recorded_segmentation is not None and recorded_segmentation != segmentation["counts"]:
        raise ValueError("Recorded floor segmentation counts disagree with the saved transformed points")

    starts, ends, thicknesses = floor_column_geometry(surface, fill)
    recorded_intersection = fill.get("intersection") or {}
    if recorded_intersection.get("accepted") is not None and int(recorded_intersection["accepted"]) != len(starts):
        raise ValueError("Recorded floor-column count disagrees with the saved surface points")
    if not len(starts):
        raise ValueError("Floor reconstruction contains no accepted surface columns")

    lower3, upper3 = pad_bounds(retained, voxels, starts, ends)
    floor_samples = floor_plane_samples(segmentation["plane"], lower3, upper3)
    floor_rejected = transformed[segmentation["floor_rejected_mask"]]
    bounds_rejected = transformed[segmentation["bounds_rejected_mask"]]
    scene_frame = str(getattr(calibration, "scene_frame", "scene"))
    top_bounds = pad_bounds(transformed[:, [0, 2]], voxels[:, [0, 2]], floor_samples[:, [0, 2]])
    elevation_bounds = pad_bounds(transformed[:, [0, 1]], voxels[:, [0, 1]], floor_samples[:, [0, 1]])
    panel_size = (700, 470)

    top = make_panel(
        *panel_size,
        "Floor segmentation — top view",
        "Green: retained dough; red: floor/clearance rejection; magenta: scene-bounds rejection; cyan: floor samples",
    )
    draw_scatter(top, floor_samples, (0, 2), CYAN, *top_bounds, f"{scene_frame} X (m)", f"{scene_frame} Z (m)", max_points, dot_radius=0)
    draw_scatter(top, retained, (0, 2), GREEN, *top_bounds, f"{scene_frame} X (m)", f"{scene_frame} Z (m)", max_points, dot_radius=1, overlay=True)
    draw_scatter(top, floor_rejected, (0, 2), RED, *top_bounds, f"{scene_frame} X (m)", f"{scene_frame} Z (m)", max_points, dot_radius=2, overlay=True)
    draw_scatter(top, bounds_rejected, (0, 2), MAGENTA, *top_bounds, f"{scene_frame} X (m)", f"{scene_frame} Z (m)", max_points, dot_radius=2, overlay=True)

    elevation = make_panel(
        *panel_size,
        "Floor segmentation — elevation",
        "The calibrated floor and clearance rejection are shown in metric scene coordinates",
    )
    draw_scatter(elevation, floor_samples, (0, 1), CYAN, *elevation_bounds, f"{scene_frame} X (m)", f"{scene_frame} Y (m)", max_points, dot_radius=0)
    draw_scatter(elevation, retained, (0, 1), GREEN, *elevation_bounds, f"{scene_frame} X (m)", f"{scene_frame} Y (m)", max_points, dot_radius=1, overlay=True)
    draw_scatter(elevation, floor_rejected, (0, 1), RED, *elevation_bounds, f"{scene_frame} X (m)", f"{scene_frame} Y (m)", max_points, dot_radius=2, overlay=True)
    draw_scatter(elevation, bounds_rejected, (0, 1), MAGENTA, *elevation_bounds, f"{scene_frame} X (m)", f"{scene_frame} Y (m)", max_points, dot_radius=2, overlay=True)

    fill_axis = "xyz".index(str(fill["axis"]))
    horizontal_axis = next(axis for axis in range(3) if axis != fill_axis)
    profile_axes = (horizontal_axis, fill_axis)
    profile_bounds = pad_bounds(voxels[:, profile_axes], starts[:, profile_axes], ends[:, profile_axes])
    thickness_low = float(np.min(thicknesses))
    thickness_high = float(np.max(thicknesses))
    thickness_colors = value_colors(thicknesses, thickness_low, thickness_high)
    profile = make_panel(
        *panel_size,
        "Per-column fill to floor",
        "Lines connect visible surface columns to their calibrated floor intersections; color encodes local thickness",
    )
    draw_scatter(profile, voxels, profile_axes, BLUE, *profile_bounds, f"{scene_frame} {'XYZ'[horizontal_axis]} (m)", f"{scene_frame} {'XYZ'[fill_axis]} (m)", max_points, dot_radius=0)
    draw_segments(profile, starts, ends, profile_axes, thickness_colors, *profile_bounds, max_segments=600, width=1)
    draw_scatter(profile, starts, profile_axes, thickness_colors, *profile_bounds, f"{scene_frame} {'XYZ'[horizontal_axis]} (m)", f"{scene_frame} {'XYZ'[fill_axis]} (m)", max_points, dot_radius=2, overlay=True)
    draw_scatter(profile, ends, profile_axes, CYAN, *profile_bounds, f"{scene_frame} {'XYZ'[horizontal_axis]} (m)", f"{scene_frame} {'XYZ'[fill_axis]} (m)", max_points, dot_radius=1, overlay=True)

    intersection_statistics = recorded_intersection.get("local_thickness_m") or {
        "min": thickness_low,
        "max": thickness_high,
        "mean": float(np.mean(thicknesses)),
        "median": float(np.median(thicknesses)),
        "p05": float(np.quantile(thicknesses, 0.05)),
        "p95": float(np.quantile(thicknesses, 0.95)),
    }
    volume = float(reconstruction["object_volume_m3"])
    summary = make_panel(*panel_size, "Floor-fill reconstruction summary", "Values are read from and checked against reconstruction metadata")
    draw_key_value_rows(summary, [
        ("Object volume", f"{volume * 1e6:.3f} mL ({volume:.9g} m³)"),
        ("Voxel count / size", f"{int(reconstruction['voxel_count']):,} / {float(reconstruction['voxel_size']) * 1000:.2f} mm"),
        ("Retained source points", f"{segmentation['counts']['kept']:,} of {segmentation['counts']['input']:,}"),
        ("Floor / bounds rejection", f"{segmentation['counts']['at_or_below_clearance']:,} / {segmentation['counts']['outside_scene_bounds']:,}"),
        ("Accepted / rejected columns", f"{len(starts):,} / {int(recorded_intersection.get('rejected', 0)):,}"),
        ("Thickness p05 .. p95", f"{float(intersection_statistics['p05']) * 1000:.2f} .. {float(intersection_statistics['p95']) * 1000:.2f} mm"),
    ])
    diagnostics = {
        "floor_plane_scene": segmentation["plane"].tolist(),
        "floor_clearance_m": segmentation["clearance_m"],
        "segmentation_counts": segmentation["counts"],
        "accepted_columns": int(len(starts)),
        "local_thickness_m": intersection_statistics,
        "object_volume_m3": volume,
        "voxel_count": int(reconstruction["voxel_count"]),
        "voxel_size_m": float(reconstruction["voxel_size"]),
    }
    return quad_canvas([top, elevation, profile, summary]), diagnostics


def make_reference_frame_dashboard(
    scene_points: np.ndarray,
    reconstruction: dict[str, Any],
    calibration: Any,
    max_points: int,
) -> tuple[Image.Image, dict[str, Any]]:
    transforms = calibration_reference_transforms(calibration)
    origins = np.asarray([transform[:3, 3] for _, transform in transforms], dtype=np.float32)
    object_lower, object_upper = pad_bounds(scene_points)
    lower3, upper3 = pad_bounds(scene_points, origins)
    plane = normalized_plane(reconstruction["fill"]["floor_plane_scene"])
    floor_samples = floor_plane_samples(plane, lower3, upper3)
    top_bounds = pad_bounds(scene_points[:, [0, 2]], origins[:, [0, 2]], floor_samples[:, [0, 2]])
    elevation_bounds = pad_bounds(scene_points[:, [0, 1]], origins[:, [0, 1]], floor_samples[:, [0, 1]])
    axis_length = float(np.clip(np.max(object_upper - object_lower) * 0.18, 0.015, 0.10))
    scene_frame = str(getattr(calibration, "scene_frame", "scene"))
    panel_size = (700, 470)

    top = make_panel(*panel_size, "Metric reference frames — top view", "Red/green/blue arrows are each frame's local X/Y/Z axes")
    draw_scatter(top, floor_samples, (0, 2), CYAN, *top_bounds, f"{scene_frame} X (m)", f"{scene_frame} Z (m)", max_points, dot_radius=0)
    draw_scatter(top, scene_points, (0, 2), MUTED, *top_bounds, f"{scene_frame} X (m)", f"{scene_frame} Z (m)", max_points, dot_radius=1, overlay=True)
    draw_reference_axes(top, transforms, (0, 2), *top_bounds, axis_length)

    elevation = make_panel(*panel_size, "Metric reference frames — elevation", "Frame origins and orientations are drawn without fitted translation or scale")
    draw_scatter(elevation, floor_samples, (0, 1), CYAN, *elevation_bounds, f"{scene_frame} X (m)", f"{scene_frame} Y (m)", max_points, dot_radius=0)
    draw_scatter(elevation, scene_points, (0, 1), MUTED, *elevation_bounds, f"{scene_frame} X (m)", f"{scene_frame} Y (m)", max_points, dot_radius=1, overlay=True)
    draw_reference_axes(elevation, transforms, (0, 1), *elevation_bounds, axis_length)

    transform_panel = make_panel(*panel_size, "Reference-frame origins", "All origins are expressed in the calibrated scene frame")
    transform_rows = [
        (label, f"[{matrix[0, 3]:+.4f}, {matrix[1, 3]:+.4f}, {matrix[2, 3]:+.4f}] m")
        for label, matrix in transforms
    ]
    draw_key_value_rows(transform_panel, transform_rows[:6])

    provenance = getattr(calibration, "provenance", {}) or {}
    detail_panel = make_panel(*panel_size, "AprilTag and floor reference", "Physical measurements used to establish metric scene geometry")
    tag_size = provenance.get("tag_edge_size_m")
    tag_text = "not recorded" if tag_size is None else f"{float(tag_size) * 1000:.3f} mm"
    draw_key_value_rows(detail_panel, [
        ("Calibration schema", str(getattr(calibration, "schema", "unknown"))),
        ("Source → scene", f"{getattr(calibration, 'source_frame', 'source')} → {scene_frame}"),
        ("AprilTag", f"{provenance.get('tag_family', 'unknown')} id {provenance.get('tag_id', 'unknown')}"),
        ("Measured tag edge", tag_text),
        ("Floor plane [n, d]", "[" + ", ".join(f"{value:+.5f}" for value in plane) + "]"),
        ("Fitted scale", "none (metric rigid transforms)" if getattr(calibration, "is_metric", False) else "legacy calibration"),
    ])
    diagnostics = {
        "axis_length_m": axis_length,
        "frames": [
            {"name": label, "scene_from_frame": matrix.tolist()}
            for label, matrix in transforms
        ],
        "tag_family": provenance.get("tag_family"),
        "tag_id": provenance.get("tag_id"),
        "tag_edge_size_m": tag_size,
    }
    return quad_canvas([top, elevation, transform_panel, detail_panel]), diagnostics


def depth_panels(
    real_depth: np.ndarray,
    real_valid: np.ndarray,
    virtual_depth: np.ndarray,
    virtual_valid: np.ndarray,
) -> tuple[Image.Image, dict[str, Any]]:
    common = real_valid & virtual_valid
    all_valid = real_valid | virtual_valid
    finite = np.concatenate([real_depth[real_valid], virtual_depth[virtual_valid]])
    low, high = np.quantile(finite, [0.01, 0.99])
    residual = virtual_depth - real_depth
    residual_limit = float(np.quantile(np.abs(residual[common]), 0.99)) if np.any(common) else 0.001
    residual_limit = max(residual_limit, 0.001)
    coverage_rgb = np.full((*real_depth.shape, 3), BACKGROUND, dtype=np.uint8)
    coverage_rgb[real_valid & ~virtual_valid] = BLUE
    coverage_rgb[~real_valid & virtual_valid] = ORANGE
    coverage_rgb[common] = GREEN

    panel_size = (700, 470)
    panels: list[Image.Image] = []
    specs = [
        ("Real calibrated z-buffer depth", "Raw metric depth; not independently normalized display depth", depth_rgb(real_depth, real_valid, float(low), float(high))),
        ("Taichi initial-state z-buffer depth", "Raw metric depth from the same pinhole camera", depth_rgb(virtual_depth, virtual_valid, float(low), float(high))),
        ("Pixel-level signed depth residual", "Taichi depth minus real depth; red = farther, blue = nearer", np.where(common[..., None], signed_colors(residual.reshape(-1), residual_limit).reshape(*residual.shape, 3), np.asarray(BACKGROUND, dtype=np.uint8))),
        ("Visibility agreement", "Green: both; blue: real only; orange: Taichi only", coverage_rgb),
    ]
    for index, (title, subtitle, rgb) in enumerate(specs):
        panel = make_panel(*panel_size, title, subtitle)
        image = resize_for_panel(rgb, 620, 340)
        panel.paste(image, (50, 88))
        draw = ImageDraw.Draw(panel)
        if index < 2:
            colors = value_colors(np.linspace(low, high, 620), float(low), float(high))
            panel.paste(Image.fromarray(np.repeat(colors[None], 8, axis=0)), (50, 432))
            draw.text((50, 445), f"Shared scale: {low:.4f} m (near) to {high:.4f} m (far)", fill=MUTED, font=font(11))
        elif index == 2:
            colors = signed_colors(np.linspace(-residual_limit, residual_limit, 620), residual_limit)
            panel.paste(Image.fromarray(np.repeat(colors[None], 8, axis=0)), (50, 432))
            draw.text((50, 445), f"-{residual_limit * 1000:.2f} mm             0             +{residual_limit * 1000:.2f} mm; common pixels only", fill=MUTED, font=font(11))
        else:
            draw.text((50, 445), "Dark pixels: neither observed; not evidence of zero depth error", fill=MUTED, font=font(11))
        panels.append(panel)
    diagnostics = {
        "real_visible_pixels": int(real_valid.sum()),
        "virtual_visible_pixels": int(virtual_valid.sum()),
        "common_visible_pixels": int(common.sum()),
        "common_pixel_fraction_of_union": float(common.sum() / max(all_valid.sum(), 1)),
        "pixel_residual_bias_m": float(residual[common].mean()) if np.any(common) else None,
        "pixel_residual_p95_absolute_m": float(np.quantile(np.abs(residual[common]), 0.95)) if np.any(common) else None,
        "pixel_residual_color_limit_m": residual_limit,
    }
    return quad_canvas(panels), diagnostics


def metric_mm(value: float | None) -> str:
    return "N/A (no common support)" if value is None else f"{value * 1000:.2f} mm"


def draw_metric_cards(panel: Image.Image, metrics: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    draw = ImageDraw.Draw(panel)
    cards = [
        ("Footprint IoU", f"{metrics['footprint_iou']:.4f}"),
        ("Bounding-box area ratio", f"{metrics['area_ratio']:.3f}x"),
        ("Grid-cell depth p95", metric_mm(metrics.get('depth_p95_absolute_error'))),
        ("Candidate outside metric grid", f"{diagnostics['candidate_points_outside_evaluator_grid']:,} pts"),
    ]
    x, y = 35, 95
    for title, value in cards:
        draw.rounded_rectangle((x, y, x + 290, y + 82), radius=10, fill=BACKGROUND, outline=GRID)
        draw.text((x + 14, y + 12), title, fill=MUTED, font=font(12))
        draw.text((x + 14, y + 38), value, fill=TEXT, font=font(22))
        y += 95


def make_footprint_dashboard(
    reference: np.ndarray,
    candidate: np.ndarray,
    metrics: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    cell_size = float(metrics["cell_size"])
    metric_lower = np.asarray(metrics["grid_lower_xy"], dtype=np.float32)
    metric_upper = np.asarray(metrics["grid_upper_xy"], dtype=np.float32)
    reference_grid = occupancy_grid(reference, metric_lower, metric_upper, cell_size)
    candidate_grid = occupancy_grid(candidate, metric_lower, metric_upper, cell_size)
    metric_cells = np.floor((candidate[:, :2] - metric_lower) / cell_size).astype(np.int64)
    metric_shape = np.asarray(metrics["grid_shape"], dtype=np.int64)
    inside_metric = np.all((metric_cells >= 0) & (metric_cells < metric_shape), axis=1)
    diagnostics = {
        "evaluator_grid_lower_xy_m": metric_lower.tolist(),
        "evaluator_grid_upper_xy_m": metric_upper.tolist(),
        "candidate_points_outside_evaluator_grid": int((~inside_metric).sum()),
        "candidate_fraction_outside_evaluator_grid": float((~inside_metric).mean()),
    }

    panel_size = (700, 470)
    union = make_panel(*panel_size, "Unaligned footprint occupancy", "Green: overlap; blue: real only; orange: Taichi only. Uses the evaluator grid spanning both clouds.")
    union.paste(resize_for_panel(occupancy_rgb(reference_grid, candidate_grid), 620, 340), (50, 88))

    metric_grid = make_panel(*panel_size, "Evaluator grid boundary diagnostic", "Grid bounds include both clouds. Red points would indicate an evaluator-range defect.")
    bounds = (metric_lower, metric_upper)
    draw_scatter(metric_grid, reference, (0, 1), BLUE, *bounds, "camera optical X (m)", "camera optical Y (m)", 16000, dot_radius=1)
    draw_scatter(metric_grid, candidate[inside_metric], (0, 1), ORANGE, *bounds, "camera optical X (m)", "camera optical Y (m)", 16000, dot_radius=1, overlay=True)
    draw_scatter(metric_grid, candidate[~inside_metric], (0, 1), RED, *bounds, "camera optical X (m)", "camera optical Y (m)", 16000, dot_radius=2, overlay=True)

    profile = make_panel(*panel_size, "Footprint width profiles", "Per-cell occupancy counts: blue real, orange Taichi; axes use the evaluator grid shown above")
    draw = ImageDraw.Draw(profile)
    ref_x = reference_grid.sum(axis=1)
    cand_x = candidate_grid.sum(axis=1)
    ref_y = reference_grid.sum(axis=0)
    cand_y = candidate_grid.sum(axis=0)
    left, top, right, bottom = 55, 95, 650, 400
    draw.rectangle((left, top, right, bottom), outline=GRID)
    maximum = max(int(ref_x.max()), int(cand_x.max()), int(ref_y.max()), int(cand_y.max()), 1)
    for values, color, offset in ((ref_x, BLUE, 0), (cand_x, ORANGE, 0), (ref_y, CYAN, 150), (cand_y, MAGENTA, 150)):
        points = []
        for index, value in enumerate(values):
            x = left + int((right - left) * index / max(len(values) - 1, 1))
            y = bottom - int((bottom - top - 10) * value / maximum) + offset // 5
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
    draw.text((left, 415), "blue/orange: X-axis profiles; cyan/magenta: Y-axis profiles", fill=MUTED, font=font(11))

    cards = make_panel(*panel_size, "Benchmark metric summary", "All values are from the unaligned static evaluator")
    draw_metric_cards(cards, metrics, diagnostics)
    return quad_canvas([union, metric_grid, profile, cards]), diagnostics


def write_html_index(
    output_dir: Path,
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    floor_diagnostics: dict[str, Any] | None = None,
) -> None:
    cards = [
        ("Footprint IoU", f"{metrics['footprint_iou']:.4f}"),
        ("Bounding-box area ratio", f"{metrics['area_ratio']:.3f}x"),
        ("Grid-cell depth p95", metric_mm(metrics.get('depth_p95_absolute_error'))),
        ("Evaluator-grid exclusions", f"{diagnostics['candidate_points_outside_evaluator_grid']:,} candidate points"),
    ]
    if floor_diagnostics is not None:
        thickness = floor_diagnostics["local_thickness_m"]
        cards.extend([
            ("Reconstructed volume", f"{floor_diagnostics['object_volume_m3'] * 1e6:.3f} mL"),
            ("Floor columns", f"{floor_diagnostics['accepted_columns']:,} accepted"),
            ("Thickness p05–p95", f"{float(thickness['p05']) * 1000:.2f}–{float(thickness['p95']) * 1000:.2f} mm"),
        ])
    card_html = "".join(f"<article><h2>{html.escape(label)}</h2><strong>{html.escape(value)}</strong></article>" for label, value in cards)
    images = [
        ("Source and reconstruction", "01_source_and_reconstruction.png", "Observed point cloud, volume fill, visible-surface reservation, and sampled particles."),
        ("Calibration and initialized scene", "02_calibration_and_scene.png", "Source-to-scene calibration and the actual Taichi particle state before the first MPM substep."),
        ("Depth and visibility", "03_depth_and_visibility.png", "Raw z-buffer depth, signed residuals, and pixel-level visibility agreement."),
        ("Footprint and metric diagnostics", "04_footprint_and_metrics.png", "Unaligned occupancy plus the evaluator-grid-boundary diagnostic."),
    ]
    if floor_diagnostics is not None:
        images.extend([
            ("Floor segmentation and volume fill", "05_floor_reconstruction.png", "Retained and rejected points, calibrated floor samples, per-column intersections, local thickness, and reconstructed volume."),
            ("Metric reference frames", "06_reference_frames.png", "Scene, source, camera, and selected AprilTag origins and axes expressed in the metric scene frame."),
        ])
    sections = "".join(
        f"<section><h1>{html.escape(title)}</h1><p>{html.escape(description)}</p><img src='{filename}' alt='{html.escape(title)}'></section>"
        for title, filename, description in images
    )
    warning = ""
    if diagnostics["candidate_points_outside_evaluator_grid"]:
        warning = (
            "<aside><strong>Review item:</strong> candidate points fall outside the evaluator grid despite the grid being "
            "derived from both clouds. Review the rasterization bounds before using IoU as a final acceptance criterion.</aside>"
        )
    content = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Static Top-View Benchmark Diagnostics</title>
<style>
body {{ margin: 0; background:#111721; color:#e5ebf1; font:16px system-ui,sans-serif; }}
main {{ max-width:1480px; padding:28px; margin:auto; }} h1 {{ margin-bottom:4px; }} p {{ color:#a4b2c2; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin:24px 0; }}
article, aside {{ background:#192230; border:1px solid #3d4a5b; border-radius:10px; padding:16px; }} article h2 {{ color:#a4b2c2; font-size:14px; margin:0 0 8px; }} article strong {{ font-size:26px; }}
section {{ margin:30px 0 42px; }} img {{ width:100%; border:1px solid #3d4a5b; border-radius:8px; background:#111721; }} aside {{ border-color:#f6a743; color:#f6d59b; line-height:1.5; }} code {{ color:#74ca89; }}
</style></head><body><main>
<h1>Static top-view benchmark diagnostics</h1>
<aside><strong>Initialization only — dynamics not evaluated.</strong> One captured observation is compared with its reconstructed initial particle state before any MPM substep. These scores do not establish material behavior, tool response, time alignment, or dynamic fidelity. Hidden volume and camera calibration remain modeling assumptions.</aside>
<p>Visual evidence for reconstruction, calibration, initial-state export, and unaligned scoring. This report does not alter either point cloud. Depth errors are in calibrated scene units, not necessarily unscaled source-camera units.</p>
<p>Depth p95 below is evaluated on optical XY grid cells, not pixels. Pixel depth p95 is {metric_mm(diagnostics.get('pixel_residual_p95_absolute_m'))} on common visible support. The area ratio is a bounding-box ratio, not occupied footprint area. Exported grayscale depth PNGs are independently normalized and must not be used to assess dynamics.</p>
<div class=\"cards\">{card_html}</div>{warning}{sections}
<p>Machine-readable diagnostics: <code>visualization_diagnostics.json</code></p>
</main></body></html>"""
    (output_dir / "index.html").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_scatter_points <= 0:
        raise ValueError("--max-scatter-points must be positive")
    reconstruction = load_json(args.reconstruction_metadata)
    taichi = load_json(args.taichi_metadata)
    evaluation = load_json(args.metrics)
    calibration = load_calibration(args.calibration)
    calibration_source_frame = str(getattr(calibration, "source_frame"))
    calibration_scene_frame = str(getattr(calibration, "scene_frame", "world"))
    if evaluation.get("calibration", {}).get("fingerprint") != calibration.fingerprint:
        raise ValueError("Metric artifact calibration does not match --calibration")
    if taichi.get("calibration", {}).get("fingerprint") != calibration.fingerprint:
        raise ValueError("Taichi artifact calibration does not match --calibration")

    reconstruction_dir = args.reconstruction_metadata.parent
    outputs = reconstruction["outputs"]
    array_frames = reconstruction.get("array_frames", {})
    source_value = outputs.get("source_filtered_points_xyz") or outputs["real_points_xyz"]
    source = np.load(resolve_artifact(source_value, args.reconstruction_metadata)).astype(np.float32)
    observed = np.load(resolve_artifact(outputs["real_points_xyz"], args.reconstruction_metadata)).astype(np.float32)
    voxels = np.load(resolve_artifact(outputs["voxel_centers_xyz"], args.reconstruction_metadata)).astype(np.float32)
    surface = np.load(resolve_artifact(outputs["surface_particles_xyz"], args.reconstruction_metadata)).astype(np.float32)
    particles = np.load(resolve_artifact(outputs["sampled_particles_xyz"], args.reconstruction_metadata)).astype(np.float32)
    reconstruction_frame = array_frames.get("real_points_xyz", calibration_source_frame)
    for key in ("voxel_centers_xyz", "surface_particles_xyz", "sampled_particles_xyz"):
        if array_frames.get(key, reconstruction_frame) != reconstruction_frame:
            raise ValueError(f"Reconstruction array {key} does not use frame {reconstruction_frame!r}")
    if reconstruction_frame == calibration_scene_frame:
        observed_scene = observed
    elif reconstruction_frame == calibration_source_frame:
        observed_scene = apply_calibration(observed, calibration)
    else:
        raise ValueError(
            f"Reconstruction points use frame {reconstruction_frame!r}, expected "
            f"{calibration_source_frame!r} or {calibration_scene_frame!r}"
        )
    transformed_scene = None
    transformed_value = outputs.get("scene_transformed_points_xyz")
    if transformed_value is not None:
        transformed_scene = np.load(resolve_artifact(transformed_value, args.reconstruction_metadata)).astype(np.float32)
        if array_frames.get("scene_transformed_points_xyz") != calibration_scene_frame:
            raise ValueError("scene_transformed_points_xyz must use the calibrated scene frame")
    view = primary_view(taichi)
    taichi_particles = np.load(resolve_artifact(taichi["frames"][0]["particles"], args.taichi_metadata)).astype(np.float32)
    virtual_depth = np.load(resolve_artifact(view["depth_array"], args.taichi_metadata)).astype(np.float32)
    virtual_optical = np.load(resolve_artifact(view["pointcloud_array"], args.taichi_metadata)).astype(np.float32)[:, :3]
    evaluator_filename = "real_visible_camera_optical_xyz.npy"
    evaluator_candidates = (
        args.metrics.parent / evaluator_filename,
        reconstruction_dir / evaluator_filename,
    )
    real_optical_path = next((path for path in evaluator_candidates if path.is_file()), None)
    if real_optical_path is None:
        raise FileNotFoundError(
            f"Missing evaluator output {evaluator_filename!r} beside {args.metrics} or {args.reconstruction_metadata}"
        )
    real_optical = np.load(real_optical_path).astype(np.float32)
    calibrated_source = apply_calibration(source, calibration)
    real_depth, real_nearest, *_ = rasterize_depth(
        observed_scene,
        int(view["width"]),
        int(view["height"]),
        calibration.camera,
        splat_radius=int(view.get("splat_radius", 0)),
    )
    real_valid = real_nearest >= 0
    virtual_valid = virtual_depth < float(calibration.camera["zFar"])

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dashboard = make_source_reconstruction_dashboard(
        observed,
        voxels,
        surface,
        particles,
        reconstruction_frame,
        args.max_scatter_points,
    )
    source_dashboard.save(output_dir / "01_source_and_reconstruction.png")
    calibration_dashboard = make_calibration_dashboard(source, calibrated_source, taichi_particles, calibration, args.max_scatter_points)
    calibration_dashboard.save(output_dir / "02_calibration_and_scene.png")
    depth_dashboard, depth_diagnostics = depth_panels(real_depth, real_valid, virtual_depth, virtual_valid)
    depth_dashboard.save(output_dir / "03_depth_and_visibility.png")
    footprint_dashboard, footprint_diagnostics = make_footprint_dashboard(real_optical, virtual_optical, evaluation["metrics"])
    footprint_dashboard.save(output_dir / "04_footprint_and_metrics.png")

    floor_diagnostics = None
    reference_diagnostics = None
    if reconstruction.get("fill", {}).get("mode") == "floor":
        if transformed_scene is None:
            raise ValueError("Floor diagnostics require scene_transformed_points_xyz")
        floor_dashboard, floor_diagnostics = make_floor_reconstruction_dashboard(
            transformed_scene,
            observed_scene,
            surface,
            voxels,
            reconstruction,
            calibration,
            args.max_scatter_points,
        )
        floor_dashboard.save(output_dir / "05_floor_reconstruction.png")
        reference_dashboard, reference_diagnostics = make_reference_frame_dashboard(
            observed_scene,
            reconstruction,
            calibration,
            args.max_scatter_points,
        )
        reference_dashboard.save(output_dir / "06_reference_frames.png")

    diagnostics = {
        "visualization": "static-topview-diagnostics/v1",
        "calibration_fingerprint": calibration.fingerprint,
        "input_points": {
            "source": int(len(source)),
            "reconstruction_visible": int(len(observed)),
            "voxels": int(len(voxels)),
            "surface_particles": int(len(surface)),
            "reconstruction_particles": int(len(particles)),
            "taichi_particles": int(len(taichi_particles)),
            "real_visible_optical": int(len(real_optical)),
            "virtual_visible_optical": int(len(virtual_optical)),
        },
        "depth": depth_diagnostics,
        "footprint": footprint_diagnostics,
        "floor_reconstruction": floor_diagnostics,
        "reference_frames": reference_diagnostics,
        "inputs": {
            "reconstruction_metadata": str(args.reconstruction_metadata),
            "taichi_metadata": str(args.taichi_metadata),
            "metrics": str(args.metrics),
            "calibration": str(args.calibration),
        },
    }
    (output_dir / "visualization_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    write_html_index(
        output_dir,
        evaluation["metrics"],
        {**depth_diagnostics, **footprint_diagnostics},
        floor_diagnostics,
    )
    print(f"Wrote visual benchmark report: {output_dir / 'index.html'}")
    print(f"Wrote visualization diagnostics: {output_dir / 'visualization_diagnostics.json'}")


if __name__ == "__main__":
    main()
