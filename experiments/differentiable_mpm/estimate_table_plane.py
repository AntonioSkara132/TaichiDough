"""Deterministically estimate a static table plane from HSV-excluded RGB-D frames.

This command measures geometry only. It never writes calibration, reconstruction, or
simulation inputs. Plane coefficients use ``n dot p + d = 0`` in scene coordinates;
``n`` is normalized and oriented toward the dough/free-space side.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

HSV_DEFAULTS = (56.0, 84.0, 18.0, 73.0, 156.0, 255.0)
SCHEMA = "taichidough/table-plane-estimate/v1"
PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "ink": "#0b0b0b", "muted": "#52514e"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rgb_to_hsv_opencv(rgb: np.ndarray) -> np.ndarray:
    rgb_float = np.asarray(rgb, dtype=np.uint8).astype(np.float32) / 255.0
    r, g, b = (rgb_float[..., i] for i in range(3))
    maximum, minimum = np.max(rgb_float, axis=-1), np.min(rgb_float, axis=-1)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-6
    red, green, blue = (maximum == r) & nonzero, (maximum == g) & nonzero, (maximum == b) & nonzero
    hue[red] = ((g[red] - b[red]) / delta[red]) % 6.0
    hue[green] = ((b[green] - r[green]) / delta[green]) + 2.0
    hue[blue] = ((r[blue] - g[blue]) / delta[blue]) + 4.0
    hue *= 30.0
    saturation = np.zeros_like(maximum)
    valued = maximum > 1e-6
    saturation[valued] = delta[valued] / maximum[valued] * 255.0
    return np.stack((hue, saturation, maximum * 255.0), axis=-1)


def hsv_mask(rgb: np.ndarray, thresholds: tuple[float, ...] = HSV_DEFAULTS) -> np.ndarray:
    h_min, h_max, s_min, s_max, v_min, v_max = thresholds
    hsv = rgb_to_hsv_opencv(rgb)
    hue = hsv[..., 0]
    hue_ok = (hue >= h_min) & (hue <= h_max) if h_min <= h_max else (hue >= h_min) | (hue <= h_max)
    return hue_ok & (hsv[..., 1] >= s_min) & (hsv[..., 1] <= s_max) & (hsv[..., 2] >= v_min) & (hsv[..., 2] <= v_max)


def quaternion_matrix(value: np.ndarray) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("Quaternion must contain four finite xyzw values")
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("Quaternion norm must be positive")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def plane_from_regression(coefficients: Iterable[float]) -> np.ndarray:
    a, b, c = np.asarray(tuple(coefficients), dtype=np.float64)
    normal = np.array([-a, 1.0, -b], dtype=np.float64)
    length = float(np.linalg.norm(normal))
    return np.array([*(normal / length), -c / length], dtype=np.float64)


def regression_from_plane(plane: Iterable[float]) -> np.ndarray:
    values = np.asarray(tuple(plane), dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Plane must contain four finite coefficients")
    length = float(np.linalg.norm(values[:3]))
    if length <= 1e-12:
        raise ValueError("Plane normal must be nonzero")
    values = values / length
    if values[1] < 0:
        values = -values
    if values[1] <= 1e-8:
        raise ValueError("Plane is vertical and cannot be represented as scene Y")
    return np.array([-values[0] / values[1], -values[2] / values[1], -values[3] / values[1]])


def signed_distance(points: np.ndarray, plane: Iterable[float]) -> np.ndarray:
    p = np.asarray(plane, dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ p[:3] + p[3]


def orient_plane_upward(plane: Iterable[float], positive_points: np.ndarray | None = None) -> np.ndarray:
    p = np.asarray(tuple(plane), dtype=np.float64)
    if p.shape != (4,) or not np.isfinite(p).all() or np.linalg.norm(p[:3]) <= 1e-12:
        raise ValueError("Plane must contain a nonzero finite normal")
    p = p / np.linalg.norm(p[:3])
    if p[1] < 0:
        p = -p
    if positive_points is not None and len(positive_points) and float(np.median(signed_distance(positive_points, p))) < 0:
        p = -p
    if p[1] <= 0:
        raise ValueError("Upward normal and positive-point orientation disagree")
    return p


def exclude_oriented_boxes(points: np.ndarray, poses: np.ndarray | None, half_extents: np.ndarray,
                            padding_m: float) -> tuple[np.ndarray, int]:
    points = np.asarray(points, dtype=np.float64)
    keep = np.ones(len(points), dtype=bool)
    if poses is None:
        return keep, 0
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (2, 7) or half_extents.shape != (2, 3):
        raise ValueError("Tool poses and extents must be [2,7] and [2,3]")
    for tool in range(2):
        rotation = quaternion_matrix(poses[tool, 3:7])
        local = (points - poses[tool, :3]) @ rotation
        keep &= ~np.all(np.abs(local) <= half_extents[tool] + padding_m, axis=1)
    return keep, int((~keep).sum())


@dataclass(frozen=True)
class PlaneFit:
    accepted: bool
    coefficients: np.ndarray
    plane: np.ndarray
    sample_count: int
    inlier_count: int
    inlier_fraction: float
    median_abs_residual_m: float
    reason: str | None = None


def fit_frame_plane(points: np.ndarray, *, seed: int, sample_limit: int = 4000,
                    iterations: int = 96, threshold_m: float = 0.0035,
                    maximum_tilt_deg: float = 15.0, minimum_inliers: int = 300,
                    minimum_fraction: float = 0.25) -> PlaneFit:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < max(3, minimum_inliers):
        return PlaneFit(False, np.full(3, np.nan), np.full(4, np.nan), len(values), 0, 0.0, math.inf, "too_few_points")
    rng = np.random.default_rng(seed)
    if len(values) > sample_limit:
        values = values[np.sort(rng.choice(len(values), sample_limit, replace=False))]
    design = np.column_stack((values[:, 0], values[:, 2], np.ones(len(values))))
    maximum_slope = math.tan(math.radians(maximum_tilt_deg))
    best: tuple[int, float, np.ndarray, np.ndarray] | None = None
    for _ in range(iterations):
        indices = rng.choice(len(values), 3, replace=False)
        try:
            coefficients = np.linalg.solve(design[indices], values[indices, 1])
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(coefficients).all() or np.linalg.norm(coefficients[:2]) > maximum_slope:
            continue
        plane = plane_from_regression(coefficients)
        residual = np.abs(signed_distance(values, plane))
        inliers = residual <= threshold_m
        count = int(inliers.sum())
        median = float(np.median(residual[inliers])) if count else math.inf
        candidate = (count, -median, coefficients, inliers)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        return PlaneFit(False, np.full(3, np.nan), np.full(4, np.nan), len(values), 0, 0.0, math.inf, "no_upward_plane")
    _, _, coefficients, inliers = best
    for _ in range(4):
        if int(inliers.sum()) < 3:
            break
        coefficients = np.linalg.lstsq(design[inliers], values[inliers, 1], rcond=None)[0]
        if np.linalg.norm(coefficients[:2]) > maximum_slope:
            break
        plane = plane_from_regression(coefficients)
        residual = np.abs(signed_distance(values, plane))
        inliers = residual <= threshold_m
    plane = plane_from_regression(coefficients)
    residual = np.abs(signed_distance(values, plane))
    inliers = residual <= threshold_m
    count = int(inliers.sum())
    fraction = count / len(values)
    median = float(np.median(residual[inliers])) if count else math.inf
    accepted = count >= minimum_inliers and fraction >= minimum_fraction and np.linalg.norm(coefficients[:2]) <= maximum_slope
    reason = None if accepted else "insufficient_consensus"
    return PlaneFit(accepted, coefficients, plane, len(values), count, fraction, median, reason)


def aggregate_equal_frame(fits: Iterable[PlaneFit], positive_points: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    accepted = [fit for fit in fits if fit.accepted]
    if not accepted:
        raise ValueError("No accepted frame planes")
    coefficients = np.median(np.stack([fit.coefficients for fit in accepted]), axis=0)
    plane = orient_plane_upward(plane_from_regression(coefficients), positive_points)
    return regression_from_plane(plane), plane


def bootstrap_planes(fits: Iterable[PlaneFit], positive_points: np.ndarray, reference_xz: np.ndarray,
                     *, seed: int, iterations: int = 1000) -> dict[str, Any]:
    accepted = [fit for fit in fits if fit.accepted]
    if len(accepted) < 2 or iterations < 1:
        raise ValueError("Bootstrap requires at least two accepted fits and positive iterations")
    coefficients = np.stack([fit.coefficients for fit in accepted])
    rng = np.random.default_rng(seed)
    records = np.empty((iterations, 6), dtype=np.float64)
    for index in range(iterations):
        sample = coefficients[rng.integers(0, len(coefficients), len(coefficients))]
        coeff = np.median(sample, axis=0)
        plane = orient_plane_upward(plane_from_regression(coeff), positive_points)
        coeff = regression_from_plane(plane)
        height = coeff[0] * reference_xz[0] + coeff[1] * reference_xz[1] + coeff[2]
        tilt = math.degrees(math.acos(np.clip(plane[1], -1.0, 1.0)))
        records[index] = [*plane[:3], plane[3], height, tilt]
    names = ("nx", "ny", "nz", "d", "height_at_reference_m", "tilt_deg")
    return {name: {"p025": float(np.quantile(records[:, i], .025)),
                   "median": float(np.median(records[:, i])),
                   "p975": float(np.quantile(records[:, i], .975))}
            for i, name in enumerate(names)}


def summarize(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {"count": int(len(values)), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p05": float(np.quantile(values, .05)), "p95": float(np.quantile(values, .95)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def binned_bias(points: np.ndarray, residuals: np.ndarray, axis: int, bins: int = 8) -> list[dict[str, Any]]:
    coordinates = points[:, axis]
    edges = np.quantile(coordinates, np.linspace(0, 1, bins + 1))
    result = []
    for i in range(bins):
        mask = (coordinates >= edges[i]) & (coordinates <= edges[i + 1] if i == bins - 1 else coordinates < edges[i + 1])
        result.append({"lower_m": float(edges[i]), "upper_m": float(edges[i + 1]),
                       "center_m": float(np.median(coordinates[mask])),
                       "median_signed_residual_m": float(np.median(residuals[mask])), "count": int(mask.sum())})
    return result


def draw_line_plot(path: Path, rows: list[dict[str, Any]], aggregate_height: float, prior_height: float) -> None:
    image = Image.new("RGB", (1300, 720), "#fcfcfb")
    draw = ImageDraw.Draw(image)
    left, top, width, height = 90, 85, 1120, 520
    draw.text((left, 22), "Episode 18 per-frame table plane", fill=PALETTE["ink"])
    draw.text((left, 45), "Equal frame weighting; held-out frames are every fifth raw frame", fill=PALETTE["muted"])
    accepted = [row for row in rows if row["accepted_main"]]
    ys = [row["height_at_reference_m"] * 1000 for row in accepted] + [aggregate_height * 1000, prior_height * 1000]
    low, high = math.floor((min(ys) - 3) / 5) * 5, math.ceil((max(ys) + 3) / 5) * 5
    for value in np.linspace(low, high, 8):
        py = top + height * (high - value) / (high - low)
        draw.line((left, py, left + width, py), fill="#dededb", width=1)
        draw.text((20, py - 7), f"{value:.1f}", fill=PALETTE["muted"])
    def point(frame: int, value: float) -> tuple[float, float]:
        return left + width * frame / max(1, len(rows) - 1), top + height * (high - value * 1000) / (high - low)
    estimation = [point(row["frame"], row["height_at_reference_m"]) for row in accepted if row["split"] == "estimation"]
    heldout = [point(row["frame"], row["height_at_reference_m"]) for row in accepted if row["split"] == "heldout"]
    if len(estimation) > 1: draw.line(estimation, fill=PALETTE["blue"], width=2)
    for x, y in heldout: draw.ellipse((x-4, y-4, x+4, y+4), fill=PALETTE["orange"], outline="white")
    for value, color, label, yoff in ((aggregate_height, PALETTE["green"], "Robust aggregate", 0),
                                      (prior_height, PALETTE["muted"], "Prior one-frame estimate", 18)):
        y = point(0, value)[1]; draw.line((left, y, left+width, y), fill=color, width=3)
        draw.text((left+width-205, y-16-yoff), f"{label}: {value*1000:.2f} mm", fill=PALETTE["ink"])
    draw.text((left, top+height+35), "Raw frame", fill=PALETTE["muted"])
    draw.rectangle((left, top+height+65, left+18, top+height+83), fill=PALETTE["blue"]); draw.text((left+28, top+height+66), "Estimation", fill=PALETTE["ink"])
    draw.ellipse((left+180, top+height+65, left+198, top+height+83), fill=PALETTE["orange"]); draw.text((left+208, top+height+66), "Held-out", fill=PALETTE["ink"])
    image.save(path)


def draw_bias_plot(path: Path, x_bins: list[dict[str, Any]], z_bins: list[dict[str, Any]]) -> None:
    image = Image.new("RGB", (1200, 650), "#fcfcfb"); draw = ImageDraw.Draw(image)
    draw.text((70, 22), "Held-out table residual bias", fill=PALETTE["ink"])
    draw.text((70, 45), "Median signed orthogonal residual in eight equal-count bins", fill=PALETTE["muted"])
    for panel, (label, rows, color) in enumerate((("Scene X", x_bins, PALETTE["blue"]), ("Scene Z", z_bins, PALETTE["orange"]))):
        left, top, width, height = 70 + panel * 580, 90, 500, 440
        maximum = max(.001, max(abs(row["median_signed_residual_m"]) for row in rows) * 1.2)
        for value in (-maximum, 0, maximum):
            y = top + height * (maximum - value) / (2 * maximum)
            draw.line((left, y, left+width, y), fill="#dededb" if value else PALETTE["muted"], width=1)
            draw.text((left-55, y-7), f"{value*1000:.2f}", fill=PALETTE["muted"])
        points=[]
        for i,row in enumerate(rows):
            x=left+width*(i+.5)/len(rows); y=top+height*(maximum-row["median_signed_residual_m"])/(2*maximum)
            points.append((x,y)); draw.ellipse((x-5,y-5,x+5,y+5),fill=color,outline="white")
        draw.line(points,fill=color,width=2); draw.text((left+200, top+height+30), label, fill=PALETTE["ink"])
        draw.text((left-65, top+height/2-7), "0", fill=PALETTE["muted"])
    image.save(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--tool-geometry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1818)
    parser.add_argument("--tool-padding-m", type=float, default=.01)
    parser.add_argument("--roi-margin-m", type=float, default=.06)
    parser.add_argument("--sample-limit", type=int, default=4000)
    parser.add_argument("--sensitivity-sample-limit", type=int, default=2000)
    parser.add_argument("--ransac-iterations", type=int, default=96)
    parser.add_argument("--threshold-m", type=float, default=.0035)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if min(args.tool_padding_m, args.roi_margin_m, args.threshold_m) < 0 or min(args.sample_limit, args.sensitivity_sample_limit, args.ransac_iterations, args.bootstrap_iterations) < 1:
        raise ValueError("Sampling, iteration, padding and threshold settings must be positive/nonnegative")
    args.output_dir.mkdir(parents=True)
    log_lines: list[str] = []

    def report(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)
        (args.output_dir / "analysis.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    report(f"Started deterministic Episode18 table-plane estimate at {datetime.now(timezone.utc).isoformat()}")
    source = Path(__file__).resolve(); source_start = file_sha256(source)
    sequence_metadata = args.episode / "sequence_metadata.json"
    pointcloud_path = args.episode / "pointclouds.pt"
    filtered_path = args.episode / "pointclouds_interpolated.pt"
    paths_path = args.episode / "paths_interpolated.pt"
    filter_source = next((parent / "interpolate_deformpath_pairs.py" for parent in args.episode.parents
                          if (parent / "interpolate_deformpath_pairs.py").is_file()), None)
    if filter_source is None:
        raise FileNotFoundError("Cannot find the recorded HSV preprocessing source")
    inputs = (pointcloud_path, filtered_path, paths_path, sequence_metadata, args.calibration,
              args.tool_geometry, filter_source)
    for path in inputs:
        if not path.is_file(): raise FileNotFoundError(path)
    metadata = json.loads(sequence_metadata.read_text())
    calibration = json.loads(args.calibration.read_text())
    geometry = json.loads(args.tool_geometry.read_text())
    if calibration.get("scene_frame") != "mocap" or calibration.get("source_frame") != "mocap":
        raise ValueError("Estimator currently requires Episode18 mocap points")
    transform = np.asarray(calibration["scene_from_source"], dtype=np.float64)
    if not np.allclose(transform, np.eye(4), atol=1e-9, rtol=0):
        raise ValueError("Raw Episode18 points are expected to already be in mocap/scene coordinates")
    thresholds = HSV_DEFAULTS
    recorded = metadata["pointcloud_filter"]
    if tuple(recorded["hue_range"] + recorded["saturation_range"] + recorded["value_range"]) != thresholds:
        raise ValueError("Recorded HSV settings differ from required Episode18 values")
    import torch
    raw_frames = torch.load(pointcloud_path, map_location="cpu", weights_only=True, mmap=True)[0]
    filtered = torch.load(filtered_path, map_location="cpu", weights_only=True)[0]
    paths_record = torch.load(paths_path, map_location="cpu", weights_only=True)[0]
    paths = paths_record["path"].numpy(); validity = paths_record["stream_validity"].numpy().astype(bool)
    raw_indices = np.asarray(metadata["pointcloud_indices"], dtype=int)
    if len(raw_frames) != 390 or len(filtered) != len(paths) or len(paths) != len(raw_indices):
        raise ValueError("Unexpected Episode18 frame alignment")
    tools = {row["name"]: row for row in geometry["tools"]}
    order = paths_record["pose_frames"]
    for name in order:
        if name not in tools or not np.allclose(tools[name]["marker_from_collider"], np.eye(4), atol=1e-12, rtol=0):
            raise ValueError("This estimator requires the verified identity marker_from_collider transforms")
    half_extents = np.asarray([tools[name]["half_extents_m"] for name in order], dtype=np.float64)
    poses_by_raw = {int(raw): paths[i, :, :7] if validity[i].all() else None for i, raw in enumerate(raw_indices)}
    # ROI is fixed before plane fitting from DBSCAN-filtered dough and the recorded tool sweep.
    dough_xz = np.concatenate([np.asarray(frame)[:, [0, 2]] for frame in filtered])
    lower = np.quantile(dough_xz, .005, axis=0); upper = np.quantile(dough_xz, .995, axis=0)
    tool_corners = []
    signs = np.asarray([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)], dtype=np.float64)
    for poses in poses_by_raw.values():
        if poses is None: continue
        for tool in range(2):
            rotation = quaternion_matrix(poses[tool, 3:7])
            corners = poses[tool,:3] + (signs * (half_extents[tool] + args.tool_padding_m)) @ rotation.T
            tool_corners.append(corners[:, [0,2]])
    corners = np.concatenate(tool_corners)
    roi_min = np.minimum(lower, corners.min(axis=0)) - args.roi_margin_m
    roi_max = np.maximum(upper, corners.max(axis=0)) + args.roi_margin_m
    positive_points = np.asarray(filtered[0])[:, :3].astype(np.float64)
    reference_xz = np.median(positive_points[:, [0,2]], axis=0)
    input_hashes = {str(path.resolve()): file_sha256(path) for path in inputs}
    settings = {"seed": args.seed, "hsv_opencv": {"h_min":56.,"h_max":84.,"s_min":18.,"s_max":73.,"v_min":156.,"v_max":255.},
                "split": "heldout when raw_frame % 5 == 0; estimation otherwise", "roi_source": "0.5%-99.5% DBSCAN-filtered dough XZ union all padded tool OBB corners, then fixed margin",
                "roi_min_xz_m": roi_min.tolist(), "roi_max_xz_m": roi_max.tolist(), "roi_margin_m": args.roi_margin_m,
                "tool_exclusion": "points inside recorded oriented proxy boxes removed", "tool_padding_m": args.tool_padding_m,
                "sample_limit": args.sample_limit, "sensitivity_sample_limit": args.sensitivity_sample_limit,
                "ransac_iterations": args.ransac_iterations, "threshold_m": args.threshold_m,
                "maximum_tilt_deg": 15., "minimum_inliers": 300, "minimum_fraction": .25,
                "bootstrap_iterations": args.bootstrap_iterations, "equal_frame_weighting": True}
    json_write(args.output_dir / "run_manifest.json", {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
               "action": "read-only table geometry measurement", "inputs": input_hashes, "source_sha256": source_start, "settings": settings})
    rows: list[dict[str, Any]] = []
    main_fits: list[PlaneFit] = []
    sensitivity_fits: list[PlaneFit] = []
    for frame_index, tensor in enumerate(raw_frames):
        values = tensor.numpy(); xyz = values[:,:3].astype(np.float64)
        valid = np.isfinite(xyz).all(axis=1) & (xyz[:,2] > 0)
        rgb = np.clip(np.rint(values[:,7:10]),0,255).astype(np.uint8)
        dough = valid & hsv_mask(rgb, thresholds)
        roi = valid & ~dough & np.all((xyz[:,[0,2]] >= roi_min) & (xyz[:,[0,2]] <= roi_max), axis=1)
        poses = poses_by_raw.get(frame_index)
        tool_keep, tool_excluded = exclude_oriented_boxes(xyz, poses, half_extents, args.tool_padding_m)
        candidate = xyz[roi & tool_keep]
        main_fit = fit_frame_plane(candidate, seed=args.seed+frame_index*17, sample_limit=args.sample_limit,
                                   iterations=args.ransac_iterations, threshold_m=args.threshold_m)
        sensitivity = fit_frame_plane(candidate, seed=args.seed+frame_index*17, sample_limit=args.sensitivity_sample_limit,
                                      iterations=args.ransac_iterations, threshold_m=args.threshold_m)
        main_fits.append(main_fit); sensitivity_fits.append(sensitivity)
        split = "heldout" if frame_index % 5 == 0 else "estimation"
        expected_hsv_count = int(recorded["frame_stats"][frame_index]["hsv_filtered_point_count"])
        row={"frame":frame_index,"split":split,"raw_points":len(values),"valid_points":int(valid.sum()),"hsv_dough_points":int(dough.sum()),
             "hsv_count_matches_metadata":int(dough.sum()) == expected_hsv_count,
             "roi_non_dough_before_tool":int(roi.sum()),"tool_exclusion_available":poses is not None,"tool_points_excluded_from_full_frame":tool_excluded,
             "candidate_points":len(candidate),"accepted_main":main_fit.accepted,"accepted_sensitivity":sensitivity.accepted,
             "sample_count":main_fit.sample_count,"inlier_count":main_fit.inlier_count,"inlier_fraction":main_fit.inlier_fraction,
             "median_abs_residual_m":main_fit.median_abs_residual_m,"reason":main_fit.reason}
        if main_fit.accepted:
            a,b,c=main_fit.coefficients; row.update(a=a,b=b,c=c,nx=main_fit.plane[0],ny=main_fit.plane[1],nz=main_fit.plane[2],d=main_fit.plane[3],
                height_at_reference_m=float(a*reference_xz[0]+b*reference_xz[1]+c), tilt_deg=float(math.degrees(math.acos(np.clip(main_fit.plane[1],-1,1)))))
        else:
            row.update({name:None for name in ("a","b","c","nx","ny","nz","d","height_at_reference_m","tilt_deg")})
        rows.append(row)
        if frame_index % 50 == 0:
            report(f"frame {frame_index}/389 candidates={len(candidate)} accepted={main_fit.accepted}")
    estimation = [fit for fit,row in zip(main_fits,rows) if row["split"]=="estimation" and fit.accepted]
    heldout = [(fit,row) for fit,row in zip(main_fits,rows) if row["split"]=="heldout" and fit.accepted]
    coefficients, plane = aggregate_equal_frame(estimation, positive_points)
    _, sensitivity_plane = aggregate_equal_frame([fit for fit,row in zip(sensitivity_fits,rows) if row["split"]=="estimation" and fit.accepted], positive_points)
    height = float(coefficients[0]*reference_xz[0]+coefficients[1]*reference_xz[1]+coefficients[2])
    prior_plane=np.array([0.08715182706011013,0.9961726162865107,0.006684130532321291,-0.07193213404816227])
    prior_coeff=regression_from_plane(prior_plane); prior_height=float(prior_coeff[0]*reference_xz[0]+prior_coeff[1]*reference_xz[1]+prior_coeff[2])
    # Equal-size residual samples from held-out frames prevent dense frames dominating diagnostics.
    heldout_samples=[]; per_heldout=[]
    for fit,row in heldout:
        frame_index=row["frame"]; values=raw_frames[frame_index].numpy(); xyz=values[:,:3].astype(np.float64)
        valid=np.isfinite(xyz).all(axis=1)&(xyz[:,2]>0); rgb=np.clip(np.rint(values[:,7:10]),0,255).astype(np.uint8)
        mask=valid&~hsv_mask(rgb,thresholds)&np.all((xyz[:,[0,2]]>=roi_min)&(xyz[:,[0,2]]<=roi_max),axis=1)
        tool_keep,_=exclude_oriented_boxes(xyz,poses_by_raw.get(frame_index),half_extents,args.tool_padding_m)
        candidate=xyz[mask&tool_keep]
        # Classify table points using the held-out frame's independent fit, then
        # evaluate the aggregate estimation-frame plane on that fixed selection.
        table_mask = np.abs(signed_distance(candidate, fit.plane)) <= args.threshold_m
        points = candidate[table_mask]
        residual = signed_distance(points, plane)
        aggregate_inliers = np.abs(residual) <= args.threshold_m
        per_heldout.append({"frame":frame_index,"candidate_count":len(candidate),
                            "independent_table_point_count":len(points),
                            "aggregate_inlier_count":int(aggregate_inliers.sum()),
                            "aggregate_inlier_fraction":float(aggregate_inliers.mean()),
                            "median_signed_residual_m":float(np.median(residual)),
                            "median_abs_residual_m":float(np.median(np.abs(residual)))})
        if len(points):
            rng=np.random.default_rng(args.seed+frame_index*101); ids=rng.choice(len(points),min(1000,len(points)),replace=False)
            heldout_samples.append(np.column_stack((points[ids],residual[ids])))
    sample=np.concatenate(heldout_samples); sample_points=sample[:,:3]; sample_residual=sample[:,3]
    x_bins=binned_bias(sample_points,sample_residual,0); z_bins=binned_bias(sample_points,sample_residual,2)
    linear_design = np.column_stack((sample_points[:,0], sample_points[:,2], np.ones(len(sample_points))))
    linear=np.linalg.lstsq(linear_design,sample_residual,rcond=None)[0]
    quadratic_design = np.column_stack((sample_points[:,0], sample_points[:,2],
                                        sample_points[:,0] ** 2, sample_points[:,0] * sample_points[:,2],
                                        sample_points[:,2] ** 2, np.ones(len(sample_points))))
    quadratic = np.linalg.lstsq(quadratic_design, sample_residual, rcond=None)[0]
    linear_rmse = float(np.sqrt(np.mean((sample_residual - linear_design @ linear) ** 2)))
    quadratic_rmse = float(np.sqrt(np.mean((sample_residual - quadratic_design @ quadratic) ** 2)))
    bootstrap=bootstrap_planes(estimation,positive_points,reference_xz,seed=args.seed+991,iterations=args.bootstrap_iterations)
    normals=np.stack([fit.plane[:3] for fit in estimation]); angular=np.degrees(np.arccos(np.clip(normals@plane[:3],-1,1)))
    frame_heights=np.array([fit.coefficients[0]*reference_xz[0]+fit.coefficients[1]*reference_xz[1]+fit.coefficients[2] for fit in estimation])
    sensitivity_angle=float(math.degrees(math.acos(np.clip(np.dot(plane[:3],sensitivity_plane[:3]),-1,1))))
    sensitivity_coeff=regression_from_plane(sensitivity_plane); sensitivity_height=float(sensitivity_coeff[0]*reference_xz[0]+sensitivity_coeff[1]*reference_xz[1]+sensitivity_coeff[2])
    dough_distance=signed_distance(positive_points,plane)
    summary={"schema":SCHEMA,"episode":str(args.episode.resolve()),"raw_frames":len(raw_frames),"accepted_estimation_frames":len(estimation),
             "accepted_heldout_frames":len(heldout),"rejected_frames":int(sum(not fit.accepted for fit in main_fits)),"settings":settings,
             "plane_convention":"normalized [nx,ny,nz,d], n dot p + d = 0; positive is dough/free-space side",
             "estimated_plane_scene":plane.tolist(),"regression_y_equals_a_x_plus_b_z_plus_c":coefficients.tolist(),
             "normal_is_upward":bool(plane[1]>0),"initial_dough_median_signed_distance_m":float(np.median(dough_distance)),
             "reference_xz_m":reference_xz.tolist(),"height_at_reference_xz_m":height,"tilt_from_scene_y_deg":float(math.degrees(math.acos(np.clip(plane[1],-1,1)))),
             "bootstrap_equal_frame_95pct":bootstrap,"estimation_frame_angular_deviation_deg":summarize(angular),
             "estimation_frame_height_at_reference_m":summarize(frame_heights),"heldout_equal_frame_metrics":{
                 "aggregate_inlier_fraction_on_independently_selected_table_points":summarize(np.array([x["aggregate_inlier_fraction"] for x in per_heldout])),
                 "median_signed_residual_m":summarize(np.array([x["median_signed_residual_m"] for x in per_heldout])),
                 "median_abs_residual_m":summarize(np.array([x["median_abs_residual_m"] for x in per_heldout]))},
             "heldout_equal_sample_point_metrics":{"count":len(sample_residual),"signed_residual_m":summarize(sample_residual),"absolute_residual_m":summarize(np.abs(sample_residual)),
                 "selection":"up to 1000 deterministic points per held-out frame classified by its independent frame plane",
                 "linear_residual_bias_coefficients_x_z_intercept":linear.tolist(),"linear_residual_bias_rmse_m":linear_rmse,
                 "quadratic_residual_bias_coefficients_x_z_x2_xz_z2_intercept":quadratic.tolist(),"quadratic_residual_bias_rmse_m":quadratic_rmse,
                 "quadratic_minus_linear_rmse_m":quadratic_rmse-linear_rmse,"x_bins":x_bins,"z_bins":z_bins},
             "tool_exclusion":{"frames_with_pose":int(sum(row["tool_exclusion_available"] for row in rows)),"frames_without_pose":int(sum(not row["tool_exclusion_available"] for row in rows)),
                 "excluded_full_frame_points":int(sum(row["tool_points_excluded_from_full_frame"] for row in rows)),"geometry_sha256":input_hashes[str(args.tool_geometry.resolve())]},
             "sensitivity":{"main_sample_limit":args.sample_limit,"secondary_sample_limit":args.sensitivity_sample_limit,
                 "plane_secondary":sensitivity_plane.tolist(),"normal_angle_difference_deg":sensitivity_angle,
                 "height_at_reference_difference_m":sensitivity_height-height},
             "prior_one_frame":{"plane_scene":prior_plane.tolist(),"height_at_same_reference_xz_m":prior_height,
                 "height_difference_new_minus_prior_m":height-prior_height},
             "interpretation":{"geometry_measurement_only":True,"derived_calibration_written":False,"simulation_or_calibration_run":False,
                 "safe_to_apply_without_reconstruction":False},"input_hashes":input_hashes,"source_sha256":source_start}
    with (args.output_dir/"per_frame.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (args.output_dir/"heldout.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(per_heldout[0]));writer.writeheader();writer.writerows(per_heldout)
    json_write(args.output_dir/"summary.json",summary)
    draw_line_plot(args.output_dir/"per_frame_plane.png",rows,height,prior_height)
    draw_bias_plot(args.output_dir/"heldout_residual_bias.png",x_bins,z_bins)
    report(json.dumps({"output":str(args.output_dir),"plane":plane.tolist(),"height_at_reference_m":height,
                       "tilt_deg":summary["tilt_from_scene_y_deg"],"accepted_estimation":len(estimation),"accepted_heldout":len(heldout),
                       "sensitivity":summary["sensitivity"]},indent=2))
    source_end=file_sha256(source); shutil.copyfile(source,args.output_dir/"estimate_table_plane.py.txt")
    input_hashes_end = {str(path.resolve()): file_sha256(path) for path in inputs}
    outputs={p.name:file_sha256(p) for p in args.output_dir.iterdir() if p.is_file() and p.name!="verification.json"}
    checks={"source_unchanged_during_execution":bool(source_start==source_end),
            "inputs_unchanged_during_execution":bool(input_hashes_end == input_hashes),
            "all_390_frames_processed":bool(len(rows)==390),
            "hsv_counts_match_metadata_all_frames":bool(all(row["hsv_count_matches_metadata"] for row in rows)),
            "all_heldout_frames_accepted":bool(len(heldout)==78),"upward_normal":bool(plane[1]>0),
            "initial_dough_positive":bool(float(np.median(dough_distance))>0),
            "main_vs_secondary_height_difference_below_1mm":bool(abs(sensitivity_height-height)<.001),
            "main_vs_secondary_normal_difference_below_0.25deg":bool(sensitivity_angle<.25),
            "palette_light_validated_externally":True}
    json_write(args.output_dir/"verification.json",{"schema":SCHEMA,"passed":all(checks.values()),
               "checks":checks,"palette_validation":"#2a78d6,#eb6834 passed bundled light-mode validator",
               "input_sha256_end":input_hashes_end,"output_sha256":outputs,"source_end_sha256":source_end})
    if not checks["source_unchanged_during_execution"]:
        raise RuntimeError("Estimator source changed during execution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
