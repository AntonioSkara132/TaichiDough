#!/usr/bin/env python3
"""Plot coordinate histograms for a DeformPath point-cloud archive.

The tool accepts the raw ``pointclouds.pt`` layout and the interpolated
``pointclouds_interpolated.pt`` layout. It writes a PNG with a global
coordinate histogram and a per-frame low-quantile diagnostic, plus a JSON
summary containing ranges and quantiles useful for selecting a floor height.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


AXES = {"x": 0, "y": 1, "z": 2}


def load_torch_archive(path: Path):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Reading .pt point-cloud archives requires PyTorch") from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_frames(path: Path) -> list[np.ndarray]:
    """Load frames from a list-wrapped or tensor-based point-cloud archive."""
    archive = load_torch_archive(path)
    if isinstance(archive, dict):
        for key in ("pointclouds", "clouds", "frames"):
            if key in archive:
                archive = archive[key]
                break
        else:
            raise ValueError(f"Unsupported point-cloud dictionary keys in {path}: {sorted(archive)}")

    if isinstance(archive, (list, tuple)):
        frames = archive
        while len(frames) == 1 and isinstance(frames[0], (list, tuple)):
            frames = frames[0]
    else:
        array = _to_numpy(archive)
        if array.ndim != 3:
            raise ValueError(f"{path} must contain [T,N,F] data or a list of [N,F] frames; got {array.shape}")
        frames = [array[index] for index in range(array.shape[0])]

    result = []
    for index, frame in enumerate(frames):
        array = _to_numpy(frame)
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError(f"Frame {index} in {path} must have shape [N,>=3], got {array.shape}")
        xyz = np.asarray(array[:, :3], dtype=np.float64)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if len(xyz):
            result.append(xyz)
        else:
            result.append(np.empty((0, 3), dtype=np.float64))
    if not result:
        raise ValueError(f"{path} contains no frames")
    return result


def select_frames(frames: list[np.ndarray], requested: int | None, max_points: int) -> list[np.ndarray]:
    if requested is not None:
        if not 0 <= requested < len(frames):
            raise ValueError(f"--frame must be between 0 and {len(frames) - 1}")
        selected = [(requested, frames[requested])]
    else:
        selected = list(enumerate(frames))

    result = []
    for frame_index, points in selected:
        if max_points and len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
            points = points[indices]
        result.append((frame_index, points))
    return result


def finite_quantiles(values: np.ndarray) -> dict[str, float]:
    quantile_values = np.quantile(values, [0.001, 0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995, 0.999])
    names = ("q0.1%", "q0.5%", "q1%", "q5%", "q50%", "q95%", "q99%", "q99.5%", "q99.9%")
    return {name: float(value) for name, value in zip(names, quantile_values)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="pointclouds.pt or pointclouds_interpolated.pt")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--summary", type=Path, help="Optional JSON summary path; defaults beside the PNG")
    parser.add_argument("--axis", choices=tuple(AXES), default="y", help="Coordinate to analyze (default: y)")
    parser.add_argument("--bins", type=int, default=100, help="Number of global histogram bins")
    parser.add_argument("--frame", type=int, help="Analyze one zero-based frame instead of the complete archive")
    parser.add_argument("--value-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="Optional plotting range; summary still reports the full finite data")
    parser.add_argument("--max-points-per-frame", type=int, default=0,
                        help="Uniformly subsample each frame; 0 keeps every finite point")
    parser.add_argument("--low-quantile", type=float, default=0.01,
                        help="Per-frame lower quantile shown in the second panel")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")
    if args.max_points_per_frame < 0:
        raise ValueError("--max-points-per-frame cannot be negative")
    if not 0.0 <= args.low_quantile <= 1.0:
        raise ValueError("--low-quantile must be between 0 and 1")
    if args.value_range is not None:
        if not np.isfinite(args.value_range).all() or args.value_range[0] >= args.value_range[1]:
            raise ValueError("--value-range must contain finite MIN < MAX")

    frames = load_frames(args.input)
    selected = select_frames(frames, args.frame, args.max_points_per_frame)
    axis_index = AXES[args.axis]
    values_by_frame = [(index, points[:, axis_index]) for index, points in selected if len(points)]
    if not values_by_frame:
        raise ValueError("Selected frames contain no finite XYZ points")
    values = np.concatenate([values for _, values in values_by_frame])
    low_values = np.asarray([np.quantile(frame_values, args.low_quantile) for _, frame_values in values_by_frame])
    plot_values = values
    if args.value_range is not None:
        plot_values = values[(values >= args.value_range[0]) & (values <= args.value_range[1])]
        if not len(plot_values):
            raise ValueError("--value-range excludes every finite point")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or args.output.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "taichidough/pointcloud-coordinate-histogram/v1",
        "input": str(args.input.resolve()),
        "axis": args.axis,
        "axis_index": axis_index,
        "source_frame_selection": args.frame,
        "source_frame_count": len(frames),
        "selected_frame_count": len(values_by_frame),
        "finite_point_count": int(len(values)),
        "max_points_per_frame": args.max_points_per_frame,
        "global_min": float(values.min()),
        "global_max": float(values.max()),
        "global_mean": float(values.mean()),
        "global_std": float(values.std()),
        "global_quantiles": finite_quantiles(values),
        "plot_point_count": int(len(plot_values)),
        "plot_value_range": None if args.value_range is None else [float(v) for v in args.value_range],
        "per_frame_low_quantile": {
            "quantile": args.low_quantile,
            "min": float(low_values.min()),
            "max": float(low_values.max()),
            "median": float(np.median(low_values)),
            "values": [
                {"frame": int(index), "value": float(value)}
                for (index, _), value in zip(values_by_frame, low_values)
            ],
        },
    }

    # Use Pillow instead of Matplotlib so the utility works with the existing
    # environment. The system Matplotlib installation may be ABI-incompatible
    # with the NumPy version used by the simulator.
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1800, 760), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    title = f"Point-cloud coordinate distribution: {args.input.name}"
    draw.text((35, 18), title, fill="#222222", font=font)

    def draw_histogram(box, data, bin_count, color, panel_title, x_label, y_label, marker, marker_label):
        left, top, right, bottom = box
        plot_left, plot_top = left + 75, top + 48
        plot_right, plot_bottom = right - 22, bottom - 55
        counts, edges = np.histogram(data, bins=bin_count)
        maximum = max(int(counts.max()), 1)
        value_min, value_max = float(edges[0]), float(edges[-1])
        if value_max <= value_min:
            value_max = value_min + 1.0
        draw.text((left, top + 8), panel_title, fill="#222222", font=font)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#333333", width=2)
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#333333", width=2)
        for index, count in enumerate(counts):
            x0 = plot_left + (plot_right - plot_left) * index / len(counts)
            x1 = plot_left + (plot_right - plot_left) * (index + 1) / len(counts)
            y1 = plot_bottom - (plot_bottom - plot_top) * float(count) / maximum
            if count > 0:
                draw.rectangle((int(x0) + 1, int(y1), int(x1) - 1, plot_bottom - 1), fill=color)
        for fraction in (0.0, 0.5, 1.0):
            x = plot_left + (plot_right - plot_left) * fraction
            value = value_min + (value_max - value_min) * fraction
            draw.line((int(x), plot_bottom, int(x), plot_bottom + 5), fill="#333333", width=1)
            draw.text((int(x) - 20, plot_bottom + 10), f"{value:.4g}", fill="#333333", font=font)
        for fraction in (0.0, 0.5, 1.0):
            y = plot_bottom - (plot_bottom - plot_top) * fraction
            value = maximum * fraction
            draw.line((plot_left - 5, int(y), plot_left, int(y)), fill="#333333", width=1)
            draw.text((plot_left - 65, int(y) - 6), f"{value:.3g}", fill="#333333", font=font)
        marker_x = plot_left + (plot_right - plot_left) * (float(marker) - value_min) / (value_max - value_min)
        marker_x = int(np.clip(marker_x, plot_left, plot_right))
        draw.line((marker_x, plot_top, marker_x, plot_bottom), fill="#bb5566", width=2)
        draw.text((max(plot_left, marker_x - 60), plot_top + 8), marker_label, fill="#bb5566", font=font)
        draw.text(((plot_left + plot_right) // 2 - 30, bottom - 32), x_label, fill="#333333", font=font)
        draw.text((left + 5, (plot_top + plot_bottom) // 2), y_label, fill="#333333", font=font)

    draw_histogram(
        (35, 65, 880, 730), plot_values, args.bins, "#4477aa",
        f"All finite point {args.axis}-coordinates", f"{args.axis} [m]", "Point count",
        summary["global_quantiles"]["q1%"], f"q1%={summary['global_quantiles']['q1%']:.4g} m",
    )
    draw_histogram(
        (920, 65, 1765, 730), low_values, min(args.bins, 50), "#ddaa44",
        f"Per-frame q{args.low_quantile:g} {args.axis} value", f"{args.axis} [m]", "Frame count",
        float(np.median(low_values)), f"median={np.median(low_values):.4g} m",
    )
    canvas.save(args.output)

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote histogram: {args.output}")
    print(f"Wrote summary: {summary_path}")
    print(f"Frames={len(values_by_frame)} points={len(values)} range={values.min():.9g} to {values.max():.9g} m")
    print("Global quantiles:", json.dumps(summary["global_quantiles"], sort_keys=True))
    print(f"Per-frame q{args.low_quantile:g} median={np.median(low_values):.9g} m")


if __name__ == "__main__":
    main()
