#!/usr/bin/env python3
"""Render narrow raw-depth cross-sections against an explicit calibrated floor."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True,
                        help="DeformPath3 episode directory containing pointclouds.pt")
    parser.add_argument("--calibration", type=Path, required=True,
                        help="Scene calibration used by the fit being inspected")
    parser.add_argument("--frame", type=int, default=30,
                        help="Raw point-cloud frame index")
    parser.add_argument("--slice-center-x", type=float, default=0.5,
                        help="Scene X coordinate of the Z/Y cross-section")
    parser.add_argument("--slice-center-z", type=float, default=0.5,
                        help="Scene Z coordinate of the X/Y cross-section")
    parser.add_argument("--slice-width", type=float, default=0.01,
                        help="Full cross-section slab width in metres")
    parser.add_argument("--y-range", type=float, nargs=2, default=(-0.03, 0.20),
                        metavar=("MIN", "MAX"), help="Displayed scene-Y interval in metres")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def floor_height(plane: np.ndarray, horizontal: np.ndarray, fixed: float,
                 horizontal_axis: int, fixed_axis: int) -> np.ndarray:
    if abs(plane[1]) < 1e-12:
        raise ValueError("Calibrated floor plane cannot be plotted as scene Y")
    return -(plane[horizontal_axis] * horizontal + plane[fixed_axis] * fixed + plane[3]) / plane[1]


def main() -> int:
    args = parse_args()
    episode = args.episode.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not math.isfinite(args.slice_width) or args.slice_width <= 0:
        raise ValueError("slice-width must be positive and finite")
    if not all(math.isfinite(value) for value in
               (args.slice_center_x, args.slice_center_z, *args.y_range)):
        raise ValueError("slice centres and y-range must be finite")
    if args.y_range[0] >= args.y_range[1]:
        raise ValueError("y-range requires MIN < MAX")

    metadata_path = episode / "conversion_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    output_frame = metadata.get("output_frame")
    source_frame = calibration.get("source_frame")
    if output_frame != source_frame:
        raise ValueError(
            f"Recorded point coordinates are in {output_frame!r}, but calibration "
            f"scene_from_source expects {source_frame!r}")

    transform = np.asarray(calibration["scene_from_source"], dtype=np.float64)
    floor = np.asarray(calibration["floor_plane_scene"], dtype=np.float64)
    if transform.shape != (4, 4) or floor.shape != (4,):
        raise ValueError("Expected scene_from_source [4,4] and floor_plane_scene [4]")
    if not np.isfinite(transform).all() or not np.isfinite(floor).all():
        raise ValueError("Calibration transform and floor plane must be finite")

    pointcloud_path = episode / "pointclouds.pt"
    clouds = torch.load(pointcloud_path, map_location="cpu", weights_only=False)
    if not isinstance(clouds, list) or len(clouds) != 1 or not isinstance(clouds[0], list):
        raise ValueError("Expected pointclouds.pt to contain one raw point-cloud sequence")
    frames = clouds[0]
    if not 0 <= args.frame < len(frames):
        raise ValueError(f"frame must be in [0, {len(frames) - 1}]")
    raw = frames[args.frame].detach().cpu().numpy()
    if raw.ndim != 2 or raw.shape[1] < 10:
        raise ValueError("Expected raw point-cloud columns XYZ...RGB")

    scene_xyz = raw[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    rgb = np.clip(raw[:, 7:10] / 255.0, 0.0, 1.0)
    half_width = args.slice_width / 2
    sections = (
        {
            "horizontal": 0, "fixed": 2, "center": args.slice_center_z,
            "label": "X", "fixed_label": "Z",
            "mask": np.abs(scene_xyz[:, 2] - args.slice_center_z) <= half_width,
        },
        {
            "horizontal": 2, "fixed": 0, "center": args.slice_center_x,
            "label": "Z", "fixed_label": "X",
            "mask": np.abs(scene_xyz[:, 0] - args.slice_center_x) <= half_width,
        },
    )
    if any(not np.any(section["mask"]) for section in sections):
        raise ValueError("At least one cross-section contains no points; adjust its centre or width")

    figure, axes = plt.subplots(1, 2, figsize=(15, 7.5), sharey=True, layout="constrained")
    section_records = []
    for axis, section in zip(axes, sections, strict=True):
        mask = section["mask"]
        points = scene_xyz[mask]
        colors = rgb[mask]
        horizontal = section["horizontal"]
        order = np.argsort(points[:, 1])
        points = points[order]
        colors = colors[order]
        axis.scatter(points[:, horizontal], points[:, 1], s=1.0, c=colors,
                     marker=".", linewidths=0, alpha=0.8, rasterized=True)
        low, high = np.quantile(points[:, horizontal], (0.002, 0.998))
        line_x = np.linspace(low, high, 256)
        line_y = floor_height(floor, line_x, section["center"], horizontal, section["fixed"])
        axis.plot(line_x, line_y, color="#e69f00", linewidth=2.0, linestyle="--",
                  label="calibrated floor")
        axis.set_xlim(low, high)
        axis.set_ylim(*args.y_range)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(f"scene {section['label']} (m)")
        axis.set_title(
            f"{section['fixed_label']} = {section['center']:.3f} m ± {half_width * 1000:.1f} mm\n"
            f"{len(points):,} of {len(raw):,} raw points",
            fontsize=11)
        axis.grid(color="#d8d8d8", linewidth=0.5)
        axis.legend(loc="upper right", framealpha=0.95)
        section_records.append({
            "horizontal_axis": section["label"],
            "fixed_axis": section["fixed_label"],
            "fixed_center_m": section["center"],
            "point_count": int(len(points)),
            "horizontal_range_m": [float(low), float(high)],
            "y_range_m": [float(points[:, 1].min()), float(points[:, 1].max())],
        })
    axes[0].set_ylabel("scene Y (m)")
    figure.suptitle(
        f"Episode 18 raw-depth cross-sections — frame {args.frame} — calibrated table frame",
        fontsize=14, fontweight="bold")
    figure.text(
        0.5, 0.015,
        f"Raw XYZ frame: {output_frame}; transformed once with scene_from_source. "
        f"Slab width: {args.slice_width * 1000:.1f} mm. No segmentation or floor removal.",
        ha="center", fontsize=9)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, facecolor="white")
    plt.close(figure)
    manifest_path = output.with_suffix(".json")
    manifest = {
        "schema": "taichidough/raw-depth-floor-cross-section/v1",
        "episode": str(episode),
        "frame": args.frame,
        "pointclouds": str(pointcloud_path),
        "pointclouds_sha256": sha256(pointcloud_path),
        "conversion_metadata": str(metadata_path),
        "conversion_metadata_sha256": sha256(metadata_path),
        "calibration": str(calibration_path),
        "calibration_sha256": sha256(calibration_path),
        "input_coordinate_frame": output_frame,
        "calibration_source_frame": source_frame,
        "calibration_scene_frame": calibration.get("scene_frame"),
        "transform": "scene_from_source",
        "floor_plane_scene": floor.tolist(),
        "raw_point_count": int(len(raw)),
        "slice_width_m": args.slice_width,
        "display_y_range_m": list(args.y_range),
        "sections": section_records,
        "output": str(output),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"RAW_POINTS={len(raw)}")
    for record in section_records:
        print(f"{record['fixed_axis']}_SLICE_POINTS={record['point_count']}")
    print(f"OUTPUT={output}")
    print(f"MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
