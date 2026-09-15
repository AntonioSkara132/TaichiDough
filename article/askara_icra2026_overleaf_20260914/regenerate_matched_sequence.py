#!/usr/bin/env python3
"""Regenerate the local matched-sequence figure without an internal title."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from PIL import Image

REPO = Path("/home/antonio/diplomski_antonio/diplomski/TaichiDough")
ARTICLE = REPO / "article/askara_icra2026_overleaf_20260914"
OUT = ARTICLE / "source_revised/figures"
RUN = REPO / "experiments/differentiable_mpm/runs/episode18-table-aligned-registered-tools_fit_20260912T222338_9ab2d0/validation_e2a82ee1_strict/evaluation"
BLUE = "#1769AA"
ORANGE = "#D55E00"
MUTED = "#5B6268"
GRID = "#D9DEE2"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def axes_grid(value: object, rows: int, columns: int) -> list[list[Axes]]:
    array = np.asarray(value, dtype=object).reshape(rows, columns)
    return [
        [cast(Axes, array[row, column]) for column in range(columns)]
        for row in range(rows)
    ]


def main() -> int:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    metrics_path = RUN / "dynamic_topview_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    frames = [0, 30, 60, 75, 97]
    records = {int(item["source_frame"]): item for item in metrics["frames"]}
    arrays: list[dict[str, np.ndarray]] = []
    rgb_images: list[np.ndarray] = []
    rgb_manifest = OUT / "recorded_rgb_frames.json"
    source_files = [metrics_path, rgb_manifest]

    for frame in frames:
        path = RUN / "arrays" / f"frame_{frame:06d}.npz"
        rgb_path = OUT / f"recorded_rgb_frame_{frame:06d}.png"
        source_files.extend((path, rgb_path))
        with np.load(path) as data:
            arrays.append({key: data[key].copy() for key in data.files})
        rgb_images.append(np.asarray(Image.open(rgb_path).convert("RGB")))

    union = np.zeros_like(arrays[0]["real_valid"], dtype=bool)
    depth_values: list[np.ndarray] = []
    for data in arrays:
        union |= data["real_valid"] | data["sim_valid"]
        depth_values.extend(
            [
                data["real_depth"][data["real_valid"]],
                data["sim_depth"][data["sim_valid"]],
            ]
        )
    yy, xx = np.nonzero(union)
    margin = 18
    y0, y1 = max(0, yy.min() - margin), min(union.shape[0], yy.max() + margin + 1)
    x0, x1 = max(0, xx.min() - margin), min(union.shape[1], xx.max() + margin + 1)
    finite_depth = np.concatenate(depth_values)
    vmin, vmax = np.quantile(finite_depth, [0.01, 0.99])
    cmap = mpl.colormaps["cividis"].copy()
    cmap.set_bad("white")

    fig, axes_raw = plt.subplots(
        2, len(frames), figsize=(7.05, 2.72), layout="constrained"
    )
    axes = axes_grid(axes_raw, 2, len(frames))
    for column, (frame, data, rgb) in enumerate(
        zip(frames, arrays, rgb_images, strict=True)
    ):
        record = records[frame]
        phase = "initial" if frame == 0 else ("fit" if frame <= 60 else "continuation")
        axes[0][column].imshow(rgb[y0:y1, x0:x1], interpolation="nearest")
        simulated = np.ma.masked_where(
            ~data["sim_valid"][y0:y1, x0:x1],
            data["sim_depth"][y0:y1, x0:x1],
        )
        axes[1][column].imshow(
            simulated,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        for row in range(2):
            axis = axes[row][column]
            axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in axis.spines.values():
                spine.set_linewidth(0.65)
                spine.set_edgecolor(GRID)
        color = MUTED if phase == "initial" else (BLUE if phase == "fit" else ORANGE)
        axes[0][column].set_title(
            f"{record['time_s']:.2f} s\n{phase}",
            fontsize=7.3,
            color=color,
            fontweight="semibold",
        )
    axes[0][0].set_ylabel("Recorded\nRGB", fontsize=8.2, fontweight="semibold")
    axes[1][0].set_ylabel("Simulated\ndepth", fontsize=8.2, fontweight="semibold")

    fig.savefig(OUT / "matched_sequence.pdf", bbox_inches="tight")
    fig.savefig(
        OUT / "matched_sequence.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    manifest = {
        "schema": "taichidough/icra-figure-provenance/v1",
        "figure": "matched_sequence",
        "sources": [
            {"path": str(path.relative_to(REPO)), "sha256": digest(path)}
            for path in source_files
        ],
        "frames": [
            {
                "source_frame": frame,
                "time_s": float(records[frame]["time_s"]),
                "phase": "initial"
                if frame == 0
                else ("fit" if frame <= 60 else "temporal_continuation"),
            }
            for frame in frames
        ],
        "camera_calibration_fingerprint": metrics["calibration"]["fingerprint"],
        "crop_pixels_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "depth_display_quantiles_m": [float(vmin), float(vmax)],
        "limitations": [
            "Top row uses exact-timestamp RGB frames extracted from the original ROS 2 bag.",
            "Simulated depth contains dough only; simulated tool occlusion is not modeled.",
            "Frames 61-97 are temporal continuation in the same episode, not independent-motion validation.",
        ],
        "display": "No internal figure title; panel timestamps and phase labels retained.",
    }
    (OUT / "matched_sequence.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
