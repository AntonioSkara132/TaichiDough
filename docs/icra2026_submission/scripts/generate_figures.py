#!/usr/bin/env python3
"""Generate publication figures from existing TaichiDough run artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import cast

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "figures"
BLUE = "#1769AA"
ORANGE = "#D55E00"
INK = "#202428"
MUTED = "#5B6268"
GRID = "#D9DEE2"
LIGHT_BLUE = "#E8F2FA"
LIGHT_ORANGE = "#FBEDE5"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_manifest(name: str, payload: dict) -> None:
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def axes_grid(value: object, rows: int, columns: int) -> list[list[Axes]]:
    array = np.asarray(value, dtype=object).reshape(rows, columns)
    return [[cast(Axes, array[row, column]) for column in range(columns)]
            for row in range(rows)]


def parameter_history() -> None:
    run = REPO / "experiments/differentiable_mpm/runs/dataset_fit_20260913T195731_c5aeaabd"
    history = REPO / "experiments/differentiable_mpm/runs/dataset_fit_20260913T195731_c5aeaabd_visualizations_prancer/dataset_fit_20260913T195731_c5aeaabd/retained_states.csv"
    manifest_path = run / "run_manifest.json"
    result_path = run / "result.json"
    with history.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    manifest = json.loads(manifest_path.read_text())
    result = json.loads(result_path.read_text())
    iterations = np.array([int(row["iteration"]) for row in rows])
    bounds = manifest["identity"]["dataset_identity"]["dataset"]["shared_parameters"]["bounds"]

    panels = [
        ("loss", "Training objective", None, None),
        ("youngs_modulus", r"Young's modulus $E$ (kPa)", 1e-3, bounds["youngs_modulus"]),
        ("viscosity", r"Viscosity $\eta$ (Pa s)", 1.0, bounds["viscosity"]),
        ("poisson_ratio", r"Poisson ratio $\nu$", 1.0, bounds["poisson_ratio"]),
        ("plastic_min", r"Lower stretch $\sigma_{\min}$", 1.0, bounds["plastic_min"]),
        ("plastic_max", r"Upper stretch $\sigma_{\max}$", 1.0, bounds["plastic_max"]),
    ]
    fig, axes_raw = plt.subplots(2, 3, figsize=(7.05, 4.15), sharex=True, layout="constrained")
    axes = axes_grid(axes_raw, 2, 3)
    for axis, (key, label, scale, limit) in zip(
            [axis for row in axes for axis in row], panels, strict=True):
        values = np.array([float(row[key]) for row in rows])
        if scale is not None:
            values *= scale
        axis.plot(iterations, values, color=BLUE, linewidth=1.7, marker="o", markersize=3.2,
                  markerfacecolor="white", markeredgewidth=0.9)
        axis.scatter(iterations[-1], values[-1], s=42, facecolor="white", edgecolor=ORANGE,
                     linewidth=1.8, zorder=5)
        if limit is not None:
            low, high = np.array(limit, dtype=float) * scale
            data_span = max(float(np.ptp(values)), 1e-9)
            bound_span = high - low
            if data_span / bound_span >= 0.18:
                axis.axhspan(low, high, color=LIGHT_BLUE, alpha=0.28, zorder=-3)
                axis.axhline(low, color=MUTED, linewidth=0.55, linestyle=":")
                axis.axhline(high, color=MUTED, linewidth=0.55, linestyle=":")
            else:
                padding = data_span * 0.22
                axis.set_ylim(values.min() - padding, values.max() + padding)
                axis.text(0.03, 0.94, f"bounds {low:g}–{high:g}", transform=axis.transAxes,
                          va="top", fontsize=6.2, color=MUTED)
        axis.set_title(label, fontsize=8.3, color=INK, pad=3)
        axis.grid(axis="y", color=GRID, linewidth=0.55)
        axis.tick_params(labelsize=7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(iterations[0] - 0.3, iterations[-1] + 0.6)
    for axis in axes[-1]:
        axis.set_xlabel("Iteration", fontsize=8)
    axes[0][0].annotate("best saved", (iterations[-1], float(rows[-1]["loss"])),
                        xytext=(-30, 12), textcoords="offset points", fontsize=6.8,
                        arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.8})
    fig.suptitle("Interrupted five-parameter fit: accepted-state history", fontsize=10.2,
                 fontweight="semibold", color=INK)
    save(fig, "parameter_history")
    write_manifest("parameter_history", {
        "schema": "taichidough/icra-figure-provenance/v1",
        "figure": "parameter_history",
        "sources": [
            {"path": str(history.relative_to(REPO)), "sha256": digest(history)},
            {"path": str(manifest_path.relative_to(REPO)), "sha256": digest(manifest_path)},
            {"path": str(result_path.relative_to(REPO)), "sha256": digest(result_path)},
        ],
        "accepted_states": len(rows),
        "accepted_updates": int(result["accepted_updates"]),
        "terminal_status": result["status"],
        "terminal_error_type": result["error_type"],
        "best_loss": float(rows[-1]["loss"]),
        "display": "No smoothing or omitted accepted states; modulus converted from Pa to kPa.",
    })


def matched_sequence() -> None:
    root = REPO / "experiments/differentiable_mpm/runs/episode18-table-aligned-registered-tools_fit_20260912T222338_9ab2d0/validation_e2a82ee1_strict/evaluation"
    metrics_path = root / "dynamic_topview_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    frames = [0, 30, 60, 75, 97]
    records = {int(item["source_frame"]): item for item in metrics["frames"]}
    arrays = []
    rgb_images = []
    rgb_manifest = OUT / "recorded_rgb_frames.json"
    source_files = [metrics_path, rgb_manifest]
    for frame in frames:
        path = root / "arrays" / f"frame_{frame:06d}.npz"
        rgb_path = OUT / f"recorded_rgb_frame_{frame:06d}.png"
        source_files.extend((path, rgb_path))
        with np.load(path) as data:
            arrays.append({key: data[key].copy() for key in data.files})
        rgb_images.append(np.asarray(Image.open(rgb_path).convert("RGB")))

    union = np.zeros_like(arrays[0]["real_valid"], dtype=bool)
    depth_values = []
    for data in arrays:
        union |= data["real_valid"] | data["sim_valid"]
        depth_values.extend([data["real_depth"][data["real_valid"]], data["sim_depth"][data["sim_valid"]]])
    yy, xx = np.nonzero(union)
    margin = 18
    y0, y1 = max(0, yy.min() - margin), min(union.shape[0], yy.max() + margin + 1)
    x0, x1 = max(0, xx.min() - margin), min(union.shape[1], xx.max() + margin + 1)
    finite_depth = np.concatenate(depth_values)
    vmin, vmax = np.quantile(finite_depth, [0.01, 0.99])
    cmap = mpl.colormaps["cividis"].copy()
    cmap.set_bad("white")

    fig, axes_raw = plt.subplots(2, len(frames), figsize=(7.05, 2.95), layout="constrained")
    axes = axes_grid(axes_raw, 2, len(frames))
    image = None
    for column, (frame, data, rgb) in enumerate(zip(frames, arrays, rgb_images, strict=True)):
        record = records[frame]
        phase = "initial" if frame == 0 else ("fit" if frame <= 60 else "continuation")
        axes[0][column].imshow(rgb[y0:y1, x0:x1], interpolation="nearest")
        simulated = np.ma.masked_where(
            ~data["sim_valid"][y0:y1, x0:x1], data["sim_depth"][y0:y1, x0:x1])
        image = axes[1][column].imshow(simulated, cmap=cmap, vmin=vmin, vmax=vmax,
                                      interpolation="nearest")
        for row in range(2):
            axis = axes[row][column]
            axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in axis.spines.values():
                spine.set_linewidth(0.65)
                spine.set_edgecolor(GRID)
        color = MUTED if phase == "initial" else (BLUE if phase == "fit" else ORANGE)
        axes[0][column].set_title(f"{record['time_s']:.2f} s\n{phase}", fontsize=7.3,
                                  color=color, fontweight="semibold")
    axes[0][0].set_ylabel("Recorded\nRGB", fontsize=8.2, fontweight="semibold")
    axes[1][0].set_ylabel("Simulated\ndepth", fontsize=8.2, fontweight="semibold")
    assert image is not None
    fig.suptitle("Camera-matched Episode 18 observation and simulation", fontsize=10.2,
                 fontweight="semibold", color=INK)
    save(fig, "matched_sequence")
    write_manifest("matched_sequence", {
        "schema": "taichidough/icra-figure-provenance/v1",
        "figure": "matched_sequence",
        "sources": [{"path": str(path.relative_to(REPO)), "sha256": digest(path)} for path in source_files],
        "frames": [{"source_frame": frame, "time_s": float(records[frame]["time_s"]),
                    "phase": "initial" if frame == 0 else ("fit" if frame <= 60 else "temporal_continuation")}
                   for frame in frames],
        "camera_calibration_fingerprint": metrics["calibration"]["fingerprint"],
        "crop_pixels_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "depth_display_quantiles_m": [float(vmin), float(vmax)],
        "limitations": [
            "Top row uses exact-timestamp RGB frames extracted from the original Rosbag.",
            "Simulated depth contains dough only; simulated tool occlusion is not modeled.",
            "Frames 61-97 are temporal continuation in the same episode, not independent-motion validation.",
        ],
    })


def pipeline_overview() -> None:
    fig, axis_raw = plt.subplots(figsize=(7.05, 2.15))
    axis = cast(Axes, axis_raw)
    axis.set_xlim(0, 13.0)
    axis.set_ylim(0, 3.7)
    axis.axis("off")
    stages = [
        (0.10, 2.0, "RGB-D\nobservation", "depth + intrinsics"),
        (2.50, 2.0, "Metric\nreconstruction", "tag, floor, volume"),
        (4.90, 2.0, "Tool replay\n+ MLS-MPM", "two SDF spatulas"),
        (7.30, 2.0, "Partial-view loss\n+ adjoint", "depth + coverage"),
        (9.70, 2.0, "Parameter\noptimization", r"$E,\nu,\eta,\sigma_{min},\sigma_{max}$"),
    ]
    width, height = 2.08, 1.15
    for index, (x, y, title, subtitle) in enumerate(stages):
        fill = LIGHT_BLUE if index < 3 else LIGHT_ORANGE
        edge = BLUE if index < 3 else ORANGE
        box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.08,rounding_size=0.08",
                             linewidth=1.15, edgecolor=edge, facecolor=fill)
        axis.add_patch(box)
        axis.text(x + width / 2, y + 0.72, title, ha="center", va="center", fontsize=6.7,
                  fontweight="semibold", color=INK)
        axis.text(x + width / 2, y + 0.24, subtitle, ha="center", va="center", fontsize=5.6,
                  color=MUTED)
        if index < len(stages) - 1:
            axis.add_patch(FancyArrowPatch((x + width + 0.07, y + height / 2),
                                           (stages[index + 1][0] - 0.08, y + height / 2),
                                           arrowstyle="-|>", mutation_scale=10, linewidth=1.0,
                                           color=MUTED))
    axis.add_patch(FancyArrowPatch((10.32, 1.93), (10.32, 1.22), arrowstyle="-|>",
                                   mutation_scale=10, linewidth=1.0, color=MUTED))
    future = FancyBboxPatch((7.75, 0.23), 5.05, 0.82, boxstyle="round,pad=0.07,rounding_size=0.07",
                            linewidth=1.1, linestyle="--", edgecolor=MUTED, facecolor="#F5F6F7")
    axis.add_patch(future)
    axis.text(10.275, 0.64, "Future: synthetic trajectories for\nbimanual shaping policies",
              ha="center", va="center", fontsize=7.0, fontweight="semibold", color=INK)
    axis.text(0.15, 3.48, "Completed identification pipeline", fontsize=8.4,
              fontweight="semibold", color=INK)
    save(fig, "pipeline_overview")
    write_manifest("pipeline_overview", {
        "schema": "taichidough/icra-figure-provenance/v1",
        "figure": "pipeline_overview",
        "kind": "schematic",
        "completed_stages": ["single-camera capture", "metric calibration and reconstruction",
                             "recorded two-tool MLS-MPM replay", "partial-view differentiation",
                             "bounded parameter optimization"],
        "future_stage": "bimanual robot policy learning and execution",
        "note": "The dashed future block denotes planned work, not a completed experiment.",
    })


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    parameter_history()
    matched_sequence()
    pipeline_overview()
    print(f"Wrote figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
