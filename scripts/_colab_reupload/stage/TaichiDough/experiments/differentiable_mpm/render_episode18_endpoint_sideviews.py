"""Render recorded dough, table, and registered tools at Episode18 endpoints."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from render_chunk01_mid_side import (
    euler_xyz_matrix,
    load_binary_stl,
    quaternion_xyzw_matrix,
)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DATA = ROOT / "data/episode18_table_aligned_v1"
REGISTERED = ROOT / "data/episode18_registered_tools_v1"
PREPARED = ROOT / "runs/episode18_rgb_topview_20260915T081318Z/chunks/chunk01/simulation/prepared_inputs.json"
OUTPUT = ROOT / "runs/episode18_endpoint_sideviews_20260915"


def render(points: np.ndarray, poses: np.ndarray, names: list[str], index: int,
           raw: int, elapsed: float, label: str, assets: list[dict], scale: float) -> Path:
    mesh_paths = [REGISTERED / "ur_spathla_collision_solid.stl",
                  REGISTERED / "gen3_spathla_collision_solid.stl"]
    colors = ["#e76f3c", "#19a979"]
    labels = ["UR5e tool", "Kinova tool"]
    triangles_world = []
    for asset, mesh_path, pose in zip(assets, mesh_paths, poses, strict=True):
        triangles = load_binary_stl(mesh_path) * scale
        triangles = triangles @ euler_xyz_matrix(asset["visual_rpy_rad"]).T
        triangles += np.asarray(asset["visual_origin_m"])
        triangles = triangles @ quaternion_xyzw_matrix(pose[3:]).T + pose[:3]
        triangles_world.append(triangles)

    points = points[np.isfinite(points).all(axis=1) & (points[:, 1] >= 0.003)]
    all_geometry = np.concatenate([points, *(mesh.reshape(-1, 3) for mesh in triangles_world)])
    low, high = all_geometry.min(axis=0), all_geometry.max(axis=0)
    low[1], high[1] = min(low[1], 0.0), max(high[1], 0.0)
    origin = np.array([0.5, 0.0, 0.5])
    points_mm = (points - origin) * 1000.0

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "#242522", "axes.labelcolor": "#242522",
                         "xtick.color": "#60615c", "ytick.color": "#60615c",
                         "axes.edgecolor": "#aaa9a2", "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for triangles, color, tool_label in zip(triangles_world, colors, labels, strict=True):
        mesh_mm = (triangles - origin) * 1000.0
        ax.add_collection(PolyCollection(mesh_mm[:, :, [2, 1]], facecolors=color,
                                         edgecolors=color, linewidths=0.12, alpha=0.24))
        center = mesh_mm.reshape(-1, 3).mean(axis=0)
        ax.annotate(tool_label, (center[2], center[1]), xytext=(12, 12),
                    textcoords="offset points", fontsize=10,
                    arrowprops={"arrowstyle": "-", "color": "#666", "lw": 0.8})
    ax.scatter(points_mm[:, 2], points_mm[:, 1], s=7, linewidths=0, alpha=0.72,
               color="#2878c8", rasterized=True, zorder=4)
    ax.axhline(0.0, color="#696b64", linewidth=1.4, zorder=3)
    ax.text((high[2] - origin[2]) * 1000 + 8, 2, "Table: Y = 0", ha="right",
            va="bottom", color="#60615c", fontsize=10)
    ax.set_xlim((low[2] - origin[2]) * 1000 - 15, (high[2] - origin[2]) * 1000 + 15)
    ax.set_ylim(-15, (high[1] - origin[1]) * 1000 + 15)
    ax.set_xlabel("Z offset (mm)")
    ax.set_ylabel("Height above table, Y (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#e6e7e1", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Episode18 {label}: recorded dough, registered tools, and table",
                 loc="left", fontsize=15, fontweight="bold", pad=16)
    ax.text(0.0, 1.015,
            f"Side view looking along +X · aligned frame {index} · original ordinal {raw} · t = {elapsed:.3f} s",
            transform=ax.transAxes, fontsize=10, color="#60615c")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#2878c8",
                      markersize=7, label="Recorded dough")]
    handles.extend(Patch(facecolor=color, alpha=0.55, label=tool_label)
                   for color, tool_label in zip(colors, labels, strict=True))
    ax.legend(handles=handles, loc="upper right", frameon=False)
    fig.tight_layout()
    target = OUTPUT / f"episode18_{label}_sideview.png"
    fig.savefig(target, dpi=200)
    plt.close(fig)
    return target


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prepared = json.loads(PREPARED.read_text())
    collision = prepared["provenance"]["collision"]
    with np.load(DATA / "observed_points_scene.npz", allow_pickle=False) as observed:
        times = observed["times"]
        original = observed["original_indices"]
        offsets = observed["offsets"]
        points_all = observed["points"]
        selected = [(0, "first"), (len(times) - 1, "last")]
        frames = [(index, label, points_all[offsets[index]:offsets[index + 1]].copy(),
                   int(original[index]), float(times[index] - times[0]))
                  for index, label in selected]
    with np.load(DATA / "tool_trajectories_scene.npz", allow_pickle=False) as tools:
        poses_all = tools["poses"]
        names = tools["names"].tolist()
    outputs = []
    for index, label, points, raw, elapsed in frames:
        target = render(points, poses_all[index], names, index, raw, elapsed, label,
                        collision["assets_in_stream_order"], float(collision["mesh_scale"]))
        outputs.append({"label": label, "aligned_frame": index, "original_ordinal": raw,
                        "elapsed_s": elapsed, "output": str(target.relative_to(REPO))})
        print(target)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "schema": "taichidough/episode18-endpoint-sideviews/v1",
        "view_direction": "+X", "projection": "orthographic Z-Y",
        "floor_y_m": 0.0, "geometry": "recorded visible dough points and registered collision meshes",
        "tool_names": names, "frames": outputs,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
