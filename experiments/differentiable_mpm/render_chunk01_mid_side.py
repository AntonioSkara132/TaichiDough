"""Render the simulated dough, floor, and registered tools at chunk01 midpoint."""
from __future__ import annotations

import json
from pathlib import Path
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RUN = ROOT / "runs/episode18_rgb_topview_20260915T081318Z/chunks/chunk01/simulation"
OUTPUT = ROOT / "runs/episode18_rgb_topview_20260915T081318Z/sideview/chunk01_sideview.png"
FRAME = 9


def load_binary_stl(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    count = struct.unpack_from("<I", payload, 80)[0]
    expected = 84 + 50 * count
    if len(payload) != expected:
        raise ValueError(f"Expected binary STL, got {len(payload)} bytes instead of {expected}: {path}")
    records = np.frombuffer(payload, dtype=np.dtype([
        ("normal", "<f4", (3,)), ("vertices", "<f4", (9,)), ("attribute", "<u2")
    ]), offset=84, count=count)
    return records["vertices"].reshape(-1, 3, 3).astype(np.float64)


def euler_xyz_matrix(angles: list[float]) -> np.ndarray:
    x, y, z = angles
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def quaternion_xyzw_matrix(quaternion: list[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm([x, y, z, w])
    x, y, z, w = np.array([x, y, z, w]) / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads((RUN / "simulation_result.json").read_text())
    prepared = json.loads((RUN / "prepared_inputs.json").read_text())
    record = next(item for item in result["frames"] if item["source_frame"] == FRAME)
    points = np.load(RUN / record["particles"]).astype(np.float64)
    assets = prepared["provenance"]["collision"]["assets_in_stream_order"]
    mesh_scale = float(prepared["provenance"]["collision"]["mesh_scale"])
    local_meshes = [
        ROOT / "data/episode18_registered_tools_v1/ur_spathla_collision_solid.stl",
        ROOT / "data/episode18_registered_tools_v1/gen3_spathla_collision_solid.stl",
    ]
    colors = ["#e76f3c", "#19a979"]
    labels = ["UR5e tool", "Kinova tool"]
    triangles_world = []
    for asset, mesh_path, pose in zip(assets, local_meshes, record["tool_poses"], strict=True):
        triangles = load_binary_stl(mesh_path) * mesh_scale
        visual_rotation = euler_xyz_matrix(asset["visual_rpy_rad"])
        triangles = triangles @ visual_rotation.T + np.asarray(asset["visual_origin_m"])
        pose_rotation = quaternion_xyzw_matrix(pose[3:])
        triangles = triangles @ pose_rotation.T + np.asarray(pose[:3])
        triangles_world.append(triangles)

    all_geometry = np.concatenate([points, *(mesh.reshape(-1, 3) for mesh in triangles_world)])
    low, high = all_geometry.min(axis=0), all_geometry.max(axis=0)
    low[1], high[1] = min(low[1], 0.0), max(high[1], 0.0)
    origin = np.array([0.5, 0.0, 0.5])
    points_mm = (points - origin) * 1000.0

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": "#242522",
        "axes.labelcolor": "#242522", "xtick.color": "#60615c", "ytick.color": "#60615c",
        "axes.edgecolor": "#aaa9a2", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
    })
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for triangles, color, label in zip(triangles_world, colors, labels, strict=True):
        mesh_mm = (triangles - origin) * 1000.0
        ax.add_collection(PolyCollection(mesh_mm[:, :, [2, 1]], facecolors=color,
                                         edgecolors=color, linewidths=0.12, alpha=0.24))
        center = mesh_mm.reshape(-1, 3).mean(axis=0)
        ax.annotate(label, (center[2], center[1]), xytext=(12, 12), textcoords="offset points",
                    fontsize=10, arrowprops={"arrowstyle": "-", "color": "#666", "lw": 0.8})
    ax.scatter(points_mm[:, 2], points_mm[:, 1], s=4, linewidths=0, alpha=0.65,
               color="#2878c8", rasterized=True, zorder=4)
    ax.axhline(0.0, color="#696b64", linewidth=1.4, zorder=3)
    ax.text((high[2] - origin[2]) * 1000 + 8, 2, "Table: Y = 0", ha="right", va="bottom",
            color="#60615c", fontsize=10)
    ax.set_xlim((low[2] - origin[2]) * 1000 - 15, (high[2] - origin[2]) * 1000 + 15)
    ax.set_ylim(-15, (high[1] - origin[1]) * 1000 + 15)
    ax.set_xlabel("Z offset (mm)")
    ax.set_ylabel("Height above table, Y (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#e6e7e1", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Chunk01 midpoint: dough, registered tools, and table", loc="left",
                 fontsize=16, fontweight="bold", pad=16)
    ax.text(0.0, 1.015,
            f"Side view looking along +X · local frame {FRAME} · original ordinal {record['original_source_frame']} · t = {record['sim_time_s']:.4f} s",
            transform=ax.transAxes, fontsize=10, color="#60615c")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#2878c8",
                      markersize=7, label="Simulated dough")]
    handles.extend(Patch(facecolor=color, alpha=0.55, label=label)
                   for color, label in zip(colors, labels, strict=True))
    ax.legend(handles=handles, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200)
    plt.close(fig)
    manifest = {
        "chunk": 1, "local_frame": FRAME, "original_pointcloud_ordinal": record["original_source_frame"],
        "sim_time_s": record["sim_time_s"], "view_direction": "+X", "projection": "orthographic Z-Y",
        "floor_y_m": 0.0, "dough_source": str((RUN / record["particles"]).relative_to(REPO)),
        "tool_meshes": [str(path.relative_to(REPO)) for path in local_meshes],
        "tool_poses": record["tool_poses"], "output": str(OUTPUT.relative_to(REPO)),
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
