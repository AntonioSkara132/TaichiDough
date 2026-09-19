#!/usr/bin/env python3
"""Render Episode18 STL tools over a different recorded colored depth frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from scipy.spatial import cKDTree

try:
    from experiments.differentiable_mpm.episode3_tool_identity import hsv_mask
except ImportError:
    from episode3_tool_identity import hsv_mask


REPO = Path(__file__).resolve().parents[2]
DEFAULT_EPISODE = REPO.parent / "data/deformpath_training/preprocessed_dataset/episode18_dynamics"
DEFAULT_REGISTRATION = Path(__file__).resolve().parent / "data/episode18_manual_pose_frame0300_v1/episode18_tool_pose_frame0300_alignment.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data/episode18_manual_pose_frame0300_v1/episode18_tool_overlay_frame0420.png"
TOOL_HSV = (0.0, 69.0, 96.0, 255.0, 189.0, 255.0)


def load_torch(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


def read_binary_stl(path: Path, visual: np.ndarray, scale: float) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"STL is too short: {path}")
    count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + count * 50:
        raise ValueError(f"Only binary STL is supported: {path}")
    values = np.frombuffer(raw, dtype=np.dtype([
        ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")
    ]), offset=84)
    return transform(values["vertices"].reshape(-1, 3) * scale, visual).reshape(-1, 3, 3)


def set_equal_limits(ax, points: np.ndarray, pad: float = 0.01) -> None:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float((upper - lower).max()) / 2.0 + pad, 0.03)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def run(episode_dir: Path, registration_path: Path, frame: int, output: Path) -> None:
    episode_dir = episode_dir.resolve()
    registration_path = registration_path.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)

    raw_payload = load_torch(episode_dir / "pointclouds.pt")
    usable_payload = load_torch(episode_dir / "pointclouds_interpolated.pt")
    paths_payload = load_torch(episode_dir / "paths_interpolated.pt")
    raw_frames = raw_payload[0]
    usable_frames = usable_payload[0]
    path = paths_payload[0]["path"]
    if not 0 <= frame < len(usable_frames):
        raise ValueError(f"--frame must be between 0 and {len(usable_frames) - 1}")
    if tuple(path.shape) != (len(usable_frames), 2, 14):
        raise ValueError("Pose and usable pointcloud counts do not match")

    selected = usable_frames[frame].numpy()
    raw_ordinal = int(round(float(np.median(selected[:, 4]))))
    raw = raw_frames[raw_ordinal].numpy()
    valid = (raw[:, 5] > 0.5) & np.isfinite(raw[:, :3]).all(axis=1)
    xyz = raw[valid, :3].astype(np.float64)
    rgb = np.clip(np.rint(raw[valid, 7:10]), 0, 255).astype(np.uint8)
    tool_mask = hsv_mask(rgb, TOOL_HSV)
    tool_points = xyz[tool_mask]
    if len(tool_points) == 0:
        raise ValueError("Selected frame has no tool-colored points")

    report = json.loads(registration_path.read_text(encoding="utf-8"))
    results = {row["name"]: row for row in report["results"]}
    meshes: dict[str, np.ndarray] = {}
    for record in report["results"]:
        name = record["name"]
        manifest_tool = next(item for item in json.loads(
            (registration_path.parent / "frame_manifest.json").read_text(encoding="utf-8")
        )["tools"] if item["name"] == name)
        visual = np.asarray(manifest_tool["tool_link_from_scaled_raw_visual"], dtype=np.float64)
        stl = Path(manifest_tool["collision_stl"]["path"])
        meshes[name] = read_binary_stl(stl, visual, float(manifest_tool["stl_scale_to_metres"]))

    poses = path[frame].numpy().astype(np.float64)
    names = ["UR5e_spathla", "gen3_spathla"]
    colors = {"UR5e_spathla": "#157fbe", "gen3_spathla": "#ef8a24"}
    predicted_meshes: dict[str, np.ndarray] = {}
    for index, name in enumerate(names):
        marker_from_mesh = np.asarray(results[name]["candidate_marker_from_mesh"], dtype=np.float64)
        pose_matrix = np.eye(4, dtype=np.float64)
        from scipy.spatial.transform import Rotation
        pose_matrix[:3, :3] = Rotation.from_quat(poses[index, 3:7]).as_matrix()
        pose_matrix[:3, 3] = poses[index, :3]
        predicted_meshes[name] = transform(
            transform(meshes[name].reshape(-1, 3), marker_from_mesh), pose_matrix
        )
    mesh_distances = np.stack([
        cKDTree(predicted_meshes[name]).query(tool_points, workers=1)[0]
        for name in names
    ], axis=1)
    owners = np.argmin(mesh_distances, axis=1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#fbfcfa")
    fig.suptitle(
        f"Episode 18 tool registration overlay · usable frame {frame} · raw frame {raw_ordinal}",
        fontsize=17,
        fontweight="bold",
    )
    for index, name in enumerate(names):
        result = results[name]
        marker_from_mesh = np.asarray(result["candidate_marker_from_mesh"], dtype=np.float64)
        world_triangles = predicted_meshes[name].reshape(-1, 3, 3)

        own_tool_points = tool_points[owners == index]
        if len(own_tool_points) == 0:
            raise ValueError(f"No tool-colored points assigned to {name}")
        tool_center = own_tool_points.mean(axis=0)
        local_context = xyz[np.all(np.abs(xyz - tool_center) < 0.16, axis=1)]
        if len(local_context) == 0:
            local_context = xyz
        all_points = np.concatenate((local_context, own_tool_points, world_triangles.reshape(-1, 3)), axis=0)

        top = axes[0, index]
        top_points = np.concatenate((local_context[:, [0, 2]], tool_points[:, [0, 2]], world_triangles.reshape(-1, 3)[:, [0, 2]]), axis=0)
        top.scatter(local_context[:, 0], local_context[:, 2], s=1, c="#aeb7b1", alpha=0.12)
        top.scatter(own_tool_points[:, 0], own_tool_points[:, 2], s=7, c="#d62728", alpha=0.72, label="tool depth points")
        top.add_collection(PolyCollection(world_triangles[:, :, [0, 2]], facecolors=colors[name], edgecolors="#173f55", linewidths=0.15, alpha=0.42))
        top.scatter([], [], s=35, c=colors[name], label="registered STL")
        top.set_title(f"{name} · top projection (X/Z)")
        top.set_xlabel("X (m)")
        top.set_ylabel("Z (m)")
        set_equal_limits(top, top_points)
        top.grid(alpha=0.25)
        top.legend(loc="upper left", fontsize=8)

        side = axes[1, index]
        side_points = np.concatenate((local_context[:, [0, 1]], tool_points[:, [0, 1]], world_triangles.reshape(-1, 3)[:, [0, 1]]), axis=0)
        side.scatter(local_context[:, 0], local_context[:, 1], s=1, c="#aeb7b1", alpha=0.12)
        side.scatter(own_tool_points[:, 0], own_tool_points[:, 1], s=7, c="#d62728", alpha=0.72, label="tool depth points")
        side.add_collection(PolyCollection(world_triangles[:, :, [0, 1]], facecolors=colors[name], edgecolors="#173f55", linewidths=0.15, alpha=0.42))
        side.scatter([], [], s=35, c=colors[name], label="registered STL")
        side.set_title(f"{name} · side projection (X/Y)")
        side.set_xlabel("X (m)")
        side.set_ylabel("Y (m)")
        set_equal_limits(side, side_points)
        side.grid(alpha=0.25)
        side.legend(loc="upper left", fontsize=8)

    fig.text(0.02, 0.015, "Red = HSV-selected recorded tool points; colored surface = saved Blender candidate transform; gray = nearby depth context", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(json.dumps({"output": str(output), "usable_frame": frame, "raw_frame": raw_ordinal, "tool_points": int(len(tool_points))}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--frame", type=int, default=420, help="Usable interpolated frame, different from the calibration frame 300.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.episode_dir, args.registration, args.frame, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
