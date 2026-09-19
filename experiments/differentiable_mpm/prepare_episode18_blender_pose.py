#!/usr/bin/env python3
"""Prepare an Episode18 frame for manual collision-tool pose editing in Blender.

The chunk files intentionally contain filtered XYZ-only point clouds.  This
tool therefore reads the RGB-preserving raw ``pointclouds.pt`` at the episode
root and uses ``pointclouds_interpolated.pt`` only to map a usable local frame
to its original raw pointcloud ordinal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from experiments.differentiable_mpm.episode3_tool_identity import hsv_mask
except ImportError:
    from episode3_tool_identity import hsv_mask


REPO = Path(__file__).resolve().parents[2]
DEFAULT_EPISODE = REPO.parent / "data/deformpath_training/preprocessed_dataset/episode18_dynamics"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data/episode18_manual_pose_frame0000_v1"
TOOL_NAMES = ("UR5e_spathla", "gen3_spathla")
POSE_FEATURES = (
    "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz",
    "wx", "wy", "wz", "relative_timestamp_s",
)
TOOL_HSV = (0.0, 69.0, 96.0, 255.0, 189.0, 255.0)
BLENDER_FROM_SCENE = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
MESHES = (
    {
        "name": "UR5e_spathla",
        "filename": "ur_spathla_collision_solid.stl",
        "visual_origin_m": (-0.002395874, -0.017992075, -0.019913439),
        "visual_rpy_rad": (4.5910, 1.379415965, -1.740698498),
    },
    {
        "name": "gen3_spathla",
        "filename": "gen3_spathla_collision_solid.stl",
        "visual_origin_m": (-0.02345833, -0.02261066, -0.01297941),
        "visual_rpy_rad": (3.12897712, 0.06996522, -3.11041673),
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_torch(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]


def write_binary_ply(path: Path, points: np.ndarray) -> None:
    values = np.ascontiguousarray(points, dtype="<f4")
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("PLY points must be a finite [N,3] array")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(values)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(values.tobytes(order="C"))


def validate_rigid(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation determinant is not +1")
    return result


def prepare(*, episode_dir: Path, frame: int, output_dir: Path, max_context_points: int, seed: int) -> dict[str, Any]:
    episode_dir = episode_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")

    raw_path = episode_dir / "pointclouds.pt"
    interpolated_path = episode_dir / "pointclouds_interpolated.pt"
    paths_path = episode_dir / "paths_interpolated.pt"
    calibration_path = episode_dir / "scene_calibration_v2.json"
    conversion_path = episode_dir / "conversion_metadata.json"
    for path in (raw_path, interpolated_path, paths_path, calibration_path, conversion_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw_payload = load_torch(raw_path)
    interpolated_payload = load_torch(interpolated_path)
    paths_payload = load_torch(paths_path)
    if not (isinstance(raw_payload, list) and len(raw_payload) == 1):
        raise ValueError("pointclouds.pt must contain one episode list")
    if not (isinstance(interpolated_payload, list) and len(interpolated_payload) == 1):
        raise ValueError("pointclouds_interpolated.pt must contain one episode list")
    if not (isinstance(paths_payload, list) and len(paths_payload) == 1):
        raise ValueError("paths_interpolated.pt must contain one episode dictionary")
    raw_frames = raw_payload[0]
    usable_frames = interpolated_payload[0]
    paths = paths_payload[0]
    if not 0 <= frame < len(usable_frames):
        raise ValueError(f"--frame must be between 0 and {len(usable_frames) - 1}")
    if tuple(paths.get("pose_frames", ())) != TOOL_NAMES:
        raise ValueError("Unexpected tool pose order")
    path_tensor = paths.get("path")
    if tuple(path_tensor.shape) != (len(usable_frames), 2, 14):
        raise ValueError("Interpolated path shape does not match usable pointcloud frames")

    selected = usable_frames[frame].numpy()
    if selected.ndim != 2 or selected.shape[1] < 5:
        raise ValueError("Interpolated pointcloud layout is unsupported")
    raw_ordinal = int(round(float(np.median(selected[:, 4]))))
    if not 0 <= raw_ordinal < len(raw_frames):
        raise ValueError(f"Interpolated frame maps to invalid raw ordinal {raw_ordinal}")
    raw = raw_frames[raw_ordinal].numpy()
    if raw.ndim != 2 or raw.shape[1] != 10:
        raise ValueError("Episode18 raw pointclouds must use [N,10] layout")
    if not np.allclose(raw[:, 4], raw_ordinal, atol=0.0):
        raise ValueError("Raw frame-number column does not match its ordinal")
    valid = (raw[:, 5] > 0.5) & np.isfinite(raw[:, :3]).all(axis=1)
    if not valid.any():
        raise ValueError("Selected raw pointcloud has no valid points")
    xyz_source = raw[valid, :3].astype(np.float64)
    rgb = np.clip(np.rint(raw[valid, 7:10]), 0, 255).astype(np.uint8)
    relative_time = float(np.median(raw[valid, 6]))

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    scene_from_source = validate_rigid(np.asarray(calibration["scene_from_source"]), "scene_from_source")
    blender_from_scene = validate_rigid(BLENDER_FROM_SCENE, "blender_from_scene")
    blender_from_mocap = validate_rigid(blender_from_scene @ scene_from_source, "blender_from_mocap")
    xyz_blender = transform_points(xyz_source, blender_from_mocap)
    tool = xyz_blender[hsv_mask(rgb, TOOL_HSV)]
    if len(tool) == 0:
        raise ValueError("Selected frame has no points in the configured tool HSV range")
    rng = np.random.default_rng(seed)
    context = xyz_blender if len(xyz_blender) <= max_context_points else xyz_blender[np.sort(rng.choice(len(xyz_blender), max_context_points, replace=False))]

    interpolated_time = float(path_tensor[frame, 0, 13].item())
    if not np.isclose(interpolated_time, relative_time, atol=2e-3):
        raise ValueError(f"Pointcloud/path time mismatch: {relative_time} vs {interpolated_time}")
    poses = []
    marker_matrices = []
    for index, name in enumerate(TOOL_NAMES):
        pose = path_tensor[frame, index].numpy().astype(np.float64)
        if not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:7]), 1.0, atol=1e-5):
            raise ValueError(f"Invalid pose for {name}")
        poses.append(pose)
        marker_matrices.append(validate_rigid(blender_from_mocap @ pose_matrix(pose), f"blender_from_{name}"))

    mesh_paths = {item["name"]: REPO / "meshes" / item["filename"] for item in MESHES}
    for path in mesh_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        context_path = temporary / "context_points.ply"
        tool_path = temporary / "tool_hsv_points.ply"
        write_binary_ply(context_path, context)
        write_binary_ply(tool_path, tool)
        conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
        timestamps = conversion["episode"]["timestamps_ns"]
        generated = {
            context_path.name: {"sha256": sha256(context_path), "size_bytes": context_path.stat().st_size, "points": len(context)},
            tool_path.name: {"sha256": sha256(tool_path), "size_bytes": tool_path.stat().st_size, "points": len(tool)},
        }
        tools = []
        for index, item in enumerate(MESHES):
            visual = np.eye(4, dtype=np.float64)
            visual[:3, :3] = rpy_matrix(item["visual_rpy_rad"])
            visual[:3, 3] = item["visual_origin_m"]
            tools.append({
                "name": item["name"],
                "collision_stl": {"path": str(mesh_paths[item["name"]]), "sha256": sha256(mesh_paths[item["name"]]), "units": "millimetres", "coordinate_frame": "raw_stl_visual"},
                "tool_link_from_scaled_raw_visual": visual.tolist(),
                "stl_scale_to_metres": 0.001,
                "source_pose_xyzw": poses[index][:7].tolist(),
                "source_from_marker": pose_matrix(poses[index]).tolist(),
                "blender_from_marker": marker_matrices[index].tolist(),
                "initial_marker_from_mesh": np.eye(4).tolist(),
            })
        manifest = {
            "schema": "taichidough/episode3-blender-pose-frame/v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "manual_pose_preparation",
            "validated_geometry": False,
            "episode18_registration_used": False,
            "dataset_name": "episode18_dynamics",
            "frame": {
                "usable_interpolated_ordinal": frame,
                "raw_pointcloud_ordinal": raw_ordinal,
                "header_stamp_ns": str(timestamps[raw_ordinal]),
                "relative_timestamp_s": relative_time,
                "valid_context_points_before_downsample": len(xyz_source),
                "context_points_written": len(context),
                "tool_hsv_points_written": len(tool),
            },
            "tool_hsv_opencv": {"h": [0, 69], "s": [96, 255], "v": [189, 255]},
            "coordinate_frames": {
                "point_source": calibration.get("source_frame", "mocap"),
                "scene": calibration.get("scene_frame", "mocap"),
                "blender": "right-handed Z-up; [x,y,z]_scene -> [x,-z,y]_blender",
                "matrix_convention": "target_from_source, column-vector homogeneous transforms",
                "scene_from_source": scene_from_source.tolist(),
                "blender_from_scene": blender_from_scene.tolist(),
                "blender_from_mocap": blender_from_mocap.tolist(),
            },
            "composition": "blender_from_mocap @ source_from_marker @ marker_from_mesh @ tool_link_from_scaled_raw_visual @ positive_0.001_scale @ raw_stl_mm",
            "tools": tools,
            "point_files": generated,
            "source_hashes": {str(path): sha256(path) for path in (raw_path, interpolated_path, paths_path, calibration_path, conversion_path)},
            "settings": {"max_context_points": max_context_points, "seed": seed, "raw_source_ordinal": raw_ordinal},
            "operations_not_run": ["calibration", "automatic_registration", "MPM", "gradient", "material_fit", "dataset_creation"],
        }
        (temporary / "frame_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (temporary / "README.txt").write_text(
            "Episode18 manual collision-tool pose scene.\n\n"
            "Move and rotate only Registration_UR5e_spathla and Registration_gen3_spathla.\n"
            "Do not move Marker_* or scale the tool meshes. Gray points are context;\n"
            "red points are the configured tool HSV mask.\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--frame", type=int, default=0, help="Usable interpolated frame index, not raw pointcloud ordinal.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-context-points", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        manifest = prepare(
            episode_dir=args.episode_dir,
            frame=args.frame,
            output_dir=args.output,
            max_context_points=args.max_context_points,
            seed=args.seed,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}")
        return 2
    print(json.dumps({"output": str(args.output.resolve()), "frame": manifest["frame"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
