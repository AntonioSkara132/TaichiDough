#!/usr/bin/env python3
"""Prepare one corrected Episode3 frame for manual tool-pose editing in Blender."""
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

from experiments.differentiable_mpm.episode3_tool_identity import hsv_mask  # pyright: ignore[reportMissingImports]


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
STAGING_ROOT = REPOSITORY_ROOT.parent / "data/deformpath_training/staging/episode3_dynamics_dataset_filt_v1"
DEFAULT_EXPORT = STAGING_ROOT / "full_export_corrected"
DEFAULT_SELECTION = STAGING_ROOT / "range_selection.json"
DEFAULT_CALIBRATION = EXPERIMENT_ROOT / "data/episode3_dynamics_dataset_filt_v1/scene_calibration_table_aligned_v2.json"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "data/episode3_manual_pose_frame0431_v1"
TOOL_NAMES = ("UR5e_spathla", "gen3_spathla")
TOOL_HSV = (0.0, 69.0, 96.0, 255.0, 189.0, 255.0)
POINT_FEATURES = (
    "x", "y", "z", "normalized_frame_time", "frame_number", "valid_flag",
    "relative_timestamp_s", "r", "g", "b",
)
POSE_FEATURES = (
    "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz",
    "wx", "wy", "wz", "relative_timestamp_s",
)
EXPECTED_HASHES = {
    "pointclouds.pt": "cf3f72a8ba7630a298f1203f83a96d6299fe0608af6423fe5b802aa251cf8dba",
    "paths.pt": "dc06565bb73ddba7d792dd001407cbd0423762409f74d34cdacb5bc0cf84143c",
    "conversion_metadata.json": "51e2b3afd5295eaa330c584bd0af47cf24e8edd77b8039c0ba02d651334f589f",
}
EXPECTED_CALIBRATION_SHA256 = "e6e0eb95e6d75a90e09ca9757b471609f5ba8014a3d24d2e353cc4f179fdc1c7"
EXPECTED_SOURCE_FINGERPRINT = "473a99dace623f9e0b5db93528b3985d3748cbac69d00a1b85990b3a99a24708"
MESHES = (
    {
        "name": "UR5e_spathla",
        "filename": "ur_spathla_collision_solid.stl",
        "sha256": "65e2afa80894c45dde9ef1791786c8652d1de22a0a6b39de098fb2567552b251",
        "visual_origin_m": (-0.002395874, -0.017992075, -0.019913439),
        "visual_rpy_rad": (4.5910, 1.379415965, -1.740698498),
    },
    {
        "name": "gen3_spathla",
        "filename": "gen3_spathla_collision_solid.stl",
        "sha256": "5bc2afd71f3bd31201996ad33d6f91600a5e3899b1c9b280f990106cf89f043d",
        "visual_origin_m": (-0.02345833, -0.02261066, -0.01297941),
        "visual_rpy_rad": (3.12897712, 0.06996522, -3.11041673),
    },
)
BLENDER_FROM_SCENE = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


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
    value = np.asarray(pose, dtype=np.float64)
    if value.shape[0] < 7:
        raise ValueError("Tool pose must contain XYZ and XYZW quaternion")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(value[3:7]).as_matrix()
    result[:3, 3] = value[:3]
    return result


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    return values @ matrix[:3, :3].T + matrix[:3, 3]


def validate_rigid(matrix: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation determinant is not +1")
    return value


def deterministic_sample(points: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    values = np.asarray(points)
    if maximum <= 0 or len(values) <= maximum:
        return values.copy()
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(values), maximum, replace=False))
    return values[indices]


def write_binary_ply(path: Path, points: np.ndarray) -> None:
    values = np.ascontiguousarray(points, dtype="<f4")
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("PLY points must be a finite [N,3] array")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(values)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(values.tobytes(order="C"))


def retained_membership(frame: int, ranges: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for item in ranges:
        start = int(item["start_pointcloud_ordinal"])
        end = int(item["end_pointcloud_ordinal_exclusive"])
        if start <= frame < end:
            return {
                "id": item["id"],
                "output_name": item["output_name"],
                "chunk_local_frame": frame - start,
            }
    return None


def prepare(
    *,
    frame: int,
    export_dir: Path,
    selection_path: Path,
    calibration_path: Path,
    output_dir: Path,
    max_context_points: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    output = output_dir.resolve()
    if not output.is_relative_to(EXPERIMENT_ROOT) or output == EXPERIMENT_ROOT:
        raise ValueError("Output must be inside experiments/differentiable_mpm")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    export = export_dir.resolve()
    sources = {name: export / name for name in EXPECTED_HASHES}
    sources["table_aligned_calibration"] = calibration_path.resolve()
    sources["range_selection"] = selection_path.resolve()
    mesh_paths = {item["name"]: REPOSITORY_ROOT / "meshes" / item["filename"] for item in MESHES}
    for path in (*sources.values(), *mesh_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    stats_before = {str(path): source_stat(path) for path in (*sources.values(), *mesh_paths.values())}
    hashes = {}
    for name, expected in EXPECTED_HASHES.items():
        actual = sha256(sources[name])
        if actual != expected:
            raise ValueError(f"Corrected source hash differs for {name}")
        hashes[str(sources[name])] = actual
    calibration_hash = sha256(sources["table_aligned_calibration"])
    if calibration_hash != EXPECTED_CALIBRATION_SHA256:
        raise ValueError("Table-aligned calibration hash differs")
    hashes[str(sources["table_aligned_calibration"])] = calibration_hash
    selection_hash = sha256(sources["range_selection"])
    hashes[str(sources["range_selection"])] = selection_hash
    for item in MESHES:
        path = mesh_paths[item["name"]]
        actual = sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"Collision STL hash differs for {item['name']}")
        hashes[str(path)] = actual

    selection = json.loads(sources["range_selection"].read_text())
    if selection.get("schema") != "taichidough/deformpath-range-selection/v1":
        raise ValueError("Range selection schema is not supported")
    if selection["source"].get("fingerprint") != EXPECTED_SOURCE_FINGERPRINT:
        raise ValueError("Episode3 source fingerprint differs")
    metadata = json.loads(sources["conversion_metadata.json"].read_text())
    if metadata.get("output_frame") != "mocap":
        raise ValueError("Corrected point clouds must be in mocap coordinates")
    if tuple(metadata.get("pose_frames", ())) != TOOL_NAMES:
        raise ValueError("Pose frame order differs")
    if tuple(metadata["pointcloud_layout"]["axis_1_features"]) != POINT_FEATURES:
        raise ValueError("Point-cloud layout differs")
    if tuple(metadata["path_layout"]["axis_1_features"]) != POSE_FEATURES:
        raise ValueError("Path layout differs")
    frame_count = int(metadata["episode"]["num_frames"])
    if frame_count != 2017 or not 0 <= frame < frame_count:
        raise ValueError(f"Frame must be between 0 and {frame_count - 1}")

    calibration = json.loads(sources["table_aligned_calibration"].read_text())
    scene_from_source = validate_rigid(np.asarray(calibration["scene_from_source"]), name="scene_from_source")
    validate_rigid(BLENDER_FROM_SCENE, name="blender_from_scene")
    blender_from_mocap = validate_rigid(BLENDER_FROM_SCENE @ scene_from_source, name="blender_from_mocap")

    point_payload = torch.load(sources["pointclouds.pt"], map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(point_payload, list) or len(point_payload) != 1 or len(point_payload[0]) != frame_count:
        raise ValueError("Point-cloud archive layout differs")
    values = point_payload[0][frame].numpy()
    if values.ndim != 2 or values.shape[1] != 10:
        raise ValueError("Selected point-cloud frame layout differs")
    frame_numbers = values[:, 4]
    if not np.allclose(frame_numbers, frame, atol=0.0):
        raise ValueError("Point-cloud frame-number column differs")
    valid = (values[:, 5] > 0.5) & np.isfinite(values[:, :3]).all(axis=1)
    xyz_mocap = values[valid, :3].astype(np.float64)
    rgb_values = values[valid, 7:10]
    if len(xyz_mocap) == 0 or not np.isfinite(rgb_values).all():
        raise ValueError("Selected point cloud is empty or has invalid RGB")
    if float(rgb_values.min()) < 0.0 or float(rgb_values.max()) > 255.0:
        raise ValueError("RGB values are outside 0..255")
    rgb = np.clip(np.rint(rgb_values), 0, 255).astype(np.uint8)
    frame_times = values[valid, 6].astype(np.float64)
    relative_time = float(np.median(frame_times))
    if float(np.max(np.abs(frame_times - relative_time))) > 1e-5:
        raise ValueError("Point timestamp is not constant within the selected frame")

    path_payload = torch.load(sources["paths.pt"], map_location="cpu", weights_only=True, mmap=True)
    if tuple(path_payload.get("pose_frames", ())) != TOOL_NAMES:
        raise ValueError("Path archive tool order differs")
    if tuple(path_payload.get("feature_order", ())) != POSE_FEATURES:
        raise ValueError("Path archive feature order differs")
    poses = []
    marker_matrices = []
    for name in TOOL_NAMES:
        stream = path_payload["streams"][name]
        if tuple(stream.shape) != (frame_count, 14):
            raise ValueError(f"Path stream layout differs for {name}")
        pose = stream[frame].numpy().astype(np.float64)
        if not np.isfinite(pose).all():
            raise ValueError(f"Path row is not finite for {name}")
        if not np.isclose(np.linalg.norm(pose[3:7]), 1.0, atol=1e-5):
            raise ValueError(f"Quaternion is not normalized for {name}")
        if not np.isclose(pose[13], relative_time, atol=1e-5):
            raise ValueError(f"Point and pose timestamps differ for {name}")
        poses.append(pose)
        marker_matrices.append(validate_rigid(blender_from_mocap @ pose_matrix(pose), name=f"blender_from_{name}"))

    xyz_blender = transform_points(xyz_mocap, blender_from_mocap)
    tool = xyz_blender[hsv_mask(rgb, TOOL_HSV)]
    if len(tool) == 0:
        raise ValueError("The selected frame has no points in the tool HSV range")
    context = deterministic_sample(xyz_blender, max_context_points, seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        context_path = temporary / "context_points.ply"
        tool_path = temporary / "tool_hsv_points.ply"
        write_binary_ply(context_path, context)
        write_binary_ply(tool_path, tool)
        (temporary / "README.txt").write_text(
            "Episode3 manual collision-tool pose scene.\n\n"
            "Open episode3_tool_pose_frame0431.blend. Move and rotate only\n"
            "Registration_UR5e_spathla and Registration_gen3_spathla.\n"
            "Use G to move, R to rotate, N for numeric transforms, Home to frame\n"
            "the scene, and Ctrl+S to save. Do not move Marker_* or scale tools.\n"
            "Gray points are context; red points pass the requested tool HSV mask.\n"
        )
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
                "collision_stl": {"path": str(mesh_paths[item["name"]]), "sha256": item["sha256"], "units": "millimetres", "coordinate_frame": "raw_stl_visual"},
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
            "frame": {
                "raw_pointcloud_ordinal": frame,
                "header_stamp_ns": str(metadata["episode"]["timestamps_ns"][frame]),
                "relative_timestamp_s": relative_time,
                "retained_membership": retained_membership(frame, selection["retained_ranges"]),
                "valid_context_points_before_downsample": len(xyz_blender),
                "context_points_written": len(context),
                "tool_hsv_points_written": len(tool),
            },
            "tool_hsv_opencv": {"h": [0, 69], "s": [96, 255], "v": [189, 255]},
            "coordinate_frames": {
                "point_source": "mocap",
                "scene": "episode3-table-aligned",
                "blender": "right-handed Z-up; [x,y,z]_scene -> [x,-z,y]_blender",
                "matrix_convention": "target_from_source, column-vector homogeneous transforms",
                "scene_from_source": scene_from_source.tolist(),
                "blender_from_scene": BLENDER_FROM_SCENE.tolist(),
                "blender_from_mocap": blender_from_mocap.tolist(),
            },
            "composition": "blender_from_mocap @ source_from_marker @ marker_from_mesh @ tool_link_from_scaled_raw_visual @ positive_0.001_scale @ raw_stl_mm",
            "tools": tools,
            "point_files": generated,
            "source_hashes": hashes,
            "source_stats_before": stats_before,
            "settings": {"max_context_points": max_context_points, "seed": seed},
            "operations_not_run": ["calibration", "automatic_registration", "MPM", "gradient", "material_fit", "dataset_creation"],
        }
        manifest_path = temporary / "frame_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        stats_after = {str(path): source_stat(path) for path in (*sources.values(), *mesh_paths.values())}
        if stats_after != stats_before:
            raise RuntimeError("An Episode3 input changed during preparation")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=431)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-context-points", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        manifest = prepare(
            frame=args.frame,
            export_dir=args.export_dir,
            selection_path=args.selection,
            calibration_path=args.calibration,
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
