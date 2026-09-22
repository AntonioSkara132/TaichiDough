#!/usr/bin/env python3
"""Prepare one direct-mocap-corrected Episode3 frame for manual tool-pose editing in Blender.

This is a lighter sibling of prepare_episode3_blender_pose.py: it consumes the
export produced by export_episode3_dynamics_deformpath_direct_mocap.py (camera/
mocap calibration taken from the bag's own 'mocap' -> 'camera_link' TF, no tag4
recalibration) instead of the older, hash-pinned 'full_export_corrected' bundle.
It does not carry that script's provenance pinning (frame count, source hashes) --
this is a fresh, unvalidated-lineage export for manual visualization/alignment,
not yet wired into the paper-facing provenance chain.

Output layout (a fresh directory under experiments/differentiable_mpm/data/):
  context_points.ply     - all valid dough+tool points for the frame (gray, for context)
  tool_hsv_points.ply    - points passing a tool-colored HSV mask (red, sanity check)
  frame_manifest.json    - per-tool source pose, mesh info, and the blender_from_*
                            matrices needed to place each collision mesh
  README.txt             - Blender import/alignment instructions
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.differentiable_mpm.episode3_tool_identity import hsv_mask  # pyright: ignore[reportMissingImports]

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_EXPORT = Path(
    "/home/antonio/diplomski_antonio/diplomski/data/deformpath_training/staging/"
    "episode3_alt_dynamics_export_direct_mocap_v1"
)
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "data/episode3_manual_pose_direct_mocap_v1"
TOOL_NAMES = ("UR5e_spathla", "gen3_spathla")
TOOL_HSV = (0.0, 69.0, 96.0, 255.0, 189.0, 255.0)
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
# Same fixed right-handed Z-up convention used by prepare_episode3_blender_pose.py,
# applied here directly to the export's native pointcloud/pose frame instead of a
# post-calibration "scene" frame: [x,y,z]_camera -> [x,-z,y]_blender.
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
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(value[3:7]).as_matrix()
    result[:3, 3] = value[:3]
    return result


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    return values @ matrix[:3, :3].T + matrix[:3, 3]


def deterministic_sample(points: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    values = np.asarray(points)
    if maximum <= 0 or len(values) <= maximum:
        return values.copy()
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(values), maximum, replace=False))
    return values[indices]


def write_binary_ply(path: Path, points: np.ndarray) -> None:
    values = np.ascontiguousarray(points, dtype="<f4")
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


def prepare(*, frame: int, export_dir: Path, output_dir: Path, max_context_points: int, seed: int) -> dict[str, Any]:
    import torch

    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    export = export_dir.resolve()
    pointclouds_path = export / "pointclouds.pt"
    paths_path = export / "paths.pt"
    metadata_path = export / "conversion_metadata.json"
    mesh_paths = {item["name"]: REPOSITORY_ROOT / "meshes" / item["filename"] for item in MESHES}
    for path in (pointclouds_path, paths_path, metadata_path, *mesh_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text())
    if tuple(metadata.get("pose_frames", ())) != TOOL_NAMES:
        raise ValueError("Pose frame order differs from expected (UR5e_spathla, gen3_spathla)")

    point_payload = torch.load(pointclouds_path, map_location="cpu", weights_only=True)
    frame_count = len(point_payload[0])
    if not 0 <= frame < frame_count:
        raise ValueError(f"Frame must be between 0 and {frame_count - 1}")

    values = point_payload[0][frame].numpy()
    valid = (values[:, 5] > 0.5) & np.isfinite(values[:, :3]).all(axis=1)
    xyz_camera = values[valid, :3].astype(np.float64)
    rgb_values = values[valid, 7:10]
    if len(xyz_camera) == 0:
        raise ValueError("Selected point cloud is empty")
    rgb = np.clip(np.rint(rgb_values), 0, 255).astype(np.uint8)
    frame_times = values[valid, 6].astype(np.float64)
    relative_time = float(np.median(frame_times))

    path_payload = torch.load(paths_path, map_location="cpu", weights_only=True)
    poses = []
    marker_matrices = []
    for name in TOOL_NAMES:
        stream = path_payload["streams"][name]
        pose = stream[frame].numpy().astype(np.float64)
        if not np.isfinite(pose).all():
            raise ValueError(f"Path row is not finite for {name} at frame {frame}; pick another frame")
        poses.append(pose)
        marker_matrices.append(BLENDER_FROM_SCENE @ pose_matrix(pose))

    xyz_blender = transform_points(xyz_camera, BLENDER_FROM_SCENE)
    tool_mask = hsv_mask(rgb, TOOL_HSV)
    tool = xyz_blender[tool_mask]
    context = deterministic_sample(xyz_blender, max_context_points, seed)

    output.mkdir(parents=True, exist_ok=False)
    context_path = output / "context_points.ply"
    tool_path = output / "tool_hsv_points.ply"
    write_binary_ply(context_path, context)
    if len(tool) > 0:
        write_binary_ply(tool_path, tool)
    (output / "README.txt").write_text(
        "Episode3 manual collision-tool pose scene (direct-mocap-corrected export).\n\n"
        "Import context_points.ply and (if present) tool_hsv_points.ply as point\n"
        "clouds, then import each tool's collision STL and apply its\n"
        "'blender_from_marker' matrix from frame_manifest.json as the mesh object's\n"
        "world matrix -- that places it exactly where the corrected mocap pipeline\n"
        "currently thinks it is. Then nudge each mesh (move/rotate only, do not\n"
        "scale) until it lines up with the tool-colored points, and read off the\n"
        "residual transform: that residual is the marker-to-tool-tip offset this\n"
        "episode has been missing.\n\n"
        "Gray points are context (whole dough+scene point cloud). Red points pass\n"
        "the tool HSV mask used elsewhere in the Episode3 pipeline, as a sanity\n"
        "check on where the tool already shows up in the depth camera's own colors.\n"
    )

    tools = []
    for index, item in enumerate(MESHES):
        visual = np.eye(4, dtype=np.float64)
        visual[:3, :3] = rpy_matrix(item["visual_rpy_rad"])
        visual[:3, 3] = item["visual_origin_m"]
        mesh_path = mesh_paths[item["name"]]
        tools.append({
            "name": item["name"],
            "collision_stl": {"path": str(mesh_path), "sha256": sha256(mesh_path), "units": "millimetres"},
            "tool_link_from_scaled_raw_visual": visual.tolist(),
            "stl_scale_to_metres": 0.001,
            "source_pose_xyzw": poses[index][:7].tolist(),
            "source_from_marker": pose_matrix(poses[index]).tolist(),
            "blender_from_marker": marker_matrices[index].tolist(),
        })

    manifest = {
        "schema": "taichidough/episode3-blender-pose-frame-direct-mocap/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "manual_pose_preparation",
        "validated_geometry": False,
        "source_export_dir": str(export),
        "source_export_camera_to_mocap_mode": metadata.get("camera_to_mocap", {}).get("mode"),
        "frame": {
            "raw_pointcloud_ordinal": frame,
            "relative_timestamp_s": relative_time,
            "context_points_written": len(context),
            "tool_hsv_points_written": len(tool),
        },
        "tool_hsv_opencv": {"h": [TOOL_HSV[0], TOOL_HSV[1]], "s": [TOOL_HSV[2], TOOL_HSV[3]], "v": [TOOL_HSV[4], TOOL_HSV[5]]},
        "coordinate_frames": {
            "point_source": metadata.get("pointcloud_frame"),
            "blender": "right-handed Z-up; [x,y,z]_camera -> [x,-z,y]_blender",
            "matrix_convention": "target_from_source, column-vector homogeneous transforms",
            "blender_from_scene": BLENDER_FROM_SCENE.tolist(),
        },
        "composition": "blender_from_marker @ tool_link_from_scaled_raw_visual @ 0.001_scale @ raw_stl_mm",
        "tools": tools,
        "operations_not_run": ["automatic_registration", "MPM", "gradient", "material_fit", "dataset_creation"],
    }
    (output / "frame_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=800)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-context-points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    manifest = prepare(
        frame=args.frame,
        export_dir=args.export_dir,
        output_dir=args.output,
        max_context_points=args.max_context_points,
        seed=args.seed,
    )
    print(f"Wrote {args.output}")
    print(json.dumps(manifest["frame"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
