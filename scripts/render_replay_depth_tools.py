#!/usr/bin/env python3
"""Render an existing SDF replay with dough depth and animated STL tool meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    from deformpath_topview import project_camera_points, rigid_inverse, transform_points
    from taichi_viscoelastic_mpm_scene import (
        KINOVA_TOOL_VISUAL_ORIGIN,
        KINOVA_TOOL_VISUAL_RPY,
        UR_TOOL_VISUAL_ORIGIN,
        UR_TOOL_VISUAL_RPY,
        load_binary_stl,
        mesh_in_tool_frame,
        quaternion_to_matrix,
        rpy_to_matrix,
    )
except ImportError:
    from .deformpath_topview import project_camera_points, rigid_inverse, transform_points
    from .taichi_viscoelastic_mpm_scene import (
        KINOVA_TOOL_VISUAL_ORIGIN,
        KINOVA_TOOL_VISUAL_RPY,
        UR_TOOL_VISUAL_ORIGIN,
        UR_TOOL_VISUAL_RPY,
        load_binary_stl,
        mesh_in_tool_frame,
        quaternion_to_matrix,
        rpy_to_matrix,
    )


TOOL_COLORS = np.array([[54, 137, 214], [218, 82, 72]], dtype=np.float32)
DOUGH_COLOR = np.array([209, 142, 76], dtype=np.float32)


def find_view(frame: dict[str, Any], name: str) -> dict[str, Any]:
    for view in frame.get("views", []):
        if view.get("name") == name:
            return view
    raise ValueError(f"Replay frame {frame.get('frame')} has no {name!r} view")


def validate_camera(camera: dict[str, Any]) -> None:
    required = ("width", "height", "fx", "fy", "cx", "cy", "zNear", "zFar", "scene_from_camera")
    missing = [key for key in required if key not in camera]
    if missing:
        raise ValueError("Replay camera is missing " + ", ".join(missing))
    if int(camera["width"]) <= 0 or int(camera["height"]) <= 0:
        raise ValueError("Replay camera dimensions must be positive")
    if not np.isfinite([float(camera[key]) for key in ("fx", "fy", "cx", "cy", "zNear", "zFar")]).all():
        raise ValueError("Replay camera intrinsics must be finite")
    if float(camera["zNear"]) <= 0 or float(camera["zFar"]) <= float(camera["zNear"]):
        raise ValueError("Replay camera depth range is invalid")
    transform = np.asarray(camera["scene_from_camera"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Replay camera scene_from_camera must be a finite 4x4 matrix")


def validate_tool_poses(frame: dict[str, Any]) -> np.ndarray:
    poses = np.asarray(frame.get("tool_poses_scene"), dtype=np.float64)
    if poses.shape != (2, 7) or not np.isfinite(poses).all():
        raise ValueError(f"Replay frame {frame.get('frame')} has invalid tool_poses_scene")
    if np.any(np.linalg.norm(poses[:, 3:], axis=1) < 1e-8):
        raise ValueError(f"Replay frame {frame.get('frame')} has a zero-norm tool quaternion")
    return poses


def mesh_scene_vertices(link_vertices: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Place vertices already expressed in mesh-tool-link coordinates into scene coordinates."""
    values = np.asarray(link_vertices, dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or pose.shape != (7,):
        raise ValueError("Expected [N, 3] tool-link vertices and an XYZW scene pose")
    return values @ quaternion_to_matrix(pose[3:]).T + pose[:3]


def project_scene_vertices(vertices_scene: np.ndarray, camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    camera_from_scene = rigid_inverse(np.asarray(camera["scene_from_camera"], dtype=np.float64))
    camera_vertices = transform_points(vertices_scene, camera_from_scene)
    return project_camera_points(camera_vertices, camera), camera_vertices[:, 2]


def rasterize_triangles(
    depth: np.ndarray,
    owner: np.ndarray,
    vertices_scene: np.ndarray,
    camera: dict[str, Any],
    tool_index: int,
) -> None:
    """Depth-test unindexed scene triangles into the shared replay depth buffer."""
    projected, camera_z = project_scene_vertices(vertices_scene, camera)
    height, width = depth.shape
    near = float(camera["zNear"])
    far = float(camera["zFar"])
    triangles = projected.reshape(-1, 3, 3)
    triangle_z = camera_z.reshape(-1, 3)
    for uvz, z_values in zip(triangles, triangle_z):
        if not np.isfinite(uvz).all() or np.any(z_values <= near) or np.any(z_values >= far):
            continue
        x0, y0 = uvz[0, :2]
        x1, y1 = uvz[1, :2]
        x2, y2 = uvz[2, :2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denominator) < 1e-10:
            continue
        xmin = max(0, int(np.ceil(min(x0, x1, x2) - 0.5)))
        xmax = min(width - 1, int(np.floor(max(x0, x1, x2) - 0.5)))
        ymin = max(0, int(np.ceil(min(y0, y1, y2) - 0.5)))
        ymax = min(height - 1, int(np.floor(max(y0, y1, y2) - 0.5)))
        if xmin > xmax or ymin > ymax:
            continue
        grid_x, grid_y = np.meshgrid(
            np.arange(xmin, xmax + 1, dtype=np.float64) + 0.5,
            np.arange(ymin, ymax + 1, dtype=np.float64) + 0.5,
        )
        weight0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
        weight1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
        weight2 = 1.0 - weight0 - weight1
        inside = (weight0 >= -1e-8) & (weight1 >= -1e-8) & (weight2 >= -1e-8)
        inverse_depth = weight0 / z_values[0] + weight1 / z_values[1] + weight2 / z_values[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            triangle_depth = 1.0 / inverse_depth
        update = inside & np.isfinite(triangle_depth) & (triangle_depth > near) & (triangle_depth < depth[ymin:ymax + 1, xmin:xmax + 1])
        if np.any(update):
            region_depth = depth[ymin:ymax + 1, xmin:xmax + 1]
            region_owner = owner[ymin:ymax + 1, xmin:xmax + 1]
            region_depth[update] = triangle_depth[update]
            region_owner[update] = tool_index


def colorize_depth(depth: np.ndarray, owner: np.ndarray, z_near: float, z_far: float) -> np.ndarray:
    valid = depth < z_far - 1e-6
    normalized = np.clip((depth - z_near) / max(z_far - z_near, 1e-6), 0.0, 1.0)
    brightness = 0.38 + 0.62 * (1.0 - normalized)
    rgb = np.full((*depth.shape, 3), 18.0, dtype=np.float32)
    dough = valid & (owner < 0)
    rgb[dough] = DOUGH_COLOR * brightness[dough, None]
    for tool_index, color in enumerate(TOOL_COLORS):
        pixels = owner == tool_index
        rgb[pixels] = color * brightness[pixels, None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def render_frame(
    frame: dict[str, Any],
    camera: dict[str, Any],
    tool_link_meshes: list[np.ndarray],
) -> np.ndarray:
    poses = validate_tool_poses(frame)
    depth_path = Path(frame_view_path(frame, camera, "depth_array"))
    dough_depth = np.load(depth_path).astype(np.float64)
    expected = (int(camera["height"]), int(camera["width"]))
    if dough_depth.shape != expected or not np.isfinite(dough_depth).all():
        raise ValueError(f"Replay dough depth {depth_path} does not match the calibrated camera")
    depth = dough_depth.copy()
    owner = np.full(depth.shape, -1, dtype=np.int8)
    for tool_index, (mesh, pose) in enumerate(zip(tool_link_meshes, poses)):
        rasterize_triangles(depth, owner, mesh_scene_vertices(mesh, pose), camera, tool_index)
    return colorize_depth(depth, owner, float(camera["zNear"]), float(camera["zFar"]))


def frame_view_path(frame: dict[str, Any], camera: dict[str, Any], key: str) -> str:
    for view in frame["views"]:
        if int(view.get("width", -1)) == int(camera["width"]) and int(view.get("height", -1)) == int(camera["height"]):
            if key in view:
                return str(view[key])
    raise ValueError(f"Replay frame {frame.get('frame')} has no {key}")


def load_tool_link_meshes(ur_mesh: Path, kinova_mesh: Path, scale: float) -> list[np.ndarray]:
    ur_raw, _ = load_binary_stl(ur_mesh, scale)
    kinova_raw, _ = load_binary_stl(kinova_mesh, scale)
    return [
        mesh_in_tool_frame(ur_raw, UR_TOOL_VISUAL_ORIGIN, rpy_to_matrix(UR_TOOL_VISUAL_RPY)),
        mesh_in_tool_frame(kinova_raw, KINOVA_TOOL_VISUAL_ORIGIN, rpy_to_matrix(KINOVA_TOOL_VISUAL_RPY)),
    ]


def add_label(image: np.ndarray, frame: dict[str, Any], modulus_pa: float | None) -> Image.Image:
    result = Image.fromarray(image, "RGB")
    draw = ImageDraw.Draw(result)
    label = f"source {frame.get('source_frame', frame.get('frame'))}  |  t={float(frame.get('sim_time_s', 0.0)):.3f}s"
    if modulus_pa is not None:
        label += f"  |  E={modulus_pa:.6g} Pa"
    draw.rectangle((10, 10, min(result.width - 10, 520), 44), fill=(0, 0, 0))
    draw.text((18, 17), label, fill=(255, 255, 255))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=7.572)
    parser.add_argument("--youngs-modulus", type=float, default=None)
    parser.add_argument("--ur-tool-mesh", type=Path, default=Path(__file__).resolve().parents[1] / "meshes/ur_spathla.stl")
    parser.add_argument("--kinova-tool-mesh", type=Path, default=Path(__file__).resolve().parents[1] / "meshes/gen3_spathla.stl")
    parser.add_argument("--tool-mesh-scale", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.tool_mesh_scale <= 0:
        raise ValueError("--fps and --tool-mesh-scale must be positive")
    replay_dir = args.replay_dir.resolve()
    metadata = json.loads((replay_dir / "camera_parameters.json").read_text())
    if (metadata.get("replay") or {}).get("tool_pose_frame") != "mesh_tool_link":
        raise ValueError("Tool mesh rendering requires an SDF replay with mesh_tool_link poses")
    frames = metadata.get("frames") or []
    if not frames:
        raise ValueError("Replay metadata contains no frames")
    camera = find_view(frames[0], "deformpath_top")
    validate_camera(camera)
    for frame in frames:
        view = find_view(frame, "deformpath_top")
        if any(view.get(key) != camera.get(key) for key in ("width", "height", "fx", "fy", "cx", "cy", "zNear", "zFar", "scene_from_camera")):
            raise ValueError("Replay camera changes between frames")
        validate_tool_poses(frame)
    tool_link_meshes = load_tool_link_meshes(args.ur_tool_mesh, args.kinova_tool_mesh, args.tool_mesh_scale)
    frames_dir = args.frames_dir or replay_dir / "depth_tools_frames"
    if frames_dir.exists() and any(frames_dir.iterdir()):
        raise ValueError(f"Frame output directory must be empty: {frames_dir}")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        image = render_frame(frame, camera, tool_link_meshes)
        add_label(image, frame, args.youngs_modulus).save(frames_dir / f"frame_{index:06d}.png")
        print(f"Rendered {index + 1}/{len(frames)} source={frame.get('source_frame')}", flush=True)
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-framerate", f"{args.fps:.9g}", "-start_number", "0",
        "-i", str(frames_dir / "frame_%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.output_video),
    ], check=True)
    print(f"Wrote {args.output_video}")


if __name__ == "__main__":
    main()
