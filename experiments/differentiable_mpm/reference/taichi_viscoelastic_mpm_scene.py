import argparse
import hashlib
import json
import socket
import struct
import time as wall_time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import taichi as ti
from PIL import Image

try:
    from deformpath_topview import (
        apply_calibration,
        calibration_metadata,
        depth_to_pointcloud as topview_depth_to_pointcloud,
        format_pointcloud,
        load_calibration,
        rasterize_depth,
    )
except ImportError:
    from .deformpath_topview import (
        apply_calibration,
        calibration_metadata,
        depth_to_pointcloud as topview_depth_to_pointcloud,
        format_pointcloud,
        load_calibration,
        rasterize_depth,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "taichi_mpm_camera_views"
SCENE_CENTER = [0.5, 0.32, 0.5]
TOOL_Y = 0.32
TOOL_Z_OFFSET = 0.24
TOOL_TRAVEL = 0.28
DOUGH_RADIUS = [0.14, 0.055, 0.12]


class MpmInvalidStateError(RuntimeError):
    """Raised when a particle cannot safely use the quadratic MPM grid stencil."""

    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__(
            "MPM invalid state: "
            f"{diagnostic['failure_kind']} for particle {diagnostic['particle_index']} "
            f"at substep {diagnostic['substep']}"
        )


UR_TOOL_VISUAL_ORIGIN = np.array([-0.002395874, -0.017992075, -0.019913439], dtype=np.float32)
UR_TOOL_VISUAL_RPY = np.array([4.5910, 1.379415965, -1.740698498], dtype=np.float32)
KINOVA_TOOL_VISUAL_ORIGIN = np.array([-0.02345833, -0.02261066, -0.01297941], dtype=np.float32)
KINOVA_TOOL_VISUAL_RPY = np.array([3.12897712, 0.06996522, -3.11041673], dtype=np.float32)

TOOL_INITIAL_POSES = np.array(
    [
        [SCENE_CENTER[0], TOOL_Y, SCENE_CENTER[2] + TOOL_Z_OFFSET, 0.0, 0.0, 0.0, 1.0],
        [SCENE_CENTER[0], TOOL_Y, SCENE_CENTER[2] - TOOL_Z_OFFSET, 0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def find_tool_mesh(filename: str, override: Path | None = None) -> Path:
    """Use explicit meshes, the sourced ROS package, or bundled standalone assets."""
    if override is not None:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Tool mesh does not exist: {path}")
        return path
    try:
        from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
    except ImportError:
        pass
    else:
        try:
            installed = Path(get_package_share_directory("ur_dual_bringup")) / "meshes" / filename
        except PackageNotFoundError:
            pass
        else:
            if installed.is_file():
                return installed
    bundled = PROJECT_ROOT / "meshes" / filename
    if not bundled.is_file():
        raise FileNotFoundError(f"Tool mesh missing: {bundled}; provide --ur-tool-mesh/--kinova-tool-mesh")
    return bundled


def load_binary_stl(path: Path, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Load an STL as an unindexed triangle mesh for Taichi's scene renderer."""
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path} is too short to be a binary STL file")
    triangle_count = struct.unpack_from("<I", raw, 80)[0]
    expected_size = 84 + triangle_count * 50
    if len(raw) != expected_size:
        raise ValueError(
            f"{path} is not a supported binary STL file "
            f"(expected {expected_size} bytes, found {len(raw)})"
        )
    triangle_dtype = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    triangles = np.frombuffer(raw, dtype=triangle_dtype, count=triangle_count, offset=84)
    vertices = np.array(triangles["vertices"].reshape(-1, 3) * scale, dtype=np.float32)
    indices = np.arange(vertices.shape[0], dtype=np.int32)
    return vertices, indices


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def transformed_tool_mesh(
    vertices: np.ndarray,
    pose: np.ndarray,
    visual_origin: np.ndarray,
    visual_rotation: np.ndarray,
) -> np.ndarray:
    """Apply mesh-to-link URDF origin, then link-to-scene tool pose."""
    local_vertices = vertices @ visual_rotation.T + visual_origin
    return local_vertices @ quaternion_to_matrix(pose[3:]).T + pose[:3]


def mesh_in_tool_frame(vertices: np.ndarray, visual_origin: np.ndarray, visual_rotation: np.ndarray) -> np.ndarray:
    """Apply the URDF visual transform, yielding mesh vertices in tool-link axes."""
    return vertices @ visual_rotation.T + visual_origin


def _mark_triangle_surface(surface: np.ndarray, triangle: np.ndarray, minimum: np.ndarray, spacing: np.ndarray) -> None:
    """Conservatively rasterize one triangle into a local SDF voxel grid."""
    longest_edge = max(
        np.linalg.norm(triangle[1] - triangle[0]),
        np.linalg.norm(triangle[2] - triangle[1]),
        np.linalg.norm(triangle[0] - triangle[2]),
    )
    subdivisions = max(1, int(np.ceil(longest_edge / float(np.min(spacing)))))
    samples = []
    for row in range(subdivisions + 1):
        for column in range(subdivisions + 1 - row):
            u = row / subdivisions
            v = column / subdivisions
            samples.append(triangle[0] + u * (triangle[1] - triangle[0]) + v * (triangle[2] - triangle[0]))
    indices = np.rint((np.asarray(samples) - minimum) / spacing).astype(np.int32)
    indices = np.clip(indices, 0, np.asarray(surface.shape) - 1)
    for offset_x in (-1, 0, 1):
        for offset_y in (-1, 0, 1):
            for offset_z in (-1, 0, 1):
                marked = np.clip(indices + (offset_x, offset_y, offset_z), 0, np.asarray(surface.shape) - 1)
                surface[marked[:, 0], marked[:, 1], marked[:, 2]] = True


def _outside_voxels(surface: np.ndarray) -> np.ndarray:
    """Flood-fill empty boundary voxels; unvisited empty voxels are inside the mesh."""
    outside = np.zeros_like(surface, dtype=bool)
    queue = deque()
    nx, ny, nz = surface.shape
    for x in range(nx):
        for y in range(ny):
            for z in (0, nz - 1):
                if not surface[x, y, z] and not outside[x, y, z]:
                    outside[x, y, z] = True
                    queue.append((x, y, z))
    for x in range(nx):
        for z in range(nz):
            for y in (0, ny - 1):
                if not surface[x, y, z] and not outside[x, y, z]:
                    outside[x, y, z] = True
                    queue.append((x, y, z))
    for y in range(ny):
        for z in range(nz):
            for x in (0, nx - 1):
                if not surface[x, y, z] and not outside[x, y, z]:
                    outside[x, y, z] = True
                    queue.append((x, y, z))
    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            xn, yn, zn = x + dx, y + dy, z + dz
            if 0 <= xn < nx and 0 <= yn < ny and 0 <= zn < nz and not surface[xn, yn, zn] and not outside[xn, yn, zn]:
                outside[xn, yn, zn] = True
                queue.append((xn, yn, zn))
    return outside


def _distance_transform_1d(values: np.ndarray, spacing: float) -> np.ndarray:
    """Exact squared Euclidean distance transform for a one-dimensional line."""
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return values.copy()
    count = len(values)
    sites = np.empty(count, dtype=np.int32)
    boundaries = np.empty(count + 1, dtype=np.float64)
    site_count = 0
    sites[0] = finite[0]
    boundaries[0], boundaries[1] = -np.inf, np.inf
    for point in finite[1:]:
        while True:
            previous = sites[site_count]
            boundary = ((values[point] + (point * spacing) ** 2) - (values[previous] + (previous * spacing) ** 2)) / (2.0 * spacing * (point - previous))
            if boundary > boundaries[site_count]:
                break
            site_count -= 1
            if site_count < 0:
                break
        if site_count < 0:
            site_count = 0
            sites[0] = point
            boundaries[0], boundaries[1] = -np.inf, np.inf
        else:
            site_count += 1
            sites[site_count] = point
            boundaries[site_count] = boundary
            boundaries[site_count + 1] = np.inf
    output = np.empty(count, dtype=np.float64)
    site_index = 0
    for point in range(count):
        while boundaries[site_index + 1] < point * spacing:
            site_index += 1
        source = sites[site_index]
        output[point] = values[source] + ((point - source) * spacing) ** 2
    return output


def _euclidean_distance_to_surface(surface: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    squared = np.where(surface, 0.0, np.inf).astype(np.float64)
    for axis, voxel_size in enumerate(spacing):
        squared = np.moveaxis(squared, axis, 0)
        for index in np.ndindex(squared.shape[1:]):
            squared[(slice(None),) + index] = _distance_transform_1d(squared[(slice(None),) + index], float(voxel_size))
        squared = np.moveaxis(squared, 0, axis)
    return np.sqrt(squared).astype(np.float32)


def build_mesh_sdf(vertices: np.ndarray, resolution: int, padding: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a conservative signed distance field for a closed binary-STL mesh."""
    if resolution < 16:
        raise ValueError("tool_sdf_resolution must be at least 16")
    if len(vertices) == 0 or len(vertices) % 3:
        raise ValueError("Mesh must contain a non-empty sequence of triangle vertices")
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    base_spacing = np.max(upper - lower) / max(resolution - 5, 1)
    margin = max(float(padding), float(base_spacing) * 2.0)
    lower -= margin
    upper += margin
    spacing = (upper - lower) / (resolution - 1)
    surface = np.zeros((resolution, resolution, resolution), dtype=bool)
    for triangle in vertices.reshape(-1, 3, 3):
        _mark_triangle_surface(surface, triangle, lower, spacing)
    if not surface.any():
        raise ValueError("Mesh SDF rasterization produced no surface voxels")
    outside = _outside_voxels(surface)
    distance = _euclidean_distance_to_surface(surface, spacing)
    signed_distance = np.where(outside | surface, distance, -distance).astype(np.float32)
    if not np.any(signed_distance < 0.0):
        raise ValueError(
            "SDF source has no negative interior voxels; it cannot provide signed solid collision. "
            "Use a validated watertight collision mesh rather than a visual STL."
        )
    gradients = np.stack(np.gradient(signed_distance, *spacing, edge_order=1), axis=-1).astype(np.float32)
    gradient_norm = np.linalg.norm(gradients, axis=-1, keepdims=True)
    gradients /= np.maximum(gradient_norm, 1e-8)
    return signed_distance, gradients, lower.astype(np.float32), spacing.astype(np.float32)


def create_mesh_collision_fields(meshes: list[np.ndarray], resolution: int, padding: float):
    """Upload two local-mesh SDFs to Taichi fields used by the MPM kernels."""
    volumes = [build_mesh_sdf(mesh, resolution, padding) for mesh in meshes]
    sdf = ti.field(dtype=ti.f32, shape=(len(volumes), resolution, resolution, resolution))
    gradients = ti.Vector.field(3, dtype=ti.f32, shape=(len(volumes), resolution, resolution, resolution))
    minimums = ti.Vector.field(3, dtype=ti.f32, shape=len(volumes))
    spacings = ti.Vector.field(3, dtype=ti.f32, shape=len(volumes))
    sdf.from_numpy(np.stack([volume[0] for volume in volumes]))
    gradients.from_numpy(np.stack([volume[1] for volume in volumes]))
    minimums.from_numpy(np.stack([volume[2] for volume in volumes]))
    spacings.from_numpy(np.stack([volume[3] for volume in volumes]))
    statistics = []
    for index, (distance, _gradient, minimum, spacing) in enumerate(volumes):
        stats = {
            "negative_voxels": int(np.count_nonzero(distance < 0.0)),
            "zero_voxels": int(np.count_nonzero(distance == 0.0)),
            "positive_voxels": int(np.count_nonzero(distance > 0.0)),
            "minimum_signed_distance_m": float(distance.min()),
            "maximum_signed_distance_m": float(distance.max()),
            "minimum_scene_m": minimum.tolist(),
            "maximum_scene_m": (minimum + spacing * (resolution - 1)).tolist(),
            "voxel_spacing_scene_m": spacing.tolist(),
        }
        statistics.append(stats)
        print(
            f"Tool SDF {index}: {resolution}^3, bounds={stats['minimum_scene_m']} to "
            f"{stats['maximum_scene_m']}, range={distance.min():.4f} to {distance.max():.4f} m, "
            f"negative_voxels={stats['negative_voxels']}"
        )
    return SimpleNamespace(sdf=sdf, gradients=gradients, minimums=minimums, spacings=spacings, resolution=resolution, statistics=statistics)


CAMERA_VIEWS = {
    "front_dough": {
        "position": [0.5, 0.38, 1.25],
        "lookAt": SCENE_CENTER,
        "fieldOfView": 57.0,
        "zNear": 0.01,
        "zFar": 2.0,
    },
    "top_dough": {
        "position": [0.5, 0.8, 0.5],
        "lookAt": SCENE_CENTER,
        "fieldOfView": 57.0,
        "zNear": 0.01,
        "zFar": 2.0,
    },
    "deformpath_top": {
        "position": [0.5, 0.8, 0.5],
        "lookAt": SCENE_CENTER,
        "up": [0.0, 0.0, -1.0],
        "fieldOfView": 57.0,
        "zNear": 0.01,
        "zFar": 2.0,
    },
    "side_dough": {
        "position": [1.25, 0.36, 0.5],
        "lookAt": SCENE_CENTER,
        "fieldOfView": 57.0,
        "zNear": 0.01,
        "zFar": 2.0,
    },
}


def _load_torch_tensor(path):
    try:
        import torch
    except ImportError as exc:
        raise ImportError(f"Loading {path} requires torch because it is not a .npy file.") from exc

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("sampled_particles_xyz", "particles", "points", "pointcloud", "current_frame"):
            if key in obj:
                obj = obj[key]
                break
    if hasattr(obj, "detach"):
        obj = obj.detach().cpu().numpy()
    return np.asarray(obj)


def load_initial_particles(path):
    path = Path(path)
    if path.suffix == ".npy":
        points = np.load(path)
    else:
        points = _load_torch_tensor(path)

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"{path} must contain an array/tensor with shape [N, >=3], got {points.shape}.")

    points = points[:, :3]
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        raise ValueError(f"{path} did not contain any finite XYZ points.")
    return points


def particle_array_sha256(points):
    array = np.ascontiguousarray(np.asarray(points, dtype=np.float32))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _rigid_matrix(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0.0):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7, rtol=0.0):
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix


def load_tool_geometry(path):
    geometry_path = Path(path)
    raw = geometry_path.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid tool geometry JSON in {geometry_path}") from exc
    if not isinstance(document, dict) or document.get("schema") != "taichidough/tool-geometry/v1":
        raise ValueError("Tool geometry must use schema taichidough/tool-geometry/v1")
    tools = document.get("tools")
    if not isinstance(tools, list) or len(tools) != 2:
        raise ValueError("Tool geometry must contain exactly two tools")

    names = []
    half_extents = []
    marker_from_collider = []
    marker_from_mesh = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tools[{index}] must be an object")
        name = tool.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"tools[{index}].name must be a nonempty string")
        extents = np.asarray(tool.get("half_extents_m"), dtype=np.float64)
        if extents.shape != (3,) or not np.isfinite(extents).all() or np.any(extents <= 0.0):
            raise ValueError(f"tools[{index}].half_extents_m must contain three positive finite values")
        collider_transform = _rigid_matrix(tool.get("marker_from_collider"), f"tools[{index}].marker_from_collider")
        mesh_transform = _rigid_matrix(tool.get("marker_from_mesh", np.eye(4)), f"tools[{index}].marker_from_mesh")
        names.append(name.strip())
        half_extents.append(extents.tolist())
        marker_from_collider.append(collider_transform.tolist())
        marker_from_mesh.append(mesh_transform.tolist())
    if len(set(names)) != 2:
        raise ValueError("Tool geometry names must be unique")
    return {
        "schema": "taichidough/tool-geometry/v1",
        "names": names,
        "half_extents_m": half_extents,
        "marker_from_collider": marker_from_collider,
        "marker_from_mesh": marker_from_mesh,
        "fingerprint": hashlib.sha256(raw).hexdigest(),
        "source": str(geometry_path.resolve()),
        "proxy": bool(document.get("proxy", False)),
    }


def legacy_tool_geometry(half_extents, marker_offset):
    extents = np.asarray(half_extents, dtype=np.float64)
    offset = np.asarray(marker_offset, dtype=np.float64)
    if extents.shape != (3,) or not np.isfinite(extents).all() or np.any(extents <= 0.0):
        raise ValueError("Tool half-extents must contain three positive finite values")
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Tool marker offset must contain three finite values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = offset
    result = {
        "schema": "taichidough/tool-geometry/v1",
        "names": ["tool_0", "tool_1"],
        "half_extents_m": [extents.tolist(), extents.tolist()],
        "marker_from_collider": [transform.tolist(), transform.tolist()],
        "marker_from_mesh": [np.eye(4).tolist(), np.eye(4).tolist()],
        "source": "legacy_cli",
        "proxy": True,
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["fingerprint"] = hashlib.sha256(payload).hexdigest()
    return result


def resolve_tool_geometry(geometry_path=None, half_extents=None, marker_offset=None):
    if geometry_path is not None:
        if half_extents is not None or marker_offset is not None:
            raise ValueError("--tool-geometry cannot be combined with --tool-half-extents or --tool-marker-offset")
        return load_tool_geometry(geometry_path)
    return legacy_tool_geometry(
        (0.05, 0.05, 0.05) if half_extents is None else half_extents,
        (0.0, 0.0, 0.0) if marker_offset is None else marker_offset,
    )


def align_tool_geometry(tool_geometry, expected_names):
    names = [str(name) for name in expected_names]
    if len(names) != 2 or len(set(names)) != 2 or any(not name for name in names):
        raise ValueError("Captured replay must contain two uniquely named tool streams")
    result = dict(tool_geometry)
    raw_names = list(result.get("names", []))
    if result.get("source") == "legacy_cli":
        order = [0, 1]
    else:
        if len(raw_names) != 2 or set(raw_names) != set(names):
            raise ValueError(f"Tool geometry names {raw_names} do not match pose streams {names}")
        order = [raw_names.index(name) for name in names]
    result["names"] = names
    result["half_extents_m"] = [result["half_extents_m"][index] for index in order]
    result["marker_from_collider"] = [result["marker_from_collider"][index] for index in order]
    result["marker_from_mesh"] = [result["marker_from_mesh"][index] for index in order]
    if result.get("source") == "legacy_cli":
        fingerprint_payload = {key: value for key, value in result.items() if key != "fingerprint"}
        payload = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
        result["fingerprint"] = hashlib.sha256(payload).hexdigest()
    return result


def replay_marker_transforms(tool_geometry, collision_mode):
    transform_key = "marker_from_mesh" if collision_mode == "sdf" else "marker_from_collider"
    if collision_mode not in {"box", "sdf", "none"}:
        raise ValueError(f"Unsupported tool collision mode: {collision_mode}")
    transforms = np.asarray(tool_geometry.get(transform_key), dtype=np.float64)
    if transforms.shape != (2, 4, 4):
        raise ValueError(f"Tool geometry {transform_key} must have shape [2, 4, 4]")
    return np.stack([
        _rigid_matrix(transform, f"{transform_key}[{index}]") for index, transform in enumerate(transforms)
    ])


def resolve_sim_tool_half_extents(args):
    legacy = getattr(args, "tool_half_extents", None)
    if legacy is None:
        legacy = (0.05, 0.05, 0.05)
    by_tool = getattr(args, "tool_half_extents_by_tool", None)
    if by_tool is None:
        by_tool = [legacy, legacy]
    values = np.asarray(by_tool, dtype=np.float64)
    if values.shape != (2, 3) or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("Simulator tool half-extents must contain two positive finite XYZ vectors")
    return tuple(tuple(float(value) for value in row) for row in values)


def _positive_finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def normalize_plane(plane):
    if isinstance(plane, dict):
        if "coefficients" in plane:
            values = plane["coefficients"]
        elif "normal" in plane:
            normal = np.asarray(plane["normal"], dtype=np.float64)
            if "point" in plane:
                point = np.asarray(plane["point"], dtype=np.float64)
                values = [*normal, -float(np.dot(normal, point))]
            else:
                offset = plane.get("offset", plane.get("d", plane.get("distance")))
                if offset is None:
                    raise ValueError("Floor plane with a normal requires offset, d, distance, or point")
                values = [*normal, offset]
        else:
            values = [plane.get(key) for key in ("a", "b", "c", "d")]
    else:
        values = plane
    coefficients = np.asarray(values, dtype=np.float64)
    if coefficients.shape != (4,) or not np.isfinite(coefficients).all():
        raise ValueError("Floor plane must contain four finite coefficients")
    normal_norm = float(np.linalg.norm(coefficients[:3]))
    if normal_norm <= 1e-12:
        raise ValueError("Floor plane normal must be nonzero")
    return coefficients / normal_norm


def floor_y_from_plane(plane, tilt_tolerance=1e-6):
    normalized = normalize_plane(plane)
    if abs(normalized[0]) > tilt_tolerance or abs(normalized[2]) > tilt_tolerance:
        raise ValueError("The reconstructed floor plane is tilted and cannot be used by the constant-Y collider")
    if abs(abs(normalized[1]) - 1.0) > tilt_tolerance:
        raise ValueError("The reconstructed floor plane is incompatible with the constant-Y collider")
    floor_y = -normalized[3] / normalized[1]
    if not np.isfinite(floor_y):
        raise ValueError("The reconstructed floor height is not finite")
    return float(floor_y)


def compute_mass_properties(
    particle_count,
    grid_size,
    density,
    object_volume_m3=None,
    object_mass_kg=None,
):
    if int(particle_count) != particle_count or particle_count <= 0:
        raise ValueError("Particle count must be a positive integer")
    if int(grid_size) != grid_size or grid_size <= 0:
        raise ValueError("Grid size must be a positive integer")
    density = _positive_finite(density, "Density")

    if object_volume_m3 is None:
        if object_mass_kg is not None:
            raise ValueError("--object-mass-kg requires reconstructed object volume metadata")
        particle_volume = (0.5 / float(grid_size)) ** 3
        total_volume = particle_volume * int(particle_count)
        total_mass = density * total_volume
        mass_source = "grid_derived_particle_volume_and_density"
    else:
        total_volume = _positive_finite(object_volume_m3, "Object volume")
        if object_mass_kg is None:
            total_mass = density * total_volume
            mass_source = "reconstructed_volume_and_density"
        else:
            total_mass = _positive_finite(object_mass_kg, "Object mass")
            density = total_mass / total_volume
            mass_source = "reconstructed_volume_and_measured_mass"
        particle_volume = total_volume / int(particle_count)

    particle_mass = total_mass / int(particle_count)
    for value, name in (
        (total_volume, "Object volume"),
        (density, "Density"),
        (total_mass, "Total mass"),
        (particle_volume, "Particle volume"),
        (particle_mass, "Particle mass"),
    ):
        _positive_finite(value, name)
    return {
        "mass_source": mass_source,
        "object_volume_m3": float(total_volume),
        "density_kg_m3": float(density),
        "total_mass_kg": float(total_mass),
        "particle_volume_m3": float(particle_volume),
        "particle_mass_kg": float(particle_mass),
        "particle_count": int(particle_count),
        "grid_size": int(grid_size),
    }


def validate_reconstructed_particle_options(axis_map=None, fit=None, scale=None, offset=None):
    if axis_map not in (None, "xyz"):
        raise ValueError("Reconstructed scene-coordinate particles cannot use an axis-map transform")
    if fit not in (None, "none"):
        raise ValueError("Reconstructed scene-coordinate particles cannot use a fitting transform")
    if scale is not None and (not np.isfinite(scale) or not np.isclose(scale, 1.0)):
        raise ValueError("Reconstructed scene-coordinate particles cannot use an additional scale")
    if offset is not None:
        offset_array = np.asarray(offset, dtype=np.float64)
        if offset_array.shape != (3,) or not np.isfinite(offset_array).all() or not np.allclose(offset_array, 0.0):
            raise ValueError("Reconstructed scene-coordinate particles cannot use an additional offset")


def _metadata_output_matches(metadata_path, recorded_path, actual_path):
    if not recorded_path:
        return True
    recorded = Path(recorded_path).expanduser()
    candidates = [recorded.resolve()]
    if not recorded.is_absolute():
        candidates.append((Path(metadata_path).resolve().parent / recorded).resolve())
    actual = Path(actual_path).resolve()
    return actual in candidates


def load_reconstruction_metadata(metadata_path, particles_path, loaded_particles, calibration=None):
    metadata_path = Path(metadata_path)
    raw = metadata_path.read_bytes()
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid reconstruction metadata JSON in {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Reconstruction metadata must contain a JSON object")
    schema = str(metadata.get("schema", ""))
    if schema != "voxel_dough_reconstruction/v2":
        raise ValueError("Scene-coordinate initialization requires voxel_dough_reconstruction/v2 metadata")

    object_volume = _positive_finite(metadata.get("object_volume_m3"), "Reconstructed object volume")
    voxel_size = _positive_finite(metadata.get("voxel_size"), "Reconstruction voxel size")
    voxel_count = metadata.get("voxel_count", metadata.get("voxel_centers"))
    if isinstance(voxel_count, bool) or not isinstance(voxel_count, int) or voxel_count <= 0:
        raise ValueError("Reconstruction voxel count must be a positive integer")
    expected_volume = voxel_count * voxel_size**3
    if not np.isclose(object_volume, expected_volume, rtol=1e-9, atol=1e-15):
        raise ValueError("Reconstructed object volume is inconsistent with voxel count and voxel size")
    sampled_count = metadata.get("sampled_particles")
    if sampled_count is not None and sampled_count != len(loaded_particles):
        raise ValueError("Initial particle count does not match reconstruction metadata")

    fill_info = metadata.get("fill") or {}
    if fill_info.get("mode") != "floor":
        raise ValueError("Scene-coordinate volume initialization requires floor-mode reconstruction metadata")
    floor_plane = fill_info.get("floor_plane_scene", metadata.get("floor_plane_scene"))
    if floor_plane is None:
        raise ValueError("Reconstruction metadata does not contain a scene floor plane")
    normalized_plane = normalize_plane(floor_plane)

    calibration_info = metadata.get("calibration") or {}
    calibration_schema = metadata.get("calibration_schema") or calibration_info.get("schema")
    schema_text = str(calibration_schema).lower()
    if not (calibration_info.get("is_metric") is True and schema_text.endswith("/v2")):
        raise ValueError("Reconstruction metadata must identify a metric v2 calibration")
    scene_frame = calibration_info.get("scene_frame")
    particle_frame = (metadata.get("array_frames") or {}).get("sampled_particles_xyz")
    if not scene_frame:
        scene_frame = particle_frame
    if not particle_frame or particle_frame != scene_frame:
        raise ValueError("Sampled particles are not identified as scene-coordinate data")

    recorded_output = (metadata.get("outputs") or {}).get("sampled_particles_xyz")
    if recorded_output and not _metadata_output_matches(metadata_path, recorded_output, particles_path):
        raise ValueError("Initial particle path does not match reconstruction metadata")
    expected_hash = metadata.get("sampled_particles_sha256")
    actual_hash = particle_array_sha256(loaded_particles)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("Initial particle SHA-256 does not match reconstruction metadata")

    calibration_fingerprint = metadata.get("calibration_fingerprint") or calibration_info.get("fingerprint")
    if calibration is not None and calibration_fingerprint:
        if getattr(calibration, "fingerprint", None) != calibration_fingerprint:
            raise ValueError("Initial particle calibration fingerprint does not match reconstruction metadata")
    return {
        "metadata": metadata,
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "particle_sha256": actual_hash,
        "object_volume_m3": object_volume,
        "floor_plane_scene": normalized_plane.tolist(),
        "floor_y": floor_y_from_plane(normalized_plane),
        "scene_frame": scene_frame,
        "calibration_fingerprint": calibration_fingerprint,
    }


def parse_axis_map(axis_map):
    compact = axis_map.replace(",", "").replace(" ", "").lower()
    result = []
    i = 0
    while i < len(compact):
        sign = 1.0
        if compact[i] == "-":
            sign = -1.0
            i += 1
        elif compact[i] == "+":
            i += 1
        if i >= len(compact) or compact[i] not in "xyz":
            raise ValueError(f"Invalid --initial-particles-axis-map '{axis_map}'. Use values like xyz, xzy, x-z-y.")
        result.append((sign, "xyz".index(compact[i])))
        i += 1

    if len(result) != 3:
        raise ValueError(f"Invalid --initial-particles-axis-map '{axis_map}'. It must define three destination axes.")
    if sorted(index for _, index in result) != [0, 1, 2]:
        raise ValueError(f"Invalid --initial-particles-axis-map '{axis_map}'. Each source axis must be used once.")
    return result


def map_initial_particle_axes(points, axis_map):
    mapping = parse_axis_map(axis_map)
    mapped = np.empty_like(points[:, :3], dtype=np.float32)
    for dest_axis, (sign, src_axis) in enumerate(mapping):
        mapped[:, dest_axis] = sign * points[:, src_axis]
    return mapped


def fit_initial_particles_to_scene(points, args, calibration=None):
    if calibration is not None:
        if args.initial_particles_fit != "none" or args.initial_particles_raw_scene_coordinates:
            raise ValueError(
                "--initial-particles-calibration requires --initial-particles-fit none and cannot use "
                "--initial-particles-raw-scene-coordinates."
            )
        mapped = apply_calibration(points, calibration)
        if not np.isfinite(mapped).all() or np.any(mapped <= 0.0) or np.any(mapped >= 1.0):
            raise ValueError("Calibrated initial particles must be finite and strictly inside the unit Taichi scene.")
        return mapped.astype(np.float32, copy=False)

    points = map_initial_particle_axes(points, args.initial_particles_axis_map)
    if args.initial_particles_raw_scene_coordinates:
        return points.astype(np.float32, copy=False)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    source_center = 0.5 * (mins + maxs)
    centered = points - source_center
    half_extent = np.maximum(0.5 * (maxs - mins), 1e-6)
    target_half_extent = np.asarray(DOUGH_RADIUS, dtype=np.float32)

    if args.initial_particles_fit == "anisotropic":
        scale = target_half_extent / half_extent
    elif args.initial_particles_fit == "isotropic":
        scale = np.full(3, np.min(target_half_extent / half_extent), dtype=np.float32)
    else:
        scale = np.ones(3, dtype=np.float32)

    offset = np.asarray(args.initial_particles_offset, dtype=np.float32)
    scene_center = np.asarray(SCENE_CENTER, dtype=np.float32) + offset
    return (scene_center + centered * scale * args.initial_particles_scale).astype(np.float32)


def mpm_grid_stencil_is_safe(position, grid_size):
    """Return whether the existing Taichi integer stencil base stays in the grid."""
    values = np.asarray(position, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all() or int(grid_size) < 3:
        return False
    base = np.trunc(values * int(grid_size) - 0.5).astype(np.int64)
    return bool(np.all(base >= 0) and np.all(base + 2 < int(grid_size)))


def resize_initial_particles(points, count, seed):
    if count == len(points):
        return points.astype(np.float32, copy=False)

    rng = np.random.default_rng(seed)
    if count < len(points):
        indices = rng.choice(len(points), size=count, replace=False)
    else:
        indices = rng.choice(len(points), size=count, replace=True)
    return points[indices].astype(np.float32, copy=False)


def build_sim(args, mesh_collision=None, return_sdf_contact_diagnostics=False):
    dim = 3
    n_particles = args.particles
    n_grid = args.grid
    tool_vis_resolution = 14
    tool_vis_count = 2 * tool_vis_resolution * tool_vis_resolution * tool_vis_resolution
    dx = 1.0 / n_grid
    inv_dx = float(n_grid)
    dt = args.dt

    mass_properties = getattr(args, "mass_properties", None)
    if mass_properties is None:
        mass_properties = compute_mass_properties(n_particles, n_grid, args.density)
    p_vol = mass_properties["particle_volume_m3"]
    p_mass = mass_properties["particle_mass_kg"]
    E = args.youngs_modulus
    nu = args.poisson_ratio
    mu_0 = E / (2 * (1 + nu))
    lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))
    viscosity = args.viscosity
    gravity = args.gravity
    floor_y = args.floor_y
    tool_close_time = args.tool_close_time
    tool_motion_start = args.tool_motion_start
    tool_contact_padding = args.tool_contact_padding
    tool_contact_friction = args.tool_contact_friction
    tool_contact_absorption = args.tool_contact_absorption
    tool_stickiness = args.tool_stickiness
    floor_friction = args.floor_friction
    floor_stickiness = args.floor_stickiness
    floor_absorption = args.floor_absorption
    floor_plastic_damping_band = args.floor_plastic_damping_band
    velocity_damping = args.velocity_damping
    pure_viscoelastic = 1 if args.pure_viscoelastic else 0
    plastic_min = args.plastic_min
    plastic_max = args.plastic_max
    plastic_velocity_damping = args.plastic_velocity_damping
    plastic_affine_damping = args.plastic_affine_damping
    use_jp = 0 if not args.use_jp else 1
    jp_hardening = args.jp_hardening
    jp_min = args.jp_min
    jp_max = args.jp_max
    scripted_tools = 0 if args.ros_control or getattr(args, "replay_episode", None) else 1
    tool_half_by_tool = resolve_sim_tool_half_extents(args)
    tool_half_0 = tool_half_by_tool[0]
    tool_half_1 = tool_half_by_tool[1]
    collision_mode = getattr(args, "tool_collision", "box")
    record_sdf_contact_diagnostics = bool(getattr(args, "record_sdf_contact_diagnostics", False))
    if collision_mode not in {"sdf", "box", "none"}:
        raise ValueError(f"Unsupported tool collision mode: {collision_mode}")
    if collision_mode == "sdf" and mesh_collision is None:
        raise ValueError("SDF collision mode requires prebuilt mesh collision fields")
    use_mesh_collision = collision_mode == "sdf"
    use_box_collision = collision_mode == "box"
    scene_center_x = SCENE_CENTER[0]
    scene_center_y = SCENE_CENTER[1]
    scene_center_z = SCENE_CENTER[2]
    dough_radius_x = DOUGH_RADIUS[0]
    dough_radius_y = DOUGH_RADIUS[1]
    dough_radius_z = DOUGH_RADIUS[2]
    tool_y = TOOL_Y
    tool_z_offset = TOOL_Z_OFFSET
    tool_travel = TOOL_TRAVEL

    x = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
    v = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
    C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
    F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
    Jp = ti.field(dtype=ti.f32, shape=n_particles)
    yielded = ti.field(dtype=ti.i32, shape=n_particles)
    grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
    grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
    tool_x = ti.Vector.field(dim, dtype=ti.f32, shape=tool_vis_count)
    tool_center = ti.Vector.field(dim, dtype=ti.f32, shape=2)
    tool_quat = ti.Vector.field(4, dtype=ti.f32, shape=2)
    tool_velocity = ti.Vector.field(dim, dtype=ti.f32, shape=2)
    tool_angular_velocity = ti.Vector.field(dim, dtype=ti.f32, shape=2)
    invalid_pre_p2g_state = ti.field(dtype=ti.i32, shape=())
    invalid_pre_p2g_stencil = ti.field(dtype=ti.i32, shape=())
    invalid_post_g2p_state = ti.field(dtype=ti.i32, shape=())
    invalid_post_g2p_stencil = ti.field(dtype=ti.i32, shape=())
    sdf_grid_evaluated = ti.field(dtype=ti.i32, shape=2)
    sdf_grid_valid = ti.field(dtype=ti.i32, shape=2)
    sdf_grid_invalid = ti.field(dtype=ti.i32, shape=2)
    sdf_grid_min_distance = ti.field(dtype=ti.f32, shape=2)
    sdf_grid_inside = ti.field(dtype=ti.i32, shape=2)
    sdf_grid_candidate = ti.field(dtype=ti.i32, shape=2)
    sdf_grid_applied = ti.field(dtype=ti.i32, shape=2)
    sdf_grid_inward_removed = ti.field(dtype=ti.i32, shape=2)
    sdf_particle_evaluated = ti.field(dtype=ti.i32, shape=2)
    sdf_particle_valid = ti.field(dtype=ti.i32, shape=2)
    sdf_particle_invalid = ti.field(dtype=ti.i32, shape=2)
    sdf_particle_min_distance = ti.field(dtype=ti.f32, shape=2)
    sdf_particle_inside = ti.field(dtype=ti.i32, shape=2)
    sdf_particle_candidate = ti.field(dtype=ti.i32, shape=2)
    sdf_particle_applied = ti.field(dtype=ti.i32, shape=2)
    sdf_particle_inward_removed = ti.field(dtype=ti.i32, shape=2)

    @ti.func
    def quat_to_matrix(q):
        xq, yq, zq, wq = q[0], q[1], q[2], q[3]
        return ti.Matrix([
            [1.0 - 2.0 * (yq * yq + zq * zq), 2.0 * (xq * yq - zq * wq), 2.0 * (xq * zq + yq * wq)],
            [2.0 * (xq * yq + zq * wq), 1.0 - 2.0 * (xq * xq + zq * zq), 2.0 * (yq * zq - xq * wq)],
            [2.0 * (xq * zq - yq * wq), 2.0 * (yq * zq + xq * wq), 1.0 - 2.0 * (xq * xq + yq * yq)],
        ])

    @ti.kernel
    def set_tool_state(poses: ti.types.ndarray(), velocities: ti.types.ndarray()):
        for i in range(2):
            tool_center[i] = ti.Vector([poses[i, 0], poses[i, 1], poses[i, 2]])
            tool_quat[i] = ti.Vector([poses[i, 3], poses[i, 4], poses[i, 5], poses[i, 6]])
            tool_velocity[i] = ti.Vector([velocities[i, 0], velocities[i, 1], velocities[i, 2]])
            tool_angular_velocity[i] = ti.Vector([0.0, 0.0, 0.0])
            if velocities.shape[1] >= 6:
                tool_angular_velocity[i] = ti.Vector([velocities[i, 3], velocities[i, 4], velocities[i, 5]])

    @ti.func
    def tool_pose_and_velocity(tool_id, t):
        motion_time = ti.max(t - tool_motion_start, 0.0)
        progress = 0.0
        if t >= tool_motion_start:
            progress = ti.min(motion_time / tool_close_time, 1.0)

        center = tool_center[tool_id]
        cvel = tool_velocity[tool_id]
        if scripted_tools == 1:
            if tool_id == 0:
                center = ti.Vector([scene_center_x, tool_y, scene_center_z + tool_z_offset - tool_travel * progress])
                if t >= tool_motion_start and progress < 1.0:
                    cvel = ti.Vector([0.0, 0.0, -tool_travel / tool_close_time])
                else:
                    cvel = ti.Vector([0.0, 0.0, 0.0])
            else:
                center = ti.Vector([scene_center_x, tool_y, scene_center_z - tool_z_offset + tool_travel * progress])
                if t >= tool_motion_start and progress < 1.0:
                    cvel = ti.Vector([0.0, 0.0, tool_travel / tool_close_time])
                else:
                    cvel = ti.Vector([0.0, 0.0, 0.0])

        return center, cvel

    @ti.func
    def box_collision_velocity_and_normal(pos, t):
        hit = 0
        normal = ti.Vector([0.0, 0.0, 0.0])
        collider_v = ti.Vector([0.0, 0.0, 0.0])

        for k in ti.static(range(2)):
            inflated_half = ti.Vector([tool_half_0[0], tool_half_0[1], tool_half_0[2]]) + tool_contact_padding
            if ti.static(k == 1):
                inflated_half = ti.Vector([tool_half_1[0], tool_half_1[1], tool_half_1[2]]) + tool_contact_padding
            center, cvel = tool_pose_and_velocity(k, t)
            rot = quat_to_matrix(tool_quat[k])
            q = rot.transpose() @ (pos - center)
            aq = ti.abs(q)
            inside = aq.x < inflated_half.x and aq.y < inflated_half.y and aq.z < inflated_half.z
            if inside:
                penetration = inflated_half - aq
                min_pen = penetration.x
                local_normal = ti.Vector([1.0, 0.0, 0.0])
                if q.x < 0.0:
                    local_normal = ti.Vector([-1.0, 0.0, 0.0])

                if penetration.y < min_pen:
                    min_pen = penetration.y
                    local_normal = ti.Vector([0.0, 1.0, 0.0])
                    if q.y < 0.0:
                        local_normal = ti.Vector([0.0, -1.0, 0.0])

                if penetration.z < min_pen:
                    local_normal = ti.Vector([0.0, 0.0, 1.0])
                    if q.z < 0.0:
                        local_normal = ti.Vector([0.0, 0.0, -1.0])

                hit = 1
                normal = rot @ local_normal
                collider_v = cvel + tool_angular_velocity[k].cross(pos - center)

        return hit, normal, collider_v

    @ti.func
    def project_particle_out_of_tools(pos, vel, t):
        new_pos = pos
        new_vel = vel

        for k in ti.static(range(2)):
            inflated_half = ti.Vector([tool_half_0[0], tool_half_0[1], tool_half_0[2]]) + tool_contact_padding
            if ti.static(k == 1):
                inflated_half = ti.Vector([tool_half_1[0], tool_half_1[1], tool_half_1[2]]) + tool_contact_padding
            center, cvel = tool_pose_and_velocity(k, t)
            rot = quat_to_matrix(tool_quat[k])
            q = rot.transpose() @ (new_pos - center)
            aq = ti.abs(q)
            inside = aq.x < inflated_half.x and aq.y < inflated_half.y and aq.z < inflated_half.z
            if inside:
                penetration = inflated_half - aq
                min_pen = penetration.x
                if penetration.y < min_pen:
                    min_pen = penetration.y
                if penetration.z < min_pen:
                    min_pen = penetration.z

                local_normal = ti.Vector([0.0, 0.0, 0.0])
                if min_pen == penetration.x:
                    if q.x < 0.0:
                        q.x = -(inflated_half.x + 1e-4)
                        local_normal = ti.Vector([-1.0, 0.0, 0.0])
                    else:
                        q.x = inflated_half.x + 1e-4
                        local_normal = ti.Vector([1.0, 0.0, 0.0])
                elif min_pen == penetration.y:
                    if q.y < 0.0:
                        q.y = -(inflated_half.y + 1e-4)
                        local_normal = ti.Vector([0.0, -1.0, 0.0])
                    else:
                        q.y = inflated_half.y + 1e-4
                        local_normal = ti.Vector([0.0, 1.0, 0.0])
                else:
                    if q.z < 0.0:
                        q.z = -(inflated_half.z + 1e-4)
                        local_normal = ti.Vector([0.0, 0.0, -1.0])
                    else:
                        q.z = inflated_half.z + 1e-4
                        local_normal = ti.Vector([0.0, 0.0, 1.0])

                normal = rot @ local_normal
                new_pos = center + rot @ q

                cvel += tool_angular_velocity[k].cross(new_pos - center)
                rel_v = new_vel - cvel
                vn = rel_v.dot(normal)
                if vn < 0.0:
                    rel_v -= normal * vn
                rel_v *= tool_contact_friction * (1.0 - tool_contact_absorption)
                rel_v *= 1.0 - tool_stickiness
                new_vel = cvel + rel_v

        return new_pos, new_vel

    @ti.func
    def mesh_sdf_at(tool_id, local_pos):
        """Trilinearly sample one tool's local signed-distance volume."""
        grid_pos = (local_pos - mesh_collision.minimums[tool_id]) / mesh_collision.spacings[tool_id]
        valid = (
            grid_pos.x >= 0.0 and grid_pos.x <= mesh_collision.resolution - 1 and
            grid_pos.y >= 0.0 and grid_pos.y <= mesh_collision.resolution - 1 and
            grid_pos.z >= 0.0 and grid_pos.z <= mesh_collision.resolution - 1
        )
        distance = 1e6
        gradient = ti.Vector([0.0, 0.0, 0.0])
        if valid:
            distance = 0.0
            maximum_base = mesh_collision.resolution - 2
            ix = ti.max(0, ti.min(maximum_base, ti.cast(ti.floor(grid_pos.x), ti.i32)))
            iy = ti.max(0, ti.min(maximum_base, ti.cast(ti.floor(grid_pos.y), ti.i32)))
            iz = ti.max(0, ti.min(maximum_base, ti.cast(ti.floor(grid_pos.z), ti.i32)))
            fx = ti.max(0.0, ti.min(1.0, grid_pos.x - ti.cast(ix, ti.f32)))
            fy = ti.max(0.0, ti.min(1.0, grid_pos.y - ti.cast(iy, ti.f32)))
            fz = ti.max(0.0, ti.min(1.0, grid_pos.z - ti.cast(iz, ti.f32)))
            for dx, dy, dz in ti.static(ti.ndrange(2, 2, 2)):
                wx = fx if dx == 1 else 1.0 - fx
                wy = fy if dy == 1 else 1.0 - fy
                wz = fz if dz == 1 else 1.0 - fz
                weight = wx * wy * wz
                index = ti.Vector([ix + dx, iy + dy, iz + dz])
                distance += weight * mesh_collision.sdf[tool_id, index.x, index.y, index.z]
                gradient += weight * mesh_collision.gradients[tool_id, index.x, index.y, index.z]
            gradient /= ti.max(gradient.norm(), 1e-6)
        return valid, distance, gradient

    @ti.func
    def mesh_collision_velocity_and_normal(pos, t):
        hit = 0
        selected_tool = -1
        normal = ti.Vector([0.0, 0.0, 0.0])
        collider_v = ti.Vector([0.0, 0.0, 0.0])
        closest_distance = 1e6
        for k in ti.static(range(2)):
            center, cvel = tool_pose_and_velocity(k, t)
            rotation = quat_to_matrix(tool_quat[k])
            valid, distance, local_normal = mesh_sdf_at(k, rotation.transpose() @ (pos - center))
            if valid and distance < tool_contact_padding:
                if ti.static(record_sdf_contact_diagnostics):
                    ti.atomic_add(sdf_grid_candidate[k], 1)
                if distance < closest_distance:
                    hit = 1
                    selected_tool = k
                    closest_distance = distance
                    normal = rotation @ local_normal
                    collider_v = cvel + tool_angular_velocity[k].cross(pos - center)
        return hit, selected_tool, normal, collider_v

    @ti.func
    def project_particle_out_of_meshes(pos, vel, t):
        new_pos = pos
        new_vel = vel
        for k in ti.static(range(2)):
            center, cvel = tool_pose_and_velocity(k, t)
            rotation = quat_to_matrix(tool_quat[k])
            local_pos = rotation.transpose() @ (new_pos - center)
            valid, distance, local_normal = mesh_sdf_at(k, local_pos)
            if valid and distance < tool_contact_padding:
                if ti.static(record_sdf_contact_diagnostics):
                    ti.atomic_add(sdf_particle_candidate[k], 1)
                    ti.atomic_add(sdf_particle_applied[k], 1)
                penetration = tool_contact_padding - distance + 1e-4
                normal = rotation @ local_normal
                new_pos += normal * penetration
                cvel += tool_angular_velocity[k].cross(new_pos - center)
                relative_velocity = new_vel - cvel
                normal_velocity = relative_velocity.dot(normal)
                if normal_velocity < 0.0:
                    relative_velocity -= normal * normal_velocity
                    if ti.static(record_sdf_contact_diagnostics):
                        ti.atomic_add(sdf_particle_inward_removed[k], 1)
                relative_velocity *= tool_contact_friction * (1.0 - tool_contact_absorption)
                relative_velocity *= 1.0 - tool_stickiness
                new_vel = cvel + relative_velocity
        return new_pos, new_vel

    @ti.kernel
    def initialize():
        for i in range(n_particles):
            # Random points in a dough-like ellipsoid centered in the unit scene.
            p = ti.Vector([0.0, 0.0, 0.0])
            accepted = False
            for _ in range(32):
                candidate = ti.Vector([
                    ti.random(ti.f32) * 2.0 - 1.0,
                    ti.random(ti.f32) * 2.0 - 1.0,
                    ti.random(ti.f32) * 2.0 - 1.0,
                ])
                if candidate.dot(candidate) <= 1.0 and not accepted:
                    p = candidate
                    accepted = True
            if not accepted:
                p = ti.Vector([0.0, 0.0, 0.0])

            x[i] = ti.Vector([scene_center_x, scene_center_y, scene_center_z]) + p * ti.Vector([
                dough_radius_x,
                dough_radius_y,
                dough_radius_z,
            ])
            v[i] = ti.Vector([0.0, 0.0, 0.0])
            C[i] = ti.Matrix.zero(ti.f32, dim, dim)
            F[i] = ti.Matrix.identity(ti.f32, dim)
            Jp[i] = 1.0
            yielded[i] = 0

        tool_center[0] = ti.Vector([scene_center_x, tool_y, scene_center_z + tool_z_offset])
        tool_center[1] = ti.Vector([scene_center_x, tool_y, scene_center_z - tool_z_offset])
        tool_quat[0] = ti.Vector([0.0, 0.0, 0.0, 1.0])
        tool_quat[1] = ti.Vector([0.0, 0.0, 0.0, 1.0])
        tool_velocity[0] = ti.Vector([0.0, 0.0, 0.0])
        tool_velocity[1] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def update_tool_visuals(time: ti.f32):
        motion_time = ti.max(time - tool_motion_start, 0.0)
        progress = 0.0
        if time >= tool_motion_start:
            progress = ti.min(motion_time / tool_close_time, 1.0)

        for p in range(tool_vis_count):
            local_id = p % (tool_vis_resolution * tool_vis_resolution * tool_vis_resolution)
            ix = local_id % tool_vis_resolution
            iy = (local_id // tool_vis_resolution) % tool_vis_resolution
            iz = local_id // (tool_vis_resolution * tool_vis_resolution)
            uvw = ti.Vector([
                ix / (tool_vis_resolution - 1),
                iy / (tool_vis_resolution - 1),
                iz / (tool_vis_resolution - 1),
            ])
            tool_id = 0
            half = ti.Vector([tool_half_0[0], tool_half_0[1], tool_half_0[2]])
            if p >= tool_vis_count // 2:
                tool_id = 1
                half = ti.Vector([tool_half_1[0], tool_half_1[1], tool_half_1[2]])
            local = (uvw * 2.0 - 1.0) * half

            center = tool_center[tool_id]
            if scripted_tools == 1:
                if tool_id == 0:
                    center = ti.Vector([scene_center_x, tool_y, scene_center_z + tool_z_offset - tool_travel * progress])
                else:
                    center = ti.Vector([scene_center_x, tool_y, scene_center_z - tool_z_offset + tool_travel * progress])

            tool_x[p] = center + quat_to_matrix(tool_quat[tool_id]) @ local

    @ti.func
    def scalar_is_finite(value):
        return not ti.math.isnan(value) and not ti.math.isinf(value)

    @ti.func
    def particle_state_is_finite(p):
        finite = scalar_is_finite(Jp[p])
        for axis in ti.static(range(dim)):
            finite = finite and scalar_is_finite(x[p][axis]) and scalar_is_finite(v[p][axis])
            for column in ti.static(range(dim)):
                finite = finite and scalar_is_finite(C[p][axis, column]) and scalar_is_finite(F[p][axis, column])
        return finite

    @ti.func
    def position_has_safe_grid_stencil(position):
        # This exactly matches the base conversion used by P2G and G2P below.
        # Taichi's float-to-int conversion truncates toward zero, so checking a
        # continuous lower bound would reject valid floor-adjacent particles.
        base = (position * inv_dx - 0.5).cast(int)
        return (
            base.x >= 0 and base.x + 2 < n_grid
            and base.y >= 0 and base.y + 2 < n_grid
            and base.z >= 0 and base.z + 2 < n_grid
        )

    @ti.kernel
    def reset_invalid_state():
        invalid_pre_p2g_state[None] = n_particles
        invalid_pre_p2g_stencil[None] = n_particles
        invalid_post_g2p_state[None] = n_particles
        invalid_post_g2p_stencil[None] = n_particles

    @ti.kernel
    def reset_sdf_contact_diagnostics():
        for tool_id in range(2):
            sdf_grid_evaluated[tool_id] = 0
            sdf_grid_valid[tool_id] = 0
            sdf_grid_invalid[tool_id] = 0
            sdf_grid_min_distance[tool_id] = ti.math.inf
            sdf_grid_inside[tool_id] = 0
            sdf_grid_candidate[tool_id] = 0
            sdf_grid_applied[tool_id] = 0
            sdf_grid_inward_removed[tool_id] = 0
            sdf_particle_evaluated[tool_id] = 0
            sdf_particle_valid[tool_id] = 0
            sdf_particle_invalid[tool_id] = 0
            sdf_particle_min_distance[tool_id] = ti.math.inf
            sdf_particle_inside[tool_id] = 0
            sdf_particle_candidate[tool_id] = 0
            sdf_particle_applied[tool_id] = 0
            sdf_particle_inward_removed[tool_id] = 0

    @ti.kernel
    def substep_kernel(time: ti.f32):
        for I in ti.grouped(grid_m):
            grid_v[I] = ti.Vector.zero(ti.f32, dim)
            grid_m[I] = 0.0

        for p in x:
            if not particle_state_is_finite(p):
                ti.atomic_min(invalid_pre_p2g_state[None], p)
            elif not position_has_safe_grid_stencil(x[p]):
                ti.atomic_min(invalid_pre_p2g_stencil[None], p)
            else:
                F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * C[p]) @ F[p]
                old_J = F[p].determinant()
                yielded[p] = 0
                if pure_viscoelastic == 0:
                    U, sig, V = ti.svd(F[p])
                    for d in ti.static(range(dim)):
                        unclamped = sig[d, d]
                        clamped = ti.min(ti.max(unclamped, plastic_min), plastic_max)
                        if ti.abs(unclamped - clamped) > 1e-6:
                            yielded[p] = 1
                        sig[d, d] = clamped
                    F[p] = U @ sig @ V.transpose()
                    if use_jp == 1:
                        new_J = F[p].determinant()
                        Jp[p] = ti.min(ti.max(Jp[p] * old_J / new_J, jp_min), jp_max)

                if not particle_state_is_finite(p):
                    ti.atomic_min(invalid_pre_p2g_state[None], p)
                else:
                    base = (x[p] * inv_dx - 0.5).cast(int)
                    fx = x[p] * inv_dx - base.cast(float)
                    w = [
                        0.5 * (1.5 - fx) ** 2,
                        0.75 - (fx - 1.0) ** 2,
                        0.5 * (fx - 0.5) ** 2,
                    ]
                    J = F[p].determinant()
                    hardening = ti.exp(jp_hardening * (1.0 - Jp[p]))
                    mu = mu_0 * hardening
                    la = lambda_0 * hardening
                    r, _ = ti.polar_decompose(F[p])
                    elastic_stress = 2 * mu * (F[p] - r) @ F[p].transpose()
                    elastic_stress += ti.Matrix.identity(ti.f32, dim) * la * J * (J - 1)
                    viscous_stress = viscosity * (C[p] + C[p].transpose())
                    stress = -dt * p_vol * 4 * inv_dx * inv_dx * (elastic_stress + viscous_stress)
                    affine = stress + p_mass * C[p]

                    for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                        offset = ti.Vector([i, j, k])
                        dpos = (offset.cast(float) - fx) * dx
                        weight = w[i].x * w[j].y * w[k].z
                        grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                        grid_m[base + offset] += weight * p_mass

        for I in ti.grouped(grid_m):
            if grid_m[I] > 0:
                grid_v[I] = grid_v[I] / grid_m[I]
                grid_v[I].y += dt * gravity

                pos = I.cast(float) * dx

                if pos.y <= floor_y and grid_v[I].y < 0.0:
                    grid_v[I].y *= -floor_absorption
                    grid_v[I].x *= floor_friction * (1.0 - floor_stickiness)
                    grid_v[I].z *= floor_friction * (1.0 - floor_stickiness)

                hit = 0
                selected_tool = -1
                normal = ti.Vector([0.0, 0.0, 0.0])
                collider_v = ti.Vector([0.0, 0.0, 0.0])
                if ti.static(use_mesh_collision):
                    hit, selected_tool, normal, collider_v = mesh_collision_velocity_and_normal(pos, time)
                elif ti.static(use_box_collision):
                    hit, normal, collider_v = box_collision_velocity_and_normal(pos, time)
                if hit == 1:
                    if ti.static(use_mesh_collision and record_sdf_contact_diagnostics):
                        ti.atomic_add(sdf_grid_applied[selected_tool], 1)
                    rel_v = grid_v[I] - collider_v
                    vn = rel_v.dot(normal)
                    if vn < 0.0:
                        rel_v -= normal * vn
                        if ti.static(use_mesh_collision and record_sdf_contact_diagnostics):
                            ti.atomic_add(sdf_grid_inward_removed[selected_tool], 1)
                    rel_v *= tool_contact_friction * (1.0 - tool_contact_absorption)
                    rel_v *= 1.0 - tool_stickiness
                    grid_v[I] = collider_v + rel_v

                bound = 3
                if I.x < bound and grid_v[I].x < 0:
                    grid_v[I].x = 0
                if I.x > n_grid - bound and grid_v[I].x > 0:
                    grid_v[I].x = 0
                if I.y > n_grid - bound and grid_v[I].y > 0:
                    grid_v[I].y = 0
                if I.z < bound and grid_v[I].z < 0:
                    grid_v[I].z = 0
                if I.z > n_grid - bound and grid_v[I].z > 0:
                    grid_v[I].z = 0

        for p in x:
            if not particle_state_is_finite(p):
                ti.atomic_min(invalid_post_g2p_state[None], p)
            elif not position_has_safe_grid_stencil(x[p]):
                ti.atomic_min(invalid_post_g2p_stencil[None], p)
            else:
                base = (x[p] * inv_dx - 0.5).cast(int)
                fx = x[p] * inv_dx - base.cast(float)
                w = [
                    0.5 * (1.5 - fx) ** 2,
                    0.75 - (fx - 1.0) ** 2,
                    0.5 * (fx - 0.5) ** 2,
                ]
                new_v = ti.Vector.zero(ti.f32, dim)
                new_C = ti.Matrix.zero(ti.f32, dim, dim)
                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (offset.cast(float) - fx) * dx
                    g_v = grid_v[base + offset]
                    weight = w[i].x * w[j].y * w[k].z
                    new_v += weight * g_v
                    new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
                v[p] = new_v * velocity_damping
                x[p] += dt * v[p]
                projected_x = x[p]
                projected_v = v[p]
                if ti.static(use_mesh_collision):
                    projected_x, projected_v = project_particle_out_of_meshes(x[p], v[p], time)
                elif ti.static(use_box_collision):
                    projected_x, projected_v = project_particle_out_of_tools(x[p], v[p], time)
                floor_contact = 0
                if projected_x.y < floor_y:
                    projected_x.y = floor_y
                    if projected_v.y < 0.0:
                        projected_v.y *= -floor_absorption
                    projected_v.x *= floor_friction * (1.0 - floor_stickiness)
                    projected_v.z *= floor_friction * (1.0 - floor_stickiness)
                    floor_contact = 1
                x[p] = projected_x
                v[p] = projected_v
                if x[p].y < floor_y + floor_plastic_damping_band:
                    if floor_contact == 0:
                        if v[p].y < 0.0:
                            v[p].y *= -floor_absorption
                        v[p].x *= floor_friction * (1.0 - floor_stickiness)
                        v[p].z *= floor_friction * (1.0 - floor_stickiness)
                    new_C *= plastic_affine_damping
                if yielded[p] == 1:
                    v[p] *= plastic_velocity_damping
                    new_C *= plastic_affine_damping
                C[p] = new_C
                if not particle_state_is_finite(p):
                    ti.atomic_min(invalid_post_g2p_state[None], p)
                elif not position_has_safe_grid_stencil(x[p]):
                    ti.atomic_min(invalid_post_g2p_stencil[None], p)

    invalid_kind_fields = (
        ("pre_p2g_nonfinite", invalid_pre_p2g_state),
        ("pre_p2g_stencil_out_of_bounds", invalid_pre_p2g_stencil),
        ("post_g2p_nonfinite", invalid_post_g2p_state),
        ("post_g2p_stencil_out_of_bounds", invalid_post_g2p_stencil),
    )
    substep_ordinal = 0

    def json_safe(value):
        if isinstance(value, np.ndarray):
            return json_safe(value.tolist())
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, (float, np.floating)):
            numeric = float(value)
            if not np.isfinite(numeric):
                return "nan" if np.isnan(numeric) else ("inf" if numeric > 0 else "-inf")
            return numeric
        return value

    def substep(time):
        nonlocal substep_ordinal
        reset_invalid_state()
        substep_kernel(time)
        failure_kind = None
        particle_index = n_particles
        for kind, field in invalid_kind_fields:
            index = int(field[None])
            if index < n_particles:
                failure_kind = kind
                particle_index = index
                break
        if failure_kind is not None:
            positions = x.to_numpy()
            velocities = v.to_numpy()
            affine = C.to_numpy()
            deformation = F.to_numpy()
            plastic_volume = Jp.to_numpy()
            position = positions[particle_index]
            grid_coordinate = position * inv_dx
            stencil_base = None
            if np.isfinite(grid_coordinate).all():
                stencil_base = np.trunc(grid_coordinate - 0.5).astype(np.int64)
            diagnostic = {
                "schema": "taichidough/mpm-invalid-state/v1",
                "failure_kind": failure_kind,
                "particle_index": particle_index,
                "substep": substep_ordinal,
                "sim_time_s": float(time),
                "grid_size": n_grid,
                "grid_coordinate": json_safe(grid_coordinate),
                "stencil_base": json_safe(stencil_base),
                "safe_stencil_base_interval": [0, n_grid - 3],
                "position_scene": json_safe(position),
                "velocity_scene": json_safe(velocities[particle_index]),
                "affine_velocity": json_safe(affine[particle_index]),
                "deformation_gradient": json_safe(deformation[particle_index]),
                "plastic_volume_ratio": json_safe(float(plastic_volume[particle_index])),
                "youngs_modulus_pa": float(E),
                "tool_collision": collision_mode,
            }
            substep_ordinal += 1
            raise MpmInvalidStateError(diagnostic)
        substep_ordinal += 1

    def sdf_contact_diagnostics():
        grid_candidates = sdf_grid_candidate.to_numpy()
        grid_applied = sdf_grid_applied.to_numpy()
        grid_inward_removed = sdf_grid_inward_removed.to_numpy()
        particle_candidates = sdf_particle_candidate.to_numpy()
        particle_applied = sdf_particle_applied.to_numpy()
        particle_inward_removed = sdf_particle_inward_removed.to_numpy()
        return [
            {
                "grid_nodes": {
                    "contact_candidates": int(grid_candidates[index]),
                    "applied_responses": int(grid_applied[index]),
                    "inward_normal_velocity_removed": int(grid_inward_removed[index]),
                },
                "particles": {
                    "contact_candidates": int(particle_candidates[index]),
                    "applied_responses": int(particle_applied[index]),
                    "inward_normal_velocity_removed": int(particle_inward_removed[index]),
                },
            }
            for index in range(2)
        ]

    result = x, tool_x, initialize, substep, update_tool_visuals, set_tool_state
    if return_sdf_contact_diagnostics:
        return (*result, reset_sdf_contact_diagnostics, sdf_contact_diagnostics)
    return result


def compute_camera_basis(position, look_at):
    position = np.asarray(position, dtype=np.float32)
    look_at = np.asarray(look_at, dtype=np.float32)
    forward = look_at - position
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    corrected_up = np.cross(right, forward)
    return right, corrected_up, forward


def render_particle_depth(points, width, height, view_name, config, output_dir, frame_idx, splat_radius=0):
    depth, nearest_indices, right, up, forward, camera_pos = rasterize_depth(
        points, width, height, config, splat_radius=splat_radius
    )
    finite = nearest_indices >= 0
    normalized = np.zeros_like(depth, dtype=np.uint8)
    if finite.any():
        values = depth[finite]
        normalized[finite] = ((1.0 - (values - values.min()) / max(values.max() - values.min(), 1e-6)) * 255).astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{view_name}_depth_{frame_idx:06d}"
    depth_path = output_dir / f"{stem}.npy"
    np.save(depth_path, depth)
    Image.fromarray(normalized, "L").save(output_dir / f"{stem}.png")
    camera_metadata = {key: value for key, value in config.items()}
    return {
        "name": view_name,
        "depth_array": str(depth_path),
        "depth_image": str(output_dir / f"{stem}.png"),
        **camera_metadata,
        "width": width,
        "height": height,
        "splat_radius": splat_radius,
        "_depth": depth,
        "_nearest_indices": nearest_indices,
        "_right": right,
        "_up": up,
        "_forward": forward,
        "_camera_pos": camera_pos,
    }


def depth_to_pointcloud(
    depth,
    nearest_indices,
    right,
    up,
    forward,
    camera_pos,
    width,
    height,
    config,
    max_points,
    pointcloud_format,
    pointcloud_frame="world",
):
    visible = nearest_indices >= 0
    pixel_y, pixel_x = np.nonzero(visible)
    if pixel_x.size == 0:
        columns = 7 if pointcloud_format == "deformpath7" else 3
        return np.empty((0, columns), dtype=np.float32)

    z = depth[pixel_y, pixel_x]
    f = height / (2.0 * np.tan(np.radians(config["fieldOfView"]) / 2.0))
    cam_x = (pixel_x.astype(np.float32) + 0.5 - width * 0.5) * z / f
    cam_y = (height * 0.5 - (pixel_y.astype(np.float32) + 0.5)) * z / f
    if pointcloud_frame == "world":
        xyz = (
            camera_pos[None, :]
            + cam_x[:, None] * right[None, :]
            + cam_y[:, None] * up[None, :]
            + z[:, None] * forward[None, :]
        ).astype(np.float32)
    elif pointcloud_frame == "camera":
        xyz = np.stack([cam_x, cam_y, z], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported depth pointcloud frame: {pointcloud_frame}")

    if max_points > 0 and xyz.shape[0] > max_points:
        selected = np.linspace(0, xyz.shape[0] - 1, max_points).round().astype(np.int64)
        xyz = xyz[selected]

    if pointcloud_format == "xyz":
        return xyz
    if pointcloud_format == "deformpath7":
        extra = np.zeros((xyz.shape[0], 4), dtype=np.float32)
        extra[:, 2] = 1.0
        return np.concatenate([xyz, extra], axis=1)
    raise ValueError(f"Unsupported depth pointcloud format: {pointcloud_format}")


def quaternion_to_matrix(quaternion):
    x, y, z, w = normalize_quaternion(quaternion)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def matrix_to_quaternion(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([(matrix[2, 1] - matrix[1, 2]) / s,
                      (matrix[0, 2] - matrix[2, 0]) / s,
                      (matrix[1, 0] - matrix[0, 1]) / s, 0.25 * s])
    else:
        axis = int(np.argmax(np.diag(matrix)))
        nxt = (axis + 1) % 3
        last = (axis + 2) % 3
        s = 2.0 * np.sqrt(max(1.0 + matrix[axis, axis] - matrix[nxt, nxt] - matrix[last, last], 1e-12))
        q = np.zeros(4)
        q[axis] = 0.25 * s
        q[3] = (matrix[last, nxt] - matrix[nxt, last]) / s
        q[nxt] = (matrix[nxt, axis] + matrix[axis, nxt]) / s
        q[last] = (matrix[last, axis] + matrix[axis, last]) / s
    return normalize_quaternion(q)


def normalize_quaternion(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quaternion / norm


def quaternion_multiply(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float32,
    )


def quaternion_conjugate(quaternion):
    return np.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]], dtype=np.float32)


def integrate_orientation(quaternion, angular_velocity, dt):
    angular_velocity = np.asarray(angular_velocity, dtype=np.float32)
    angle = np.linalg.norm(angular_velocity) * dt
    if angle < 1e-12:
        return normalize_quaternion(quaternion)

    axis = angular_velocity / np.linalg.norm(angular_velocity)
    half_angle = 0.5 * angle
    delta = np.array(
        [
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle),
            np.cos(half_angle),
        ],
        dtype=np.float32,
    )
    return normalize_quaternion(quaternion_multiply(delta, quaternion))


def angular_velocity_from_quaternions(old_quaternion, new_quaternion, dt):
    dt = max(float(dt), 1e-8)
    delta = quaternion_multiply(new_quaternion, quaternion_conjugate(old_quaternion))
    delta = normalize_quaternion(delta)
    if delta[3] < 0.0:
        delta = -delta
    vector_norm = np.linalg.norm(delta[:3])
    if vector_norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(vector_norm, delta[3])
    axis = delta[:3] / vector_norm
    return (axis * angle / dt).astype(np.float32)


class UdpRigidBoxControl:
    def __init__(self, receive_ports=(5005, 5007), transmit_ports=(5006, 5008), host="127.0.0.1", max_vel=1.0):
        self.host = host
        self.receive_sockets = []
        self.transmit_sockets = []
        self.transmit_addrs = [(host, port) for port in transmit_ports]
        self.poses = TOOL_INITIAL_POSES.copy()
        self.velocities = np.zeros((2, 6), dtype=np.float32)
        self.pose_control_active = np.zeros(2, dtype=bool)
        self.max_vel = max_vel

        for port in receive_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.setblocking(False)
            self.receive_sockets.append(sock)

        for _ in transmit_ports:
            self.transmit_sockets.append(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))

        print(f"ROS-style UDP box control listening on {receive_ports}, transmitting poses on {transmit_ports}")

    def close(self):
        for sock in self.receive_sockets + self.transmit_sockets:
            sock.close()

    def poll_and_integrate(self, dt):
        pose_driven = np.zeros(2, dtype=bool)
        for i, sock in enumerate(self.receive_sockets):
            try:
                while True:
                    data, _ = sock.recvfrom(1024)
                    msg = json.loads(data.decode())
                    if "vel" in msg:
                        vel = np.asarray(msg["vel"], dtype=np.float32)
                        if vel.size < 6:
                            vel = np.pad(vel, (0, 6 - vel.size))
                        self.velocities[i] = vel[:6]
                        self.pose_control_active[i] = False
                    elif "pose" in msg:
                        pose = np.asarray(msg["pose"], dtype=np.float32)
                        if pose.size >= 7:
                            old_pose = self.poses[i].copy()
                            pose_dt = float(msg.get("dt", dt))
                            self.poses[i] = pose[:7]
                            self.poses[i, 3:7] = normalize_quaternion(self.poses[i, 3:7])
                            self.velocities[i, :3] = (self.poses[i, :3] - old_pose[:3]) / max(pose_dt, 1e-8)
                            self.velocities[i, 3:6] = angular_velocity_from_quaternions(
                                old_pose[3:7], self.poses[i, 3:7], pose_dt
                            )
                            self.pose_control_active[i] = True
                            pose_driven[i] = True
            except BlockingIOError:
                pass

        linear = np.clip(self.velocities[:, :3], -self.max_vel, self.max_vel)
        angular = self.velocities[:, 3:6]
        for i in range(2):
            if not self.pose_control_active[i]:
                self.poses[i, :3] += linear[i] * dt
                self.poses[i, 3:7] = integrate_orientation(self.poses[i, 3:7], angular[i], dt)
            elif not pose_driven[i]:
                linear[i] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return self.poses.astype(np.float32), linear.astype(np.float32)

    def transmit_poses(self):
        for i, sock in enumerate(self.transmit_sockets):
            msg = {"pose": self.poses[i].tolist()}
            sock.sendto(json.dumps(msg).encode(), self.transmit_addrs[i])


class DoughCenterTransmitter:
    def __init__(self, port=5010, host="127.0.0.1"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (host, port)
        print(f"Publishing dough center over UDP to {host}:{port}")

    def send(self, center):
        msg = {"center": np.asarray(center, dtype=float).tolist()}
        self.sock.sendto(json.dumps(msg).encode(), self.addr)

    def close(self):
        self.sock.close()


def save_frame_outputs(points, args, views, run_dir, frame_idx, step, view_configs, completed_substeps=None):
    if completed_substeps is None:
        completed_substeps = step * args.substeps_per_frame
    particles_path = run_dir / f"particles_{frame_idx:06d}.npy"
    np.save(particles_path, points)
    frame = {
        "frame": frame_idx, "step": step, "particles": str(particles_path.resolve()), "views": [],
        "simulation_step": completed_substeps, "completed_substeps": completed_substeps,
        "sim_time_s": completed_substeps * args.dt, "initial_state": completed_substeps == 0,
    }
    for view_name in views:
        config = view_configs[view_name]
        rendered = render_particle_depth(
            points,
            args.depth_width if args.save_depth_pointclouds else args.width,
            args.depth_height if args.save_depth_pointclouds else args.height,
            view_name,
            config,
            run_dir,
            frame_idx,
            args.depth_splat_radius if args.save_depth_pointclouds else 0,
        )
        if args.save_depth_pointclouds:
            pointcloud = topview_depth_to_pointcloud(
                rendered.pop("_depth"),
                rendered.pop("_nearest_indices"),
                rendered.pop("_right"),
                rendered.pop("_up"),
                rendered.pop("_forward"),
                rendered.pop("_camera_pos"),
                config,
                max_points=args.depth_pointcloud_max_points,
                frame=args.depth_pointcloud_frame,
            )
            formatted = format_pointcloud(pointcloud, args.depth_pointcloud_format)
            pointcloud_path = run_dir / f"{view_name}_pointcloud_{frame_idx:06d}.npy"
            np.save(pointcloud_path, formatted)
            rendered["pointcloud_array"] = str(pointcloud_path)
            rendered["pointcloud_frame"] = args.depth_pointcloud_frame
            rendered["pointcloud_format"] = args.depth_pointcloud_format
            rendered["pointcloud_count"] = int(len(formatted))
            if args.depth_pointcloud_save_pt:
                try:
                    import torch
                except ImportError as exc:
                    raise ImportError("--depth-pointcloud-save-pt requires torch.") from exc
                pointcloud_pt_path = pointcloud_path.with_suffix(".pt")
                torch.save(torch.from_numpy(formatted), pointcloud_pt_path)
                rendered["pointcloud_tensor"] = str(pointcloud_pt_path)
        else:
            for key in ("_depth", "_nearest_indices", "_right", "_up", "_forward", "_camera_pos"):
                rendered.pop(key)
        frame["views"].append(rendered)
    return frame


def draw_gui_frame(window, canvas, scene, camera, particles, tools, time, free_camera=True, movement_speed=0.03, frame_path=None):
    if free_camera:
        camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.RMB)
    else:
        camera.position(0., 1.35, -1.35)
        camera.lookat(*SCENE_CENTER)
        camera.up(0.0, 1.0, 0.0)
    scene.set_camera(camera)
    scene.ambient_light((0.35, 0.35, 0.35))
    scene.point_light(pos=(0.4, 0.9, 1.1), color=(1.0, 1.0, 1.0))
    scene.particles(particles, radius=0.006, color=(0.78, 0.55, 0.36))
    scene.particles(tools, radius=0.004, color=(0.35, 0.42, 0.50))
    canvas.scene(scene)
    if frame_path is not None:
        window.save_image(str(frame_path))
    window.show()


def create_video_writer(output_path, fps, frame_size):
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open video writer for {output_path}")
    return writer


def main():
    parser = argparse.ArgumentParser(description="Taichi MLS-MPM viscoelastic dough scene prototype.")
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=2e-4)
    parser.add_argument("--substeps-per-frame", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--view", choices=list(CAMERA_VIEWS), action="append")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--save-initial-frame", action="store_true", help="Save a step-0 frame immediately after initialization.")
    parser.add_argument("--save-depth-pointclouds", action="store_true", help="Export z-buffered virtual depth point clouds with saved frames.")
    parser.add_argument("--depth-pointcloud-save-pt", action="store_true", help="Also write exported virtual point clouds as Torch tensors.")
    parser.add_argument("--depth-pointcloud-frame", choices=("world", "camera", "camera_optical"), default="camera")
    parser.add_argument("--depth-pointcloud-format", choices=("xyz", "deformpath7"), default="deformpath7")
    parser.add_argument("--depth-pointcloud-max-points", type=int, default=0, help="Maximum points per exported cloud; 0 keeps all visible pixels.")
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--depth-splat-radius", type=int, default=3, help="Pixel radius for virtual depth splats.")
    parser.add_argument("--youngs-modulus", type=float, default=2000)
    parser.add_argument("--poisson-ratio", type=float, default=0.35)
    parser.add_argument("--viscosity", type=float, default=2.5)
    parser.add_argument("--density", type=float, default=1100.0)
    parser.add_argument(
        "--object-mass-kg",
        type=float,
        default=None,
        help="Independent object mass measurement. Requires reconstructed volume metadata and determines density.",
    )
    parser.add_argument("--gravity", type=float, default=-9.81)
    parser.add_argument("--floor-y", type=float, default=0.20)
    parser.add_argument("--floor-friction", type=float, default=0.7)
    parser.add_argument(
        "--floor-absorption",
        type=float,
        default=0.0,
        help="Normal bounce kept at floor impact. 0 removes downward velocity, 1 is fully elastic bounce.",
    )
    parser.add_argument("--replay-episode", type=Path, help="Replay captured tools on simulation time and export at observation timestamps (headless).")
    parser.add_argument("--replay-start-frame", type=int, default=0)
    parser.add_argument("--replay-end-frame", type=int, default=None, help="Inclusive retained frame index; defaults to the last observation. --steps is ignored for replay.")
    parser.add_argument("--replay-stride", type=int, default=1, help="Observation export stride; tool interpolation still uses every captured pose.")
    parser.add_argument("--replay-max-gap", type=float, default=0.1, help="Largest allowed tool interpolation gap in seconds.")
    parser.add_argument(
        "--tool-geometry",
        type=Path,
        default=None,
        help="Two-tool collider geometry JSON using schema taichidough/tool-geometry/v1.",
    )
    parser.add_argument(
        "--tool-half-extents",
        nargs=3,
        type=float,
        default=None,
        help="Legacy shared proxy half-extents in scene metres. Defaults to 0.05 m per axis.",
    )
    parser.add_argument(
        "--tool-marker-offset",
        nargs=3,
        type=float,
        default=None,
        help="Legacy shared marker-local offset to proxy center in source metres. Defaults to zero.",
    )
    parser.add_argument("--tool-close-time", type=float, default=0.04)
    parser.add_argument("--tool-motion-start", type=float, default=1.0)
    parser.add_argument("--tool-contact-padding", type=float, default=0.035)
    parser.add_argument(
        "--record-sdf-contact-diagnostics",
        action="store_true",
        help="Write per-replay-frame SDF proximity and applied-contact measurements.",
    )
    parser.add_argument(
        "--tool-collision",
        choices=("sdf", "box", "none"),
        default="box",
        help="Tool collision geometry: calibrated box proxy (default), STL signed-distance field, or none.",
    )
    parser.add_argument(
        "--tool-sdf-resolution",
        type=int,
        default=64,
        help="Voxel resolution per STL axis for --tool-collision sdf.",
    )
    parser.add_argument(
        "--ur-tool-mesh",
        type=Path,
        default=None,
        help="Override the UR visual STL; used for rendering only, not solid-SDF collision.",
    )
    parser.add_argument(
        "--kinova-tool-mesh",
        type=Path,
        default=None,
        help="Override the Kinova visual STL; used for rendering only, not solid-SDF collision.",
    )
    parser.add_argument(
        "--ur-tool-collision-mesh",
        type=Path,
        default=None,
        help="Override the validated UR watertight collision STL.",
    )
    parser.add_argument(
        "--kinova-tool-collision-mesh",
        type=Path,
        default=None,
        help="Override the validated Kinova watertight collision STL.",
    )
    parser.add_argument(
        "--tool-mesh-scale",
        type=float,
        default=0.001,
        help="Scale applied to STL coordinates in millimetres.",
    )
    parser.add_argument("--tool-contact-friction", type=float, default=0.75)
    parser.add_argument(
        "--tool-contact-absorption",
        type=float,
        default=0.0,
        help="Extra damping at tool contacts. 0 keeps old response, 1 sticks to the tool velocity.",
    )
    parser.add_argument(
        "--tool-stickiness",
        type=float,
        default=0.0,
        help="Blend contact velocity toward tool velocity. 0 disables sticky tool contact.",
    )
    parser.add_argument(
        "--floor-stickiness",
        type=float,
        default=0.0,
        help="Extra damping of horizontal velocity at floor contact. 0 disables sticky floor contact.",
    )
    parser.add_argument(
        "--floor-plastic-damping-band",
        type=float,
        default=0.02,
        help="Height above the floor where particle velocity-gradient damping is applied.",
    )
    parser.add_argument("--velocity-damping", type=float, default=0.998)
    parser.add_argument(
        "--pure-viscoelastic",
        action="store_true",
        help="Disable SVD clamp plasticity and use the older viscoelastic-only model.",
    )
    parser.add_argument("--plastic-min", type=float, default=0.88, help="Minimum singular value kept in F.")
    parser.add_argument("--plastic-max", type=float, default=1.08, help="Maximum singular value kept in F.")
    parser.add_argument(
        "--plastic-velocity-damping",
        type=float,
        default=0.92,
        help="Extra velocity damping applied only to particles that yielded this step.",
    )
    parser.add_argument(
        "--plastic-affine-damping",
        type=float,
        default=0.80,
        help="Extra C/velocity-gradient damping applied only to particles that yielded this step.",
    )
    parser.add_argument("--use-jp", action="store_true", help="Track accumulated plastic volume change Jp.")
    parser.add_argument(
        "--jp-hardening",
        type=float,
        default=0.0,
        help="Hardening coefficient applied as exp(jp_hardening * (1 - Jp)).",
    )
    parser.add_argument("--jp-min", type=float, default=0.6)
    parser.add_argument("--jp-max", type=float, default=2.0)
    parser.add_argument("--cpu", action="store_true", help="Use CPU backend instead of GPU.")
    parser.add_argument("--gui", action="store_true", help="Open a live Taichi 3D viewer.")
    parser.add_argument("--no-save", action="store_true", help="Run without writing camera/particle frames.")
    parser.add_argument("--gui-fps-substeps", type=int, default=4, help="MPM substeps between GUI redraws.")
    parser.add_argument("--free-camera", action="store_true", help="Allow mouse/keyboard control of the GUI camera.", default=True)
    parser.add_argument("--camera-speed", type=float, default=0.01, help="Movement speed for --free-camera.")
    parser.add_argument("--record-video", action="store_true", help="Record live GUI frames to an MP4.")
    parser.add_argument("--video-path", type=Path, default=OUTPUT_DIR / "viscoelastic_mpm_gui.mp4")
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument(
        "--video-simulation-time",
        action="store_true",
        help="Set video FPS so playback duration matches simulated time exactly.",
    )
    parser.add_argument(
        "--keep-video-frames",
        action="store_true",
        help="Keep temporary PNG frames used for video encoding.",
    )
    parser.add_argument(
        "--ros-control",
        action="store_true",
        help="Control the two box tools with the same UDP vel/pose ports used by the SOFA ROS bridge.",
    )
    parser.add_argument("--ros-tool-max-vel", type=float, default=1.0)
    parser.add_argument("--publish-dough-center", action="store_true", help="Publish particle mean center over UDP.", default=True)
    parser.add_argument(
        "--no-publish-dough-center",
        action="store_false",
        dest="publish_dough_center",
        help="Disable UDP publishing of the particle mean center.",
    )
    parser.add_argument("--dough-center-port", type=int, default=5010)
    parser.add_argument(
        "--initial-particles",
        type=Path,
        default=None,
        help="Initialize dough particle positions from a .npy or .pt file with shape [N, >=3].",
    )
    parser.add_argument(
        "--initial-particles-metadata",
        type=Path,
        default=None,
        help="Reconstruction metadata for scene-coordinate initial particles and conserved object volume.",
    )
    parser.add_argument(
        "--initial-particles-calibration",
        type=Path,
        default=None,
        help="Calibration JSON used for camera and replay geometry. Reconstructed particles are not transformed again.",
    )
    parser.add_argument(
        "--initial-particles-axis-map",
        type=str,
        default=None,
        help=(
            "Source axes used for destination Taichi x,y,z. Legacy initialization defaults to xzy; "
            "reconstruction metadata defaults to xyz and rejects non-identity maps."
        ),
    )
    parser.add_argument(
        "--initial-particles-fit",
        choices=["anisotropic", "isotropic", "none"],
        default=None,
        help="How to scale loaded particles into the current dough bounding box. Legacy initialization defaults to anisotropic.",
    )
    parser.add_argument(
        "--initial-particles-scale",
        type=float,
        default=None,
        help="Extra scale multiplier applied after fitting loaded particles. Defaults to 1.",
    )
    parser.add_argument(
        "--initial-particles-offset",
        type=float,
        nargs=3,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Scene-coordinate offset added after fitting loaded particles. Defaults to zero.",
    )
    parser.add_argument(
        "--initial-particles-raw-scene-coordinates",
        action="store_true",
        help="Use loaded XYZ directly as Taichi scene coordinates, only applying the axis map.",
    )
    parser.add_argument(
        "--initial-particles-seed",
        type=int,
        default=0,
        help="Seed used when --particles requires subsampling or repeating loaded particles.",
    )
    parser.add_argument(
        "--timing-report-interval",
        type=float,
        default=2.0,
        help="Wall-clock seconds between sim/real speed reports. 0 disables reports.",
    )
    args = parser.parse_args()
    if args.depth_pointcloud_max_points < 0 or args.depth_splat_radius < 0:
        raise ValueError("Depth pointcloud limits and splat radius must be non-negative")
    if not np.isfinite(args.dt) or args.dt <= 0 or min(args.substeps_per_frame, args.gui_fps_substeps, args.save_every) <= 0 or args.steps < 0:
        raise ValueError("dt, substeps-per-frame, gui-fps-substeps and save-every must be positive; steps cannot be negative")
    if args.tool_mesh_scale <= 0.0:
        raise ValueError("--tool-mesh-scale must be positive")
    if args.tool_sdf_resolution < 16:
        raise ValueError("--tool-sdf-resolution must be at least 16")
    tool_geometry = resolve_tool_geometry(
        args.tool_geometry,
        args.tool_half_extents,
        args.tool_marker_offset,
    )
    args.tool_half_extents_by_tool = tool_geometry["half_extents_m"]
    args.tool_marker_from_collider = tool_geometry["marker_from_collider"]
    args.tool_marker_from_mesh = tool_geometry["marker_from_mesh"]
    if tool_geometry["source"] == "legacy_cli":
        args.tool_half_extents = tuple(tool_geometry["half_extents_m"][0])
        args.tool_marker_offset = tuple(
            float(value) for value in np.asarray(tool_geometry["marker_from_collider"][0])[:3, 3]
        )
    if args.initial_particles_metadata is not None and args.initial_particles is None:
        raise ValueError("--initial-particles-metadata requires --initial-particles")

    calibration = load_calibration(args.initial_particles_calibration) if args.initial_particles_calibration else None
    reconstruction_info = None
    initial_particles = None
    if args.initial_particles is not None:
        loaded_particles = load_initial_particles(args.initial_particles)
        if args.initial_particles_metadata is not None:
            validate_reconstructed_particle_options(
                args.initial_particles_axis_map,
                args.initial_particles_fit,
                args.initial_particles_scale,
                args.initial_particles_offset,
            )
            reconstruction_info = load_reconstruction_metadata(
                args.initial_particles_metadata,
                args.initial_particles,
                loaded_particles,
                calibration,
            )
            if np.any(loaded_particles <= 0.0) or np.any(loaded_particles >= 1.0):
                raise ValueError("Reconstructed scene-coordinate particles must lie strictly inside the unit Taichi scene")
            fitted_particles = loaded_particles
            args.initial_particles_axis_map = "xyz"
            args.initial_particles_fit = "none"
            args.initial_particles_scale = 1.0
            args.initial_particles_offset = (0.0, 0.0, 0.0)
            args.initial_particles_raw_scene_coordinates = True
            args.floor_y = reconstruction_info["floor_y"]
        else:
            args.initial_particles_axis_map = args.initial_particles_axis_map or "xzy"
            args.initial_particles_fit = args.initial_particles_fit or "anisotropic"
            args.initial_particles_scale = 1.0 if args.initial_particles_scale is None else args.initial_particles_scale
            args.initial_particles_offset = args.initial_particles_offset or (0.0, 0.0, 0.0)
            fitted_particles = fit_initial_particles_to_scene(loaded_particles, args, calibration)
        if args.particles is None:
            args.particles = len(fitted_particles)
        initial_particles = resize_initial_particles(fitted_particles, args.particles, args.initial_particles_seed)
        print(
            f"Loaded {len(loaded_particles)} initial particles from {args.initial_particles}; "
            f"using {len(initial_particles)} particles in the simulation.",
            flush=True,
        )
    else:
        args.initial_particles_axis_map = args.initial_particles_axis_map or "xzy"
        args.initial_particles_fit = args.initial_particles_fit or "anisotropic"
        args.initial_particles_scale = 1.0 if args.initial_particles_scale is None else args.initial_particles_scale
        args.initial_particles_offset = args.initial_particles_offset or (0.0, 0.0, 0.0)
        if args.particles is None:
            args.particles = 24000

    args.mass_properties = compute_mass_properties(
        args.particles,
        args.grid,
        args.density,
        reconstruction_info["object_volume_m3"] if reconstruction_info is not None else None,
        args.object_mass_kg,
    )
    args.density = args.mass_properties["density_kg_m3"]
    if not np.isfinite(args.floor_y):
        raise ValueError("Floor height must be finite")

    replay = None
    if args.replay_episode:
        if (
            args.gui
            or args.ros_control
            or args.no_save
            or not args.initial_particles
            or calibration is None
            or reconstruction_info is None
        ):
            raise ValueError("Replay requires headless saved output, reconstruction metadata, calibration, and no UDP tool control")
        if args.replay_stride < 1:
            raise ValueError("Replay stride must be positive")
        try:
            from deformpath_dynamics import ToolReplay, load_observation_sequence
        except ImportError:
            from .deformpath_dynamics import ToolReplay, load_observation_sequence
        observation_sequence = load_observation_sequence(args.replay_episode)
        tool_geometry = align_tool_geometry(tool_geometry, observation_sequence.names)
        args.tool_half_extents_by_tool = tool_geometry["half_extents_m"]
        args.tool_marker_from_collider = tool_geometry["marker_from_collider"]
        args.tool_marker_from_mesh = tool_geometry["marker_from_mesh"]
        initialization = reconstruction_info["metadata"]
        if (
            initialization.get("frame") != args.replay_start_frame
            or Path(initialization.get("episode_dir", "")).resolve() != observation_sequence.episode_dir
            or initialization.get("pointclouds_name") != "pointclouds_interpolated.pt"
            or reconstruction_info.get("calibration_fingerprint") != calibration.fingerprint
        ):
            raise ValueError("Initial reconstruction must match the replay episode, start frame and calibration")
        replay_end = len(observation_sequence.times) - 1 if args.replay_end_frame is None else args.replay_end_frame
        replay = ToolReplay(
            observation_sequence,
            calibration,
            args.replay_start_frame,
            replay_end,
            max_gap_s=args.replay_max_gap,
            marker_from_tool_frames=replay_marker_transforms(tool_geometry, args.tool_collision),
        )
        args.save_initial_frame = True
        args.save_depth_pointclouds = True
        args.depth_pointcloud_frame = "camera_optical"
        args.depth_pointcloud_format = "xyz"
        args.view = ["deformpath_top"]
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("Replay output directory must be empty; preserve previous benchmark artifacts")

    ti.init(arch=ti.cpu if args.cpu else ti.gpu)

    mesh_collision = None
    if args.tool_collision == "sdf":
        ur_tool_mesh_path = find_tool_mesh("ur_spathla_collision_solid.stl", args.ur_tool_collision_mesh)
        kinova_tool_mesh_path = find_tool_mesh("gen3_spathla_collision_solid.stl", args.kinova_tool_collision_mesh)
        ur_tool_mesh, _ = load_binary_stl(ur_tool_mesh_path, args.tool_mesh_scale)
        kinova_tool_mesh, _ = load_binary_stl(kinova_tool_mesh_path, args.tool_mesh_scale)
        print(f"UR solid collision mesh: {ur_tool_mesh_path}")
        print(f"Kinova solid collision mesh: {kinova_tool_mesh_path}")
        print(f"Building solid STL collision SDFs at {args.tool_sdf_resolution}^3 voxels per tool...")
        mesh_collision = create_mesh_collision_fields(
            [
                mesh_in_tool_frame(ur_tool_mesh, UR_TOOL_VISUAL_ORIGIN, rpy_to_matrix(UR_TOOL_VISUAL_RPY)),
                mesh_in_tool_frame(kinova_tool_mesh, KINOVA_TOOL_VISUAL_ORIGIN, rpy_to_matrix(KINOVA_TOOL_VISUAL_RPY)),
            ],
            args.tool_sdf_resolution,
            args.tool_contact_padding,
        )
        mesh_collision.asset_paths = [str(ur_tool_mesh_path.resolve()), str(kinova_tool_mesh_path.resolve())]
        mesh_collision.asset_sha256 = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (ur_tool_mesh_path, kinova_tool_mesh_path)]
        collision_manifest = PROJECT_ROOT / "meshes" / "tool_collision_meshes_v1.json"
        use_default_collision_assets = args.ur_tool_collision_mesh is None and args.kinova_tool_collision_mesh is None
        if use_default_collision_assets:
            if not collision_manifest.is_file():
                raise FileNotFoundError(f"Missing solid collision manifest: {collision_manifest}")
            manifest = json.loads(collision_manifest.read_text())
            if manifest.get("schema") != "taichidough/tool-collision-meshes/v1":
                raise ValueError(f"Unsupported solid collision manifest schema: {manifest.get('schema')!r}")
            expected = {entry.get("name"): entry.get("collision_sha256") for entry in manifest.get("tools", [])}
            actual = dict(zip(("UR5e_spathla", "gen3_spathla"), mesh_collision.asset_sha256))
            if expected != actual:
                raise ValueError("Solid collision mesh hashes do not match tool_collision_meshes_v1.json")
            mesh_collision.manifest_path = str(collision_manifest.resolve())
            mesh_collision.manifest_sha256 = hashlib.sha256(collision_manifest.read_bytes()).hexdigest()
        else:
            mesh_collision.manifest_path = None
            mesh_collision.manifest_sha256 = None

    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    view_configs = dict(CAMERA_VIEWS)
    if calibration is not None:
        view_configs["deformpath_top"] = calibration.camera
    views = args.view or list(view_configs)

    (
        x, tool_x, initialize, substep, update_tool_visuals, set_tool_state,
        reset_sdf_contact_diagnostics, sdf_contact_diagnostics,
    ) = build_sim(args, mesh_collision, return_sdf_contact_diagnostics=True)
    initialize()
    if initial_particles is not None:
        x.from_numpy(initial_particles)
    if replay is not None:
        set_tool_state(*replay.at(0.0))
    else:
        set_tool_state(TOOL_INITIAL_POSES, np.zeros((2, 3), dtype=np.float32))
    update_tool_visuals(0.0)
    reset_sdf_contact_diagnostics()

    def run_substep(current_time):
        try:
            substep(current_time)
        except MpmInvalidStateError as exc:
            invalid_state_path = run_dir / "invalid_state.json"
            invalid_state_path.write_text(json.dumps(exc.diagnostic, indent=2, allow_nan=False))
            print(f"{exc}; wrote {invalid_state_path}", flush=True)
            raise

    ros_control = UdpRigidBoxControl(max_vel=args.ros_tool_max_vel) if args.ros_control else None
    dough_center_tx = DoughCenterTransmitter(port=args.dough_center_port) if args.publish_dough_center else None

    metadata = {
        "scene": "Taichi 3D MLS-MPM approximation of scene_with_camera.py",
        "model": "compressible corotated/Neo-Hookean stress plus viscosity, with optional SVD clamp plasticity",
        "frames": [],
        "video": {},
        "time_convention": "simulation_step counts completed MPM substeps; sim_time_s = simulation_step * dt; step counts outer integration batches",
        "calibration": calibration_metadata(calibration) if calibration is not None else None,
        "calibration_fingerprint": (
            reconstruction_info.get("calibration_fingerprint")
            if reconstruction_info is not None
            else getattr(calibration, "fingerprint", None)
        ),
        "initial_particles_metadata_fingerprint": (
            reconstruction_info["metadata_sha256"] if reconstruction_info is not None else None
        ),
        "initialization": (
            {
                "particle_path": str(args.initial_particles.resolve()),
                "particle_sha256": reconstruction_info["particle_sha256"],
                "metadata_path": reconstruction_info["metadata_path"],
                "metadata_sha256": reconstruction_info["metadata_sha256"],
                "scene_frame": reconstruction_info["scene_frame"],
                "floor_plane_scene": reconstruction_info["floor_plane_scene"],
            }
            if reconstruction_info is not None
            else None
        ),
        "mass": args.mass_properties,
        "mass_source": args.mass_properties["mass_source"],
        "object_volume_m3": args.mass_properties["object_volume_m3"],
        "density_kg_m3": args.mass_properties["density_kg_m3"],
        "total_mass_kg": args.mass_properties["total_mass_kg"],
        "particle_volume_m3": args.mass_properties["particle_volume_m3"],
        "particle_mass_kg": args.mass_properties["particle_mass_kg"],
        "tool_geometry": tool_geometry,
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }

    frame_idx = 0
    if args.save_initial_frame:
        initial_frame = save_frame_outputs(x.to_numpy(), args, views, run_dir, frame_idx, 0, view_configs)
        initial_frame["simulation_step"] = 0
        initial_frame["initial_state"] = True
        metadata["frames"].append(initial_frame)
        frame_idx += 1

    if replay is not None:
        replay_is_sdf = args.tool_collision == "sdf"
        write_sdf_contact_diagnostics = replay_is_sdf and args.record_sdf_contact_diagnostics
        metadata["replay"] = {
            "mode": "captured_trajectory_sdf_tools" if replay_is_sdf else "captured_trajectory_proxy_tools",
            "tool_collision": args.tool_collision,
            "tool_pose_frame": "mesh_tool_link" if replay_is_sdf else "collider",
            "episode_dir": str(observation_sequence.episode_dir),
            "sequence_fingerprint": observation_sequence.fingerprint,
            "source_start_frame": replay.start, "source_end_frame": replay.end,
            "source_time_origin_s": float(observation_sequence.times[replay.start]),
            "tool_names": observation_sequence.names,
            "tool_geometry": tool_geometry,
            "max_interpolation_gap_s": args.replay_max_gap,
            "tool_contact_padding_scene_m": args.tool_contact_padding,
            "solid_collision_assets": (
                {
                    "paths": mesh_collision.asset_paths,
                    "sha256": mesh_collision.asset_sha256,
                    "manifest_path": mesh_collision.manifest_path,
                    "manifest_sha256": mesh_collision.manifest_sha256,
                    "sdf_statistics": mesh_collision.statistics,
                }
                if replay_is_sdf else None
            ),
            "limitations": [
                "SDF collision approximates the physical tool surfaces with voxelized STL meshes."
                if replay_is_sdf else (
                    "Box colliders approximate the physical tool surfaces."
                    if args.tool_collision == "box" else "Tool collision is disabled."
                ),
                "Camera calibration is supplied, not independently verified.",
                "Depth exports contain dough only; tool occlusion is not modeled.",
                "Velocity uses the derivative of piecewise-linear positions and SLERP orientations.",
                "Volume comes from floor-filled voxel reconstruction; unobserved side geometry remains approximate.",
            ],
        }
        if write_sdf_contact_diagnostics:
            contact_debug_path = run_dir / "replay_sdf_contact_debug.json"
            metadata["replay"]["sdf_contact_diagnostics"] = {
                "schema": "taichidough/replay-sdf-contact-debug/v1",
                "path": str(contact_debug_path.resolve()),
            }
            contact_debug = {
                "schema": "taichidough/replay-sdf-contact-debug/v1",
                "units": "scene metres and collision-branch executions",
                "tool_collision": "sdf",
                "tool_names": observation_sequence.names,
                "sequence_fingerprint": observation_sequence.fingerprint,
                "tool_contact_padding_scene_m": float(args.tool_contact_padding),
                "tool_sdf_resolution": int(args.tool_sdf_resolution),
                "predicate": {
                    "valid_sdf_sample": "point is inside the finite local SDF volume",
                    "inside_mesh": "valid && signed_distance_m < 0",
                    "contact_candidate": "valid && signed_distance_m < tool_contact_padding_scene_m",
                    "grid_applied_response": "selected candidate executes the grid velocity response",
                    "particle_applied_response": "candidate executes post-G2P particle projection; counts can include both tools for one particle",
                },
                "sdf_local_bounds": [
                    {
                        "minimum_scene_m": minimum.tolist(),
                        "maximum_scene_m": (minimum + spacing * (args.tool_sdf_resolution - 1)).tolist(),
                        "voxel_spacing_scene_m": spacing.tolist(),
                    }
                    for minimum, spacing in zip(mesh_collision.minimums.to_numpy(), mesh_collision.spacings.to_numpy())
                ],
                "frames": [],
            }

            def append_contact_debug(frame, window_start_substep, snapshot=False):
                contact_debug["frames"].append({
                    "frame": int(frame["frame"]),
                    "simulation_step": int(frame["simulation_step"]),
                    "completed_substeps": int(frame["completed_substeps"]),
                    "sim_time_s": float(frame["sim_time_s"]),
                    "source_frame": int(frame["source_frame"]),
                    "original_source_frame": int(frame["original_source_frame"]),
                    "collection_window_start_substep": int(window_start_substep),
                    "collection_window_end_substep": int(frame["completed_substeps"]),
                    "snapshot_without_substep_evaluations": bool(snapshot),
                    "tools": sdf_contact_diagnostics(),
                })
        if tool_geometry["source"] == "legacy_cli":
            metadata["replay"]["tool_half_extents_scene_m"] = list(args.tool_half_extents)
            metadata["replay"]["tool_marker_offset_source_m"] = list(args.tool_marker_offset)
        def annotate_replay_frame(frame, source_index):
            target = float(observation_sequence.times[source_index] - observation_sequence.times[replay.start])
            poses, velocities = replay.at(min(frame["sim_time_s"], replay.times[-1]))
            frame.update({
                "source_frame": source_index, "original_source_frame": observation_sequence.original_indices[source_index],
                "source_timestamp_s": float(observation_sequence.times[source_index]), "target_time_s": target,
                "pairing_error_s": frame["sim_time_s"] - target,
                "tool_poses_scene": poses.tolist(), "tool_velocities_scene": velocities.tolist(),
                "tool_validity": observation_sequence.valid[source_index].tolist(),
            })
        annotate_replay_frame(metadata["frames"][0], replay.start)
        if write_sdf_contact_diagnostics:
            append_contact_debug(metadata["frames"][0], 0, snapshot=True)
            reset_sdf_contact_diagnostics()
        source_indices = list(range(replay.start + args.replay_stride, replay.end + 1, args.replay_stride))
        if replay.end > replay.start and (not source_indices or source_indices[-1] != replay.end):
            source_indices.append(replay.end)
        completed = 0
        for source_index in source_indices:
            window_start_substep = completed
            target = float(observation_sequence.times[source_index] - observation_sequence.times[replay.start])
            target_substeps = int(np.ceil(target / args.dt - 1e-10))
            while completed < target_substeps:
                current_time = completed * args.dt
                set_tool_state(*replay.at(min(current_time, replay.times[-1])))
                run_substep(current_time)
                completed += 1
            set_tool_state(*replay.at(min(completed * args.dt, replay.times[-1])))
            update_tool_visuals(completed * args.dt)
            points = x.to_numpy()
            if not np.isfinite(points).all():
                raise ValueError(f"Nonfinite simulated particles at time {completed * args.dt}")
            frame = save_frame_outputs(points, args, views, run_dir, frame_idx,
                                       completed // args.substeps_per_frame, view_configs, completed)
            annotate_replay_frame(frame, source_index)
            metadata["frames"].append(frame)
            if write_sdf_contact_diagnostics:
                append_contact_debug(frame, window_start_substep)
                reset_sdf_contact_diagnostics()
            frame_idx += 1
            print(f"Replay: source={source_index} sim={frame['sim_time_s']:.6f}s lag={frame['pairing_error_s']:.6f}s", flush=True)
        metadata_path = run_dir / "camera_parameters.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        if write_sdf_contact_diagnostics:
            contact_debug_path.write_text(json.dumps(contact_debug, indent=2))
        if dough_center_tx is not None:
            dough_center_tx.close()
        replay_label = "SDF mesh-frame" if args.tool_collision == "sdf" else "proxy-tool"
        print(f"Wrote {frame_idx} timestamped {replay_label} replay frames to {run_dir}")
        return

    step = 0
    timing_start_wall = wall_time.perf_counter()
    timing_start_sim = 0.0
    timing_last_wall = timing_start_wall
    timing_last_sim = timing_start_sim

    def report_timing(sim_time):
        nonlocal timing_last_wall, timing_last_sim
        if args.timing_report_interval <= 0.0:
            return
        now_wall = wall_time.perf_counter()
        wall_dt = now_wall - timing_last_wall
        if wall_dt < args.timing_report_interval:
            return
        sim_dt = sim_time - timing_last_sim
        total_wall = max(now_wall - timing_start_wall, 1e-12)
        total_sim = sim_time - timing_start_sim
        instant_ratio = sim_dt / max(wall_dt, 1e-12)
        average_ratio = total_sim / total_wall
        suggested_deformpath_hz = 30.0 * instant_ratio
        print(
            "Timing: "
            f"sim={sim_time:.3f}s wall={total_wall:.3f}s "
            f"sim/real={instant_ratio:.3f} avg={average_ratio:.3f} "
            f"suggested_deformpath_rate={suggested_deformpath_hz:.2f} Hz",
            flush=True,
        )
        timing_last_wall = now_wall
        timing_last_sim = sim_time

    if args.gui:
        window = ti.ui.Window("Taichi Viscoelastic MPM Dough", (args.width, args.height), vsync=True)
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()
        camera.position(0., 1.35, 1.35)
        camera.lookat(*SCENE_CENTER)
        camera.up(0.0, 1.0, 0.0)
        video_writer = None
        video_frame_dir = run_dir / "video_frames"
        gui_frame_idx = 0

        if args.record_video:
            seconds_per_gui_frame = args.dt * args.substeps_per_frame * args.gui_fps_substeps
            if args.video_simulation_time:
                args.video_fps = 1.0 / seconds_per_gui_frame
            metadata["video"] = {
                "path": str(args.video_path),
                "fps": args.video_fps,
                "seconds_per_gui_frame": seconds_per_gui_frame,
                "simulation_time_playback": args.video_simulation_time,
            }
            print(
                "Recording video at "
                f"{args.video_fps:.6g} fps; each frame is {seconds_per_gui_frame:.6g} simulated seconds."
            )
            video_frame_dir.mkdir(parents=True, exist_ok=True)
            video_writer = create_video_writer(args.video_path, args.video_fps, (args.width, args.height))

        while window.running and step < args.steps:
            for _ in range(min(args.gui_fps_substeps, args.steps - step)):
                if ros_control is not None:
                    poses, linear_velocities = ros_control.poll_and_integrate(args.dt * args.substeps_per_frame)
                    set_tool_state(poses, linear_velocities)
                    ros_control.transmit_poses()
                time = step * args.dt * args.substeps_per_frame
                for _ in range(args.substeps_per_frame):
                    run_substep(time)
                    time += args.dt
                step += 1

            update_tool_visuals(step * args.dt * args.substeps_per_frame)
            if dough_center_tx is not None:
                dough_center_tx.send(x.to_numpy().mean(axis=0))
            frame_path = None
            if args.record_video:
                frame_path = video_frame_dir / f"gui_{gui_frame_idx:06d}.png"
            draw_gui_frame(
                window,
                canvas,
                scene,
                camera,
                x,
                tool_x,
                step * args.dt * args.substeps_per_frame,
                free_camera=args.free_camera,
                movement_speed=args.camera_speed,
                frame_path=frame_path,
            )

            if args.record_video:
                import cv2

                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"Could not read recorded GUI frame {frame_path}")
                video_writer.write(frame)
                if not args.keep_video_frames:
                    frame_path.unlink()
                gui_frame_idx += 1

            if not args.no_save and (step % args.save_every == 0 or step == args.steps):
                metadata["frames"].append(save_frame_outputs(x.to_numpy(), args, views, run_dir, frame_idx, step, view_configs))
                frame_idx += 1
            report_timing(step * args.dt * args.substeps_per_frame)

        if video_writer is not None:
            video_writer.release()
            print(f"Wrote GUI video to {args.video_path}")
    else:
        for step in range(1, args.steps + 1):
            if ros_control is not None:
                poses, linear_velocities = ros_control.poll_and_integrate(args.dt * args.substeps_per_frame)
                set_tool_state(poses, linear_velocities)
                ros_control.transmit_poses()
            time = (step - 1) * args.dt * args.substeps_per_frame
            for _ in range(args.substeps_per_frame):
                run_substep(time)
                time += args.dt
            update_tool_visuals(time)
            if dough_center_tx is not None:
                dough_center_tx.send(x.to_numpy().mean(axis=0))

            if not args.no_save and (step % args.save_every == 0 or step == args.steps):
                metadata["frames"].append(save_frame_outputs(x.to_numpy(), args, views, run_dir, frame_idx, step, view_configs))
                frame_idx += 1
            report_timing(step * args.dt * args.substeps_per_frame)

    metadata_path = run_dir / "camera_parameters.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {frame_idx} Taichi MPM frames to {run_dir}")
    print(f"Wrote metadata to {metadata_path}")
    if ros_control is not None:
        ros_control.close()
    if dough_center_tx is not None:
        dough_center_tx.close()


if __name__ == "__main__":
    main()
