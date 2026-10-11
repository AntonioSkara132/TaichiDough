"""Shared geometry utilities for static DeformPath-to-Taichi top-view matching."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCENE_CALIBRATION_V2 = "taichidough/scene-calibration/v2"


class CalibrationCamera(dict[str, Any]):
    """Camera values with an optional camera-optical pose kept outside JSON keys."""

    def __init__(self, values: dict[str, Any], scene_from_camera: np.ndarray | None = None):
        super().__init__(values)
        self.scene_from_camera = scene_from_camera


@dataclass(frozen=True)
class TopViewCalibration:
    schema: str | None
    name: str
    source_frame: str
    scene_frame: str
    camera: dict[str, Any]
    fingerprint: str
    scene_from_source: np.ndarray
    scene_from_camera: np.ndarray
    floor_plane_scene: np.ndarray | None
    is_metric: bool
    axis_map: str
    uniform_scale: float
    translation: tuple[float, float, float]
    provenance: dict[str, Any]
    fingerprints: dict[str, Any]
    diagnostics: dict[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _as_rigid_matrix(value: Any, field: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{field} must be a 4x4 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{field} must contain only finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0):
        raise ValueError(f"{field} must have homogeneous last row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0.0):
        raise ValueError(f"{field} rotation must be orthonormal with unit metric scale")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1e-7, rtol=0.0):
        raise ValueError(f"{field} rotation determinant must be +1, got {determinant:.9g}")
    return matrix


def _validate_intrinsic_camera(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("calibration camera must be an object")
    required = ("width", "height", "fx", "fy", "cx", "cy", "zNear", "zFar")
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"calibration camera is missing {', '.join(missing)}")
    camera = dict(value)
    for field in ("width", "height"):
        number = camera[field]
        if isinstance(number, bool) or not isinstance(number, (int, np.integer)) or int(number) <= 0:
            raise ValueError(f"calibration camera {field} must be a positive integer")
        camera[field] = int(number)
    for field in ("fx", "fy", "cx", "cy", "zNear", "zFar"):
        camera[field] = float(camera[field])
        if not np.isfinite(camera[field]):
            raise ValueError(f"calibration camera {field} must be finite")
    if camera["fx"] <= 0.0 or camera["fy"] <= 0.0:
        raise ValueError("calibration camera fx and fy must be positive")
    if camera["zNear"] < 0.0 or camera["zFar"] <= camera["zNear"]:
        raise ValueError("calibration camera must satisfy 0 <= zNear < zFar")
    return camera


def _legacy_scene_from_source(axis_map: str, uniform_scale: float, translation: tuple[float, float, float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = 0.0
    for destination, (sign, source) in enumerate(parse_axis_map(axis_map)):
        matrix[destination, source] = sign * uniform_scale
    matrix[:3, 3] = translation
    return matrix


def _legacy_scene_from_camera(camera: dict[str, Any]) -> np.ndarray:
    right, up, forward = compute_camera_basis(camera["position"], camera["lookAt"], camera.get("up"))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = right
    matrix[:3, 1] = -up
    matrix[:3, 2] = forward
    matrix[:3, 3] = np.asarray(camera["position"], dtype=np.float64)
    return matrix


def load_calibration(path: str | Path) -> TopViewCalibration:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    schema = data.get("schema")
    fingerprint = hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()

    if schema is None:
        required = {"name", "axis_map", "uniform_scale", "translation", "camera", "source_frame"}
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"{path} is missing calibration fields: {', '.join(missing)}")
        translation_values = tuple(float(value) for value in data["translation"])
        if len(translation_values) != 3 or not np.isfinite(translation_values).all():
            raise ValueError("calibration translation must contain three finite values")
        translation = (translation_values[0], translation_values[1], translation_values[2])
        uniform_scale = float(data["uniform_scale"])
        if not np.isfinite(uniform_scale) or uniform_scale <= 0:
            raise ValueError("calibration uniform_scale must be positive and finite")
        axis_map = str(data["axis_map"])
        parse_axis_map(axis_map)
        camera_values = dict(data["camera"])
        for field in ("position", "lookAt", "fieldOfView", "zNear", "zFar"):
            if field not in camera_values:
                raise ValueError(f"calibration camera is missing {field}")
        for field in ("position", "lookAt"):
            if len(camera_values[field]) != 3:
                raise ValueError(f"calibration camera {field} must contain three values")
        scene_from_camera = _legacy_scene_from_camera(camera_values)
        return TopViewCalibration(
            schema=None,
            name=str(data["name"]),
            source_frame=str(data["source_frame"]),
            scene_frame=str(data.get("scene_frame", "world")),
            camera=CalibrationCamera(camera_values, scene_from_camera),
            fingerprint=fingerprint,
            scene_from_source=_legacy_scene_from_source(axis_map, uniform_scale, translation),
            scene_from_camera=scene_from_camera,
            floor_plane_scene=None,
            is_metric=False,
            axis_map=axis_map,
            uniform_scale=uniform_scale,
            translation=translation,
            provenance={},
            fingerprints={},
            diagnostics={},
        )

    if schema != SCENE_CALIBRATION_V2:
        raise ValueError(f"Unsupported calibration schema {schema!r}")
    required = {"name", "source_frame", "scene_frame", "camera", "scene_from_source", "scene_from_camera"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{path} is missing calibration fields: {', '.join(missing)}")
    source_frame = str(data["source_frame"])
    scene_frame = str(data["scene_frame"])
    if not source_frame or not scene_frame:
        raise ValueError("source_frame and scene_frame must be nonempty")
    scene_from_source = _as_rigid_matrix(data["scene_from_source"], "scene_from_source")
    scene_from_camera = _as_rigid_matrix(data["scene_from_camera"], "scene_from_camera")
    floor_plane = data.get("floor_plane_scene")
    if floor_plane is not None:
        floor_plane = np.asarray(floor_plane, dtype=np.float64)
        if floor_plane.shape != (4,) or not np.isfinite(floor_plane).all():
            raise ValueError("floor_plane_scene must contain four finite values")
        normal_length = float(np.linalg.norm(floor_plane[:3]))
        if not np.isclose(normal_length, 1.0, atol=1e-7, rtol=0.0):
            raise ValueError("floor_plane_scene normal must have unit length")
    camera_values = _validate_intrinsic_camera(data["camera"])
    axis_map = str(data.get("axis_map", "xyz"))
    parse_axis_map(axis_map)
    uniform_scale = float(data.get("uniform_scale", 1.0))
    if not np.isfinite(uniform_scale) or not np.isclose(uniform_scale, 1.0, atol=1e-9, rtol=0.0):
        raise ValueError("v2 uniform_scale compatibility value must be 1 for metric calibration")
    translation = (
        float(scene_from_source[0, 3]),
        float(scene_from_source[1, 3]),
        float(scene_from_source[2, 3]),
    )
    return TopViewCalibration(
        schema=SCENE_CALIBRATION_V2,
        name=str(data["name"]),
        source_frame=source_frame,
        scene_frame=scene_frame,
        camera=CalibrationCamera(
            {**camera_values, "scene_from_camera": scene_from_camera.tolist()}, scene_from_camera
        ),
        fingerprint=fingerprint,
        scene_from_source=scene_from_source,
        scene_from_camera=scene_from_camera,
        floor_plane_scene=floor_plane,
        is_metric=True,
        axis_map=axis_map,
        uniform_scale=uniform_scale,
        translation=translation,
        provenance=dict(data.get("provenance", {})),
        fingerprints=dict(data.get("fingerprints", {})),
        diagnostics=dict(data.get("diagnostics", {})),
    )


def calibration_metadata(calibration: TopViewCalibration) -> dict[str, Any]:
    if calibration.schema != SCENE_CALIBRATION_V2:
        return {
            "name": calibration.name,
            "source_frame": calibration.source_frame,
            "axis_map": calibration.axis_map,
            "uniform_scale": calibration.uniform_scale,
            "translation": list(calibration.translation),
            "camera": dict(calibration.camera),
            "fingerprint": calibration.fingerprint,
        }
    result: dict[str, Any] = {
        "schema": calibration.schema,
        "name": calibration.name,
        "source_frame": calibration.source_frame,
        "scene_frame": calibration.scene_frame,
        "scene_from_source": calibration.scene_from_source.tolist(),
        "scene_from_camera": calibration.scene_from_camera.tolist(),
        "floor_plane_scene": None if calibration.floor_plane_scene is None else calibration.floor_plane_scene.tolist(),
        "camera": dict(calibration.camera),
        "is_metric": calibration.is_metric,
        "fingerprint": calibration.fingerprint,
    }
    if calibration.provenance:
        result["provenance"] = calibration.provenance
    if calibration.fingerprints:
        result["fingerprints"] = calibration.fingerprints
    if calibration.diagnostics:
        result["diagnostics"] = calibration.diagnostics
    return result


def parse_axis_map(axis_map: str) -> list[tuple[float, int]]:
    compact = axis_map.replace(",", "").replace(" ", "").lower()
    result: list[tuple[float, int]] = []
    index = 0
    while index < len(compact):
        sign = 1.0
        if compact[index] == "-":
            sign = -1.0
            index += 1
        elif compact[index] == "+":
            index += 1
        if index >= len(compact) or compact[index] not in "xyz":
            raise ValueError(f"Invalid axis map {axis_map!r}; use xyz, xzy, or x-zy.")
        result.append((sign, "xyz".index(compact[index])))
        index += 1
    if len(result) != 3 or sorted(source for _, source in result) != [0, 1, 2]:
        raise ValueError(f"Invalid axis map {axis_map!r}; each source axis must be used once.")
    return result


def rigid_inverse(transform: np.ndarray) -> np.ndarray:
    """Return the inverse of a rigid 4x4 transform."""
    matrix = _as_rigid_matrix(transform, "transform")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return result


def compose_rigid_transforms(*transforms: np.ndarray) -> np.ndarray:
    """Compose transforms in matrix order, from rightmost source to leftmost target."""
    result = np.eye(4, dtype=np.float64)
    for index, transform in enumerate(transforms):
        result = result @ _as_rigid_matrix(transform, f"transforms[{index}]")
    return _as_rigid_matrix(result, "composed transform")


def transform_points(points: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.shape == () or values.shape[-1] != 3:
        raise ValueError(f"points must end in three coordinates, got {values.shape}")
    matrix = _as_rigid_matrix(target_from_source, "target_from_source")
    return values @ matrix[:3, :3].T + matrix[:3, 3]


def transform_vectors(vectors: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.shape == () or values.shape[-1] != 3:
        raise ValueError(f"vectors must end in three coordinates, got {values.shape}")
    matrix = _as_rigid_matrix(target_from_source, "target_from_source")
    return values @ matrix[:3, :3].T


def transform_plane(plane_source: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    plane = np.asarray(plane_source, dtype=np.float64)
    if plane.shape != (4,) or not np.isfinite(plane).all():
        raise ValueError("plane_source must contain four finite values")
    normal_length = float(np.linalg.norm(plane[:3]))
    if normal_length <= 0.0:
        raise ValueError("plane_source normal must be nonzero")
    normalized = plane / normal_length
    transformed = rigid_inverse(target_from_source).T @ normalized
    transformed /= np.linalg.norm(transformed[:3])
    return transformed


def transform_rigid_pose(pose_source: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    """Transform a rigid pose matrix from source coordinates into target coordinates."""
    return compose_rigid_transforms(target_from_source, pose_source)


# Descriptive aliases used by callers that prefer longer names.
invert_rigid_transform = rigid_inverse
transform_pose = transform_rigid_pose


def project_camera_points(points_camera_optical: np.ndarray, camera: dict[str, Any]) -> np.ndarray:
    """Project optical-frame points to [u, v, depth] using exact pinhole intrinsics."""
    intrinsic = _validate_intrinsic_camera(camera)
    points = np.asarray(points_camera_optical, dtype=np.float64)
    if points.shape == () or points.shape[-1] != 3:
        raise ValueError(f"points_camera_optical must end in three coordinates, got {points.shape}")
    depth = points[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intrinsic["fx"] * points[..., 0] / depth + intrinsic["cx"]
        v = intrinsic["fy"] * points[..., 1] / depth + intrinsic["cy"]
    return np.stack([u, v, depth], axis=-1)


def unproject_camera_points(pixels: np.ndarray, depth: np.ndarray, camera: dict[str, Any]) -> np.ndarray:
    """Unproject optical pixels and metric depth into camera-optical coordinates."""
    intrinsic = _validate_intrinsic_camera(camera)
    pixel_values = np.asarray(pixels, dtype=np.float64)
    depth_values = np.asarray(depth, dtype=np.float64)
    if pixel_values.shape == () or pixel_values.shape[-1] != 2:
        raise ValueError(f"pixels must end in two coordinates, got {pixel_values.shape}")
    try:
        depth_values = np.broadcast_to(depth_values, pixel_values.shape[:-1])
    except ValueError as exc:
        raise ValueError("depth must broadcast to the pixel dimensions") from exc
    x = (pixel_values[..., 0] - intrinsic["cx"]) * depth_values / intrinsic["fx"]
    y = (pixel_values[..., 1] - intrinsic["cy"]) * depth_values / intrinsic["fy"]
    return np.stack([x, y, depth_values], axis=-1)


camera_project = project_camera_points
camera_unproject = unproject_camera_points


def apply_calibration(points: np.ndarray, calibration: TopViewCalibration) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected [N, >=3] points, got {points.shape}")
    if calibration.schema == SCENE_CALIBRATION_V2:
        return transform_points(points[:, :3], calibration.scene_from_source).astype(np.float32)
    mapped = np.empty((len(points), 3), dtype=np.float32)
    for destination, (sign, source) in enumerate(parse_axis_map(calibration.axis_map)):
        mapped[:, destination] = sign * points[:, source]
    return mapped * calibration.uniform_scale + np.asarray(calibration.translation, dtype=np.float32)


def load_deformpath_frame(episode_dir: str | Path, pointclouds_name: str, frame_index: int) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Loading DeformPath .pt files requires torch.") from exc

    path = Path(episode_dir) / pointclouds_name
    sequence: Any = torch.load(path, map_location="cpu")
    if isinstance(sequence, list) and len(sequence) == 1 and isinstance(sequence[0], list):
        sequence = sequence[0]
    if isinstance(sequence, dict):
        for key in ("pointclouds", "points", "frames"):
            if key in sequence:
                sequence = sequence[key]
                break
    if not isinstance(sequence, (list, tuple)):
        raise ValueError(f"Unsupported pointcloud structure in {path}")
    if not 0 <= frame_index < len(sequence):
        raise IndexError(f"Frame {frame_index} is outside sequence length {len(sequence)}")

    frame = sequence[frame_index]
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    return np.asarray(frame, dtype=np.float32)


def filter_xyz(points: np.ndarray, trim_quantile: float = 0.005) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected [N, >=3] points, got {points.shape}")
    xyz = points[:, :3]
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if trim_quantile > 0 and len(xyz) > 10:
        quantile = float(np.clip(trim_quantile, 0.0, 0.2))
        low = np.quantile(xyz, quantile, axis=0)
        high = np.quantile(xyz, 1.0 - quantile, axis=0)
        xyz = xyz[np.all((xyz >= low) & (xyz <= high), axis=1)]
    if len(xyz) == 0:
        raise ValueError("No finite points remain after filtering")
    return xyz.astype(np.float32, copy=False)


def compute_camera_basis(position: Any, look_at: Any, up_hint: Any | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.asarray(position, dtype=np.float32)
    look_at = np.asarray(look_at, dtype=np.float32)
    forward = look_at - position
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError("Camera position and lookAt must differ")
    forward /= forward_norm
    up = np.asarray(up_hint if up_hint is not None else [0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    corrected_up = np.cross(right, forward)
    return right, corrected_up, forward


def _intrinsic_camera_pose(camera: dict[str, Any]) -> np.ndarray | None:
    value = getattr(camera, "scene_from_camera", None)
    if value is None and "scene_from_camera" in camera:
        value = camera["scene_from_camera"]
    return None if value is None else _as_rigid_matrix(value, "camera scene_from_camera")


def _has_exact_intrinsics(camera: dict[str, Any]) -> bool:
    return all(field in camera for field in ("width", "height", "fx", "fy", "cx", "cy", "zNear", "zFar"))


def rasterize_depth(
    points: np.ndarray,
    width: int,
    height: int,
    camera: dict[str, Any],
    splat_radius: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if width <= 0 or height <= 0:
        raise ValueError("Depth dimensions must be positive")
    point_values = np.asarray(points, dtype=np.float32)
    if point_values.ndim != 2 or point_values.shape[1] < 3:
        raise ValueError(f"Expected [N, >=3] points, got {point_values.shape}")

    exact_intrinsics = _has_exact_intrinsics(camera)
    if exact_intrinsics:
        intrinsic = _validate_intrinsic_camera(camera)
        if width != intrinsic["width"] or height != intrinsic["height"]:
            raise ValueError(
                f"Depth dimensions {width}x{height} do not match calibrated camera "
                f"{intrinsic['width']}x{intrinsic['height']}"
            )
        scene_from_camera = _intrinsic_camera_pose(camera)
        if scene_from_camera is None:
            scene_from_camera = np.eye(4, dtype=np.float64)
        camera_from_scene = rigid_inverse(scene_from_camera)
        camera_xyz = transform_points(point_values[:, :3], camera_from_scene)
        projected = project_camera_points(camera_xyz, intrinsic)
        cam_x = camera_xyz[:, 0]
        cam_y = -camera_xyz[:, 1]
        cam_z = camera_xyz[:, 2]
        pixel_u = projected[:, 0]
        pixel_v = projected[:, 1]
        right = scene_from_camera[:3, 0].astype(np.float32)
        up = (-scene_from_camera[:3, 1]).astype(np.float32)
        forward = scene_from_camera[:3, 2].astype(np.float32)
        camera_pos = scene_from_camera[:3, 3].astype(np.float32)
    else:
        right, up, forward = compute_camera_basis(camera["position"], camera["lookAt"], camera.get("up"))
        camera_pos = np.asarray(camera["position"], dtype=np.float32)
        rel = point_values[:, :3] - camera_pos[None, :]
        cam_x = rel @ right
        cam_y = rel @ up
        cam_z = rel @ forward
        focal = height / (2.0 * np.tan(np.radians(float(camera["fieldOfView"])) / 2.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            pixel_u = focal * cam_x / cam_z + width * 0.5
            pixel_v = height * 0.5 - focal * cam_y / cam_z

    finite_projection = np.isfinite(pixel_u) & np.isfinite(pixel_v)
    pixel_x = np.full(len(point_values), -1, dtype=np.int32)
    pixel_y = np.full(len(point_values), -1, dtype=np.int32)
    if exact_intrinsics:
        pixel_x[finite_projection] = np.floor(pixel_u[finite_projection]).astype(np.int32)
        pixel_y[finite_projection] = np.floor(pixel_v[finite_projection]).astype(np.int32)
    else:
        # NumPy's conversion truncates toward zero; retain the original v1 edge behavior.
        pixel_x[finite_projection] = pixel_u[finite_projection].astype(np.int32)
        pixel_y[finite_projection] = pixel_v[finite_projection].astype(np.int32)
    valid = (
        np.isfinite(cam_z)
        & (cam_z > float(camera["zNear"]))
        & (cam_z < float(camera["zFar"]))
        & (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    depth = np.full((height, width), float(camera["zFar"]), dtype=np.float32)
    nearest_indices = np.full((height, width), -1, dtype=np.int32)
    radius = max(0, int(splat_radius))
    radius_sq = radius * radius
    for point_index in np.flatnonzero(valid):
        px, py, z = int(pixel_x[point_index]), int(pixel_y[point_index]), cam_z[point_index]
        for offset_y in range(-radius, radius + 1):
            yy = py + offset_y
            if yy < 0 or yy >= height:
                continue
            for offset_x in range(-radius, radius + 1):
                if offset_x * offset_x + offset_y * offset_y > radius_sq:
                    continue
                xx = px + offset_x
                if xx < 0 or xx >= width:
                    continue
                if z < depth[yy, xx]:
                    depth[yy, xx] = z
                    nearest_indices[yy, xx] = point_index
    return depth, nearest_indices, right, up, forward, camera_pos


def depth_to_pointcloud(
    depth: np.ndarray,
    nearest_indices: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
    camera_pos: np.ndarray,
    camera: dict[str, Any],
    max_points: int = 0,
    frame: str = "world",
) -> np.ndarray:
    pixel_y, pixel_x = np.nonzero(nearest_indices >= 0)
    if len(pixel_x) == 0:
        return np.empty((0, 3), dtype=np.float32)
    z = depth[pixel_y, pixel_x]
    height, width = depth.shape
    if _has_exact_intrinsics(camera):
        intrinsic = _validate_intrinsic_camera(camera)
        if width != intrinsic["width"] or height != intrinsic["height"]:
            raise ValueError(
                f"Depth dimensions {width}x{height} do not match calibrated camera "
                f"{intrinsic['width']}x{intrinsic['height']}"
            )
        pixels = np.stack(
            [pixel_x.astype(np.float64) + 0.5, pixel_y.astype(np.float64) + 0.5], axis=1
        )
        optical_xyz = unproject_camera_points(pixels, z, intrinsic).astype(np.float32)
        camera_xyz = optical_xyz * np.array([1.0, -1.0, 1.0], dtype=np.float32)
        if frame == "camera":
            xyz = camera_xyz
        elif frame == "camera_optical":
            xyz = optical_xyz
        elif frame == "world":
            scene_from_camera = _intrinsic_camera_pose(camera)
            if scene_from_camera is None:
                scene_from_camera = np.eye(4, dtype=np.float64)
            xyz = transform_points(optical_xyz, scene_from_camera)
        else:
            raise ValueError(f"Unsupported pointcloud frame {frame!r}")
    else:
        focal = height / (2.0 * np.tan(np.radians(float(camera["fieldOfView"])) / 2.0))
        cam_x = (pixel_x.astype(np.float32) + 0.5 - width * 0.5) * z / focal
        cam_y = (height * 0.5 - (pixel_y.astype(np.float32) + 0.5)) * z / focal
        camera_xyz = np.stack([cam_x, cam_y, z], axis=1).astype(np.float32)
        if frame == "camera":
            xyz = camera_xyz
        elif frame == "camera_optical":
            xyz = camera_xyz * np.array([1.0, -1.0, 1.0], dtype=np.float32)
        elif frame == "world":
            xyz = camera_pos[None, :] + cam_x[:, None] * right[None, :] + cam_y[:, None] * up[None, :] + z[:, None] * forward[None, :]
        else:
            raise ValueError(f"Unsupported pointcloud frame {frame!r}")
    if max_points > 0 and len(xyz) > max_points:
        indices = np.linspace(0, len(xyz) - 1, max_points).round().astype(np.int64)
        xyz = xyz[indices]
    return np.asarray(xyz, dtype=np.float32)


def format_pointcloud(xyz: np.ndarray, pointcloud_format: str) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    if pointcloud_format == "xyz":
        return xyz
    if pointcloud_format == "deformpath7":
        extra = np.zeros((len(xyz), 4), dtype=np.float32)
        extra[:, 2] = 1.0
        return np.concatenate([xyz, extra], axis=1)
    raise ValueError(f"Unsupported pointcloud format {pointcloud_format!r}")


def camera_pointcloud(points: np.ndarray, camera: dict[str, Any], width: int, height: int, splat_radius: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth, nearest, right, up, forward, position = rasterize_depth(points, width, height, camera, splat_radius)
    return depth_to_pointcloud(depth, nearest, right, up, forward, position, camera, frame="camera"), depth, nearest


def fixed_grid_metrics(reference: np.ndarray, candidate: np.ndarray, cell_size: float) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32)[:, :3]
    candidate = np.asarray(candidate, dtype=np.float32)[:, :3]
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    if len(reference) == 0 or len(candidate) == 0:
        raise ValueError("Both point clouds must be nonempty")

    ref_min = reference[:, :2].min(axis=0)
    ref_max = reference[:, :2].max(axis=0)
    all_xy = np.concatenate([reference[:, :2], candidate[:, :2]], axis=0)
    padding_cells = 2
    lower = (np.floor(all_xy.min(axis=0) / cell_size) - padding_cells) * cell_size
    upper = (np.ceil(all_xy.max(axis=0) / cell_size) + padding_cells) * cell_size
    shape = np.maximum(np.rint((upper - lower) / cell_size).astype(int), 1)

    def grid(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cells = np.floor((points[:, :2] - lower) / cell_size).astype(np.int64)
        valid = np.all((cells >= 0) & (cells < shape), axis=1)
        image = np.zeros(tuple(shape), dtype=bool)
        depth = np.full(tuple(shape), np.nan, dtype=np.float32)
        for (ix, iy), z in zip(cells[valid], points[valid, 2]):
            image[ix, iy] = True
            if not np.isfinite(depth[ix, iy]) or z < depth[ix, iy]:
                depth[ix, iy] = z
        return image, depth

    ref_occupancy, ref_depth = grid(reference)
    candidate_occupancy, candidate_depth = grid(candidate)
    intersection = ref_occupancy & candidate_occupancy
    union = ref_occupancy | candidate_occupancy
    common_depth = ref_depth[intersection] - candidate_depth[intersection]
    ref_extent = ref_max - ref_min
    candidate_extent = candidate[:, :2].max(axis=0) - candidate[:, :2].min(axis=0)

    result: dict[str, Any] = {
        "cell_size": cell_size,
        "grid_lower_xy": lower.tolist(),
        "grid_upper_xy": upper.tolist(),
        "grid_shape": shape.tolist(),
        "reference_cells": int(ref_occupancy.sum()),
        "candidate_cells": int(candidate_occupancy.sum()),
        "intersection_cells": int(intersection.sum()),
        "union_cells": int(union.sum()),
        "footprint_iou": float(intersection.sum() / max(union.sum(), 1)),
        "reference_extent_xy": ref_extent.tolist(),
        "candidate_extent_xy": candidate_extent.tolist(),
        "extent_ratio_xy": (candidate_extent / np.maximum(ref_extent, 1e-8)).tolist(),
        "reference_area": float(np.prod(ref_extent)),
        "candidate_area": float(np.prod(candidate_extent)),
        "area_ratio": float(np.prod(candidate_extent) / max(np.prod(ref_extent), 1e-8)),
        "centroid_delta_xy": (candidate[:, :2].mean(axis=0) - reference[:, :2].mean(axis=0)).tolist(),
        "common_footprint_coverage": float(intersection.sum() / max(ref_occupancy.sum(), 1)),
    }
    if len(common_depth):
        result.update(
            depth_bias=float(common_depth.mean()),
            depth_median_absolute_error=float(np.median(np.abs(common_depth))),
            depth_p95_absolute_error=float(np.quantile(np.abs(common_depth), 0.95)),
        )
    else:
        result.update(depth_bias=None, depth_median_absolute_error=None, depth_p95_absolute_error=None)
    return result
