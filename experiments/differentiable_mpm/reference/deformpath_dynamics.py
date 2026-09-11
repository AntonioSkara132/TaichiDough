"""Timestamped DeformPath observations and deterministic selected-tool-frame replay."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from deformpath_topview import TopViewCalibration, apply_calibration, load_calibration, parse_axis_map
except ImportError:
    from .deformpath_topview import TopViewCalibration, apply_calibration, load_calibration, parse_axis_map


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact(value: str, metadata_path: Path) -> Path:
    path = Path(value)
    for candidate in (path, metadata_path.parent / path, metadata_path.parent / path.name):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve {value!r} from {metadata_path}")


def resolve_episode_calibration(episode_dir: Path) -> tuple[TopViewCalibration, dict[str, Any]]:
    """Load and verify the metric calibration copied into a processed episode."""
    episode_dir = Path(episode_dir).resolve()
    metadata_path = episode_dir / "sequence_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Processed episode metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    record = metadata.get("calibration")
    if not isinstance(record, dict) or record.get("status") != "available":
        reason = record.get("reason", "no attached calibration") if isinstance(record, dict) else "no calibration record"
        raise ValueError(f"Processed episode has no usable metric calibration: {reason}")
    artifact = resolve_artifact(str(record.get("path", "")), metadata_path)
    if record.get("sha256") != file_fingerprint(artifact):
        raise ValueError("Episode calibration SHA-256 does not match sequence metadata")
    calibration = load_calibration(artifact)
    if not calibration.is_metric or calibration.schema != "taichidough/scene-calibration/v2":
        raise ValueError("Episode calibration must use metric taichidough/scene-calibration/v2")
    if record.get("fingerprint") != calibration.fingerprint:
        raise ValueError("Episode calibration fingerprint does not match sequence metadata")
    if calibration.source_frame != "mocap" or calibration.scene_frame != "mocap":
        raise ValueError("Episode calibration must use mocap as both source and scene frames")
    if calibration.floor_plane_scene is None:
        raise ValueError("Episode calibration must include floor_plane_scene")
    return calibration, {"metadata_path": str(metadata_path), "artifact_path": str(artifact), **record}


def _normalized_plane(value: Any, label: str) -> np.ndarray:
    plane = np.asarray(value, dtype=np.float64)
    if plane.shape != (4,) or not np.isfinite(plane).all():
        raise ValueError(f"{label} must contain four finite coefficients")
    length = float(np.linalg.norm(plane[:3]))
    if length <= 1e-12:
        raise ValueError(f"{label} normal must be nonzero")
    return plane / length


def load_scene_point_filter(
    simulation_metadata: dict[str, Any],
    simulation_metadata_path: Path,
    calibration: TopViewCalibration,
) -> ScenePointFilter | None:
    initialization = simulation_metadata.get("initialization")
    if initialization is None:
        return None
    if not isinstance(initialization, dict) or not initialization.get("metadata_path"):
        raise ValueError("Simulation initialization metadata is incomplete")
    reconstruction_path = resolve_artifact(initialization["metadata_path"], simulation_metadata_path)
    expected_fingerprint = initialization.get("metadata_sha256")
    actual_fingerprint = file_fingerprint(reconstruction_path)
    if expected_fingerprint != actual_fingerprint:
        raise ValueError("Reconstruction metadata fingerprint does not match the simulation")
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    if reconstruction.get("schema") != "voxel_dough_reconstruction/v2":
        raise ValueError("Scene filtering requires voxel_dough_reconstruction/v2 metadata")
    if reconstruction.get("calibration_fingerprint") != calibration.fingerprint:
        raise ValueError("Reconstruction calibration fingerprint does not match evaluation calibration")
    fill = reconstruction.get("fill")
    if not isinstance(fill, dict) or fill.get("mode") != "floor":
        raise ValueError("Scene filtering requires floor-mode reconstruction metadata")
    plane = _normalized_plane(fill.get("floor_plane_scene"), "Reconstruction floor plane")
    calibration_plane = getattr(calibration, "floor_plane_scene", None)
    if calibration_plane is None or not np.allclose(
        plane, _normalized_plane(calibration_plane, "Calibration floor plane"), atol=1e-7, rtol=0
    ):
        raise ValueError("Reconstruction floor plane does not match evaluation calibration")
    clearance_value = fill.get("floor_clearance_m")
    if clearance_value is None:
        raise ValueError("Reconstruction floor clearance is missing")
    clearance = float(clearance_value)
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError("Reconstruction floor clearance must be finite and non-negative")
    bounds = fill.get("scene_bounds")
    bounds_min = bounds_max = None
    if bounds is not None:
        if not isinstance(bounds, dict):
            raise ValueError("Reconstruction scene bounds must contain min and max vectors")
        bounds_min = np.asarray(bounds.get("min"), dtype=np.float64)
        bounds_max = np.asarray(bounds.get("max"), dtype=np.float64)
        if (
            bounds_min.shape != (3,)
            or bounds_max.shape != (3,)
            or not np.isfinite(bounds_min).all()
            or not np.isfinite(bounds_max).all()
            or np.any(bounds_min >= bounds_max)
        ):
            raise ValueError("Reconstruction scene bounds must be finite ordered XYZ vectors")
    scene_frame = str((reconstruction.get("calibration") or {}).get("scene_frame", ""))
    if not scene_frame or scene_frame != getattr(calibration, "scene_frame", scene_frame):
        raise ValueError("Reconstruction scene frame does not match evaluation calibration")
    return ScenePointFilter(
        floor_plane_scene=plane,
        floor_clearance_m=clearance,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        reconstruction_metadata_sha256=actual_fingerprint,
        scene_frame=scene_frame,
    )


def filter_scene_points(points: np.ndarray, point_filter: ScenePointFilter | None) -> tuple[np.ndarray, dict[str, int]]:
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Scene points must have shape [N, 3], got {values.shape}")
    finite = np.isfinite(values).all(axis=1)
    if point_filter is None:
        kept = finite
    else:
        signed_distance = values @ point_filter.floor_plane_scene[:3] + point_filter.floor_plane_scene[3]
        kept = finite & np.isfinite(signed_distance) & (signed_distance >= point_filter.floor_clearance_m)
        if point_filter.bounds_min is not None and point_filter.bounds_max is not None:
            kept &= np.all(
                (values >= point_filter.bounds_min[None, :]) & (values <= point_filter.bounds_max[None, :]),
                axis=1,
            )
    return values[kept], {
        "input": int(len(values)),
        "nonfinite": int((~finite).sum()),
        "kept": int(kept.sum()),
        "rejected": int((~kept).sum()),
    }


def quaternion_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("Quaternion must be finite and nonzero")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def matrix_quaternion(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    # The eigenvector form remains well-conditioned at rotations of pi.
    k = np.array([
        [m[0, 0]-m[1, 1]-m[2, 2], m[0, 1]+m[1, 0], m[0, 2]+m[2, 0], m[2, 1]-m[1, 2]],
        [m[0, 1]+m[1, 0], m[1, 1]-m[0, 0]-m[2, 2], m[1, 2]+m[2, 1], m[0, 2]-m[2, 0]],
        [m[0, 2]+m[2, 0], m[1, 2]+m[2, 1], m[2, 2]-m[0, 0]-m[1, 1], m[1, 0]-m[0, 1]],
        [m[2, 1]-m[1, 2], m[0, 2]-m[2, 0], m[1, 0]-m[0, 1], np.trace(m)],
    ]) / 3.0
    q = np.linalg.eigh(k)[1][:, -1]
    return q if q[3] >= 0 else -q


def slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
    dot = float(q0 @ q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = (1 - fraction) * q0 + fraction * q1
        return q / np.linalg.norm(q)
    angle = np.arccos(np.clip(dot, -1, 1))
    return (np.sin((1 - fraction) * angle) * q0 + np.sin(fraction * angle) * q1) / np.sin(angle)


@dataclass
class ObservationSequence:
    points: list[np.ndarray]
    poses: np.ndarray
    valid: np.ndarray
    times: np.ndarray
    names: list[str]
    original_indices: list[int]
    episode_dir: Path
    fingerprint: str


@dataclass(frozen=True)
class ToolGeometry:
    names: tuple[str, str]
    half_extents_m: np.ndarray
    marker_from_collider: np.ndarray
    marker_from_mesh: np.ndarray
    fingerprint: str
    source: str
    proxy: bool


@dataclass(frozen=True)
class ScenePointFilter:
    floor_plane_scene: np.ndarray
    floor_clearance_m: float
    bounds_min: np.ndarray | None
    bounds_max: np.ndarray | None
    reconstruction_metadata_sha256: str
    scene_frame: str


def _rigid_matrix(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0):
        raise ValueError(f"{label} must have homogeneous last row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7, rtol=0):
        raise ValueError(f"{label} rotation must have determinant +1")
    return matrix


def load_tool_geometry(path: str | Path, expected_names: list[str] | tuple[str, ...] | None = None) -> ToolGeometry:
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "taichidough/tool-geometry/v1":
        raise ValueError("Tool geometry must use schema taichidough/tool-geometry/v1")
    rows = data.get("tools")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("Tool geometry must define exactly two tools")
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"]:
            raise ValueError("Every tool geometry entry needs a nonempty name")
        if row["name"] in by_name:
            raise ValueError(f"Duplicate tool geometry name {row['name']!r}")
        by_name[row["name"]] = row
    names = tuple(expected_names) if expected_names is not None else tuple(by_name)
    if len(names) != 2 or set(names) != set(by_name):
        raise ValueError(f"Tool geometry names {sorted(by_name)} do not match pose streams {list(names)}")
    ordered = [by_name[name] for name in names]
    half_extents = np.asarray([row.get("half_extents_m") for row in ordered], dtype=np.float64)
    if half_extents.shape != (2, 3) or not np.isfinite(half_extents).all() or np.any(half_extents <= 0):
        raise ValueError("Tool half_extents_m must contain two positive finite XYZ vectors")
    marker_from_collider = np.stack(
        [_rigid_matrix(row.get("marker_from_collider"), f"marker_from_collider for {row['name']}") for row in ordered]
    )
    marker_from_mesh = np.stack([
        _rigid_matrix(row.get("marker_from_mesh", np.eye(4)), f"marker_from_mesh for {row['name']}")
        for row in ordered
    ])
    return ToolGeometry(
        names=(str(names[0]), str(names[1])),
        half_extents_m=half_extents.astype(np.float32),
        marker_from_collider=marker_from_collider,
        marker_from_mesh=marker_from_mesh,
        fingerprint=file_fingerprint(path),
        source=str(path),
        proxy=bool(data.get("proxy", False)),
    )


def tool_geometry_from_metadata(
    data: dict[str, Any],
    expected_names: list[str] | tuple[str, ...] | None = None,
) -> ToolGeometry:
    if data.get("schema") != "taichidough/tool-geometry/v1":
        raise ValueError("Embedded tool geometry must use schema taichidough/tool-geometry/v1")
    raw_names = data.get("names")
    if not isinstance(raw_names, list) or len(raw_names) != 2 or any(
        not isinstance(name, str) or not name for name in raw_names
    ) or len(set(raw_names)) != 2:
        raise ValueError("Embedded tool geometry must contain two unique nonempty names")
    names = tuple(expected_names) if expected_names is not None else tuple(raw_names)
    if len(names) != 2 or set(names) != set(raw_names):
        raise ValueError(f"Tool geometry names {raw_names} do not match pose streams {list(names)}")
    order = [raw_names.index(name) for name in names]
    half_extents = np.asarray(data.get("half_extents_m"), dtype=np.float64)
    marker_from_collider = np.asarray(data.get("marker_from_collider"), dtype=np.float64)
    marker_from_mesh = np.asarray(data.get("marker_from_mesh", np.broadcast_to(np.eye(4), (2, 4, 4))), dtype=np.float64)
    if half_extents.shape != (2, 3) or not np.isfinite(half_extents).all() or np.any(half_extents <= 0):
        raise ValueError("Embedded tool half_extents_m must contain two positive finite XYZ vectors")
    if marker_from_collider.shape != (2, 4, 4):
        raise ValueError("Embedded marker_from_collider must have shape [2, 4, 4]")
    if marker_from_mesh.shape != (2, 4, 4):
        raise ValueError("Embedded marker_from_mesh must have shape [2, 4, 4]")
    marker_from_collider = np.stack([
        _rigid_matrix(marker_from_collider[index], f"marker_from_collider for {raw_names[index]}")
        for index in order
    ])
    marker_from_mesh = np.stack([
        _rigid_matrix(marker_from_mesh[index], f"marker_from_mesh for {raw_names[index]}")
        for index in order
    ])
    fingerprint = data.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Embedded tool geometry needs a nonempty fingerprint")
    source = data.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError("Embedded tool geometry needs a nonempty source")
    return ToolGeometry(
        names=(str(names[0]), str(names[1])),
        half_extents_m=half_extents[order].astype(np.float32),
        marker_from_collider=marker_from_collider,
        marker_from_mesh=marker_from_mesh,
        fingerprint=fingerprint,
        source=source,
        proxy=bool(data.get("proxy", False)),
    )


def legacy_tool_geometry(
    names: list[str] | tuple[str, str],
    half_extents_m: Any,
    marker_offset_m: Any,
) -> ToolGeometry:
    if len(names) != 2 or len(set(names)) != 2 or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Legacy tool geometry requires two uniquely named tools")
    half = np.asarray(half_extents_m, dtype=np.float64)
    if half.shape == (3,):
        half = np.broadcast_to(half, (2, 3)).copy()
    if half.shape != (2, 3) or not np.isfinite(half).all() or np.any(half <= 0):
        raise ValueError("Legacy tool half-extents must be one or two positive finite XYZ vectors")
    offset = np.asarray(marker_offset_m, dtype=np.float64)
    if offset.shape == (3,):
        offset = np.broadcast_to(offset, (2, 3)).copy()
    if offset.shape != (2, 3) or not np.isfinite(offset).all():
        raise ValueError("Legacy tool marker offsets must be one or two finite XYZ vectors")
    marker_from_collider = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    marker_from_collider[:, :3, 3] = offset
    marker_from_mesh = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    payload = {
        "names": list(names),
        "half_extents_m": half.tolist(),
        "marker_offsets_m": offset.tolist(),
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return ToolGeometry(
        names=(str(names[0]), str(names[1])),
        half_extents_m=half.astype(np.float32),
        marker_from_collider=marker_from_collider,
        marker_from_mesh=marker_from_mesh,
        fingerprint=fingerprint,
        source="legacy-cli",
        proxy=True,
    )


def tool_geometry_metadata(geometry: ToolGeometry) -> dict[str, Any]:
    return {
        "schema": "taichidough/tool-geometry/v1",
        "names": list(geometry.names),
        "half_extents_m": geometry.half_extents_m.tolist(),
        "marker_from_collider": geometry.marker_from_collider.tolist(),
        "marker_from_mesh": geometry.marker_from_mesh.tolist(),
        "fingerprint": geometry.fingerprint,
        "source": geometry.source,
        "proxy": geometry.proxy,
    }


def load_torch_archive(torch_module: Any, path: Path) -> Any:
    """Load tensor-only archives on current and older PyTorch releases."""
    try:
        return torch_module.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch_module.load(path, map_location="cpu")


def load_observation_sequence(episode_dir: Path, pointclouds_name: str = "pointclouds_interpolated.pt") -> ObservationSequence:
    import torch

    episode_dir = Path(episode_dir).resolve()
    points_path = episode_dir / pointclouds_name
    paths_path = episode_dir / "paths_interpolated.pt"
    clouds = load_torch_archive(torch, points_path)
    if isinstance(clouds, list) and len(clouds) == 1 and isinstance(clouds[0], list):
        clouds = clouds[0]
    paths = load_torch_archive(torch, paths_path)
    if isinstance(paths, list) and len(paths) == 1:
        paths = paths[0]
    if not isinstance(clouds, (list, tuple)) or not isinstance(paths, dict):
        raise ValueError("Expected an interpolated pointcloud episode and timestamped path dictionary")
    poses = np.asarray(paths["path"], dtype=np.float64)
    valid = np.asarray(paths["stream_validity"], dtype=bool)
    names = list(paths["pose_frames"])
    if poses.shape != (len(clouds), 2, 14) or valid.shape != (len(clouds), 2) or len(names) != 2 or len(clouds) == 0:
        raise ValueError("Expected matching nonempty pointclouds, two [T, 2, 14] tool streams and validity masks")
    timestamps = poses[:, :, 13]
    if not np.isfinite(timestamps).all() or not np.allclose(timestamps[:, 0], timestamps[:, 1], atol=1e-6, rtol=0):
        raise ValueError("Tool streams must share finite observation timestamps")
    times = timestamps[:, 0]
    if np.any(np.diff(times) <= 0):
        raise ValueError("Observation timestamps must be strictly increasing")
    points = [np.asarray(cloud, dtype=np.float32) for cloud in clouds]
    if any(p.ndim != 2 or p.shape[1] < 3 for p in points):
        raise ValueError("Every observation must have shape [N, >=3]")
    metadata_path = episode_dir / "sequence_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    indices = metadata.get("pointcloud_indices", list(range(len(points))))
    if len(indices) != len(points):
        raise ValueError("Original-frame index mapping does not match the observation count")
    fingerprint = hashlib.sha256((file_fingerprint(points_path) + file_fingerprint(paths_path)).encode()).hexdigest()
    return ObservationSequence(points, poses, valid, times, names, indices, episode_dir, fingerprint)


class ToolReplay:
    def __init__(
        self,
        sequence: ObservationSequence,
        calibration: TopViewCalibration,
        start: int,
        end: int,
        marker_offset: np.ndarray | None = None,
        max_gap_s: float = 0.1,
        marker_from_tool_frames: np.ndarray | None = None,
        marker_from_colliders: np.ndarray | None = None,
    ):
        if not 0 <= start <= end < len(sequence.times):
            raise ValueError("Replay frame range is outside the observation sequence")
        if not np.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError("Replay maximum gap must be finite and positive")
        selected = slice(start, end + 1)
        if not sequence.valid[selected].all() or not np.isfinite(sequence.poses[selected]).all():
            raise ValueError("Replay requires both tools to be valid and finite throughout the selected interval")
        self.times = sequence.times[selected] - sequence.times[start]
        if np.any(np.diff(self.times) > max_gap_s):
            raise ValueError("Tool interpolation gap exceeds --replay-max-gap; choose a continuous interval")

        if getattr(calibration, "is_metric", False):
            rotation = np.asarray(getattr(calibration, "scene_from_source"), dtype=np.float64)[:3, :3]
        else:
            rotation = np.zeros((3, 3))
            for dest, (sign, src) in enumerate(parse_axis_map(calibration.axis_map)):
                rotation[dest, src] = sign
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-7, rtol=0
        ):
            raise ValueError("Tool replay requires a proper rigid rotation")

        if marker_from_tool_frames is not None and marker_from_colliders is not None:
            raise ValueError("Use either marker_from_tool_frames or marker_from_colliders, not both")
        if marker_from_tool_frames is None:
            marker_from_tool_frames = marker_from_colliders
        if marker_from_tool_frames is None:
            offset = np.asarray(marker_offset if marker_offset is not None else [0, 0, 0], dtype=np.float64)
            if offset.shape == (3,):
                offset = np.broadcast_to(offset, (2, 3)).copy()
            if offset.shape != (2, 3) or not np.isfinite(offset).all():
                raise ValueError("Tool marker offset must contain one or two finite XYZ vectors")
            marker_from_tool_frames = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
            marker_from_tool_frames[:, :3, 3] = offset
        else:
            if marker_offset is not None and np.any(np.asarray(marker_offset, dtype=float) != 0):
                raise ValueError("Use either marker offsets or marker_from_tool_frames, not both")
            marker_from_tool_frames = np.asarray(marker_from_tool_frames, dtype=np.float64)
            if marker_from_tool_frames.shape != (2, 4, 4):
                raise ValueError("marker_from_tool_frames must have shape [2, 4, 4]")
            marker_from_tool_frames = np.stack([
                _rigid_matrix(matrix, f"marker_from_tool_frame[{index}]")
                for index, matrix in enumerate(marker_from_tool_frames)
            ])
        self.marker_from_tool_frames = marker_from_tool_frames
        self.marker_from_colliders = marker_from_tool_frames

        self.poses = sequence.poses[selected, :, :7].copy()
        for i in range(len(self.times)):
            for tool in range(2):
                source_from_marker = np.eye(4, dtype=np.float64)
                source_from_marker[:3, :3] = quaternion_matrix(self.poses[i, tool, 3:7])
                source_from_marker[:3, 3] = self.poses[i, tool, :3]
                source_from_tool_frame = source_from_marker @ marker_from_tool_frames[tool]
                self.poses[i, tool, :3] = apply_calibration(source_from_tool_frame[None, :3, 3], calibration)[0]
                self.poses[i, tool, 3:7] = matrix_quaternion(rotation @ source_from_tool_frame[:3, :3])
        self.start = start
        self.end = end

    def at(self, sim_time_s: float) -> tuple[np.ndarray, np.ndarray]:
        if sim_time_s < -1e-9 or sim_time_s > self.times[-1] + 1e-9:
            raise ValueError("Tool replay cannot extrapolate outside captured timestamps")
        if len(self.times) == 1:
            return self.poses[0].astype(np.float32), np.zeros((2, 6), dtype=np.float32)
        index = min(max(int(np.searchsorted(self.times, sim_time_s, side="right")) - 1, 0), len(self.times) - 2)
        duration = self.times[index + 1] - self.times[index]
        fraction = float(np.clip((sim_time_s - self.times[index]) / duration, 0, 1))
        a, b = self.poses[index], self.poses[index + 1]
        poses = (1 - fraction) * a + fraction * b
        velocities = np.zeros((2, 6), dtype=np.float64)
        velocities[:, :3] = (b[:, :3] - a[:, :3]) / duration
        for tool in range(2):
            poses[tool, 3:7] = slerp(a[tool, 3:7], b[tool, 3:7], fraction)
            delta = matrix_quaternion(quaternion_matrix(b[tool, 3:7]) @ quaternion_matrix(a[tool, 3:7]).T)
            norm = np.linalg.norm(delta[:3])
            if norm > 1e-10:
                velocities[tool, 3:] = delta[:3] / norm * (2 * np.arctan2(norm, delta[3]) / duration)
        return poses.astype(np.float32), velocities.astype(np.float32)


def paired_frame_indices(times: np.ndarray, targets: np.ndarray, tolerance_s: float) -> list[int | None]:
    times, targets = np.asarray(times), np.asarray(targets)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Simulation timestamps must be finite and strictly increasing")
    if not np.isfinite(targets).all() or not np.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("Pairing targets and tolerance must be finite; tolerance cannot be negative")
    result = []
    for target in targets:
        index = int(np.argmin(np.abs(times - target)))
        result.append(index if abs(times[index] - target) <= tolerance_s + 1e-9 else None)
    return result


def depth_comparison(real: np.ndarray, real_valid: np.ndarray, sim: np.ndarray, sim_valid: np.ndarray) -> dict:
    if real.shape != sim.shape or real.shape != real_valid.shape or sim.shape != sim_valid.shape:
        raise ValueError("Depth and validity arrays must share a shape")
    common, union = real_valid & sim_valid, real_valid | sim_valid
    residual = (sim - real)[common]
    n_real, n_sim = int(real_valid.sum()), int(sim_valid.sum())
    result = {
        "real_visible_pixels": n_real, "sim_visible_pixels": n_sim,
        "common_visible_pixels": int(common.sum()), "union_visible_pixels": int(union.sum()),
        "real_only_pixels": int((real_valid & ~sim_valid).sum()),
        "sim_only_pixels": int((sim_valid & ~real_valid).sum()),
        "pixel_iou": float(common.sum() / union.sum()) if union.any() else None,
        "real_coverage": float(common.sum() / n_real) if n_real else None,
        "sim_coverage": float(common.sum() / n_sim) if n_sim else None,
        "pixel_bias_m": float(residual.mean()) if len(residual) else None,
        "pixel_mae_m": float(np.abs(residual).mean()) if len(residual) else None,
        "pixel_p95_m": float(np.quantile(np.abs(residual), .95)) if len(residual) else None,
    }
    return result


def surface_summary(points: np.ndarray, cell_size: float) -> dict:
    if cell_size <= 0:
        raise ValueError("Metric cell size must be positive")
    if not len(points):
        return {key: None for key in ("area_m2", "bbox_area_m2", "centroid_x_m", "centroid_y_m", "width_x_m", "width_y_m", "depth_median_m")}
    cells = np.unique(np.floor(points[:, :2] / cell_size).astype(np.int64), axis=0)
    widths = np.ptp(points[:, :2], axis=0)
    return {
        "area_m2": float(len(cells) * cell_size ** 2), "bbox_area_m2": float(np.prod(widths)),
        "centroid_x_m": float(points[:, 0].mean()), "centroid_y_m": float(points[:, 1].mean()),
        "width_x_m": float(widths[0]), "width_y_m": float(widths[1]), "depth_median_m": float(np.median(points[:, 2])),
    }
