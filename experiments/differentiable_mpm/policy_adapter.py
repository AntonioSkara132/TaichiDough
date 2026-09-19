"""Build simulator tool controls from an exported dom_retrieval trajectory.

The policy predicts XYZ only. Recorded quaternions and the calibrated marker-to-tool
transforms remain external inputs. This module is NumPy-only so policy export and
simulator execution can be tested separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .state import ToolControl
from .reference.deformpath_dynamics import matrix_quaternion, quaternion_matrix, slerp

CONDITIONS = (
    "recorded_full_pose",
    "recorded_xyz_fixed_orientation",
    "predicted_xyz_fixed_orientation",
    "hold_position",
    "predicted_xyz_recorded_orientation",
)


def _finite(name, values, shape=None):
    values = np.asarray(values, dtype=np.float64)
    if shape is not None and values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite values")
    return values


def _unit_quaternions(values):
    values = _finite("quaternions", values)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Quaternions must be nonzero")
    return values / norms


def _interpolate_positions(times, values, query):
    times = _finite("sample times", times)
    values = _finite("positions", values)
    query = _finite("query times", query)
    if times.ndim != 1 or values.ndim != 3 or values.shape[:2] != (len(times), 2):
        raise ValueError("Position samples must be [N,2,3] with one time per sample")
    if values.shape[2] != 3 or len(times) < 1 or np.any(np.diff(times) <= 0):
        raise ValueError("Position sample times must be strictly increasing")
    if np.any(query < times[0] - 1e-9) or np.any(query > times[-1] + 1e-9):
        raise ValueError("Query time lies outside recorded trajectory")
    result = np.empty((len(query), 2, 3), dtype=np.float64)
    for tool in range(2):
        for axis in range(3):
            result[:, tool, axis] = np.interp(query, times, values[:, tool, axis])
    return result


def _interpolate_quaternions(times, values, query):
    values = _unit_quaternions(values)
    query = _finite("query times", query)
    if len(times) == 1:
        return np.broadcast_to(values[0], (len(query), 2, 4)).copy()
    result = np.empty((len(query), 2, 4), dtype=np.float64)
    for row, current in enumerate(query):
        right = min(max(int(np.searchsorted(times, current, side="right")), 1), len(times) - 1)
        left = right - 1
        fraction = np.clip((current - times[left]) / (times[right] - times[left]), 0., 1.)
        for tool in range(2):
            result[row, tool] = slerp(values[left, tool], values[right, tool], float(fraction))
    return result


def _tool_pose_transform(raw_poses, scene_from_source, marker_from_tool_frames):
    raw_poses = _finite("raw poses", raw_poses)
    if raw_poses.ndim != 3 or raw_poses.shape[1:] != (2, 7):
        raise ValueError("Raw poses must have shape [N,2,7]")
    scene_from_source = _finite("scene_from_source", scene_from_source, (4, 4))
    marker_from_tool_frames = _finite("marker_from_tool_frames", marker_from_tool_frames, (2, 4, 4))
    result = np.empty_like(raw_poses)
    for frame in range(len(raw_poses)):
        for tool in range(2):
            source_from_marker = np.eye(4)
            source_from_marker[:3, :3] = quaternion_matrix(raw_poses[frame, tool, 3:])
            source_from_marker[:3, 3] = raw_poses[frame, tool, :3]
            scene_from_tool = scene_from_source @ source_from_marker @ marker_from_tool_frames[tool]
            result[frame, tool, :3] = scene_from_tool[:3, 3]
            result[frame, tool, 3:] = matrix_quaternion(scene_from_tool[:3, :3])
    return result


def _velocities(poses, times):
    poses = _finite("simulator poses", poses)
    times = _finite("control times", times)
    if poses.ndim != 3 or poses.shape[1:] != (2, 7) or len(times) != len(poses):
        raise ValueError("Poses must be [N,2,7] and match control times")
    velocities = np.zeros((len(poses), 2, 6), dtype=np.float64)
    if len(poses) < 2:
        return velocities
    for row in range(1, len(poses)):
        dt = float(times[row] - times[row - 1])
        if dt <= 0:
            raise ValueError("Control times must be strictly increasing")
        velocities[row, :, :3] = (poses[row, :, :3] - poses[row - 1, :, :3]) / dt
        for tool in range(2):
            delta = matrix_quaternion(quaternion_matrix(poses[row, tool, 3:]) @ quaternion_matrix(poses[row - 1, tool, 3:]).T)
            norm = np.linalg.norm(delta[:3])
            if norm > 1e-10:
                velocities[row, tool, 3:] = delta[:3] / norm * (2 * np.arctan2(norm, delta[3]) / dt)
    return velocities


def build_condition_poses(condition, control_times, recorded_times, recorded_raw_poses,
                          predicted_times=None, predicted_positions=None,
                          scene_from_source=None, marker_from_tool_frames=None):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition}; choose from {CONDITIONS}")
    control_times = _finite("control times", control_times)
    recorded_times = _finite("recorded times", recorded_times)
    recorded_raw_poses = _finite("recorded raw poses", recorded_raw_poses)
    if recorded_raw_poses.shape != (len(recorded_times), 2, 7) or len(recorded_times) < 1:
        raise ValueError("Recorded raw poses must have shape [N,2,7]")
    if np.any(np.diff(recorded_times) <= 0) and len(recorded_times) > 1:
        raise ValueError("Recorded times must be strictly increasing")
    if scene_from_source is None:
        scene_from_source = np.eye(4)
    if marker_from_tool_frames is None:
        marker_from_tool_frames = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    recorded_positions = recorded_raw_poses[:, :, :3]
    recorded_quaternions = recorded_raw_poses[:, :, 3:]
    query = np.clip(control_times, recorded_times[0], recorded_times[-1])
    positions = _interpolate_positions(recorded_times, recorded_positions, query)
    orientations = _interpolate_quaternions(recorded_times, recorded_quaternions, query)
    if condition == "recorded_full_pose":
        raw_positions, raw_orientations = positions, orientations
    elif condition == "recorded_xyz_fixed_orientation":
        raw_positions = positions
        raw_orientations = np.broadcast_to(_unit_quaternions(recorded_quaternions[0]), orientations.shape).copy()
    elif condition == "hold_position":
        raw_positions = np.broadcast_to(recorded_positions[0], positions.shape).copy()
        raw_orientations = np.broadcast_to(_unit_quaternions(recorded_quaternions[0]), orientations.shape).copy()
    else:
        if predicted_times is None or predicted_positions is None:
            raise ValueError("Predicted conditions require exported prediction times and positions")
        predicted_times = _finite("predicted times", predicted_times)
        predicted_positions = _finite("predicted positions", predicted_positions)
        if predicted_positions.shape != (len(predicted_times), 2, 3) or len(predicted_times) < 2:
            raise ValueError("Predicted positions must have shape [T,2,3] with T>=2")
        raw_positions = _interpolate_positions(predicted_times, predicted_positions, query)
        if condition == "predicted_xyz_fixed_orientation":
            raw_orientations = np.broadcast_to(_unit_quaternions(recorded_quaternions[0]), orientations.shape).copy()
        else:
            raw_orientations = orientations
    raw = np.concatenate((raw_positions, raw_orientations), axis=-1)
    transformed = _tool_pose_transform(raw, scene_from_source, marker_from_tool_frames)
    return transformed, _velocities(transformed, control_times)


@dataclass
class ConditionControls:
    poses: np.ndarray
    velocities: np.ndarray
    dt: float

    def __post_init__(self):
        self.poses = _finite("condition poses", self.poses)
        self.velocities = _finite("condition velocities", self.velocities)
        if self.poses.shape != (len(self.velocities), 2, 7) or self.velocities.shape[1:] != (2, 6):
            raise ValueError("Condition arrays must have shapes [N,2,7] and [N,2,6]")
        if self.dt <= 0 or not np.isfinite(self.dt):
            raise ValueError("Control dt must be finite and positive")
        for pose in self.poses:
            ToolControl(pose, np.zeros((2, 6)), 0.).validate()

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return ToolControl(self.poses[index].astype(np.float32), self.velocities[index].astype(np.float32), index * self.dt)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_condition_archive(archive, condition, *, expected_dt=None, expected_steps=None,
                           expected_duration=None):
    """Load one validated condition archive for simulator execution."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition}; choose from {CONDITIONS}")
    archive = Path(archive).expanduser().resolve()
    root = archive if archive.is_dir() else archive.parent
    manifest_path = archive / "manifest.json" if archive.is_dir() else archive
    if not manifest_path.is_file():
        raise ValueError(f"Condition archive manifest was not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read condition archive manifest: {manifest_path}") from error
    if manifest.get("schema") != "taichidough/mpm-policy-controls/v1":
        raise ValueError("Unsupported policy-control archive schema")
    rows = manifest.get("conditions")
    if not isinstance(rows, dict) or condition not in rows or not isinstance(rows[condition], dict):
        raise ValueError(f"Condition archive has no metadata for {condition!r}")
    row = rows[condition]
    filename = row.get("file")
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"Condition archive has no file for {condition!r}")
    data_path = (root / filename).resolve()
    if not data_path.is_relative_to(root) or not data_path.is_file():
        raise ValueError("Condition archive file must exist inside the archive directory")
    dt = manifest.get("control_dt_s")
    duration = manifest.get("duration_s")
    times = np.asarray(manifest.get("control_times_s"), dtype=np.float64)
    if not isinstance(dt, (int, float, np.integer, np.floating)) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Condition archive control_dt_s must be finite and positive")
    if not isinstance(duration, (int, float, np.integer, np.floating)) or not np.isfinite(duration) or duration < 0:
        raise ValueError("Condition archive duration_s must be finite and nonnegative")
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or abs(float(times[0])) > 1e-8:
        raise ValueError("Condition archive control_times_s must be finite and start at zero")
    if len(times) > 1:
        deltas = np.diff(times)
        if np.any(deltas <= 0) or not np.allclose(deltas, float(dt), rtol=1e-5, atol=1e-8):
            raise ValueError("Condition archive control_times_s must use the declared regular timestep")
    if float(times[-1]) > float(duration) + 1e-8:
        raise ValueError("Condition archive duration_s is shorter than its control times")
    if expected_dt is not None and not np.isclose(float(dt), float(expected_dt), rtol=1e-5, atol=1e-8):
        raise ValueError(f"Condition archive dt {float(dt):g} does not match simulator dt {float(expected_dt):g}")
    with np.load(data_path, allow_pickle=False) as loaded:
        if set(loaded.files) != {"poses", "velocities"}:
            raise ValueError("Condition archive must contain exactly poses and velocities")
        poses = np.asarray(loaded["poses"], dtype=np.float64)
        velocities = np.asarray(loaded["velocities"], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (2, 7) or len(poses) != len(times) or len(poses) == 0:
        raise ValueError("Condition archive poses must have shape [N,2,7] matching control times")
    if velocities.shape != (len(poses), 2, 6):
        raise ValueError("Condition archive velocities must have shape [N,2,6]")
    if row.get("pose_shape") != list(poses.shape) or row.get("velocity_shape") != list(velocities.shape):
        raise ValueError("Condition archive manifest dimensions do not match the selected arrays")
    controls = ConditionControls(poses, velocities, float(dt))
    if expected_steps is not None and len(controls) < int(expected_steps):
        raise ValueError("Condition archive does not contain enough controls for the simulation")
    horizon = (len(controls) - 1) * float(dt)
    if expected_duration is not None and float(duration) + 1e-8 < float(expected_duration):
        raise ValueError("Condition archive duration does not cover the simulation horizon")
    if expected_duration is not None and len(controls) * float(dt) + 1e-8 < float(expected_duration):
        raise ValueError("Condition archive controls do not cover the simulation horizon")
    return controls, {
        "kind": "policy_archive", "archive_dir": str(root),
        "manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
        "condition": condition, "file": str(data_path), "file_sha256": _sha256(data_path),
        "pose_shape": list(poses.shape), "velocity_shape": list(velocities.shape),
        "control_dt_s": float(dt), "duration_s": float(duration),
        "control_count": len(controls), "control_horizon_s": horizon,
        "control_times_s": times.tolist(),
    }


def save_conditions(output, conditions):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "taichidough/mpm-policy-controls/v1", "conditions": {}}
    for name, (poses, velocities) in conditions.items():
        if name not in CONDITIONS:
            raise ValueError(f"Unknown condition {name}")
        np.savez_compressed(output / f"{name}.npz", poses=poses, velocities=velocities)
        manifest["conditions"][name] = {"file": f"{name}.npz", "pose_shape": list(poses.shape),
                                         "velocity_shape": list(velocities.shape)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
