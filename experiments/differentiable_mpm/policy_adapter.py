"""Build simulator tool controls from an exported dom_retrieval trajectory.

The policy predicts XYZ only. Recorded quaternions and the calibrated marker-to-tool
transforms remain external inputs. This module is NumPy-only so policy export and
simulator execution can be tested separately.
"""
from __future__ import annotations

from dataclasses import dataclass
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
