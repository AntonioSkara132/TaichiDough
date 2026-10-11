"""Build simulator tool controls from dom_retrieval trajectories and live policies.

The simulator applies only linear XYZ actions. Recorded or predicted quaternions
therefore remain external to the action adapter, while calibrated scene transforms
are applied before controls reach the solver. Tensor imports stay lazy so the
NumPy-only archive helpers remain usable without PyTorch.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping
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


def gym_actions_from_controls(controls):
    """Convert ``ConditionControls`` velocities to ``DifferentiableMPMEnv`` actions."""
    velocities = _finite("control velocities", controls.velocities)
    if velocities.ndim != 3 or velocities.shape[1:] != (2, 6):
        raise ValueError("Control velocities must have shape [T,2,6]")
    actions = velocities[:, :, :3].reshape(len(velocities), 6).astype(np.float32)
    return actions


def replay_controls_in_gym(env, controls, *, max_steps=None):
    """Replay policy controls through a Gymnasium differentiable MPM environment."""
    actions = gym_actions_from_controls(controls)
    if max_steps is not None:
        actions = actions[:int(max_steps)]
    observations, info = env.reset()
    trajectory = [(observations, 0.0, False, False, info)]
    for action in actions:
        result = env.step(action)
        trajectory.append(result)
        if result[2] or result[3]:
            break
    return trajectory


def sample_point_cloud(points, count: int) -> np.ndarray:
    """Return a deterministic fixed-size XYZ cloud.

    The ordering and repeat behavior match dom_retrieval's ``sample_points``
    helper, while keeping this adapter independent of the training package.
    Coordinates are never centered or scaled here; the model and reward must
    receive the same scene-frame units.
    """
    xyz = np.asarray(points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] < 3 or len(xyz) == 0:
        raise ValueError("Point cloud must be a nonempty [N,>=3] array")
    if count < 1:
        raise ValueError("Point count must be positive")
    xyz = xyz[:, :3]
    if not np.isfinite(xyz).all():
        raise ValueError("Point cloud must contain finite XYZ values")
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    indices = np.floor(np.arange(count, dtype=np.float64) * len(xyz) / count).astype(np.int64)
    return np.ascontiguousarray(xyz[order[indices]], dtype=np.float32)


def symmetric_chamfer_distance(first, second) -> float:
    """Compute the unsquared, symmetric mean nearest-neighbour distance.

    The result is in the coordinate units of the input clouds.  A KD-tree is
    used instead of a dense pairwise matrix so full MPM particle arrays do not
    cause quadratic memory use.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if (first.ndim != 2 or second.ndim != 2 or first.shape[1:] != (3,)
            or second.shape[1:] != (3,) or not len(first) or not len(second)):
        raise ValueError("Point sets must be nonempty [N,3] and [M,3] arrays")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Point sets must contain finite values")
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - scipy is a project dependency
        raise RuntimeError("scipy is required for Chamfer distance") from error
    first_to_second = cKDTree(second).query(first, k=1, workers=1)[0]
    second_to_first = cKDTree(first).query(second, k=1, workers=1)[0]
    return float(np.mean(first_to_second) + np.mean(second_to_first))


def _scene_target_points(points, *, calibration=None, point_filter=None,
                         trim_quantile: float | None = None,
                         apply_calibration: Callable | None = None,
                         filter_scene_points: Callable | None = None) -> np.ndarray:
    """Convert one recorded cloud to finite scene-frame XYZ points."""
    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] < 3 or not len(values):
        raise ValueError("Recorded target cloud must be a nonempty [N,>=3] array")
    values = values[:, :3]
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values):
        raise ValueError("Recorded target cloud has no finite points")

    if trim_quantile is not None:
        quantile = float(trim_quantile)
        if not np.isfinite(quantile) or not 0 <= quantile <= .2:
            raise ValueError("trim_quantile must be finite and between zero and 0.2")
        if quantile > 0 and len(values) > 10:
            low = np.quantile(values, quantile, axis=0)
            high = np.quantile(values, 1. - quantile, axis=0)
            values = values[np.all((values >= low) & (values <= high), axis=1)]

    if apply_calibration is None and calibration is not None:
        from .reference_adapter import get_reference_modules
        apply_calibration = get_reference_modules().topview.apply_calibration
    if apply_calibration is not None:
        values = apply_calibration(values, calibration)

    if point_filter is not None:
        if filter_scene_points is None:
            from .reference_adapter import get_reference_modules
            filter_scene_points = get_reference_modules().dynamics.filter_scene_points
        filtered = filter_scene_points(values, point_filter)
        values = filtered[0] if isinstance(filtered, tuple) else filtered
    values = np.asarray(values, dtype=np.float32)
    values = values[:, :3] if values.ndim == 2 and values.shape[1] >= 3 else values
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("No finite target points remain after scene conversion")
    return np.ascontiguousarray(values)


def final_episode_target_cloud(prepared, *, point_count: int | None = None,
                               point_filter=None, trim_quantile: float | None = None,
                               apply_calibration: Callable | None = None,
                               filter_scene_points: Callable | None = None) -> np.ndarray:
    """Load the last recorded point-cloud frame in the simulator scene frame.

    ``prepared`` is a ``PreparedExperiment``.  The target is intentionally read
    from ``sequence.points[-1]`` rather than from the initial state or a scored
    subset.  Supplying the same scene point filter used during preparation is
    recommended; it keeps the target and initial MPM particles in one metric
    scene frame.
    """
    sequence = getattr(prepared, "sequence", None)
    if sequence is None or not getattr(sequence, "points", None):
        raise ValueError("Prepared experiment has no recorded point-cloud sequence")
    if trim_quantile is None:
        config = getattr(prepared, "config", None)
        observation = getattr(config, "observation", None)
        trim_quantile = getattr(observation, "trim_quantile", None)
    target = _scene_target_points(
        sequence.points[-1], calibration=getattr(prepared, "calibration", None),
        point_filter=point_filter, trim_quantile=trim_quantile,
        apply_calibration=apply_calibration, filter_scene_points=filter_scene_points,
    )
    return sample_point_cloud(target, point_count) if point_count is not None else target


@dataclass
class TerminalChamferReward:
    """Gym reward callback for distance to one immutable terminal target cloud."""

    target_cloud: np.ndarray
    point_count: int | None = None
    cloud_extractor: Callable | None = None
    live_representation: str = "filled_volume"
    target_representation: str = "filled_volume"

    def __post_init__(self):
        if self.live_representation != self.target_representation:
            raise ValueError(
                "Live and target clouds must use the same representation; "
                f"got {self.live_representation!r} and {self.target_representation!r}"
            )
        target = np.asarray(self.target_cloud, dtype=np.float32)
        if target.ndim != 2 or target.shape[1] != 3 or not len(target) or not np.isfinite(target).all():
            raise ValueError("target_cloud must be a nonempty finite [N,3] array")
        self.target_cloud = (sample_point_cloud(target, self.point_count)
                             if self.point_count is not None else np.ascontiguousarray(target))

    def __call__(self, previous, current, action, tool_poses) -> float:
        del previous, action, tool_poses
        particles = getattr(current, "x", current)
        cloud = self.cloud_extractor(particles) if self.cloud_extractor is not None else particles
        if self.point_count is not None:
            cloud = sample_point_cloud(cloud, self.point_count)
        return -symmetric_chamfer_distance(cloud, self.target_cloud)


def attach_terminal_reward(env, target_cloud, *, point_count: int | None = None,
                           cloud_extractor: Callable | None = None,
                           live_representation: str = "filled_volume",
                           target_representation: str = "filled_volume") -> TerminalChamferReward:
    """Install and return a terminal-target reward on a Gym environment."""
    reward = TerminalChamferReward(
        target_cloud,
        point_count=point_count,
        cloud_extractor=cloud_extractor,
        live_representation=live_representation,
        target_representation=target_representation,
    )
    env.reward_fn = reward
    return reward


@dataclass
class _ObservationFrame:
    timestamp: float
    cloud: np.ndarray
    tool_state: np.ndarray


class LiveObservationBuffer:
    """Build dom_retrieval batches from successive Gym observations."""

    def __init__(self, point_count: int, *, history_length: int = 0,
                 history_window_seconds: float = .7,
                 history_max_gap_seconds: float = .2,
                 tool_state_dim: int = 6,
                 cloud_extractor: Callable | None = None,
                 include_tool_positions: bool = False):
        if point_count < 1 or history_length < 0:
            raise ValueError("point_count must be positive and history_length nonnegative")
        if history_length and (history_window_seconds <= 0 or history_max_gap_seconds <= 0):
            raise ValueError("History window and maximum gap must be positive")
        if tool_state_dim not in (6, 14):
            raise ValueError("tool_state_dim must be 6 or 14")
        self.point_count = int(point_count)
        self.history_length = int(history_length)
        self.history_window_seconds = float(history_window_seconds)
        self.history_max_gap_seconds = float(history_max_gap_seconds)
        self.tool_state_dim = int(tool_state_dim)
        self.cloud_extractor = cloud_extractor
        self.include_tool_positions = bool(include_tool_positions)
        self._frames: list[_ObservationFrame] = []

    def reset(self):
        self._frames.clear()

    @property
    def frames(self) -> tuple[_ObservationFrame, ...]:
        return tuple(self._frames)

    @property
    def latest(self) -> _ObservationFrame:
        if not self._frames:
            raise RuntimeError("Observation buffer is empty")
        return self._frames[-1]

    def append(self, particles, tool_poses, timestamp: float):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("Observation timestamp must be finite")
        if self._frames and timestamp <= self._frames[-1].timestamp:
            raise ValueError("Observation timestamps must be strictly increasing")
        raw = np.asarray(particles, dtype=np.float32)
        cloud = self.cloud_extractor(raw) if self.cloud_extractor is not None else raw
        cloud = sample_point_cloud(cloud, self.point_count)
        poses = _finite("tool poses", tool_poses)
        if poses.shape != (2, 7):
            raise ValueError("Tool poses must have shape [2,7]")
        if not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1., atol=1e-5):
            raise ValueError("Tool pose quaternions must be unit length")
        if self.tool_state_dim == 6:
            state = poses[:, :3].reshape(-1)
        else:
            state = poses.reshape(-1)
        self._frames.append(_ObservationFrame(timestamp, cloud, state.astype(np.float32, copy=True)))

    def _history(self) -> list[_ObservationFrame]:
        latest = self.latest
        selected = [latest]
        for frame in reversed(self._frames[:-1]):
            if latest.timestamp - frame.timestamp > self.history_window_seconds:
                break
            if selected[0].timestamp - frame.timestamp > self.history_max_gap_seconds:
                break
            selected.insert(0, frame)
            if len(selected) >= self.history_length:
                break
        return selected

    def as_torch(self, device=None):
        """Return a single-query ObservationBatch accepted by dom_retrieval."""
        import torch
        latest = self.latest
        if not self.history_length:
            result = {"observation": torch.from_numpy(latest.cloud[None]).to(device=device)}
            if self.include_tool_positions:
                result["tool_positions"] = torch.from_numpy(latest.tool_state[None]).to(device=device)
            return result if self.include_tool_positions else result["observation"]

        selected = self._history()
        clouds = np.zeros((self.history_length, self.point_count, 3), dtype=np.float32)
        tools = np.zeros((self.history_length, self.tool_state_dim), dtype=np.float32)
        times = np.zeros(self.history_length, dtype=np.float32)
        mask = np.zeros(self.history_length, dtype=bool)
        for index, frame in enumerate(selected):
            clouds[index] = frame.cloud
            tools[index] = frame.tool_state
            times[index] = np.float32(frame.timestamp - latest.timestamp)
            mask[index] = True
        return {
            "observation": torch.from_numpy(latest.cloud[None]).to(device=device),
            "history_clouds": torch.from_numpy(clouds[None]).to(device=device),
            "history_tool_positions": torch.from_numpy(tools[None]).to(device=device),
            "history_times": torch.from_numpy(times[None]).to(device=device),
            "history_mask": torch.from_numpy(mask[None]).to(device=device),
        }



def _waypoint_positions(waypoints) -> np.ndarray:
    values = _finite("waypoints", waypoints)
    if values.ndim != 3 or values.shape[1] != 2 or values.shape[2] not in (3, 7):
        if values.ndim == 2 and values.shape[1] in (6, 14):
            values = values.reshape(len(values), 2, values.shape[1] // 2)
        else:
            raise ValueError("Waypoints must have shape [T,2,3], [T,2,7], [T,6] or [T,14]")
    return values[..., :3]


def waypoints_to_actions(waypoints, current_tool_positions, action_dt: float, *,
                         steps: int | None = None, duration: float | None = None,
                         max_tool_velocity: float | None = None) -> np.ndarray:
    """Convert absolute XYZ waypoints into Gym's per-step linear velocities."""
    positions = _waypoint_positions(waypoints)
    current = _finite("current tool positions", current_tool_positions)
    if current.shape == (2, 7):
        current = current[:, :3]
    if current.shape != (2, 3):
        raise ValueError("current_tool_positions must have shape [2,3] or [2,7]")
    if not np.isfinite(action_dt) or action_dt <= 0:
        raise ValueError("action_dt must be finite and positive")
    if len(positions) < 2:
        raise ValueError("At least two waypoints are required")
    if steps is None:
        steps = len(positions) - 1
    if steps < 1:
        raise ValueError("steps must be positive")
    if duration is None:
        duration = float(action_dt) * int(steps)
    if not np.isfinite(duration) or duration <= 0 or duration < action_dt * steps - 1e-12:
        raise ValueError("duration must cover all requested action steps")
    source_times = np.linspace(0., float(duration), len(positions))
    query_times = np.arange(1, int(steps) + 1, dtype=np.float64) * float(action_dt)
    if query_times[-1] > duration + 1e-9:
        raise ValueError("duration must cover all requested action steps")
    sampled = np.empty((steps, 2, 3), dtype=np.float64)
    for tool in range(2):
        for axis in range(3):
            sampled[:, tool, axis] = np.interp(query_times, source_times, positions[:, tool, axis])
    previous = np.concatenate((current[None], sampled[:-1]), axis=0)
    velocities = (sampled - previous) / float(action_dt)
    if max_tool_velocity is not None:
        if not np.isfinite(max_tool_velocity) or max_tool_velocity <= 0:
            raise ValueError("max_tool_velocity must be finite and positive")
        velocities = np.clip(velocities, -max_tool_velocity, max_tool_velocity)
    return velocities.reshape(steps, 6).astype(np.float32)


class DomRetrievalGymAdapter:
    """Run a dom_retrieval policy through ``DifferentiableMPMEnv``.

    The adapter replans at every Gym step by default.  Set
    ``replan_each_step=False`` to execute one predicted plan open-loop.
    ``action_hook(action, observation, info)`` is called after policy conversion
    and before ``env.step`` so an RL residual or safety controller can modify the
    action without changing the observation/model code.
    Non-absolute coordinate checkpoints require both observation and prediction
    conversion callbacks; this prevents a local demonstration frame from being
    silently sent to the calibrated simulator.
    """

    def __init__(self, env, model, *, config: Mapping | None = None,
                 point_count: int | None = None, history_length: int | None = None,
                 history_window_seconds: float | None = None,
                 history_max_gap_seconds: float | None = None,
                 tool_state_dim: int | None = None, device: str = "cpu",
                 trajectory_mean=None, trajectory_std=None,
                 observation_transform: Callable | None = None,
                 prediction_to_scene: Callable | None = None,
                 cloud_extractor: Callable | None = None,
                 action_hook: Callable | None = None,
                 replan_each_step: bool = False, plan_duration_s: float | None = None):
        self.env, self.model, self.config = env, model, dict(config or {})
        data = self.config.get("data", {})
        history = self.config.get("history", {})
        self.point_count = int(point_count if point_count is not None else data.get("points", 0))
        if self.point_count < 1:
            raise ValueError("point_count is required or config.data.points must be positive")
        history_enabled = bool(history.get("enabled", False))
        default_length = int(data.get("history_length", 0)) if history_enabled else 0
        self.history_length = int(default_length if history_length is None else history_length)
        if history_enabled and self.history_length < 1:
            raise ValueError("History-enabled policies require a positive history_length")
        if not history_enabled and self.history_length:
            raise ValueError("history_length is nonzero but config history is disabled")
        self.history_window_seconds = float(
            data.get("history_window_seconds", .7) if history_window_seconds is None else history_window_seconds)
        self.history_max_gap_seconds = float(
            data.get("history_max_gap_seconds", .2) if history_max_gap_seconds is None else history_max_gap_seconds)
        if tool_state_dim is None:
            tool_state_dim = 14 if bool(data.get("include_tool_quaternions", False)) else 6
        self.tool_state_dim = int(tool_state_dim)
        static_tools = bool(self.config.get("proprioception", {}).get("enabled", False))
        if self.history_length and static_tools:
            raise ValueError("Temporal history and static proprioception cannot both be enabled")
        self.device = device
        self.observation_transform = observation_transform
        self.prediction_to_scene = prediction_to_scene
        self.action_hook = action_hook
        self.replan_each_step = bool(replan_each_step)
        self.action_dt = float(env.action_substeps * env.config.dt)
        if plan_duration_s is not None and (
                not np.isfinite(plan_duration_s) or plan_duration_s < self.action_dt):
            raise ValueError("plan_duration_s must cover at least one Gym action")
        self.plan_duration_s = plan_duration_s
        self._step_count = 0
        self._pending_actions: list[np.ndarray] = []
        self._last_observation = None
        self._last_info = None
        self._output_mean = None if trajectory_mean is None else np.asarray(trajectory_mean, dtype=np.float32)
        self._output_std = None if trajectory_std is None else np.asarray(trajectory_std, dtype=np.float32)
        if (self._output_mean is None) != (self._output_std is None):
            raise ValueError("trajectory_mean and trajectory_std must be supplied together")
        if self._output_mean is not None and (
                self._output_mean.ndim != 1 or self._output_std.shape != self._output_mean.shape
                or not np.isfinite(self._output_mean).all() or not np.isfinite(self._output_std).all()
                or np.any(self._output_std <= 0)):
            raise ValueError("Trajectory normalization statistics must be finite 1D arrays with positive std")
        self.buffer = LiveObservationBuffer(
            self.point_count, history_length=self.history_length,
            history_window_seconds=self.history_window_seconds,
            history_max_gap_seconds=self.history_max_gap_seconds,
            tool_state_dim=self.tool_state_dim, cloud_extractor=cloud_extractor,
            include_tool_positions=static_tools,
        )
        mode = self.config.get("coordinates", {}).get("mode", "absolute")
        if mode != "absolute" and (observation_transform is None or prediction_to_scene is None):
            raise ValueError("Non-absolute checkpoints require observation_transform and prediction_to_scene")

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._step_count = 0
        self._pending_actions.clear()
        self.buffer.reset()
        self.buffer.append(observation["particles"], observation["tool_poses"], info.get("sim_time", 0.0))
        self._last_observation, self._last_info = observation, info
        return observation, info

    def _model_input(self):
        value = self.buffer.as_torch(device=self.device)
        return self.observation_transform(value) if self.observation_transform is not None else value

    def _prediction(self):
        import torch
        model_input = self._model_input()
        with torch.no_grad():
            output = self.model(model_input)
        prediction = output.prediction if hasattr(output, "prediction") else output
        if not isinstance(prediction, torch.Tensor):
            raise ValueError("Policy output must be a tensor or expose a tensor prediction")
        if prediction.ndim != 3 or prediction.shape[0] != 1:
            raise ValueError("Policy prediction must have shape [1,T,D]")
        prediction = prediction[0].detach().cpu().numpy()
        if self._output_mean is not None:
            if prediction.shape[-1] != self._output_mean.size or self._output_std.shape != self._output_mean.shape:
                raise ValueError("Trajectory normalization dimensions do not match policy output")
            prediction = prediction * self._output_std + self._output_mean
        if self.prediction_to_scene is not None:
            prediction = self.prediction_to_scene(prediction, model_input, output)
        return _waypoint_positions(prediction)

    def plan_actions(self, *, steps: int | None = None) -> np.ndarray:
        if not self.buffer.frames:
            raise RuntimeError("Call reset before planning")
        remaining = self.env.episode_steps - self._step_count
        if steps is None:
            steps = remaining
        steps = min(int(steps), int(remaining))
        if steps < 1:
            raise RuntimeError("The Gym episode has ended")
        duration = self.plan_duration_s
        if duration is None:
            duration = self.action_dt * steps
        elif duration < self.action_dt * steps - 1e-12:
            steps = max(1, int(np.floor(duration / self.action_dt + 1e-10)))
        return waypoints_to_actions(
            self._prediction(), self.buffer.latest.tool_state.reshape(2, -1), self.action_dt,
            steps=steps, duration=duration, max_tool_velocity=self.env.max_tool_velocity,
        )

    def action(self) -> np.ndarray:
        if self.replan_each_step or not self._pending_actions:
            self._pending_actions = [row for row in self.plan_actions()]
        if not self._pending_actions:
            raise RuntimeError("No policy action available")
        action = np.asarray(self._pending_actions.pop(0), dtype=np.float32)
        return action if self.action_hook is None else np.asarray(
            self.action_hook(action, self._last_observation, self._last_info), dtype=np.float32)

    def step(self, action=None):
        if action is None:
            action = self.action()
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        self.buffer.append(observation["particles"], observation["tool_poses"], info.get("sim_time", self._step_count * self.action_dt))
        self._last_observation, self._last_info = observation, info
        return observation, reward, terminated, truncated, info

    def rollout(self, *, seed=None, options=None, max_steps: int | None = None):
        result = [self.reset(seed=seed, options=options)]
        limit = self.env.episode_steps if max_steps is None else min(int(max_steps), self.env.episode_steps)
        for _ in range(limit):
            transition = self.step()
            result.append(transition)
            if transition[2] or transition[3]:
                break
        return result
