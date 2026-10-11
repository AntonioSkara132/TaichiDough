"""Run a pretrained dom_retrieval policy in the differentiable MPM gym.

The adapter keeps the simulator interface unchanged: the simulator still receives
one flattened pair of XYZ velocities at every step. A policy prediction is a
trajectory of tool positions, which is converted to velocities and replayed
open-loop before the next policy query.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from dom_retrieval.data.preprocessing import sample_points
from dom_retrieval.dynamics.evaluation import chamfer_distance
from dom_retrieval.training.trainer import load_experiment, physical_output

from .state import InvalidStateError


class CheckpointCompatibilityError(ValueError):
    """The checkpoint cannot consume the live simulator observations."""


@dataclass(frozen=True)
class LivePolicySpec:
    point_count: int
    horizon: int
    output_dim: int
    history_length: int = 0
    history_window_seconds: float = 0.0
    history_tool_dim: int = 14
    proprioception_tool_dim: int = 0
    coordinate_mode: str = "absolute"

    @property
    def uses_history(self) -> bool:
        return self.history_length > 0


def _finite_cloud(points: Any, name: str) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 3 or not len(values):
        raise ValueError(f"{name} must be a nonempty [N,3] point cloud")
    values = np.ascontiguousarray(values[:, :3])
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite XYZ values")
    return values


def load_final_goal_cloud(path: str | Path) -> np.ndarray:
    """Load one goal cloud, taking the final frame from a ``[T,P,3]`` array."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Goal point cloud was not found: {path}")
    if path.suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "points" in archive:
                values = archive["points"]
            elif len(archive.files) == 1:
                values = archive[archive.files[0]]
            else:
                raise ValueError("Goal NPZ must contain a points array")
    else:
        raise ValueError("Goal cloud must be stored as .npy or .npz")
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[-1]
    return _finite_cloud(values, "goal cloud")


def _sample_cloud(points: Any, count: int, name: str) -> torch.Tensor:
    values = _finite_cloud(points, name)
    return sample_points(torch.from_numpy(values), int(count))


def negative_chamfer(
    current: Any,
    target: Any,
    *,
    sample_count: int = 1024,
) -> float:
    """Return negative symmetric mean nearest-neighbour distance in metres."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    current_values = _sample_cloud(current, sample_count, "current cloud")
    target_values = _sample_cloud(target, sample_count, "target cloud")
    return -float(chamfer_distance(current_values, target_values))


class NegativeChamferReward:
    """Gym reward callback with a goal fixed at the beginning of an episode."""

    def __init__(
        self,
        *,
        sample_count: int = 1024,
        live_representation: str = "filled_volume",
        goal_representation: str = "filled_volume",
        cloud_extractor: Callable[[Any], Any] | None = None,
    ) -> None:
        if live_representation != goal_representation:
            raise ValueError(
                "Live and goal point clouds must use the same representation; "
                f"got {live_representation!r} and {goal_representation!r}"
            )
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        self.sample_count = int(sample_count)
        self.live_representation = live_representation
        self.goal_representation = goal_representation
        self.goal_cloud: np.ndarray | None = None
        self.cloud_extractor = cloud_extractor
        self.last_components: dict[str, float] = {}

    def set_goal(self, goal_cloud: Any) -> None:
        self.goal_cloud = _finite_cloud(goal_cloud, "goal cloud")
        self.last_components = {}

    def __call__(self, previous_state, current_state, action, tool_poses) -> float:
        del previous_state, action, tool_poses
        if self.goal_cloud is None:
            raise RuntimeError("Set the recorded goal cloud before stepping the environment")
        particles = current_state.x
        if self.cloud_extractor is not None:
            particles = self.cloud_extractor(particles)
        current = _sample_cloud(particles, self.sample_count, "current cloud")
        target = _sample_cloud(self.goal_cloud, self.sample_count, "target cloud")
        distances = torch.cdist(current.float(), target.float())
        current_to_target = float(distances.min(dim=1).values.mean())
        target_to_current = float(distances.min(dim=0).values.mean())
        self.last_components = {
            "chamfer_current_to_goal_m": current_to_target,
            "chamfer_goal_to_current_m": target_to_current,
            "chamfer_m": current_to_target + target_to_current,
        }
        return -self.last_components["chamfer_m"]


class _HistoryBuffer:
    def __init__(self, spec: LivePolicySpec) -> None:
        self.spec = spec
        self.entries: deque[tuple[float, np.ndarray, np.ndarray]] = deque(
            maxlen=spec.history_length
        )

    def reset(self) -> None:
        self.entries.clear()

    def append(self, time_s: float, cloud: Any, tool_poses: Any) -> None:
        cloud = _finite_cloud(cloud, "observation cloud")
        poses = np.asarray(tool_poses, dtype=np.float32)
        if poses.shape != (2, 7) or not np.isfinite(poses).all():
            raise ValueError("tool_poses must be finite with shape [2,7]")
        if self.spec.history_tool_dim == 14:
            tools = poses.reshape(-1)
        elif self.spec.history_tool_dim == 6:
            tools = poses[:, :3].reshape(-1)
        else:
            raise ValueError("history_tool_dim must be 6 or 14")
        current_time = float(time_s)
        self.entries.append((current_time, cloud, tools.copy()))
        while (
            len(self.entries) > 1
            and current_time - self.entries[0][0] > self.spec.history_window_seconds
        ):
            self.entries.popleft()

    def observation(self) -> dict[str, torch.Tensor]:
        if not self.entries:
            raise RuntimeError("History is empty")
        if len(self.entries) > self.spec.history_length:
            raise RuntimeError("History exceeds the configured length")
        point_count = len(self.entries[-1][1])
        length = self.spec.history_length
        clouds = np.zeros((length, point_count, 3), dtype=np.float32)
        tools = np.zeros((length, self.spec.history_tool_dim), dtype=np.float32)
        times = np.zeros(length, dtype=np.float32)
        mask = np.zeros(length, dtype=bool)
        now = self.entries[-1][0]
        for row, (time_s, cloud, tool) in enumerate(self.entries):
            if len(cloud) != point_count:
                raise ValueError("All history clouds must have the same point count")
            clouds[row] = cloud
            tools[row] = tool
            times[row] = time_s - now
            mask[row] = True
        current = clouds[len(self.entries) - 1]
        return {
            "observation": torch.from_numpy(current.copy()),
            "history_clouds": torch.from_numpy(clouds),
            "history_tool_positions": torch.from_numpy(tools),
            "history_times": torch.from_numpy(times),
            "history_mask": torch.from_numpy(mask),
        }


@dataclass
class RolloutResult:
    actions: np.ndarray
    predictions: np.ndarray
    rewards: np.ndarray
    infos: list[dict[str, Any]]
    terminated: bool
    truncated: bool


class DomRetrievalPolicyRunner:
    """Load and query a checkpoint against live simulator observations."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        cloud_extractor: Callable[[Any], Any] | None = None,
    ) -> None:
        try:
            model, split, mean, std, config = load_experiment(checkpoint, device)
        except (RuntimeError, ValueError, KeyError, OSError) as error:
            raise CheckpointCompatibilityError(
                "Could not load the dom_retrieval checkpoint. Verify that its "
                "dataset snapshot, history configuration, and memory metadata "
                "match the saved state."
            ) from error
        if not split.train:
            raise CheckpointCompatibilityError("Checkpoint has no training demonstrations")
        self.model = model.eval()
        self.device = torch.device(device)
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)
        self.config = config
        self.cloud_extractor = cloud_extractor
        self.spec = self._build_spec(split)
        if self.spec.coordinate_mode != "absolute":
            raise CheckpointCompatibilityError(
                "Live TaichiDough inference currently supports only coordinates.mode=absolute"
            )
        self.history = _HistoryBuffer(self.spec) if self.spec.uses_history else None
        self.current_observation: dict[str, Any] | None = None

    def _build_spec(self, split) -> LivePolicySpec:
        sample = split.train[0]
        point_count = int(sample.observation.shape[-2])
        horizon = int(getattr(self.model, "horizon", sample.trajectory.shape[0]))
        output_dim = int(getattr(self.model, "action_dim", sample.trajectory.shape[-1]))
        if output_dim not in (6, 14):
            raise CheckpointCompatibilityError(
                f"Gym bridge requires a 6D or 14D trajectory, got {output_dim}"
            )
        if horizon < 2:
            raise CheckpointCompatibilityError("Gym bridge requires at least two waypoints")
        history_module = next(
            (module for module in self.model.modules()
             if module.__class__.__name__ == "HistoryEncoder"),
            None,
        )
        proprioception_module = next(
            (module for module in self.model.modules()
             if module.__class__.__name__ == "ProprioceptiveEncoder"),
            None,
        )
        data_config = self.config.get("data", {})
        history_config = self.config.get("history", {})
        if history_module is None:
            history_length = 0
            history_window = 0.0
            history_tool_dim = 14
        else:
            history_length = int(data_config.get(
                "history_length", history_config.get("length", 8)
            ))
            history_window = float(data_config.get(
                "history_window_seconds", history_config.get("window_seconds", 0.7)
            ))
            history_tool_dim = int(getattr(history_module, "pose_dim", 14))
            if history_length < 1 or history_window < 0:
                raise CheckpointCompatibilityError("Checkpoint history settings are invalid")
        proprioception_tool_dim = (
            int(getattr(proprioception_module, "pose_dim", 0))
            if proprioception_module is not None else 0
        )
        if proprioception_tool_dim not in (0, 6, 14):
            raise CheckpointCompatibilityError("Checkpoint proprioception pose dimension is invalid")
        coordinate_mode = str(self.config.get("coordinates", {}).get("mode", "absolute"))
        return LivePolicySpec(
            point_count=point_count,
            horizon=horizon,
            output_dim=output_dim,
            history_length=history_length,
            history_window_seconds=history_window,
            history_tool_dim=history_tool_dim,
            proprioception_tool_dim=proprioception_tool_dim,
            coordinate_mode=coordinate_mode,
        )

    def reset(self, observation: Mapping[str, Any], time_s: float = 0.0) -> None:
        self.current_observation = dict(observation)
        if self.history is not None:
            self.history.reset()
            self._append_observation(observation, time_s)

    def _live_cloud(self, particles: Any) -> Any:
        return self.cloud_extractor(particles) if self.cloud_extractor is not None else particles

    def _append_observation(self, observation: Mapping[str, Any], time_s: float) -> None:
        cloud = _sample_cloud(
            self._live_cloud(observation["particles"]),
            self.spec.point_count,
            "observation cloud",
        ).numpy()
        poses = np.asarray(observation["tool_poses"], dtype=np.float32)
        if self.history is not None:
            self.history.append(time_s, cloud, poses)
        self.current_observation = dict(observation)

    def observe(self, observation: Mapping[str, Any], time_s: float) -> None:
        if self.history is None:
            self.current_observation = dict(observation)
        else:
            self._append_observation(observation, time_s)

    def _pose_vector(self, dimension: int) -> torch.Tensor:
        if self.current_observation is None:
            raise RuntimeError("Call reset before predict")
        poses = np.asarray(self.current_observation["tool_poses"], dtype=np.float32)
        if poses.shape != (2, 7) or not np.isfinite(poses).all():
            raise ValueError("tool_poses must be finite with shape [2,7]")
        if dimension == 14:
            values = poses.reshape(-1)
        elif dimension == 6:
            values = poses[:, :3].reshape(-1)
        else:
            raise ValueError("Tool pose dimension must be 6 or 14")
        return torch.from_numpy(values).unsqueeze(0).to(self.device)

    def _batch(self) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.current_observation is None:
            raise RuntimeError("Call reset before predict")
        cloud = _sample_cloud(
            self._live_cloud(self.current_observation["particles"]),
            self.spec.point_count,
            "observation cloud",
        )
        needs_current_pose = (
            self.spec.proprioception_tool_dim > 0
            or getattr(self.model, "pose_parameterization", "absolute") == "orientation_delta"
            or self.config.get("trajectory_representation", "absolute") == "relative_to_start"
        )
        if self.history is not None:
            batch = self.history.observation()
            batch = {name: value.unsqueeze(0).to(self.device) for name, value in batch.items()}
            if self.spec.proprioception_tool_dim > 0:
                batch["tool_positions"] = self._pose_vector(self.spec.proprioception_tool_dim)
            elif needs_current_pose and self.spec.history_tool_dim not in (6, 14):
                raise CheckpointCompatibilityError("History pose dimension is invalid")
            return batch
        batch_cloud = cloud.unsqueeze(0).to(self.device)
        if needs_current_pose:
            dimension = self.spec.proprioception_tool_dim or 14
            return {
                "observation": batch_cloud,
                "tool_positions": self._pose_vector(dimension),
            }
        return batch_cloud

    def predict_positions(self) -> np.ndarray:
        batch = self._batch()
        with torch.inference_mode():
            try:
                output = self.model(batch, query_groups=["__live__"])
            except TypeError:
                output = self.model(batch)
            output = physical_output(output, self.mean, self.std)
        prediction = output.prediction[0].detach().cpu()
        if prediction.ndim != 2 or prediction.shape[0] != self.spec.horizon:
            raise CheckpointCompatibilityError("Policy returned an unexpected trajectory horizon")
        if self.spec.output_dim == 6:
            positions = prediction.reshape(self.spec.horizon, 2, 3)
        else:
            positions = prediction[:, (0, 1, 2, 7, 8, 9)].reshape(self.spec.horizon, 2, 3)
        values = positions.numpy().astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Policy returned nonfinite tool positions")
        return values


def trajectory_to_actions(
    positions: Any,
    current_tool_poses: Any,
    control_dt: float,
) -> np.ndarray:
    """Convert [T,2,3] positions to [T,6] two-tool XYZ velocities."""
    values = np.asarray(positions, dtype=np.float64)
    poses = np.asarray(current_tool_poses, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (2, 3) or len(values) < 1:
        raise ValueError("Predicted positions must have shape [T,2,3]")
    if poses.shape != (2, 7):
        raise ValueError("Current tool poses must have shape [2,7]")
    if not np.isfinite(values).all() or not np.isfinite(poses).all():
        raise ValueError("Positions and tool poses must be finite")
    if not np.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be finite and positive")
    previous = np.concatenate((poses[None, :, :3], values[:-1]), axis=0)
    actions = ((values - previous) / float(control_dt)).reshape(len(values), 6)
    if not np.isfinite(actions).all():
        raise ValueError("Converted actions are nonfinite")
    return actions.astype(np.float32)


class DomRetrievalGymController:
    """Execute full predicted trajectories and then replan."""

    def __init__(
        self,
        env,
        runner: DomRetrievalPolicyRunner,
        reward: NegativeChamferReward,
        goal_cloud: Any,
    ) -> None:
        self.env = env
        self.runner = runner
        self.reward = reward
        self.goal_cloud = _finite_cloud(goal_cloud, "goal cloud")
        self.control_dt = float(env.config.dt) * int(env.action_substeps)

    def run(self, *, max_steps: int | None = None) -> RolloutResult:
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be positive")
        observation, info = self.env.reset()
        self.reward.set_goal(self.goal_cloud)
        self.runner.reset(observation, 0.0)
        actions: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        rewards: list[float] = []
        infos: list[dict[str, Any]] = [dict(info)]
        terminated = truncated = False
        elapsed_steps = 0
        while not (terminated or truncated):
            predicted = self.runner.predict_positions()
            predictions.append(predicted.copy())
            open_loop = trajectory_to_actions(
                predicted, observation["tool_poses"], self.control_dt
            )
            for action in open_loop:
                if max_steps is not None and elapsed_steps >= max_steps:
                    truncated = True
                    break
                try:
                    observation, reward, terminated, truncated, info = self.env.step(action)
                except InvalidStateError as error:
                    terminated = True
                    truncated = False
                    info = {"failure": "invalid_mpm_state", "error": str(error)}
                    actions.append(action.copy())
                    rewards.append(float("-inf"))
                    infos.append(info)
                    break
                actions.append(action.copy())
                rewards.append(float(reward))
                step_info = dict(info)
                if self.reward.last_components:
                    step_info["chamfer"] = dict(self.reward.last_components)
                infos.append(step_info)
                elapsed_steps += 1
                self.runner.observe(observation, elapsed_steps * self.control_dt)
                if max_steps is not None and elapsed_steps >= max_steps and not terminated and not truncated:
                    truncated = True
                if terminated or truncated:
                    break
        return RolloutResult(
            actions=np.asarray(actions, dtype=np.float32).reshape(-1, 6),
            predictions=np.asarray(predictions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            infos=infos,
            terminated=terminated,
            truncated=truncated,
        )


def load_callable(spec: str) -> Callable[..., Any]:
    """Load ``module:function`` for project-specific environment construction."""
    if ":" not in spec:
        raise ValueError("Callable must use the module:function form")
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise ValueError(f"Callable was not found: {spec}")
    return function


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def save_rollout(result: RolloutResult, output: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output.with_suffix(".npz"),
        actions=result.actions,
        predictions=result.predictions,
        rewards=result.rewards,
    )
    report = {
        "terminated": result.terminated,
        "truncated": result.truncated,
        "steps": int(len(result.rewards)),
        "return": float(result.rewards.sum()),
        "infos": json_safe(result.infos),
    }
    if metadata:
        report.update(json_safe(dict(metadata)))
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
