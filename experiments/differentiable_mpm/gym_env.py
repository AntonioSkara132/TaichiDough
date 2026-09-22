"""Gymnasium adapter for the forward differentiable MPM solver."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .runtime import init_runtime
from .solver import Stepper
from .state import ParticleState, SimulationConfig, SDFData, ToolControl


class DifferentiableMPMEnv(gym.Env):
    """Forward-only Gymnasium environment around ``differentiable_mpm.Stepper``.

    Actions are two tool linear velocities, flattened as ``[tool0_xyz, tool1_xyz]``.
    The environment does not use the solver's checkpointed or reverse-mode APIs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        simulation_config: SimulationConfig,
        initial_state: ParticleState,
        parameters: Mapping[str, float],
        sdf: SDFData | None = None,
        *,
        tool_poses: np.ndarray | None = None,
        action_substeps: int = 1,
        episode_steps: int | None = None,
        max_tool_velocity: float = 1.0,
        particle_count: int | None = None,
        particle_seed: int = 0,
        reward_fn: Callable | None = None,
        runtime_backend: str = "cpu",
        runtime_cpu_threads: int = 1,
        runtime_debug: bool = False,
        runtime_seed: int = 0,
    ):
        if action_substeps < 1:
            raise ValueError("action_substeps must be positive")
        if episode_steps is not None and episode_steps < 1:
            raise ValueError("episode_steps must be positive")
        if max_tool_velocity <= 0 or not np.isfinite(max_tool_velocity):
            raise ValueError("max_tool_velocity must be finite and positive")

        initial_state.validate()
        source_count = len(initial_state.x)
        if particle_count is not None:
            if particle_count < 1 or particle_count > source_count:
                raise ValueError("particle_count must be between one and the input count")
            rng = np.random.default_rng(particle_seed)
            indices = np.sort(rng.choice(source_count, particle_count, replace=False))
            initial_state = ParticleState(
                **{name: getattr(initial_state, name)[indices].copy()
                   for name in ("x", "v", "C", "F", "Jp")}
            )
            scale = source_count / particle_count
            simulation_config = replace(
                simulation_config,
                n_particles=particle_count,
                particle_mass=simulation_config.particle_mass * scale,
                particle_volume=simulation_config.particle_volume * scale,
            )
        elif simulation_config.n_particles != source_count:
            raise ValueError("simulation_config.n_particles does not match initial_state")

        if simulation_config.n_particles != len(initial_state.x):
            raise ValueError("simulation_config.n_particles does not match selected particles")
        if episode_steps is None:
            episode_steps = 1
        capacity = action_substeps * episode_steps + 2

        init_runtime(
            backend=runtime_backend,
            precision=simulation_config.precision,
            cpu_threads=runtime_cpu_threads,
            debug=runtime_debug,
            seed=runtime_seed,
        )
        self.config = simulation_config
        self.initial_state = initial_state.copy()
        self.parameters = dict(parameters)
        self.sdf = sdf
        self.action_substeps = action_substeps
        self.episode_steps = episode_steps
        self.max_tool_velocity = float(max_tool_velocity)
        self.reward_fn = reward_fn
        self._capacity = capacity
        self._initial_tool_poses = self._validate_poses(tool_poses)
        self.action_space = spaces.Box(
            low=-self.max_tool_velocity, high=self.max_tool_velocity,
            shape=(6,), dtype=np.float32,
        )
        self.observation_space = spaces.Dict({
            "particles": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.config.n_particles, 3), dtype=np.float32,
            ),
            "tool_poses": spaces.Box(
                low=-np.inf, high=np.inf, shape=(2, 7), dtype=np.float32,
            ),
        })
        self._stepper = None
        self._slot = 0
        self._step_count = 0
        self._tool_poses = self._initial_tool_poses.copy()

    @staticmethod
    def _validate_poses(poses):
        if poses is None:
            return ToolControl.stationary().poses.astype(np.float32)
        poses = np.asarray(poses, dtype=np.float32)
        control = ToolControl(poses, np.zeros((2, 6), dtype=np.float32))
        control.validate()
        return poses.copy()

    @classmethod
    def from_prepared_experiment(cls, prepared, **kwargs):
        """Construct from an existing ``PreparedExperiment`` without re-preparing data."""
        tool_poses = None
        if len(prepared.controls):
            tool_poses = prepared.controls[0].poses
        kwargs.setdefault("episode_steps", prepared.total_steps)
        return cls(
            prepared.simulation_config,
            prepared.initial_state,
            prepared.parameters,
            prepared.sdf,
            tool_poses=tool_poses,
            **kwargs,
        )

    def _new_stepper(self):
        self._stepper = Stepper(
            self.config, self.parameters, capacity=self._capacity, sdf=self.sdf
        )
        self._stepper.load_state(0, self.initial_state)
        self._slot = 0

    def _observation(self):
        state = self._stepper.state(self._slot)
        return {
            "particles": state.x.astype(np.float32, copy=False),
            "tool_poses": self._tool_poses.astype(np.float32, copy=True),
        }

    def _info(self):
        return {
            "step": self._step_count,
            "sim_time": self._step_count * self.action_substeps * self.config.dt,
            "particle_count": self.config.n_particles,
            "diagnostics": self._stepper.diagnostics(),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._new_stepper()
        self._step_count = 0
        self._tool_poses = self._initial_tool_poses.copy()
        return self._observation(), self._info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Action must be finite with shape (6,)")
        action = np.clip(action, -self.max_tool_velocity, self.max_tool_velocity)
        velocities = np.zeros((2, 6), dtype=self.config.numpy_dtype)
        velocities[:, :3] = action.reshape(2, 3)
        previous = self._stepper.state(self._slot)
        for _ in range(self.action_substeps):
            self._tool_poses[:, :3] += velocities[:, :3] * self.config.dt
            control = ToolControl(
                self._tool_poses.astype(self.config.numpy_dtype),
                velocities,
                (self._step_count + 1) * self.config.dt,
            )
            self._stepper.advance(self._slot, control)
            self._slot += 1
        self._step_count += 1
        observation = self._observation()
        current = self._stepper.state(self._slot)
        reward = 0.0 if self.reward_fn is None else float(
            self.reward_fn(previous, current, action, self._tool_poses.copy())
        )
        truncated = self._step_count >= self.episode_steps
        return observation, reward, False, truncated, self._info()

    def close(self):
        self._stepper = None
