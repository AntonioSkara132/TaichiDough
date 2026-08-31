#!/usr/bin/env python3
"""Gymnasium/LeRobot-compatible wrapper for the Taichi viscoelastic MPM scene."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import taichi as ti
from gymnasium import spaces

try:
    from taichi_viscoelastic_mpm_scene import SCENE_CENTER, TOOL_INITIAL_POSES, build_sim
except ModuleNotFoundError:
    from .taichi_viscoelastic_mpm_scene import SCENE_CENTER, TOOL_INITIAL_POSES, build_sim


def make_sim_args(**overrides):
    defaults = {
        "particles": 12000,
        "grid": 40,
        "dt": 2e-4,
        "substeps_per_frame": 8,
        "youngs_modulus": 1000.0,
        "poisson_ratio": 0.35,
        "viscosity": 2.5,
        "density": 1100.0,
        "gravity": -9.81,
        "floor_y": 0.20,
        "floor_friction": 0.7,
        "floor_absorption": 0.0,
        "tool_close_time": 0.04,
        "tool_motion_start": 1.0,
        "tool_contact_padding": 0.035,
        "tool_contact_friction": 0.75,
        "tool_contact_absorption": 0.0,
        "tool_stickiness": 0.0,
        "floor_stickiness": 0.0,
        "floor_plastic_damping_band": 0.02,
        "velocity_damping": 0.998,
        "pure_viscoelastic": False,
        "plastic_min": 0.88,
        "plastic_max": 1.08,
        "plastic_velocity_damping": 0.92,
        "plastic_affine_damping": 0.80,
        "use_jp": False,
        "jp_hardening": 0.0,
        "jp_min": 0.6,
        "jp_max": 2.0,
        "ros_control": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TaichiViscoelasticMPMEnv(gym.Env):
    """Headless MPM environment suitable for LeRobot-style RL/data collection.

    Action is two 3D tool linear velocities in simulation units:
        [left_vx, left_vy, left_vz, right_vx, right_vy, right_vz]

    Reward is:
        -mean_i ||x_i - mean_j(x_j)||^2
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sim_args=None,
        episode_steps=250,
        action_substeps=8,
        max_tool_velocity=0.20,
        observation_particles=4096,
        seed=None,
        arch=None,
    ):
        super().__init__()
        self.sim_args = sim_args or make_sim_args()
        self.episode_steps = int(episode_steps)
        self.action_substeps = int(action_substeps)
        self.max_tool_velocity = float(max_tool_velocity)
        self.observation_particles = min(int(observation_particles), int(self.sim_args.particles))
        self.rng = np.random.default_rng(seed)

        ti.init(arch=arch or ti.cpu)
        self.x, self.tool_x, self.initialize_kernel, self.substep_kernel, self.update_tool_visuals, self.set_tool_state = (
            build_sim(self.sim_args)
        )

        self.tool_poses = TOOL_INITIAL_POSES.copy()
        self.tool_velocities = np.zeros((2, 3), dtype=np.float32)
        self.elapsed_sim_time = 0.0
        self.step_count = 0
        self.particle_indices = np.arange(self.observation_particles)

        self.action_space = spaces.Box(
            low=-self.max_tool_velocity,
            high=self.max_tool_velocity,
            shape=(6,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "observation.state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.observation_particles * 3 + 14,),
                    dtype=np.float32,
                ),
                "observation.particles": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.observation_particles, 3),
                    dtype=np.float32,
                ),
                "observation.tool_poses": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2, 7),
                    dtype=np.float32,
                ),
            }
        )

    def _sample_particle_indices(self):
        n_particles = int(self.sim_args.particles)
        if self.observation_particles >= n_particles:
            self.particle_indices = np.arange(n_particles)
        else:
            self.particle_indices = self.rng.choice(n_particles, size=self.observation_particles, replace=False)

    def _particle_array(self):
        return self.x.to_numpy().astype(np.float32, copy=False)

    def _reward(self, particles):
        center = particles.mean(axis=0, keepdims=True)
        squared_distances = np.sum((particles - center) ** 2, axis=1)
        return -float(np.mean(squared_distances))

    def _observation(self):
        particles = self._particle_array()
        sampled_particles = particles[self.particle_indices]
        tool_poses = self.tool_poses.astype(np.float32, copy=True)
        state = np.concatenate([sampled_particles.reshape(-1), tool_poses.reshape(-1)]).astype(np.float32)
        return {
            "observation.state": state,
            "observation.particles": sampled_particles.astype(np.float32, copy=False),
            "observation.tool_poses": tool_poses,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.initialize_kernel()
        self.tool_poses = TOOL_INITIAL_POSES.copy()
        self.tool_velocities.fill(0.0)
        self.set_tool_state(self.tool_poses, self.tool_velocities)
        self.update_tool_visuals(0.0)
        self.elapsed_sim_time = 0.0
        self.step_count = 0
        self._sample_particle_indices()
        obs = self._observation()
        reward = self._reward(self._particle_array())
        return obs, {"reward": reward, "sim_time": self.elapsed_sim_time}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2, 3)
        action = np.clip(action, -self.max_tool_velocity, self.max_tool_velocity)
        self.tool_velocities = action

        for _ in range(self.action_substeps):
            self.tool_poses[:, :3] += self.tool_velocities * self.sim_args.dt
            self.set_tool_state(self.tool_poses, self.tool_velocities)
            self.substep_kernel(self.elapsed_sim_time)
            self.elapsed_sim_time += self.sim_args.dt

        self.update_tool_visuals(self.elapsed_sim_time)
        self.step_count += 1

        particles = self._particle_array()
        obs = self._observation()
        reward = self._reward(particles)
        terminated = False
        truncated = self.step_count >= self.episode_steps
        info = {
            "sim_time": self.elapsed_sim_time,
            "dough_center": particles.mean(axis=0).astype(np.float32),
            "dough_spread_reward": reward,
        }
        return obs, reward, terminated, truncated, info

    @property
    def lerobot_features(self):
        return {
            "observation.state": {
                "dtype": "float32",
                "shape": (self.observation_particles * 3 + 14,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["left_vx", "left_vy", "left_vz", "right_vx", "right_vy", "right_vz"],
            },
        }

    def close(self):
        pass


def smoke_test(args):
    sim_args = make_sim_args(
        particles=args.particles,
        grid=args.grid,
        dt=args.dt,
        youngs_modulus=args.youngs_modulus,
        viscosity=args.viscosity,
        floor_friction=args.floor_friction,
        tool_contact_padding=args.tool_contact_padding,
        tool_contact_friction=args.tool_contact_friction,
    )
    env = TaichiViscoelasticMPMEnv(
        sim_args=sim_args,
        episode_steps=args.episode_steps,
        action_substeps=args.action_substeps,
        max_tool_velocity=args.max_tool_velocity,
        observation_particles=args.observation_particles,
        seed=args.seed,
        arch=ti.cpu if args.cpu else ti.gpu,
    )
    obs, info = env.reset(seed=args.seed)
    print("reset reward", info["reward"], "state shape", obs["observation.state"].shape)
    action = np.array([0.0, 0.0, -0.05, 0.0, 0.0, 0.05], dtype=np.float32)
    for step in range(args.smoke_steps):
        obs, reward, terminated, truncated, info = env.step(action)
        print(step, "reward", reward, "sim_time", info["sim_time"], "center", info["dough_center"])
        if terminated or truncated:
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test for the Taichi LeRobot/Gymnasium wrapper.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--particles", type=int, default=4000)
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--dt", type=float, default=2e-4)
    parser.add_argument("--episode-steps", type=int, default=250)
    parser.add_argument("--action-substeps", type=int, default=8)
    parser.add_argument("--max-tool-velocity", type=float, default=0.20)
    parser.add_argument("--observation-particles", type=int, default=1024)
    parser.add_argument("--youngs-modulus", type=float, default=1000.0)
    parser.add_argument("--viscosity", type=float, default=2.5)
    parser.add_argument("--floor-friction", type=float, default=0.7)
    parser.add_argument("--tool-contact-padding", type=float, default=0.035)
    parser.add_argument("--tool-contact-friction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-steps", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    smoke_test(parse_args())
