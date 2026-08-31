#!/usr/bin/env python3
"""Train a small Stable-Baselines3 policy on the Taichi MPM wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import taichi as ti
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from taichi_lerobot_env import TaichiViscoelasticMPMEnv, make_sim_args


class StateOnlyWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space["observation.state"]

    def observation(self, observation):
        return observation["observation.state"]


class ProgressCallback(BaseCallback):
    def __init__(self, log_every=100):
        super().__init__()
        self.log_every = int(log_every)

    def _on_step(self):
        if self.num_timesteps % self.log_every == 0:
            rewards = self.locals.get("rewards")
            if rewards is not None and len(rewards) > 0:
                print(f"timesteps={self.num_timesteps} reward={float(rewards[0]):.6f}", flush=True)
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train SB3 on TaichiViscoelasticMPMEnv.")
    parser.add_argument("--algo", choices=("sac", "ppo"), default="sac")
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("data/taichi_sb3_runs/smoke"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--observation-particles", type=int, default=128)
    parser.add_argument("--episode-steps", type=int, default=64)
    parser.add_argument("--action-substeps", type=int, default=8)
    parser.add_argument("--max-tool-velocity", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def make_env(args):
    sim_args = make_sim_args(
        particles=args.particles,
        grid=args.grid,
        ros_control=True,
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
    return Monitor(StateOnlyWrapper(env))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args)

    policy_kwargs = {"net_arch": [128, 128]}
    if args.algo == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=max(1000, args.timesteps),
            learning_starts=min(50, max(1, args.timesteps // 5)),
            batch_size=32,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=args.seed,
            device="cpu",
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=min(64, args.episode_steps),
            batch_size=32,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=args.seed,
            device="cpu",
        )

    config_path = args.output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"output_dir": str(args.output_dir)}, f, indent=2)

    model.learn(total_timesteps=args.timesteps, callback=ProgressCallback(args.log_every))
    model_path = args.output_dir / f"{args.algo}_taichi_mpm"
    model.save(model_path)
    env.close()
    print(f"saved model to {model_path}.zip")


if __name__ == "__main__":
    main()
