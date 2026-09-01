#!/usr/bin/env python3
"""Train a small Stable-Baselines3 policy on the Taichi MPM wrapper."""

from __future__ import annotations

import argparse
import json
import time
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


class EpisodeStatsCallback(BaseCallback):
    def __init__(self, total_timesteps, report_every_episodes=10, log_every_timesteps=0):
        super().__init__()
        self.total_timesteps = int(total_timesteps)
        self.report_every_episodes = int(report_every_episodes)
        self.log_every_timesteps = int(log_every_timesteps)
        self.start_time = None
        self.episode_start_time = None
        self.episode_returns = []
        self.episode_final_rewards = []
        self.episode_lengths = []
        self.episode_wall_times = []
        self.current_return = 0.0
        self.current_length = 0
        self.current_final_reward = 0.0
        self.best_mean_return = -float("inf")

    @staticmethod
    def format_duration(seconds):
        seconds = max(float(seconds), 0.0)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if hours:
            return f"{hours:d}h {minutes:02d}m {secs:04.1f}s"
        if minutes:
            return f"{minutes:d}m {secs:04.1f}s"
        return f"{secs:.1f}s"

    def _on_training_start(self):
        self.start_time = time.perf_counter()
        self.episode_start_time = self.start_time

    def _on_step(self):
        now = time.perf_counter()
        if self.log_every_timesteps > 0 and self.num_timesteps % self.log_every_timesteps == 0:
            rewards = self.locals.get("rewards")
            if rewards is not None and len(rewards) > 0:
                print(f"timesteps={self.num_timesteps} reward={float(rewards[0]):.6f}", flush=True)

        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is None or dones is None:
            return True

        reward = float(rewards[0])
        self.current_return += reward
        self.current_length += 1
        self.current_final_reward = reward

        if bool(dones[0]):
            self.episode_returns.append(self.current_return)
            self.episode_final_rewards.append(self.current_final_reward)
            self.episode_lengths.append(self.current_length)
            self.episode_wall_times.append(now - self.episode_start_time)

            episodes = len(self.episode_returns)
            if episodes % self.report_every_episodes == 0:
                window = min(self.report_every_episodes, episodes)
                recent_returns = self.episode_returns[-window:]
                recent_final_rewards = self.episode_final_rewards[-window:]
                recent_lengths = self.episode_lengths[-window:]
                recent_wall_times = self.episode_wall_times[-window:]
                mean_return = sum(recent_returns) / window
                mean_final_reward = sum(recent_final_rewards) / window
                mean_length = sum(recent_lengths) / window
                mean_episode_time = sum(recent_wall_times) / window
                self.best_mean_return = max(self.best_mean_return, mean_return)

                elapsed = now - self.start_time
                steps_per_second = self.num_timesteps / max(elapsed, 1e-12)
                remaining_steps = max(self.total_timesteps - self.num_timesteps, 0)
                eta = remaining_steps / max(steps_per_second, 1e-12)
                print(
                    "episodes={episodes} timesteps={steps}/{total} "
                    "avg_return={avg_return:.6f} avg_final_reward={avg_final:.6f} "
                    "best_avg_return={best:.6f} avg_len={avg_len:.1f} "
                    "avg_episode_time={episode_time} elapsed={elapsed} eta={eta} "
                    "steps_per_sec={sps:.3f}".format(
                        episodes=episodes,
                        steps=self.num_timesteps,
                        total=self.total_timesteps,
                        avg_return=mean_return,
                        avg_final=mean_final_reward,
                        best=self.best_mean_return,
                        avg_len=mean_length,
                        episode_time=self.format_duration(mean_episode_time),
                        elapsed=self.format_duration(elapsed),
                        eta=self.format_duration(eta),
                        sps=steps_per_second,
                    ),
                    flush=True,
                )

            self.current_return = 0.0
            self.current_length = 0
            self.current_final_reward = 0.0
            self.episode_start_time = now

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
    parser.add_argument("--report-every-episodes", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=0)
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

    model.learn(
        total_timesteps=args.timesteps,
        callback=EpisodeStatsCallback(
            total_timesteps=args.timesteps,
            report_every_episodes=args.report_every_episodes,
            log_every_timesteps=args.log_every,
        ),
    )
    model_path = args.output_dir / f"{args.algo}_taichi_mpm"
    model.save(model_path)
    env.close()
    print(f"saved model to {model_path}.zip")


if __name__ == "__main__":
    main()
