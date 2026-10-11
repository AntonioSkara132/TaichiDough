"""Run a dom_retrieval checkpoint through a TaichiDough gym factory.

The environment factory is project-specific and must have the signature
``factory(reward_fn=...)``. This keeps simulator preparation out of the model
adapter while making the rollout command reproducible.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

TAICHIDOUGH_ROOT = Path(__file__).resolve().parents[1]
if str(TAICHIDOUGH_ROOT) not in sys.path:
    sys.path.insert(0, str(TAICHIDOUGH_ROOT))

from experiments.differentiable_mpm.dom_retrieval_adapter import (
    DomRetrievalGymController,
    DomRetrievalPolicyRunner,
    NegativeChamferReward,
    load_callable,
    load_final_goal_cloud,
    save_rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--goal-cloud", required=True, type=Path,
                        help=".npy/.npz goal cloud; a [T,P,3] file uses its final frame")
    parser.add_argument("--env-factory", required=True,
                        help="module:function returning DifferentiableMPMEnv; accepts reward_fn=")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--live-representation", default="filled_volume")
    parser.add_argument("--goal-representation", default="filled_volume")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    goal = load_final_goal_cloud(args.goal_cloud)
    reward = NegativeChamferReward(
        sample_count=args.sample_count,
        live_representation=args.live_representation,
        goal_representation=args.goal_representation,
    )
    factory = load_callable(args.env_factory)
    env = factory(reward_fn=reward)
    try:
        runner = DomRetrievalPolicyRunner(args.checkpoint, device=args.device)
        controller = DomRetrievalGymController(env, runner, reward, goal)
        result = controller.run(max_steps=args.max_steps)
        save_rollout(
            result,
            args.output,
            metadata={
                "checkpoint": str(args.checkpoint.resolve()),
                "goal_cloud": str(args.goal_cloud.resolve()),
                "sample_count": args.sample_count,
                "live_representation": args.live_representation,
                "goal_representation": args.goal_representation,
                "open_loop_waypoints": runner.spec.horizon,
            },
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
