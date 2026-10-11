from __future__ import annotations

from types import SimpleNamespace
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

TAICHIDOUGH = Path(__file__).resolve().parents[2]
DOM_RETRIEVAL = TAICHIDOUGH / "data" / "deformpath_training"
if str(TAICHIDOUGH) not in sys.path:
    sys.path.insert(0, str(TAICHIDOUGH))
if str(DOM_RETRIEVAL) not in sys.path:
    sys.path.insert(0, str(DOM_RETRIEVAL))

from experiments.differentiable_mpm.dom_retrieval_adapter import (  # noqa: E402
    DomRetrievalGymController,
    LivePolicySpec,
    NegativeChamferReward,
    _HistoryBuffer,
    negative_chamfer,
    trajectory_to_actions,
)


class _State:
    def __init__(self, points):
        self.x = np.asarray(points, dtype=np.float32)


class _FakeEnv:
    def __init__(self, reward_fn):
        self.config = SimpleNamespace(dt=0.1)
        self.action_substeps = 1
        self.reward_fn = reward_fn
        self.actions = []
        self.step_count = 0
        self.episode_steps = 3
        self._points = np.zeros((4, 3), dtype=np.float32)
        self._poses = np.zeros((2, 7), dtype=np.float32)
        self._poses[:, 6] = 1.

    def _observation(self):
        return {"particles": self._points.copy(), "tool_poses": self._poses.copy()}

    def reset(self):
        self.step_count = 0
        self.actions.clear()
        return self._observation(), {"step": 0}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.actions.append(action.copy())
        self.step_count += 1
        self._points[:, 0] += 0.01
        reward = self.reward_fn(
            _State(self._points), _State(self._points), action, self._poses
        )
        return self._observation(), reward, False, self.step_count >= self.episode_steps, {"step": self.step_count}


class _FakeRunner:
    def __init__(self):
        self.spec = SimpleNamespace(horizon=3)
        self.calls = 0
        self.observations = []

    def reset(self, observation, time_s=0.0):
        self.observations = [(time_s, observation)]

    def observe(self, observation, time_s):
        self.observations.append((time_s, observation))

    def predict_positions(self):
        self.calls += 1
        return np.array([
            [[0.1, 0., 0.], [0.1, 0., 0.]],
            [[0.2, 0., 0.], [0.2, 0., 0.]],
            [[0.3, 0., 0.], [0.3, 0., 0.]],
        ], dtype=np.float32)


class AdapterTests(unittest.TestCase):
    def test_negative_chamfer_zero_for_identical_clouds(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32)
        self.assertAlmostEqual(negative_chamfer(points, points, sample_count=2), 0.)

    def test_negative_chamfer_is_symmetric_mean_distance(self):
        first = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32)
        second = first + np.array([0., 0., 1.], dtype=np.float32)
        self.assertAlmostEqual(negative_chamfer(first, second, sample_count=2), -2., places=6)

    def test_history_is_a_valid_chronological_prefix(self):
        spec = LivePolicySpec(2, 3, 6, history_length=4, history_window_seconds=0.7, history_tool_dim=14)
        history = _HistoryBuffer(spec)
        poses = np.zeros((2, 7), dtype=np.float32)
        poses[:, 6] = 1.
        cloud = np.zeros((2, 3), dtype=np.float32)
        history.append(1.0, cloud, poses)
        history.append(1.1, cloud + 1., poses)
        batch = history.observation()
        lengths = __import__("dom_retrieval.observations", fromlist=["validate_history"]).validate_history(
            {key: value.unsqueeze(0) for key, value in batch.items()}
        )
        self.assertEqual(lengths.tolist(), [2])
        self.assertTrue(torch.equal(batch["observation"], batch["history_clouds"][1]))

    def test_trajectory_to_actions_uses_current_pose_for_first_waypoint(self):
        poses = np.zeros((2, 7), dtype=np.float32)
        poses[:, 6] = 1.
        positions = np.array([
            [[1., 0., 0.], [0., 1., 0.]],
            [[2., 0., 0.], [0., 2., 0.]],
        ], dtype=np.float32)
        actions = trajectory_to_actions(positions, poses, 0.5)
        np.testing.assert_allclose(actions[0], [2., 0., 0., 0., 2., 0.])
        np.testing.assert_allclose(actions[1], [2., 0., 0., 0., 2., 0.])

    def test_controller_executes_full_prediction_before_replanning(self):
        reward = NegativeChamferReward(sample_count=4)
        env = _FakeEnv(reward)
        runner = _FakeRunner()
        goal = np.zeros((4, 3), dtype=np.float32)
        result = DomRetrievalGymController(env, runner, reward, goal).run()
        self.assertEqual(len(env.actions), 3)
        self.assertEqual(runner.calls, 1)
        self.assertEqual(result.actions.shape, (3, 6))
        self.assertEqual(result.rewards.shape, (3,))
        self.assertTrue(result.truncated)


if __name__ == "__main__":
    unittest.main()
