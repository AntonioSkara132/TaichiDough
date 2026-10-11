import unittest

import numpy as np

from experiments.differentiable_mpm.policy_adapter import (
    CONDITIONS,
    DomRetrievalGymAdapter,
    LiveObservationBuffer,
    TerminalChamferReward,
    build_condition_poses,
    final_episode_target_cloud,
    sample_point_cloud,
    save_conditions,
    symmetric_chamfer_distance,
    waypoints_to_actions,
)


class PolicyAdapterTests(unittest.TestCase):
    def setUp(self):
        self.times = np.array([0., 1.])
        self.poses = np.zeros((2, 2, 7), dtype=float)
        self.poses[:, :, 6] = 1.
        self.poses[0, 0, :3] = [-.1, .2, .3]
        self.poses[0, 1, :3] = [.1, .2, .3]
        self.poses[1] = self.poses[0]
        self.poses[1, :, 0] += .1
        self.predicted_times = np.array([0., 1.])
        self.predicted = self.poses[:, :, :3].copy()
        self.predicted[:, :, 0] += .02

    def test_conditions_have_valid_poses_and_velocities(self):
        for condition in CONDITIONS:
            poses, velocities = build_condition_poses(
                condition, np.linspace(0., 1., 6), self.times, self.poses,
                self.predicted_times, self.predicted,
            )
            self.assertEqual(poses.shape, (6, 2, 7))
            self.assertEqual(velocities.shape, (6, 2, 6))
            self.assertTrue(np.allclose(np.linalg.norm(poses[:, :, 3:], axis=-1), 1.))
            if condition == "hold_position":
                self.assertTrue(np.allclose(velocities, 0.))

    def test_predicted_path_does_not_teleport_start(self):
        poses, _ = build_condition_poses(
            "predicted_xyz_fixed_orientation", np.array([0., 1.]), self.times, self.poses,
            self.predicted_times, self.predicted,
        )
        self.assertTrue(np.allclose(poses[0, :, :3], self.predicted[0]))
        self.assertFalse(np.allclose(poses[0, :, :3], self.poses[0, :, :3]))

    def test_save_manifest(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            poses, velocities = build_condition_poses(
                "hold_position", np.array([0., .1]), self.times, self.poses,
            )
            manifest = save_conditions(directory, {"hold_position": (poses, velocities)})
            self.assertEqual(manifest["schema"], "taichidough/mpm-policy-controls/v1")
            self.assertTrue((__import__("pathlib").Path(directory) / "hold_position.npz").is_file())


    def test_sample_point_cloud_is_order_independent_and_repeats(self):
        points = np.array([[2., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
        sampled = sample_point_cloud(points[::-1], 5)
        self.assertTrue(np.allclose(sampled[:, 0], [0., 0., 1., 1., 2.]))

    def test_chamfer_is_symmetric_mean_and_reward_is_negative(self):
        first = np.array([[0., 0., 0.], [1., 0., 0.]])
        second = np.array([[0., 0., 0.]])
        self.assertAlmostEqual(symmetric_chamfer_distance(first, second), .5)
        reward = TerminalChamferReward(second)
        state = type("State", (), {"x": first})()
        self.assertAlmostEqual(reward(None, state, None, None), -.5)

    def test_reward_rejects_mismatched_cloud_representations(self):
        with self.assertRaises(ValueError):
            TerminalChamferReward(
                np.zeros((2, 3)),
                live_representation="filled_volume",
                target_representation="partial_camera_surface",
            )

    def test_history_buffer_keeps_current_cloud_at_last_valid_slot(self):
        buffer = LiveObservationBuffer(2, history_length=3, history_window_seconds=1.,
                                       history_max_gap_seconds=.5)
        poses = np.zeros((2, 7), dtype=np.float32)
        poses[:, 6] = 1.
        for index, time in enumerate((0., .1, .2)):
            particles = np.array([[index, 0., 0.], [index + 1., 0., 0.]])
            buffer.append(particles, poses, time)
        batch = buffer.as_torch()
        self.assertTrue(np.array_equal(batch["history_mask"].numpy(), [[True, True, True]]))
        self.assertTrue(np.allclose(batch["history_times"].numpy(), [[-.2, -.1, 0.]]))
        self.assertTrue(np.allclose(batch["observation"].numpy()[0], [[2., 0., 0.], [3., 0., 0.]]))
        self.assertTrue(np.allclose(batch["history_clouds"].numpy()[0, -1], batch["observation"].numpy()[0]))

    def test_history_gap_keeps_current_only(self):
        buffer = LiveObservationBuffer(1, history_length=3, history_window_seconds=1.,
                                       history_max_gap_seconds=.1)
        poses = np.zeros((2, 7), dtype=np.float32)
        poses[:, 6] = 1.
        buffer.append([[0., 0., 0.]], poses, 0.)
        buffer.append([[1., 0., 0.]], poses, .2)
        batch = buffer.as_torch()
        self.assertTrue(np.array_equal(batch["history_mask"].numpy(), [[True, False, False]]))
        self.assertTrue(np.allclose(batch["history_times"].numpy(), [[0., 0., 0.]]))

    def test_waypoints_to_actions_uses_current_pose_as_start(self):
        waypoints = np.array([
            [[0., 0., 0.], [0., 0., 0.]],
            [[1., 0., 0.], [0., 1., 0.]],
            [[2., 0., 0.], [0., 2., 0.]],
        ])
        actions = waypoints_to_actions(waypoints, np.zeros((2, 3)), .5)
        self.assertEqual(actions.shape, (2, 6))
        self.assertTrue(np.allclose(actions, [[2., 0., 0., 0., 2., 0.],
                                              [2., 0., 0., 0., 2., 0.]]))

    def test_final_target_uses_last_frame_and_scene_callbacks(self):
        class Sequence:
            points = [np.array([[0., 0., 0.]]), np.array([[1., 0., 0.], [2., 0., 0.]])]

        class Prepared:
            sequence = Sequence()
            calibration = "calibration"
            config = type("Config", (), {"observation": type("Observation", (), {"trim_quantile": 0.})()})()

        target = final_episode_target_cloud(
            Prepared(), point_count=3,
            apply_calibration=lambda points, calibration: points + [0., 1., 0.],
        )
        self.assertTrue(np.allclose(target[:, 1], 1.))
        self.assertTrue(np.allclose(target[:, 0], [1., 1., 2.]))

    def test_live_adapter_rollout_calls_action_hook(self):
        class Config:
            dt = .1

        class FakeEnv:
            config = Config()
            action_substeps = 1
            episode_steps = 2
            max_tool_velocity = 10.

            def __init__(self):
                self.time = 0.

            @staticmethod
            def _poses():
                poses = np.zeros((2, 7), dtype=np.float32)
                poses[:, 6] = 1.
                return poses

            def reset(self, *, seed=None, options=None):
                del seed, options
                self.time = 0.
                return {"particles": np.zeros((2, 3), dtype=np.float32),
                        "tool_poses": self._poses()}, {"sim_time": 0.}

            def step(self, action):
                self.time += .1
                self.last_action = action
                done = self.time >= .2
                return {"particles": np.zeros((2, 3), dtype=np.float32),
                        "tool_poses": self._poses()}, 0., False, done, {"sim_time": self.time}

        class FakeModel:
            def __call__(self, observation):
                import torch
                self.last_input = observation
                return torch.tensor([[[0., 0., 0., 0., 0., 0.],
                                      [.2, 0., 0., 0., .2, 0.]]])

        seen = []
        env = FakeEnv()
        model = FakeModel()
        adapter = DomRetrievalGymAdapter(
            env, model, config={"data": {"points": 2}, "history": {"enabled": False}},
            action_hook=lambda action, observation, info: seen.append((observation, info)) or action,
        )
        result = adapter.rollout()
        self.assertEqual(len(result), 3)
        self.assertEqual(len(seen), 2)
        self.assertEqual(model.last_input.shape, (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
