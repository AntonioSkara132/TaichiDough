import unittest

import numpy as np

from experiments.differentiable_mpm.policy_adapter import (
    CONDITIONS,
    build_condition_poses,
    save_conditions,
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


if __name__ == "__main__":
    unittest.main()
