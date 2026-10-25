"""Pure-array tests for the velocity-controlled tool path; no simulation required."""
import unittest

import numpy as np

from experiments.differentiable_mpm.state import ToolControl
from experiments.differentiable_mpm.trajectory_optimization import (
    controls_from_velocities, project_speed, smoothness, velocity_path_gradient,
)


class VelocityPathTests(unittest.TestCase):
    def test_integrated_midpoints_and_velocities(self):
        base = ToolControl.stationary()
        base.poses[0, :3] = [.4, .3, .2]
        velocities = np.array([[.1, -.1, .2], [.2, .1, -.1]])
        controls = controls_from_velocities(base, velocities, 0.01)
        np.testing.assert_allclose(controls[0].poses[0, :3], [.4005, .2995, .201])
        np.testing.assert_allclose(controls[1].poses[0, :3], [.402, .2995, .2015])
        np.testing.assert_array_equal(controls[1].velocities[0, :3], velocities[1])
        np.testing.assert_array_equal(controls[0].poses[1], base.poses[1])
        controls[0].poses[0, 0] = 5
        self.assertEqual(base.poses[0, 0], .4)

    def test_vjp_against_central_difference(self):
        rng = np.random.default_rng(21)
        velocities = rng.normal(size=(4, 3)) * .1
        base = ToolControl.stationary()
        pose_bar = rng.normal(size=(4, 2, 7))
        velocity_bar = rng.normal(size=(4, 2, 6))
        dt = .0002

        def objective(v):
            controls = controls_from_velocities(base, v, dt)
            return sum(np.sum(c.poses * pose_bar[t]) + np.sum(c.velocities * velocity_bar[t])
                       for t, c in enumerate(controls))

        g = velocity_path_gradient({"poses": pose_bar, "velocities": velocity_bar}, dt)
        for t in range(4):
            for j in range(3):
                direction = np.zeros_like(velocities)
                direction[t, j] = 1
                fd = (objective(velocities + 1e-4 * direction) -
                      objective(velocities - 1e-4 * direction)) / 2e-4
                self.assertAlmostEqual(g[t, j], fd, places=7)

    def test_smoothness_and_speed_projection(self):
        rng = np.random.default_rng(22)
        v, d = rng.normal(size=(2, 4, 3))
        value, g = smoothness(v, .1)
        fd = (smoothness(v + 1e-5 * d, .1)[0] - smoothness(v - 1e-5 * d, .1)[0]) / 2e-5
        self.assertAlmostEqual(np.sum(g * d), fd, places=9)
        self.assertGreaterEqual(value, 0)
        self.assertTrue(np.all(np.linalg.norm(project_speed(v, .35), axis=1) <= .35 + 1e-12))
        np.testing.assert_allclose(project_speed(v * 0, .35), 0)


if __name__ == "__main__":
    unittest.main()
