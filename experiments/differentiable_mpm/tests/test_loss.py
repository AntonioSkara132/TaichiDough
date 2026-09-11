"""Run independently: python3 -m unittest experiments.differentiable_mpm.tests.test_loss -v."""
import unittest

import numpy as np
import taichi as ti

from experiments.differentiable_mpm.loss import LossConfig, Observation, ObservationLoss
from experiments.differentiable_mpm.renderer import Camera, LocalSplatRenderer
from experiments.differentiable_mpm.state import InvalidStateError


class ObservationLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1,
                offline_cache=False, enable_fallback=False)
        cls.camera = Camera(24, 20, 30.0, 30.0, 12.0, 10.0, np.eye(4))
        cls.initial = np.array([[-0.013, -0.006, 1.0], [0.027, 0.019, 1.012]])
        depth = np.ones((20, 24))
        mask = np.zeros((20, 24), dtype=bool)
        mask[9:12, 11:15] = True
        cls.observation = Observation(1, 2, depth + 0.002, mask, depth, mask)
        cls.objective = ObservationLoss(cls.camera, LossConfig(), cls.initial, precision="f64")

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def finite_difference(self, objective, positions, observation, h):
        result = np.zeros_like(positions)
        for i in range(len(positions)):
            for j in range(3):
                plus, minus = positions.copy(), positions.copy()
                plus[i, j] += h
                minus[i, j] -= h
                result[i, j] = (
                    objective.value_and_grad_positions(plus, observation, False).value
                    - objective.value_and_grad_positions(minus, observation, False).value
                ) / (2 * h)
        return result

    def test_coordinate_and_directional_position_gradients(self):
        positions = self.initial + 0.003
        result = self.objective.value_and_grad_positions(positions, self.observation)
        errors = []
        for h in (1e-5, 1e-6, 1e-7):
            numeric = self.finite_difference(self.objective, positions, self.observation, h)
            errors.append(float(np.max(np.abs(numeric - result.gradient))))
        self.assertLess(errors[1], 1e-6)
        self.assertLess(errors[2], 1e-6)
        self.assertLess(errors[1], errors[0] * 0.05)
        direction = np.array([[0.1, -0.2, 0.3], [-0.2, 0.3, 0.1]])
        h = 1e-6
        numeric = (self.objective.value_and_grad_positions(positions + h * direction, self.observation, False).value
                   - self.objective.value_and_grad_positions(positions - h * direction, self.observation, False).value) / (2 * h)
        self.assertAlmostEqual(float(np.sum(direction * result.gradient)), numeric, delta=1e-6)
        again = self.objective.value_and_grad_positions(positions, self.observation)
        np.testing.assert_array_equal(again.gradient, result.gradient)
        self.assertEqual(again.value, result.value)

    def test_equal_depth_anchor_switch_has_correct_derivative(self):
        positions = self.initial.copy()
        positions[:, 2] = 1.004
        config = LossConfig(distance_weight=0.0)
        objective = ObservationLoss(self.camera, config, self.initial, precision="f64")
        result = objective.value_and_grad_positions(positions, self.observation)
        numeric = self.finite_difference(objective, positions, self.observation, 1e-7)
        np.testing.assert_allclose(result.gradient, numeric, rtol=2e-5, atol=2e-6)
        self.assertGreater(np.linalg.norm(result.gradient), 0.01)

    def test_metric_projection_and_scaled_intrinsics(self):
        angle = 0.24
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [0.04, -0.03, 0.2]
        camera = Camera(24, 20, 30, 28, 11.2, 9.4, transform)
        optical = np.array([[0.02, -0.01, 1.0]])
        scene = (optical - transform[:3, 3]) @ rotation
        renderer = LocalSplatRenderer(camera, 1, precision="f64")
        result = renderer.forward(scene)
        np.testing.assert_allclose(result.optical_positions, optical, atol=1e-14)
        np.testing.assert_allclose(result.projected_positions, [[11.8, 9.12]], atol=1e-13)
        half = camera.scaled(12, 10)
        self.assertEqual(half.fx, 15)
        self.assertEqual(half.cy, 4.7)
        depth = np.zeros((20, 24))
        valid = np.zeros((20, 24), bool)
        depth[8, 10], valid[8, 10] = 1.4, True
        point = camera.optical_points(depth, valid)[0]
        np.testing.assert_allclose(point, [(10.5 - 11.2) * 1.4 / 30,
                                           (8.5 - 9.4) * 1.4 / 28, 1.4])

    def test_positive_coverage_has_lateral_gradient(self):
        config = LossConfig(depth_weight=0.0, coverage_weight=1.0, distance_weight=0.0)
        objective = ObservationLoss(self.camera, config, self.initial, precision="f64")
        positions = self.initial + [0.008, -0.004, 0.002]
        result = objective.value_and_grad_positions(positions, self.observation)
        self.assertGreater(np.linalg.norm(result.gradient[:, :2]), 0.01)
        numeric = self.finite_difference(objective, positions, self.observation, 1e-7)
        np.testing.assert_allclose(result.gradient, numeric, rtol=2e-5, atol=2e-6)
        # Values outside the observed mask are unknown and cannot change the loss.
        depth = self.observation.observed_depth.copy()
        depth[~self.observation.observed_valid] = np.nan
        other = Observation(1, 2, depth, self.observation.observed_valid,
                            self.observation.observed_initial_depth,
                            self.observation.observed_initial_valid)
        unknown = objective.value_and_grad_positions(positions, other)
        self.assertEqual(result.value, unknown.value)
        np.testing.assert_array_equal(result.gradient, unknown.gradient)

    def test_no_overlap_retains_penalties_and_lateral_gradient(self):
        positions = self.initial + [0.28, 0.0, 0.0]
        result = self.objective.value_and_grad_positions(positions, self.observation)
        self.assertEqual(result.diagnostics["current_overlap_pixels"], 0)
        self.assertEqual(result.diagnostics["fixed_depth_pixels"], 12)
        self.assertAlmostEqual(result.components["positive_coverage"], 1.0)
        self.assertAlmostEqual(result.components["depth_change"], 3.5)
        self.assertGreater(result.gradient[:, 0].sum(), 1.0)
        improved = self.objective.value_and_grad_positions(positions - 1e-5 * result.gradient,
                                                          self.observation, False)
        self.assertLess(improved.value, result.value)
        numeric = self.finite_difference(self.objective, positions, self.observation, 1e-7)
        np.testing.assert_allclose(result.gradient, numeric, rtol=2e-5, atol=2e-6)

    def test_occluded_interior_is_not_compared_as_an_exterior(self):
        front = np.array([[-0.013, -0.006, 1.0]])
        # The back particle projects to precisely the same continuous coordinate.
        pair = np.concatenate((front, front * 1.1), axis=0)
        points = front.copy()
        mask = np.zeros((20, 24), bool)
        mask[9:11, 11:13] = True
        depth = np.ones((20, 24))
        observation = Observation(1, 1, depth + 0.002, mask, depth, mask, points)
        objective = ObservationLoss(self.camera, LossConfig(), pair, precision="f64")
        positions = pair.copy()
        positions[0] += [0.001, 0.002, 0.003]
        result = objective.value_and_grad_positions(positions, observation)
        self.assertEqual(result.diagnostics["visible_predicted_particles"], 1)
        self.assertLess(np.linalg.norm(result.gradient[1]), 1e-4)
        self.assertGreater(np.linalg.norm(result.gradient[0]), 0.1)
        moved_back = positions.copy()
        moved_back[1] *= 1.03
        moved = objective.value_and_grad_positions(moved_back, observation, False)
        self.assertAlmostEqual(result.value, moved.value, delta=1e-6)
        numeric = self.finite_difference(objective, positions, observation, 1e-7)
        np.testing.assert_allclose(result.gradient, numeric, rtol=2e-5, atol=2e-6)

    def test_offscreen_empty_nonfinite_and_missing_initial_support_fail(self):
        with self.assertRaises(InvalidStateError):
            self.objective.value_and_grad_positions(self.initial + [100, 0, 0], self.observation)
        mask = np.zeros((20, 24), bool)
        empty = Observation(1, 2, self.observation.observed_depth, mask,
                            self.observation.observed_initial_depth, mask)
        with self.assertRaises(InvalidStateError):
            self.objective.value_and_grad_positions(self.initial, empty)
        missing_initial = Observation(1, 2, self.observation.observed_depth,
                                      self.observation.observed_valid,
                                      self.observation.observed_initial_depth, mask)
        with self.assertRaises(InvalidStateError):
            self.objective.value_and_grad_positions(self.initial, missing_initial)
        positions = self.initial.copy()
        positions[0, 0] = np.nan
        with self.assertRaises(InvalidStateError):
            self.objective.value_and_grad_positions(positions, self.observation)
        bad_depth = self.observation.observed_depth.copy()
        bad_depth[10, 12] = np.inf
        invalid = Observation(1, 2, bad_depth, self.observation.observed_valid,
                              self.observation.observed_initial_depth,
                              self.observation.observed_initial_valid)
        with self.assertRaises(InvalidStateError):
            self.objective.value_and_grad_positions(self.initial, invalid)

    def test_near_far_clipping_and_clipped_particle_gradients(self):
        camera = Camera(24, 20, 30.0, 30.0, 12.0, 10.0, np.eye(4), near_m=0.9, far_m=1.02)
        self.assertEqual(camera.scaled(12, 10).far_m, 1.02)
        self.assertEqual(camera.as_dict()["far_m"], 1.02)
        depths = np.array([0.9, 1.02, 1.019, 0.901, 0.899, 1.021])
        positions = np.column_stack((np.zeros(6), np.zeros(6), depths))
        renderer = LocalSplatRenderer(camera, len(positions), precision="f64")
        rendered = renderer.forward(positions)
        np.testing.assert_array_equal(rendered.active_particles, [False, False, True, True, False, False])
        objective = ObservationLoss(camera, LossConfig(), self.initial, precision="f64")
        current = self.initial.copy()
        current[0] += [0.001, -0.001, 0.003]
        current[1, 2] = 1.03
        record = objective.value_and_grad_positions(current, self.observation)
        self.assertEqual(record.diagnostics["active_predicted_particles"], 1)
        np.testing.assert_array_equal(record.gradient[1], [0.0, 0.0, 0.0])
        numeric = self.finite_difference(objective, current, self.observation, 1e-7)
        np.testing.assert_allclose(record.gradient, numeric, rtol=2e-5, atol=2e-6)
        with self.assertRaises(InvalidStateError):
            objective.value_and_grad_positions(self.initial + [0, 0, 0.1], self.observation)
        far_depth = self.observation.observed_depth.copy()
        far_depth[self.observation.observed_valid] = 1.03
        outside = Observation(1, 2, far_depth, self.observation.observed_valid,
                              self.observation.observed_initial_depth,
                              self.observation.observed_initial_valid)
        with self.assertRaises(InvalidStateError):
            objective.value_and_grad_positions(self.initial, outside)
        for far in (0.8, 0.9, np.inf):
            with self.assertRaises(ValueError):
                Camera(24, 20, 30, 30, 12, 10, np.eye(4), near_m=0.9, far_m=far)

    def test_v2_depth_is_continuous_at_last_splat_boundary(self):
        initial = np.array([[0.5 / 30.0, 0.5 / 30.0, 1.0]])
        depth = np.ones((20, 24))
        mask = np.zeros((20, 24), bool)
        mask[10, 12] = True
        observation = Observation(1, 1, depth + 0.002, mask, depth, mask)
        settings = dict(depth_weight=1.0, coverage_weight=0.0, distance_weight=0.0, min_observed_pixels=1)
        v2 = ObservationLoss(self.camera, LossConfig(**settings), initial, precision="f64")
        v1 = ObservationLoss(self.camera, LossConfig(version="partial-visible-splats-v1", **settings),
                             initial, precision="f64")
        boundary = initial + [2.0 / 30.0, 0.0, 0.0]
        inside = boundary - [1e-7, 0.0, 0.0]
        outside = boundary + [1e-7, 0.0, 0.0]
        before = v2.value_and_grad_positions(inside, observation)
        after = v2.value_and_grad_positions(outside, observation)
        self.assertAlmostEqual(before.value, after.value, delta=1e-8)
        self.assertAlmostEqual(after.value, 3.5)
        np.testing.assert_array_equal(after.gradient, np.zeros((1, 3)))
        old_before = v1.value_and_grad_positions(inside, observation, False)
        old_after = v1.value_and_grad_positions(outside, observation, False)
        self.assertGreater(abs(old_after.value - old_before.value), 3.0)
        nearby = boundary - [1e-4, 0.0, 0.0]
        derivative = v2.value_and_grad_positions(nearby, observation).gradient
        numeric = self.finite_difference(v2, nearby, observation, 1e-7)
        np.testing.assert_allclose(derivative, numeric, rtol=2e-5, atol=2e-6)

    def test_v1_is_selectable_and_reproduces_previous_value(self):
        config = LossConfig(version="partial-visible-splats-v1")
        old = ObservationLoss(self.camera, config, self.initial, precision="f64")
        value = old.value_and_grad_positions(self.initial + 0.003, self.observation)
        self.assertAlmostEqual(value.value, 2.910763571327664, places=10)
        self.assertEqual(value.diagnostics["loss_version"], "partial-visible-splats-v1")
        self.assertEqual(LossConfig().version, "partial-visible-splats-v2")
        self.assertEqual(LossConfig(**config.as_dict()), config)
        self.assertNotEqual(config.fingerprint(self.camera), LossConfig().fingerprint(self.camera))
        with self.assertRaises(ValueError):
            LossConfig(version="unknown")

    def test_fingerprint_records_loss_and_camera(self):
        config = LossConfig()
        self.assertEqual(config.fingerprint(self.camera), LossConfig().fingerprint(self.camera))
        self.assertNotEqual(config.fingerprint(self.camera), LossConfig(footprint_radius=3).fingerprint(self.camera))
        self.assertNotEqual(config.fingerprint(self.camera), config.fingerprint(self.camera.scaled(12, 10)))


if __name__ == "__main__":
    unittest.main()
