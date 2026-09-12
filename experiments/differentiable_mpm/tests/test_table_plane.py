"""Host-only tests for the deterministic table-plane estimator."""
import unittest

import numpy as np

from experiments.differentiable_mpm.estimate_table_plane import (
    PlaneFit,
    aggregate_equal_frame,
    bootstrap_planes,
    exclude_oriented_boxes,
    fit_frame_plane,
    orient_plane_upward,
    plane_from_regression,
    regression_from_plane,
    signed_distance,
)


class TablePlaneTests(unittest.TestCase):
    def synthetic_points(self, coefficients, count=3000, noise=0.0005, seed=3):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-0.35, 0.35, count)
        z = rng.uniform(0.35, 1.05, count)
        a, b, c = coefficients
        y = a * x + b * z + c + rng.normal(0.0, noise, count)
        return np.column_stack((x, y, z))

    def test_recovers_known_tilted_plane_with_heavy_outliers(self):
        expected = np.array([-0.075, 0.018, 0.071])
        plane_points = self.synthetic_points(expected, count=3500)
        rng = np.random.default_rng(12)
        outliers = np.column_stack((
            rng.uniform(-0.35, 0.35, 2500),
            rng.uniform(-0.15, 0.35, 2500),
            rng.uniform(0.35, 1.05, 2500),
        ))
        fit = fit_frame_plane(
            np.concatenate((plane_points, outliers)), seed=41, sample_limit=6000,
            iterations=180, threshold_m=0.0025, minimum_inliers=1000,
        )
        self.assertTrue(fit.accepted)
        np.testing.assert_allclose(fit.coefficients, expected, atol=2e-4, rtol=0)
        self.assertGreater(fit.inlier_fraction, 0.5)
        self.assertLess(fit.median_abs_residual_m, 0.0005)

    def test_excludes_rotated_tool_clusters(self):
        angle = np.deg2rad(90.0)
        quaternion = np.array([0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)])
        poses = np.array([
            [0.1, 0.08, 0.7, *quaternion],
            [-0.2, 0.12, 0.5, 0.0, 0.0, 0.0, 1.0],
        ])
        half_extents = np.array([[0.04, 0.03, 0.08], [0.02, 0.02, 0.02]])
        points = np.array([
            [0.17, 0.08, 0.7],
            [0.1, 0.08, 0.75],
            [-0.2, 0.12, 0.5],
            [0.35, 0.05, 0.8],
        ])
        keep, excluded = exclude_oriented_boxes(points, poses, half_extents, 0.0)
        np.testing.assert_array_equal(keep, [False, True, False, True])
        self.assertEqual(excluded, 2)

    def test_upward_orientation_and_positive_side(self):
        coefficients = np.array([-0.08, -0.01, 0.07])
        plane = plane_from_regression(coefficients)
        points_above = np.array([[0.0, 0.09, 0.7], [0.1, 0.08, 0.6]])
        oriented = orient_plane_upward(-plane, points_above)
        self.assertGreater(oriented[1], 0)
        self.assertGreater(np.median(signed_distance(points_above, oriented)), 0)
        np.testing.assert_allclose(regression_from_plane(oriented), coefficients, atol=1e-14)

    def test_equal_frame_weighting_ignores_point_count(self):
        def fit(coefficients, sample_count):
            plane = plane_from_regression(coefficients)
            return PlaneFit(True, np.asarray(coefficients), plane, sample_count,
                            sample_count, 1.0, 0.0001)

        fits = [
            fit([0.0, 0.0, 0.04], 100000),
            fit([0.0, 0.0, 0.08], 100),
            fit([0.0, 0.0, 0.081], 100),
        ]
        coefficients, _ = aggregate_equal_frame(fits)
        self.assertAlmostEqual(coefficients[2], 0.08)

    def test_bootstrap_is_deterministic(self):
        fits = []
        for height in np.linspace(0.048, 0.052, 12):
            coefficients = np.array([-0.08, -0.01, height])
            fits.append(PlaneFit(True, coefficients, plane_from_regression(coefficients),
                                 1000, 900, 0.9, 0.0005))
        positive = np.array([[0.0, 0.08, 0.7], [0.1, 0.09, 0.8]])
        first = bootstrap_planes(fits, positive, np.array([0.0, 0.7]), seed=91, iterations=200)
        second = bootstrap_planes(fits, positive, np.array([0.0, 0.7]), seed=91, iterations=200)
        self.assertEqual(first, second)
        self.assertLess(first["height_at_reference_m"]["p025"], first["height_at_reference_m"]["p975"])

    def test_sample_limit_sensitivity_is_stable(self):
        expected = np.array([-0.07, 0.012, 0.065])
        points = self.synthetic_points(expected, count=12000, seed=22)
        rng = np.random.default_rng(23)
        outliers = rng.uniform([-0.35, -0.1, 0.35], [0.35, 0.25, 1.05], (3000, 3))
        points = np.concatenate((points, outliers))
        large = fit_frame_plane(points, seed=7, sample_limit=4000, iterations=120,
                                threshold_m=0.003, minimum_inliers=500)
        small = fit_frame_plane(points, seed=7, sample_limit=2000, iterations=120,
                                threshold_m=0.003, minimum_inliers=300)
        self.assertTrue(large.accepted and small.accepted)
        angle = np.degrees(np.arccos(np.clip(np.dot(large.plane[:3], small.plane[:3]), -1, 1)))
        self.assertLess(angle, 0.05)
        self.assertLess(abs(large.coefficients[2] - small.coefficients[2]), 0.0005)

    def test_rejects_malformed_and_insufficient_input(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            fit_frame_plane(np.zeros((20, 2)), seed=1)
        with self.assertRaisesRegex(ValueError, "Quaternion"):
            exclude_oriented_boxes(
                np.zeros((2, 3)), np.zeros((2, 7)), np.ones((2, 3)), 0.0,
            )
        rejected = fit_frame_plane(np.zeros((20, 3)), seed=1)
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "too_few_points")


if __name__ == "__main__":
    unittest.main()
