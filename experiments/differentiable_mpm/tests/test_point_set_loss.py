"""Host point-loss checks plus separate CPU-f64 renderer checks."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import taichi as ti

from experiments.differentiable_mpm.loss import LossConfig
from experiments.differentiable_mpm.loss_options import (
    PAPER_LOSS_VERSIONS, PaperLossConfig, is_paper_loss, loss_temporal_reduction,
    make_observation_loss, parse_loss_config,
)
from experiments.differentiable_mpm.point_set_loss import (
    PointSetLoss, PointSetObservation, assignment_memory_estimate,
    assignment_value_and_grad, chamfer_value_and_grad, sampled_indices,
    soft_iou_value_and_grad, voxel_centroids,
)
from experiments.differentiable_mpm.renderer import Camera
from experiments.differentiable_mpm.state import InvalidStateError


CAMERA = Camera(24, 20, 30.0, 30.0, 12.0, 10.0, np.eye(4))


def finite_difference(objective, positions, observation, h=1e-6):
    result = np.zeros_like(positions, dtype=np.float64)
    for index in np.ndindex(positions.shape):
        plus, minus = positions.copy(), positions.copy()
        plus[index] += h
        minus[index] -= h
        result[index] = (objective.value_and_grad_positions(plus, observation, False).value
                         - objective.value_and_grad_positions(minus, observation, False).value) / (2 * h)
    return result


class PaperLossConfigTests(unittest.TestCase):
    def test_default_and_old_version_are_unchanged(self):
        self.assertEqual(parse_loss_config({}), LossConfig())
        old = LossConfig(version="partial-visible-splats-v1", depth_weight=0.2)
        restored = parse_loss_config(old.as_dict())
        self.assertEqual(restored, old)
        self.assertEqual(restored.fingerprint(CAMERA), old.fingerprint(CAMERA))
        self.assertFalse(is_paper_loss(restored))
        self.assertEqual(loss_temporal_reduction(restored), "mean")

    def test_paper_roundtrip_and_temporal_reductions(self):
        for version in PAPER_LOSS_VERSIONS:
            with self.subTest(version=version):
                config = parse_loss_config({"version": version})
                self.assertTrue(is_paper_loss(config))
                self.assertEqual(parse_loss_config(config.as_dict()), config)
                self.assertEqual(loss_temporal_reduction(config), "sum" if version.startswith("empm") else "mean")
                self.assertEqual(len(config.fingerprint(CAMERA)), 64)
        self.assertEqual(PaperLossConfig("dpsi-prt-cd-v1").target_source, "external")
        self.assertEqual(PaperLossConfig("empm-offline-v1").tracking_weight, 1)

    def test_unrelated_keys_and_invalid_options_are_rejected(self):
        cases = [
            {"version": "dpsi-pcd-cd-v1", "depth_weight": 1},
            {"version": "dpsi-pcd-cd-v1", "tracking_weight": 0},
            {"version": "dpsi-prt-cd-v1", "target_voxel_size_m": 0.005},
            {"version": "empm-offline-v1", "mask_weight": 1},
            {"version": "dpsi-prt-cd-v1", "target_source": "recorded_cloud"},
            {"version": "dpsi-pcd-cd-v1", "predicted_sample_count": True},
            {"version": "dpsi-pcd-cd-v1", "target_sample_count": 0},
            {"version": "dpsi-pcd-cd-v1", "sampling_seed": -1},
            {"version": "dpsi-pcd-cd-v1", "min_target_points": 1.5},
            {"version": "empm-offline-v1", "tracking_weight": float("nan")},
            {"version": "empm-offline-v1", "geometric_weight": 0, "tracking_weight": 0},
            {"version": "empm-offline-v1", "empty_track_policy": "ignore"},
            {"version": "empm-mask-inspired-v1", "mask_weight": 0},
            {"version": "empm-mask-inspired-v1", "mask_epsilon": 0},
            {"version": "empm-mask-inspired-v1", "footprint_radius": 9},
            {"version": "not-a-loss"},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                parse_loss_config(case)
        with self.assertRaises(ValueError):
            PaperLossConfig("dpsi-pcd-cd-v1", tracking_weight=0.4)

    def test_factory_preserves_old_constructor(self):
        initial = np.array([[0.0, 0.0, 1.0]])
        settings = LossConfig()
        with patch("experiments.differentiable_mpm.loss.ObservationLoss") as old:
            returned = make_observation_loss(CAMERA, settings, initial, "f64")
            old.assert_called_once_with(CAMERA, settings, initial, precision="f64")
            self.assertIs(returned, old.return_value)
        paper = make_observation_loss(CAMERA, PaperLossConfig("dpsi-pcd-cd-v1"), initial, "f64")
        self.assertIsInstance(paper, PointSetLoss)

    def test_changed_options_and_camera_change_fingerprint(self):
        a = PaperLossConfig("empm-offline-v1", tracking_weight=0)
        b = PaperLossConfig("empm-offline-v1", tracking_weight=1)
        self.assertNotEqual(a.fingerprint(CAMERA), b.fingerprint(CAMERA))
        self.assertNotEqual(a.fingerprint(CAMERA), a.fingerprint(CAMERA.scaled(12, 10)))


class PointGeometryTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        self.target = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    def test_hand_computed_chamfer_values(self):
        value, gradient, components, _ = chamfer_value_and_grad(self.positions, self.target)
        self.assertEqual(value, 4)
        self.assertEqual(components["chamfer_target_to_simulation"], 2)
        np.testing.assert_array_equal(gradient, [[-2, 0, 0], [2, 0, 0]])
        squared, squared_gradient, _, _ = chamfer_value_and_grad(self.positions, self.target, squared=True)
        self.assertEqual(squared, 2)
        np.testing.assert_array_equal(squared_gradient, gradient)

    def test_repeated_nearest_matches_accumulate(self):
        predicted = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        target = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        value, gradient, _, _ = chamfer_value_and_grad(predicted, target)
        self.assertEqual(value, 14)
        np.testing.assert_array_equal(gradient, [[-4, 0, 0], [1, 0, 0]])

    def test_rectangular_assignment_and_unmatched_particles(self):
        predicted = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        value, gradient, components, diagnostics = assignment_value_and_grad(
            predicted, np.array([[1., 0, 0], [3., 0, 0]]), max_pairs=100, max_bytes=10000)
        self.assertEqual(value, 2)
        self.assertEqual(components["target_to_simulation_assignment"], value)
        self.assertEqual(diagnostics["unmatched_simulated_points"], 1)
        np.testing.assert_array_equal(gradient, [[-1, 0, 0], [1, 0, 0], [0, 0, 0]])

    def test_assignment_is_not_independent_nearest_neighbors(self):
        predicted = np.array([[0., 0, 0], [10., 0, 0]])
        target = np.array([[1., 0, 0], [2., 0, 0]])
        value, _, _, _ = assignment_value_and_grad(predicted, target, max_pairs=100, max_bytes=10000)
        self.assertEqual(value, 9)
        self.assertNotEqual(value, 3)

    def test_assignment_limits_refuse_before_allocating(self):
        module = "experiments.differentiable_mpm.point_set_loss.cdist"
        for limits in ({"max_pairs": 3, "max_bytes": 10000},
                       {"max_pairs": 100, "max_bytes": assignment_memory_estimate(2, 2) - 1}):
            with self.subTest(limits=limits), patch(module) as allocator, self.assertRaisesRegex(ValueError, "allocation refused"):
                assignment_value_and_grad(self.positions, self.target, **limits)
            allocator.assert_not_called()
        with patch(module) as allocator, self.assertRaisesRegex(ValueError, "n_target <= n_simulated"):
            assignment_value_and_grad(self.positions[:1], self.target, max_pairs=100, max_bytes=10000)
        allocator.assert_not_called()
        self.assertGreater(assignment_memory_estimate(24000, 24000), 4_600_000_000)

    def test_unsquared_zero_subgradient_and_ties_are_repeatable(self):
        duplicated = np.zeros((3, 3))
        for metric in ("cd", "emd"):
            if metric == "cd":
                result = chamfer_value_and_grad(duplicated, duplicated)
            else:
                result = assignment_value_and_grad(duplicated, duplicated, max_pairs=100, max_bytes=10000)
            self.assertEqual(result[0], 0)
            np.testing.assert_array_equal(result[1], 0)
        predicted = np.array([[-1., 0, 0], [1., 0, 0]])
        target = np.zeros((1, 3))
        first = chamfer_value_and_grad(predicted, target)
        second = chamfer_value_and_grad(predicted, target)
        self.assertEqual(first[0], 3)
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[3], second[3])

    def test_objective_gradients_for_all_point_versions(self):
        positions = np.array([[0.1, 0.02, 0.5], [0.9, 0.15, 0.7], [1.5, -0.2, 0.8]])
        target = np.array([[0.25, 0.05, 0.6], [0.8, 0.1, 0.6]])
        for version in PAPER_LOSS_VERSIONS[:-1]:
            values = {"version": version}
            if version == "empm-offline-v1":
                values["tracking_weight"] = 0
            objective = PointSetLoss(CAMERA, parse_loss_config(values), positions, "f64")
            obs = PointSetObservation(1, 2, target, target_representation=(
                "inferred_volume" if "-prt-" in version else "partial_observed"))
            result = objective.value_and_grad_positions(positions, obs)
            for h in (1e-4, 1e-5, 1e-6):
                with self.subTest(version=version, h=h):
                    numeric = finite_difference(objective, positions, obs, h)
                    np.testing.assert_allclose(result.gradient, numeric, rtol=2e-6, atol=2e-7)
            no_grad = objective.value_and_grad_positions(positions, obs, False)
            self.assertIsNone(no_grad.gradient)
            self.assertEqual(no_grad.value, result.value)
            self.assertAlmostEqual(result.value, sum(result.components[k] * result.diagnostics["component_weights"][k]
                                                      for k in result.components))

    def test_sampling_is_explicit_repeatable_and_particle_fixed(self):
        positions = np.column_stack((np.arange(20), np.zeros((20, 2)))).astype(float)
        config = PaperLossConfig("dpsi-pcd-cd-v1", predicted_sample_count=5,
                                 target_sample_count=4, sampling_seed=13)
        first, second = PointSetLoss(CAMERA, config, positions, "f64"), PointSetLoss(CAMERA, config, positions, "f64")
        np.testing.assert_array_equal(first.predicted_ids, second.predicted_ids)
        obs = PointSetObservation(1, 2, positions + [0.2, 0.1, 0.0])
        result = first.value_and_grad_positions(positions, obs)
        changed = first.value_and_grad_positions(positions + 0.01, obs)
        self.assertEqual(result.diagnostics["prediction_indices_sha256"], changed.diagnostics["prediction_indices_sha256"])
        self.assertEqual(result.diagnostics["target_indices_sha256"], changed.diagnostics["target_indices_sha256"])
        self.assertEqual(result.diagnostics["prediction_used_points"], 5)
        self.assertEqual(result.diagnostics["target_used_points"], 4)
        self.assertTrue(result.diagnostics["sampling_adaptation"])
        inactive = np.setdiff1d(np.arange(len(positions)), first.predicted_ids)
        np.testing.assert_array_equal(result.gradient[inactive], 0)
        np.testing.assert_array_equal(sampled_indices(2, 5, 0), [0, 1])

    def test_voxel_downsampling_is_deterministic_and_metric(self):
        points = np.array([[0.001, 0.001, 0.001], [0.003, 0.002, 0.001],
                           [0.011, 0.001, 0.001]])
        expected = np.array([[0.002, 0.0015, 0.001], [0.011, 0.001, 0.001]])
        np.testing.assert_allclose(voxel_centroids(points, 0.005), expected)
        np.testing.assert_allclose(voxel_centroids(points[::-1], 0.005), expected)
        np.testing.assert_array_equal(voxel_centroids(points, 0), points)

    def test_empty_invalid_points_representation_and_state_are_rejected(self):
        objective = PointSetLoss(CAMERA, PaperLossConfig("dpsi-pcd-cd-v1"), self.positions, "f64")
        for target in (np.empty((0, 3)), [[np.nan, 0, 0]], [[1, 2]]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                objective.value_and_grad_positions(self.positions, PointSetObservation(1, 2, target))
        with self.assertRaises(ValueError):
            objective.value_and_grad_positions(self.positions, PointSetObservation(1, 2, self.target,
                                               target_representation="inferred_volume"))
        volume = PointSetLoss(CAMERA, PaperLossConfig("dpsi-prt-cd-v1"), self.positions, "f64")
        with self.assertRaises(ValueError):
            volume.value_and_grad_positions(self.positions, PointSetObservation(1, 2, self.target))
        with self.assertRaises(InvalidStateError):
            objective.value_and_grad_positions(self.positions * np.nan, PointSetObservation(1, 2, self.target))

    def test_large_chamfer_does_not_use_dense_assignment_cost(self):
        predicted = np.column_stack((np.linspace(0, 1, 30000), np.zeros((30000, 2))))
        target = np.array([[0.1, 0.01, 0], [0.9, -0.01, 0]])
        with patch("experiments.differentiable_mpm.point_set_loss.cdist", side_effect=AssertionError("dense allocation")):
            value, gradient, _, _ = chamfer_value_and_grad(predicted, target)
        self.assertTrue(np.isfinite(value))
        self.assertEqual(gradient.shape, predicted.shape)


class PointTrackingTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.array([[0.1, 0.2, 0.8], [0.5, 0.1, 1.0], [-0.2, 0.4, 0.9]])
        self.obs = PointSetObservation(2, 3, self.positions + 0.02,
                                      track_particle_ids=np.array([0, 2]),
                                      track_positions_scene=np.array([[0.2, 0.1, 0.7], [np.nan, np.nan, np.nan]]),
                                      track_valid=np.array([True, False]))

    def test_tracking_gradients_masks_and_unweighted_addition(self):
        config = PaperLossConfig("empm-offline-v1", geometric_weight=0.3, tracking_weight=2.0)
        objective = PointSetLoss(CAMERA, config, self.positions, "f64")
        result = objective.value_and_grad_positions(self.positions, self.obs)
        np.testing.assert_allclose(result.gradient, finite_difference(objective, self.positions, self.obs), rtol=1e-7, atol=1e-9)
        self.assertAlmostEqual(result.components["tracked_squared_distance"], 0.03)
        self.assertEqual(result.diagnostics["valid_tracked_points"], 1)
        self.assertFalse(result.diagnostics["geometry_only_ablation"])
        self.assertAlmostEqual(result.value, 0.3 * (result.components["chamfer_target_to_simulation"]
                                                   + result.components["chamfer_simulation_to_target"])
                               + 2 * result.components["tracked_squared_distance"])

    def test_tracking_only_and_particle_ids_not_reassigned(self):
        config = PaperLossConfig("empm-offline-v1", geometric_weight=0, tracking_weight=1,
                                 predicted_sample_count=1)
        objective = PointSetLoss(CAMERA, config, self.positions, "f64")
        self.obs.points_scene = np.empty((0, 3))
        result = objective.value_and_grad_positions(self.positions, self.obs)
        expected = np.zeros_like(self.positions)
        expected[0] = [-0.2, 0.2, 0.2]
        np.testing.assert_allclose(result.gradient, expected)
        moved = self.positions.copy()
        moved[0] += 2
        changed = objective.value_and_grad_positions(moved, self.obs)
        np.testing.assert_allclose(changed.gradient[0], expected[0] + 4)
        np.testing.assert_array_equal(changed.gradient[1:], 0)

    def test_missing_tracks_and_bad_ids_fail(self):
        objective = PointSetLoss(CAMERA, PaperLossConfig("empm-offline-v1"), self.positions, "f64")
        with self.assertRaisesRegex(ValueError, "track_particle_ids"):
            objective.value_and_grad_positions(self.positions, PointSetObservation(1, 2, self.positions))
        for ids in (np.array([0, 0]), np.array([-1, 2]), np.array([0, 3]), np.array([0., 2.])):
            self.obs.track_particle_ids = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                objective.value_and_grad_positions(self.positions, self.obs)

    def test_zero_tracking_weight_is_explicit_ablation(self):
        objective = PointSetLoss(CAMERA, PaperLossConfig("empm-offline-v1", tracking_weight=0), self.positions, "f64")
        result = objective.value_and_grad_positions(self.positions, PointSetObservation(1, 2, self.positions + 0.1))
        self.assertTrue(result.diagnostics["geometry_only_ablation"])
        self.assertNotIn("tracked_squared_distance", result.components)

    def test_empty_tracks_error_or_explicit_skip(self):
        self.obs.track_valid[:] = False
        strict = PointSetLoss(CAMERA, PaperLossConfig("empm-offline-v1"), self.positions, "f64")
        with self.assertRaisesRegex(ValueError, "no valid tracks"):
            strict.value_and_grad_positions(self.positions, self.obs)
        skip = PointSetLoss(CAMERA, PaperLossConfig("empm-offline-v1", empty_track_policy="skip"), self.positions, "f64")
        result = skip.value_and_grad_positions(self.positions, self.obs)
        self.assertEqual(result.components["tracked_squared_distance"], 0)
        self.assertGreater(result.value, 0)
        self.assertTrue(result.diagnostics["tracking_skipped"])


class MaskFormulaTests(unittest.TestCase):
    def test_hand_computed_value_and_gradient(self):
        coverage = np.array([[0.2, 0.8], [0.4, 0.9]])
        foreground = np.array([[1, 1], [0, 0]])
        known = np.array([[1, 1], [1, 0]])
        value, gradient = soft_iou_value_and_grad(coverage, foreground, known)
        self.assertAlmostEqual(value, 1 - (1 + 1e-8) / (2.4 + 1e-8))
        self.assertEqual(gradient[1, 1], 0)
        for index in np.ndindex(coverage.shape):
            plus, minus = coverage.copy(), coverage.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            numeric = (soft_iou_value_and_grad(plus, foreground, known)[0]
                       - soft_iou_value_and_grad(minus, foreground, known)[0]) / 2e-6
            self.assertAlmostEqual(gradient[index], numeric, delta=1e-9)

    def test_unknown_pixels_are_excluded_and_empty_support_rejected(self):
        coverage = np.array([[0.2, 0.8], [0.4, 0.9]])
        foreground = np.array([[1, 1], [0, 0]])
        known = np.array([[1, 1], [1, 0]])
        first = soft_iou_value_and_grad(coverage, foreground, known)
        coverage[1, 1], foreground[1, 1] = 0.1, 1
        second = soft_iou_value_and_grad(coverage, foreground, known)
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        with self.assertRaises(ValueError):
            soft_iou_value_and_grad(coverage, foreground, np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            soft_iou_value_and_grad(coverage, foreground + 0.1, known)


class ObservationPreflightTests(unittest.TestCase):
    def test_sampling_diagnostics_match_evaluation_without_mutating_targets(self):
        positions = np.array([[0., 0, 1], [0.01, 0, 1], [0.02, 0, 1]])
        points = np.array([[0.001, 0, 1], [0.002, 0, 1], [0.006, 0, 1],
                           [0.012, 0, 1], [0.017, 0, 1]])
        original = points.copy()
        config = PaperLossConfig("dpsi-pcd-emd-v1", target_voxel_size_m=0.005,
                                 target_sample_count=2, predicted_sample_count=2, sampling_seed=7)
        objective = PointSetLoss(CAMERA, config, positions, "f64")
        obs = PointSetObservation(2, 3, points, metadata={"source_kind": "synthetic_fixture"})
        module = "experiments.differentiable_mpm.point_set_loss"
        with patch(module + ".cdist", side_effect=AssertionError("cost allocation")), \
                patch(module + ".chamfer_value_and_grad", side_effect=AssertionError("loss evaluation")), \
                patch("experiments.differentiable_mpm.renderer.LocalSplatRenderer",
                      side_effect=AssertionError("renderer allocation")):
            prepared = objective.prepare_observation(obs)
        json.dumps(prepared, allow_nan=False)
        self.assertIsNone(objective.renderer)
        self.assertEqual(prepared["target_input_points"], 5)
        self.assertEqual(prepared["target_voxel_points"], 4)
        self.assertEqual(prepared["target_used_points"], 2)
        result = objective.value_and_grad_positions(positions, obs)
        for key, value in prepared.items():
            self.assertEqual(result.diagnostics[key], value)
        again = objective.prepare_observation(obs)
        self.assertEqual(again, prepared)
        np.testing.assert_array_equal(obs.points_scene, original)
        np.testing.assert_array_equal(points, original)
        self.assertEqual(obs.metadata, {"source_kind": "synthetic_fixture"})

    def test_preflight_refuses_assignment_cardinality_and_memory(self):
        positions = np.array([[0., 0, 1], [0.01, 0, 1]])
        for config, target, message in (
            (PaperLossConfig("dpsi-pcd-emd-v1"), np.vstack((positions, [[0.03, 0, 1]])), "n_target"),
            (PaperLossConfig("dpsi-pcd-emd-v1", max_assignment_pairs=3), positions, "allocation refused"),
        ):
            objective = PointSetLoss(CAMERA, config, positions, "f64")
            with self.subTest(message=message), patch("experiments.differentiable_mpm.point_set_loss.cdist") as costs:
                with self.assertRaisesRegex(ValueError, message):
                    objective.prepare_observation(PointSetObservation(1, 2, target))
                costs.assert_not_called()

    def test_preflight_validates_tracks_without_evaluating_loss(self):
        positions = np.array([[0., 0, 1], [0.01, 0, 1]])
        objective = PointSetLoss(CAMERA, PaperLossConfig("empm-offline-v1", geometric_weight=0), positions, "f64")
        obs = PointSetObservation(1, 2, np.empty((0, 3)))
        with self.assertRaisesRegex(ValueError, "track_particle_ids"):
            objective.prepare_observation(obs)
        obs.track_particle_ids = np.array([0, 1])
        obs.track_positions_scene = positions.copy()
        obs.track_valid = np.array([True, False])
        with patch.object(objective, "_tracking", side_effect=AssertionError("tracking loss")):
            diagnostics = objective.prepare_observation(obs)
        self.assertEqual(diagnostics["valid_tracked_points"], 1)
        self.assertEqual(diagnostics["target_used_points"], 0)
        obs.track_valid[:] = False
        with self.assertRaisesRegex(ValueError, "no valid tracks"):
            objective.prepare_observation(obs)
        skip = PointSetLoss(CAMERA, PaperLossConfig("empm-offline-v1", empty_track_policy="skip",
                                                   geometric_weight=0), positions, "f64")
        self.assertTrue(skip.prepare_observation(obs)["tracking_skipped"])

    def test_mask_preflight_checks_support_without_renderer(self):
        positions = np.array([[0., 0, 1]])
        objective = PointSetLoss(CAMERA, PaperLossConfig("empm-mask-inspired-v1", geometric_weight=0), positions, "f64")
        foreground = np.zeros((20, 24), bool)
        foreground[9:12, 11:15] = True
        obs = PointSetObservation(1, 2, np.empty((0, 3)), foreground_mask=foreground,
                                  known_mask=np.ones_like(foreground))
        with patch("experiments.differentiable_mpm.renderer.LocalSplatRenderer",
                   side_effect=AssertionError("renderer initialization")):
            diagnostics = objective.prepare_observation(obs)
        self.assertIsNone(objective.renderer)
        self.assertEqual(diagnostics["known_mask_pixels"], 480)
        self.assertEqual(diagnostics["known_foreground_pixels"], 12)
        obs.known_mask[:] = False
        with self.assertRaisesRegex(ValueError, "no known pixels"):
            objective.prepare_observation(obs)
        obs.known_mask = np.ones((2, 2), bool)
        with self.assertRaisesRegex(ValueError, "known_mask"):
            objective.prepare_observation(obs)


class MaskRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1,
                offline_cache=False, enable_fallback=False)

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def test_actual_mask_only_position_gradient_and_camera_transform(self):
        angle = 0.23
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, [0.03, -0.02, 0.1]
        camera = Camera(24, 20, 30, 30, 12, 10, transform)
        optical = np.array([[-0.013, -0.006, 1.0], [0.027, 0.019, 1.012]])
        initial = (optical - transform[:3, 3]) @ rotation
        objective = PointSetLoss(camera, PaperLossConfig("empm-mask-inspired-v1", geometric_weight=0), initial, "f64")
        foreground = np.zeros((20, 24), bool)
        foreground[9:12, 11:15] = True
        obs = PointSetObservation(1, 2, np.empty((0, 3)), foreground_mask=foreground,
                                  known_mask=np.ones_like(foreground))
        positions = initial + [0.006, -0.003, 0.002]
        result = objective.value_and_grad_positions(positions, obs)
        self.assertEqual(set(result.components), {"segmentation_soft_iou"})
        self.assertGreater(np.linalg.norm(result.gradient), 0.01)
        for h in (1e-5, 1e-6, 1e-7):
            with self.subTest(h=h):
                numeric = finite_difference(objective, positions, obs, h)
                np.testing.assert_allclose(result.gradient, numeric, rtol=3e-5, atol=3e-6)
        again = objective.value_and_grad_positions(positions, obs)
        np.testing.assert_array_equal(result.gradient, again.gradient)
        self.assertEqual(result.value, again.value)

    def test_geometry_and_mask_weights_combine_without_normalization(self):
        initial = np.array([[-0.013, -0.006, 1.0], [0.027, 0.019, 1.012]])
        config = PaperLossConfig("empm-mask-inspired-v1", geometric_weight=2, mask_weight=3)
        objective = PointSetLoss(CAMERA, config, initial, "f64")
        foreground = np.zeros((20, 24), bool)
        foreground[9:12, 11:15] = True
        obs = PointSetObservation(1, 2, initial, foreground_mask=foreground,
                                  known_mask=np.ones_like(foreground))
        positions = initial + [0.003, 0.001, 0.002]
        result = objective.value_and_grad_positions(positions, obs)
        expected = sum(result.components[key] * result.diagnostics["component_weights"][key]
                       for key in result.components)
        self.assertAlmostEqual(result.value, expected)
        np.testing.assert_allclose(result.gradient, finite_difference(objective, positions, obs, 1e-7),
                                   rtol=3e-5, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
