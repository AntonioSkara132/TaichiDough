"""Fixed-pair line search, support checks and recovery for partial-view ICP."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.differentiable_mpm import partial_view_icp as icp


CAMERA = {'fx': 500, 'fy': 500, 'cx': 320, 'cy': 240, 'width': 640, 'height': 480}


def plane_fixture():
    axis = np.linspace(-.03, .03, 7)
    xy = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)
    points = np.column_stack((xy, np.zeros(len(xy))))
    normals = np.tile([0., 0., 1.], (len(points), 1))
    camera_from_marker = np.eye(4)
    camera_from_marker[2, 3] = .5
    frame = {'points': points + [0, 0, .004], 'camera_from_marker': camera_from_marker}
    config = icp.IcpConfig(iterations=1, trim_quantile=1, min_frame_correspondences=10,
                           min_total_correspondences=10, min_visible=10,
                           max_step_translation_m=.1, backtracking=True)
    return [frame], (points, normals), config


class BacktrackingTests(unittest.TestCase):
    def test_overshoot_is_reduced_and_fixed_pair_objective_decreases(self):
        frames, model, config = plane_fixture()
        solve = np.linalg.lstsq

        def overshoot(*args, **kwargs):
            step, residual, rank, singular = solve(*args, **kwargs)
            return 3 * step, residual, rank, singular

        with patch.object(icp.np.linalg, 'lstsq', side_effect=overshoot):
            result = icp.fit_icp_detailed(frames, model, CAMERA, config=config)
        record = result.history[0]
        self.assertEqual(record['backtracks'], 1)
        self.assertEqual(record['accepted_fraction'], .5)
        self.assertEqual(record['line_search_trials'][0]['rejection'], 'objective_increase')
        self.assertLess(record['objective_after'], record['objective_before'])
        self.assertAlmostEqual(result.transform[2, 3], .006, places=7)
        self.assertGreaterEqual(record['minimum_support_ratio'], config.min_support_ratio)

    def test_ascent_direction_is_rejected_without_changing_transform(self):
        frames, model, config = plane_fixture()
        solve = np.linalg.lstsq

        def ascent(*args, **kwargs):
            step, residual, rank, singular = solve(*args, **kwargs)
            return -step, residual, rank, singular

        with patch.object(icp.np.linalg, 'lstsq', side_effect=ascent):
            result = icp.fit_icp_detailed(frames, model, CAMERA, config=config)
        self.assertEqual(result.status, 'line_search_failed')
        np.testing.assert_array_equal(result.transform, np.eye(4))
        self.assertEqual(len(result.history[0]['line_search_trials']), config.max_backtracks + 1)
        self.assertTrue(all(row['rejection'] == 'objective_increase'
                            for row in result.history[0]['line_search_trials']))

    def test_improving_candidate_cannot_drop_correspondence_support(self):
        frames, model, config = plane_fixture()
        original = icp._trimmed_correspondences

        def drop_support(frame, samples, correction, camera, limit, options):
            selected, distances = original(frame, samples, correction, camera, limit, options)
            if correction[2, 3] > 0 and selected is not None:
                selected = tuple(value[:5] for value in selected)
            return selected, distances

        with patch.object(icp, '_trimmed_correspondences', side_effect=drop_support):
            result = icp.fit_icp_detailed(frames, model, CAMERA, config=config)
        self.assertEqual(result.status, 'line_search_failed')
        np.testing.assert_array_equal(result.transform, np.eye(4))
        trials = result.history[0]['line_search_trials']
        self.assertTrue(all(row['rejection'] == 'support_loss' for row in trials))
        self.assertTrue(all(row['objective'] < result.history[0]['objective_before'] for row in trials))

    def test_visibility_loss_rejects_candidate(self):
        frames, model, config = plane_fixture()
        original = icp._trimmed_correspondences

        def no_visibility(frame, samples, correction, camera, limit, options):
            if correction[2, 3] > 0:
                raise ValueError('No visible samples')
            return original(frame, samples, correction, camera, limit, options)

        with patch.object(icp, '_trimmed_correspondences', side_effect=no_visibility):
            result = icp.fit_icp_detailed(frames, model, CAMERA, config=config)
        self.assertEqual(result.status, 'line_search_failed')
        np.testing.assert_array_equal(result.transform, np.eye(4))

    def test_bound_retries_smaller_step_and_default_keeps_original_stop(self):
        frames, model, config = plane_fixture()
        bounded = replace(config, max_total_translation_m=.003)
        candidate = icp.fit_icp_detailed(frames, model, CAMERA, config=bounded)
        self.assertAlmostEqual(candidate.transform[2, 3], .002, places=7)
        self.assertEqual(candidate.history[0]['line_search_trials'][0]['rejection'], 'motion_bound')
        baseline = icp.fit_icp_detailed(frames, model, CAMERA, config=replace(bounded, backtracking=False))
        self.assertEqual(baseline.status, 'bounded_stop')
        np.testing.assert_array_equal(baseline.transform, np.eye(4))
        self.assertEqual(baseline.history, [{'iteration': 0, 'bounded_stop': True}])

    def test_nonfinite_update_is_rejected(self):
        frames, model, config = plane_fixture()
        with patch.object(icp.np.linalg, 'lstsq', return_value=(np.full(6, np.nan), None, None, None)):
            result = icp.fit_icp_detailed(frames, model, CAMERA, config=config)
        self.assertEqual(result.status, 'nonfinite_update')
        np.testing.assert_array_equal(result.transform, np.eye(4))

    def test_sparse_observation_leaves_initial_pose_unchanged(self):
        frames, model, config = plane_fixture()
        frames[0]['points'] = frames[0]['points'][:5]
        initial = np.eye(4)
        initial[0, 3] = .001
        result = icp.fit_icp_detailed(frames, model, CAMERA, initial=initial, config=config)
        self.assertEqual(result.status, 'insufficient_correspondences')
        np.testing.assert_array_equal(result.transform, initial)

    def test_invalid_search_settings_rejected(self):
        frames, model, config = plane_fixture()
        for changes in ({'max_backtracks': -1}, {'max_backtracks': True}, {'max_backtracks': 1.5},
                        {'backtrack_factor': 0}, {'backtrack_factor': 1}, {'backtrack_factor': np.nan},
                        {'min_support_ratio': 0}, {'min_support_ratio': 1.1},
                        {'tangential_weight': -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                icp.fit_icp_detailed(frames, model, CAMERA, config=replace(config, **changes))

    def test_known_transform_recovery_with_partial_views(self):
        vertices = np.array([[0, 0, 0], [.10, 0, 0], [0, .08, 0], [.03, .02, .07]]) - [.04, .03, .02]
        triangles = vertices[np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])]
        model = icp.sample_mesh(triangles, 18000, 7)
        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_euler('xyz', [3, -2, 4], degrees=True).as_matrix()
        expected[:3, 3] = [.004, -.003, .006]
        rng = np.random.default_rng(8)
        frames = []
        for center in ([-.20, -.1, -.45], [.2, .1, -.45], [.25, -.2, .35], [-.2, .25, .35]):
            center = np.array(center)
            z = -center / np.linalg.norm(center)
            x = np.cross([0, 1, 0], z)
            x /= np.linalg.norm(x)
            camera_from_marker = np.eye(4)
            camera_from_marker[:3, :3] = np.stack([x, np.cross(z, x), z])
            camera_from_marker[:3, 3] = -camera_from_marker[:3, :3] @ center
            points, _ = icp.visible_model(*model, expected, camera_from_marker, CAMERA)
            points = points[rng.choice(len(points), min(1000, len(points)), replace=False)]
            frames.append({'points': points + rng.normal(0, .0001, points.shape),
                           'camera_from_marker': camera_from_marker})
        result = icp.fit_icp_detailed(frames, model, CAMERA, config=icp.IcpConfig(backtracking=True))
        self.assertLess(np.linalg.norm(result.transform[:3, 3] - expected[:3, 3]), .0005)
        angle = Rotation.from_matrix(result.transform[:3, :3] @ expected[:3, :3].T).magnitude()
        self.assertLess(np.rad2deg(angle), .5)
        accepted = [row for row in result.history if 'objective_after' in row]
        self.assertGreater(len(accepted), 0)
        for row in accepted:
            self.assertLessEqual(row['objective_after'], row['objective_before'] + 1e-18)
            self.assertGreaterEqual(row['minimum_support_ratio'], .9)
        self.assertTrue(np.isfinite(result.scaled_plane_hessian).all())


if __name__ == '__main__':
    unittest.main()
