import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reconstruct_voxel_dough_from_deformpath as reconstruction


class FloorReconstructionTests(unittest.TestCase):
    def test_legacy_fill_selection(self):
        self.assertEqual(
            reconstruction.resolve_fill_settings(None, None, None),
            ("thickness", 0.045, None, True),
        )
        self.assertEqual(
            reconstruction.resolve_fill_settings(None, 0.02, 0.4),
            ("limit", None, 0.4, True),
        )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            reconstruction.resolve_fill_settings("floor", 0.02, None)

    def test_horizontal_plane_intersection(self):
        points = np.array([[0.2, 0.5, 0.3], [0.7, 0.8, 0.4]], dtype=np.float32)
        thickness, stats = reconstruction.intersect_points_with_plane(
            points,
            np.array([0.0, -1.0, 0.0]),
            [0.0, 1.0, 0.0, -0.2],
            0.01,
            1.0,
        )
        np.testing.assert_allclose(thickness, [0.3, 0.6], atol=1e-6)
        self.assertEqual(stats["accepted"], 2)
        self.assertAlmostEqual(stats["local_thickness_m"]["mean"], 0.45, places=6)

    def test_tilted_plane_intersection_varies_by_column(self):
        points = np.array([[0.0, 0.6, 0.0], [0.0, 0.6, 0.2]], dtype=np.float32)
        thickness, stats = reconstruction.intersect_points_with_plane(
            points,
            np.array([0.0, -1.0, 0.0]),
            [0.0, 1.0, 0.5, -0.2],
            0.01,
            1.0,
        )
        np.testing.assert_allclose(thickness, [0.4, 0.5], atol=1e-6)
        self.assertEqual(stats["accepted"], 2)

    def test_parallel_wrong_direction_and_limits_are_counted(self):
        points = np.array([[0.0, 0.5, 0.0], [0.0, 0.205, 0.0], [0.0, 1.5, 0.0]])
        parallel, parallel_stats = reconstruction.intersect_points_with_plane(
            points, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0, -0.2], 0.01, 1.0
        )
        self.assertTrue(np.isnan(parallel).all())
        self.assertEqual(parallel_stats["rejection_counts"]["parallel"], 3)

        wrong, wrong_stats = reconstruction.intersect_points_with_plane(
            points[:1], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0, -0.2], 0.01, 1.0
        )
        self.assertTrue(np.isnan(wrong).all())
        self.assertEqual(wrong_stats["rejection_counts"]["wrong_direction"], 1)

        limited, limited_stats = reconstruction.intersect_points_with_plane(
            points, [0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -0.2], 0.01, 1.0
        )
        self.assertTrue(np.isnan(limited[1]))
        self.assertTrue(np.isnan(limited[2]))
        self.assertEqual(limited_stats["rejection_counts"]["below_minimum_thickness"], 1)
        self.assertEqual(limited_stats["rejection_counts"]["above_maximum_thickness"], 1)

    def test_floor_segmentation_uses_signed_distance_and_bounds(self):
        points = np.array(
            [
                [0.5, 0.24, 0.5],
                [0.5, 0.205, 0.5],
                [1.2, 0.4, 0.5],
                [0.4, 0.4, 0.4],
            ],
            dtype=np.float32,
        )
        kept, counts = reconstruction.segment_scene_points(
            points,
            [0.0, 1.0, 0.0, -0.2],
            0.01,
            (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
        )
        np.testing.assert_allclose(kept, points[[0, 3]])
        self.assertEqual(counts["at_or_below_clearance"], 1)
        self.assertEqual(counts["outside_scene_bounds"], 1)

    def test_floor_transform_is_applied_once(self):
        calibration = SimpleNamespace(
            schema="topview_calibration/v2",
            is_metric=True,
            scene_frame="taichi_scene",
            floor_plane_scene=[0.0, 1.0, 0.0, -0.2],
        )
        source = np.array([[0.2, 0.4, 0.3], [0.3, 0.5, 0.4]], dtype=np.float32)
        transformed = source + np.array([0.1, 0.0, 0.0], dtype=np.float32)
        with mock.patch.object(reconstruction, "apply_calibration", return_value=transformed) as apply:
            all_scene, segmented, _, counts = reconstruction.transform_and_segment_floor_points(
                source, calibration, 0.01
            )
        apply.assert_called_once_with(source, calibration)
        np.testing.assert_allclose(all_scene, transformed)
        np.testing.assert_allclose(segmented, transformed)
        self.assertEqual(counts["kept"], 2)

    def test_floor_mode_requires_metric_v2_calibration(self):
        legacy = SimpleNamespace(schema="topview_calibration/v1", is_metric=False, scene_frame="scene", floor_plane_scene=[0, 1, 0, 0])
        with self.assertRaisesRegex(ValueError, "metric v2"):
            reconstruction.require_metric_floor_calibration(legacy)

    def test_sampling_hash_is_deterministic(self):
        voxels = np.array([[0.2, 0.3, 0.4], [0.3, 0.3, 0.4]], dtype=np.float32)
        first = reconstruction.sample_particles(voxels, 10, 0.01, 7)
        second = reconstruction.sample_particles(voxels, 10, 0.01, 7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(reconstruction.particle_array_sha256(first), reconstruction.particle_array_sha256(second))


if __name__ == "__main__":
    unittest.main()
