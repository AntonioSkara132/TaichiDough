"""Focused host checks for table-aligned calibration geometry."""
import copy
import unittest

import numpy as np

from experiments.differentiable_mpm.table_alignment import (
    compose_table_calibration,
    table_alignment_transform,
)


class TableAlignmentTests(unittest.TestCase):
    plane = np.array([0.08762950795145376, 0.9961169753375586,
                      0.008487684050421995, -0.07360768236928783])

    def test_plane_mapping_preserves_signed_distance_and_rigidity(self):
        transform = table_alignment_transform(self.plane, translation_xz=(0.1, -0.2))
        rng = np.random.default_rng(18)
        points = rng.uniform([-0.2, 0.0, 0.3], [0.6, 0.2, 1.0], (100, 3))
        moved = points @ transform[:3, :3].T + transform[:3, 3]
        expected = points @ self.plane[:3] + self.plane[3]
        np.testing.assert_allclose(moved[:, 1], expected, atol=1e-14, rtol=0)
        np.testing.assert_allclose(np.linalg.inv(transform).T @ self.plane,
                                   [0.0, 1.0, 0.0, 0.0], atol=1e-14, rtol=0)
        np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3],
                                   np.eye(3), atol=1e-14, rtol=0)
        self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1.0)
        horizontal = table_alignment_transform([0, 2, 0, -0.1])
        np.testing.assert_allclose(horizontal[:3, :3], np.eye(3))
        self.assertAlmostEqual(horizontal[1, 3], -0.05)

    def test_calibration_camera_and_tool_pose_invariance(self):
        source = np.eye(4)
        source[:3, 3] = [0.1, 0.02, -0.2]
        camera = np.eye(4)
        camera[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        camera[:3, 3] = [0.3, 0.6, 0.8]
        original = {
            "schema": "taichidough/scene-calibration/v2", "name": "episode18",
            "source_frame": "mocap", "scene_frame": "mocap",
            "scene_from_source": source.tolist(), "scene_from_camera": camera.tolist(),
            "floor_plane_scene": [0, 1, 0, 0],
            "camera": {"fx": 615.0, "fy": 616.0, "cx": 327.0, "cy": 240.0,
                       "width": 640, "height": 480, "zNear": 0.01, "zFar": 5.0},
        }
        unchanged = copy.deepcopy(original)
        derived = compose_table_calibration(original, self.plane, translation_xz=(-0.1, 0.2))
        transform = table_alignment_transform(self.plane, translation_xz=(-0.1, 0.2))
        new_source = np.asarray(derived["scene_from_source"])
        new_camera = np.asarray(derived["scene_from_camera"])
        np.testing.assert_allclose(np.linalg.inv(new_camera) @ new_source,
                                   np.linalg.inv(camera) @ source, atol=1e-14, rtol=0)
        marker = np.eye(4)
        marker[:3, 3] = [0.2, 0.08, 0.7]
        marker_from_mesh = np.eye(4)
        marker_from_mesh[:3, 3] = [0.04, -0.02, 0.01]
        old_tool = source @ marker @ marker_from_mesh
        new_tool = new_source @ marker @ marker_from_mesh
        np.testing.assert_allclose(new_tool, transform @ old_tool, atol=1e-14, rtol=0)
        self.assertEqual(original, unchanged)
        self.assertEqual(derived["camera"], original["camera"])
        self.assertEqual(derived["scene_frame"], "table-aligned")
        self.assertEqual(derived["floor_plane_scene"], [0.0, 1.0, 0.0, 0.0])
        self.assertFalse(derived["provenance"]["table_alignment"]["source_tensors_transformed"])

    def test_rejects_invalid_plane_and_nonrigid_calibration(self):
        for plane in ([0, 0, 0, 0], [0, -1, 0, 0], [np.nan, 1, 0, 0], [0, 1, 0]):
            with self.subTest(plane=plane), self.assertRaises(ValueError):
                table_alignment_transform(plane)
        with self.assertRaisesRegex(ValueError, "translation_xz"):
            table_alignment_transform(self.plane, translation_xz=(0.0, np.inf))
        scaled = np.eye(4)
        scaled[0, 0] = 2.0
        document = {"schema": "taichidough/scene-calibration/v2", "source_frame": "mocap",
                    "scene_frame": "mocap", "camera": {}, "scene_from_source": scaled.tolist(),
                    "scene_from_camera": np.eye(4).tolist()}
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            compose_table_calibration(document, self.plane)


if __name__ == "__main__":
    unittest.main()
