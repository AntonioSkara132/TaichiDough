import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from render_replay_depth_tools import mesh_scene_vertices, project_scene_vertices, rasterize_triangles
from taichi_viscoelastic_mpm_scene import quaternion_to_matrix


class ReplayDepthToolsRendererTests(unittest.TestCase):
    def setUp(self):
        self.camera = {
            "width": 20,
            "height": 20,
            "fx": 10.0,
            "fy": 10.0,
            "cx": 10.0,
            "cy": 10.0,
            "zNear": 0.1,
            "zFar": 10.0,
            "scene_from_camera": np.eye(4).tolist(),
        }

    def test_scene_mesh_uses_xyzw_link_pose(self):
        vertices = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        half_turn_z = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        pose = np.array([2.0, 3.0, 4.0, *half_turn_z], dtype=np.float64)
        expected = vertices @ quaternion_to_matrix(half_turn_z).T + pose[:3]
        np.testing.assert_allclose(mesh_scene_vertices(vertices, pose), expected)

    def test_project_scene_vertices_uses_calibrated_intrinsics(self):
        projected, depths = project_scene_vertices(np.array([[1.0, 2.0, 2.0]]), self.camera)
        np.testing.assert_allclose(projected, [[15.0, 20.0, 2.0]])
        np.testing.assert_allclose(depths, [2.0])

    def test_nearest_triangle_replaces_existing_dough_depth(self):
        depth = np.full((20, 20), 3.0, dtype=np.float64)
        owner = np.full((20, 20), -1, dtype=np.int8)
        near_triangle = np.array([[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0]])
        rasterize_triangles(depth, owner, near_triangle, self.camera, 0)
        self.assertEqual(owner[10, 10], 0)
        self.assertAlmostEqual(depth[10, 10], 1.0)
        far_triangle = near_triangle.copy()
        far_triangle[:, 2] = 2.0
        rasterize_triangles(depth, owner, far_triangle, self.camera, 1)
        self.assertEqual(owner[10, 10], 0)
        self.assertAlmostEqual(depth[10, 10], 1.0)

    def test_behind_camera_triangle_is_rejected(self):
        depth = np.full((20, 20), 3.0, dtype=np.float64)
        owner = np.full((20, 20), -1, dtype=np.int8)
        triangle = np.array([[-0.5, -0.5, -1.0], [0.5, -0.5, -1.0], [0.0, 0.5, -1.0]])
        rasterize_triangles(depth, owner, triangle, self.camera, 0)
        np.testing.assert_array_equal(owner, -1)
        np.testing.assert_array_equal(depth, 3.0)


if __name__ == "__main__":
    unittest.main()
