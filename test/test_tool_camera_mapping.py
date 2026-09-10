"""Validate STL transforms and signed-distance-field helpers without ROS or a viewer."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "dough_scene", PACKAGE / "scripts/taichi_viscoelastic_mpm_scene.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load the Taichi scene module")
scene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scene)


class ToolMeshCollisionTest(unittest.TestCase):
    def test_mesh_sdf_is_signed_and_has_outward_gradients(self):
        corners = np.array([
            [-.05, -.05, -.05], [.05, -.05, -.05], [.05, .05, -.05], [-.05, .05, -.05],
            [-.05, -.05, .05], [.05, -.05, .05], [.05, .05, .05], [-.05, .05, .05],
        ], dtype=np.float32)
        faces = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ])
        distance, gradients, lower, spacing = scene.build_mesh_sdf(corners[faces].reshape(-1, 3), 40, .01)

        def sample(point):
            index = np.rint((np.asarray(point) - lower) / spacing).astype(int)
            return distance[tuple(index)], gradients[tuple(index)]

        inside, _ = sample([0, 0, 0])
        outside, gradient = sample([.058, 0, 0])
        self.assertLess(inside, 0.0)
        self.assertGreater(outside, 0.0)
        self.assertGreater(gradient[0], 0.8)

    def test_mesh_visual_transform_uses_link_then_pose(self):
        vertices = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float32)
        pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        origin = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        actual = scene.transformed_tool_mesh(vertices, pose, origin, np.eye(3, dtype=np.float32))
        np.testing.assert_allclose(actual, vertices + pose[:3] + origin)

    def test_marker_to_mesh_transform_targets_the_sdf_link_frame(self):
        raw_vertex = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        visual_origin = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        local_sdf_vertex = scene.mesh_in_tool_frame(raw_vertex, visual_origin, np.eye(3))
        source_from_marker = np.eye(4)
        source_from_marker[:3, 3] = [1.0, 2.0, 3.0]
        marker_from_mesh = np.eye(4)
        marker_from_mesh[:3, 3] = [0.4, -0.1, 0.2]
        scene_from_mesh = source_from_marker @ marker_from_mesh
        local_homogeneous = np.concatenate([local_sdf_vertex[0], [1.0]])
        np.testing.assert_allclose(
            scene_from_mesh @ local_homogeneous,
            [1.5, 2.1, 3.5, 1.0],
        )

    def test_bundled_meshes_are_binary_and_transformable(self):
        for filename, origin, rpy in (
            ("ur_spathla.stl", scene.UR_TOOL_VISUAL_ORIGIN, scene.UR_TOOL_VISUAL_RPY),
            ("gen3_spathla.stl", scene.KINOVA_TOOL_VISUAL_ORIGIN, scene.KINOVA_TOOL_VISUAL_RPY),
        ):
            with self.subTest(mesh=filename):
                vertices, _ = scene.load_binary_stl(PACKAGE / "meshes" / filename, 0.001)
                local = scene.mesh_in_tool_frame(vertices, origin, scene.rpy_to_matrix(rpy))
                self.assertTrue(np.isfinite(local).all())
                self.assertEqual(local.shape, vertices.shape)


if __name__ == "__main__":
    unittest.main()
