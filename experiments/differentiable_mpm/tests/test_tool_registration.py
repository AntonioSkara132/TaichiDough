"""Small known-transform and visibility checks for partial tool registration."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.differentiable_mpm.register_tool_clouds import (
    sample_mesh, visible_model, fit_icp, transform,
)


class ToolRegistrationTests(unittest.TestCase):
    def test_shared_known_transform_from_partial_views(self):
        vertices = np.array([[0, 0, 0], [.10, 0, 0], [0, .08, 0], [.03, .02, .07]]) - [.04, .03, .02]
        triangles = vertices[np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])]
        model = sample_mesh(triangles, 24000, 7)
        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_euler('xyz', [3, -2, 4], degrees=True).as_matrix()
        expected[:3, 3] = [.004, -.003, .006]
        camera = {'fx': 500, 'fy': 500, 'cx': 320, 'cy': 240, 'width': 640, 'height': 480}
        rng = np.random.default_rng(8)
        frames = []
        # Opposite viewpoints expose non-coplanar faces; one planar face alone is insufficient.
        for center in ([-.20, -.1, -.45], [.2, .1, -.45], [.25, -.2, .35], [-.2, .25, .35]):
            center = np.array(center)
            z = -center / np.linalg.norm(center)
            x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x)
            y = np.cross(z, x)
            camera_from_marker = np.eye(4)
            camera_from_marker[:3, :3] = np.stack([x, y, z])
            camera_from_marker[:3, 3] = -camera_from_marker[:3, :3] @ center
            points, _ = visible_model(*model, expected, camera_from_marker, camera)
            selected = points[rng.choice(len(points), 1200, replace=False)]
            selected += rng.normal(0, .0001, selected.shape)
            frames.append({'points': selected, 'camera_from_marker': camera_from_marker})
        fitted, _, information = fit_icp(frames, model, camera, iterations=35)
        translation_error = np.linalg.norm(fitted[:3, 3] - expected[:3, 3])
        angle_error = Rotation.from_matrix(fitted[:3, :3] @ expected[:3, :3].T).magnitude()
        self.assertLess(translation_error, .0005)
        self.assertLess(np.rad2deg(angle_error), .5)
        self.assertTrue(np.isfinite(information).all())

    def test_transform_composition_uses_marker_local_correction(self):
        marker = np.eye(4)
        marker[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        marker[:3, 3] = [.2, .1, .6]
        correction = np.eye(4); correction[:3, 3] = [.01, 0, 0]
        actual = transform([[0, 0, 0]], marker @ correction)[0]
        np.testing.assert_allclose(actual, [.2, .11, .6], atol=1e-12)
        self.assertFalse(np.allclose(actual, marker[:3, 3] + correction[:3, 3]))


if __name__ == '__main__':
    unittest.main()
