"""Small analytic tests for triangle queries and sampled distance construction."""
import unittest
from unittest.mock import patch

import numpy as np

from experiments.differentiable_mpm.triangle_sdf import (
    BUILDER_VERSION, TriangleDistanceQuery, build_tool_sdf, builder_identity, grid_bounds,
)


def box_triangles(half=(.03, .02, .01), center=(0., 0., 0.), rotation=None):
    vertices = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    vertices *= np.asarray(half)
    if rotation is not None:
        vertices = vertices @ np.asarray(rotation).T
    vertices += center
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                      [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                      [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return vertices[faces]


def box_distance(points, half):
    q = np.abs(points) - np.asarray(half)
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)


class TriangleDistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.query = TriangleDistanceQuery(box_triangles())

    def test_distances_match_independent_box_formula(self):
        points = np.random.default_rng(10).uniform(-.05, .05, (250, 3))
        points = np.vstack((points, [[0., 0., 0.], [.03, 0., 0.], [.031, 0., 0.], [.029, 0., 0.]]))
        distances, normals = self.query.query(points)
        np.testing.assert_allclose(distances, box_distance(points, [.03, .02, .01]), atol=1e-8, rtol=2e-6)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1., atol=2e-7)
        self.assertLess(distances[-1], 0)
        self.assertGreater(distances[-2], 0)

    def test_outward_normals_inside_outside_and_on_each_face(self):
        points, expected = [], []
        for axis, half in enumerate((.03, .02, .01)):
            for sign in (-1., 1.):
                normal = np.eye(3)[axis] * sign
                for offset in (-.001, 0., .001):
                    points.append(normal * (half + offset))
                    expected.append(normal)
        _, normals = self.query.query(np.asarray(points))
        np.testing.assert_allclose(normals, expected, atol=3e-6)

    def test_translated_rotated_thin_box(self):
        angle = .43
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                             [-np.sin(angle), 0., np.cos(angle)]])
        center, half = np.array([.12, -.07, .03]), np.array([.04, .025, .0007])
        query = TriangleDistanceQuery(box_triangles(half, center, rotation))
        local = np.array([[0., 0., -.001], [0., 0., -.0004], [0., 0., .0004], [0., 0., .001]])
        distances, normals = query.query(local @ rotation.T + center)
        np.testing.assert_allclose(distances, box_distance(local, half), atol=3e-8)
        expected = np.array([[0., 0., -1.], [0., 0., -1.], [0., 0., 1.], [0., 0., 1.]]) @ rotation.T
        np.testing.assert_allclose(normals, expected, atol=6e-5)

    def test_empty_and_shaped_queries(self):
        distances, normals = self.query.query(np.zeros((2, 3, 3)))
        self.assertEqual(distances.shape, (2, 3))
        self.assertEqual(normals.shape, (2, 3, 3))
        distances, normals = self.query.query(np.empty((0, 3)))
        self.assertEqual(distances.shape, (0,))
        self.assertEqual(normals.shape, (0, 3))
        for points in ([[np.nan, 0, 0]], [[np.inf, 0, 0]], [[0, 0]], [[1e100, 0, 0]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                self.query.query(points)

    def test_rejects_invalid_meshes_without_changing_inputs(self):
        triangles = box_triangles()
        reversed_face = triangles.copy()
        reversed_face[0] = reversed_face[0, ::-1]
        degenerate = triangles.copy()
        degenerate[0, 0] = degenerate[0, 1]
        nonfinite = triangles.copy()
        nonfinite[0, 0, 0] = np.nan
        for mesh in (triangles[:-1], reversed_face, degenerate, nonfinite, triangles[:, ::-1],
                     np.concatenate((triangles, box_triangles(center=(.2, 0., 0.)))),
                     np.concatenate((triangles, box_triangles(half=(.005, .005, .005))))):
            copy = mesh.copy()
            with self.subTest(faces=len(mesh)), self.assertRaises(ValueError):
                TriangleDistanceQuery(mesh)
            np.testing.assert_array_equal(mesh, copy)

    def test_rejects_self_crossing_connected_mesh(self):
        triangles = box_triangles(half=(1., 1., 1.))
        triangles[np.all(triangles == [1., 1., 1.], axis=-1)] = [-2., 0., 0.]
        with self.assertRaisesRegex(ValueError, "self-intersecting"):
            TriangleDistanceQuery(triangles)

    def test_identical_float32_geometry_reuses_exact_validation(self):
        from experiments.differentiable_mpm import triangle_sdf

        with patch.object(triangle_sdf, "_validate_mesh", wraps=triangle_sdf._validate_mesh) as validate:
            query = TriangleDistanceQuery(box_triangles().astype(np.float32))
            self.assertEqual(validate.call_count, 1)
            self.assertEqual(query.metadata["source"], query.metadata["query_mesh"])
        with patch.object(triangle_sdf, "_validate_mesh", wraps=triangle_sdf._validate_mesh) as validate:
            TriangleDistanceQuery(box_triangles())
            self.assertEqual(validate.call_count, 2)

    def test_rejects_float32_geometry_collapse(self):
        with self.assertRaisesRegex(ValueError, "float32 query mesh"):
            TriangleDistanceQuery(box_triangles(half=(.01, .01, .01), center=(1e8, 1e8, 1e8)))

    def test_ray_settings_and_identity(self):
        for nsamples in (0, 2, True, 1.5):
            with self.subTest(nsamples=nsamples), self.assertRaises(ValueError):
                TriangleDistanceQuery(box_triangles(), nsamples=nsamples)
        identity = builder_identity()
        self.assertEqual(identity["builder"], BUILDER_VERSION)
        self.assertEqual(identity["nsamples"], 5)
        self.assertNotEqual(identity, builder_identity(nsamples=7))
        self.assertEqual(len(identity["builder_sha256"]), 64)


class SampledTriangleFieldTests(unittest.TestCase):
    def test_fields_match_analytic_samples_and_legacy_layout(self):
        triangles = box_triangles()
        distance, normal, minimum, spacing = build_tool_sdf(triangles.reshape(-1, 3), 15, .002)
        self.assertEqual(distance.shape, (15, 15, 15))
        self.assertEqual(normal.shape, (15, 15, 15, 3))
        self.assertEqual(minimum.shape, (3,))
        self.assertEqual(spacing.shape, (3,))
        for array in (distance, normal, minimum, spacing):
            self.assertEqual(array.dtype, np.float32)
            self.assertTrue(np.isfinite(array).all())
        coordinates = np.stack(np.meshgrid(*[np.arange(15)] * 3, indexing="ij"), axis=-1)
        points = minimum.astype(np.float64) + coordinates * spacing.astype(np.float64)
        np.testing.assert_allclose(distance, box_distance(points, [.03, .02, .01]), atol=1e-8, rtol=2e-6)
        self.assertTrue(np.any(distance < 0))

    def test_chunk_size_does_not_change_field(self):
        small = build_tool_sdf(box_triangles(), 9, .001, chunk_size=17)
        large = build_tool_sdf(box_triangles(), 9, .001, chunk_size=10000)
        for left, right in zip(small, large):
            np.testing.assert_array_equal(left, right)

    def test_bounds_include_padding_and_two_final_intervals(self):
        triangles = box_triangles(center=(.1, -.08, .02))
        for resolution in (6, 17, 128):
            for padding in (0., .0013, .00390625):
                minimum, spacing = grid_bounds(triangles, resolution, padding)
                last = minimum.astype(np.float64) + (resolution - 1) * spacing.astype(np.float64)
                lower, upper = triangles.reshape(-1, 3).min(axis=0), triangles.reshape(-1, 3).max(axis=0)
                self.assertTrue(np.all(lower - minimum >= padding + 2 * spacing.astype(np.float64)))
                self.assertTrue(np.all(last - upper >= padding + 2 * spacing.astype(np.float64)))

    def test_no_interior_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no negative interior"):
            build_tool_sdf(box_triangles(half=(.04, .03, 1e-6)), 8, .005)

    def test_invalid_grid_settings(self):
        for resolution in (0, 5, True, 8.5):
            with self.subTest(resolution=resolution), self.assertRaises(ValueError):
                grid_bounds(box_triangles(), resolution, 0.)
        for padding in (-1., np.nan, np.inf, True, [], "invalid"):
            with self.subTest(padding=padding), self.assertRaises(ValueError):
                grid_bounds(box_triangles(), 8, padding)
        for chunk in (0, -1, True, 2.5):
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                build_tool_sdf(box_triangles(), 8, 0., chunk_size=chunk)


if __name__ == "__main__":
    unittest.main()
