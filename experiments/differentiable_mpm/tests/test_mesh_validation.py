"""Exact self-intersection validation, including permitted adjacent contacts."""
import hashlib
from pathlib import Path
import unittest

import numpy as np

from experiments.differentiable_mpm import mesh_validation as module


class MeshValidationTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float64)

    def check_pair(self, b, expected_valid, first=None):
        result = module.validate_mesh_intersections(np.array([self.a if first is None else first, b], dtype=np.float64))
        self.assertTrue(result["complete"])
        self.assertEqual(result["valid"], expected_valid, result)
        self.assertEqual(result["candidate_pair_count"], result["disjoint_pair_count"]
                         + result["allowed_adjacent_pair_count"] + result["unexpected_intersection_count"])
        return result

    def test_closed_tetrahedron_adjacency(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        triangles = vertices[[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]]
        result = module.validate_mesh_intersections(triangles)
        self.assertTrue(result["valid"])
        self.assertTrue(result["complete"])
        self.assertEqual(result["candidate_pair_count"], 6)
        self.assertEqual(result["allowed_adjacent_pair_count"], 6)
        self.assertEqual(result["rational_pair_count"], 0)

    def test_aabb_reject(self):
        result = self.check_pair(self.a + [0, 0, 1], True)
        self.assertEqual(result["candidate_pair_count"], 0)

    def test_plane_reject_with_overlapping_boxes(self):
        a = [[0, 0, 0], [2, 0, 2], [0, 2, 0]]
        b = np.asarray(a) + [0, 0, 0.25]
        result = self.check_pair(b, True, first=a)
        self.assertEqual(result["candidate_pair_count"], 1)
        self.assertEqual(result["disjoint_pair_count"], 1)

    def test_crossing_nonadjacent_triangles(self):
        result = self.check_pair([[0.5, 0.5, -1], [0.5, 0.5, 1], [1.5, 0.5, 0]], False)
        self.assertEqual(result["unexpected_intersection_count"], 1)
        self.assertEqual(len(result["examples"][0]["intersection_points_rational"]), 2)

    def test_shared_edge_non_coplanar_is_allowed(self):
        result = self.check_pair([[2, 0, 0], [0, 0, 0], [1, 0, 1]], True)
        self.assertEqual(result["allowed_adjacent_pair_count"], 1)

    def test_shared_edge_coplanar_opposite_sides_allowed(self):
        result = self.check_pair([[2, 0, 0], [0, 0, 0], [1, -1, 0]], True)
        self.assertEqual(result["allowed_adjacent_pair_count"], 1)

    def test_shared_edge_does_not_allow_same_side_overlap(self):
        result = self.check_pair([[2, 0, 0], [0, 0, 0], [1, 1, 0]], False)
        self.assertEqual(result["examples"][0]["shared_vertex_count"], 2)

    def test_shared_vertex_non_coplanar_allowed(self):
        result = self.check_pair([[0, 0, 0], [-1, 0, 1], [0, -1, 1]], True)
        self.assertEqual(result["allowed_adjacent_pair_count"], 1)

    def test_shared_vertex_coplanar_allowed(self):
        result = self.check_pair([[0, 0, 0], [-1, 0, 0], [0, -1, 0]], True)
        self.assertEqual(result["allowed_adjacent_pair_count"], 1)

    def test_shared_vertex_does_not_allow_extra_segment(self):
        result = self.check_pair([[0, 0, 0], [0.5, 0.5, -1], [0.5, 0.5, 1]], False)
        self.assertEqual(result["examples"][0]["shared_vertex_count"], 1)

    def test_shared_vertex_does_not_allow_coplanar_overlap(self):
        self.check_pair([[0, 0, 0], [1, 0.5, 0], [0.5, 1, 0]], False)

    def test_non_topological_point_contact_rejected(self):
        result = self.check_pair([[1, 0, 0], [1, -1, 1], [1, 1, 1]], False)
        self.assertEqual(result["examples"][0]["shared_vertex_count"], 0)
        self.assertEqual(result["examples"][0]["intersection_points"], [[1, 0, 0]])

    def test_coplanar_overlap_without_contained_vertices(self):
        result = self.check_pair([[0, -2, 0], [-2, 1, 0], [2, 1, 0]], False,
                                 first=[[0, 2, 0], [-2, -1, 0], [2, -1, 0]])
        self.assertEqual(len(result["examples"][0]["intersection_points"]), 6)

    def test_coplanar_disjoint_with_overlapping_boxes(self):
        self.check_pair([[1.5, 1.5, 0], [2, 1.5, 0], [1.5, 2, 0]], True)

    def test_duplicate_faces_rejected_in_either_winding(self):
        for b in (self.a, self.a[::-1]):
            result = self.check_pair(b, False)
            self.assertEqual(result["examples"][0]["shared_vertex_count"], 3)

    def test_degenerate_fails_without_claiming_complete(self):
        result = module.validate_mesh_intersections([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]])
        self.assertFalse(result["valid"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["degenerate_face_indices"], [0])

    def test_tiny_nonzero_triangle_is_not_degenerate(self):
        result = module.validate_mesh_intersections([[[0, 0, 0], [1e-200, 0, 0], [0, 1e-200, 0]]])
        self.assertTrue(result["valid"])
        self.assertEqual(result["degenerate_face_count"], 0)

    def test_no_tolerance_turns_small_gap_into_contact(self):
        result = self.check_pair(self.a + [0, 0, 2 ** -100], True)
        self.assertEqual(result["unexpected_intersection_count"], 0)

    def test_power_of_two_scaling_preserves_result(self):
        pair = np.array([self.a, [[0.5, 0.5, -1], [0.5, 0.5, 1], [1.5, 0.5, 0]]])
        for exponent in (-100, 0, 100):
            result = module.validate_mesh_intersections(pair * (2.0 ** exponent))
            self.assertFalse(result["valid"])
            self.assertEqual(result["unexpected_intersection_count"], 1)

    def test_examples_limit_does_not_stop_complete_search(self):
        triangles = np.array([self.a, self.a, self.a])
        result = module.validate_mesh_intersections(triangles, max_reported_pairs=1)
        self.assertTrue(result["complete"])
        self.assertEqual(result["unexpected_intersection_count"], 3)
        self.assertEqual(len(result["examples"]), 1)
        self.assertEqual(len(module.validate_mesh_intersections(triangles, max_reported_pairs=0)["examples"]), 0)

    def test_invalid_inputs_raise(self):
        for values in ([], np.zeros((2, 3)), np.zeros((1, 4, 3)), np.full((1, 3, 3), np.nan)):
            with self.assertRaises(ValueError):
                module.validate_mesh_intersections(values)
        for maximum in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                module.validate_mesh_intersections([self.a], max_reported_pairs=maximum)

    def test_independent_pair_predicate_agrees_on_random_integer_triangles(self):
        from experiments.differentiable_mpm.prepare_blade_collision_meshes import exact_triangle_intersection
        random = np.random.default_rng(412)
        checked = 0
        while checked < 100:
            pair = random.integers(-5, 6, size=(2, 3, 3)).astype(float)
            if len(np.unique(pair.reshape(-1, 3), axis=0)) != 6:
                continue
            reference = exact_triangle_intersection(*pair)
            if reference["intersects"] is None:
                continue
            report = module.validate_mesh_intersections(pair)
            self.assertTrue(report["complete"])
            self.assertEqual(report["valid"], not reference["intersects"], pair.tolist())
            checked += 1

    def test_actual_adjacent_blade_overlap_is_not_ignored(self):
        from experiments.differentiable_mpm.prepare_blade_collision_meshes import DEFAULT_SOURCE, read_binary_stl, exact_triangle_intersection
        path = DEFAULT_SOURCE / "ur_spathla_x10_closed.stl"
        if not path.is_file():
            self.skipTest("Pinned local mesh is not present")
        triangles = read_binary_stl(path.read_bytes())[[6533, 6535]]
        reference = exact_triangle_intersection(*triangles)
        self.assertTrue(reference["intersects"])
        self.assertEqual(reference["kind"], "intersection-segment")
        report = module.validate_mesh_intersections(triangles)
        self.assertTrue(report["complete"])
        self.assertFalse(report["valid"])
        self.assertEqual(report["examples"][0]["shared_vertex_count"], 1)

    def test_flat_input_and_identity(self):
        self.assertTrue(module.validate_mesh_intersections(self.a)["valid"])
        identity = module.validation_identity()
        self.assertEqual(identity["source_sha256"], hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
        self.assertEqual(identity["imported_geometry_helpers"], [])


if __name__ == "__main__":
    unittest.main()
