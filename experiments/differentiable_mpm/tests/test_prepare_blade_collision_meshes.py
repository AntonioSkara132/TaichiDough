"""Record-preserving extraction and independent triangle-intersection evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.differentiable_mpm import prepare_blade_collision_meshes as module


def binary_stl(triangles):
    records = np.zeros(len(triangles), dtype=module.STL_RECORD)
    records["vertices"] = triangles
    records["normal"] = np.arange(len(triangles) * 3).reshape(-1, 3)
    records["attribute"] = np.arange(len(triangles)) + 13
    return b"preserved test header".ljust(80, b" ") + struct.pack("<I", len(triangles)) + records.tobytes()


def tetrahedron(offset=0):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32) + offset
    return vertices[[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]]


def synthetic_source(data, bounds=((0, 0, 0), (1, 1, 1)), faces=4):
    return module.BladeSource("test", "source.stl", hashlib.sha256(data).hexdigest(), bounds, faces)


class BladeExtractionTests(unittest.TestCase):
    def test_complete_binary_records_are_preserved(self):
        # Interleaving ensures extraction records original, noncontiguous indices.
        triangles = np.empty((8, 3, 3), dtype=np.float32)
        triangles[::2], triangles[1::2] = tetrahedron(), tetrahedron(10)
        data = binary_stl(triangles)
        extracted, original, indices, components = module.extract_blade(data, synthetic_source(data))
        np.testing.assert_array_equal(indices, [0, 2, 4, 6])
        self.assertEqual(extracted[:80], data[:80])
        for output_index, source_index in enumerate(indices):
            self.assertEqual(extracted[84 + 50 * output_index:84 + 50 * (output_index + 1)],
                             data[84 + 50 * source_index:84 + 50 * (source_index + 1)])
        np.testing.assert_array_equal(module.read_binary_stl(extracted), original[indices])
        self.assertEqual([row["faces"] for row in components], [4, 4])

    def test_hash_is_checked_before_geometry(self):
        data = binary_stl(tetrahedron())
        source = module.BladeSource("test", "source.stl", "0" * 64, ((0, 0, 0), (1, 1, 1)), 4)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            module.extract_blade(data, source)

    def test_bounds_and_face_count_identify_component(self):
        data = binary_stl(tetrahedron())
        for source in (synthetic_source(data, bounds=((0, 0, 0), (2, 1, 1))), synthetic_source(data, faces=3)):
            with self.assertRaisesRegex(ValueError, "Expected one pinned blade"):
                module.extract_blade(data, source)

    def test_connectivity_does_not_merge_nearby_vertices(self):
        triangles = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              [[np.nextafter(np.float32(1), np.float32(2)), 0, 0], [3, 0, 0], [3, 1, 0]]], dtype=np.float32)
        self.assertEqual(len(module.connected_components(triangles)[2]), 2)
        triangles[1, 0] = triangles[0, 0]
        self.assertEqual(len(module.connected_components(triangles)[2]), 1)

    def test_invalid_stl_is_rejected(self):
        data = binary_stl(tetrahedron())
        for invalid in (b"", data[:-1], data + b"extra", b" " * 80 + struct.pack("<I", 0)):
            with self.assertRaises(ValueError):
                module.read_binary_stl(invalid)
        triangles = tetrahedron()
        triangles[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            module.read_binary_stl(binary_stl(triangles))

    def test_topology_of_closed_tetrahedron(self):
        topology = module.mesh_topology(tetrahedron())
        self.assertEqual(topology["boundary_edges"], 0)
        self.assertEqual(topology["nonmanifold_edges"], 0)
        self.assertEqual(topology["inconsistent_winding_edges"], 0)
        self.assertEqual(topology["degenerate_faces"], 0)
        self.assertAlmostEqual(topology["signed_volume_mm3"], 1 / 6)

    def test_existing_output_is_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "sentinel"
            sentinel.write_bytes(b"unchanged")
            with self.assertRaises(FileExistsError):
                module.prepare_blade_collision_meshes("missing-source", directory)
            self.assertEqual(sentinel.read_bytes(), b"unchanged")
            self.assertEqual(list(Path(directory).iterdir()), [sentinel])

    def test_source_failure_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "new"
            source = module.BladeSource("test", "bad.stl", "0" * 64, ((0, 0, 0), (1, 1, 1)))
            (root / source.filename).write_bytes(b"wrong hash")
            with patch.object(module, "SOURCES", (source,)), self.assertRaisesRegex(ValueError, "SHA-256"):
                module.prepare_blade_collision_meshes(root, output)
            self.assertFalse(output.exists())


class ExactIntersectionTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float32)

    def test_proper_intersection_segment(self):
        b = [[0.5, 0.5, -1], [0.5, 0.5, 1], [1.5, 0.5, 0]]
        result = module.exact_triangle_intersection(self.a, b)
        self.assertEqual(result["kind"], "intersection-segment")
        self.assertTrue(result["strict_plane_straddle"])
        self.assertAlmostEqual(result["maximum_point_separation_mm"], 1)

    def test_disjoint_planes_and_coplanar_triangles(self):
        for b in (self.a + [0, 0, 1], self.a + [4, 0, 0]):
            self.assertFalse(module.exact_triangle_intersection(self.a, b)["intersects"])

    def test_tiny_positive_separation_is_not_rounded_to_contact(self):
        b = self.a.astype(np.float64) + [0, 0, 2 ** -60]
        self.assertFalse(module.exact_triangle_intersection(self.a, b)["intersects"])

    def test_exact_point_contact(self):
        b = [[2, 0, 0], [3, 0, 1], [3, 1, 1]]
        result = module.exact_triangle_intersection(self.a, b)
        self.assertEqual(result["kind"], "point-contact")
        self.assertEqual(result["points_mm"], [[2, 0, 0]])

    def test_coplanar_overlap_is_winding_independent(self):
        b = np.array([[0.25, 0.25, 0], [1, 0.25, 0], [0.25, 1, 0]])
        for first in (self.a, self.a[::-1]):
            for second in (b, b[::-1]):
                self.assertEqual(module.exact_triangle_intersection(first, second)["kind"], "coplanar-overlap")

    def test_coplanar_edge_contact(self):
        b = [[0, 0, 0], [2, 0, 0], [1, -1, 0]]
        result = module.exact_triangle_intersection(self.a, b)
        self.assertEqual(result["kind"], "coplanar-contact")
        self.assertAlmostEqual(result["maximum_point_separation_mm"], 2)

    def test_degenerate_triangle_is_not_accepted(self):
        result = module.exact_triangle_intersection(self.a, [[0, 0, 0]] * 3)
        self.assertEqual(result["kind"], "degenerate")
        self.assertIsNone(result["intersects"])

    def test_symmetry(self):
        cases = [[[0.5, 0.5, -1], [0.5, 0.5, 1], [1.5, 0.5, 0]],
                 [[0.25, 0.25, 0], [1, 0.25, 0], [0.25, 1, 0]],
                 [[2, 0, 0], [3, 0, 1], [3, 1, 1]]]
        for b in cases:
            forward = module.exact_triangle_intersection(self.a, b)
            reverse = module.exact_triangle_intersection(b, self.a)
            self.assertEqual(forward.pop("first_vertices_against_second_plane"),
                             reverse.pop("second_vertices_against_first_plane"))
            self.assertEqual(forward.pop("second_vertices_against_first_plane"),
                             reverse.pop("first_vertices_against_second_plane"))
            self.assertEqual(forward, reverse)

    def test_straddling_planes_with_disjoint_triangles(self):
        b = [[10, 0.5, -1], [10, 0.5, 1], [11, 0.5, 0]]
        self.assertFalse(module.exact_triangle_intersection(self.a, b)["intersects"])

    def test_coplanar_edge_crossings_without_contained_vertices(self):
        a = [[0, 2, 0], [-2, -1, 0], [2, -1, 0]]
        b = [[0, -2, 0], [-2, 1, 0], [2, 1, 0]]
        result = module.exact_triangle_intersection(a, b)
        self.assertEqual(result["kind"], "coplanar-overlap")
        self.assertEqual(len(result["points_rational_mm"]), 6)


@unittest.skipUnless(importlib.util.find_spec("open3d") is not None, "Open3D is required for mesh auditing")
class GeometryAuditTests(unittest.TestCase):
    def test_valid_extraction_and_views(self):
        data = binary_stl(np.concatenate((tetrahedron(), tetrahedron(10))))
        source = synthetic_source(data)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / source.filename).write_bytes(data)
            with patch.object(module, "SOURCES", (source,)):
                result = module.prepare_blade_collision_meshes(root, root / "output")
            self.assertTrue(result["accepted"])
            manifest = json.loads(Path(result["manifest"]).read_text())
            self.assertEqual(manifest["qualification"], "qualified")
            row = manifest["tools"][0]
            self.assertEqual(row["source_triangle_indices"], [0, 1, 2, 3])
            self.assertIn("<svg", (root / "output" / row["views"]).read_text())
            self.assertEqual((root / source.filename).read_bytes(), data)

    @unittest.skipUnless(all((module.DEFAULT_SOURCE / source.filename).is_file() for source in module.SOURCES),
                         "Pinned local collision meshes are not present")
    def test_actual_blades_remain_unqualified_and_records_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "blades"
            result = module.prepare_blade_collision_meshes(module.DEFAULT_SOURCE, output, write_views=False)
            self.assertFalse(result["accepted"])
            manifest = json.loads(Path(result["manifest"]).read_text())
            self.assertEqual(manifest["qualification"], "not-qualified")
            for source, row in zip(module.SOURCES, manifest["tools"]):
                original = (module.DEFAULT_SOURCE / source.filename).read_bytes()
                extracted = (output / row["collision_mesh"]).read_bytes()
                self.assertEqual(len(row["source_triangle_indices"]), 3522)
                self.assertEqual(extracted[84:], b"".join(original[84 + 50 * i:84 + 50 * (i + 1)]
                                                        for i in row["source_triangle_indices"]))
                audit = json.loads((output / row["qualification_report"]).read_text())
                self.assertFalse(audit["accepted"])
                self.assertEqual(audit["reported_intersection_pairs"], 2)
                self.assertEqual(audit["topology"]["boundary_edges"], 0)
                self.assertEqual(audit["topology"]["inconsistent_winding_edges"], 0)
                pairs = audit["exact_pair_checks"]
                self.assertEqual([pair["kind"] for pair in pairs], ["disjoint", "intersection-segment"])
                self.assertFalse(pairs[0]["intersects"])
                self.assertTrue(pairs[1]["intersects"])
                self.assertEqual(len(pairs[1]["points_rational_mm"]), 2)
                self.assertGreater(pairs[1]["maximum_point_separation_mm"], 0)
                self.assertLess(pairs[1]["maximum_point_separation_mm"], 1e-5)


if __name__ == "__main__":
    unittest.main()
