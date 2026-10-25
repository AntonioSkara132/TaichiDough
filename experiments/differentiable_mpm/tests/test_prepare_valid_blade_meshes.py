"""Pinned local retriangulation preserves vertices and passes exact mesh checks."""
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.differentiable_mpm import prepare_valid_blade_meshes as module
from experiments.differentiable_mpm import prepare_blade_collision_meshes as extraction


class PatchBoundTests(unittest.TestCase):
    def test_bound_is_certified_against_exact_squared_distance(self):
        result = module.patch_distance_bound([0, 0, 0], [1, 0, 0], [0.5, 1e-6, 2e-6])
        bound = result["hausdorff_upper_bound_mm"]
        squared = Fraction(result["squared_bound_rational_mm2"])
        self.assertGreaterEqual(Fraction(bound) ** 2, squared)
        self.assertLess(Fraction(math.nextafter(bound, 0)) ** 2, squared)
        self.assertEqual(result["edge_projection_fraction"], "1/2")

    def test_outside_projection_and_zero_edge_rejected(self):
        for a, b, c in (([0, 0, 0], [0, 0, 0], [1, 0, 0]), ([0, 0, 0], [1, 0, 0], [2, 1, 0])):
            with self.assertRaises(ValueError):
                module.patch_distance_bound(a, b, c)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "unchanged"
            marker.write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                module.prepare_valid_blade_meshes("missing", directory)
            self.assertEqual(marker.read_bytes(), b"original")

    def test_source_failure_writes_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            with self.assertRaises(FileNotFoundError):
                module.prepare_valid_blade_meshes(directory, output)
            self.assertFalse(output.exists())


@unittest.skipUnless(all((extraction.DEFAULT_SOURCE / source.filename).is_file() for source in extraction.SOURCES),
                     "Pinned local meshes are not present")
class ActualBladeRetriangulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = []
        for source in extraction.SOURCES:
            raw = (extraction.DEFAULT_SOURCE / source.filename).read_bytes()
            extracted, _, indices, _ = extraction.extract_blade(raw, source)
            corrected, report = module.apply_pinned_flips(extracted, indices, source)
            cls.cases.append((source, raw, extracted, corrected, report))

    def test_all_vertex_coordinates_and_bounds_are_preserved(self):
        for source, _, extracted, corrected, report in self.cases:
            before = extraction.read_binary_stl(extracted)
            after = extraction.read_binary_stl(corrected)
            np.testing.assert_array_equal(np.unique(before.reshape(-1, 3), axis=0), np.unique(after.reshape(-1, 3), axis=0))
            np.testing.assert_array_equal(before.min(axis=(0, 1)), after.min(axis=(0, 1)))
            np.testing.assert_array_equal(before.max(axis=(0, 1)), after.max(axis=(0, 1)))
            self.assertEqual(report["vertex_displacement_mm"], 0)
            self.assertLess(report["hausdorff_upper_bound_mm"], 0.000015)
            self.assertEqual(len(report["flips"]), 16 if source.name == "UR5e_spathla" else 15)

    def test_only_pinned_face_records_change(self):
        for source, _, extracted, corrected, report in self.cases:
            changed = [i for i in range(3522)
                       if extracted[84 + 50 * i:84 + 50 * (i + 1)] != corrected[84 + 50 * i:84 + 50 * (i + 1)]]
            self.assertEqual(changed, report["changed_blade_triangle_indices"])
            self.assertEqual(len(changed), 32 if source.name == "UR5e_spathla" else 30)
            before = np.frombuffer(extracted, dtype=extraction.STL_RECORD, offset=84)
            after = np.frombuffer(corrected, dtype=extraction.STL_RECORD, offset=84)
            np.testing.assert_array_equal(before["attribute"], after["attribute"])
            self.assertEqual(extracted[:84], corrected[:84])

    def test_bound_and_disjoint_flip_records_are_verified(self):
        for _, _, _, _, report in self.cases:
            seen = set()
            for flip in report["flips"]:
                self.assertFalse(seen & set(flip["source_triangle_indices"]))
                seen.update(flip["source_triangle_indices"])
                self.assertGreaterEqual(Fraction(flip["hausdorff_upper_bound_mm"]) ** 2,
                                        Fraction(flip["squared_bound_rational_mm2"]))
                self.assertLess(flip["old_sliver_altitude_mm"], module.MAX_SLIVER_ALTITUDE_MM)
                self.assertGreater(flip["new_minimum_altitude_mm"], module.MAX_SLIVER_ALTITUDE_MM)

    def test_repeat_application_is_byte_deterministic(self):
        for source, _, extracted, corrected, report in self.cases:
            repeated, second_report = module.apply_pinned_flips(extracted, np.asarray(report["source_triangle_indices"]), source)
            self.assertEqual(repeated, corrected)
            self.assertEqual(second_report, report)

    def test_invalid_pinned_pair_is_rejected(self):
        source, _, extracted, _, report = self.cases[0]
        with patch.dict(module.SOURCE_FACE_PAIRS, {source.name: ((5030, 3819),)}):
            with self.assertRaisesRegex(ValueError, "Pinned neighbor"):
                module.apply_pinned_flips(extracted, np.asarray(report["source_triangle_indices"]), source)

    @unittest.skipUnless(importlib.util.find_spec("open3d") is not None, "Open3D is required")
    def test_saved_assets_are_qualified_only_as_meshes(self):
        from experiments.differentiable_mpm.reference_adapter import get_reference_modules, reference_policy
        with reference_policy("frozen"):
            simulator = get_reference_modules().simulator
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            result = module.prepare_valid_blade_meshes(extraction.DEFAULT_SOURCE, output)
            self.assertTrue(result["mesh_geometry_valid"])
            self.assertEqual(result["field_qualification"], "pending")
            manifest = json.loads(Path(result["manifest"]).read_text())
            self.assertFalse(manifest["vertex_positions_modified"])
            self.assertFalse(manifest["tracking_transforms_modified"])
            for row, case in zip(manifest["tools"], self.cases):
                source, original, _, expected, _ = case
                path = output / row["collision_mesh"]
                self.assertEqual(path.read_bytes(), expected)
                self.assertEqual(hashlib.sha256(expected).hexdigest(), row["collision_sha256"])
                audit = json.loads((output / row["geometry_report"]).read_text())
                for key in ("raw_mm_validation", "production_tool_frame_m_validation"):
                    validation = audit[key]["intersection_validation"]
                    self.assertTrue(validation["valid"])
                    self.assertTrue(validation["complete"])
                    self.assertEqual(validation["unexpected_intersection_count"], 0)
                    self.assertEqual(validation["degenerate_face_count"], 0)
                loaded, _ = simulator.load_binary_stl(path, 0.001)
                prefix = "UR" if source.name == "UR5e_spathla" else "KINOVA"
                origin = getattr(simulator, prefix + "_TOOL_VISUAL_ORIGIN")
                rpy = getattr(simulator, prefix + "_TOOL_VISUAL_RPY")
                expected_transformed = simulator.mesh_in_tool_frame(loaded, origin, simulator.rpy_to_matrix(rpy))
                actual_transformed, _ = module.production_transform(extraction.read_binary_stl(expected), source.name)
                np.testing.assert_array_equal(actual_transformed.reshape(-1, 3), expected_transformed)
                self.assertEqual((extraction.DEFAULT_SOURCE / source.filename).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
