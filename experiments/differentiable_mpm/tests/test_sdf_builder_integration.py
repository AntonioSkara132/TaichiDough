"""Version selection, qualification and cache behavior without geometry kernels."""
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.differentiable_mpm.config import ExperimentConfig, FrameWindow, REQUIRED_PATHS
from experiments.differentiable_mpm.data import collision_inputs
from experiments.differentiable_mpm.dataset_config import _common_settings
from experiments.differentiable_mpm.tests.helpers import temporary_directory


class BuilderIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = temporary_directory(prefix="sdf-integration-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.names = ["UR5e_spathla", "gen3_spathla"]
        self.keys = ["ur_collision_mesh", "kinova_collision_mesh"]
        paths = {key: self.directory / key for key in REQUIRED_PATHS}
        paths.update({key: self.directory / key for key in self.keys})
        paths["collision_manifest"] = self.directory / "manifest.json"
        rows = [{"name": name, "collision_sha256": key, "units": "millimetres",
                 "coordinate_frame": "raw_stl_visual",
                 "topology": {"boundary_edges": 0, "nonmanifold_edges": 0}}
                for name, key in zip(self.names, self.keys)]
        paths["collision_manifest"].write_text(json.dumps({
            "schema": "taichidough/tool-collision-meshes/v1", "tools": rows}))
        self.config = ExperimentConfig("fixture", paths, {},
            {"n_particles": 1, "tool_collision": "sdf", "tool_contact_padding": .001,
             "tool_grid_contact_padding": .004},
            tool_sdf_resolution=16, training=FrameWindow(1, 2), validation=FrameWindow(3, 4))
        self.records = {key: {"sha256": key} for key in self.keys + ["collision_manifest"]}
        distance = np.ones((16, 16, 16), np.float32)
        distance[8, 8, 8] = -.001
        self.volume = (distance, np.ones((16, 16, 16, 3), np.float32),
                       np.zeros(3, np.float32), np.ones(3, np.float32))
        self.simulator = SimpleNamespace(
            __file__=__file__, UR_TOOL_VISUAL_ORIGIN=np.zeros(3), UR_TOOL_VISUAL_RPY=np.zeros(3),
            KINOVA_TOOL_VISUAL_ORIGIN=np.zeros(3), KINOVA_TOOL_VISUAL_RPY=np.zeros(3),
            load_binary_stl=Mock(return_value=(np.zeros((3, 3), np.float32), None)),
            mesh_in_tool_frame=lambda vertices, origin, rotation: vertices,
            rpy_to_matrix=lambda rpy: np.eye(3), build_mesh_sdf=Mock(return_value=self.volume))
        self.core = ModuleType("experiments.differentiable_mpm.triangle_sdf")
        self.core.build_tool_sdf = Mock(return_value=self.volume)
        self.core.builder_identity = lambda: {"version": "triangle-test"}
        self.qualifier = ModuleType("experiments.differentiable_mpm.qualify_tool_sdf")
        self.qualifier.qualify_field = Mock(return_value={"passed": True})
        self.qualifier.qualify_resolution = Mock(return_value={"passed": True})
        self.qualifier.qualification_identity = lambda profile, spacing: {"version": "qualification-test",
                                                                          "profile": profile, "spacing": spacing}
        self.qualifier.qualification_settings = Mock(side_effect=lambda profile, spacing: {"profile_marker": profile})
        self.addCleanup(patch.stopall)
        patch.dict("sys.modules", {self.core.__name__: self.core,
                                   self.qualifier.__name__: self.qualifier}).start()
        patch.dict(os.environ, {}, clear=True).start()

    def prepare(self, build=True):
        return collision_inputs(self.config, self.records, self.names, self.simulator, build)

    def test_default_and_invalid_selection(self):
        self.assertEqual(self.config.validate().tool_sdf_builder, "legacy-voxel-v1")
        for value in (None, True, [], "unknown"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "tool_sdf_builder"):
                replace(self.config, tool_sdf_builder=value).validate()
        changed = replace(self.config, tool_sdf_builder="triangle-distance-v1")
        changed.validate()
        self.assertNotEqual(changed.fingerprint, self.config.fingerprint)
        self.assertEqual(changed.as_dict()["tool_sdf_builder"], "triangle-distance-v1")

    def test_dataset_rejects_mixed_geometry_settings(self):
        baseline = _common_settings(self.config)
        for options in ({"tool_sdf_builder": "triangle-distance-v1"},
                        {"tool_sdf_builder": "triangle-distance-v1", "tool_sdf_qualification": "grid-relative-v1"},
                        {"tool_sdf_resolution": 32}, {"tool_mesh_scale": .002}):
            self.assertNotEqual(baseline, _common_settings(replace(self.config, **options)))

    def test_qualification_profile_selection(self):
        self.assertEqual(self.config.validate().tool_sdf_qualification, "strict-v2")
        with self.assertRaisesRegex(ValueError, "tool_sdf_qualification must be"):
            replace(self.config, tool_sdf_builder="triangle-distance-v1", tool_sdf_qualification="loose").validate()
        with self.assertRaisesRegex(ValueError, "only to the triangle-distance-v1 builder"):
            replace(self.config, tool_sdf_qualification="grid-relative-v1").validate()
        changed = replace(self.config, tool_sdf_builder="triangle-distance-v1", tool_sdf_qualification="grid-relative-v1")
        changed.validate()
        self.assertNotEqual(changed.fingerprint, replace(self.config, tool_sdf_builder="triangle-distance-v1").fingerprint)

    def test_grid_relative_profile_reaches_every_qualification_call(self):
        self.config.tool_sdf_builder = "triangle-distance-v1"
        self.config.tool_sdf_qualification = "grid-relative-v1"
        self.config.simulation["grid"] = 40
        _, provenance = self.prepare()
        self.qualifier.qualification_settings.assert_called_with("grid-relative-v1", 1 / 40)
        self.assertEqual(provenance["construction"]["qualification"]["profile"], "grid-relative-v1")
        for call in self.qualifier.qualify_resolution.call_args_list + self.qualifier.qualify_field.call_args_list:
            self.assertEqual(call.kwargs, {"profile_marker": "grid-relative-v1"})

    def test_legacy_builder_retains_particle_padding(self):
        sdf, provenance = self.prepare()
        self.assertEqual(sdf.resolution, 16)
        self.assertEqual(provenance["padding_m"], .001)
        self.assertEqual(self.simulator.build_mesh_sdf.call_count, 2)
        self.assertEqual(self.simulator.build_mesh_sdf.call_args.args[2], .001)
        self.core.build_tool_sdf.assert_not_called()
        self.qualifier.qualify_field.assert_not_called()

    def test_triangle_builder_qualifies_larger_grid_band(self):
        self.config.tool_sdf_builder = "triangle-distance-v1"
        _, provenance = self.prepare()
        self.assertEqual(self.core.build_tool_sdf.call_args.args[2], .004)
        self.assertEqual(provenance["padding_m"], .004)
        self.assertEqual(len(provenance["qualification"]), 2)
        self.assertIn("arrays", provenance["qualification"][0])
        self.simulator.build_mesh_sdf.assert_not_called()

    def test_explicitly_unqualified_geometry_is_rejected_before_build(self):
        self.config.tool_sdf_builder = "triangle-distance-v1"
        path = self.config.paths["collision_manifest"]
        document = json.loads(path.read_text())
        document["qualification"] = "not-qualified"
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "source geometry not-qualified"):
            self.prepare()
        self.core.build_tool_sdf.assert_not_called()

    def test_no_build_does_not_query_geometry(self):
        self.config.tool_sdf_builder = "triangle-distance-v1"
        sdf, provenance = self.prepare(False)
        self.assertIsNone(sdf)
        self.assertIsNone(provenance["qualification"])
        self.core.build_tool_sdf.assert_not_called()
        self.qualifier.qualify_field.assert_not_called()

    def test_cache_separates_versions_and_requalifies_hits(self):
        os.environ["TAICHIDOUGH_SDF_CACHE"] = str(self.directory / "cache")
        self.prepare()
        legacy_files = set((self.directory / "cache").glob("*.npz"))
        self.config.tool_sdf_builder = "triangle-distance-v1"
        self.prepare()
        self.assertEqual(len(set((self.directory / "cache").glob("*.npz")) - legacy_files), 2)
        self.core.build_tool_sdf.reset_mock()
        self.qualifier.qualify_field.reset_mock()
        os.environ["TAICHIDOUGH_SDF_CACHE_READONLY"] = "1"
        self.prepare()
        self.core.build_tool_sdf.assert_not_called()
        self.assertEqual(self.qualifier.qualify_field.call_count, 2)
        self.qualifier.qualify_field.return_value = {"passed": False, "reason": "bad field"}
        with self.assertRaisesRegex(ValueError, "Cached tool SDF qualification failed"):
            self.prepare()

    def test_failed_preflight_does_not_allocate_or_publish_field(self):
        os.environ["TAICHIDOUGH_SDF_CACHE"] = str(self.directory / "cache")
        self.config.tool_sdf_builder = "triangle-distance-v1"
        self.qualifier.qualify_resolution.return_value = {"passed": False, "reason": "missing interior"}
        with self.assertRaisesRegex(ValueError, "resolution preflight failed"):
            self.prepare()
        self.core.build_tool_sdf.assert_not_called()
        self.qualifier.qualify_field.assert_not_called()
        self.assertEqual(list((self.directory / "cache").glob("*.npz")), [])

    def test_failed_qualification_does_not_publish_cache(self):
        os.environ["TAICHIDOUGH_SDF_CACHE"] = str(self.directory / "cache")
        self.config.tool_sdf_builder = "triangle-distance-v1"
        self.qualifier.qualify_field.return_value = {"passed": False, "reason": "missing interior"}
        with self.assertRaisesRegex(ValueError, "Tool SDF qualification failed"):
            self.prepare()
        self.assertEqual(list((self.directory / "cache").glob("*.npz")), [])


if __name__ == "__main__":
    unittest.main()
