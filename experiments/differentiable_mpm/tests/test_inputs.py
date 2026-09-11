"""Input integrity, timestamp scheduling and strict-frame validation tests."""
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.differentiable_mpm.config import (
    EXPERIMENT_ROOT, REPOSITORY_ROOT, ExperimentConfig, FrameWindow, ObservationSettings,
    file_sha256, load_config, verify_input_paths,
)
from experiments.differentiable_mpm.data import camera_from_calibration, scaled_camera, visible_optical_points
from experiments.differentiable_mpm.evaluate import fresh_directory, require_complete_frames, require_owned_output
from experiments.differentiable_mpm.replay import RecordedControls, observation_schedule
from experiments.differentiable_mpm import evaluate
from experiments.differentiable_mpm.tests import helpers
from experiments.differentiable_mpm.tests.helpers import temporary_directory


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = temporary_directory(prefix="input-tests-")
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture_config(self):
        episode = self.directory / "episode"
        episode.mkdir()
        for name in ("pointclouds_interpolated.pt", "paths_interpolated.pt"):
            (episode / name).write_bytes(b"tensor fixture")
        paths = {"episode": episode}
        for name in ("calibration", "initial_particles", "reconstruction_metadata", "tool_geometry"):
            paths[name] = self.directory / name
            paths[name].write_bytes(b"fixture")
        return ExperimentConfig("fixture", paths, {k: file_sha256(v) for k, v in paths.items() if k != "episode"},
                                {"n_particles": 1}, training=FrameWindow(1, 2), validation=FrameWindow(3, 4))

    def test_expected_hash_is_checked(self):
        config = self.fixture_config()
        records = verify_input_paths(config)
        self.assertEqual(records["initial_particles"]["sha256"], config.expected_sha256["initial_particles"])
        config.paths["initial_particles"].write_bytes(b"different contents")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            verify_input_paths(config)

    def test_missing_paths_are_not_relocated(self):
        config = self.fixture_config()
        config.paths["initial_particles"] = self.directory / "different_workspace" / "initial_particles"
        with self.assertRaisesRegex(FileNotFoundError, "Explicit initial_particles"):
            verify_input_paths(config)

    def test_examples_select_plasticity_explicitly(self):
        pure = load_config(EXPERIMENT_ROOT / "configs" / "episode18_viscoelastic.json")
        plastic = load_config(EXPERIMENT_ROOT / "configs" / "episode18_stretch_clamp.json")
        self.assertEqual(pure.simulation["plasticity"], "none")
        self.assertEqual(plastic.simulation["plasticity"], "stretch-clamp")
        self.assertEqual(plastic.mass_kg, 0.25)
        self.assertEqual(plastic.simulation["tool_contact_padding"], 1 / 384)
        pure.fit_parameters.append("plastic_min")
        with self.assertRaisesRegex(ValueError, "requires explicit plasticity"):
            pure.validate()

    def test_examples_select_training_loss_version_explicitly(self):
        from experiments.differentiable_mpm.loss import LossConfig

        for name in ("episode18_viscoelastic.json", "episode18_stretch_clamp.json"):
            config = load_config(EXPERIMENT_ROOT / "configs" / name)
            self.assertEqual(config.loss["version"], "partial-visible-splats-v2")
            self.assertEqual(LossConfig(**config.loss).as_dict()["version"], "partial-visible-splats-v2")

    def test_explicit_overrides_do_not_change_unrelated_paths(self):
        path = EXPERIMENT_ROOT / "configs" / "episode18_viscoelastic.json"
        base = load_config(path)
        override = self.directory / "remote_layout" / "episode"
        changed = load_config(path, {"episode": override})
        self.assertEqual(changed.paths["episode"], override)
        self.assertEqual(changed.paths["calibration"], base.paths["calibration"])
        self.assertEqual(changed.expected_sha256, base.expected_sha256)
        self.assertTrue(changed.paths["tool_geometry"].is_relative_to(REPOSITORY_ROOT))

    def test_reject_inactive_plasticity_unknown_backend_and_overlap(self):
        config = self.fixture_config()
        config.fit_parameters = ["plastic_max"]
        with self.assertRaises(ValueError):
            config.validate()
        config.fit_parameters = ["youngs_modulus"]
        config.backend = "gpu"
        with self.assertRaisesRegex(ValueError, "explicitly select"):
            config.validate()
        config.backend = "cpu"
        config.validation = FrameWindow(2, 4)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            config.validate()

    def test_unphysical_parameter_bounds_fail_before_simulation(self):
        config = self.fixture_config()
        config.parameter_bounds = {"poisson_ratio": [0.1, 0.5]}
        with self.assertRaises(ValueError):
            config.validate()
        config.parameter_bounds = {}
        config.simulation["n_particles"] = 1.5
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            config.validate()

    def test_training_camera_scales_metric_intrinsics(self):
        calibration = SimpleNamespace(camera={"width": 640, "height": 480, "fx": 600., "fy": 620.,
                                              "cx": 321., "cy": 239., "zNear": 0.01, "zFar": 5.},
                                      scene_from_camera=np.eye(4))
        camera = scaled_camera(calibration, 160, 120)
        self.assertEqual(camera["fx"], 150.)
        self.assertEqual(camera["fy"], 155.)
        self.assertEqual(camera["cx"], 80.25)
        self.assertEqual(calibration.camera["width"], 640)
        depth = np.ones((120, 160))
        valid = np.zeros((120, 160), dtype=bool)
        valid[30, 40] = True
        points = visible_optical_points(depth, valid, camera)
        np.testing.assert_allclose(points[0], [(40.5 - 80.25) / 150., (30.5 - 59.75) / 155., 1.], atol=1e-7)

    def test_training_camera_preserves_calibrated_clipping_planes(self):
        scene_from_camera = np.eye(4)
        scene_from_camera[:3, 3] = [0.2, 0.4, 0.7]
        calibration = SimpleNamespace(camera={"width": 640, "height": 480, "fx": 600., "fy": 620.,
                                              "cx": 321., "cy": 239., "zNear": 0.025, "zFar": 0.85},
                                      scene_from_camera=scene_from_camera)
        reference_helpers = SimpleNamespace(topview=SimpleNamespace(rigid_inverse=np.linalg.inv))
        with patch("experiments.differentiable_mpm.data.get_reference_modules", return_value=reference_helpers):
            camera = camera_from_calibration(calibration, 160, 120)
        self.assertEqual(camera.near_m, 0.025)
        self.assertEqual(camera.far_m, 0.85)
        self.assertEqual(camera.fx, 150.)
        np.testing.assert_array_equal(camera.camera_from_scene[:3, 3], [-0.2, -0.4, -0.7])
        self.assertEqual(calibration.camera["zFar"], 0.85)

    def test_timestamp_ceiling_and_original_indices(self):
        schedule = observation_schedule([123., 123.0004, 123.00061], [10, 11, 15], 2, .0002)
        self.assertEqual([frame.completed_substeps for frame in schedule], [0, 2, 4])
        self.assertEqual([frame.original_source_frame for frame in schedule], [10, 11, 15])
        self.assertAlmostEqual(schedule[-1].sim_time_s, .0008)
        self.assertGreater(schedule[-1].pairing_error_s, 0)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            observation_schedule([0., .2, .2], [1, 2, 3], 2, .001)
        with self.assertRaisesRegex(ValueError, "unique"):
            observation_schedule([0., .2], [1, 1], 1, .001)

    def test_multiple_observations_can_share_state_index(self):
        schedule = observation_schedule([0., .001, .002], [0, 1, 2], 2, .01)
        self.assertEqual([frame.completed_substeps for frame in schedule], [0, 1, 1])

    def test_start_controls_and_end_clamping(self):
        class Replay:
            times = np.array([0., .025])

            def __init__(self):
                self.calls = []

            def at(self, time):
                self.calls.append(time)
                poses = np.zeros((2, 7), dtype=np.float32)
                poses[:, 0] = time
                poses[:, 6] = 1
                return poses, np.zeros((2, 6), dtype=np.float32)

        replay = Replay()
        controls = RecordedControls(replay, 3, .01)
        np.testing.assert_allclose(replay.calls, [0., .01, .02])
        self.assertEqual(controls[1].time, .01)
        end = controls.at_completed_step(3)
        self.assertEqual(replay.calls[-1], .025)
        self.assertEqual(end.time, .03)
        self.assertFalse(controls.poses.flags.writeable)

    def test_exact_frame_completeness_rejects_missing_duplicates_reordering(self):
        require_complete_frames([{"source_frame": i} for i in range(3)], range(3))
        for actual in ([0, 2], [0, 1, 1], [0, 2, 1], [False, 1, 2]):
            with self.assertRaises(ValueError):
                require_complete_frames([{"source_frame": i} for i in actual], range(3))

    def test_fresh_output_never_replaces_existing_content(self):
        path = self.directory / "output"
        with patch.object(evaluate, "EXPERIMENT_ROOT", self.directory):
            fresh_directory(path)
            (path / "keep.json").write_text("preserve")
            with self.assertRaisesRegex(ValueError, "not empty"):
                fresh_directory(path)
            self.assertEqual((path / "keep.json").read_text(), "preserve")
            with self.assertRaisesRegex(ValueError, "inside experiments"):
                require_owned_output(REPOSITORY_ROOT / "output" / "existing")

    def test_scratch_helper_uses_current_job_directory(self):
        job = self.directory / "new-job"
        with patch.dict(os.environ, {"CLAUDE_JOB_DIR": str(job)}):
            self.assertFalse((job / "tmp").exists())
            root = helpers.scratch_root()
            self.assertEqual(root, job / "tmp")
            with helpers.temporary_directory(prefix="first-") as first, helpers.temporary_directory(prefix="second-") as second:
                self.assertEqual(Path(first).parent, root)
                self.assertEqual(Path(second).parent, root)
                self.assertNotEqual(first, second)
            self.assertFalse(Path(first).exists())
            self.assertFalse(Path(second).exists())
            self.assertTrue(root.is_dir())

    def test_scratch_helper_has_repository_relative_fallback(self):
        experiment = self.directory / "copied-experiment"
        with patch.dict(os.environ, {"CLAUDE_JOB_DIR": ""}), patch.object(helpers, "EXPERIMENT_ROOT", experiment):
            expected = experiment / "runs" / "test_tmp"
            self.assertFalse(expected.exists())
            self.assertEqual(helpers.scratch_root(), expected)
            with helpers.temporary_directory() as child:
                self.assertEqual(Path(child).parent, expected)
            self.assertTrue(expected.is_dir())

    def test_invalid_observation_settings(self):
        for settings in ({"width": 0}, {"splat_radius": -1}, {"trim_quantile": .5}):
            with self.assertRaises(ValueError):
                ObservationSettings(**settings)


if __name__ == "__main__":
    unittest.main()
