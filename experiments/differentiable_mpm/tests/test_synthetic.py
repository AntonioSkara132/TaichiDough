"""Small synthetic definitions and a real CPU joint-calibration subprocess test."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

import numpy as np

from experiments.differentiable_mpm.synthetic import (
    FIT_PARAMETERS, INITIAL_PARAMETERS, TRUTH, SyntheticConfig,
    excitation_diagnostics, make_initial_state, parameter_space, parse_args,
    simulation_config, synthetic_camera,
)


REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "experiments" / "differentiable_mpm" / "runs"


class SyntheticDefinitionTests(unittest.TestCase):
    def test_all_five_parameters_are_active_and_bounded(self):
        space = parameter_space()
        self.assertEqual(space.fit, FIT_PARAMETERS)
        self.assertEqual(len(space.fit), 5)
        self.assertEqual(space.plasticity, "stretch-clamp")
        decoded = space.physical(space.coordinates())
        for name in FIT_PARAMETERS:
            self.assertAlmostEqual(decoded[name], INITIAL_PARAMETERS[name])
            self.assertNotEqual(TRUTH[name], INITIAL_PARAMETERS[name])

    def test_known_excitation_activates_both_stretch_limits(self):
        config = SyntheticConfig()
        state = make_initial_state(config)
        state.validate()
        diagnostics = excitation_diagnostics(state, config)
        for key in ("truth_lower_active_particles", "truth_upper_active_particles",
                    "initial_lower_active_particles", "initial_upper_active_particles"):
            self.assertEqual(diagnostics[key], 64)
        self.assertTrue(np.all(np.linalg.det(state.F) > 0))
        simulation = simulation_config(config)
        self.assertAlmostEqual(simulation.particle_mass / simulation.particle_volume, 800.0)
        self.assertEqual(simulation.plasticity, "stretch-clamp")
        self.assertEqual(simulation.tool_collision, "none")
        self.assertFalse(simulation.use_jp)

    def test_unused_excitation_changes_deformation_and_velocity(self):
        config = SyntheticConfig()
        training, heldout = make_initial_state(config), make_initial_state(config, heldout=True)
        np.testing.assert_array_equal(training.x, heldout.x)
        self.assertGreater(np.linalg.norm(training.F - heldout.F), 0.1)
        self.assertGreater(np.linalg.norm(training.C - heldout.C), 1.0)
        self.assertGreater(np.linalg.norm(training.v - heldout.v), 0.1)
        np.testing.assert_array_equal(training.F, make_initial_state(config).F)
        camera = synthetic_camera()
        optical = training.x @ camera.camera_from_scene[:3, :3].T + camera.camera_from_scene[:3, 3]
        self.assertTrue(np.all(optical[:, 2] > camera.near_m))

    def test_explicit_optimizer_and_prediction_temperature_options(self):
        required = ["--backend", "cpu", "--precision", "f64", "--output-dir", str(RUNS / "unused")]
        default = parse_args(required)
        self.assertEqual(default.learning_rate_policy, "persistent-v1")
        self.assertEqual(default.visibility_temperature_m, 0.01)
        self.assertEqual(default.target_visibility_temperature_m, 0.002)
        changed = parse_args(required + ["--learning-rate-policy", "recover-v1", "--learning-rate-growth", "2",
                                         "--visibility-temperature-m", "0.002"])
        self.assertEqual(changed.learning_rate_policy, "recover-v1")
        self.assertEqual(changed.learning_rate_growth, 2.0)
        self.assertEqual(changed.visibility_temperature_m, 0.002)
        self.assertEqual(changed.target_visibility_temperature_m, 0.002)
        for values in ({"visibility_temperature_m": 0}, {"target_visibility_temperature_m": -1}):
            with self.assertRaises(ValueError):
                SyntheticConfig(**values)

    def test_timestep_schedule_and_invalid_settings(self):
        config = SyntheticConfig(steps=12, segment_length=4, observation_count=3)
        self.assertEqual(config.observation_steps, (4, 8, 12))
        for values in ({"steps": 1}, {"segment_length": 0}, {"precision": "half"},
                       {"dt": 0.1}, {"seed": -1}, {"grid": 48}, {"observation_count": 100}):
            with self.assertRaises(ValueError):
                SyntheticConfig(**values)


class SyntheticCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = RUNS / "tests" / ("synthetic_" + uuid.uuid4().hex)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO) + os.pathsep + environment.get("PYTHONPATH", "")
        command = [sys.executable, "-m", "experiments.differentiable_mpm.synthetic",
                   "--backend", "cpu", "--precision", "f64", "--steps", "12",
                   "--segment-length", "4", "--observation-count", "3",
                   "--iterations", "4", "--learning-rate", "0.1", "--output-dir", str(cls.output)]
        process = subprocess.run(command, cwd=REPO, env=environment, text=True,
                                 capture_output=True, timeout=300)
        if process.returncode:
            raise RuntimeError("Synthetic calibration subprocess failed:\n" + process.stdout + "\n" + process.stderr)
        cls.report = json.loads((cls.output / "result.json").read_text())

    def test_actual_joint_fit_reduces_visible_observation_loss(self):
        report = self.report
        self.assertEqual(report["fitted_parameters"], list(FIT_PARAMETERS))
        self.assertGreater(report["accepted_updates"], 0)
        self.assertLess(report["training"]["selected_loss"], report["training"]["initial_loss"])
        self.assertGreater(report["training"]["relative_loss_reduction"], 0.01)
        gradient = report["training"]["initial_coordinate_gradient"]
        self.assertEqual(set(gradient), set(FIT_PARAMETERS))
        self.assertTrue(all(np.isfinite(value) and abs(value) > 1e-8 for value in gradient.values()))
        self.assertTrue(report["runtime"]["forward_verified"])
        self.assertTrue(report["runtime"]["backward_verified"])
        self.assertEqual(report["runtime"]["backend"], "cpu")

    def test_targets_are_partial_observations_without_id_correspondence(self):
        report = self.report
        self.assertFalse(report["targets"]["particle_correspondence_used"])
        for observation in report["targets"]["training"]:
            self.assertGreater(observation["observed_pixels"], 4)
            self.assertGreater(observation["observed_points"], 0)
            self.assertLess(observation["observed_points"], 64)
        self.assertFalse(report["heldout"]["used_for_selection"])
        self.assertTrue(all(np.isfinite(report["heldout"][name]) for name in ("initial_loss", "selected_loss", "truth_loss")))
        self.assertIn("non-unique", report["interpretation"])

    def test_checkpointed_gradient_and_run_records(self):
        diagnostics = self.report["training"]["initial_gradient_diagnostics"]
        self.assertEqual(diagnostics["total_steps"], 12)
        self.assertEqual(diagnostics["segment_length"], 4)
        self.assertEqual(diagnostics["observation_gradient_injections"], [1, 1, 1])
        self.assertEqual(diagnostics["branch_summary_steps_checked"], 12)
        self.assertEqual(set(diagnostics["recomputed_endpoint_max_abs"]), {"x", "v", "C", "F", "Jp"})
        for name in ("run_manifest.json", "optimizer_state.json", "optimization_history.json", "synthetic_targets.json", "events.jsonl"):
            self.assertTrue((self.output / name).is_file(), name)
        manifest = json.loads((self.output / "run_manifest.json").read_text())
        self.assertEqual(manifest["identity"]["parameter_space"]["fit"], list(FIT_PARAMETERS))
        self.assertEqual(manifest["identity"]["simulation"]["dt"], 0.001)


if __name__ == "__main__":
    unittest.main()
