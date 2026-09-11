# pyright: reportMissingImports=false

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sweep_episode18_viscosity as sweep
from material_calibration import file_sha256
from sweep_episode18_viscosity import (
    FIXED_TOOL_CONTACT_PADDING_M,
    FIXED_YOUNGS_MODULUS_PA,
    REPLAY_END_FRAME,
    VISCOSITY_VALUES_PA_S,
    build_case_argv,
    viscosity_cases,
)


class Episode18ViscositySweepTests(unittest.TestCase):
    def baseline(self):
        return [
            "python3", "sim.py", "--cpu", "--youngs-modulus", "2000", "--grid", "48",
            "--tool-collision", "sdf", "--tool-sdf-resolution", "64",
            "--replay-start-frame", "0", "--replay-end-frame", "387", "--replay-stride", "4",
            "--tool-contact-padding", "0", "--tool-contact-friction", "0.2", "--viscosity", "0",
            "--output-dir", "/old", "--replay-episode", "/episode",
            "--initial-particles-calibration", "/calibration.json", "--dt", "0.0002", "--unrelated", "keep-me",
        ]

    def test_viscosity_values_and_isolated_rewrites(self):
        self.assertEqual([case["viscosity_pa_s"] for case in viscosity_cases()], list(VISCOSITY_VALUES_PA_S))
        argv = build_case_argv(self.baseline(), 2.5, 0.17, Path("/simulation"))
        self.assertEqual(argv[argv.index("--youngs-modulus") + 1], format(FIXED_YOUNGS_MODULUS_PA, ".12g"))
        self.assertEqual(argv[argv.index("--tool-contact-padding") + 1], format(FIXED_TOOL_CONTACT_PADDING_M, ".17g"))
        self.assertEqual(argv[argv.index("--tool-contact-friction") + 1], "0.17")
        self.assertEqual(argv[argv.index("--viscosity") + 1], "2.5")
        self.assertEqual(argv[argv.index("--replay-start-frame") + 1], "0")
        self.assertEqual(argv[argv.index("--replay-stride") + 1], "1")
        self.assertEqual(argv[argv.index("--replay-end-frame") + 1], str(REPLAY_END_FRAME))
        self.assertEqual(argv[argv.index("--output-dir") + 1], "/simulation")
        self.assertIn("--cpu", argv)
        self.assertEqual(argv[argv.index("--unrelated") + 1], "keep-me")

    def test_selected_modulus_and_backend_overrides(self):
        argv = build_case_argv(self.baseline(), 1.0, 0.2, Path("/simulation"), youngs_modulus_pa=4500.0, cpu=False)
        self.assertEqual(argv[argv.index("--youngs-modulus") + 1], "4500")
        self.assertNotIn("--cpu", argv)
        self.assertNotIn("--gpu", argv)
        gpu_baseline = [value for value in self.baseline() if value != "--cpu"]
        self.assertNotIn("--cpu", build_case_argv(gpu_baseline, 0.0, 0.2, Path("/simulation")))
        cpu_argv = build_case_argv(gpu_baseline, 0.0, 0.2, Path("/simulation"), cpu=True)
        self.assertEqual(cpu_argv.count("--cpu"), 1)
        self.assertEqual(build_case_argv(self.baseline(), 0.0, 0.2, Path("/simulation"), cpu=True).count("--cpu"), 1)

    def test_rejects_unapproved_viscosity_and_negative_friction(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            build_case_argv(self.baseline(), 3.0, 0.2, Path("/simulation"))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            build_case_argv(self.baseline(), 0.0, -0.1, Path("/simulation"))

    def test_rejects_nonpositive_and_nonfinite_modulus(self):
        for value in (0.0, -1.0, math.inf, -math.inf, math.nan):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "youngs-modulus must be positive and finite"):
                build_case_argv(self.baseline(), 0.0, 0.2, Path("/simulation"), youngs_modulus_pa=value)

    def test_cli_backend_defaults_and_mutual_exclusion(self):
        self.assertIsNone(sweep.parse_args(["--tool-contact-friction", "0.2"]).cpu)
        self.assertTrue(sweep.parse_args(["--tool-contact-friction", "0.2", "--cpu"]).cpu)
        self.assertFalse(sweep.parse_args(["--tool-contact-friction", "0.2", "--gpu"]).cpu)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            sweep.parse_args(["--tool-contact-friction", "0.2", "--cpu", "--gpu"])

    def input_fixture(self, directory):
        directory = Path(directory)
        baseline = self.baseline()
        simulator = directory / "simulator.py"
        simulator.write_text("# simulated test source\n")
        baseline[1] = str(simulator)
        episode = directory / "episode"
        episode.mkdir()
        for filename in ("pointclouds_interpolated.pt", "paths_interpolated.pt", "sequence_metadata.json"):
            (episode / filename).write_text("test input\n")
        baseline = sweep.replace_option(baseline, "--replay-episode", str(episode))
        for option in (
            "--initial-particles", "--initial-particles-metadata", "--initial-particles-calibration", "--tool-geometry",
            "--ur-tool-mesh", "--kinova-tool-mesh", "--ur-tool-collision-mesh", "--kinova-tool-collision-mesh",
        ):
            path = directory / (option[2:] + ".data")
            path.write_text("test input\n")
            if option in baseline:
                baseline = sweep.replace_option(baseline, option, str(path))
            else:
                baseline.extend([option, str(path)])
        return baseline

    def valid_loss(self):
        return {
            "valid": True, "weighted_total": 0.25, "failure_reason": None,
            "counts": {"total": 60, "valid": 60, "invalid": 0},
            "frames": [{"source_frame": index, "valid": True} for index in range(1, 61)],
        }

    def fake_run(self, argv, stdout_path, stderr_path, timeout_s):
        stdout_path.write_text("test execution\n")
        stderr_path.write_text("")
        output = Path(argv[argv.index("--output-dir") + 1])
        output.mkdir(parents=True)
        filename = "dynamic_topview_metrics.json" if "--taichi-metadata" in argv else "camera_parameters.json"
        (output / filename).write_text("{}\n")
        return {"returncode": 0, "duration_s": 0.01, "failure_reason": None}

    def run_case(self, root, baseline, loss=None, **kwargs):
        with mock.patch.object(sweep, "run", side_effect=self.fake_run) as run_mock, mock.patch.object(
            sweep, "score_evaluation_artifact", return_value=self.valid_loss() if loss is None else loss,
        ):
            result = sweep.execute_case(root, baseline, viscosity_cases()[0], 0.2, 10.0, **{
                "resume": False, "retry_failures": False, "youngs_modulus_pa": 4500.0, **kwargs,
            })
        return result, run_mock

    def test_default_collision_assets_and_manifest_are_fingerprinted(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline = self.input_fixture(temporary)
            defaults = {}
            for option, filename in (
                ("--ur-tool-collision-mesh", "ur_spathla_collision_solid.stl"),
                ("--kinova-tool-collision-mesh", "gen3_spathla_collision_solid.stl"),
            ):
                index = baseline.index(option)
                del baseline[index:index + 2]
                path = Path(temporary) / filename
                path.write_text("test default collision mesh\n")
                defaults[filename] = path
            simulation = Path(temporary) / "simulation"
            simulator = build_case_argv(baseline, 0.0, 0.2, simulation)
            evaluator = sweep.evaluator_argv(baseline, simulation, Path(temporary) / "evaluation")
            with mock.patch("taichi_viscoelastic_mpm_scene.find_tool_mesh", side_effect=lambda filename, override: defaults[filename]) as resolve_mesh:
                first = sweep.case_fingerprint(simulator, evaluator)
                self.assertEqual(resolve_mesh.call_args_list, [
                    mock.call("ur_spathla_collision_solid.stl", None),
                    mock.call("gen3_spathla_collision_solid.stl", None),
                ])
                defaults["ur_spathla_collision_solid.stl"].write_text("changed default collision mesh\n")
                changed = sweep.case_fingerprint(simulator, evaluator)
            self.assertNotEqual(first["input_files"]["ur_tool_collision_mesh"]["sha256"], changed["input_files"]["ur_tool_collision_mesh"]["sha256"])
            manifest = first["input_files"]["collision_manifest"]
            self.assertEqual(Path(manifest["path"]).name, "tool_collision_meshes_v1.json")
            self.assertEqual(manifest["sha256"], file_sha256(Path(manifest["path"])))

    def test_identical_resume_reuses_only_matching_complete_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline = self.input_fixture(temporary)
            root = Path(temporary) / "sweep"
            result, first_run = self.run_case(root, baseline)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["youngs_modulus_pa"], 4500.0)
            self.assertEqual(result["backend_request"], "cpu")
            self.assertEqual(result["fingerprint"]["source_files"]["simulator"]["sha256"], file_sha256(Path(baseline[1])))
            self.assertEqual(first_run.call_count, 2)
            repeated, second_run = self.run_case(root, baseline, resume=True)
            self.assertEqual(repeated, result)
            second_run.assert_not_called()

    def test_resume_rejects_changed_source_inputs_and_settings_without_overwriting(self):
        for change in ("source", "particles", "sequence", "modulus", "backend"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                baseline = self.input_fixture(temporary)
                root = Path(temporary) / "sweep"
                self.run_case(root, baseline)
                result_path = root / "cases" / "viscosity_0" / "result.json"
                previous = result_path.read_bytes()
                kwargs: dict[str, bool | float] = {"resume": True, "retry_failures": True}
                if change == "source":
                    Path(baseline[1]).write_text("# corrected solver\n")
                elif change == "particles":
                    Path(baseline[baseline.index("--initial-particles") + 1]).write_text("changed particles\n")
                elif change == "sequence":
                    (Path(baseline[baseline.index("--replay-episode") + 1]) / "paths_interpolated.pt").write_text("changed motion\n")
                elif change == "modulus":
                    kwargs["youngs_modulus_pa"] = 5000.0
                else:
                    kwargs["cpu"] = False
                with self.assertRaisesRegex(ValueError, "fingerprint is missing or differs"):
                    self.run_case(root, baseline, **kwargs)
                self.assertEqual(result_path.read_bytes(), previous)

    def test_retry_failures_cannot_overwrite_a_completed_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline = self.input_fixture(temporary)
            root = Path(temporary) / "sweep"
            self.run_case(root, baseline)
            result_path = root / "cases" / "viscosity_0" / "result.json"
            previous = result_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "Completed case already exists"):
                self.run_case(root, baseline, retry_failures=True)
            self.assertEqual(result_path.read_bytes(), previous)

    def test_resume_rejects_legacy_unfingerprinted_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline = self.input_fixture(temporary)
            root = Path(temporary) / "sweep"
            case_dir = root / "cases" / "viscosity_0"
            case_dir.mkdir(parents=True)
            result_path = case_dir / "result.json"
            result_path.write_text(json.dumps({"status": "complete", "loss": self.valid_loss()}))
            previous = result_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "fingerprint is missing or differs"):
                self.run_case(root, baseline, resume=True, retry_failures=True)
            self.assertEqual(result_path.read_bytes(), previous)

    def test_invalid_incomplete_and_nonfinite_losses_are_failed(self):
        invalid = self.valid_loss()
        invalid.update(valid=False, weighted_total=None, failure_reason="one frame is invalid")
        missing_frame = self.valid_loss()
        missing_frame["frames"].pop()
        nonfinite = self.valid_loss()
        nonfinite["weighted_total"] = math.nan
        for loss in (invalid, missing_frame, nonfinite):
            with self.subTest(loss=loss.get("failure_reason")), tempfile.TemporaryDirectory() as temporary:
                baseline = self.input_fixture(temporary)
                result, _ = self.run_case(Path(temporary) / "sweep", baseline, loss=loss)
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["loss"])
                self.assertIn("could not score evaluation", result["failure_reason"])

    def test_retry_failed_case_preserves_prior_partial_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline = self.input_fixture(temporary)
            root = Path(temporary) / "sweep"
            failed, _ = self.run_case(root, baseline, loss={"valid": False, "failure_reason": "test failure"})
            original = Path(failed["simulation_dir"]) / "camera_parameters.json"
            previous = original.read_bytes()
            retried, _ = self.run_case(root, baseline, resume=True, retry_failures=True)
            self.assertEqual(retried["status"], "complete")
            self.assertEqual(failed["cache_key"], retried["cache_key"])
            self.assertIn("attempt_001", retried["simulation_dir"])
            self.assertEqual(original.read_bytes(), previous)
            repeated, run_mock = self.run_case(root, baseline, resume=True)
            self.assertEqual(repeated, retried)
            run_mock.assert_not_called()

    def test_source_change_during_execution_fails_the_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline = self.input_fixture(temporary)
            root = Path(temporary) / "sweep"

            def changing_run(argv, stdout_path, stderr_path, timeout_s):
                result = self.fake_run(argv, stdout_path, stderr_path, timeout_s)
                Path(baseline[1]).write_text("# source changed during test execution\n")
                return result

            with mock.patch.object(sweep, "run", side_effect=changing_run), mock.patch.object(sweep, "score_evaluation_artifact", return_value=self.valid_loss()):
                result = sweep.execute_case(root, baseline, viscosity_cases()[0], 0.2, 10.0, False, False, youngs_modulus_pa=4500.0)
            self.assertEqual(result["status"], "failed")
            self.assertIn("changed during execution", result["failure_reason"])

    def test_main_report_has_selected_modulus_and_preserved_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline = self.input_fixture(temporary)
            source = Path(temporary) / "command.json"
            source.write_text(json.dumps(baseline))
            output = Path(temporary) / "sweep"
            args = ["--source-command", str(source), "--output-dir", str(output), "--tool-contact-friction", "0.2", "--youngs-modulus", "4500"]
            with mock.patch.object(sweep, "rebase_workspace_paths", side_effect=list), mock.patch.object(sweep, "run", side_effect=self.fake_run), mock.patch.object(sweep, "score_evaluation_artifact", return_value=self.valid_loss()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sweep.main(args), 0)
            report = json.loads((output / "viscosity_sweep.json").read_text())
            self.assertEqual(report["fixed_parameters"]["youngs_modulus_pa"], 4500.0)
            self.assertEqual(report["fixed_parameters"]["backend_request"], "cpu")
            self.assertEqual(len(report["cases"]), 5)
            self.assertTrue(all(case["status"] == "complete" for case in report["cases"]))

    def test_resume_rejects_altered_output_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline = self.input_fixture(temporary)
            root = Path(temporary) / "sweep"
            result, _ = self.run_case(root, baseline)
            (Path(result["simulation_dir"]) / "camera_parameters.json").write_text("changed output\n")
            with self.assertRaisesRegex(ValueError, "Completed case output differs"):
                self.run_case(root, baseline, resume=True)

    def test_validate_only_checks_inputs_and_expanded_commands_without_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline = self.input_fixture(temporary)
            source = Path(temporary) / "command.json"
            source.write_text(json.dumps(baseline))
            output = Path(temporary) / "unused-output"
            args = ["--source-command", str(source), "--output-dir", str(output), "--tool-contact-friction", "0.2", "--youngs-modulus", "4500", "--gpu", "--validate-only"]
            with mock.patch.object(sweep, "rebase_workspace_paths", side_effect=list), mock.patch.object(sweep, "run") as run_mock, contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(sweep.main(args), 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["youngs_modulus_pa"], 4500.0)
            self.assertEqual(report["backend_request"], "gpu")
            self.assertEqual(len(report["commands"]), 5)
            for command in report["commands"]:
                argv = command["simulator"]
                self.assertNotIn("--cpu", argv)
                self.assertEqual(argv[argv.index("--youngs-modulus") + 1], "4500")
            self.assertFalse(output.exists())
            run_mock.assert_not_called()
            Path(baseline[baseline.index("--initial-particles") + 1]).unlink()
            with mock.patch.object(sweep, "rebase_workspace_paths", side_effect=list), self.assertRaises(FileNotFoundError):
                sweep.main(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
