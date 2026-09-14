"""Host-only tests for sequential Cartesian dataset calibration attempts."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import signal
import unittest

from experiments.differentiable_mpm import calibrate_dataset_attempts as cli
from experiments.differentiable_mpm.config import REQUIRED_PATHS, SCHEMA as EPISODE_SCHEMA

MATERIAL_NAMES = ("youngs_modulus", "poisson_ratio", "viscosity", "plastic_min", "plastic_max")
DATASET_SCHEMA = "taichidough/differentiable-dataset/v1"
from experiments.differentiable_mpm.results import RunStore, SCHEMA as RUN_SCHEMA, canonical_hash
from experiments.differentiable_mpm.state import DEFAULT_PARAMETERS
from experiments.differentiable_mpm.tests.helpers import temporary_directory


class AttemptLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = temporary_directory(prefix="attempt-launcher-")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.dataset_path = self.root / "dataset.json"
        config_path = self.root / "episode.json"
        initial = {name: DEFAULT_PARAMETERS[name] for name in MATERIAL_NAMES}
        initial.update(youngs_modulus=6000.0, poisson_ratio=0.47, viscosity=0.0,
                       plastic_min=0.9, plastic_max=1.1)
        config = {
            "schema": EPISODE_SCHEMA,
            "name": "episode",
            "paths": {name: str(self.root / "inputs" / name) for name in REQUIRED_PATHS},
            "expected_sha256": {},
            "simulation": {"n_particles": 1, "grid": 8, "precision": "f64",
                           "plasticity": "stretch-clamp", "physics_version": "corrected-v1"},
            "parameters": dict(DEFAULT_PARAMETERS, **initial),
            "fit_parameters": list(MATERIAL_NAMES),
            "parameter_bounds": {
                "youngs_modulus": [2000.0, 60000.0], "poisson_ratio": [0.45, 0.49],
                "viscosity": [0.0, 60.0], "plastic_min": [0.7, 0.999],
                "plastic_max": [1.001, 1.3],
            },
            "training": {"start_frame": 1, "end_frame": 2},
            "validation": {"start_frame": 3, "end_frame": 4},
            "optimizer": {}, "segment_length": 1,
        }
        config_path.write_text(json.dumps(config))
        self.dataset_path.write_text(json.dumps({
            "schema": DATASET_SCHEMA, "name": "attempt-fixture",
            "shared_parameters": {"initial": initial, "fit": list(MATERIAL_NAMES),
                                  "bounds": config["parameter_bounds"]},
            "episodes": [{"id": "train", "config": str(config_path), "membership": "training",
                          "scored_window": {"start_frame": 1, "end_frame": 2}}],
        }))
        self.source = {"fixture": "source-v1"}
        self.calls = []

    def args(self, output=None, *extra):
        values = ["--dataset", str(self.dataset_path),
                  "--initial-youngs-modulus", "3000", "--initial-youngs-modulus", "5000",
                  "--initial-viscosity", "1", "--initial-viscosity", "4"]
        if output is not None:
            values.extend(("--output-dir", str(output)))
        values.extend(extra)
        return cli.parse_args(values)

    @staticmethod
    def _value(command, option):
        return command[command.index(option) + 1]

    def write_success(self, command, loss):
        calibration = Path(self._value(command, "--output-dir"))
        calibration.mkdir(parents=True, exist_ok=True)
        dataset = cli._load_dataset_host_only(self._value(command, "--dataset"), initial_overrides={
            "youngs_modulus": float(self._value(command, "--initial-youngs-modulus")),
            "viscosity": float(self._value(command, "--initial-viscosity")),
        })
        dataset_identity = {"dataset": dataset.as_dict(), "dataset_fingerprint": dataset.fingerprint}
        identity = {"action": "dataset-fit", "dataset_identity": dataset_identity, "optimizer": {}}
        identity_hash = canonical_hash(identity)
        (calibration / "run_manifest.json").write_text(json.dumps({
            "schema": RUN_SCHEMA, "identity": identity, "identity_sha256": identity_hash,
        }))
        values = dict(dataset.shared_initial)
        selection = {
            "schema": "taichidough/dataset-material-selection/v1", "identity_sha256": identity_hash,
            "dataset_identity": dataset_identity, "dataset_fingerprint": dataset.fingerprint,
            "shared_material_parameters": values, "training_value": loss,
        }
        (calibration / "selected_parameters.json").write_text(json.dumps(selection))
        result = {
            "status": "converged_parameters",
            "optimization": {"status": "converged_parameters", "best_value": loss},
            "shared_material_parameters": values,
            "independent_evaluation_skipped": False,
            "evaluations": {episode.id: {"strict_evaluation": {"valid": True}}
                            for episode in dataset.episodes},
        }
        (calibration / "result.json").write_text(json.dumps(result))

    def successful_runner(self, command, **kwargs):
        self.calls.append(list(command))
        Path(kwargs["log_path"]).write_text("concise child output\n")
        youngs = float(self._value(command, "--initial-youngs-modulus"))
        viscosity = float(self._value(command, "--initial-viscosity"))
        self.write_success(command, youngs / 10000 + viscosity / 100)
        return cli.ProcessOutcome(0)

    def execute(self, args, runner=None):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return cli.execute(args, process_runner=runner or self.successful_runner,
                               source_provider=lambda: self.source, repository_root=self.root,
                               store_factory=lambda path, identity, resume=False:
                               RunStore(path, identity, resume=resume, allowed_root=self.root))

    def test_validate_only_has_e_major_order_exact_commands_and_no_files(self):
        output = self.root / "validation-only"
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = cli.execute(self.args(output, "--validate-only", "--backend", "cpu",
                               "--parameter-stability-updates", "3"),
                               process_runner=lambda *args, **kwargs: self.fail("process launched"),
                               source_provider=lambda: self.source, repository_root=self.root)
        self.assertEqual(code, 0)
        self.assertFalse(output.exists())
        report = json.loads(stream.getvalue())
        self.assertEqual([(row["youngs_modulus"], row["viscosity"]) for row in report["attempts"]],
                         [(3000.0, 1.0), (3000.0, 4.0), (5000.0, 1.0), (5000.0, 4.0)])
        for row in report["attempts"]:
            self.assertEqual(row["argv"][:4], [str(Path(cli.sys.executable).absolute()), "-m",
                             "experiments.differentiable_mpm.calibrate_dataset", "fit"])
            self.assertIn("--fit-log", row["argv"])
            self.assertIn("--parameter-stability-updates", row["argv"])
        self.assertEqual(len({row["id"] for row in report["attempts"]}), 4)

    def test_explicit_python_symlink_is_preserved(self):
        selected = self.root / "selected-python"
        selected.symlink_to(Path(cli.sys.executable).resolve())
        plan = cli.build_plan(self.args(None, "--python", str(selected)), source_provider=lambda: self.source)
        self.assertEqual(plan["python"], selected.absolute())
        self.assertEqual(plan["attempts"][0]["ordinal"], 1)

    def test_invalid_grid_and_retry_arguments_are_rejected(self):
        cases = [
            ["--dataset", str(self.dataset_path), "--initial-youngs-modulus", "3000",
             "--initial-youngs-modulus", "3000", "--initial-viscosity", "1"],
            ["--dataset", str(self.dataset_path), "--initial-youngs-modulus", "1000",
             "--initial-viscosity", "1"],
            ["--dataset", str(self.dataset_path), "--initial-youngs-modulus", "3000",
             "--initial-viscosity", "nan"],
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                cli.build_plan(cli.parse_args(values), source_provider=lambda: self.source)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_args(["--dataset", str(self.dataset_path), "--initial-youngs-modulus", "3000",
                            "--initial-viscosity", "1", "--retry-failed"])

    def test_sequential_run_writes_incremental_summaries_and_verified_best(self):
        output = self.root / "successful"
        self.assertEqual(self.execute(self.args(output)), 0)
        self.assertEqual(len(self.calls), 4)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["status"], "completed")
        self.assertTrue(all(row["status"] == "completed" for row in summary["attempts"]))
        self.assertEqual([row["invocations"] for row in summary["attempts"]], [1, 1, 1, 1])
        best = json.loads((output / "best_attempt.json").read_text())
        self.assertEqual(best["attempt_id"], summary["attempts"][0]["id"])
        selected = Path(best["selected_parameters"])
        self.assertTrue(selected.is_file())
        self.assertEqual(cli._sha256(selected), best["selected_parameters_sha256"])
        self.assertEqual(len(list(output.glob("attempts/*/invocations/0001/started.json"))), 4)
        self.assertEqual(len(list(output.glob("attempts/*/invocations/0001/finished.json"))), 4)
        self.assertTrue((output / "summary.csv").is_file())

    def test_failed_attempt_is_preserved_and_retried_only_when_requested(self):
        output = self.root / "retry"
        count = 0

        def first_runner(command, **kwargs):
            nonlocal count
            count += 1
            self.calls.append(list(command))
            Path(kwargs["log_path"]).write_text("failed\n")
            if count == 1:
                calibration = Path(self._value(command, "--output-dir"))
                calibration.mkdir(parents=True)
                (calibration / "run_manifest.json").write_text("{}")
                return cli.ProcessOutcome(3)
            self.write_success(command, float(count))
            return cli.ProcessOutcome(0)

        self.assertEqual(self.execute(self.args(output), first_runner), 2)
        first_summary = json.loads((output / "summary.json").read_text())
        failed_id = first_summary["attempts"][0]["id"]
        self.calls.clear()
        self.assertEqual(self.execute(self.args(output, "--resume"), first_runner), 2)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.execute(self.args(output, "--resume", "--retry-failed"), first_runner), 0)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["attempts"][0]["invocations"], 2)
        retry_command = json.loads((output / "attempts" / failed_id / "invocations" / "0002" /
                                    "started.json").read_text())["argv"]
        self.assertNotIn("--resume", retry_command)
        self.assertIn("calibration_retry_0002", self._value(retry_command, "--output-dir"))
        self.assertTrue((output / "attempts" / failed_id / "calibration" / "run_manifest.json").is_file())
        self.assertTrue((output / "attempts" / failed_id / "invocations" / "0001" /
                         "finished.json").is_file())

    def test_interruption_stops_scheduling_and_pending_attempt_resumes(self):
        output = self.root / "interrupt"

        def interrupting(command, **kwargs):
            self.calls.append(list(command))
            Path(kwargs["log_path"]).write_text("interrupted\n")
            kwargs["interruption"].requested = True
            kwargs["interruption"].signum = signal.SIGINT
            return cli.ProcessOutcome(130, True, ("SIGINT",))

        self.assertEqual(self.execute(self.args(output), interrupting), 130)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual([row["status"] for row in summary["attempts"]],
                         ["interrupted", "pending", "pending", "pending"])
        self.calls.clear()
        self.assertEqual(self.execute(self.args(output, "--resume")), 0)
        self.assertEqual(len(self.calls), 4)
        resumed = json.loads((output / "summary.json").read_text())
        self.assertTrue(all(row["status"] == "completed" for row in resumed["attempts"]))

    def test_artifact_verification_accepts_completed_iteration_budget(self):
        args = self.args(self.root / "unused-budget")
        plan = cli.build_plan(args, source_provider=lambda: self.source)
        attempt = plan["attempts"][0]
        command = cli.child_command(plan, attempt, self.root / "budget-child")
        self.write_success(command, 0.5)
        result_path = self.root / "budget-child" / "result.json"
        result = json.loads(result_path.read_text())
        result["status"] = "budget_exhausted"
        result["optimization"]["status"] = "budget_exhausted"
        result_path.write_text(json.dumps(result))
        self.assertEqual(cli.verify_completed(self.root / "budget-child", attempt)["training_value"], 0.5)

    def test_artifact_verification_rejects_skipped_or_invalid_strict_evaluation(self):
        args = self.args(self.root / "unused")
        plan = cli.build_plan(args, source_provider=lambda: self.source)
        attempt = plan["attempts"][0]
        command = cli.child_command(plan, attempt, self.root / "child")
        self.write_success(command, 0.5)
        result_path = self.root / "child" / "result.json"
        result = json.loads(result_path.read_text())
        result["independent_evaluation_skipped"] = True
        result_path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "skipped"):
            cli.verify_completed(self.root / "child", attempt)
        result["independent_evaluation_skipped"] = False
        result["evaluations"]["train"]["strict_evaluation"]["valid"] = False
        result_path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "strict evaluation failed"):
            cli.verify_completed(self.root / "child", attempt)


if __name__ == "__main__":
    unittest.main()
