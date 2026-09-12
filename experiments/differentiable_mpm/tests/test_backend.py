"""Backend-report and failure-handling tests; no hardware initialization here."""

from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.differentiable_mpm import backend_check
from experiments.differentiable_mpm.tests.helpers import temporary_directory


def _valid_result():
    state = {"x": [[.4, .41, .42]], "v": [[.01, -.02, .03]],
             "C": np.eye(3)[None].tolist(), "F": np.eye(3)[None].tolist(), "Jp": [1.0]}
    return {"status": "execution_passed", "precision": "f32", "fixture_version": "fixture",
            "source_hashes": {"solver.py": "test_hash"}, "configuration": {"grid": 12},
            "parameters": {"youngs_modulus": 4000.0, "viscosity": 8.0},
            "forward_state": state, "initial_state_adjoint": deepcopy(state),
            "branch_counts": [{"yielded": 1, "particle_floor": 1}],
            "observation": {"value": 0.8, "position_gradient": [[1.0, -2.0, 3.0]]},
            "parameter_gradients": {"youngs_modulus": .001, "viscosity": .1}}


class BackendReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = temporary_directory(prefix="backend-tests-")
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_identical_results_pass(self):
        result = _valid_result()
        comparison = backend_check.compare_results(result, deepcopy(result))
        self.assertTrue(comparison["passed"])
        self.assertTrue(all(check["passed"] for check in comparison["checks"]))

    def test_wrong_gradient_or_discrete_decision_fails(self):
        reference = _valid_result()
        candidate = deepcopy(reference)
        candidate["parameter_gradients"]["youngs_modulus"] *= -1
        comparison = backend_check.compare_results(reference, candidate)
        self.assertFalse(comparison["passed"])
        failed = [check["name"] for check in comparison["checks"] if not check["passed"]]
        self.assertEqual(failed, ["scaled_parameter_gradient_youngs_modulus"])
        candidate = deepcopy(reference)
        candidate["branch_counts"][0]["yielded"] = 2
        self.assertFalse(backend_check.compare_results(reference, candidate)["passed"])

    def test_nonfinite_values_or_wrong_dimensions_fail(self):
        reference = _valid_result()
        for value in ([[float("nan"), 0, 0]], [[0, 0]]):
            candidate = deepcopy(reference)
            candidate["forward_state"]["x"] = value
            self.assertFalse(backend_check.compare_results(reference, candidate)["passed"])

    def test_mismatched_execution_identity_fails(self):
        reference = _valid_result()
        changes = {"status": "failed", "precision": "f64", "fixture_version": "other",
                   "source_hashes": {"solver.py": "different"}, "configuration": {"grid": 48}}
        for name, value in changes.items():
            candidate = deepcopy(reference)
            candidate[name] = value
            comparison = backend_check.compare_results(reference, candidate)
            self.assertFalse(comparison["passed"])
            self.assertEqual(comparison["checks"], [])

    def test_p2g_mode_reaches_both_workers_and_fixture(self):
        root = self.directory / "runs"
        for mode in ("atomic", "serial"):
            with self.subTest(mode=mode), patch.object(backend_check, "EVIDENCE_ROOT", root), \
                    patch.object(backend_check, "_launch", return_value=_valid_result()) as launch:
                arguments = ["--backend", "cuda", "--compare-cpu", "--output-dir", str(root / mode)]
                if mode == "serial":
                    arguments += ["--p2g-mode", mode]
                self.assertEqual(backend_check.main(arguments), 0)
                self.assertEqual([call.args[0] for call in launch.call_args_list], ["cuda", "cpu"])
                self.assertTrue(all(call.args[4] == mode for call in launch.call_args_list))
                result = json.loads((root / mode / "backend_check.json").read_text())
                self.assertEqual(result["p2g_mode"], mode)
                self.assertEqual(backend_check._fixture("f32", mode)[0].p2g_mode, mode)
        with patch.object(backend_check, "_worker", return_value=0) as worker:
            self.assertEqual(backend_check.main(["--backend", "cuda", "--p2g-mode", "serial",
                                                 "--_worker-report", "test.json"]), 0)
            worker.assert_called_once_with("cuda", "f32", "test.json", "serial", "corrected-v1")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            backend_check.main(["--backend", "cuda", "--p2g-mode", "parallel"])

    def test_physics_version_reaches_workers_fixture_and_failure_record(self):
        root = self.directory / 'physics_runs'
        for version in ('corrected-v1', 'legacy-v1'):
            flags = [] if version == 'corrected-v1' else ['--physics-version', version]
            with patch.object(backend_check, 'EVIDENCE_ROOT', root), \
                    patch.object(backend_check, '_launch', return_value=_valid_result()) as launch:
                self.assertEqual(backend_check.main(['--backend', 'cuda', '--compare-cpu',
                    '--output-dir', str(root / version), *flags]), 0)
                self.assertEqual([call.args[5] for call in launch.call_args_list], [version, version])
            self.assertEqual(backend_check._fixture('f32', physics_version=version)[0].physics_version, version)
            directory = self.directory / version
            directory.mkdir()
            with patch.object(backend_check.subprocess, 'run', return_value=SimpleNamespace(returncode=-11)) as run:
                report = backend_check._launch('cuda', 'f32', directory, 1, physics_version=version)
            argv = run.call_args.args[0]
            self.assertEqual(argv[argv.index('--physics-version') + 1], version)
            self.assertEqual(report['physics_version'], version)
        with patch.object(backend_check, '_worker', return_value=0) as worker:
            backend_check.main(['--backend', 'cuda', '--physics-version', 'legacy-v1', '--_worker-report', 'p.json'])
            worker.assert_called_once_with('cuda', 'f32', 'p.json', 'atomic', 'legacy-v1')
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            backend_check.main(['--backend', 'cpu', '--physics-version', 'unknown'])
        reference, candidate = _valid_result(), _valid_result()
        reference['configuration']['physics_version'] = 'corrected-v1'
        candidate['configuration']['physics_version'] = 'legacy-v1'
        self.assertFalse(backend_check.compare_results(reference, candidate)['passed'])

    def test_serial_mode_reaches_subprocess_and_failure_record(self):
        directory = self.directory / "serial_process"
        directory.mkdir()
        with patch.object(backend_check.subprocess, "run", return_value=SimpleNamespace(returncode=-11)) as run:
            result = backend_check._launch("cuda", "f32", directory, 1, "serial")
        argv = run.call_args.args[0]
        self.assertEqual(argv[argv.index("--p2g-mode") + 1], "serial")
        self.assertEqual(result["p2g_mode"], "serial")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(json.loads((directory / "command.json").read_text())["argv"], argv)

    def test_different_p2g_modes_refuse_comparison(self):
        reference = _valid_result()
        candidate = deepcopy(reference)
        reference["configuration"]["p2g_mode"] = "atomic"
        candidate["configuration"]["p2g_mode"] = "serial"
        comparison = backend_check.compare_results(reference, candidate)
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["checks"], [])

    def test_evidence_refuses_existing_or_external_directory(self):
        root = self.directory / "runs"
        with patch.object(backend_check, "EVIDENCE_ROOT", root):
            selected = backend_check.create_evidence_directory(root / "new")
            self.assertEqual(selected, root / "new")
            with self.assertRaises(FileExistsError):
                backend_check.create_evidence_directory(root / "new")
            with self.assertRaises(ValueError):
                backend_check.create_evidence_directory(self.directory / "outside")

    def test_process_failure_preserves_requested_backend(self):
        directory = self.directory / "failed_cuda"
        directory.mkdir()
        with patch.object(backend_check.subprocess, "run", return_value=SimpleNamespace(returncode=-11)) as run:
            result = backend_check._launch("cuda", "f32", directory, 1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["requested_backend"], "cuda")
        self.assertEqual(result["process_failure"]["returncode"], -11)
        argv = run.call_args.args[0]
        self.assertEqual(argv[argv.index("--backend") + 1], "cuda")
        self.assertEqual(run.call_count, 1)
        self.assertFalse(result.get("runtime", {}).get("initialization_verified", False))
        stored = json.loads((directory / "result.json").read_text())
        self.assertEqual(stored["status"], "failed")

    def test_failed_requested_backend_does_not_start_cpu_comparison(self):
        root = self.directory / "runs"
        failed = {"status": "failed", "requested_backend": "cuda", "precision": "f32"}
        with patch.object(backend_check, "EVIDENCE_ROOT", root), \
                patch.object(backend_check, "_launch", return_value=failed) as launch:
            code = backend_check.main(["--backend", "cuda", "--compare-cpu", "--output-dir", str(root / "failure")])
        self.assertEqual(code, 2)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(launch.call_args.args[0], "cuda")
        result = json.loads((root / "failure" / "backend_check.json").read_text())
        self.assertIsNone(result["cpu_reference"])
        self.assertFalse(result["execution_verified"])
        self.assertIn("not started", result["cpu_comparison"]["reason"])

    def test_hardware_execution_requires_increasing_process_counters(self):
        before = [{"driver": "amdgpu", "pci_device": "0000:75:00.0", "client_id": "42",
                   "engine_time_ns": {"compute": 100}}]
        self.assertFalse(backend_check.drm_activity(before, deepcopy(before))["verified"])
        self.assertFalse(backend_check.drm_activity([], [])["verified"])
        after = deepcopy(before)
        after[0]["engine_time_ns"]["compute"] = 200
        activity = backend_check.drm_activity(before, after)
        self.assertTrue(activity["verified"])
        self.assertEqual(activity["active_clients"][0]["engine_delta_ns"]["compute"], 100)

    def test_fp64_comparison_uses_its_tighter_tolerances(self):
        reference = _valid_result()
        candidate = deepcopy(reference)
        candidate["parameter_gradients"]["youngs_modulus"] *= 1.005
        self.assertTrue(backend_check.compare_results(reference, candidate)["passed"])
        reference["precision"] = candidate["precision"] = "f64"
        self.assertFalse(backend_check.compare_results(reference, candidate)["passed"])

    def test_timeout_is_not_reported_as_success(self):
        directory = self.directory / "timed_out_vulkan"
        directory.mkdir()
        failure = subprocess.TimeoutExpired(cmd="qualify", timeout=1)
        with patch.object(backend_check.subprocess, "run", side_effect=failure):
            result = backend_check._launch("vulkan", "f32", directory, 1)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["process_failure"]["timed_out"])
        self.assertIsNone(result["process_failure"]["returncode"])


if __name__ == "__main__":
    unittest.main()
