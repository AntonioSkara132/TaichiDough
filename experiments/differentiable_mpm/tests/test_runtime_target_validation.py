"""Host-only checks for paper-target validation before numerical runtime startup."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import Mock

import numpy as np

from experiments.differentiable_mpm.loss_options import parse_loss_config
from experiments.differentiable_mpm.loss_targets import required_target_paths
from experiments.differentiable_mpm.tests import test_calibrate_cli as cli_fixture
from experiments.differentiable_mpm.tests import test_episode_worker as worker_fixture
from experiments.differentiable_mpm.tests import test_loss_options_integration as integration_fixture


class RuntimeTargetValidationTests(integration_fixture.HostOnlyTests):
    modes = ({}, {"version": "partial-visible-splats-v1"},
             {"version": "dpsi-pcd-cd-v1"}, {"version": "dpsi-pcd-emd-v1"},
             {"version": "dpsi-prt-cd-v1"}, {"version": "dpsi-prt-emd-v1"},
             {"version": "empm-offline-v1"}, {"version": "empm-mask-inspired-v1"})

    def single_episode(self, loss):
        fixture = cli_fixture.CalibrateCliTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        config = deepcopy(fixture.config)
        config.loss = loss
        for name in required_target_paths(parse_loss_config(loss)):
            config.paths[name] = fixture.root / name
            config.expected_sha256[name] = "a" * 64
        return fixture, config

    def episode_worker(self, loss):
        fixture = worker_fixture.EpisodeWorkerTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        if loss:
            fixture.config.loss = loss
        return fixture

    def calls(self, mocks):
        recorded = Mock()
        for name in ("prepare_experiment", "init_runtime", "make_rollout"):
            recorded.attach_mock(mocks[name], name)
        return recorded

    def test_single_episode_prepares_once_in_mode_specific_order(self):
        for loss in self.modes:
            with self.subTest(loss=loss):
                fixture, config = self.single_episode(loss)
                with fixture.environment(config=config) as environment:
                    calls = self.calls(environment.mocks)
                    status = fixture.run_cli(["gradient", "--output-dir", str(fixture.root / "gradient")])
                    self.assertEqual(status, 0)
                    self.assertEqual(environment.prepared_calls, [("training", 3, True)])
                    self.assertEqual(environment.mocks["prepare_experiment"].call_count, 1)
                    prepared = environment.mocks["make_rollout"].call_args.args[0]
                    self.assertIs(prepared.config, environment.mocks["prepare_experiment"].call_args.args[0])
                order = [call[0] for call in calls.mock_calls]
                expected = ["prepare_experiment", "init_runtime"] if loss.get("version", "").startswith(
                    ("dpsi-", "empm-")) else ["init_runtime", "prepare_experiment"]
                self.assertEqual(order, expected + ["make_rollout"])

    def test_worker_prepares_once_in_mode_specific_order(self):
        for loss in self.modes:
            with self.subTest(loss=loss):
                fixture = self.episode_worker(loss)
                stack, mocks = fixture.environment()
                calls = self.calls(mocks)
                with stack:
                    status, result = fixture.execute()
                self.assertEqual(status, 0)
                self.assertTrue(result["runtime"]["initialization_verified"])
                mocks["prepare_experiment"].assert_called_once_with(
                    fixture.config, split=fixture.episode.membership, build_sdf=True,
                    scored_window=fixture.episode.scored_window)
                self.assertIs(mocks["make_rollout"].call_args.args[0], fixture.prepared)
                order = [call[0] for call in calls.mock_calls]
                expected = ["prepare_experiment", "init_runtime"] if loss.get("version", "").startswith(
                    ("dpsi-", "empm-")) else ["init_runtime", "prepare_experiment"]
                self.assertEqual(order, expected + ["make_rollout"])

    def test_single_episode_no_runtime_prepares_once_without_execution(self):
        for loss in self.modes:
            with self.subTest(loss=loss):
                fixture, config = self.single_episode(loss)
                output = fixture.root / "validate"
                with fixture.environment(config=config) as environment:
                    calls = self.calls(environment.mocks)
                    self.assertEqual(fixture.run_cli(["validate", "--no-runtime", "--output-dir", str(output)]), 0)
                self.assertEqual([call[0] for call in calls.mock_calls], ["prepare_experiment"])
                result = json.loads((output / "result.json").read_text())
                self.assertFalse(result["runtime"]["initialization_verified"])
                self.assertEqual(result["status"], "inputs_valid")

    def test_worker_no_runtime_prepares_once_without_execution(self):
        for loss in self.modes:
            with self.subTest(loss=loss):
                fixture = self.episode_worker(loss)
                request = dict(fixture.request, action="validate", compute_grad=False,
                               no_runtime=True, expected_prepared_fingerprint=None)
                stack, mocks = fixture.environment()
                calls = self.calls(mocks)
                with stack:
                    status, result = fixture.execute(request)
                self.assertEqual(status, 0)
                self.assertEqual([call[0] for call in calls.mock_calls], ["prepare_experiment"])
                self.assertFalse(result["runtime"]["initialization_verified"])
                self.assertNotIn("actual_arch", result["runtime"])

    def invalid_targets(self, kind):
        fixture = integration_fixture.ObservationPreparationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        endpoint, loss = 1, {"version": "dpsi-prt-cd-v1"}
        if kind == "assignment_budget":
            loss = {"version": "dpsi-pcd-emd-v1", "max_assignment_pairs": 1}
            message = "Assignment allocation refused"
        else:
            archive = fixture.write_external_volume()
            if kind == "hash":
                archive.write_bytes(archive.read_bytes() + b"changed")
                message = "SHA-256 mismatch"
            elif kind == "metadata":
                path = fixture.config.paths["loss_point_targets_metadata"]
                metadata = json.loads(path.read_text())
                metadata["sequence_fingerprint"] = "9" * 64
                path.write_text(json.dumps(metadata))
                fixture.config.expected_sha256["loss_point_targets_metadata"] = hashlib.sha256(path.read_bytes()).hexdigest()
                message = "sequence_fingerprint differs"
            elif kind == "nonfinite":
                with np.load(archive, allow_pickle=False) as saved:
                    arrays = {name: saved[name].copy() for name in saved.files}
                arrays["points"][0, 0] = np.nan
                np.savez_compressed(archive, **arrays)
                fixture.config.expected_sha256["loss_point_targets"] = hashlib.sha256(archive.read_bytes()).hexdigest()
                message = "finite"
            else:
                self.assertEqual(kind, "missing_frame")
                endpoint, message = 2, "missing scored frames"

        def prepare(config, **kwargs):
            return fixture.prepare(config.loss, endpoint=endpoint, scored=(endpoint,))
        return loss, prepare, message

    def test_single_episode_real_target_failures_precede_runtime(self):
        for kind in ("hash", "metadata", "nonfinite", "missing_frame", "assignment_budget"):
            for no_runtime in (False, True):
                with self.subTest(kind=kind, no_runtime=no_runtime):
                    loss, prepare, message = self.invalid_targets(kind)
                    fixture, config = self.single_episode(loss)
                    with fixture.environment(config=config) as environment:
                        environment.mocks["prepare_experiment"].side_effect = prepare
                        args = ["validate"] + (["--no-runtime"] if no_runtime else [])
                        with self.assertRaisesRegex(ValueError, message):
                            fixture.run_cli(args)
                        environment.mocks["prepare_experiment"].assert_called_once()
                        environment.mocks["init_runtime"].assert_not_called()
                        environment.mocks["make_rollout"].assert_not_called()
                        self.assertEqual(environment.stores, [])

    def test_worker_real_target_failures_precede_runtime(self):
        for kind in ("hash", "metadata", "nonfinite", "missing_frame", "assignment_budget"):
            for no_runtime in (False, True):
                with self.subTest(kind=kind, no_runtime=no_runtime):
                    loss, prepare, message = self.invalid_targets(kind)
                    fixture = self.episode_worker(loss)
                    request = dict(fixture.request, action="validate", compute_grad=False,
                                   no_runtime=no_runtime, expected_prepared_fingerprint=None)
                    stack, mocks = fixture.environment()
                    mocks["prepare_experiment"].side_effect = prepare
                    with stack:
                        status, result = fixture.execute(request)
                    self.assertEqual(status, 1)
                    self.assertEqual(result["status"], "error")
                    self.assertEqual(result["error_type"], "ValueError")
                    self.assertRegex(result["error"], message)
                    self.assertIsNone(result["runtime"])
                    self.assertIsNone(result["prepared_fingerprint"])
                    self.assertNotIn("evaluation", result)
                    mocks["prepare_experiment"].assert_called_once()
                    mocks["init_runtime"].assert_not_called()
                    mocks["make_rollout"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
