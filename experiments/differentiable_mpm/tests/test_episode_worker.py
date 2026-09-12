"""Host-only protocol, provenance and failure checks for isolated episode workers."""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.differentiable_mpm import episode_worker as worker
from experiments.differentiable_mpm.config import FrameWindow
from experiments.differentiable_mpm.results import RunStore, canonical_hash
from experiments.differentiable_mpm.state import DEFAULT_PARAMETERS, InvalidStateError, SimulationConfig
from experiments.differentiable_mpm.tests.helpers import temporary_directory


class EpisodeWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = temporary_directory(prefix="episode-worker-")
        self.directory = Path(self.temporary.name)
        self.source = {"sources": {"solver.py": "source-a"}, "dependencies": {}}
        self.parameters = dict(DEFAULT_PARAMETERS, tool_retention=1.0)
        shared = {key: self.parameters[key] for key in
                  ("youngs_modulus", "poisson_ratio", "viscosity", "plastic_min", "plastic_max")}
        self.request = {"schema": worker.REQUEST_SCHEMA, "request_id": "request-1", "episode_id": "episode-a",
                        "action": "objective", "dataset_path": str(self.directory / "dataset.json"),
                        "dataset_options": {}, "parameters": shared, "compute_grad": True,
                        "reference_policy": "frozen", "cpu_threads": 1, "ignore_recompute_mismatch": False,
                        "expected_source": self.source, "expected_dataset_fingerprint": "a" * 64,
                        "expected_prepared_fingerprint": "b" * 64}
        self.config = SimpleNamespace(parameters=dict(self.parameters), simulation={"physics_version": "corrected-v1"},
                                      backend="cpu", seed=7, segment_length=4, tool_sdf_resolution=16)
        self.episode = SimpleNamespace(id="episode-a", config=self.config, membership="validation",
                                       scored_window=FrameWindow(1, 8, 2),
                                       parameters_for=lambda shared_values: {**self.parameters, **shared_values})
        self.dataset = SimpleNamespace(fingerprint="a" * 64, episodes=(self.episode,))
        self.sim = SimulationConfig(n_particles=2, grid=8)
        self.prepared = SimpleNamespace(fingerprint="b" * 64, simulation_config=self.sim, total_steps=8,
                                       summary=lambda: {"provenance": {"scored_frames": [1, 3, 5, 7, 8]},
                                                        "observation_count": 5, "total_steps": 8,
                                                        "scored_frames": [1, 3, 5, 7, 8]})
        self.evaluation = SimpleNamespace(value=2.5, gradient={key: 0.25 for key in self.parameters},
                                          frames=[], diagnostics={"replay_consistent": True}, initial_gradient=None)
        self.rollout = Mock()
        self.rollout.value_and_gradient.return_value = self.evaluation

    def tearDown(self):
        self.temporary.cleanup()

    def environment(self):
        stack = ExitStack()
        mocks = {}
        for name, value in {
            "source_identity": self.source, "_load_dataset": self.dataset,
            "verify_reference": {}, "reference_identity": {"simulator_sha256": "c" * 64},
            "reference_policy": nullcontext(),
            "init_runtime": {"actual_arch": "Arch.x64", "backend": "cpu", "precision": "f32",
                             "initialization_verified": True},
            "prepare_experiment": self.prepared, "make_rollout": (object(), self.rollout),
            "export_and_evaluate": {"strict_evaluation": {"valid": True}},
        }.items():
            mocks[name] = stack.enter_context(patch.object(worker, name, return_value=value))
        stack.enter_context(patch.object(worker, "RunStore", side_effect=lambda path, identity:
                                        RunStore(path, identity, allowed_root=self.directory)))
        return stack, mocks

    def execute(self, request=None, name="run"):
        request = self.request if request is None else request
        output = self.directory / name
        code = worker.run_request(request, output)
        return code, json.loads((output / "result.json").read_text())

    def test_objective_identity_gradient_and_effective_parameters(self):
        with self.environment()[0]:
            code, result = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(result["schema"], worker.RESULT_SCHEMA)
        self.assertEqual(result["request_sha256"], canonical_hash(self.request))
        self.assertEqual(result["parameters"], self.parameters)
        self.assertEqual(result["evaluation"]["gradient"], self.evaluation.gradient)
        self.assertEqual(result["prepared_fingerprint"], "b" * 64)
        self.assertTrue(result["runtime"]["forward_verified"])
        self.assertTrue(result["runtime"]["backward_verified"])
        self.assertEqual(result["source_after"], self.source)
        self.rollout.value_and_gradient.assert_called_once_with(self.parameters, compute_grad=True)

    def test_candidate_does_not_change_preparation_and_whole_episode_window(self):
        stack, mocks = self.environment()
        candidate = deepcopy(self.request)
        candidate["parameters"]["viscosity"] = 22.0
        with stack:
            code, result = self.execute(candidate)
        self.assertEqual(code, 0)
        self.assertEqual(self.config.parameters["viscosity"], DEFAULT_PARAMETERS["viscosity"])
        self.assertEqual(result["parameters"]["viscosity"], 22.0)
        mocks["prepare_experiment"].assert_called_once_with(self.config, split="validation", build_sdf=True,
                                                           scored_window=self.episode.scored_window)

    def test_validation_initializes_runtime_and_reports_memory_without_rollout(self):
        request = dict(self.request, action="validate", compute_grad=False, expected_prepared_fingerprint=None)
        stack, mocks = self.environment()
        with stack:
            code, result = self.execute(request)
        self.assertEqual(code, 0)
        mocks["init_runtime"].assert_called_once()
        mocks["make_rollout"].assert_not_called()
        self.assertEqual(result["prepared_summary"]["observation_count"], 5)
        self.assertIn("memory_estimate", result["prepared_summary"])

    def test_no_runtime_validation_does_not_claim_execution(self):
        request = dict(self.request, action="validate", compute_grad=False,
                       expected_prepared_fingerprint=None, no_runtime=True)
        stack, mocks = self.environment()
        with stack:
            code, result = self.execute(request)
        self.assertEqual(code, 0)
        mocks["init_runtime"].assert_not_called()
        self.assertFalse(result["runtime"]["initialization_verified"])
        self.assertNotIn("actual_arch", result["runtime"])

    def test_identity_mismatches_are_errors_before_simulation(self):
        for key, value in (("expected_source", {"sources": {}}),
                           ("expected_dataset_fingerprint", "d" * 64),
                           ("expected_prepared_fingerprint", "e" * 64)):
            stack, mocks = self.environment()
            with stack:
                code, result = self.execute(dict(self.request, **{key: value}), name=key)
            self.assertEqual(code, 1)
            self.assertEqual(result["status"], "error")
            mocks["make_rollout"].assert_not_called()

    def test_source_change_overrides_success_or_invalidity(self):
        for index, invalid in enumerate((False, True)):
            stack, mocks = self.environment()
            mocks["source_identity"].side_effect = [self.source, {"sources": {"solver.py": "source-b"}}]
            self.rollout.value_and_gradient.side_effect = InvalidStateError("unstable") if invalid else None
            with stack:
                code, result = self.execute(name=f"changed-{index}")
            self.assertEqual(code, 1)
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["error_type"], "SourceChangedError")

    def test_numerical_invalidity_and_infrastructure_errors_are_distinct(self):
        for index, error in enumerate((InvalidStateError("bad deformation"), RuntimeError("worker runtime failed"))):
            self.rollout.value_and_gradient.side_effect = error
            with self.environment()[0]:
                code, result = self.execute(name=f"failure-{index}")
            self.assertEqual(code, 2 if index == 0 else 1)
            self.assertEqual(result["status"], "invalid" if index == 0 else "error")
            self.assertEqual(result["error"], str(error))
            self.assertNotIn("evaluation", result)

    def test_forward_objective_and_mismatch_policy(self):
        request = dict(self.request, compute_grad=False, ignore_recompute_mismatch=True)
        self.evaluation.gradient = None
        stack, mocks = self.environment()
        with stack:
            code, result = self.execute(request)
        self.assertEqual(code, 0)
        self.assertFalse(result["runtime"]["backward_verified"])
        self.assertIsNone(result["evaluation"]["gradient"])
        self.assertTrue(mocks["make_rollout"].call_args.kwargs["ignore_recompute_mismatch"])

    def test_replay_and_strict_evaluation_use_explicit_selected_parameters(self):
        for action in ("replay", "evaluate"):
            request = dict(self.request, action=action, compute_grad=False)
            stack, mocks = self.environment()
            mocks["export_and_evaluate"].return_value = {"strict_evaluation": {"valid": False}}
            with stack:
                code, result = self.execute(request, name=action)
            self.assertEqual(code, 0)
            self.assertFalse(result["export"]["strict_evaluation"]["valid"])
            self.assertEqual(mocks["export_and_evaluate"].call_args.args[2], self.parameters)
            self.assertEqual(mocks["export_and_evaluate"].call_args.kwargs["strict"], action == "evaluate")
            self.rollout.value_and_gradient.assert_not_called()
            mocks["make_rollout"].assert_not_called()
            self.assertIsNone(mocks["export_and_evaluate"].call_args.args[1])

    def test_request_validation_and_invalid_json(self):
        changes = ({"compute_grad": 1}, {"no_runtime": True}, {"unknown": 1},
                   {"expected_prepared_fingerprint": None}, {"action": "fit"},
                   {"cpu_threads": True}, {"reference_policy": "unchecked"})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                worker.validate_request(dict(self.request, **change))
        for index, text in enumerate(('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}')):
            path = self.directory / f"bad-{index}.json"
            path.write_text(text)
            with self.assertRaises(ValueError):
                worker.load_request(path)

    def test_dataset_loader_receives_exact_declared_options(self):
        options = {"backend": "cpu", "precision": "f64", "physics_version": "corrected-v1",
                   "p2g_mode": "serial", "segment_length": 4,
                   "path_overrides": {"episode-a": {"episode": "/recorded/episode-a"}}}
        with patch("experiments.differentiable_mpm.dataset_config.load_dataset", return_value=self.dataset) as load:
            result = worker._load_dataset(dict(self.request, dataset_options=options))
        self.assertIs(result, self.dataset)
        load.assert_called_once_with(self.request["dataset_path"], **options)

    def test_missing_episode_fails_before_preparation(self):
        stack, mocks = self.environment()
        with stack:
            code, result = self.execute(dict(self.request, episode_id="missing"))
        self.assertEqual(code, 1)
        mocks["prepare_experiment"].assert_not_called()
        self.assertEqual(result["status"], "error")


if __name__ == "__main__":
    unittest.main()
