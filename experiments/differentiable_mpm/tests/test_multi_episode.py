"""Host-only multi-episode objective and subprocess-protocol regressions.

Run independently with:
    python3 -m unittest experiments.differentiable_mpm.tests.test_multi_episode -v
No Taichi runtime, simulator kernel, recording, or full calibration is executed.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

from experiments.differentiable_mpm import multi_episode as multi
from experiments.differentiable_mpm.dataset_config import MATERIAL_NAMES
from experiments.differentiable_mpm.optimize import AdamOptions, ProjectedAdam
from experiments.differentiable_mpm.parameters import PhysicalParameterSpace
from experiments.differentiable_mpm.state import DEFAULT_PARAMETERS, InvalidStateError


SOURCE = {"schema": "host-test-source", "files": {"fixture": "a" * 64}}
PREPARED = "b" * 64
SHARED = {name: DEFAULT_PARAMETERS[name] for name in MATERIAL_NAMES}


class FakeEpisode:
    def __init__(self, episode_id, weight=1.0, membership="training", frames=2, horizon=3):
        self.id, self.weight, self.membership = episode_id, weight, membership
        self.frames, self.horizon = frames, horizon
        self.config = SimpleNamespace(backend="cpu", simulation={"precision": "f32"}, seed=0,
                                      parameters=dict(DEFAULT_PARAMETERS, floor_retention=.37))

    def parameters_for(self, shared):
        return {**self.config.parameters, **shared, "tool_retention": 1.0}


def dataset(*episodes, fit=("youngs_modulus", "viscosity")):
    return SimpleNamespace(episodes=tuple(episodes), fit_parameters=fit, fingerprint="d" * 64)


def evaluation(value=4.0, youngs=2.0, viscosity=6.0, diagnostics=None):
    return {"value": value, "gradient": {"youngs_modulus": youngs, "viscosity": viscosity},
            "diagnostics": {} if diagnostics is None else diagnostics}


def runtime():
    return {"backend": "cpu", "precision": "f32", "cpu_threads": 1, "debug": False,
            "seed": 0, "actual_arch": "Arch.x64", "taichi_version": [1, 7, 4],
            "host": "host-fixture", "platform": "host-test", "machine": "x86_64",
            "CUDA_VISIBLE_DEVICES": None, "device_identity": "host-test CPU",
            "initialization_verified": True, "forward_verified": True, "backward_verified": True}


class HostOnlyTests(unittest.TestCase):
    def setUp(self):
        guard = patch("taichi.init", side_effect=AssertionError("Host tests must not initialize Taichi"))
        guard.start()
        self.addCleanup(guard.stop)


class WeightTests(HostOnlyTests):
    def test_episode_weights_are_normalized(self):
        np.testing.assert_array_equal(multi.normalized_weights([FakeEpisode("a", 2), FakeEpisode("b", 6)]), [.25, .75])

    def test_huge_finite_weights_do_not_overflow_the_sum(self):
        weights = multi.normalized_weights([FakeEpisode("a", 1e308), FakeEpisode("b", 1e308), FakeEpisode("c", 5e307)])
        np.testing.assert_allclose(weights, [.4, .4, .2], atol=0, rtol=1e-15)
        self.assertTrue(np.isfinite(weights).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_small_equal_subnormal_weights_stay_positive(self):
        tiny = np.nextafter(0.0, 1.0)
        np.testing.assert_array_equal(multi.normalized_weights([FakeEpisode("a", tiny), FakeEpisode("b", tiny)]), [.5, .5])

    def test_underflowing_relative_weight_is_rejected(self):
        with np.errstate(under="ignore"):
            with self.assertRaises(ValueError):
                multi.normalized_weights([FakeEpisode("a", 1e308), FakeEpisode("b", np.nextafter(0.0, 1.0))])

    def test_empty_and_nonpositive_weights_are_rejected(self):
        for episodes in ([], [FakeEpisode("a", 0)], [FakeEpisode("a", -1)]):
            with self.subTest(episodes=episodes), self.assertRaises(ValueError):
                multi.normalized_weights(episodes)

    def test_nonfinite_or_malformed_weights_are_protocol_errors(self):
        for value in (float("nan"), float("inf"), "1", True, None):
            with self.subTest(value=value), self.assertRaises(multi.EpisodeExecutionError):
                multi.normalized_weights([FakeEpisode("a", value)])


class DatasetObjectiveTests(HostOnlyTests):
    def test_weighted_episode_means_and_shared_gradients(self):
        episodes = [FakeEpisode("short", 2, frames=2, horizon=3), FakeEpisode("long", 6, frames=29, horizon=900)]
        records = {"short": evaluation(8, 1, 2), "long": evaluation(20, 5, 10)}
        for ep in episodes:
            records[ep.id]["diagnostics"] = {"observation_count": ep.frames, "total_steps": ep.horizon}
        obj = multi.DatasetObjective(dataset(*episodes), lambda ep, params, grad: deepcopy(records[ep.id]))
        result = obj(SHARED)
        self.assertEqual(result.value, 17.0)
        self.assertEqual(result.gradient, {"youngs_modulus": 4.0, "viscosity": 8.0})
        self.assertEqual([row["weighted_value"] for row in result.diagnostics["episodes"]], [2.0, 15.0])
        # The worker values are already episode means; neither horizon nor frame
        # count may divide them a second time or reweight them across episodes.
        self.assertEqual(result.diagnostics["episodes"][1]["diagnostics"]["observation_count"], 29)

    def test_every_episode_receives_identical_independent_candidate_dict(self):
        original = dict(SHARED)
        retained, seen = [], []

        def evaluate(ep, params, grad):
            retained.append(params)
            seen.append(deepcopy(params))
            params["youngs_modulus"] = -123
            params["local_only"] = ep.id
            return evaluation()

        obj = multi.DatasetObjective(dataset(FakeEpisode("a"), FakeEpisode("b")), evaluate)
        obj(original)
        self.assertEqual(seen, [SHARED, SHARED])
        self.assertIsNot(retained[0], retained[1])
        self.assertEqual(original, SHARED)

    def test_heldout_episode_is_excluded_from_calls_and_weight_normalization(self):
        episodes = [FakeEpisode("a", 2), FakeEpisode("heldout", 1e308, "validation"), FakeEpisode("b", 6)]
        calls = []

        def evaluate(ep, params, grad):
            calls.append(ep.id)
            if ep.membership != "training":
                raise AssertionError("Heldout data entered the training objective")
            return evaluation(8 if ep.id == "a" else 20)

        obj = multi.DatasetObjective(dataset(*episodes), evaluate)
        self.assertEqual(obj(SHARED).value, 17.0)
        self.assertEqual(calls, ["a", "b"])
        np.testing.assert_array_equal(obj.weights, [.25, .75])

    def test_no_training_episodes_is_rejected(self):
        with self.assertRaises(ValueError):
            multi.DatasetObjective(dataset(FakeEpisode("heldout", membership="validation")), Mock())

    def test_value_only_does_not_require_or_return_derivatives(self):
        calls = []

        def evaluate(ep, params, grad):
            calls.append(grad)
            return {"value": 3.0, "diagnostics": {}}

        obj = multi.DatasetObjective(dataset(FakeEpisode("a"), FakeEpisode("b")), evaluate)
        result = obj.value_and_gradient(SHARED, compute_grad=False)
        self.assertEqual(calls, [False, False])
        self.assertEqual(result.value, 3.0)
        self.assertEqual(result.gradient, {})
        self.assertFalse(result.diagnostics["replay_consistency_checked"])

    def test_last_episode_invalid_rejects_the_entire_candidate(self):
        events, calls = [], []

        def evaluate(ep, params, grad):
            calls.append(ep.id)
            if ep.id == "last":
                raise InvalidStateError("Nonfinite particle state in last episode")
            return evaluation(-1000)

        obj = multi.DatasetObjective(dataset(FakeEpisode("first"), FakeEpisode("last")), evaluate, event=events.append)
        with self.assertRaises(InvalidStateError):
            obj(SHARED)
        self.assertEqual(calls, ["first", "last"])
        self.assertEqual(obj.calls, 1)
        self.assertEqual(obj.successful_calls, 0)
        self.assertEqual(events[-1]["event"], "dataset_objective_failed")
        self.assertEqual(events[-1]["completed_episodes"], ["first"])
        self.assertNotIn("dataset_objective_finished", [e["event"] for e in events])

    def test_malformed_successful_evaluations_are_not_candidate_rejections(self):
        malformed = [None, [], {"gradient": {}}, evaluation(float("nan")), evaluation(float("inf")),
                     {"value": 1.0}, {"value": 1.0, "gradient": []},
                     {"value": 1.0, "gradient": {"youngs_modulus": 2.0}},
                     evaluation(viscosity=float("nan")), evaluation(youngs="2"), evaluation(True),
                     evaluation(diagnostics=[]), evaluation(diagnostics={"replay_consistent": "false"}),
                     evaluation(diagnostics={"recompute_mismatch_counts": {"x": True}})]
        for record in malformed:
            with self.subTest(record=record):
                obj = multi.DatasetObjective(dataset(FakeEpisode("a")), lambda *args: deepcopy(record))
                with self.assertRaises(multi.EpisodeExecutionError):
                    obj(SHARED)
                self.assertEqual(obj.successful_calls, 0)

    def test_replay_mismatch_diagnostics_are_combined(self):
        records = {"a": evaluation(diagnostics={"replay_consistent": False, "recompute_mismatch_counts": {"x": 2}}),
                   "b": evaluation(diagnostics={"replay_consistent": False, "recompute_mismatch_counts": {"x": 3, "F": 1}})}
        obj = multi.DatasetObjective(dataset(FakeEpisode("a"), FakeEpisode("b")), lambda ep, p, g: records[ep.id])
        result = obj(SHARED)
        self.assertFalse(result.diagnostics["replay_consistent"])
        self.assertEqual(result.diagnostics["recompute_mismatch_counts"], {"x": 5, "F": 1})
        self.assertEqual(result.diagnostics["recompute_mismatch_count"], 6)
        self.assertEqual(obj.calls_with_mismatches, 1)

    def test_adam_rolls_back_when_only_the_last_episode_rejects_proposals(self):
        initial = dict(DEFAULT_PARAMETERS, viscosity=8.0)
        seen = []

        def evaluate(ep, params, grad):
            eta = params["viscosity"]
            seen.append((ep.id, eta))
            if ep.id == "last" and eta != initial["viscosity"]:
                raise InvalidStateError("Last episode rejects this proposal")
            return {"value": (eta + 1.0) ** 2, "gradient": {"viscosity": 2 * (eta + 1.0)}, "diagnostics": {}}

        objective = multi.DatasetObjective(dataset(FakeEpisode("first"), FakeEpisode("last"), fit=("viscosity",)), evaluate)
        space = PhysicalParameterSpace(initial, ["viscosity"], "none")
        adam = ProjectedAdam(space, objective, AdamOptions(max_backtracks=2, learning_rate=.01))
        adam.initialize()
        before = deepcopy(adam.state_dict())
        step = adam.step()
        after = adam.state_dict()
        self.assertFalse(step["accepted"])
        self.assertEqual(step["status"], "stalled_invalid")
        for key in ("coordinates", "physical_parameters", "first_moment", "second_moment", "current", "best", "accepted_updates"):
            self.assertEqual(before[key], after[key], key)
        self.assertEqual(after["evaluations"], 4)
        self.assertEqual(objective.successful_calls, 1)
        self.assertEqual([ep for ep, _ in seen], ["first", "last"] * 4)


class ProcessEvaluatorTests(HostOnlyTests):
    def setUp(self):
        super().setUp()
        job = os.environ.get("CLAUDE_JOB_DIR")
        parent = Path(job) / "tmp" if job else multi.RUN_ROOT
        parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="multi_episode_host_", dir=parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "runs"
        self.root.mkdir()
        for target, value in (("RUN_ROOT", self.root), ("source_identity", lambda: deepcopy(SOURCE))):
            patcher = patch.object(multi, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.ep = FakeEpisode("a")
        self.other = FakeEpisode("b")
        self.data = dataset(self.ep, self.other)
        self.requests = []

    def response(self, request, episode=None):
        ep = episode or next(ep for ep in self.data.episodes if ep.id == request["episode_id"])
        return {"schema": multi.RESULT_SCHEMA, "request_id": request["request_id"],
                "request_sha256": multi.canonical_hash(request), "episode_id": ep.id,
                "dataset_fingerprint": self.data.fingerprint, "source": deepcopy(SOURCE), "source_after": deepcopy(SOURCE),
                "status": "ok", "parameters": ep.parameters_for(request["parameters"]),
                "compute_grad": request["compute_grad"], "prepared_fingerprint": PREPARED,
                "runtime": runtime(), "evaluation": evaluation()}

    def make(self, mutate=None, code=0, raw=None, **kwargs):
        def launch(request_path, worker_path, request_dir):
            request = json.loads(request_path.read_text())
            self.requests.append(request)
            result = self.response(request)
            if mutate is not None:
                mutate(result, request)
            worker_path.mkdir()
            (worker_path / "result.json").write_text(raw if raw is not None else json.dumps(result))
            return code
        defaults = {"expected_source": deepcopy(SOURCE), "expected_prepared": {"a": PREPARED, "b": PREPARED},
                    "expected_runtimes": {ep.id: multi.stable_runtime(runtime()) for ep in self.data.episodes},
                    "launcher": launch}
        defaults.update(kwargs)
        return multi.EpisodeProcessEvaluator(self.data, self.root / "dataset.json", {}, self.root / "workers", **defaults)

    def test_valid_reply_and_request_parameters(self):
        evaluator = self.make()
        result = evaluator(self.ep, SHARED)
        self.assertEqual(result["value"], 4.0)
        self.assertEqual(self.requests[0]["parameters"], SHARED)
        self.assertEqual(self.requests[0]["expected_prepared_fingerprint"], PREPARED)
        self.assertTrue(self.requests[0]["compute_grad"])
        self.assertFalse(evaluator._active)

    def test_training_processes_are_sequential_and_reentrance_is_rejected(self):
        sequence = []
        evaluator = None

        def launch(request_path, worker_path, request_dir):
            request = json.loads(request_path.read_text())
            sequence.append(("start", request["episode_id"]))
            self.assertTrue(evaluator._active)
            with self.assertRaises(multi.EpisodeExecutionError):
                evaluator(self.other, SHARED)
            worker_path.mkdir()
            (worker_path / "result.json").write_text(json.dumps(self.response(request)))
            sequence.append(("finish", request["episode_id"]))
            return 0

        evaluator = self.make(launcher=launch)
        result = multi.DatasetObjective(self.data, evaluator)(SHARED)
        self.assertEqual(result.value, 4.0)
        self.assertEqual(sequence, [("start", "a"), ("finish", "a"), ("start", "b"), ("finish", "b")])
        self.assertFalse(evaluator._active)
        requests = list((self.root / "workers").glob("*/request.json"))
        self.assertEqual(len(requests), 2)
        ids = [json.loads(p.read_text())["request_id"] for p in requests]
        self.assertEqual(len(set(ids)), 2)

    def test_protocol_identity_mismatches_are_rejected(self):
        changes = {"schema": "bad", "request_id": "different", "request_sha256": "0" * 64,
                   "episode_id": "wrong", "dataset_fingerprint": "0" * 64, "source": {"changed": True}}
        for key, value in changes.items():
            with self.subTest(field=key):
                evaluator = self.make(mutate=lambda r, q, k=key, v=value: r.update({k: v}))
                with self.assertRaises(multi.EpisodeExecutionError):
                    evaluator(self.ep, SHARED)
                self.assertFalse(evaluator._active)

    def test_source_after_mismatch_is_rejected(self):
        evaluator = self.make(mutate=lambda r, q: r.update(source_after={"changed": True}))
        with self.assertRaises(multi.EpisodeExecutionError):
            evaluator(self.ep, SHARED)

    def test_parent_source_change_during_execution_is_rejected(self):
        evaluator = self.make()
        with patch.object(multi, "source_identity", return_value={"changed": True}):
            with self.assertRaises(multi.EpisodeExecutionError):
                evaluator(self.ep, SHARED)

    def test_prepared_inputs_parameters_and_gradient_mode_must_match(self):
        mutations = [lambda r, q: r.update(prepared_fingerprint="0" * 64),
                     lambda r, q: r.update(prepared_fingerprint=None),
                     lambda r, q: r["parameters"].update(viscosity=123.0),
                     lambda r, q: r.update(compute_grad=False)]
        for index, mutation in enumerate(mutations):
            with self.subTest(case=index), self.assertRaises(multi.EpisodeExecutionError):
                self.make(mutate=mutation)(self.ep, SHARED)

    def test_runtime_backend_precision_architecture_and_identity_must_match(self):
        changes = {"backend": "cuda", "precision": "f64", "actual_arch": "Arch.cuda",
                   "initialization_verified": False, "host": "different-host", "cpu_threads": 8, "seed": 99}
        for key, value in changes.items():
            with self.subTest(field=key):
                evaluator = self.make(mutate=lambda r, q, k=key, v=value: r["runtime"].update({k: v}))
                with self.assertRaises(multi.EpisodeExecutionError):
                    evaluator(self.ep, SHARED)

    def test_invalid_reply_with_valid_identity_rejects_candidate(self):
        evaluator = self.make(mutate=lambda r, q: r.update(status="invalid", error="Invalid numerical state"), code=2)
        with self.assertRaises(InvalidStateError):
            evaluator(self.ep, SHARED)
        self.assertFalse(evaluator._active)

    def test_invalid_status_does_not_bypass_other_identity_checks(self):
        mutations = {"parameters": lambda r: r["parameters"].update(viscosity=123.0),
                     "gradient_mode": lambda r: r.update(compute_grad=False),
                     "prepared": lambda r: r.update(prepared_fingerprint="0" * 64),
                     "runtime": lambda r: r["runtime"].update(actual_arch="Arch.cuda"),
                     "source_after": lambda r: r.update(source_after={"changed": True})}
        for name, mutation in mutations.items():
            def mutate(result, request, change=mutation):
                result.update(status="invalid", error="Invalid numerical state")
                change(result)
            with self.subTest(field=name):
                evaluator = self.make(mutate=mutate, code=2)
                with self.assertRaises(multi.EpisodeExecutionError):
                    evaluator(self.ep, SHARED)
                self.assertFalse(evaluator._active)

    def test_invalid_nonobjective_reply_is_not_a_candidate_rejection(self):
        evaluator = self.make(mutate=lambda r, q: r.update(status="invalid", error="Bad validation input"), code=2)
        with self.assertRaises(multi.EpisodeExecutionError):
            evaluator.run(self.ep, "validate", SHARED)

    def test_invalid_status_cannot_hide_schema_or_dataset_mismatch(self):
        def mutate(result, request):
            result.update(status="invalid", dataset_fingerprint="0" * 64)
        with self.assertRaises(multi.EpisodeExecutionError):
            self.make(mutate=mutate, code=2)(self.ep, SHARED)

    def test_abnormal_status_exit_combinations_are_process_errors(self):
        for status, code in (("invalid", 0), ("invalid", 1), ("ok", 2), ("error", 2), ("unknown", 0), ("ok", -9)):
            with self.subTest(status=status, code=code), self.assertRaises(multi.EpisodeExecutionError):
                self.make(mutate=lambda r, q, s=status: r.update(status=s), code=code)(self.ep, SHARED)

    def test_malformed_gradient_or_nonfinite_success_result_is_protocol_error(self):
        mutations = [lambda r, q: r.update(evaluation={"value": 1}),
                     lambda r, q: r["evaluation"].update(gradient={"youngs_modulus": 1}),
                     lambda r, q: r["evaluation"].update(value=float("nan")),
                     lambda r, q: r["evaluation"]["gradient"].update(viscosity="nan"),
                     lambda r, q: r["evaluation"]["gradient"].update(viscosity=True)]
        for index, mutation in enumerate(mutations):
            with self.subTest(case=index), self.assertRaises(multi.EpisodeExecutionError):
                self.make(mutate=mutation)(self.ep, SHARED)

    def test_duplicate_nonfinite_and_truncated_json_are_rejected(self):
        for text in ('{"status":"ok","status":"invalid"}', '{"value":NaN}', '{"value":Infinity}', '{'):
            with self.subTest(text=text), self.assertRaises(multi.EpisodeExecutionError):
                self.make(raw=text)(self.ep, SHARED)

    def test_missing_result_file_is_protocol_error(self):
        evaluator = self.make(launcher=lambda *args: 0)
        with self.assertRaises(multi.EpisodeExecutionError):
            evaluator(self.ep, SHARED)
        self.assertFalse(evaluator._active)

    def test_launcher_interrupt_releases_active_flag_for_next_call(self):
        evaluator = self.make(launcher=Mock(side_effect=KeyboardInterrupt))
        with self.assertRaises(KeyboardInterrupt):
            evaluator(self.ep, SHARED)
        self.assertFalse(evaluator._active)
        evaluator.launcher = self.make().launcher
        self.assertEqual(evaluator(self.ep, SHARED)["value"], 4.0)

    def test_started_event_interrupt_does_not_leave_evaluator_active(self):
        launcher = Mock()

        def event(record):
            if record["event"] == "worker_started":
                raise KeyboardInterrupt

        evaluator = self.make(launcher=launcher, event=event)
        with self.assertRaises(KeyboardInterrupt):
            evaluator(self.ep, SHARED)
        launcher.assert_not_called()
        self.assertFalse(evaluator._active)

    def test_timeout_terminates_and_reaps_process_group(self):
        process = Mock(pid=424242)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", .01), 0]
        evaluator = self.make(launcher=None, timeout_s=.01)
        with patch.object(multi.subprocess, "Popen", return_value=process) as popen, \
                patch.object(multi.os, "killpg") as killpg:
            with self.assertRaises(multi.EpisodeExecutionError):
                evaluator(self.ep, SHARED)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        killpg.assert_called_once_with(424242, signal.SIGTERM)
        self.assertEqual(process.wait.call_args_list, [call(timeout=.01), call(timeout=5)])
        self.assertFalse(evaluator._active)

    def test_interrupted_wait_terminates_and_reaps_process_group(self):
        process = Mock(pid=424243)
        process.poll.return_value = None
        process.wait.side_effect = [KeyboardInterrupt, 0]
        evaluator = self.make(launcher=None)
        with patch.object(multi.subprocess, "Popen", return_value=process), patch.object(multi.os, "killpg") as killpg:
            with self.assertRaises(KeyboardInterrupt):
                evaluator(self.ep, SHARED)
        killpg.assert_called_once_with(424243, signal.SIGTERM)
        self.assertEqual(process.wait.call_args_list, [call(timeout=None), call(timeout=5)])
        self.assertFalse(evaluator._active)

    def test_unresponsive_worker_is_killed_then_reaped(self):
        process = Mock(pid=424244)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", .01), subprocess.TimeoutExpired("worker", 5), 0]
        evaluator = self.make(launcher=None, timeout_s=.01)
        with patch.object(multi.subprocess, "Popen", return_value=process), patch.object(multi.os, "killpg") as killpg:
            with self.assertRaises(multi.EpisodeExecutionError):
                evaluator(self.ep, SHARED)
        self.assertEqual(killpg.call_args_list, [call(424244, signal.SIGTERM), call(424244, signal.SIGKILL)])
        self.assertEqual(process.wait.call_args_list, [call(timeout=.01), call(timeout=5), call()])
        self.assertFalse(evaluator._active)

    def test_already_finished_worker_is_reaped_without_signals(self):
        process = Mock(pid=424245)
        process.poll.return_value = 0
        with patch.object(multi.os, "killpg") as killpg:
            multi.EpisodeProcessEvaluator._stop(process)
        killpg.assert_not_called()
        process.wait.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
