"""Deterministic regression tests for minibatch checkpoint continuation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import continue_minibatch_fit as continuation
from experiments.differentiable_mpm import calibrate_dataset as calibration
from experiments.differentiable_mpm.optimize import AdamOptions, ObjectiveValue
from experiments.differentiable_mpm.results import canonical_hash


class Space:
    def coordinates(self):
        return np.array([0.25, -0.35], dtype=np.float64)

    def project(self, u):
        return np.clip(u, [-2, -2], [2, 2])

    def physical(self, u):
        return {"p": float(u[0]), "q": float(u[1])}

    def pullback(self, u, gradient):
        return np.array([gradient["p"], gradient["q"]], dtype=np.float64)


class Episode:
    def __init__(self, index):
        self.id = f"chunk{index:02}"
        self.membership = "training" if index <= 20 else "validation"
        self.weight = 1.0


class Dataset:
    fit_parameters = ("p", "q")
    episodes = tuple(Episode(index) for index in range(1, 26))

    def parameter_space(self):
        return Space()


class Store:
    identity_hash = "test-identity"

    def __init__(self):
        self.written = {}
        self.events = []

    def write_json(self, name, record):
        self.written[name] = deepcopy(record)

    def append_event(self, record):
        self.events.append(deepcopy(record))


class TestMinibatchContinuation(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset()
        self.options = AdamOptions(learning_rate=0.01)
        self.calls = []

    def evaluate(self, dataset, batch, parameters, *, compute_grad, runner_factory, max_workers, event):
        ids = tuple(ep.id for ep in batch)
        self.calls.append((ids, compute_grad))
        index = int(batch[0].id[-2:])
        p, q = parameters["p"], parameters["q"]
        value = (p - index / 30) ** 2 + (q + index / 20) ** 2
        gradient = {"p": 2 * (p - index / 30), "q": 2 * (q + index / 20)} if compute_grad else {}
        return ObjectiveValue(value, gradient, {"replay_consistent": True,
                                                "episodes": [{"episode_id": ep.id, "value": value,
                                                              "diagnostics": {}} for ep in batch]})

    def baseline(self, iterations):
        args = type("Args", (), {"episode_batch_size": 4, "iterations": iterations, "fit_log": "quiet",
                                "fixed_batch_order": True, "batch_shuffle_seed": 0})()
        store = Store()
        with patch.object(calibration, "_evaluate_episode_batch", side_effect=self.evaluate):
            calibration._run_minibatch_fit(self.dataset, args, self.options, store, lambda: None, lambda _: None)
        return store.written["optimizer_state.json"]["state"]

    def test_split_resume_matches_uninterrupted_171_updates(self):
        baseline = self.baseline(171)
        split = self.baseline(21)
        state = deepcopy(split)
        state.update(origin_iteration=21, continuation_objectives=0, continuation_mismatches=0)
        store = Store()
        self.calls.clear()
        with patch.object(calibration, "_evaluate_episode_batch", side_effect=self.evaluate):
            first = continuation.continue_updates(self.dataset, state, batch_size=4,
                                                  target_iteration=171, max_updates=5,
                                                  adam_options=self.options, runner_factory=lambda: None,
                                                  store=store, event=lambda _: None)
            self.assertEqual(first["iterations"], 26)
            self.assertEqual(self.calls[0][0], tuple(f"chunk{i:02}" for i in range(5, 9)))
            for _ in range(29):
                continuation.continue_updates(self.dataset, state, batch_size=4,
                                              target_iteration=171, max_updates=5,
                                              adam_options=self.options, runner_factory=lambda: None,
                                              store=store, event=lambda _: None)
        self.assertEqual(state["iterations"], 171)
        self.assertEqual(state["continuation_objectives"], 150)
        for key in ("coordinates", "first_moment", "second_moment", "physical_parameters", "history"):
            self.assertEqual(state[key], baseline[key], key)
        self.assertEqual(store.written["optimizer_state.json"]["identity_sha256"], store.identity_hash)
        self.assertEqual(len(store.events), 150)
        self.assertEqual([record["iteration"] for record in store.events], list(range(22, 172)))

    def test_corrupt_or_reordered_checkpoint_rejected(self):
        state = self.baseline(21)
        for modified in (
            lambda s: s["history"][0].update(episode_ids=["chunk03"]),
            lambda s: s.update(first_moment=[float("nan"), 0.0]),
            lambda s: s.update(accepted_updates=20),
            lambda s: s.update(coordinates=[5.0, 0.0]),
            lambda s: s.update(physical_parameters={"p": 0.0, "q": 0.0}),
            lambda s: s.update(first_moment=[0.0, 0.0]),
            lambda s: s["history"][0]["coordinate_gradient"].__setitem__(0, 123.0),
        ):
            broken = deepcopy(state)
            modified(broken)
            with self.subTest(modified=modified), self.assertRaises(ValueError):
                continuation.verify_snapshot(broken, dataset=self.dataset, batch_size=4,
                                             origin_iteration=21, target_iteration=171,
                                             adam_options=self.options)

    def test_driver_writes_and_restores_off_runtime_mirror(self):
        origin_state = self.baseline(21)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_run = root / "original"
            original_run.mkdir()
            runs = root / "runs"
            runs.mkdir()
            output = runs / "continued"
            mirror = root / "drive_backup"
            episode_config = {"backend": "cuda", "segment_length": 64,
                              "simulation": {"precision": "f64", "p2g_mode": "atomic",
                                             "physics_version": "corrected-v1"}}
            dataset_document = {"episodes": [{"config": episode_config}],
                                "shared_parameters": {"initial": {}}}
            source = {"test": "frozen"}
            dataset_identity = {"dataset": dataset_document, "source": source,
                                "dataset_fingerprint": "fixed-inputs",
                                "episodes": [{"episode_id": ep.id} for ep in self.dataset.episodes],
                                "reference_policy": "frozen", "cpu_threads": 1,
                                "ignore_recompute_mismatch": False,
                                "physics_version": "corrected-v1", "physics_reference": {"test": True}}
            optimizer = asdict(self.options)
            optimizer.update(episode_batch_size=4, objective="independent-episode-minibatch-v1")
            identity = {"action": "dataset-fit", "optimizer": optimizer,
                        "dataset_identity": dataset_identity}
            digest = canonical_hash(identity)
            documents = {
                "run_manifest.json": {"schema": "taichidough-differentiable-run-v1",
                                      "identity_sha256": digest, "identity": identity},
                "optimizer_state.json": {"schema": continuation.CHECKPOINT_SCHEMA,
                                         "identity_sha256": digest, "state": origin_state},
                "result.json": {"status": "budget_exhausted", "independent_evaluation_skipped": False,
                                "optimization": {"accepted_updates": 21,
                                                 "current_parameters": origin_state["physical_parameters"]},
                                "evaluations": {ep.id: {"strict_evaluation": {"valid": True}}
                                                for ep in self.dataset.episodes}},
                "selected_parameters.json": {"identity_sha256": digest},
            }
            for name, document in documents.items():
                (original_run / name).write_text(json.dumps(document))
            summaries = {ep.id: {"prepared_fingerprint": "fixed", "runtime": {"host": "same"}}
                         for ep in self.dataset.episodes}
            (original_run / "validated_inputs.json").write_text(json.dumps(
                {"episodes": summaries, "preflight_directory": str(root / "preflight")}))
            from experiments.differentiable_mpm.results import RunStore

            def temp_store(path, identity, *, resume=False):
                return RunStore(path, identity, resume=resume, allowed_root=runs)

            args = SimpleNamespace(dataset=root / "dataset.json", original_run=original_run,
                                   output_dir=output, target_updates=171, min_additional_updates=150,
                                   max_updates_this_run=1, mirror_dir=mirror, resume=False,
                                   allow_host_change=False,
                                   allow_partial_evaluation_after_user_request=False)
            with (patch.object(continuation, "source_identity", return_value=source),
                  patch.object(continuation, "load_dataset", return_value=self.dataset),
                  patch.object(continuation, "RunStore", side_effect=temp_store),
                  patch.object(calibration, "preflight", return_value=(summaries, {}, root / "preflight")),
                  patch.object(continuation.socket, "gethostname", return_value="same"),
                  patch.object(calibration, "_dataset_identity", return_value=dataset_identity),
                  patch.object(calibration, "_runner", return_value=None),
                  patch.object(calibration, "_evaluate_episode_batch", side_effect=self.evaluate),
                  patch.object(self.dataset, "as_dict", return_value=dataset_document, create=True),
                  patch.object(self.dataset, "fingerprint", "fixed-inputs", create=True)):
                continuation.run(args)
                self.assertEqual(json.loads((mirror / "optimizer_state.json").read_text())["state"]["iterations"], 22)
                previous_marker = (mirror / "checkpoint_sha256.json").read_text()
                self.assertEqual(json.loads((mirror / "original_run" / "run_manifest.json").read_text()),
                                 documents["run_manifest.json"])
                args.resume = True
                continuation.run(args)
                saved = json.loads((output / "optimizer_state.json").read_text())
                self.assertEqual(saved["state"]["iterations"], 23)
                self.assertEqual(json.loads((mirror / "optimizer_state.json").read_text()), saved)
                self.assertEqual(saved["state"]["history"][:21], origin_state["history"])
                (mirror / "checkpoint_sha256.json").write_text(previous_marker)
                continuation.run(args)
                self.assertEqual(json.loads((mirror / "checkpoint_sha256.json").read_text())["iterations"], 24)
                self.assertEqual(json.loads((output / "optimizer_state.json").read_text())["state"]["iterations"], 24)
                marker = json.loads((mirror / "checkpoint_sha256.json").read_text())
                marker["optimizer_state_sha256"] = "bad"
                (mirror / "checkpoint_sha256.json").write_text(json.dumps(marker))
                with self.assertRaisesRegex(ValueError, "hash does not match"):
                    continuation.run(args)

    def _mirror_fixture(self, root, *, with_validated_inputs):
        """Build the original_run fixture shared by the mirror and missing-validated-inputs tests."""
        original_run = root / "original"
        original_run.mkdir()
        runs = root / "runs"
        runs.mkdir()
        output = runs / "continued"
        mirror = root / "drive_backup"
        origin_state = self.baseline(21)
        episode_config = {"backend": "cuda", "segment_length": 64,
                          "simulation": {"precision": "f64", "p2g_mode": "atomic",
                                         "physics_version": "corrected-v1"}}
        dataset_document = {"episodes": [{"config": episode_config}],
                            "shared_parameters": {"initial": {}}}
        source = {"test": "frozen"}
        dataset_identity = {"dataset": dataset_document, "source": source,
                            "dataset_fingerprint": "fixed-inputs",
                            "episodes": [{"episode_id": ep.id} for ep in self.dataset.episodes],
                            "reference_policy": "frozen", "cpu_threads": 1,
                            "ignore_recompute_mismatch": False,
                            "physics_version": "corrected-v1", "physics_reference": {"test": True}}
        optimizer = asdict(self.options)
        optimizer.update(episode_batch_size=4, objective="independent-episode-minibatch-v1")
        identity = {"action": "dataset-fit", "optimizer": optimizer,
                    "dataset_identity": dataset_identity}
        digest = canonical_hash(identity)
        documents = {
            "run_manifest.json": {"schema": "taichidough-differentiable-run-v1",
                                  "identity_sha256": digest, "identity": identity},
            "optimizer_state.json": {"schema": continuation.CHECKPOINT_SCHEMA,
                                     "identity_sha256": digest, "state": origin_state},
            "result.json": {"status": "budget_exhausted", "independent_evaluation_skipped": False,
                            "optimization": {"accepted_updates": 21,
                                             "current_parameters": origin_state["physical_parameters"]},
                            "evaluations": {ep.id: {"strict_evaluation": {"valid": True}}
                                            for ep in self.dataset.episodes}},
            "selected_parameters.json": {"identity_sha256": digest},
        }
        for name, document in documents.items():
            (original_run / name).write_text(json.dumps(document))
        summaries = {ep.id: {"prepared_fingerprint": "fixed", "runtime": {"host": "same"}}
                     for ep in self.dataset.episodes}
        if with_validated_inputs:
            (original_run / "validated_inputs.json").write_text(json.dumps(
                {"episodes": summaries, "preflight_directory": str(root / "preflight")}))
        args = SimpleNamespace(dataset=root / "dataset.json", original_run=original_run,
                               output_dir=output, target_updates=171, min_additional_updates=150,
                               max_updates_this_run=1, mirror_dir=mirror, resume=False,
                               allow_host_change=False,
                               allow_partial_evaluation_after_user_request=False)
        return args, dataset_document, dataset_identity, source, summaries, output, runs

    def test_origin_args_drives_real_preflight_without_crashing(self):
        """_origin_args must build a Namespace that calibration.preflight() can actually use
        end-to-end (not just a mocked stand-in) -- preflight's event() closure reads args.action,
        which _origin_args previously omitted, crashing every episode with AttributeError."""
        original_identity = {"optimizer": {}, "dataset_identity": {"reference_policy": "frozen",
                                                                    "cpu_threads": 1,
                                                                    "ignore_recompute_mismatch": False}}
        args = continuation._origin_args(Path("/tmp/dataset.json"), original_identity)
        self.assertEqual(args.action, "fit")

        class FakeRunner:
            def run(self, ep, mode, shared_initial, *, no_runtime):
                return {"prepared_fingerprint": "fp", "runtime": {"host": "same"},
                        "prepared_summary": {"provenance": {"sequence_fingerprint": ep.id.rjust(64, "0")}}}

        from experiments.differentiable_mpm.results import RunStore

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def temp_store(path, identity, *, resume=False):
                return RunStore(path, identity, resume=resume, allowed_root=root)

            with (patch.object(calibration, "RUN_ROOT", root),
                  patch.object(calibration, "RunStore", side_effect=temp_store),
                  patch.object(calibration, "_runner", return_value=FakeRunner()),
                  patch.object(self.dataset, "as_dict", return_value={"episodes": []}, create=True),
                  patch.object(self.dataset, "shared_initial", {}, create=True)):
                summaries, failures, preflight_root = calibration.preflight(self.dataset, args, self.options,
                                                                             source={"test": "frozen"})
        self.assertEqual(failures, {})
        self.assertEqual(len(summaries), len(self.dataset.episodes))

    def test_missing_validated_inputs_runs_fresh_preflight_instead_of_crashing(self):
        """A genuinely fresh runtime (no archived validated_inputs.json) must not crash;
        it should fall through to the same fresh-preflight path already used for a host change,
        and the fresh preflight result must still pass the full identity check."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, dataset_document, dataset_identity, source, summaries, output, runs = self._mirror_fixture(
                root, with_validated_inputs=False)
            from experiments.differentiable_mpm.results import RunStore

            def temp_store(path, identity, *, resume=False):
                return RunStore(path, identity, resume=resume, allowed_root=runs)

            preflight_calls = []

            def fake_preflight(dataset, run_args, options, source):
                preflight_calls.append(1)
                return summaries, {}, root / "preflight"

            with (patch.object(continuation, "source_identity", return_value=source),
                  patch.object(continuation, "load_dataset", return_value=self.dataset),
                  patch.object(continuation, "RunStore", side_effect=temp_store),
                  patch.object(calibration, "preflight", side_effect=fake_preflight),
                  patch.object(continuation.socket, "gethostname", return_value="same"),
                  patch.object(calibration, "_dataset_identity", return_value=dataset_identity),
                  patch.object(calibration, "_runner", return_value=None),
                  patch.object(calibration, "_evaluate_episode_batch", side_effect=self.evaluate),
                  patch.object(self.dataset, "as_dict", return_value=dataset_document, create=True),
                  patch.object(self.dataset, "fingerprint", "fixed-inputs", create=True)):
                continuation.run(args)
            # Fresh preflight must actually have run (not a silent crash, not a silent skip).
            self.assertEqual(len(preflight_calls), 1)
            saved_validated = json.loads((output / "validated_inputs.json").read_text())
            self.assertEqual(saved_validated["episodes"], summaries)
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertIn("original_validated_inputs_absent_fresh_preflight_performed",
                          [e.get("event") for e in events])
            self.assertNotIn("original_preflight_reused", [e.get("event") for e in events])
            self.assertEqual(json.loads((output / "optimizer_state.json").read_text())["state"]["iterations"], 22)

    def test_validated_inputs_present_still_reuses_original_preflight(self):
        """Existing same-host reuse behavior is unchanged when validated_inputs.json is present."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, dataset_document, dataset_identity, source, summaries, output, runs = self._mirror_fixture(
                root, with_validated_inputs=True)
            from experiments.differentiable_mpm.results import RunStore

            def temp_store(path, identity, *, resume=False):
                return RunStore(path, identity, resume=resume, allowed_root=runs)

            def fail_preflight(*a, **k):
                raise AssertionError("preflight must not run when the original validation is reused")

            with (patch.object(continuation, "source_identity", return_value=source),
                  patch.object(continuation, "load_dataset", return_value=self.dataset),
                  patch.object(continuation, "RunStore", side_effect=temp_store),
                  patch.object(calibration, "preflight", side_effect=fail_preflight),
                  patch.object(continuation.socket, "gethostname", return_value="same"),
                  patch.object(calibration, "_dataset_identity", return_value=dataset_identity),
                  patch.object(calibration, "_runner", return_value=None),
                  patch.object(calibration, "_evaluate_episode_batch", side_effect=self.evaluate),
                  patch.object(self.dataset, "as_dict", return_value=dataset_document, create=True),
                  patch.object(self.dataset, "fingerprint", "fixed-inputs", create=True)):
                continuation.run(args)
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertIn("original_preflight_reused", [e.get("event") for e in events])
            self.assertEqual(json.loads((output / "optimizer_state.json").read_text())["state"]["iterations"], 22)

    def test_partial_evaluation_override_pins_completed_strict_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {"sources": "pinned"}
            identity = {"action": "dataset-fit", "optimizer": {"objective": "independent-episode-minibatch-v1"},
                        "dataset_identity": {"source": source,
                                             "dataset": {"shared_parameters": {"initial": {"p": 1.0}, "fit": ["p"]},
                                                         "episodes": [{"id": "chunk01", "membership": "training",
                                                                       "config": {"parameters": {"tool_retention": 1.0}}},
                                                                      {"id": "chunk02", "membership": "validation",
                                                                       "config": {"parameters": {"tool_retention": 1.0}}}]},
                                             "episodes": [{"episode_id": "chunk01"},
                                                          {"episode_id": "chunk02"}]}}
            digest = canonical_hash(identity)
            manifest = {"schema": "taichidough-differentiable-run-v1",
                        "identity_sha256": digest, "identity": identity}
            state = {"accepted_updates": 21, "physical_parameters": {"p": 1.0, "tool_retention": 1.0}}
            checkpoint = {"schema": continuation.CHECKPOINT_SCHEMA,
                          "identity_sha256": digest, "state": state}
            result = {"status": "budget_exhausted", "optimization": {"accepted_updates": 21},
                      "evaluations": {}}
            for name, record in (("run_manifest.json", manifest), ("optimizer_state.json", checkpoint),
                                 ("result.json", result)):
                (root / name).write_text(json.dumps(record))
            request_dir = root / "episodes" / "a"
            (request_dir / "worker").mkdir(parents=True)
            request = {"action": "evaluate", "episode_id": "chunk01", "request_id": "a",
                       "expected_source": source, "parameters": {"p": 1.0}}
            response = {"status": "ok", "episode_id": "chunk01", "request_id": "a",
                        "request_sha256": canonical_hash(request), "source": source,
                        "source_after": source, "export": {"strict_evaluation": {"valid": True}}}
            (request_dir / "request.json").write_text(json.dumps(request))
            (request_dir / "worker" / "result.json").write_text(json.dumps(response))
            with self.assertRaisesRegex(ValueError, "override required"):
                continuation._read_origin(root)
            _, _, _, inventory = continuation._read_origin(root, allow_partial_evaluation=True)
            self.assertEqual(inventory["completed_count"], 1)
            self.assertEqual(inventory["evaluations"][0]["episode_id"], "chunk01")
            request["parameters"]["p"] = 1.01
            response["request_sha256"] = canonical_hash(request)
            (request_dir / "request.json").write_text(json.dumps(request))
            (request_dir / "worker" / "result.json").write_text(json.dumps(response))
            with self.assertRaisesRegex(ValueError, "mismatched"):
                continuation._read_origin(root, allow_partial_evaluation=True)
            request["parameters"]["p"] = 1.0
            response["request_sha256"] = canonical_hash(request)
            (request_dir / "request.json").write_text(json.dumps(request))
            response["export"]["strict_evaluation"]["valid"] = False
            (request_dir / "worker" / "result.json").write_text(json.dumps(response))
            with self.assertRaisesRegex(ValueError, "invalid"):
                continuation._read_origin(root, allow_partial_evaluation=True)
            response["export"]["strict_evaluation"]["valid"] = True
            (request_dir / "worker" / "result.json").write_text(json.dumps(response))
            checkpoint["state"]["physical_parameters"]["tool_retention"] = 0.9
            (root / "optimizer_state.json").write_text(json.dumps(checkpoint))
            with self.assertRaisesRegex(ValueError, "tool retention"):
                continuation._read_origin(root, allow_partial_evaluation=True)

    def test_runtime_migration_only_allows_host_changes(self):
        old = {"dataset_fingerprint": "unchanged", "episodes": [
            {"episode_id": "chunk01", "prepared_fingerprint": "same",
             "runtime": {"host": "old-host", "actual_arch": "Arch.cuda", "precision": "f64"}}]}
        current = deepcopy(old)
        current["episodes"][0]["runtime"]["host"] = "new-host"
        with patch.object(calibration, "_dataset_identity", return_value=current):
            with self.assertRaisesRegex(ValueError, "preflight"):
                continuation._check_preflight_identity(None, None, None, None, old,
                                                       allow_host_change=False)
            changes = continuation._check_preflight_identity(None, None, None, None, old,
                                                              allow_host_change=True)
            self.assertEqual(changes[0]["current_host"], "new-host")
        current["episodes"][0]["runtime"]["precision"] = "f32"
        with patch.object(calibration, "_dataset_identity", return_value=current):
            with self.assertRaisesRegex(ValueError, "more than"):
                continuation._check_preflight_identity(None, None, None, None, old,
                                                       allow_host_change=True)

    def test_original_run_requires_completed_strict_evaluations_and_matching_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            identity = {"action": "dataset-fit",
                        "optimizer": {"objective": "independent-episode-minibatch-v1"},
                        "dataset_identity": {"episodes": [{"episode_id": "chunk01"}]}}
            digest = canonical_hash(identity)
            manifest = {"schema": "taichidough-differentiable-run-v1",
                        "identity_sha256": digest, "identity": identity}
            checkpoint = {"schema": continuation.CHECKPOINT_SCHEMA,
                          "identity_sha256": digest, "state": {"accepted_updates": 21}}
            result = {"status": "budget_exhausted", "independent_evaluation_skipped": False,
                      "optimization": {"accepted_updates": 21},
                      "evaluations": {"chunk01": {"strict_evaluation": {"valid": True}}}}
            for name, document in (("run_manifest.json", manifest), ("optimizer_state.json", checkpoint),
                                   ("result.json", result)):
                (path / name).write_text(json.dumps(document))
            continuation._read_origin(path)
            result["evaluations"] = {}
            (path / "result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                continuation._read_origin(path)
            result["evaluations"] = {"chunk01": {"strict_evaluation": {"valid": True}}}
            (path / "result.json").write_text(json.dumps(result))
            checkpoint["identity_sha256"] = "wrong"
            (path / "optimizer_state.json").write_text(json.dumps(checkpoint))
            with self.assertRaisesRegex(ValueError, "disagree"):
                continuation._read_origin(path)


if __name__ == "__main__":
    unittest.main()
