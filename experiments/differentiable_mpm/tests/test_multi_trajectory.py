"""Real corrected-physics multi-episode AD with explicitly mocked input preparation.

The isolated-worker check exercises episode_worker and EpisodeProcessEvaluator,
but replaces dataset loading and input preparation with deterministic synthetic
records. It does not qualify recorded-data parsing, cameras, SDFs, or inventories.
All simulation, visible loss, reverse kernels, aggregation and optimization are real.
"""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from experiments.differentiable_mpm.calibrate import evaluation_record, make_rollout
from experiments.differentiable_mpm.dataset_config import MATERIAL_NAMES
from experiments.differentiable_mpm.loss import ObservationLoss
from experiments.differentiable_mpm.multi_episode import DatasetObjective, EpisodeProcessEvaluator
from experiments.differentiable_mpm.optimize import AdamOptions, ProjectedAdam
from experiments.differentiable_mpm.results import RUN_ROOT, canonical_hash, json_value, source_identity
from experiments.differentiable_mpm.runtime import init_runtime
from experiments.differentiable_mpm.solver import Stepper
from experiments.differentiable_mpm.state import ToolControl
from experiments.differentiable_mpm.synthetic import (
    INITIAL_PARAMETERS, TRUTH, SyntheticConfig, array_hash, generate_observations,
    make_initial_state, parameter_space, simulation_config, synthetic_camera,
    synthetic_loss_config,
)


REPO = Path(__file__).resolve().parents[3]
CONFIG = SyntheticConfig(steps=12, segment_length=4, observation_count=3,
                         physics_version="corrected-v1", precision="f64")
SCOPE = ("Synthetic input preparation and dataset loading are mocked; actual corrected-v1 "
         "CPU f64 MPM, visible loss, segmented reverse AD, weighted dataset objective, "
         "ProjectedAdam and isolated episode-worker protocol are tested. No real calibration.")


def write_json(path, record):
    Path(path).write_text(json.dumps(json_value(record), indent=2, sort_keys=True, allow_nan=False) + "\n")


def fixture_dataset():
    episodes = []
    for name, motion, membership, weight in (("motion_a", "a", "training", 1.0),
                                              ("motion_b", "b", "training", 2.0),
                                              ("heldout_c", "c", "validation", 1.0)):
        config = SimpleNamespace(backend="cpu", seed=CONFIG.seed, segment_length=CONFIG.segment_length,
                                 tool_sdf_resolution=8, simulation=asdict(simulation_config(CONFIG)),
                                 motion=motion)
        episodes.append(SimpleNamespace(id=name, membership=membership, weight=weight, config=config,
                                        scored_window=None,
                                        parameters_for=lambda shared: {**INITIAL_PARAMETERS, **shared}))
    fingerprint = canonical_hash({"fixture": "synthetic-three-motions-v1", "settings": asdict(CONFIG),
                                  "episodes": [(ep.id, ep.config.motion, ep.membership, ep.weight) for ep in episodes]})
    return SimpleNamespace(episodes=tuple(episodes), fit_parameters=tuple(MATERIAL_NAMES), fingerprint=fingerprint)


def prepare_fixture(config, *, split, build_sdf, scored_window, stepper=None):
    """Prepare synthetic records after runtime initialization, without recorded inputs."""
    if split not in {"training", "validation"} or not build_sdf or scored_window is not None:
        raise ValueError("Unexpected synthetic fixture preparation arguments")
    initial = make_initial_state(CONFIG, heldout=config.motion == "b")
    if config.motion == "c":
        # A third prescribed motion, held out from the objective and optimizer.
        initial.v *= -1
        initial.C *= -1
    initial.validate()
    sim = simulation_config(CONFIG)
    if stepper is None:
        stepper = Stepper(sim, TRUTH, capacity=CONFIG.segment_length + 1)
    controls = [ToolControl.stationary(i * CONFIG.dt) for i in range(CONFIG.steps)]
    camera = synthetic_camera()
    target_loss = ObservationLoss(camera, synthetic_loss_config(visibility_temperature_m=0.002),
                                  initial.x, precision="f64")
    observations = generate_observations(stepper, initial, controls, CONFIG.observation_steps,
                                         target_loss, CONFIG.seed + 1)
    record = {"motion": config.motion, "split": split, "physics_version": sim.physics_version,
              "initial_state": {name: array_hash(array) for name, array in initial.arrays().items()},
              "observations": [{"step": obs.step, "depth": array_hash(obs.observed_depth),
                                "mask": array_hash(obs.observed_valid),
                                "points": array_hash(obs.observed_points)} for obs in observations]}
    fingerprint = canonical_hash(record)
    return SimpleNamespace(config=config, simulation_config=sim, parameters=dict(INITIAL_PARAMETERS),
                           initial_state=initial, controls=controls, observations=observations,
                           camera=camera, loss_config=synthetic_loss_config(visibility_temperature_m=0.01),
                           sdf=None, total_steps=CONFIG.steps, fingerprint=fingerprint,
                           summary=lambda: {**record, "fingerprint": fingerprint, "scope": SCOPE})


def fixture_worker(request_path, output_dir):
    """Keep the real worker; replace only unavailable recorded-input preparation."""
    from experiments.differentiable_mpm import episode_worker
    dataset = fixture_dataset()
    with patch.object(episode_worker, "_load_dataset", return_value=dataset), \
            patch.object(episode_worker, "prepare_experiment", side_effect=prepare_fixture):
        return episode_worker.main(["--request", str(request_path), "--output-dir", str(output_dir)])


class MultiTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.started = time.perf_counter()
        cls.output = RUN_ROOT / "tests" / ("multi_trajectory_" + uuid.uuid4().hex)
        cls.output.mkdir(parents=True)
        print("MULTI_TRAJECTORY_EVIDENCE", cls.output, flush=True)
        cls.source = source_identity()
        write_json(cls.output / "source_before.json", cls.source)
        write_json(cls.output / "scope.json", {"scope": SCOPE, "settings": asdict(CONFIG),
                                               "reference_policy_for_worker": "frozen"})
        cls.runtime = init_runtime("cpu", "f64", cpu_threads=1, seed=CONFIG.seed)
        cls.dataset = fixture_dataset()
        cls.calls = []
        cls.prepared, cls.rollouts, cls.individual = {}, {}, {}
        stepper = Stepper(simulation_config(CONFIG), TRUTH, capacity=CONFIG.segment_length + 1)
        for ep in cls.dataset.episodes:
            prepared = prepare_fixture(ep.config, split=ep.membership, build_sdf=True,
                                       scored_window=None, stepper=stepper)
            cls.prepared[ep.id] = prepared
            _, cls.rollouts[ep.id] = make_rollout(prepared, stepper=stepper, progress=None)
            write_json(cls.output / (ep.id + "_fixture.json"), prepared.summary())
            if ep.membership == "training":
                cls.individual[ep.id] = evaluation_record(
                    cls.rollouts[ep.id].value_and_gradient(INITIAL_PARAMETERS))
        cls.objective = DatasetObjective(cls.dataset, cls.evaluate)
        cls.initial = cls.objective(INITIAL_PARAMETERS)
        write_json(cls.output / "initial_objective.json", {
            "value": cls.initial.value, "gradient": cls.initial.gradient,
            "diagnostics": cls.initial.diagnostics, "individual": cls.individual})

    @classmethod
    def evaluate(cls, episode, shared, compute_grad):
        if set(shared) != set(MATERIAL_NAMES):
            raise AssertionError("Dataset objective must pass exactly five shared material parameters")
        cls.calls.append({"episode": episode.id, "shared": dict(shared), "compute_grad": compute_grad})
        return evaluation_record(cls.rollouts[episode.id].value_and_gradient(
            episode.parameters_for(shared), compute_grad=compute_grad))

    @classmethod
    def tearDownClass(cls):
        after = source_identity()
        write_json(cls.output / "source_after.json", after)
        write_json(cls.output / "execution.json", {"runtime": cls.runtime, "scope": SCOPE,
                                                   "elapsed_seconds": time.perf_counter() - cls.started,
                                                   "source_unchanged": after == cls.source,
                                                   "calls": cls.calls})
        if after != cls.source:
            raise AssertionError("Experiment sources changed during multi-trajectory tests")

    def test_weighted_ad_matches_individual_episodes(self):
        self.assertEqual([ep.id for ep in self.objective.episodes], ["motion_a", "motion_b"])
        np.testing.assert_allclose(self.objective.weights, [1 / 3, 2 / 3], rtol=0, atol=0)
        expected = math.fsum(self.individual[ep.id]["value"] * w
                             for ep, w in zip(self.objective.episodes, self.objective.weights))
        self.assertAlmostEqual(self.initial.value, expected, places=12)
        for name in MATERIAL_NAMES:
            expected = math.fsum(self.individual[ep.id]["gradient"][name] * w
                                 for ep, w in zip(self.objective.episodes, self.objective.weights))
            self.assertAlmostEqual(self.initial.gradient[name], expected, places=12)
            self.assertTrue(np.isfinite(expected))
            self.assertGreater(abs(expected), 1e-12)
        self.assertNotEqual(self.prepared["motion_a"].fingerprint, self.prepared["motion_b"].fingerprint)
        self.assertGreater(np.linalg.norm(self.prepared["motion_a"].initial_state.v -
                                         self.prepared["motion_b"].initial_state.v), 0.1)
        self.assertNotEqual(self.individual["motion_a"]["value"], self.individual["motion_b"]["value"])
        self.assertTrue(self.initial.diagnostics["replay_consistent"])
        for record in self.individual.values():
            self.assertEqual(record["diagnostics"]["observation_gradient_injections"], [1, 1, 1])
            self.assertEqual(record["diagnostics"]["recompute_mismatch_count"], 0)

    def test_combined_gradient_finite_differences_multiple_epsilons(self):
        steps = {"youngs_modulus": 0.01, "poisson_ratio": 1e-6, "viscosity": 0.001,
                 "plastic_min": 1e-6, "plastic_max": 1e-6}
        rows = {}
        start = len(self.calls)
        for name in MATERIAL_NAMES:
            rows[name] = []
            for multiplier in (1.0, 0.3):
                h = steps[name] * multiplier
                plus, minus = dict(INITIAL_PARAMETERS), dict(INITIAL_PARAMETERS)
                plus[name] += h
                minus[name] -= h
                fd = (self.objective.value_and_gradient(plus, compute_grad=False).value -
                      self.objective.value_and_gradient(minus, compute_grad=False).value) / (2 * h)
                ad = self.initial.gradient[name]
                relative = abs(ad - fd) / max(abs(ad), abs(fd), 1e-9)
                rows[name].append({"epsilon": h, "ad": ad, "fd": fd, "relative_error": relative})
                print(f'MULTI_FD {name} h={h:g} AD={ad:.10g} FD={fd:.10g} relative={relative:.4g}', flush=True)
        write_json(self.output / "finite_differences.json", rows)
        for name, checks in rows.items():
            self.assertLess(min(row["relative_error"] for row in checks), 1e-3, name)
        self.assertEqual({call["episode"] for call in self.calls[start:]}, {"motion_a", "motion_b"})
        self.assertTrue(all(not call["compute_grad"] for call in self.calls[start:]))

    def test_short_fit_improves_without_evaluating_heldout(self):
        start = len(self.calls)
        options = AdamOptions(learning_rate=0.1, max_backtracks=8,
                              gradient_tolerance=1e-9, loss_tolerance=1e-12)
        optimizer = ProjectedAdam(parameter_space(), self.objective, options,
                                  objective_id=self.dataset.fingerprint)
        result = optimizer.run(4)
        fit_calls = self.calls[start:].copy()
        selected = dict(result.best_parameters)
        reduction = (self.initial.value - result.best_value) / abs(self.initial.value)
        # Evaluate validation only after the selected candidate has been frozen.
        validation = next(ep for ep in self.dataset.episodes if ep.membership == "validation")
        heldout = self.evaluate(validation, {name: selected[name] for name in MATERIAL_NAMES}, False)
        write_json(self.output / "fit.json", {
            "initial_loss": self.initial.value, "selected_loss": result.best_value,
            "relative_reduction": reduction, "accepted_updates": result.accepted_updates,
            "selected_parameters": selected, "truth": TRUTH, "history": result.history,
            "heldout": heldout, "heldout_used_for_selection": False, "fit_calls": fit_calls,
            "interpretation": "Same-model synthetic optimization with different target/prediction smoothing is not physical parameter identification."})
        print(f'MULTI_FIT {self.initial.value:.10g} -> {result.best_value:.10g}; reduction={reduction:.6%}', flush=True)
        self.assertGreater(result.accepted_updates, 0)
        self.assertGreater(reduction, 0.01)
        self.assertTrue(np.isfinite(heldout["value"]))
        self.assertTrue(fit_calls)
        self.assertEqual({call["episode"] for call in fit_calls}, {"motion_a", "motion_b"})

    def test_isolated_worker_matches_direct_real_kernels(self):
        def launcher(request_path, worker_path, request_dir):
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
            command = [sys.executable, "-u", "-m", "experiments.differentiable_mpm.tests.test_multi_trajectory",
                       "--fixture-worker", str(request_path), str(worker_path)]
            with (request_dir / "stdout.log").open("w") as stdout, (request_dir / "stderr.log").open("w") as stderr:
                process = subprocess.run(command, cwd=REPO, env=env, stdout=stdout, stderr=stderr, timeout=900)
            return process.returncode

        episode = self.dataset.episodes[0]
        evaluator = EpisodeProcessEvaluator(
            self.dataset, self.output / "mocked_dataset.json", {}, self.output / "workers",
            reference_policy="frozen", cpu_threads=1, launcher=launcher,
            expected_source=self.source,
            expected_prepared={ep.id: self.prepared[ep.id].fingerprint for ep in self.dataset.episodes})
        record = evaluator(episode, INITIAL_PARAMETERS, compute_grad=True)
        direct = self.individual[episode.id]
        self.assertAlmostEqual(record["value"], direct["value"], places=11)
        self.assertEqual(set(record["gradient"]), set(direct["gradient"]))
        for name in direct["gradient"]:
            np.testing.assert_allclose(record["gradient"][name], direct["gradient"][name], rtol=1e-10, atol=1e-11)
        self.assertEqual(record["diagnostics"]["observation_gradient_injections"], [1, 1, 1])
        self.assertTrue(record["diagnostics"]["replay_consistent"])
        self.assertFalse(evaluator._active)
        write_json(self.output / "worker_equivalence.json", {"scope": SCOPE, "episode": episode.id,
                                                             "direct": direct, "isolated": record})


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--fixture-worker":
        raise SystemExit(fixture_worker(sys.argv[2], sys.argv[3]))
    unittest.main()
