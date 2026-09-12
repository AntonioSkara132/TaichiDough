"""Checkpoint orchestration tests with an analytic all-state recurrence.

Real Taichi integration checks are in test_trajectory.py. This recurrence makes
boundary-adjoint errors identifiable without relying on a particular MPM fixture.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.differentiable_mpm.checkpoint import CheckpointedRollout, estimate_memory
from experiments.differentiable_mpm.state import (
    DEFAULT_PARAMETERS, PARAMETER_NAMES, ParticleState, InvalidStateError,
)


def flatten(state):
    return np.concatenate([state.x, state.v, state.C.reshape(-1, 9),
                           state.F.reshape(-1, 9), state.Jp[:, None]], axis=1)


def unflatten(a):
    return ParticleState(a[:, :3].copy(), a[:, 3:6].copy(), a[:, 6:15].reshape(-1, 3, 3).copy(),
                         a[:, 15:24].reshape(-1, 3, 3).copy(), a[:, 24].copy())


class LinearStepper:
    def __init__(self, capacity, drift=False):
        self.capacity = capacity
        self.states = [None] * capacity
        self.grads = [None] * capacity
        self.A = 0.96 * np.eye(25) + 0.035 * np.roll(np.eye(25), 1, axis=1)
        self.B = np.sin(np.arange(25)[:, None] + 1.37 * np.arange(7)[None]) * 1e-3
        self.param_grad = np.zeros(7)
        self.p = np.zeros(7)
        self.drift = drift
        self.reverse_started = False

    def set_parameters(self, params):
        self.p = np.array([params[n] for n in PARAMETER_NAMES])
        self.reverse_started = False

    def load_state(self, slot, state):
        self.states[slot] = flatten(state)

    def state(self, slot):
        return unflatten(self.states[slot])

    def advance(self, slot, control):
        self.states[slot + 1] = self.states[slot] @ self.A.T + (self.B @ self.p) * (1 + control)
        if self.drift and self.reverse_started:
            self.states[slot + 1] += 0.1

    def diagnostics(self):
        return {"branch_counts": {"active": 1}}

    def clear_parameter_gradients(self):
        self.param_grad[:] = 0
        self.reverse_started = True

    def clear_state_gradients(self):
        self.grads = [np.zeros((1, 25)) for _ in range(self.capacity)]

    def load_adjoint(self, slot, state, add=False):
        if add:
            self.grads[slot] += flatten(state)
        else:
            self.grads[slot] = flatten(state)

    def adjoint(self, slot):
        return unflatten(self.grads[slot])

    def seed_positions(self, slot, gradient, weight=1):
        self.grads[slot][:, :3] += weight * gradient

    def reverse_step(self, slot, control):
        self.param_grad += self.B.T @ self.grads[slot + 1].sum(axis=0) * (1 + control)
        self.grads[slot] += self.grads[slot + 1] @ self.A

    def parameter_gradients(self):
        return dict(zip(PARAMETER_NAMES, self.param_grad))


@dataclass
class Observation:
    frame_index: int
    step: int
    target: np.ndarray


class QuadraticLoss:
    def value_and_grad_positions(self, positions, observation, compute_grad=True):
        residual = positions - observation.target
        return SimpleNamespace(value=float(0.5 * np.sum(residual ** 2)),
                               gradient=residual if compute_grad else None,
                               components={}, diagnostics={})


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.initial = ParticleState.initial([[0.3, 0.4, 0.5]], np.float64)
        self.controls = np.linspace(0, 0.1, 8)
        self.obs = [Observation(i, s, np.full((1, 3), 0.2 + 0.01 * i))
                    for i, s in enumerate([0, 2, 3, 3, 4, 6, 8])]
        self.parameters = dict(DEFAULT_PARAMETERS, youngs_modulus=3.0)

    def rollout(self, length, initial=None, drift=False, ignore_recompute_mismatch=False):
        return CheckpointedRollout(LinearStepper(length + 1, drift),
                                  initial or self.initial, self.controls, self.obs, QuadraticLoss(),
                                  length, replay_rtol=0, replay_atol={n: 0 for n in self.initial.arrays()},
                                  ignore_recompute_mismatch=ignore_recompute_mismatch)

    def test_segment_lengths_match_full_history(self):
        full = self.rollout(8).value_and_gradient(self.parameters)
        for length in (1, 2, 3, 4, 7):
            result = self.rollout(length).value_and_gradient(self.parameters)
            self.assertAlmostEqual(full.value, result.value, places=14)
            np.testing.assert_allclose(list(result.gradient.values()), list(full.gradient.values()), atol=1e-14)
            np.testing.assert_allclose(flatten(result.initial_gradient), flatten(full.initial_gradient), atol=1e-14)
            self.assertEqual(result.diagnostics["observation_gradient_injections"], [1] * len(self.obs))

    def test_parameter_and_initial_state_finite_differences(self):
        rollout = self.rollout(3)
        result = rollout.value_and_gradient(self.parameters)
        for name in PARAMETER_NAMES:
            epsilon = 1e-5
            plus, minus = dict(self.parameters), dict(self.parameters)
            plus[name] += epsilon
            minus[name] -= epsilon
            fd = (rollout.value_and_gradient(plus, compute_grad=False).value -
                  rollout.value_and_gradient(minus, compute_grad=False).value) / (2 * epsilon)
            self.assertAlmostEqual(result.gradient[name], fd, delta=1e-10)
        direction = np.random.default_rng(0).normal(size=(1, 25))
        eps = 1e-6
        state_flat = flatten(self.initial)
        plus = self.rollout(3, unflatten(state_flat + eps * direction)).value_and_gradient(self.parameters, compute_grad=False)
        minus = self.rollout(3, unflatten(state_flat - eps * direction)).value_and_gradient(self.parameters, compute_grad=False)
        self.assertAlmostEqual(float(np.sum(flatten(result.initial_gradient) * direction)),
                               (plus.value - minus.value) / (2 * eps), delta=1e-9)

    def test_recomputation_drift_rejects_gradient(self):
        with self.assertRaisesRegex(InvalidStateError, "Checkpoint recomputation"):
            self.rollout(3, drift=True).value_and_gradient(self.parameters)

    def test_recomputed_count_difference_reports_segment_and_rejects_before_reverse(self):
        events, reverse_calls = [], []
        rollout = self.rollout(3)
        rollout.progress = events.append
        stepper = rollout.stepper
        stepper.diagnostics = lambda: {"branch_counts": {
            "active": 2 if stepper.reverse_started else 1, "unchanged": 4}}
        stepper.reverse_step = lambda *args: reverse_calls.append(args)
        with self.assertRaisesRegex(InvalidStateError, r"step 6 .*segment 6–8"):
            rollout.value_and_gradient(self.parameters)
        self.assertEqual(reverse_calls, [])
        failure = [event for event in events if event['phase'] == 'recompute_mismatch']
        self.assertEqual(len(failure), 1)
        event = failure[0]
        self.assertEqual(event['stage'], 'segment_forward_recompute')
        self.assertEqual(event['mismatch_kind'], 'contact_counts')
        self.assertFalse(event['ignored'])
        self.assertEqual(event['mismatch_number'], 1)
        self.assertIn('backward evaluation rejected', event['message'])
        self.assertEqual(event['step'], 6)
        self.assertEqual(event['segment_start_step'], 6)
        self.assertEqual(event['segment_end_step'], 8)
        self.assertEqual(event['expected_counts'], {'active': 1, 'unchanged': 4})
        self.assertEqual(event['recomputed_counts'], {'active': 2, 'unchanged': 4})
        self.assertEqual(event['differing_counts'], {'active': {'forward': 1, 'recomputed': 2}})

    def test_missing_count_is_not_treated_as_zero(self):
        events = []
        rollout = self.rollout(3)
        rollout.progress = events.append
        with self.assertRaises(InvalidStateError):
            rollout._check_signature({'active': 0}, {}, 1, 0, 3)
        self.assertEqual(events[0]['differing_counts'], {'active': {'forward': 0, 'recomputed': None}})

    def test_zero_step_initial_observation(self):
        observation = Observation(0, 0, np.zeros((1, 3)))
        r = CheckpointedRollout(LinearStepper(2), self.initial, [], [observation], QuadraticLoss(), 1)
        result = r.value_and_gradient(self.parameters)
        self.assertTrue(all(g == 0 for g in result.gradient.values()))
        np.testing.assert_allclose(result.initial_gradient.x, self.initial.x)

    def test_nonfinite_adjoint_reports_boundary_and_component(self):
        events = []
        rollout = self.rollout(3)
        rollout.progress = events.append
        original = rollout.stepper.reverse_step
        def invalid_reverse(slot, control):
            original(slot, control)
            if slot == 0:
                rollout.stepper.grads[slot][0, 0] = np.nan
        rollout.stepper.reverse_step = invalid_reverse
        with self.assertRaisesRegex(InvalidStateError, 'adjoint at global step 6'):
            rollout.value_and_gradient(self.parameters)
        failure = [event for event in events if event['phase'] == 'invalid_adjoint']
        self.assertEqual(len(failure), 1)
        self.assertEqual(failure[0]['arrays']['x']['nonfinite_count'], 1)
        self.assertEqual(failure[0]['arrays']['x']['first_nonfinite_indices'], [[0, 0]])
        self.assertEqual(failure[0]['arrays']['v']['nonfinite_count'], 0)

    def test_ignored_count_mismatches_continue_with_unchanged_adjoint_carry(self):
        expected = self.rollout(3).value_and_gradient(self.parameters)
        events = []
        rollout = self.rollout(3, ignore_recompute_mismatch=True)
        rollout.progress = events.append
        rollout.stepper.diagnostics = lambda: {"branch_counts": {
            "active": 2 if rollout.stepper.reverse_started else 1, "unchanged": 4}}
        result = rollout.value_and_gradient(self.parameters)
        self.assertEqual(result.value, expected.value)
        np.testing.assert_array_equal(list(result.gradient.values()), list(expected.gradient.values()))
        np.testing.assert_array_equal(flatten(result.initial_gradient), flatten(expected.initial_gradient))
        self.assertTrue(result.diagnostics["ignore_recompute_mismatch"])
        self.assertFalse(result.diagnostics["replay_consistent"])
        self.assertTrue(result.diagnostics["replay_consistency_checked"])
        self.assertEqual(result.diagnostics["recompute_mismatch_counts"],
                         {"contact_counts": 8, "checkpoint_state": 0, "observation_loss": 0})
        self.assertEqual(result.diagnostics["recompute_mismatch_count"], 8)
        mismatches = [event for event in events if event["phase"] == "recompute_mismatch"]
        self.assertEqual([event["mismatch_number"] for event in mismatches], list(range(1, 9)))
        self.assertEqual([event["step"] for event in mismatches], [6, 7, 3, 4, 5, 0, 1, 2])
        for event in mismatches:
            self.assertTrue(event["ignored"])
            self.assertEqual(event["mismatch_kind"], "contact_counts")
            self.assertEqual(event["differing_counts"], {"active": {"forward": 1, "recomputed": 2}})
            self.assertIn("continuing because ignore_recompute_mismatch=True", event["message"])
        self.assertEqual(result.diagnostics["observation_gradient_injections"], [1] * len(self.obs))

    def test_ignored_state_and_loss_mismatches_are_recorded(self):
        events = []
        rollout = self.rollout(3, drift=True, ignore_recompute_mismatch=True)
        rollout.progress = events.append
        result = rollout.value_and_gradient(self.parameters)
        self.assertTrue(np.isfinite(list(result.gradient.values())).all())
        self.assertGreater(np.linalg.norm(flatten(result.initial_gradient)), 0)
        self.assertFalse(result.diagnostics["replay_consistent"])
        self.assertEqual(result.diagnostics["recompute_mismatch_counts"],
                         {"contact_counts": 0, "checkpoint_state": 15, "observation_loss": 6})
        mismatches = [event for event in events if event["phase"] == "recompute_mismatch"]
        self.assertEqual([event["mismatch_number"] for event in mismatches], list(range(1, 22)))
        state_events = [event for event in mismatches if event["mismatch_kind"] == "checkpoint_state"]
        self.assertEqual({event["step"] for event in state_events}, {3, 6, 8})
        self.assertEqual({event["array"] for event in state_events}, set(self.initial.arrays()))
        for event in state_events:
            self.assertTrue(event["ignored"])
            self.assertGreater(event["max_abs_difference"], 0)
            self.assertEqual(event["replay_atol"], 0)
            self.assertIn("Checkpoint recomputation differs", event["message"])
        for event in mismatches:
            if event["mismatch_kind"] == "observation_loss":
                self.assertTrue(event["ignored"])
                self.assertNotEqual(event["expected_value"], event["recomputed_value"])
                self.assertEqual(event["max_abs_difference"], abs(event["expected_value"] - event["recomputed_value"]))
                self.assertEqual(event["step"], self.obs[event["frame_index"]].step)
                self.assertEqual(event["replay_atol"], 1e-8)

    def test_loss_mismatch_default_rejects_and_opt_in_continues(self):
        class ChangedLoss(QuadraticLoss):
            def value_and_grad_positions(self, positions, observation, compute_grad=True):
                record = super().value_and_grad_positions(positions, observation, compute_grad)
                if compute_grad:
                    record.value += 1.0
                return record

        for ignore in (False, True):
            with self.subTest(ignore=ignore):
                events = []
                rollout = self.rollout(3, ignore_recompute_mismatch=ignore)
                rollout.loss = ChangedLoss()
                rollout.progress = events.append
                if ignore:
                    result = rollout.value_and_gradient(self.parameters)
                    self.assertEqual(result.diagnostics["recompute_mismatch_counts"],
                                     {"contact_counts": 0, "checkpoint_state": 0, "observation_loss": 7})
                    self.assertFalse(result.diagnostics["replay_consistent"])
                else:
                    with self.assertRaisesRegex(InvalidStateError, "Loss recomputation differs at frame 6"):
                        rollout.value_and_gradient(self.parameters)
                mismatches = [event for event in events if event["phase"] == "recompute_mismatch"]
                self.assertEqual(mismatches[0]["mismatch_kind"], "observation_loss")
                self.assertEqual(mismatches[0]["ignored"], ignore)
                self.assertEqual(mismatches[0]["step"], 8)
                self.assertEqual(mismatches[0]["mismatch_number"], 1)

    def test_endpoint_default_reports_and_rejects_first_array(self):
        events = []
        rollout = self.rollout(3, drift=True)
        rollout.progress = events.append
        with self.assertRaisesRegex(InvalidStateError, "Checkpoint recomputation differs at step 8, x"):
            rollout.value_and_gradient(self.parameters)
        mismatches = [event for event in events if event["phase"] == "recompute_mismatch"]
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]["mismatch_kind"], "checkpoint_state")
        self.assertEqual(mismatches[0]["array"], "x")
        self.assertFalse(mismatches[0]["ignored"])

    def test_mismatch_counts_reset_without_mutating_previous_result(self):
        events = []
        rollout = self.rollout(3, drift=True, ignore_recompute_mismatch=True)
        rollout.progress = events.append
        first = rollout.value_and_gradient(self.parameters)
        saved_counts = dict(first.diagnostics["recompute_mismatch_counts"])
        self.assertEqual(saved_counts, {"contact_counts": 0, "checkpoint_state": 15, "observation_loss": 6})
        rollout.stepper.drift = False
        second = rollout.value_and_gradient(self.parameters)
        self.assertTrue(second.diagnostics["replay_consistent"])
        self.assertEqual(second.diagnostics["recompute_mismatch_count"], 0)
        self.assertEqual(first.diagnostics["recompute_mismatch_counts"], saved_counts)
        self.assertIsNot(first.diagnostics["recompute_mismatch_counts"], second.diagnostics["recompute_mismatch_counts"])
        events.clear()
        rollout.stepper.diagnostics = lambda: {"active": 2 if rollout.stepper.reverse_started else 1}
        third = rollout.value_and_gradient(self.parameters)
        mismatches = [event for event in events if event["phase"] == "recompute_mismatch"]
        self.assertEqual(mismatches[0]["mismatch_number"], 1)
        self.assertEqual(third.diagnostics["recompute_mismatch_count"], 8)
        forward = rollout.value_and_gradient(self.parameters, compute_grad=False)
        self.assertFalse(forward.diagnostics["replay_consistency_checked"])
        self.assertEqual(forward.diagnostics["recompute_mismatch_count"], 0)
        self.assertEqual(first.diagnostics["recompute_mismatch_counts"], saved_counts)
        self.assertEqual(third.diagnostics["recompute_mismatch_counts"]["contact_counts"], 8)
        self.assertFalse(first.diagnostics["replay_consistent"])

    def test_ignore_never_accepts_nonfinite_counts_states_or_adjoints(self):
        for failure in ("count", "state", "adjoint", "parameter_gradient", "invalid_step"):
            with self.subTest(failure=failure):
                rollout = self.rollout(3, ignore_recompute_mismatch=True)
                stepper = rollout.stepper
                if failure == "count":
                    stepper.diagnostics = lambda: {"active": np.nan if stepper.reverse_started else 1}
                elif failure in ("state", "invalid_step"):
                    advance = stepper.advance
                    def invalid_advance(slot, control):
                        advance(slot, control)
                        if stepper.reverse_started:
                            if failure == "invalid_step":
                                raise InvalidStateError("Injected invalid deformation")
                            stepper.states[slot + 1][0, 15] = np.nan
                    stepper.advance = invalid_advance
                elif failure == "adjoint":
                    reverse = stepper.reverse_step
                    def invalid_reverse(slot, control):
                        reverse(slot, control)
                        stepper.grads[slot][0, 0] = np.inf
                    stepper.reverse_step = invalid_reverse
                else:
                    stepper.parameter_gradients = lambda: dict.fromkeys(PARAMETER_NAMES, np.nan)
                with self.assertRaises(InvalidStateError):
                    rollout.value_and_gradient(self.parameters)

    def test_ignore_never_accepts_nonfinite_loss_or_invalid_loss_gradient(self):
        for failure in ("forward_loss", "recomputed_loss", "gradient_nan", "gradient_dimensions"):
            with self.subTest(failure=failure):
                rollout = self.rollout(3, ignore_recompute_mismatch=True)
                evaluate = rollout.loss.value_and_grad_positions
                def invalid_loss(positions, observation, compute_grad=True):
                    record = evaluate(positions, observation, compute_grad)
                    if failure == "forward_loss" or (failure == "recomputed_loss" and compute_grad):
                        record.value = np.nan
                    if compute_grad and failure == "gradient_nan":
                        record.gradient[:] = np.nan
                    if compute_grad and failure == "gradient_dimensions":
                        record.gradient = np.zeros((1, 2))
                    return record
                rollout.loss.value_and_grad_positions = invalid_loss
                with self.assertRaises(InvalidStateError):
                    rollout.value_and_gradient(self.parameters)

    def test_ignore_flag_requires_an_explicit_boolean(self):
        for value in ("false", "true", 1, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be a boolean"):
                self.rollout(3, ignore_recompute_mismatch=value)

    def test_memory_is_bounded_by_segment(self):
        estimate = estimate_memory(24000, 10000, 64)
        self.assertLess(estimate["local_particle_history_and_adjoint_bytes"], 400_000_000)
        self.assertGreater(estimate["full_particle_history_and_adjoint_bytes"], 40_000_000_000)


if __name__ == "__main__":
    unittest.main()
