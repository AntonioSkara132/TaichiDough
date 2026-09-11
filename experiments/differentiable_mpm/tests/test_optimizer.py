"""Tests of parameter coordinates and supplied-gradient optimization (no simulator)."""
from copy import deepcopy
import json
import math
import unittest

import numpy as np

from experiments.differentiable_mpm.optimize import AdamOptions, ObjectiveValue, ProjectedAdam
from experiments.differentiable_mpm.parameters import PhysicalParameterSpace
from experiments.differentiable_mpm.state import DEFAULT_PARAMETERS, InvalidStateError, PARAMETER_NAMES


def viscosity_space(initial=0.1):
    parameters = dict(DEFAULT_PARAMETERS, viscosity=initial)
    return PhysicalParameterSpace(parameters, ["viscosity"], "none",
                                  bounds={"viscosity": (0.0, 1.0)}, scales={"viscosity": 1.0})


def viscosity_quadratic(target):
    def objective(parameters):
        difference = parameters["viscosity"] - target
        return ObjectiveValue(0.5 * difference ** 2, {"viscosity": difference},
                              {"analytic_test_objective": True})
    return objective


class ParameterTests(unittest.TestCase):
    def test_roundtrip_and_zero_viscosity(self):
        space = PhysicalParameterSpace(DEFAULT_PARAMETERS, PARAMETER_NAMES, "stretch-clamp")
        actual = space.physical(space.coordinates())
        for name in PARAMETER_NAMES:
            self.assertAlmostEqual(actual[name], DEFAULT_PARAMETERS[name], places=8)
        self.assertEqual(actual["viscosity"], 0.0)
        self.assertEqual(space.coordinates()[2], 0.0)
        self.assertEqual(space.settings(), json.loads(json.dumps(space.serialize(), allow_nan=False)))

    def test_all_coordinate_derivatives(self):
        initial = dict(DEFAULT_PARAMETERS, viscosity=13.0)
        space = PhysicalParameterSpace(initial, PARAMETER_NAMES, "stretch-clamp")
        u = space.coordinates()
        physical_gradient = dict(zip(PARAMETER_NAMES, [2e-5, -0.7, 0.05, 2.0, -4.0, 0.9, -0.4]))
        expected = space.pullback(u, physical_gradient)
        finite_difference = np.zeros_like(u)
        h = 1e-5
        for i in range(len(u)):
            offset = np.zeros_like(u)
            offset[i] = h
            plus, minus = space.physical(u + offset), space.physical(u - offset)
            finite_difference[i] = sum(physical_gradient[name] * (plus[name] - minus[name])
                                       for name in PARAMETER_NAMES) / (2 * h)
        np.testing.assert_allclose(expected, finite_difference, rtol=1e-8, atol=1e-9)
        zero = viscosity_space(0)
        self.assertEqual(zero.pullback(zero.coordinates(), {"viscosity": -2})[0], -2)
        self.assertLess(expected[3], 0)

    def test_projection_and_fixed_values(self):
        space = viscosity_space()
        self.assertEqual(space.project([-5])[0], 0.0)
        self.assertEqual(space.project([5])[0], 1.0)
        physical = space.physical(space.project([5]))
        self.assertEqual(physical["viscosity"], 1.0)
        for name in set(PARAMETER_NAMES) - {"viscosity"}:
            self.assertEqual(physical[name], DEFAULT_PARAMETERS[name])
        with self.assertRaisesRegex(ValueError, "outside"):
            space.physical([-0.1])

    def test_plastic_coordinate_bounds(self):
        space = PhysicalParameterSpace(DEFAULT_PARAMETERS, ["plastic_min", "plastic_max"], "stretch-clamp")
        low = space.physical(space.lower)
        high = space.physical(space.upper)
        self.assertEqual(low["plastic_min"], 1.0)
        self.assertEqual(low["plastic_max"], 1.0)
        self.assertEqual(high["plastic_min"], 0.5)
        self.assertEqual(high["plastic_max"], 1.5)

    def test_reject_inactive_duplicate_unknown_or_empty_parameters(self):
        for fit in (["plastic_min"], ["plastic_max"]):
            with self.assertRaisesRegex(ValueError, "explicit stretch-clamp"):
                PhysicalParameterSpace(DEFAULT_PARAMETERS, fit, "none")
        for fit in ([], ["viscosity", "viscosity"], ["yield_stress"], "viscosity"):
            with self.assertRaises(ValueError):
                PhysicalParameterSpace(DEFAULT_PARAMETERS, fit, "none")

    def test_reject_invalid_bounds_scales_and_gradients(self):
        for bounds in ({"poisson_ratio": (0.1, 0.5)}, {"viscosity": (-1, 1)},
                       {"plastic_min": (0, 1)}, {"tool_retention": (0, 2)},
                       {"youngs_modulus": (100, math.inf)}, {"viscosity": (2, 1)},
                       {"unknown": (0, 1)}):
            with self.assertRaises(ValueError):
                PhysicalParameterSpace(DEFAULT_PARAMETERS, ["viscosity"], "none", bounds=bounds)
        for scales in ({"viscosity": 0}, {"viscosity": -2}, {"viscosity": math.nan}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                PhysicalParameterSpace(DEFAULT_PARAMETERS, ["viscosity"], "none", scales=scales)
        space = viscosity_space()
        for gradient in ({}, {"viscosity": math.nan}, {"viscosity": 1, "unknown": 2}):
            with self.assertRaises(ValueError):
                space.pullback(space.coordinates(), gradient)
        with self.assertRaises(ValueError):
            space.project([math.nan])


class OptimizerTests(unittest.TestCase):
    def test_joint_analytic_optimization_uses_supplied_gradients(self):
        initial = dict(DEFAULT_PARAMETERS, viscosity=25)
        targets = dict(initial, youngs_modulus=90000, poisson_ratio=0.34, viscosity=5.0,
                       plastic_min=0.84, plastic_max=1.18, tool_retention=0.45, floor_retention=0.55)
        normalizers = dict(zip(PARAMETER_NAMES, [1e5, 0.1, 100, 0.1, 0.1, 1, 1]))
        evaluations = []

        def objective(parameters):
            evaluations.append(dict(parameters))
            value = sum(0.5 * ((parameters[name] - targets[name]) / normalizers[name]) ** 2
                        for name in PARAMETER_NAMES)
            gradient = {name: (parameters[name] - targets[name]) / normalizers[name] ** 2
                        for name in PARAMETER_NAMES}
            return ObjectiveValue(value, gradient)

        space = PhysicalParameterSpace(initial, PARAMETER_NAMES, "stretch-clamp")
        optimizer = ProjectedAdam(space, objective, AdamOptions(learning_rate=0.1))
        initial_value = optimizer.initialize().value
        result = optimizer.run(250)
        self.assertLess(result.best_value, initial_value * 1e-4)
        self.assertEqual(result.evaluations, len(evaluations))
        self.assertGreater(result.accepted_updates, 10)
        self.assertTrue(all(event["valid"] for event in result.history if event["type"] == "evaluation"))
        accepted = [event["value_after"] for event in result.history if event["type"] == "step" and event["accepted"]]
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(accepted, accepted[1:])))
        json.dumps(optimizer.state_dict(), allow_nan=False)

    def test_nonzero_gradient_at_constraint_is_converged(self):
        optimizer = ProjectedAdam(viscosity_space(0.3), viscosity_quadratic(-0.2), AdamOptions(learning_rate=0.2))
        result = optimizer.run(10)
        self.assertEqual(result.best_parameters["viscosity"], 0.0)
        self.assertEqual(result.status, "converged_gradient")
        self.assertGreater(optimizer.current.gradient["viscosity"], 0)
        self.assertGreater(result.best_value, 0)

    def test_invalid_proposal_backtracks_then_accepts(self):
        def objective(parameters):
            if parameters["viscosity"] > 0.8:
                raise InvalidStateError("Synthetic unstable candidate")
            return viscosity_quadratic(0.7)(parameters)

        optimizer = ProjectedAdam(viscosity_space(), objective, AdamOptions(learning_rate=1.0))
        optimizer.initialize()
        event = optimizer.step()
        self.assertTrue(event["accepted"])
        self.assertFalse(event["attempts"][0]["valid"])
        self.assertEqual(event["attempts"][0]["reason"], "invalid_objective")
        self.assertAlmostEqual(optimizer.u[0], 0.6, places=6)
        self.assertEqual(optimizer.accepted_updates, 1)
        self.assertEqual(optimizer.evaluations, 3)
        self.assertEqual(optimizer.learning_rate, 0.5)

    def test_all_invalid_candidates_rollback_moments_and_state(self):
        def objective(parameters):
            if parameters["viscosity"] != 0.1:
                raise InvalidStateError("Only the initial candidate is valid")
            return viscosity_quadratic(0.7)(parameters)

        optimizer = ProjectedAdam(viscosity_space(), objective, AdamOptions(max_backtracks=2))
        optimizer.initialize()
        before = optimizer.state_dict()
        result = optimizer.run(10)
        np.testing.assert_array_equal(optimizer.u, before["coordinates"])
        np.testing.assert_array_equal(optimizer.m, before["first_moment"])
        np.testing.assert_array_equal(optimizer.v, before["second_moment"])
        self.assertEqual(result.best_value, before["best"]["value"])
        self.assertEqual(result.current_value, before["current"]["value"])
        self.assertEqual(result.accepted_updates, 0)
        self.assertEqual(result.status, "stalled_invalid")
        self.assertEqual(result.evaluations, 4)
        state = json.loads(json.dumps(optimizer.state_dict(), allow_nan=False))
        restored = ProjectedAdam(viscosity_space(), objective, optimizer.options)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)

    def test_nonfinite_candidate_value_or_gradient_is_not_success(self):
        for invalid_gradient in (False, True):
            def objective(parameters):
                if parameters["viscosity"] != 0.1:
                    if invalid_gradient:
                        return ObjectiveValue(0.0, {"viscosity": math.nan})
                    return ObjectiveValue(math.nan, {"viscosity": 0.0})
                return viscosity_quadratic(0.7)(parameters)

            optimizer = ProjectedAdam(viscosity_space(), objective, AdamOptions(max_backtracks=0))
            result = optimizer.run(3)
            self.assertEqual(result.status, "stalled_invalid")
            self.assertGreater(result.best_value, 0)
            self.assertEqual(result.best_parameters["viscosity"], 0.1)
            self.assertFalse([event for event in result.history if event["type"] == "evaluation"][-1]["valid"])

    def test_gradient_overflow_is_an_invalid_candidate(self):
        space = viscosity_space()
        for derivative in (1e155, 1e308):
            def objective(parameters):
                if parameters["viscosity"] != 0.1:
                    return ObjectiveValue(0.0, {"viscosity": derivative})
                return viscosity_quadratic(0.7)(parameters)

            optimizer = ProjectedAdam(space, objective, AdamOptions(max_backtracks=0))
            result = optimizer.run(1)
            self.assertEqual(result.status, "stalled_invalid")
            self.assertEqual(result.accepted_updates, 0)
            self.assertEqual(result.best_parameters["viscosity"], 0.1)
            json.dumps(optimizer.state_dict(), allow_nan=False)

    def test_invalid_initial_objective_raises_and_remains_uninitialized(self):
        for invalid in (ObjectiveValue(math.inf, {"viscosity": 1}), ObjectiveValue(0.0, {"viscosity": math.nan})):
            optimizer = ProjectedAdam(viscosity_space(), lambda parameters: invalid)
            with self.assertRaisesRegex(InvalidStateError, "Initial objective is invalid"):
                optimizer.run(2)
            self.assertIsNone(optimizer.best_value)
            self.assertIsNone(optimizer.current)
            self.assertEqual(optimizer.status, "invalid_initial")
            self.assertEqual(optimizer.evaluations, 1)

    def test_resume_matches_uninterrupted_coordinate_moments_and_calls(self):
        options = AdamOptions(learning_rate=0.03, gradient_tolerance=0)
        objective = viscosity_quadratic(0.8)
        continuous = ProjectedAdam(viscosity_space(), objective, options, objective_id="analytic-v1")
        continuous.run(12)
        partial = ProjectedAdam(viscosity_space(), objective, options, objective_id="analytic-v1")
        partial.run(5)
        saved = json.loads(json.dumps(partial.state_dict(), allow_nan=False))
        resumed = ProjectedAdam(viscosity_space(), objective, options, objective_id="analytic-v1")
        resumed.load_state_dict(saved)
        resumed.run(7)
        np.testing.assert_array_equal(continuous.u, resumed.u)
        np.testing.assert_array_equal(continuous.m, resumed.m)
        np.testing.assert_array_equal(continuous.v, resumed.v)
        self.assertEqual(continuous.current.value, resumed.current.value)
        self.assertEqual(continuous.best_value, resumed.best_value)
        self.assertEqual(continuous.accepted_updates, resumed.accepted_updates)
        self.assertEqual(continuous.evaluations, resumed.evaluations)

    def test_restore_rejects_inconsistent_state_without_mutating_target(self):
        options = AdamOptions(learning_rate=0.03)
        source = ProjectedAdam(viscosity_space(), viscosity_quadratic(0.8), options, objective_id="test")
        source.run(3)
        state = source.state_dict()
        mutations = [
            lambda s: s["physical_parameters"].update(viscosity=0.9),
            lambda s: s["space"]["initial"].update(viscosity=0.9),
            lambda s: s["options"].update(learning_rate=0.2),
            lambda s: s.update(objective_id="different-objective"),
            lambda s: s.update(second_moment=[-1]),
            lambda s: s.update(first_moment=[0.9]),
            lambda s: s["current"]["gradient"].update(viscosity=0.9),
            lambda s: s["current"].update(value=100.0),
            lambda s: s.update(status="running"),
            lambda s: s["best"]["parameters"].update(viscosity=0.9),
            lambda s: s.update(accepted_updates=8),
            lambda s: s["history"].pop(0),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                target = ProjectedAdam(viscosity_space(), viscosity_quadratic(0.8), options, objective_id="test")
                before = target.state_dict()
                corrupted = deepcopy(state)
                mutate(corrupted)
                with self.assertRaises((ValueError, InvalidStateError)):
                    target.load_state_dict(corrupted)
                self.assertEqual(target.state_dict(), before)

    def test_every_callback_state_is_resumable(self):
        snapshots = []
        options = AdamOptions(learning_rate=1.0)

        def objective(parameters):
            if parameters["viscosity"] > 0.8:
                raise InvalidStateError("Too large")
            return viscosity_quadratic(0.7)(parameters)

        optimizer = ProjectedAdam(viscosity_space(), objective, options,
                                  callback=lambda event, state: snapshots.append((event, state)))
        optimizer.run(5)
        self.assertGreater(len(snapshots), 2)
        for event, snapshot in snapshots:
            self.assertNotEqual(event["type"], "evaluation")
            resumed = ProjectedAdam(viscosity_space(), objective, options)
            resumed.load_state_dict(json.loads(json.dumps(snapshot, allow_nan=False)))
            self.assertEqual(resumed.state_dict(), snapshot)

    def test_evaluation_budget_and_iteration_budget_are_not_convergence(self):
        optimizer = ProjectedAdam(viscosity_space(), viscosity_quadratic(0.8),
                                  AdamOptions(max_evaluations=2, learning_rate=0.01))
        result = optimizer.run(100)
        self.assertEqual(result.evaluations, 2)
        self.assertEqual(result.status, "evaluation_budget_exhausted")
        limited = ProjectedAdam(viscosity_space(), viscosity_quadratic(0.8))
        result = limited.run(0)
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.evaluations, 1)

    def test_missing_active_gradient_is_a_programming_error(self):
        optimizer = ProjectedAdam(viscosity_space(), lambda parameters: ObjectiveValue(1.0, {}))
        with self.assertRaisesRegex(ValueError, "gradient keys"):
            optimizer.run(2)
        self.assertEqual(optimizer.evaluations, 1)
        self.assertFalse(optimizer.history[-1]["valid"])

    def test_invalid_optimizer_settings(self):
        for options in ({"learning_rate": 0}, {"beta1": 1}, {"epsilon": 0},
                        {"max_backtracks": -1}, {"max_evaluations": 0},
                        {"backtrack_factor": 1}, {"loss_tolerance": -1}):
            with self.assertRaises(ValueError):
                AdamOptions(**options)


class RecoveryPolicyTests(unittest.TestCase):
    @staticmethod
    def limited_objective(parameters):
        if parameters["viscosity"] > 0.8:
            raise InvalidStateError("Synthetic unstable proposal")
        return viscosity_quadratic(0.7)(parameters)

    @staticmethod
    def rosenbrock_problem():
        initial = dict(DEFAULT_PARAMETERS, viscosity=0.1, tool_retention=0.8)
        space = PhysicalParameterSpace(initial, ["viscosity", "tool_retention"], "none",
                                       bounds={"viscosity": [0, 1]}, scales={"viscosity": 1})

        def objective(parameters):
            x, y = parameters["viscosity"], parameters["tool_retention"]
            residual = y - x * x
            return ObjectiveValue(100 * residual * residual + (1 - x) ** 2,
                                  {"viscosity": -400 * x * residual + 2 * (x - 1),
                                   "tool_retention": 200 * residual})
        return space, objective

    def test_default_persistent_policy_preserves_update_behavior(self):
        implicit = ProjectedAdam(viscosity_space(), self.limited_objective, AdamOptions(learning_rate=1))
        explicit = ProjectedAdam(viscosity_space(), self.limited_objective,
                                 AdamOptions(learning_rate=1, learning_rate_policy="persistent-v1"))
        implicit.run(3)
        explicit.run(3)
        self.assertEqual(AdamOptions().learning_rate_policy, "persistent-v1")
        self.assertEqual(AdamOptions().learning_rate_growth, 1.25)
        np.testing.assert_array_equal(implicit.u, explicit.u)
        np.testing.assert_array_equal(implicit.m, explicit.m)
        np.testing.assert_array_equal(implicit.v, explicit.v)
        self.assertEqual(implicit.evaluations, explicit.evaluations)
        steps = [event for event in implicit.history if event["type"] == "step"]
        self.assertEqual(steps[0]["proposal_learning_rate"], 1)
        self.assertEqual(steps[0]["accepted_learning_rate"], 0.5)
        for previous, current in zip(steps, steps[1:]):
            self.assertEqual(current["proposal_learning_rate"], previous["learning_rate"])

    def test_recovery_logs_growth_cap_and_actual_accepted_rate(self):
        options = AdamOptions(learning_rate=1, learning_rate_policy="recover-v1")
        optimizer = ProjectedAdam(viscosity_space(), self.limited_objective, options)
        optimizer.run(4)
        steps = [event for event in optimizer.history if event["type"] == "step"]
        self.assertEqual(steps[0]["proposal_learning_rate"], 1)
        self.assertEqual(steps[0]["accepted_learning_rate"], 0.5)
        self.assertEqual(steps[1]["proposal_learning_rate"], 0.625)
        self.assertLess(steps[1]["accepted_learning_rate"], steps[1]["proposal_learning_rate"])
        self.assertTrue(any(not attempt["valid"] for attempt in steps[1]["attempts"]))
        retained = options.learning_rate
        previous_coordinates = optimizer.space.coordinates()
        for event in steps:
            self.assertEqual(event["learning_rate_policy"], "recover-v1")
            self.assertEqual(event["proposal_learning_rate"], min(options.learning_rate, retained * 1.25))
            self.assertLessEqual(event["proposal_learning_rate"], options.learning_rate)
            if event["accepted"]:
                accepted = [attempt for attempt in event["attempts"] if attempt["accepted"]]
                self.assertEqual(len(accepted), 1)
                self.assertEqual(event["accepted_learning_rate"], accepted[0]["learning_rate"])
                self.assertEqual(event["learning_rate"], event["accepted_learning_rate"])
                delta = np.asarray(event["coordinates"]) - previous_coordinates
                slope = float(np.dot(event["coordinate_gradient_before"], delta))
                limit = event["value_before"] + options.armijo * slope + options.loss_tolerance
                self.assertLessEqual(event["value_after"], limit)
            else:
                self.assertIsNone(event["accepted_learning_rate"])
            retained = event["learning_rate"]
            previous_coordinates = np.asarray(event["coordinates"])

    def test_invalid_recovery_attempt_restores_previous_nonzero_moments(self):
        def objective(parameters):
            if parameters["viscosity"] > 0.60001:
                raise InvalidStateError("Beyond the fixed valid interval")
            return viscosity_quadratic(0.7)(parameters)

        options = AdamOptions(learning_rate=1, max_backtracks=2, learning_rate_policy="recover-v1")
        optimizer = ProjectedAdam(viscosity_space(), objective, options)
        first = optimizer.step()
        self.assertTrue(first["accepted"])
        self.assertTrue(np.any(optimizer.m != 0))
        before = optimizer.state_dict()
        rejected = optimizer.step()
        self.assertEqual(rejected["status"], "stalled_invalid")
        self.assertFalse(rejected["accepted"])
        self.assertIsNone(rejected["accepted_learning_rate"])
        self.assertEqual(rejected["proposal_learning_rate"], 0.625)
        self.assertEqual(optimizer.accepted_updates, 1)
        np.testing.assert_array_equal(optimizer.u, before["coordinates"])
        np.testing.assert_array_equal(optimizer.m, before["first_moment"])
        np.testing.assert_array_equal(optimizer.v, before["second_moment"])
        self.assertEqual(optimizer.current.value, before["current"]["value"])
        self.assertEqual(optimizer.best_value, before["best"]["value"])
        restored = ProjectedAdam(viscosity_space(), objective, options)
        state = json.loads(json.dumps(optimizer.state_dict(), allow_nan=False))
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)

    def test_recovery_budget_exhaustion_during_backtracking_is_resumable(self):
        options = AdamOptions(learning_rate=1, max_evaluations=2, learning_rate_policy="recover-v1")
        optimizer = ProjectedAdam(viscosity_space(), self.limited_objective, options)
        result = optimizer.run(10)
        self.assertEqual(result.status, "evaluation_budget_exhausted")
        self.assertEqual(result.evaluations, 2)
        self.assertEqual(result.accepted_updates, 0)
        np.testing.assert_array_equal(optimizer.u, optimizer.space.coordinates())
        np.testing.assert_array_equal(optimizer.m, np.zeros_like(optimizer.u))
        np.testing.assert_array_equal(optimizer.v, np.zeros_like(optimizer.u))
        state = json.loads(json.dumps(optimizer.state_dict(), allow_nan=False))
        restored = ProjectedAdam(viscosity_space(), self.limited_objective, options)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)
        calls = restored.evaluations
        restored.run(2)
        self.assertEqual(restored.evaluations, calls)

    def test_exact_recovery_resume_after_backtracking(self):
        space, objective = self.rosenbrock_problem()
        options = AdamOptions(learning_rate=0.03, gradient_tolerance=0,
                              learning_rate_policy="recover-v1", learning_rate_growth=1.25)
        uninterrupted = ProjectedAdam(space, objective, options, objective_id="recovery-resume-v1")
        uninterrupted.run(85)
        partial = ProjectedAdam(space, objective, options, objective_id="recovery-resume-v1")
        partial.run(55)
        self.assertLess(partial.learning_rate, options.learning_rate)
        state = json.loads(json.dumps(partial.state_dict(), allow_nan=False))
        resumed = ProjectedAdam(space, objective, options, objective_id="recovery-resume-v1")
        resumed.load_state_dict(state)
        resumed.run(30)
        for name in ("u", "m", "v", "best_u"):
            np.testing.assert_array_equal(getattr(uninterrupted, name), getattr(resumed, name))
        self.assertEqual(uninterrupted.current.value, resumed.current.value)
        self.assertEqual(uninterrupted.best_value, resumed.best_value)
        self.assertEqual(uninterrupted.evaluations, resumed.evaluations)
        self.assertEqual(uninterrupted.accepted_updates, resumed.accepted_updates)
        self.assertEqual(uninterrupted.learning_rate, resumed.learning_rate)
        self.assertEqual([event for event in uninterrupted.history if event["type"] == "step"],
                         [event for event in resumed.history if event["type"] == "step"])

    def test_recovery_callbacks_are_complete_resumable_snapshots(self):
        snapshots = []
        options = AdamOptions(learning_rate=1, learning_rate_policy="recover-v1")
        optimizer = ProjectedAdam(viscosity_space(), self.limited_objective, options,
                                  callback=lambda event, state: snapshots.append((event, state)))
        optimizer.run(4)
        for event, state in snapshots:
            self.assertNotEqual(event["type"], "evaluation")
            self.assertEqual(state["options"]["learning_rate_policy"], "recover-v1")
            self.assertEqual(state["options"]["learning_rate_growth"], 1.25)
            restored = ProjectedAdam(viscosity_space(), self.limited_objective, options)
            restored.load_state_dict(json.loads(json.dumps(state, allow_nan=False)))
            self.assertEqual(restored.state_dict(), state)

    def test_resume_rejects_policy_growth_and_rate_history_changes(self):
        options = AdamOptions(learning_rate=1, learning_rate_policy="recover-v1")
        source = ProjectedAdam(viscosity_space(), self.limited_objective, options)
        source.run(3)
        state = source.state_dict()
        for mismatched in (AdamOptions(learning_rate=1),
                           AdamOptions(learning_rate=1, learning_rate_policy="recover-v1", learning_rate_growth=1.5)):
            target = ProjectedAdam(viscosity_space(), self.limited_objective, mismatched)
            with self.assertRaisesRegex(ValueError, "options do not match"):
                target.load_state_dict(state)
        mutations = [
            lambda saved: saved.update(learning_rate=saved["learning_rate"] * 0.75),
            lambda saved: next(e for e in saved["history"] if e["type"] == "step").update(proposal_learning_rate=0.75),
            lambda saved: next(e for e in saved["history"] if e["type"] == "step").update(accepted_learning_rate=None),
            lambda saved: next(e for e in saved["history"] if e["type"] == "step").update(learning_rate_policy="persistent-v1"),
        ]
        for mutate in mutations:
            corrupted = deepcopy(state)
            mutate(corrupted)
            target = ProjectedAdam(viscosity_space(), self.limited_objective, options)
            before = target.state_dict()
            with self.assertRaises(ValueError):
                target.load_state_dict(corrupted)
            self.assertEqual(target.state_dict(), before)

    def test_recovery_improves_smooth_control_without_claiming_convergence(self):
        space, objective = self.rosenbrock_problem()
        persistent = ProjectedAdam(space, objective, AdamOptions(learning_rate=0.03))
        recovered = ProjectedAdam(space, objective, AdamOptions(learning_rate=0.03, learning_rate_policy="recover-v1"))
        baseline = persistent.run(1000)
        result = recovered.run(1000)
        self.assertLess(result.best_value, baseline.best_value * 0.2)
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(baseline.status, "budget_exhausted")
        self.assertEqual(result.accepted_updates, 1000)
        self.assertLessEqual(result.evaluations, baseline.evaluations + 10)

    def test_recovery_policy_validation_and_unit_growth_ablation(self):
        for values in ({"learning_rate_policy": "automatic"}, {"learning_rate_policy": "recover-v2"},
                       {"learning_rate_growth": 0.9}, {"learning_rate_growth": math.nan},
                       {"learning_rate_growth": math.inf}):
            with self.assertRaises(ValueError):
                AdamOptions(**values)
        a = ProjectedAdam(viscosity_space(), self.limited_objective, AdamOptions(learning_rate=1))
        b = ProjectedAdam(viscosity_space(), self.limited_objective,
                          AdamOptions(learning_rate=1, learning_rate_policy="recover-v1", learning_rate_growth=1))
        a.run(3)
        b.run(3)
        np.testing.assert_array_equal(a.u, b.u)
        np.testing.assert_array_equal(a.m, b.m)
        np.testing.assert_array_equal(a.v, b.v)
        self.assertEqual(a.evaluations, b.evaluations)


if __name__ == "__main__":
    unittest.main()
