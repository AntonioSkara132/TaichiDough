"""Projected Adam with explicit objective gradients and checked candidate steps.

The objective owns simulation and differentiation. This module never estimates
its gradients. Callbacks run only after initialization or a complete optimizer
attempt, so each supplied state can be resumed without a half-accepted update.
"""
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
import time

import numpy as np

from .parameters import PhysicalParameterSpace
from .state import InvalidStateError, PARAMETER_NAMES


@dataclass
class ObjectiveValue:
    value: float
    gradient: dict[str, float]
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AdamOptions:
    learning_rate: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_backtracks: int = 8
    backtrack_factor: float = 0.5
    min_learning_rate: float = 1e-8
    gradient_tolerance: float = 1e-6
    step_tolerance: float = 1e-12
    armijo: float = 1e-4
    loss_tolerance: float = 1e-12
    max_evaluations: int | None = None
    learning_rate_policy: str = "persistent-v1"
    learning_rate_growth: float = 1.25

    def __post_init__(self):
        numbers = (self.learning_rate, self.beta1, self.beta2, self.epsilon,
                   self.backtrack_factor, self.min_learning_rate, self.gradient_tolerance,
                   self.step_tolerance, self.armijo, self.loss_tolerance, self.learning_rate_growth)
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("Adam options must be finite")
        if self.learning_rate <= 0 or not 0 < self.min_learning_rate <= self.learning_rate:
            raise ValueError("Invalid learning-rate range")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1 or self.epsilon <= 0:
            raise ValueError("Invalid Adam moment coefficients or epsilon")
        if not 0 < self.backtrack_factor < 1 or not 0 <= self.armijo < 1:
            raise ValueError("Invalid backtracking or Armijo coefficient")
        if min(self.gradient_tolerance, self.step_tolerance, self.loss_tolerance) < 0:
            raise ValueError("Convergence and numerical tolerances must be nonnegative")
        if type(self.max_backtracks) is not int or self.max_backtracks < 0:
            raise ValueError("max_backtracks must be a nonnegative integer")
        if self.max_evaluations is not None and (type(self.max_evaluations) is not int or self.max_evaluations < 1):
            raise ValueError("max_evaluations must be a positive integer or None")
        if self.learning_rate_policy not in ("persistent-v1", "recover-v1"):
            raise ValueError("learning_rate_policy must be persistent-v1 or recover-v1")
        if self.learning_rate_growth < 1:
            raise ValueError("learning_rate_growth must be at least one")


@dataclass
class OptimizationResult:
    best_parameters: dict[str, float]
    best_value: float
    current_parameters: dict[str, float]
    current_value: float
    status: str
    iterations: int
    accepted_updates: int
    evaluations: int
    history: list[dict]


class _EvaluationBudgetExhausted(Exception):
    pass


def _plain(value):
    """Convert numerical diagnostics to JSON-compatible values without NaN JSON."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Diagnostic keys must be strings")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Diagnostic value {type(value).__name__} is not JSON-compatible")


class ProjectedAdam:
    """Minimize a fixed training objective, retaining a valid best candidate.

    ``run(iterations)`` makes at most that many additional optimizer attempts.
    Initial and proposal evaluations all count against ``max_evaluations``.
    Rejected proposals never change coordinates, moments, or accepted-update
    count. The default ``persistent-v1`` starts each attempt at the retained
    learning rate. Opt-in ``recover-v1`` first multiplies it by the growth factor,
    capped at the configured initial rate. Every proposal still undergoes the
    same gradient validity, Armijo and backtracking checks. Logs distinguish the
    first proposal rate from the rate actually accepted; neither policy changes
    the convergence criterion. If momentum does not point downhill after
    projection, use the current gradient with the same second-moment
    preconditioner and log that direction explicitly.

    A callback receives ``callback(event, state_dict)`` after a complete attempt.
    The caller must verify simulation/input identity when resuming; optional
    ``objective_id`` also checks a caller-supplied objective fingerprint here.
    """

    _TERMINAL = {"converged_gradient", "stalled_step", "stalled_invalid",
                 "stalled_descent", "evaluation_budget_exhausted", "invalid_initial"}
    _STATUSES = _TERMINAL | {"uninitialized", "ready", "running", "budget_exhausted"}

    def __init__(self, space: PhysicalParameterSpace,
                 objective: Callable[[dict[str, float]], ObjectiveValue],
                 options: AdamOptions | None = None, callback=None, objective_id: str | None = None):
        self.space = space
        self.objective = objective
        self.options = options or AdamOptions()
        self.callback = callback
        if objective_id is not None and not isinstance(objective_id, str):
            raise ValueError("objective_id must be a string or None")
        self.objective_id = objective_id
        self.u = space.coordinates()
        self.m = np.zeros_like(self.u)
        self.v = np.zeros_like(self.u)
        self.learning_rate = self.options.learning_rate
        self.iterations = 0
        self.accepted_updates = 0
        self.evaluations = 0
        self.history: list[dict] = []
        self.status = "uninitialized"
        self.current: ObjectiveValue | None = None
        self.best_u: np.ndarray | None = None
        self.best_value: float | None = None

    def _checked_objective(self, result, coordinates):
        if not isinstance(result, ObjectiveValue):
            raise TypeError("Objective must return ObjectiveValue(value, gradient, diagnostics)")
        value = float(result.value)
        if not math.isfinite(value):
            raise InvalidStateError("Objective value is not finite")
        if not isinstance(result.gradient, Mapping):
            raise TypeError("Objective gradient must map physical parameter names to derivatives")
        gradient = {name: float(derivative) for name, derivative in result.gradient.items()}
        if not all(math.isfinite(derivative) for derivative in gradient.values()):
            raise InvalidStateError("Objective physical gradient is not finite")
        unknown = set(gradient) - set(PARAMETER_NAMES)
        missing = set(self.space.fit) - set(gradient)
        if unknown or missing:
            raise ValueError(f"Invalid gradient keys: missing={sorted(missing)}, unknown={sorted(unknown)}")
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                transformed = self.space.pullback(coordinates, gradient)
        except ValueError as error:
            raise InvalidStateError(f"Invalid objective coordinate gradient: {error}") from error
        if np.max(np.abs(transformed)) > math.sqrt(np.finfo(np.float64).max):
            raise InvalidStateError("Coordinate gradient is too large for Adam squared moments")
        if not isinstance(result.diagnostics, Mapping):
            raise TypeError("Objective diagnostics must be a mapping")
        return ObjectiveValue(value, gradient, _plain(result.diagnostics))

    def _evaluate(self, coordinates, stage, backtrack=None):
        if self.options.max_evaluations is not None and self.evaluations >= self.options.max_evaluations:
            raise _EvaluationBudgetExhausted
        parameters = self.space.physical(coordinates)
        self.evaluations += 1
        event = {"type": "evaluation", "evaluation": self.evaluations,
                 "iteration": self.iterations, "stage": stage, "backtrack": backtrack,
                 "coordinates": coordinates.tolist(), "parameters": parameters}
        started = time.monotonic()
        try:
            result = self._checked_objective(self.objective(dict(parameters)), coordinates)
        except Exception as error:
            event.update(valid=False, elapsed_s=time.monotonic() - started,
                         error_type=type(error).__name__, error=str(error))
            self.history.append(event)
            if isinstance(error, (InvalidStateError, FloatingPointError, OverflowError)):
                return None, str(error)
            raise
        event.update(valid=True, elapsed_s=time.monotonic() - started,
                     value=result.value, physical_gradient=dict(result.gradient),
                     coordinate_gradient=self.space.pullback(coordinates, result.gradient).tolist(),
                     diagnostics=result.diagnostics)
        self.history.append(event)
        return result, None

    def _notify(self, event):
        event = _plain(event)
        self.history.append(event)
        if self.callback is not None:
            self.callback(deepcopy(event), self.state_dict())
        return event

    def initialize(self):
        if self.current is not None:
            return deepcopy(self.current)
        result, error = self._evaluate(self.u, "initial")
        if result is None:
            self.status = "invalid_initial"
            self._notify({"type": "initialization", "status": self.status, "error": error})
            raise InvalidStateError(f"Initial objective is invalid: {error}")
        self.current = result
        self.best_u = self.u.copy()
        self.best_value = result.value
        self.status = "ready"
        self._notify({"type": "initialization", "status": self.status, "value": result.value})
        return deepcopy(result)

    def _projected_gradient_norm(self, gradient):
        with np.errstate(over="ignore", invalid="ignore"):
            trial = self.u - gradient
        # Projection of an overflowing descent coordinate is still a bound.
        trial = np.maximum(np.minimum(trial, self.space.upper), self.space.lower)
        return float(np.linalg.norm(self.u - trial, ord=np.inf))

    def _proposal_learning_rate(self, retained_rate):
        if self.options.learning_rate_policy == "recover-v1":
            return min(self.options.learning_rate, retained_rate * self.options.learning_rate_growth)
        return retained_rate

    def step(self):
        self.initialize()
        if self.status in self._TERMINAL:
            return {"type": "status", "status": self.status, "iteration": self.iterations}
        gradient = self.space.pullback(self.u, self.current.gradient)
        projected_norm = self._projected_gradient_norm(gradient)
        if projected_norm <= self.options.gradient_tolerance:
            self.status = "converged_gradient"
            return self._notify({"type": "status", "status": self.status, "iteration": self.iterations,
                                 "projected_gradient_norm": projected_norm})
        if self.options.max_evaluations is not None and self.evaluations >= self.options.max_evaluations:
            self.status = "evaluation_budget_exhausted"
            return self._notify({"type": "status", "status": self.status, "iteration": self.iterations})
        self.iterations += 1
        self.status = "running"
        old_u = self.u.copy()
        old_value = self.current.value
        b1, b2 = self.options.beta1, self.options.beta2
        with np.errstate(over="ignore", invalid="ignore"):
            next_m = b1 * self.m + (1.0 - b1) * gradient
            next_v = b2 * self.v + (1.0 - b2) * gradient * gradient
        if not (np.isfinite(next_m).all() and np.isfinite(next_v).all()):
            raise InvalidStateError("Adam moments overflowed; rescale the loss or parameter coordinates")
        next_update = self.accepted_updates + 1
        m_hat = next_m / (1.0 - b1 ** next_update)
        v_hat = next_v / (1.0 - b2 ** next_update)
        denominator = np.sqrt(v_hat) + self.options.epsilon
        direction = -m_hat / denominator
        direction_name = "adam"
        proposal_rate = self._proposal_learning_rate(self.learning_rate)
        test_delta = self.space.project(self.u + proposal_rate * direction) - self.u
        if float(np.dot(gradient, test_delta)) >= 0:
            direction = -gradient / denominator
            direction_name = "scaled_gradient"
        rate = proposal_rate
        accepted = False
        valid_candidates = 0
        attempts = []
        stop_reason = None
        for backtrack in range(self.options.max_backtracks + 1):
            if rate < self.options.min_learning_rate:
                stop_reason = "minimum_learning_rate"
                break
            candidate_u = self.space.project(old_u + rate * direction)
            delta = candidate_u - old_u
            step_norm = float(np.linalg.norm(delta, ord=np.inf))
            if step_norm <= self.options.step_tolerance:
                stop_reason = "step_too_small"
                break
            slope = float(np.dot(gradient, delta))
            if not math.isfinite(slope) or slope >= 0:
                raise InvalidStateError("The projected optimizer direction is not a finite descent direction")
            try:
                candidate, error = self._evaluate(candidate_u, "proposal", backtrack)
            except _EvaluationBudgetExhausted:
                stop_reason = "evaluation_budget"
                break
            attempt = {"backtrack": backtrack, "learning_rate": rate, "step_norm": step_norm,
                       "evaluation": self.evaluations, "valid": candidate is not None}
            if candidate is None:
                attempt.update(accepted=False, reason="invalid_objective", error=error)
            else:
                valid_candidates += 1
                limit = old_value + self.options.armijo * slope + self.options.loss_tolerance
                if candidate.value <= limit:
                    self.u = candidate_u
                    self.current = candidate
                    self.m, self.v = next_m, next_v
                    self.accepted_updates = next_update
                    self.learning_rate = rate
                    if candidate.value < self.best_value:
                        self.best_u = candidate_u.copy()
                        self.best_value = candidate.value
                    accepted = True
                    attempt.update(accepted=True, value=candidate.value)
                else:
                    attempt.update(accepted=False, reason="insufficient_decrease",
                                   value=candidate.value, acceptance_limit=limit)
            attempts.append(attempt)
            if accepted:
                break
            rate *= self.options.backtrack_factor
        if accepted:
            self.status = "ready"
        else:
            self.learning_rate = max(min(rate, self.learning_rate), self.options.min_learning_rate)
            if stop_reason == "evaluation_budget":
                self.status = "evaluation_budget_exhausted"
            elif stop_reason in {"step_too_small", "minimum_learning_rate"}:
                self.status = "stalled_step"
            elif valid_candidates:
                self.status = "stalled_descent"
            else:
                self.status = "stalled_invalid"
        event = {"type": "step", "iteration": self.iterations, "accepted": accepted,
                 "status": self.status, "direction": direction_name,
                 "value_before": old_value, "value_after": self.current.value,
                 "coordinates": self.u.tolist(), "parameters": self.space.physical(self.u),
                 "coordinate_gradient_before": gradient.tolist(),
                 "gradient_norm": math.hypot(*gradient),
                 "projected_gradient_norm": projected_norm,
                 "step_norm": float(np.linalg.norm(self.u - old_u, ord=np.inf)),
                 "learning_rate": self.learning_rate,
                 "learning_rate_policy": self.options.learning_rate_policy,
                 "proposal_learning_rate": proposal_rate,
                 "accepted_learning_rate": rate if accepted else None,
                 "stop_reason": stop_reason,
                 "accepted_updates": self.accepted_updates, "evaluations": self.evaluations,
                 "attempts": attempts}
        return self._notify(event)

    def run(self, iterations):
        if type(iterations) is not int or iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        self.initialize()
        if self.status == "budget_exhausted":
            self.status = "ready"
        for _ in range(iterations):
            if self.status in self._TERMINAL:
                break
            self.step()
        if self.status not in self._TERMINAL:
            self.status = "budget_exhausted"
            self._notify({"type": "status", "status": self.status, "iteration": self.iterations})
        return OptimizationResult(
            self.space.physical(self.best_u), float(self.best_value), self.space.physical(self.u),
            self.current.value, self.status, self.iterations, self.accepted_updates,
            self.evaluations, deepcopy(self.history),
        )

    @staticmethod
    def _objective_dict(value):
        if value is None:
            return None
        return {"value": value.value, "gradient": dict(value.gradient), "diagnostics": deepcopy(value.diagnostics)}

    def state_dict(self):
        return {
            "schema_version": 1, "space": self.space.settings(), "options": asdict(self.options),
            "objective_id": self.objective_id, "status": self.status,
            "iterations": self.iterations, "accepted_updates": self.accepted_updates,
            "evaluations": self.evaluations, "learning_rate": self.learning_rate,
            "coordinates": self.u.tolist(), "physical_parameters": self.space.physical(self.u),
            "first_moment": self.m.tolist(), "second_moment": self.v.tolist(),
            "current": self._objective_dict(self.current),
            "best": None if self.best_u is None else {
                "coordinates": self.best_u.tolist(), "parameters": self.space.physical(self.best_u),
                "value": self.best_value,
            },
            "history": deepcopy(self.history),
        }

    def load_state_dict(self, state):
        """Validate a complete saved state before replacing any live optimizer data."""
        if not isinstance(state, Mapping) or state.get("schema_version") != 1:
            raise ValueError("Unsupported optimizer state schema")
        if state.get("space") != self.space.settings() or state.get("options") != asdict(self.options):
            raise ValueError("Saved parameter definitions or optimizer options do not match")
        if state.get("objective_id") != self.objective_id:
            raise ValueError("Saved objective fingerprint does not match")
        status = state.get("status")
        if status not in self._STATUSES or status == "running":
            raise ValueError("Unknown saved optimizer status or incomplete optimizer attempt")
        counts = [state.get(name) for name in ("iterations", "accepted_updates", "evaluations")]
        if not all(type(value) is int and value >= 0 for value in counts):
            raise ValueError("Optimizer counters must be nonnegative integers")
        iterations, updates, evaluations = counts
        if updates > iterations:
            raise ValueError("Accepted-update count exceeds optimizer attempts")
        if self.options.max_evaluations is not None and evaluations > self.options.max_evaluations:
            raise ValueError("Saved evaluation count exceeds the configured budget")
        rate = float(state["learning_rate"])
        if not math.isfinite(rate) or not self.options.min_learning_rate <= rate <= self.options.learning_rate:
            raise ValueError("Saved learning rate is outside the configured range")
        u = self.space._vector(state["coordinates"]).copy()
        if state.get("physical_parameters") != self.space.physical(u):
            raise ValueError("Saved physical values do not correspond exactly to saved coordinates")
        m = self.space._vector(state["first_moment"]).copy()
        v = self.space._vector(state["second_moment"]).copy()
        if np.any(v < 0) or (updates == 0 and (np.any(m != 0) or np.any(v != 0))):
            raise ValueError("Invalid saved Adam moments")
        current_data, best_data = state.get("current"), state.get("best")
        if (current_data is None) != (best_data is None):
            raise ValueError("Saved current and best evaluations must both exist or both be absent")
        current = None
        best_u = None
        best_value = None
        if current_data is not None:
            current = self._checked_objective(ObjectiveValue(**current_data), u)
            best_u = self.space._vector(best_data["coordinates"]).copy()
            if best_data.get("parameters") != self.space.physical(best_u):
                raise ValueError("Saved best physical values do not match their coordinates")
            best_value = float(best_data["value"])
            if not math.isfinite(best_value) or best_value > current.value:
                raise ValueError("Invalid saved best objective")
            if evaluations < 1 or status in {"uninitialized", "invalid_initial"}:
                raise ValueError("Saved initialized objective has inconsistent status/counters")
        elif iterations != 0 or updates != 0 or status not in {"uninitialized", "invalid_initial"}:
            raise ValueError("Saved uninitialized optimizer has inconsistent status/counters")
        history = deepcopy(state.get("history"))
        if not isinstance(history, list) or not all(isinstance(event, dict) for event in history):
            raise ValueError("Saved history must be a list of event records")
        evaluation_events = [event for event in history if event.get("type") == "evaluation"]
        if [event.get("evaluation") for event in evaluation_events] != list(range(1, evaluations + 1)):
            raise ValueError("Saved evaluation history is incomplete or out of order")
        step_events = [event for event in history if event.get("type") == "step"]
        if [event.get("iteration") for event in step_events] != list(range(1, iterations + 1)):
            raise ValueError("Saved optimizer-attempt history is incomplete or out of order")
        if sum(event.get("accepted") is True for event in step_events) != updates:
            raise ValueError("Saved accepted-update history does not match counters")
        expected_m = np.zeros_like(u)
        expected_v = np.zeros_like(u)
        committed = [event for event in evaluation_events if event.get("stage") == "initial" and event.get("valid") is True]
        if current is not None and len(committed) != 1:
            raise ValueError("Saved optimizer must have exactly one valid initial evaluation")
        retained_rate = self.options.learning_rate
        for event in step_events:
            if event.get("learning_rate_policy") != self.options.learning_rate_policy:
                raise ValueError("Saved optimizer step has a different learning-rate policy")
            if event.get("proposal_learning_rate") != self._proposal_learning_rate(retained_rate):
                raise ValueError("Saved proposal rate does not match the configured recovery policy")
            retained_rate = float(event["learning_rate"])
            if not math.isfinite(retained_rate) or not self.options.min_learning_rate <= retained_rate <= self.options.learning_rate:
                raise ValueError("Saved retained learning rate is outside the configured range")
            accepted_attempts = [attempt for attempt in event.get("attempts", []) if attempt.get("accepted") is True]
            if len(accepted_attempts) != int(event["accepted"] is True):
                raise ValueError("Saved optimizer step and proposal acceptance disagree")
            if not accepted_attempts and event.get("accepted_learning_rate") is not None:
                raise ValueError("A rejected optimizer attempt cannot have an accepted rate")
            if accepted_attempts:
                accepted_rate = accepted_attempts[0].get("learning_rate")
                if event.get("accepted_learning_rate") != accepted_rate or retained_rate != accepted_rate:
                    raise ValueError("Saved accepted and retained learning rates disagree")
                index = accepted_attempts[0].get("evaluation")
                if type(index) is not int or not 1 <= index <= evaluations:
                    raise ValueError("Saved accepted proposal has no corresponding evaluation")
                candidate = evaluation_events[index - 1]
                if candidate.get("valid") is not True or candidate.get("iteration") != event["iteration"]:
                    raise ValueError("Saved accepted proposal does not name a valid evaluation from its attempt")
                committed.append(candidate)
                g = self.space._vector(event["coordinate_gradient_before"])
                with np.errstate(over="ignore", invalid="ignore"):
                    expected_m = self.options.beta1 * expected_m + (1 - self.options.beta1) * g
                    expected_v = self.options.beta2 * expected_v + (1 - self.options.beta2) * g * g
        if rate != retained_rate:
            raise ValueError("Saved learning rate does not match completed optimizer history")
        if not np.array_equal(m, expected_m) or not np.array_equal(v, expected_v):
            raise ValueError("Saved Adam moments do not match accepted-gradient history")
        if current is not None:
            last = committed[-1]
            if (last.get("coordinates") != u.tolist() or last.get("parameters") != self.space.physical(u)
                    or last.get("value") != current.value or last.get("physical_gradient") != current.gradient
                    or last.get("diagnostics") != current.diagnostics):
                raise ValueError("Saved current objective does not match the last committed evaluation")
            best_event = min(committed, key=lambda event: event["value"])
            if best_event.get("coordinates") != best_u.tolist() or best_event["value"] != best_value:
                raise ValueError("Saved best objective does not match accepted evaluation history")
        _plain(history)
        self.u, self.m, self.v = u, m, v
        self.current, self.best_u, self.best_value = current, best_u, best_value
        self.iterations, self.accepted_updates, self.evaluations = iterations, updates, evaluations
        self.learning_rate, self.status, self.history = rate, status, history
