"""Segmented reverse replay retaining derivatives through every particle-state boundary."""
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from .state import InvalidStateError, ParticleState, STATE_NAMES


@dataclass
class RolloutEvaluation:
    value: float
    gradient: dict[str, float] | None
    initial_gradient: ParticleState | None
    frames: list[dict]
    diagnostics: dict


def estimate_memory(n_particles, total_steps, segment_length=64, precision="f32", grid=48, sdf_resolution=64):
    """Allocation estimates, not a measurement of the Taichi process or device allocator."""
    if min(n_particles, segment_length, grid) < 1 or total_steps < 0:
        raise ValueError("Invalid allocation dimensions")
    itemsize = 4 if precision == "f32" else 8
    segment_length = min(segment_length, max(1, total_steps))
    state_bytes = n_particles * 25 * itemsize
    checkpoint_count = (total_steps + segment_length - 1) // segment_length + 1
    return {
        "particle_state_bytes": state_bytes,
        "local_particle_history_and_adjoint_bytes": 2 * (segment_length + 1) * state_bytes,
        "host_checkpoint_bytes": checkpoint_count * state_bytes,
        "host_boundary_adjoint_bytes": state_bytes,
        "grid_primal_and_adjoint_estimate_bytes": grid ** 3 * 20 * itemsize,
        "sdf_estimate_bytes": 2 * sdf_resolution ** 3 * 4 * itemsize,
        "full_particle_history_and_adjoint_bytes": 2 * (total_steps + 1) * state_bytes,
        "note": "Material/contact scratch, loss fields, compiler and allocator overhead are additional.",
    }


class CheckpointedRollout:
    """A fixed-control replay with a mean of losses at the supplied observation states.

    The loss object returns a record with value, gradient, components and diagnostics
    from value_and_grad_positions(x, observation, compute_grad=...). Observation
    records expose integer step and frame_index. An observation at a segment start
    is differentiated by the preceding segment, except at global state zero.
    """

    def __init__(self, stepper, initial_state: ParticleState, controls: Sequence,
                 observations: Sequence, loss, segment_length=64, *, replay_rtol=1e-5,
                 replay_atol=None, loss_replay_rtol=2e-5, progress: Callable | None = None):
        initial_state.validate()
        self.stepper = stepper
        self.initial_state = initial_state.copy()
        self.controls = controls
        self.observations = list(observations)
        self.loss = loss
        self.total_steps = len(controls)
        self.segment_length = min(int(segment_length), max(1, self.total_steps))
        if self.segment_length < 1:
            raise ValueError("segment_length must be positive")
        if stepper.capacity < self.segment_length + 1:
            raise ValueError("Stepper capacity must be at least segment_length + 1")
        if not self.observations:
            raise ValueError("A calibration rollout requires at least one observation")
        self.by_step = {}
        for index, observation in enumerate(self.observations):
            if not isinstance(observation.step, (int, np.integer)) or not 0 <= observation.step <= self.total_steps:
                raise ValueError("Observation step is outside the replay horizon")
            self.by_step.setdefault(int(observation.step), []).append(index)
        frame_ids = [o.frame_index for o in self.observations]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("Duplicate observation frame indices")
        self.weight = 1.0 / len(self.observations)
        self.replay_rtol = replay_rtol
        self.replay_atol = replay_atol or {"x": 1e-6, "v": 1e-5, "C": 1e-4, "F": 1e-5, "Jp": 1e-5}
        if set(self.replay_atol) != set(STATE_NAMES):
            raise ValueError("Recomputation tolerances must specify every state array")
        if replay_rtol < 0 or any(v < 0 for v in self.replay_atol.values()):
            raise ValueError("Recomputation tolerances must be nonnegative")
        self.loss_replay_rtol = loss_replay_rtol
        self.progress = progress

    def _notify(self, phase, step, **extra):
        if self.progress is not None:
            self.progress({"phase": phase, "step": step, "total_steps": self.total_steps, **extra})

    def _signature(self):
        diagnostics = self.stepper.diagnostics()
        counts = diagnostics.get("branch_counts", diagnostics)
        if not isinstance(counts, dict) or not all(isinstance(v, (int, np.integer)) for v in counts.values()):
            raise InvalidStateError("Stepper branch summaries must be integer counts")
        return {k: int(v) for k, v in counts.items()}

    def _check_endpoint(self, expected, actual, global_step, maxima):
        actual.validate()
        for name in STATE_NAMES:
            a, b = getattr(expected, name), getattr(actual, name)
            delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
            maxima[name] = max(maxima[name], float(delta.max()))
            if not np.allclose(a, b, rtol=self.replay_rtol, atol=self.replay_atol[name]):
                raise InvalidStateError(
                    f"Checkpoint recomputation differs at step {global_step}, {name}: "
                    f"max absolute difference {delta.max():.8g}; backward evaluation rejected")

    def _check_signature(self, expected, actual, global_step, first, last):
        if actual == expected:
            return
        differences = {
            name: {"forward": expected.get(name), "recomputed": actual.get(name)}
            for name in sorted(set(expected) | set(actual))
            if expected.get(name) != actual.get(name)
        }
        self._notify("recompute_mismatch", global_step,
                     stage="segment_forward_recompute", segment_start_step=first,
                     segment_end_step=last, expected_counts=expected,
                     recomputed_counts=actual, differing_counts=differences)
        raise InvalidStateError(
            f"Contact/yield summaries differ on recomputation at step {global_step} "
            f"(segment {first}–{last}): {differences}; backward evaluation rejected")

    def _validate_adjoint(self, adjoint, global_step):
        try:
            adjoint.validate()
        except InvalidStateError as error:
            summaries = {}
            for name, values in adjoint.arrays().items():
                finite = np.isfinite(values)
                summaries[name] = {
                    "nonfinite_count": int(np.count_nonzero(~finite)),
                    "max_finite_abs": float(np.abs(values[finite]).max()) if finite.any() else None,
                    "first_nonfinite_indices": np.argwhere(~finite)[:5].tolist(),
                }
            self._notify("invalid_adjoint", global_step, arrays=summaries)
            raise InvalidStateError(
                f"Invalid trajectory adjoint at global step {global_step}: {summaries}") from error

    @staticmethod
    def _validate_record(record, n_particles, with_gradient):
        if not np.isfinite(record.value):
            raise InvalidStateError("Observation loss is nonfinite")
        if with_gradient:
            gradient = np.asarray(record.gradient)
            if gradient.shape != (n_particles, 3) or not np.isfinite(gradient).all():
                raise InvalidStateError("Observation position derivative is invalid")

    def value_and_gradient(self, parameters, *, compute_grad=True, frame_callback=None, frame_steps=None):
        start_time = perf_counter()
        export_steps = set(self.by_step) | {0} if frame_steps is None else set(frame_steps)
        if any(not isinstance(s, (int, np.integer)) or not 0 <= s <= self.total_steps for s in export_steps):
            raise ValueError("Requested export state is outside the replay")
        stepper = self.stepper
        stepper.set_parameters(parameters)
        stepper.load_state(0, self.initial_state)
        checkpoints = {0: self.initial_state.copy()}
        signatures = []
        records: list[dict[str, Any] | None] = [None] * len(self.observations)
        loss_total = 0.0
        if frame_callback is not None and 0 in export_steps:
            frame_callback(0, self.initial_state.copy())

        def observe(step, slot):
            nonlocal loss_total
            if step not in self.by_step:
                return
            positions = stepper.state(slot).x
            for index in self.by_step[step]:
                observation = self.observations[index]
                record = self.loss.value_and_grad_positions(positions, observation, compute_grad=False)
                self._validate_record(record, len(positions), False)
                records[index] = {
                    "frame_index": int(observation.frame_index), "step": step,
                    "value": float(record.value), "components": record.components,
                    "diagnostics": record.diagnostics,
                }
                loss_total += self.weight * float(record.value)
                self._notify("observation", step, frame_index=int(observation.frame_index), value=float(record.value))

        observe(0, 0)
        for global_input in range(self.total_steps):
            slot = global_input % self.segment_length
            stepper.advance(slot, self.controls[global_input])
            signatures.append(self._signature())
            global_output = global_input + 1
            observe(global_output, slot + 1)
            if frame_callback is not None and global_output in export_steps:
                frame_callback(global_output, stepper.state(slot + 1))
            if global_output % self.segment_length == 0 or global_output == self.total_steps:
                boundary = stepper.state(slot + 1)
                boundary.validate()
                if compute_grad:
                    checkpoints[global_output] = boundary
                self._notify("forward_segment", global_output)
                if global_output < self.total_steps:
                    stepper.load_state(0, boundary)
        if any(r is None for r in records):
            raise InvalidStateError("Missing observation evaluation")
        frame_records = [record for record in records if record is not None]
        forward_seconds = perf_counter() - start_time
        diagnostics: dict[str, Any] = {
            "total_steps": self.total_steps, "observation_count": len(records),
            "segment_length": self.segment_length, "forward_seconds": forward_seconds,
            "backward_seconds": 0.0, "gradient_method": "reverse-mode segmented full trajectory" if compute_grad else None,
            "branch_counts_total": {name: sum(signature.get(name, 0) for signature in signatures)
                                    for name in sorted({key for signature in signatures for key in signature})},
        }
        if not compute_grad:
            return RolloutEvaluation(loss_total, None, None, frame_records, diagnostics)

        reverse_start = perf_counter()
        self._notify("backward_start", self.total_steps)
        stepper.clear_parameter_gradients()
        carry = self.initial_state.zeros_like()
        maxima = {name: 0.0 for name in STATE_NAMES}
        observation_gradient_count = np.zeros(len(records), dtype=int)
        endpoints = sorted(checkpoints)
        loss_drift_max = 0.0

        def seed_observation(global_step, slot):
            nonlocal loss_drift_max
            if global_step not in self.by_step:
                return
            positions = stepper.state(slot).x
            for index in self.by_step[global_step]:
                record = self.loss.value_and_grad_positions(positions, self.observations[index], compute_grad=True)
                self._validate_record(record, len(positions), True)
                old_value = frame_records[index]["value"]
                loss_drift_max = max(loss_drift_max, abs(float(record.value) - old_value))
                if not np.isclose(record.value, old_value, rtol=self.loss_replay_rtol, atol=1e-8):
                    raise InvalidStateError(f"Loss recomputation differs at frame {self.observations[index].frame_index}")
                stepper.seed_positions(slot, record.gradient, weight=self.weight)
                observation_gradient_count[index] += 1

        for boundary_index in range(len(endpoints) - 1, 0, -1):
            first, last = endpoints[boundary_index - 1], endpoints[boundary_index]
            stepper.load_state(0, checkpoints[first])
            stepper.clear_state_gradients()
            for global_input in range(first, last):
                local = global_input - first
                stepper.advance(local, self.controls[global_input])
                self._check_signature(signatures[global_input], self._signature(),
                                      global_input, first, last)
            self._check_endpoint(checkpoints[last], stepper.state(last - first), last, maxima)
            stepper.load_adjoint(last - first, carry)
            for global_output in range(last, first, -1):
                local = global_output - first
                seed_observation(global_output, local)
                stepper.reverse_step(local - 1, self.controls[global_output - 1])
            carry = stepper.adjoint(0)
            self._validate_adjoint(carry, first)
            self._notify("backward_segment", first)

        if 0 in self.by_step:
            stepper.load_state(0, self.initial_state)
            stepper.clear_state_gradients()
            stepper.load_adjoint(0, carry)
            seed_observation(0, 0)
            carry = stepper.adjoint(0)
            self._validate_adjoint(carry, 0)
        if not np.all(observation_gradient_count == 1):
            raise InvalidStateError("An observation derivative was missing or injected more than once")
        gradient = stepper.parameter_gradients()
        if set(gradient) != set(parameters) or not all(np.isfinite(v) for v in gradient.values()):
            raise InvalidStateError("Parameter derivatives are missing or nonfinite")
        diagnostics.update({
            "backward_seconds": perf_counter() - reverse_start,
            "host_checkpoint_count": len(checkpoints),
            "recomputed_endpoint_max_abs": maxima,
            "recomputed_loss_max_abs": loss_drift_max,
            "observation_gradient_injections": observation_gradient_count.tolist(),
            "branch_summary_steps_checked": len(signatures),
        })
        return RolloutEvaluation(loss_total, gradient, carry, frame_records, diagnostics)
