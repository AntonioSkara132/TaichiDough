"""Finite-difference checks of loss derivatives with respect to tool poses and velocities."""
import unittest

import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation

from experiments.differentiable_mpm.checkpoint import CheckpointedRollout
from experiments.differentiable_mpm.solver import Stepper
from experiments.differentiable_mpm.state import ToolControl
from experiments.differentiable_mpm.tests.test_solver import dot_state, fixture, terminal_seed
from experiments.differentiable_mpm.tests.test_trajectory import ParticleTargetLoss, Target


def shifted(control, step):
    """Per-step controls that move the tools along their velocities, so steps differ."""
    result = ToolControl(control.poses.copy(), control.velocities.copy(), control.time)
    result.poses[:, :3] += 2e-4 * step * control.velocities[:, :3]
    return result


class ControlGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, debug=True, offline_cache=False)

    def make(self, steps, **options):
        config, state, params, control, sdf = fixture(sdf=True, **options)
        solver = Stepper(config, params, capacity=steps + 1, sdf=sdf, control_gradients=True)
        return solver, state, params, [shifted(control, t) for t in range(steps)]

    def value(self, solver, state, params, controls, seed):
        solver.set_parameters(params)
        solver.load_state(0, state)
        for t, control in enumerate(controls):
            solver.advance(t, control)
        return dot_state(solver.state(len(controls)), seed)

    def gradients(self, solver, state, params, controls, seed):
        self.value(solver, state, params, controls, seed)
        solver.clear_state_gradients()
        solver.clear_parameter_gradients()
        solver.load_adjoint(len(controls), seed)
        return [solver.reverse_step(t, controls[t]) for t in reversed(range(len(controls)))][::-1]

    def central(self, evaluate):
        return [(evaluate(h) - evaluate(-h)) / (2 * h) for h in (1e-5, 3e-6, 1e-6)]

    def assert_derivative(self, actual, candidates, context, relative=5e-4, absolute=1e-7):
        self.assertTrue(np.isfinite(actual), context)
        errors = [abs(actual - fd) for fd in candidates]
        scale = max(abs(actual), max(abs(fd) for fd in candidates), 1e-5)
        self.assertLessEqual(min(errors), absolute + relative * scale,
                             f"{context}: AD={actual:.12g}, FD={candidates}, errors={errors}")

    def check(self, steps, **options):
        solver, state, params, controls = self.make(steps, **options)
        seed = terminal_seed(state)
        gradients = self.gradients(solver, state, params, controls, seed)
        counts = solver.diagnostics()
        self.assertGreater(counts["particle_tool0"] + counts["particle_tool1"], 0)
        rng = np.random.default_rng(91)
        for t in range(steps):
            g = gradients[t]
            self.assertTrue(all(np.isfinite(a).all() for a in (g.poses, g.velocities, g.rotation)))
            position = rng.normal(size=(2, 3)) * 0.01
            velocity = rng.normal(size=(2, 6)) * np.array([0.02] * 3 + [0.3] * 3)
            axis = rng.normal(size=(2, 3))
            tangent = rng.normal(size=(2, 4))
            tangent -= np.sum(tangent * controls[t].poses[:, 3:], axis=1, keepdims=True) * controls[t].poses[:, 3:]

            def perturbed(h, kind):
                changed = list(controls)
                c = ToolControl(controls[t].poses.copy(), controls[t].velocities.copy(), controls[t].time)
                if kind == "position":
                    c.poses[:, :3] += h * position
                elif kind == "velocity":
                    c.velocities += h * velocity
                elif kind == "rotation":
                    for k in range(2):
                        rotated = Rotation.from_rotvec(h * axis[k]) * Rotation.from_quat(c.poses[k, 3:])
                        c.poses[k, 3:] = rotated.as_quat()
                else:
                    q = c.poses[:, 3:] + h * tangent
                    c.poses[:, 3:] = q / np.linalg.norm(q, axis=1, keepdims=True)
                changed[t] = c
                return self.value(solver, state, params, changed, seed)

            predicted = {"position": np.sum(g.poses[:, :3] * position), "velocity": np.sum(g.velocities * velocity),
                         "rotation": np.sum(g.rotation * axis), "quaternion": np.sum(g.poses[:, 3:] * tangent)}
            for kind, value in predicted.items():
                self.assert_derivative(value, self.central(lambda h: perturbed(h, kind)), f"{kind}, step {t}/{steps}")
        return gradients

    def test_retention_contact_one_step(self):
        gradients = self.check(1)
        self.assertGreater(np.abs(gradients[0].poses[:, :3]).max(), 0)
        self.assertGreater(np.abs(gradients[0].velocities).max(), 0)

    def test_bounded_adhesive_contact_three_steps(self):
        self.check(3, tool_contact_model="coulomb-bounded-adhesive-v2")

    def test_disabled_by_default(self):
        config, state, params, control, sdf = fixture(sdf=True)
        solver = Stepper(config, params, capacity=2, sdf=sdf)
        solver.load_state(0, state)
        solver.advance(0, control)
        solver.load_adjoint(1, terminal_seed(state))
        self.assertIsNone(solver.reverse_step(0, control))


class RolloutControlGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)

    def test_segmented_rollout_matches_finite_differences(self):
        config, state, params, control, sdf = fixture(sdf=True, tool_contact_model="coulomb-bounded-adhesive-v2")
        controls = [shifted(control, t) for t in range(6)]
        stepper = Stepper(config, params, capacity=7, sdf=sdf, control_gradients=True)
        targets = [Target(i, step, state.x + 0.002 * (i + 1)) for i, step in enumerate((2, 4, 6))]

        def rollout(length, changed=controls):
            return CheckpointedRollout(stepper, state, changed, targets, ParticleTargetLoss(), length,
                                       replay_rtol=0, replay_atol={n: 0 for n in state.arrays()})
        full = rollout(6).value_and_gradient(params, compute_control_grad=True)
        split = rollout(2).value_and_gradient(params, compute_control_grad=True)
        for name in ("poses", "velocities", "rotation"):
            self.assertEqual(full.control_gradient[name].shape[0], 6)
            np.testing.assert_allclose(full.control_gradient[name], split.control_gradient[name], rtol=1e-10, atol=1e-12)
        self.assertGreater(np.abs(full.control_gradient["poses"][:, :, :3]).max(), 0)
        direction = np.random.default_rng(5).normal(size=(6, 2, 3)) * 0.01

        def value(h):
            changed = [ToolControl(c.poses.copy(), c.velocities.copy(), c.time) for c in controls]
            for t, c in enumerate(changed):
                c.poses[:, :3] += h * direction[t]
            return rollout(2, changed).value_and_gradient(params, compute_grad=False).value
        predicted = np.sum(full.control_gradient["poses"][:, :, :3] * direction)
        fds = [(value(h) - value(-h)) / (2 * h) for h in (1e-5, 3e-6, 1e-6)]
        self.assertLessEqual(min(abs(predicted - fd) for fd in fds), 1e-7 + 5e-4 * max(abs(predicted), 1e-5),
                             f"AD={predicted}, FD={fds}")

    def test_requires_enabled_stepper(self):
        config, state, params, control, sdf = fixture(sdf=True)
        stepper = Stepper(config, params, capacity=2, sdf=sdf)
        rollout = CheckpointedRollout(stepper, state, [control], [Target(0, 1, state.x)], ParticleTargetLoss(), 1)
        with self.assertRaises(ValueError):
            rollout.value_and_gradient(params, compute_control_grad=True)


if __name__ == "__main__":
    unittest.main()
