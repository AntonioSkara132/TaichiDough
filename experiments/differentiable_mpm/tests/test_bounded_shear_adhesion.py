"""Cheap CPU contact probes for bounded normal and tangential adhesion."""
import unittest

import numpy as np
import taichi as ti

from experiments.differentiable_mpm.reference_adapter import reference_arguments
from experiments.differentiable_mpm.solver import Stepper
from experiments.differentiable_mpm.state import (
    DEFAULT_PARAMETERS, SLIP_SPEED_EPSILON, SimulationConfig, ToolControl,
    normalize_parameters, validate_tool_parameters,
)
from experiments.differentiable_mpm.tests.test_coulomb_contact import ContactProbe, plane_sdf
from experiments.differentiable_mpm.tests.test_tunable_contact import GradientProbe


MODEL = 'coulomb-bounded-shear-adhesive-v3'


class SharedAdhesionConfigTests(unittest.TestCase):
    def test_opt_in_and_existing_parameter(self):
        self.assertEqual(SimulationConfig(n_particles=1).tool_contact_model, 'retention-v1')
        config = SimulationConfig(n_particles=1, tool_collision='sdf', tool_contact_model=MODEL,
                                  tool_adhesion=150.)
        legacy = {key: value for key, value in DEFAULT_PARAMETERS.items() if key != 'tool_adhesion'}
        params = normalize_parameters(legacy, config)
        self.assertEqual(params['tool_adhesion'], 150.)
        validate_tool_parameters(params, config, ['tool_adhesion', 'tool_friction_coefficient'])
        for fitted in ('tool_stickiness', 'tool_retention'):
            with self.assertRaises(ValueError):
                validate_tool_parameters(params, config, [fitted])
        for options in ({'tool_stickiness': .1}, {'tool_contact_absorption': .1},
                        {'tool_adhesion': -1.}, {'tool_adhesion': float('nan')}):
            with self.assertRaises(ValueError):
                SimulationConfig(n_particles=1, tool_contact_model=MODEL, **options)
        with self.assertRaisesRegex(ValueError, 'preserved simulator'):
            reference_arguments(config, params)


class SharedAdhesionResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)

    def solver(self, model=MODEL, adhesion=150., mu=.3, dt=2e-4):
        config = SimulationConfig(n_particles=1, grid=12, precision='f64', dt=dt,
                                  tool_collision='sdf', tool_contact_model=model, tool_contact_padding=.004)
        params = dict(DEFAULT_PARAMETERS, tool_adhesion=adhesion, tool_friction_coefficient=mu)
        solver = Stepper(config, params, capacity=2, sdf=plane_sdf())
        control = ToolControl.stationary()
        control.poses[0, :3] = [.4, .4, .4]
        control.poses[1, :3] = [10, 10, 10]
        solver._set_control(control)
        return solver, params

    def response(self, solver, velocity, path='velocity', collider=(0, 0, 0), pos=(.401, .4, .4)):
        probe = ContactProbe()
        probe.velocity[None] = velocity
        probe.collider[None] = collider
        probe.normal[None] = [1, 0, 0]
        probe.pos[None] = pos
        getattr(probe, path + '_response')(solver)
        return np.asarray(probe.output_velocity[None])

    def test_zero_normal_load_has_bounded_shear_and_no_slip_reversal(self):
        solver, _ = self.solver(mu=0.)
        limit = 150. * solver.dt
        for tangential in (.005, .1, 2., -2.):
            actual = self.response(solver, (0, tangential, 0))
            expected = tangential * max(0., 1. - limit / np.sqrt(tangential ** 2 + SLIP_SPEED_EPSILON ** 2))
            np.testing.assert_allclose(actual, [0, expected, 0], atol=1e-14)
            self.assertGreaterEqual(actual[1] * tangential, 0.)
            self.assertLessEqual(abs(actual[1]), abs(tangential))
            self.assertLessEqual(abs(tangential - actual[1]), limit + 1e-14)
        old, _ = self.solver(model='coulomb-bounded-adhesive-v2', mu=0.)
        np.testing.assert_array_equal(self.response(old, (0, 2, 0)), [0, 2, 0])

    def test_normal_unchanged_and_shear_budgets_add(self):
        solver, _ = self.solver()
        old, _ = self.solver(model='coulomb-bounded-adhesive-v2')
        for vn in (-1., 0., .02, 1.):
            actual = self.response(solver, (vn, 2., .3))
            previous = self.response(old, (vn, 2., .3))
            self.assertEqual(actual[0], previous[0])
            normal_change = abs(vn - previous[0])
            budget = .3 * normal_change + 150. * solver.dt
            tangential = np.array([2., .3])
            expected = tangential * max(0., 1. - budget / np.sqrt(tangential @ tangential + SLIP_SPEED_EPSILON ** 2))
            np.testing.assert_allclose(actual[1:], expected, atol=1e-14)

    def test_zero_adhesion_matches_v2_on_all_contact_paths(self):
        solver, _ = self.solver(adhesion=0.)
        old, _ = self.solver(model='coulomb-bounded-adhesive-v2', adhesion=0.)
        for path in ('velocity', 'grid', 'particle'):
            for velocity in ((-1, 2, .3), (0, 2, .3), (1, 2, .3), (0, 0, 0)):
                np.testing.assert_array_equal(self.response(solver, velocity, path),
                                              self.response(old, velocity, path))

    def test_grid_and_particle_use_same_budget_and_leave_contact_region_free(self):
        solver, _ = self.solver()
        for velocity in ((-1., 2., .3), (0., 2., .3), (.02, .5, -.1)):
            expected = self.response(solver, velocity)
            for path in ('grid', 'particle'):
                np.testing.assert_allclose(self.response(solver, velocity, path), expected, atol=1e-13)
                np.testing.assert_array_equal(self.response(solver, velocity, path, pos=(.42, .4, .4)), velocity)
        collider = np.array([.3, -.2, .1])
        np.testing.assert_array_equal(self.response(solver, collider, collider=collider), collider)

    def test_tangential_budget_scales_with_dt_without_changing_parameter(self):
        for dt in (1e-4, 2e-4, 4e-4):
            solver, _ = self.solver(mu=0., dt=dt)
            actual = self.response(solver, (0, 2, 0))
            self.assertAlmostEqual(2. - actual[1], 150. * dt, places=13)

    def test_parameter_and_velocity_gradients_on_both_contact_paths(self):
        solver, params = self.solver()
        probe = GradientProbe()
        probe.normal[None] = [1, 0, 0]
        probe.pos[None] = [.401, .4, .4]
        for path in ('velocity', 'grid', 'particle'):
            for velocity in ((-1., 2., .3), (.02, .5, -.1), (1., 2., .3)):
                solver.set_parameters(params)
                solver.clear_parameter_gradients()
                probe.velocity.grad[None] = [0, 0, 0]
                probe.pos.grad[None] = [0, 0, 0]
                probe.velocity[None] = velocity
                probe.evaluate(solver, path)
                probe.loss.grad[None] = 1
                probe.evaluate.grad(solver, path)
                gradients = solver.parameter_gradients()
                velocity_grad = probe.velocity.grad[None].to_numpy().copy()
                for name in ('tool_adhesion', 'tool_friction_coefficient'):
                    h = 1e-6 * max(1., params[name])
                    samples = []
                    for sign in (1, -1):
                        solver.set_parameters(dict(params, **{name: params[name] + sign * h}))
                        probe.evaluate(solver, path)
                        samples.append(probe.loss[None])
                    fd = (samples[0] - samples[1]) / (2 * h)
                    self.assertAlmostEqual(gradients[name], fd, delta=2e-8,
                                           msg=f'{path} {velocity} {name}')
                solver.set_parameters(params)
                for axis in range(3):
                    samples = []
                    for sign in (1, -1):
                        trial = np.array(velocity)
                        trial[axis] += sign * 1e-6
                        probe.velocity[None] = trial
                        probe.evaluate(solver, path)
                        samples.append(probe.loss[None])
                    fd = (samples[0] - samples[1]) / 2e-6
                    self.assertAlmostEqual(velocity_grad[axis], fd, delta=2e-8,
                                           msg=f'{path} {velocity} velocity[{axis}]')

    def test_zero_slip_gradients_are_finite(self):
        solver, params = self.solver()
        probe = GradientProbe()
        probe.normal[None] = [1, 0, 0]
        probe.pos[None] = [.401, .4, .4]
        for path in ('velocity', 'grid', 'particle'):
            for adhesion in (0., 150.):
                for velocity in ((0., 0., 0.), (-1., 0., 0.), (1., 0., 0.)):
                    solver.set_parameters(dict(params, tool_adhesion=adhesion))
                    solver.clear_parameter_gradients()
                    probe.velocity.grad[None] = [0, 0, 0]
                    probe.pos.grad[None] = [0, 0, 0]
                    probe.velocity[None] = velocity
                    probe.evaluate(solver, path)
                    probe.loss.grad[None] = 1
                    probe.evaluate.grad(solver, path)
                    self.assertTrue(np.isfinite(list(solver.parameter_gradients().values())).all())
                    self.assertTrue(np.isfinite(probe.velocity.grad[None].to_numpy()).all())
                    self.assertTrue(np.isfinite(probe.pos.grad[None].to_numpy()).all())


if __name__ == '__main__':
    unittest.main()
