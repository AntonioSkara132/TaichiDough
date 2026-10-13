"""CPU float64 derivatives and compatibility for tunable tool contact."""
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import taichi as ti

from experiments.differentiable_mpm.calibrate import read_parameters
from experiments.differentiable_mpm.config import canonical_hash
from experiments.differentiable_mpm.solver import Stepper
from experiments.differentiable_mpm.state import (
    DEFAULT_PARAMETERS, LEGACY_PARAMETER_NAMES, SimulationConfig, ToolControl,
    normalize_parameters, validate_tool_parameters,
)
from experiments.differentiable_mpm.tests.test_coulomb_contact import ContactProbe, plane_sdf
from experiments.differentiable_mpm.tests.test_solver import fixture, terminal_seed, dot_state


@ti.data_oriented
class GradientProbe(ContactProbe):
    def __init__(self):
        super().__init__()
        self.velocity = ti.Vector.field(3, ti.f64, shape=(), needs_grad=True)
        self.pos = ti.Vector.field(3, ti.f64, shape=(), needs_grad=True)
        self.loss = ti.field(ti.f64, shape=(), needs_grad=True)

    @ti.kernel
    def evaluate(self, solver: ti.template(), path: ti.template()):
        result = self.velocity[None]
        if ti.static(path == 'velocity'):
            result = solver._tool_velocity_response(result, self.collider[None], self.normal[None])
        elif ti.static(path == 'grid'):
            result = solver._tool_grid_response(self.pos[None], result)
        elif ti.static(path == 'floor'):
            result = solver._floor_coulomb_response(result)
        else:
            position, result = solver._tool_particle_response(self.pos[None], result)
            result += position
        self.loss[None] = result.dot(ti.Vector([0.7, -0.4, 0.2]))


class CompatibilityTests(unittest.TestCase):
    def test_old_values_and_explicit_precedence(self):
        old = {name: DEFAULT_PARAMETERS[name] for name in LEGACY_PARAMETER_NAMES}
        self.assertEqual(normalize_parameters(old, {'tool_friction_coefficient': .3})['tool_friction_coefficient'], .3)
        self.assertEqual(normalize_parameters(dict(old, tool_friction_coefficient=.7),
                                             {'tool_friction_coefficient': .3})['tool_friction_coefficient'], .7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selected_parameters.json'
            identity = {'prepared': {'simulation': {'tool_friction_coefficient': .3}}}
            digest = canonical_hash(identity)
            path.write_text(json.dumps({'best_parameters': old, 'identity_sha256': digest}))
            manifest = Path(directory) / 'run_manifest.json'
            manifest.write_text(json.dumps({'identity': identity, 'identity_sha256': digest}))
            values, source = read_parameters(path, {'tool_friction_coefficient': .9})
            self.assertEqual(values['tool_friction_coefficient'], .3)
            self.assertIn('contact_defaults_source', source)
            path.write_text(json.dumps({'best_parameters': old, 'identity_sha256': 'wrong'}))
            with self.assertRaises(ValueError):
                read_parameters(path)

    def test_examples_and_physical_precedence(self):
        from experiments.differentiable_mpm.config import load_config
        from experiments.differentiable_mpm.reference_adapter import reference_arguments
        root = Path(__file__).resolve().parents[1] / 'configs'
        config = load_config(root / 'episode18_registered_friction_fit.json')
        self.assertEqual(config.fit_parameters, ['tool_friction_coefficient'])
        self.assertEqual(config.parameters['tool_stickiness'], 0)
        config.parameters['tool_friction_coefficient'] = .8
        config.validate()
        self.assertEqual(config.simulation['tool_friction_coefficient'], .8)
        joint = load_config(root / 'episode18_registered_adhesive_fit.json')
        self.assertEqual(joint.fit_parameters, ['tool_friction_coefficient', 'tool_stickiness'])
        with self.assertRaisesRegex(ValueError, 'preserved simulator'):
            reference_arguments(SimulationConfig(**joint.simulation), joint.parameters)

    def test_malformed_old_manifest_rejected(self):
        old = {name: DEFAULT_PARAMETERS[name] for name in LEGACY_PARAMETER_NAMES}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selected_parameters.json'
            path.write_text(json.dumps(old))
            manifest = Path(directory) / 'run_manifest.json'
            for value in ([], {'identity': {'prepared': []},
                               'identity_sha256': canonical_hash({'prepared': []})}):
                manifest.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    read_parameters(path)

    def test_inactive_fits_rejected(self):
        for model, fitted in [('retention-v1', ['tool_friction_coefficient']),
                              ('coulomb-v1', ['tool_stickiness']),
                              ('coulomb-adhesive-v1', ['tool_retention']),
                              ('retention-v1', ['tool_retention', 'tool_stickiness'])]:
            with self.subTest(model=model), self.assertRaises(ValueError):
                validate_tool_parameters(DEFAULT_PARAMETERS, SimulationConfig(
                    n_particles=1, tool_collision='sdf', tool_contact_model=model), fitted)
        with self.assertRaises(ValueError):
            SimulationConfig(n_particles=1, tool_contact_model='coulomb-adhesive-v1', tool_contact_absorption=.1)


class DerivativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)

    def test_contact_path_derivatives(self):
        for model in ('coulomb-v1', 'coulomb-adhesive-v1', 'retention-v1'):
            config = SimulationConfig(n_particles=1, grid=12, precision='f64', tool_collision='sdf',
                                      tool_contact_model=model, tool_contact_padding=.004)
            params = dict(DEFAULT_PARAMETERS, tool_friction_coefficient=.3,
                          tool_stickiness=0 if model == 'coulomb-v1' else .2)
            solver = Stepper(config, params, capacity=2, sdf=plane_sdf())
            control = ToolControl.stationary()
            control.poses[0, :3] = [.4, .4, .4]
            control.poses[1, :3] = [10, 10, 10]
            solver._set_control(control)
            probe = GradientProbe()
            probe.pos[None] = [.401, .4, .4]
            probe.normal[None] = [1, 0, 0]
            for path in ('velocity', 'grid', 'particle'):
                for velocity in ((-1, 2, .3), (1, 2, .3), (-1, 0, 0)):
                    probe.velocity[None] = velocity
                    solver.set_parameters(params)
                    solver.clear_parameter_gradients()
                    probe.evaluate(solver, path)
                    probe.loss.grad[None] = 1
                    probe.evaluate.grad(solver, path)
                    gradients = solver.parameter_gradients()
                    names = ['tool_friction_coefficient'] if model == 'coulomb-v1' else (
                        ['tool_stickiness'] if model == 'retention-v1' else ['tool_friction_coefficient', 'tool_stickiness'])
                    for name in names:
                        h = 1e-6
                        samples = []
                        for sign in (1, -1):
                            trial = dict(params)
                            trial[name] += sign * h
                            solver.set_parameters(trial)
                            probe.evaluate(solver, path)
                            samples.append(probe.loss[None])
                        fd = (samples[0] - samples[1]) / (2 * h)
                        self.assertAlmostEqual(gradients[name], fd, delta=2e-8,
                                               msg=f'{model} {path} {velocity} {name}')
                    if velocity == (-1, 2, .3):
                        self.assertTrue(any(abs(gradients[name]) > 1e-5 for name in names))

    def probe_gradients(self, solver, probe, params, path, names):
        """Adjoint gradients of the probe loss and their central finite differences."""
        solver.set_parameters(params)
        solver.clear_parameter_gradients()
        probe.velocity.grad[None] = [0, 0, 0]
        probe.pos.grad[None] = [0, 0, 0]
        probe.evaluate(solver, path)
        probe.loss.grad[None] = 1
        probe.evaluate.grad(solver, path)
        gradients = solver.parameter_gradients()
        differences = {}
        for name in names:
            h = 1e-6 * max(1.0, abs(params[name]))
            samples = []
            for sign in (1, -1):
                solver.set_parameters(dict(params, **{name: params[name] + sign * h}))
                probe.evaluate(solver, path)
                samples.append(probe.loss[None])
            differences[name] = (samples[0] - samples[1]) / (2 * h)
        solver.set_parameters(params)
        return gradients, differences

    def bounded_solver(self, floor='retention-v1'):
        config = SimulationConfig(n_particles=1, grid=12, precision='f64', tool_collision='sdf',
                                  tool_contact_model='coulomb-bounded-adhesive-v2', floor_contact_model=floor,
                                  tool_contact_padding=.004)
        params = dict(DEFAULT_PARAMETERS, tool_friction_coefficient=.3, tool_adhesion=150.0,
                      floor_friction_coefficient=.4)
        solver = Stepper(config, params, capacity=2, sdf=plane_sdf())
        control = ToolControl.stationary()
        control.poses[0, :3] = [.4, .4, .4]
        control.poses[1, :3] = [10, 10, 10]
        solver._set_control(control)
        return solver, params

    def test_bounded_adhesive_contact_path_derivatives(self):
        solver, params = self.bounded_solver()
        probe = GradientProbe()
        probe.pos[None] = [.401, .4, .4]
        probe.normal[None] = [1, 0, 0]
        names = ['tool_friction_coefficient', 'tool_adhesion']
        # Approach, fast separation, separation near the pull-off limit (150 m/s^2 * 2e-4 s = 0.03 m/s), pure approach.
        for path in ('velocity', 'grid', 'particle'):
            for velocity in ((-1, 2, .3), (1, 2, .3), (.02, .5, -.1), (-1, 0, 0)):
                probe.velocity[None] = velocity
                gradients, differences = self.probe_gradients(solver, probe, params, path, names)
                for name in names:
                    self.assertAlmostEqual(gradients[name], differences[name], delta=2e-8 + 1e-6 * abs(differences[name]),
                                           msg=f'{path} {velocity} {name}')
                if velocity == (.02, .5, -.1):
                    self.assertTrue(all(abs(gradients[name]) > 1e-5 for name in names), gradients)

    def test_floor_coulomb_derivatives(self):
        solver, params = self.bounded_solver(floor='coulomb-v1')
        probe = GradientProbe()
        for velocity in ((1, -.2, .3), (.05, -.2, 0), (.3, .2, -.4)):
            probe.velocity[None] = velocity
            gradients, differences = self.probe_gradients(solver, probe, params, 'floor', ['floor_friction_coefficient'])
            self.assertAlmostEqual(gradients['floor_friction_coefficient'], differences['floor_friction_coefficient'],
                                   delta=2e-8, msg=str(velocity))
            if velocity == (1, -.2, .3):
                self.assertGreater(abs(gradients['floor_friction_coefficient']), 1e-3)

    def test_co_moving_contact_gradients_are_finite(self):
        """Zero relative velocity at a contact must not produce NaN adjoints."""
        solver, params = self.bounded_solver(floor='coulomb-v1')
        probe = GradientProbe()
        probe.pos[None] = [.401, .4, .4]
        probe.normal[None] = [1, 0, 0]
        for path, velocity in (('velocity', (0, 0, 0)), ('grid', (0, 0, 0)), ('particle', (0, 0, 0)),
                               ('floor', (0, 0, 0)), ('floor', (0, -.2, 0))):
            for adhesion in (0.0, 150.0):
                probe.velocity[None] = velocity
                gradients, _ = self.probe_gradients(solver, probe, dict(params, tool_adhesion=adhesion), path, [])
                self.assertTrue(np.isfinite(list(gradients.values())).all(), f'{path} {velocity} {adhesion} {gradients}')
                self.assertTrue(np.isfinite(probe.velocity.grad[None].to_numpy()).all(), f'{path} {velocity} {adhesion}')

    def test_zero_sdf_normal_gradients_are_finite(self):
        """Mesh SDFs contain cells whose corner normals are all exactly zero."""
        sdf = plane_sdf()
        sdf.gradients[:] = 0
        for model, extra in (('coulomb-v1', {}), ('coulomb-adhesive-v1', {'tool_stickiness': .2}),
                             ('coulomb-bounded-adhesive-v2', {'tool_adhesion': 150.0})):
            config = SimulationConfig(n_particles=1, grid=12, precision='f64', tool_collision='sdf',
                                      tool_contact_model=model, tool_contact_padding=.004)
            params = dict(DEFAULT_PARAMETERS, tool_friction_coefficient=.3, **extra)
            solver = Stepper(config, params, capacity=2, sdf=sdf)
            control = ToolControl.stationary()
            control.poses[0, :3] = [.4, .4, .4]
            control.poses[1, :3] = [10, 10, 10]
            solver._set_control(control)
            probe = GradientProbe()
            probe.pos[None] = [.401, .4, .4]
            for path in ('grid', 'particle'):
                for velocity in ((0, 0, 0), (-1, 2, .3)):
                    probe.velocity[None] = velocity
                    gradients, _ = self.probe_gradients(solver, probe, params, path, [])
                    self.assertTrue(np.isfinite(list(gradients.values())).all(), f'{model} {path} {velocity}')
                    self.assertTrue(np.isfinite(probe.velocity.grad[None].to_numpy()).all(), f'{model} {path} {velocity}')
                    self.assertTrue(np.isfinite(probe.pos.grad[None].to_numpy()).all(), f'{model} {path} {velocity}')

    def test_zero_stickiness_exact_coulomb_and_withdrawal(self):
        probe = ContactProbe()
        probe.normal[None] = [1, 0, 0]
        probe.collider[None] = [-2, .3, .1]
        for velocity in ((-3, 2, .4), (0, .4, 0)):
            probe.velocity[None] = velocity
            outputs = []
            for model in ('coulomb-v1', 'coulomb-adhesive-v1'):
                solver = Stepper(SimulationConfig(n_particles=1, grid=12, precision='f64',
                                                 tool_contact_model=model), DEFAULT_PARAMETERS, capacity=2)
                probe.velocity_response(solver)
                outputs.append(np.asarray(probe.output_velocity[None]))
            np.testing.assert_array_equal(*outputs)
        solver.set_parameters(dict(DEFAULT_PARAMETERS, tool_stickiness=.2))
        probe.velocity_response(solver)
        np.testing.assert_allclose(np.asarray(probe.output_velocity[None]), [-.4, .38, .02], atol=1e-14)

    def test_three_step_mpm_contact_gradients(self):
        config, state, params, control, sdf = fixture(sdf=True, p2g_mode='serial',
                                                    tool_contact_model='coulomb-adhesive-v1')
        params.update(tool_friction_coefficient=.3, tool_stickiness=.2)
        solver = Stepper(config, params, capacity=4, sdf=sdf)
        seed = terminal_seed(state)

        def value(parameters):
            solver.set_parameters(parameters)
            solver.load_state(0, state)
            for step in range(3):
                solver.advance(step, control)
            return dot_state(solver.state(3), seed)

        value(params)
        solver.clear_state_gradients()
        solver.clear_parameter_gradients()
        solver.load_adjoint(3, seed)
        for step in reversed(range(3)):
            solver.reverse_step(step, control)
        gradients = solver.parameter_gradients()
        for name in ('tool_friction_coefficient', 'tool_stickiness'):
            h = 1e-6
            plus, minus = dict(params), dict(params)
            plus[name] += h
            minus[name] -= h
            fd = (value(plus) - value(minus)) / (2 * h)
            self.assertGreater(abs(gradients[name]), 1e-7)
            self.assertAlmostEqual(gradients[name], fd, delta=2e-6 + 3e-4 * abs(fd), msg=name)

    def rollout_gradients(self, config, state, params, control, sdf, names):
        solver = Stepper(config, params, capacity=4, sdf=sdf)
        seed = terminal_seed(state)

        def value(parameters):
            solver.set_parameters(parameters)
            solver.load_state(0, state)
            for step in range(3):
                solver.advance(step, control)
            return dot_state(solver.state(3), seed)

        value(params)
        solver.clear_state_gradients()
        solver.clear_parameter_gradients()
        solver.load_adjoint(3, seed)
        for step in reversed(range(3)):
            solver.reverse_step(step, control)
        gradients = solver.parameter_gradients()
        self.assertTrue(np.isfinite(list(gradients.values())).all(), gradients)
        for name in names:
            h = 1e-6 * max(1.0, abs(params[name]))
            fd = (value(dict(params, **{name: params[name] + h})) - value(dict(params, **{name: params[name] - h}))) / (2 * h)
            self.assertGreater(abs(gradients[name]), 1e-7, name)
            self.assertAlmostEqual(gradients[name], fd, delta=2e-6 + 3e-4 * abs(fd), msg=name)

    def test_three_step_mpm_bounded_adhesive_gradients(self):
        config, state, params, control, sdf = fixture(sdf=True, p2g_mode='serial',
                                                    tool_contact_model='coulomb-bounded-adhesive-v2')
        params.update(tool_friction_coefficient=.3, tool_adhesion=200.0)
        self.rollout_gradients(config, state, params, control, sdf, ['tool_friction_coefficient', 'tool_adhesion'])

    def test_three_step_mpm_floor_coulomb_gradients(self):
        config, state, params, control, sdf = fixture(floor=True, p2g_mode='serial')
        config = replace(config, floor_contact_model='coulomb-v1')
        params.update(floor_friction_coefficient=.4)
        self.rollout_gradients(config, state, params, control, sdf, ['floor_friction_coefficient'])


if __name__ == '__main__':
    unittest.main()
