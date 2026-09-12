"""Finite-difference checks for the staged solver, separate from reference parity."""
import unittest
from dataclasses import replace

import numpy as np
import taichi as ti

from experiments.differentiable_mpm.solver import Stepper
from experiments.differentiable_mpm.state import (
    DEFAULT_PARAMETERS, PARAMETER_NAMES, STATE_NAMES, InvalidStateError,
    ParticleState, SDFData, SimulationConfig, ToolControl,
)


def fixture(plastic=False, floor=False, sdf=False, p2g_mode="atomic"):
    config = SimulationConfig(n_particles=4, grid=12, precision="f64", dt=2e-4,
                              particle_mass=0.001, particle_volume=1e-6,
                              gravity=-2.0, plasticity="stretch-clamp" if plastic else "none",
                              use_jp=plastic, jp_hardening=2.0 if plastic else 0.0,
                              floor_y=0.028 if floor else 0.0,
                              floor_plastic_damping_band=0.01 if floor else 0.0,
                              plastic_affine_damping=0.87 if floor else 1.0,
                              tool_collision="sdf" if sdf else "none",
                              tool_contact_padding=0.004, p2g_mode=p2g_mode)
    state = ParticleState.initial([[.417, .40, .405], [.441, .415, .409],
                                   [.456, .428, .432], [.431, .421, .445]], np.float64)
    state.v[:] = [[.1, -.2, .04], [-.05, -.1, .08], [.02, -.3, -.03], [.01, -.1, .07]]
    state.C[:] = np.array([[.2, .3, .02], [-.08, -.1, .07], [.03, -.01, .15]])
    state.F[:] = np.array([[.84 if plastic else .94, .03, .01], [.0, 1.04, -.02],
                           [.01, .015, 1.17 if plastic else 1.09]])
    state.Jp[:] = [1.01, .97, 1.04, .93]
    if floor:
        state.x[:, 1] = [0.018, 0.020, 0.021, 0.023]
    params = dict(DEFAULT_PARAMETERS, youngs_modulus=4200.0, viscosity=8.0,
                  plastic_min=.91, plastic_max=1.10, tool_retention=.28, floor_retention=.43)
    control = ToolControl.stationary()
    sdf_data = None
    if sdf:
        r = 13
        axis = np.linspace(-.18, .18, r)
        coordinates = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        norm = np.linalg.norm(coordinates, axis=-1)
        distance = norm - .08
        normals = coordinates / np.maximum(norm[..., None], 1e-12)
        sdf_data = SDFData(np.stack([distance, distance]), np.stack([normals, normals]),
                           np.full((2, 3), -.18), np.full((2, 3), .36 / (r - 1)))
        control.poses[0, :3] = [.43, .40, .37]
        control.poses[1, :3] = [.42, .42, .52]
        angle = .27
        control.poses[0, 3:] = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        control.velocities[:] = [[.015, -.012, .05, .2, -.3, .4],
                                 [-.01, .012, -.035, -.1, .25, .35]]
    return config, state, params, control, sdf_data


def dot_state(a, b):
    return sum(float(np.sum(getattr(a, n) * getattr(b, n))) for n in STATE_NAMES)


def terminal_seed(state):
    rng = np.random.default_rng(527)
    result = state.zeros_like()
    for name, scale in zip(STATE_NAMES, [.3, .7, .2, .3, .2]):
        getattr(result, name)[:] = rng.normal(size=getattr(result, name).shape) * scale
    return result


@ti.kernel
def reference_grid_normalization(mass: ti.template(), momentum: ti.template(), output: ti.template()):
    for I in ti.grouped(mass):
        velocity = momentum[I]
        if mass[I] > 0:
            velocity = momentum[I] / mass[I]
        output[I] = velocity


class NormalizationAdjointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1, debug=True,
                offline_cache=False)

    def make(self):
        return Stepper(SimulationConfig(n_particles=1, grid=8, precision="f32", gravity=0.0, floor_y=-1.0),
                       capacity=2)

    def test_tiny_positive_and_nonpositive_mass(self):
        solver = self.make()
        mass = np.zeros((8, 8, 8), np.float32)
        momentum = np.zeros((8, 8, 8, 3), np.float32)
        incoming = np.zeros_like(momentum)
        cases = [((3, 3, 3), 1.50686393589534e-20), ((4, 3, 3), 1e-30),
                 ((3, 4, 3), 1e-35), ((3, 3, 4), 1e-4),
                 ((4, 4, 3), -1e-6), ((4, 4, 4), 0.0)]
        for index, value in cases:
            mass[index] = value
            scale = float(mass[index]) if value > 0 else 1.0
            momentum[index] = np.array([.4, -.2, .1]) * scale
            incoming[index] = np.array([.2, -.1, .3]) * scale
        solver.grid_m.from_numpy(mass)
        solver.grid_p.from_numpy(momentum)
        reference = ti.Vector.field(3, ti.f32, shape=(8, 8, 8))
        reference_grid_normalization(solver.grid_m, solver.grid_p, reference)
        solver._normalize_grid()
        np.testing.assert_array_equal(solver.grid_u.to_numpy(), reference.to_numpy())
        solver._grid_operations()
        np.testing.assert_array_equal(solver.grid_v.to_numpy(), reference.to_numpy())
        solver._clear_scratch_gradients()
        solver.grid_u.grad.from_numpy(incoming)
        solver._normalize_grid_backward()
        mass_gradient = solver.grid_m.grad.to_numpy()
        momentum_gradient = solver.grid_p.grad.to_numpy()
        self.assertTrue(np.isfinite(mass_gradient).all())
        self.assertTrue(np.isfinite(momentum_gradient).all())
        for index, value in cases:
            if value > 0:
                expected_p = incoming[index].astype(float) / float(mass[index])
                expected_m = -np.dot(momentum[index].astype(float) / float(mass[index]), expected_p)
            else:
                expected_p = incoming[index].astype(float)
                expected_m = 0.0
            np.testing.assert_allclose(momentum_gradient[index], expected_p, rtol=2e-6, atol=1e-8)
            np.testing.assert_allclose(mass_gradient[index], expected_m, rtol=2e-6, atol=1e-8)
        solver._normalize_grid_backward()
        np.testing.assert_array_equal(solver.grid_m.grad.to_numpy(), 2 * mass_gradient)
        np.testing.assert_array_equal(solver.grid_p.grad.to_numpy(), 2 * momentum_gradient)

    def test_recorded_mass_finite_difference(self):
        solver = self.make()
        index = (3, 3, 3)
        mass = np.float32(1.50686393589534e-20)
        momentum = np.array([1.7265644015834862e-23, -1.0280440996638743e-22,
                             -1.820572797886493e-23], np.float32)
        incoming = np.array([-3.8159966154152235e-22, -1.4763675477004859e-21,
                             -1.518090994201026e-22], np.float32)
        solver.grid_m[index] = mass
        solver.grid_p[index] = momentum
        solver._normalize_grid()
        solver._clear_scratch_gradients()
        solver.grid_u.grad[index] = incoming
        solver._normalize_grid_backward()
        derivative = float(solver.grid_m.grad[index])
        expected = -np.dot(momentum.astype(float) / float(mass), incoming.astype(float) / float(mass))
        np.testing.assert_allclose(derivative, expected, rtol=2e-6, atol=1e-9)
        finite_differences = []
        for relative_h in [0.01, 0.002, 0.0005]:
            high = np.float32(mass * (1 + relative_h))
            low = np.float32(mass * (1 - relative_h))
            solver.grid_m[index] = high
            solver._normalize_grid()
            upper = np.dot(solver.grid_u.to_numpy()[index].astype(float), incoming.astype(float))
            solver.grid_m[index] = low
            solver._normalize_grid()
            lower = np.dot(solver.grid_u.to_numpy()[index].astype(float), incoming.astype(float))
            finite_differences.append(float((upper - lower) / (float(high) - float(low))))
        self.assertLess(min(abs(value - derivative) for value in finite_differences), 2e-7)


class SerialP2GOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=4, debug=True,
                offline_cache=False)

    def test_serial_particle_order_and_analytic_velocity_adjoint(self):
        n = 4096
        config = SimulationConfig(n_particles=n, grid=8, precision="f32", particle_mass=1.0,
                                  p2g_mode="serial", gravity=0.0, floor_y=-1.0)
        solver = Stepper(config, capacity=2)
        state = ParticleState.initial(np.full((n, 3), 3.5 / 8, np.float32))
        state.v[:, 0] = np.tile(np.array([2 ** 27, 8.0, -(2 ** 27), 8.0], np.float32), n // 4)
        solver.load_state(0, state)
        solver.affine.fill(0)
        # Each of eight occupied nodes receives [2**24, 1, -2**24, 1] repeatedly.
        # Strict particle-index accumulation is 1; reassociation can change it.
        expected_mass = np.zeros((8, 8, 8), np.float32)
        expected_momentum = np.zeros((8, 8, 8, 3), np.float32)
        expected_mass[3:5, 3:5, 3:5] = n / 8
        expected_momentum[3:5, 3:5, 3:5, 0] = 1.0
        previous = None
        for _ in range(3):
            solver._clear_grid()
            solver._p2g(0)
            mass, momentum = solver.grid_m.to_numpy(), solver.grid_p.to_numpy()
            np.testing.assert_array_equal(mass, expected_mass)
            np.testing.assert_array_equal(momentum, expected_momentum)
            current = mass.tobytes(), momentum.tobytes()
            if previous is not None:
                self.assertEqual(current, previous)
            previous = current
        solver.clear_state_gradients()
        solver._clear_scratch_gradients()
        seed = np.zeros_like(expected_momentum)
        seed[3:5, 3:5, 3:5] = [1.0, -2.0, 3.0]
        solver.grid_p.grad.from_numpy(seed)
        solver._p2g.grad(0)
        expected_velocity_gradient = np.tile(np.array([1.0, -2.0, 3.0], np.float32), (n, 1))
        np.testing.assert_array_equal(solver.adjoint(0).v, expected_velocity_gradient)


class SolverGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, debug=True,
                offline_cache=False)

    def make(self, **options):
        config, state, params, control, sdf = fixture(**options)
        solver = Stepper(config, params, capacity=8, sdf=sdf)
        return solver, state, params, control

    def value(self, solver, state, params, control, seed, steps=1):
        solver.set_parameters(params)
        solver.load_state(0, state)
        for t in range(steps):
            solver.advance(t, control)
        return dot_state(solver.state(steps), seed)

    def gradients(self, solver, state, params, control, seed, steps=1):
        value = self.value(solver, state, params, control, seed, steps)
        solver.clear_state_gradients()
        solver.clear_parameter_gradients()
        solver.load_adjoint(steps, seed)
        primals = [solver.state(t) for t in range(steps + 1)]
        for t in reversed(range(steps)):
            solver.reverse_step(t, control)
        for t, before in enumerate(primals):
            for name in STATE_NAMES:
                np.testing.assert_array_equal(getattr(before, name), getattr(solver.state(t), name))
        for name in STATE_NAMES:
            np.testing.assert_array_equal(getattr(seed, name), getattr(solver.adjoint(steps), name))
        return value, solver.adjoint(0), solver.parameter_gradients()

    def assert_derivative(self, actual, candidates, context, relative=3e-4, absolute=2e-7):
        self.assertTrue(np.isfinite(actual), context)
        errors = [abs(actual - fd) for fd in candidates]
        scale = max(abs(actual), max(abs(fd) for fd in candidates), 1e-5)
        self.assertLessEqual(min(errors), absolute + relative * scale,
                             f"{context}: AD={actual:.12g}, FD={candidates}, errors={errors}")

    def check_parameter_gradients(self, solver, state, params, control, seed, steps=1, names=PARAMETER_NAMES):
        _, _, gradients = self.gradients(solver, state, params, control, seed, steps)
        for name in names:
            scale = max(abs(params[name]), 0.1)
            fds = []
            for epsilon in (2e-4, 5e-5, 1e-5):
                h = epsilon * scale
                plus, minus = dict(params), dict(params)
                plus[name] += h
                minus[name] -= h
                fds.append((self.value(solver, state, plus, control, seed, steps) -
                            self.value(solver, state, minus, control, seed, steps)) / (2 * h))
            self.assert_derivative(gradients[name], fds, f"{name}, steps={steps}")
        return gradients

    def check_state_gradients(self, solver, state, params, control, seed, steps=1):
        _, gradient, _ = self.gradients(solver, state, params, control, seed, steps)
        rng = np.random.default_rng(732)
        for name, scale in zip(STATE_NAMES, [.03, .3, .3, .1, .1]):
            direction = rng.normal(size=getattr(state, name).shape) * scale
            predicted = float(np.sum(getattr(gradient, name) * direction))
            fds = []
            for epsilon in (2e-4, 5e-5, 1e-5):
                plus, minus = state.copy(), state.copy()
                getattr(plus, name)[:] += epsilon * direction
                getattr(minus, name)[:] -= epsilon * direction
                fds.append((self.value(solver, plus, params, control, seed, steps) -
                            self.value(solver, minus, params, control, seed, steps)) / (2 * epsilon))
            self.assert_derivative(predicted, fds, f"state {name}, steps={steps}", relative=8e-4)

    def test_elastic_and_viscous_one_step(self):
        solver, state, params, control = self.make()
        seed = terminal_seed(state)
        gradients = self.check_parameter_gradients(solver, state, params, control, seed)
        self.check_state_gradients(solver, state, params, control, seed)
        self.assertNotEqual(gradients["youngs_modulus"], 0)
        self.assertNotEqual(gradients["poisson_ratio"], 0)
        self.assertNotEqual(gradients["viscosity"], 0)
        self.assertEqual(gradients["plastic_min"], 0)
        self.assertEqual(gradients["plastic_max"], 0)

    def test_active_plastic_bounds_and_updated_history(self):
        solver, state, params, control = self.make(plastic=True)
        seed = terminal_seed(state)
        gradients = self.check_parameter_gradients(solver, state, params, control, seed)
        self.check_state_gradients(solver, state, params, control, seed)
        self.assertNotEqual(gradients["plastic_min"], 0)
        self.assertNotEqual(gradients["plastic_max"], 0)

    def test_short_trajectory(self):
        solver, state, params, control = self.make(plastic=True)
        seed = terminal_seed(state)
        self.check_parameter_gradients(solver, state, params, control, seed, steps=5,
                                       names=("youngs_modulus", "poisson_ratio", "viscosity", "plastic_min", "plastic_max"))
        self.check_state_gradients(solver, state, params, control, seed, steps=5)

    def test_floor_contact(self):
        solver, state, params, control = self.make(floor=True)
        seed = terminal_seed(state)
        self.check_parameter_gradients(solver, state, params, control, seed,
                                       names=("floor_retention", "youngs_modulus", "viscosity"))
        self.check_state_gradients(solver, state, params, control, seed)
        self.assertGreater(solver.diagnostics()["particle_floor"], 0)

    def test_moving_rotating_sdf_contact(self):
        solver, state, params, control = self.make(sdf=True)
        seed = terminal_seed(state)
        self.check_parameter_gradients(solver, state, params, control, seed,
                                       names=("tool_retention", "youngs_modulus", "viscosity"))
        self.check_state_gradients(solver, state, params, control, seed)
        counts = solver.diagnostics()
        self.assertGreater(counts["grid_tool0"] + counts["grid_tool1"], 0)
        self.assertGreater(counts["particle_tool0"], 0)
        self.assertGreater(counts["particle_tool1"], 0)

    def test_floor_zero_negative_grid_mass(self):
        config, state, params, control, sdf = fixture(floor=True)
        config = replace(config, floor_y=0.0)
        state.x[:, 1] = [0.0, 1e-5, 2e-5, 3e-5]
        solver = Stepper(config, params, capacity=8)
        seed = terminal_seed(state)
        self.check_parameter_gradients(solver, state, params, control, seed,
                                       names=("floor_retention", "youngs_modulus", "viscosity"))
        self.check_state_gradients(solver, state, params, control, seed)
        mass = solver.grid_m.to_numpy()
        momentum = solver.grid_p.to_numpy()
        velocity = solver.grid_v.to_numpy()
        self.assertTrue(np.any(mass < 0), "Fixture must exercise the admitted negative stencil weights")
        np.testing.assert_array_equal(velocity[mass <= 0], momentum[mass <= 0])

    def test_atomic_default_and_serial_forward(self):
        for options in ({}, {"plastic": True, "floor": True}, {"sdf": True}):
            with self.subTest(options=options):
                config, state, params, control, sdf = fixture(plastic=options.get("plastic", False),
                                                            floor=options.get("floor", False),
                                                            sdf=options.get("sdf", False))
                self.assertEqual(config.p2g_mode, "atomic")
                if options.get("floor"):
                    config = replace(config, floor_y=0.0)
                    state.x[:, 1] = [0.0, 1e-5, 2e-5, 3e-5]
                atomic = Stepper(config, params, capacity=6, sdf=sdf)
                serial = Stepper(replace(config, p2g_mode="serial"), params, capacity=6, sdf=sdf)
                atomic.load_state(0, state)
                serial.load_state(0, state)
                original = []
                grids = []
                counts = []
                for step in range(5):
                    atomic.advance(step, control)
                    serial.advance(step, control)
                    atomic_state, serial_state = atomic.state(step + 1), serial.state(step + 1)
                    for name in STATE_NAMES:
                        np.testing.assert_allclose(getattr(serial_state, name), getattr(atomic_state, name),
                                                   rtol=2e-12, atol=2e-12, err_msg=f"{name}, step={step}")
                    self.assertEqual(serial.diagnostics(), atomic.diagnostics())
                    original.append(serial_state)
                    grids.append(tuple(getattr(serial, name).to_numpy().tobytes()
                                       for name in ("grid_m", "grid_p", "grid_u", "grid_v")))
                    counts.append(serial.diagnostics())
                    if step == 0 and options.get("floor"):
                        self.assertTrue(np.any(serial.grid_m.to_numpy() < 0))
                serial.load_state(0, state)
                for step in range(5):
                    serial.advance(step, control)
                    replay = serial.state(step + 1)
                    for name in STATE_NAMES:
                        self.assertEqual(getattr(replay, name).tobytes(), getattr(original[step], name).tobytes())
                    self.assertEqual(tuple(getattr(serial, name).to_numpy().tobytes()
                                           for name in ("grid_m", "grid_p", "grid_u", "grid_v")), grids[step])
                    self.assertEqual(serial.diagnostics(), counts[step])

    def test_serial_plastic_trajectory_gradients(self):
        solver, state, params, control = self.make(plastic=True, p2g_mode="serial")
        seed = terminal_seed(state)
        self.check_parameter_gradients(solver, state, params, control, seed, steps=5,
                                       names=("youngs_modulus", "poisson_ratio", "viscosity", "plastic_min", "plastic_max"))
        self.check_state_gradients(solver, state, params, control, seed, steps=5)
        atomic = Stepper(replace(solver.config, p2g_mode="atomic"), params, capacity=8)
        _, atomic_state_gradient, atomic_parameter_gradient = self.gradients(atomic, state, params, control, seed, steps=5)
        _, serial_state_gradient, serial_parameter_gradient = self.gradients(solver, state, params, control, seed, steps=5)
        # Compare gradients numerically; serial P2G does not serialize other reductions.
        for name in STATE_NAMES:
            np.testing.assert_allclose(getattr(serial_state_gradient, name), getattr(atomic_state_gradient, name),
                                       rtol=1e-9, atol=1e-10, err_msg=name)
        for name in PARAMETER_NAMES:
            np.testing.assert_allclose(serial_parameter_gradient[name], atomic_parameter_gradient[name],
                                       rtol=1e-9, atol=1e-10, err_msg=name)

    def test_serial_moving_rotating_sdf_gradients(self):
        solver, state, params, control = self.make(sdf=True, p2g_mode="serial")
        seed = terminal_seed(state)
        self.check_parameter_gradients(solver, state, params, control, seed,
                                       names=("tool_retention", "youngs_modulus", "viscosity"))
        self.check_state_gradients(solver, state, params, control, seed)
        counts = solver.diagnostics()
        self.assertGreater(counts["grid_tool0"] + counts["grid_tool1"], 0)
        self.assertGreater(counts["particle_tool0"], 0)
        self.assertGreater(counts["particle_tool1"], 0)

    def test_invalid_deformation_and_stencil(self):
        solver, state, params, control = self.make()
        state.F[0, 0, 0] = -1
        solver.load_state(0, state)
        with self.assertRaises(InvalidStateError):
            solver.advance(0, control)
        state.F[0] = np.eye(3)
        state.x[0, 0] = 1.5
        solver.load_state(0, state)
        with self.assertRaises(InvalidStateError):
            solver.advance(0, control)


if __name__ == "__main__":
    unittest.main()
