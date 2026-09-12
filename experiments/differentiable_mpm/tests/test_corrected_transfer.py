"""Analytic transfer, floor-padding and viscosity checks for versioned MPM physics."""
import unittest

import numpy as np
import taichi as ti

from experiments.differentiable_mpm.solver import Stepper
from experiments.differentiable_mpm.state import DEFAULT_PARAMETERS, InvalidStateError, ParticleState, SimulationConfig


AFFINE = np.array([[.2, .3, .02], [-.08, -.1, .07], [.03, -.01, .15]], dtype=np.float64)
TRANSLATION = np.array([.12, -.05, .03], dtype=np.float64)


def grid_coordinates(solver):
    axis = (np.arange(solver.allocated_grid, dtype=np.float64) + solver.grid_offset) * solver.dx
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)


def checked_load(solver, state):
    solver.load_state(0, state)
    solver._reset_diagnostics()
    solver._check_state(0, 0)
    solver._raise_invalid()


class CorrectedTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, debug=True, offline_cache=False)

    def make(self, positions, grid=24, physics_version="corrected-v1", p2g_mode="atomic"):
        state = ParticleState.initial(positions, np.float64)
        config = SimulationConfig(n_particles=len(state.x), grid=grid, precision="f64", dt=2e-4,
                                  particle_mass=.001, particle_volume=1e-6, gravity=0.0, floor_y=0.0,
                                  physics_version=physics_version, p2g_mode=p2g_mode)
        params = dict(DEFAULT_PARAMETERS, youngs_modulus=4200.0, viscosity=8.0)
        return Stepper(config, params, capacity=2), state, params

    def test_affine_reconstruction_at_three_grid_resolutions(self):
        positions = [[.437, .0, .410], [.370, .000125, .431], [.451, .401, .433]]
        for grid in (24, 48, 96):
            for version in ("corrected-v1", "legacy-v1"):
                with self.subTest(grid=grid, physics_version=version):
                    solver, state, _ = self.make(positions, grid, version)
                    checked_load(solver, state)
                    coordinates = grid_coordinates(solver)
                    velocity = coordinates @ AFFINE.T + TRANSLATION
                    solver.grid_v.from_numpy(velocity)
                    solver._g2p(0)
                    expected_scale = 1.0 if version == "corrected-v1" else solver.dx
                    expected_C = np.repeat((expected_scale * AFFINE)[None], len(positions), axis=0)
                    np.testing.assert_allclose(solver.new_C.to_numpy(), expected_C, rtol=2e-12, atol=2e-12)
                    np.testing.assert_allclose(solver.adv_v.to_numpy(), state.x @ AFFINE.T + TRANSLATION,
                                               rtol=2e-12, atol=2e-12)
                    np.testing.assert_allclose(solver.adv_x.to_numpy(),
                                               state.x + solver.dt * (state.x @ AFFINE.T + TRANSLATION),
                                               rtol=2e-12, atol=2e-12)
                    expected_size = grid + 1 if version == "corrected-v1" else grid
                    self.assertEqual(tuple(solver.grid_v.shape), (expected_size,) * 3)

    def test_ghost_grid_gradient_and_physical_coordinates(self):
        solver, state, _ = self.make([[.437, 0.0, .410]])
        checked_load(solver, state)
        solver.grid_v.from_numpy(grid_coordinates(solver) @ AFFINE.T + TRANSLATION)
        solver._g2p(0)
        solver.clear_state_gradients()
        solver._clear_scratch_gradients()
        incoming = np.array([[.3, -.2, .5]], dtype=np.float64)
        solver.adv_v.grad.from_numpy(incoming)
        solver._g2p.grad(0)
        np.testing.assert_allclose(solver.adjoint(0).x, incoming @ AFFINE, rtol=2e-12, atol=2e-12)
        gradient = solver.grid_v.grad.to_numpy()
        self.assertGreater(np.max(np.abs(gradient[:, 0, :, :])), 0,
                           "The y=-1 ghost layer must participate in the reverse transfer")
        # Downloaded array index zero is logical grid index -1, not physical zero.
        np.testing.assert_array_equal(np.array(solver.grid_v[-1, -1, -1]),
                                      solver.grid_v.to_numpy()[0, 0, 0])
        np.testing.assert_array_equal(np.array(solver.grid_v.grad[-1, -1, -1]), gradient[0, 0, 0])
        expected_corner = np.full(3, -solver.dx) @ AFFINE.T + TRANSLATION
        np.testing.assert_allclose(solver.grid_v.to_numpy()[0, 0, 0], expected_corner, rtol=0, atol=1e-15)

    def test_floor_stencil_nonnegative_mass_and_conservation(self):
        grid = 24
        positions = [[.417, 0.0, .405], [.441, 1e-7, .409],
                     [.456, .25 / grid, .432], [.431, .5 / grid, .445]]
        solver, state, _ = self.make(positions, grid, p2g_mode="serial")
        state.v[:] = [[.1, -.2, .04], [-.05, -.1, .08], [.02, -.3, -.03], [.01, -.1, .07]]
        checked_load(solver, state)
        solver.affine.fill(0)
        solver._clear_grid()
        solver._p2g(0)
        mass, momentum = solver.grid_m.to_numpy(), solver.grid_p.to_numpy()
        self.assertTrue(np.all(mass >= 0), "Floor-based quadratic weights must not produce negative mass")
        self.assertGreater(float(mass[:, 0, :].sum()), 0, "Floor-adjacent particles must use the y=-1 ghost layer")
        particle_mass = solver.config.particle_mass
        np.testing.assert_allclose(mass.sum(), len(positions) * particle_mass, rtol=2e-13, atol=1e-16)
        np.testing.assert_allclose(momentum.sum(axis=(0, 1, 2)), particle_mass * state.v.sum(axis=0),
                                   rtol=2e-13, atol=1e-16)
        coordinates = grid_coordinates(solver)
        np.testing.assert_allclose((mass[..., None] * coordinates).sum(axis=(0, 1, 2)),
                                   particle_mass * state.x.sum(axis=0), rtol=2e-13, atol=1e-16)
        solver._normalize_grid()
        # This is transfer conservation, before physical floor/contact impulses.
        solver.grid_v.from_numpy(solver.grid_u.to_numpy())
        solver._g2p(0)
        np.testing.assert_allclose(particle_mass * solver.adv_v.to_numpy().sum(axis=0),
                                   particle_mass * state.v.sum(axis=0), rtol=2e-13, atol=1e-16)
        np.testing.assert_array_equal(solver.grid_u.to_numpy()[mass == 0], momentum[mass == 0])

    def test_known_viscosity_stress_and_transfer(self):
        eta = 8.0
        symmetric_rate = AFFINE + AFFINE.T
        for grid in (24, 48, 96):
            with self.subTest(grid=grid):
                positions = (np.array([[grid // 2 + .83, grid // 2 + 1.07, grid // 2 + .69]]) / grid)
                solver, state, params = self.make(positions, grid)
                state.C[0] = AFFINE
                state.v[0] = TRANSLATION
                checked_load(solver, state)
                solver.corrected.from_numpy(np.eye(3, dtype=np.float64)[None])
                solver.rotation.from_numpy(np.eye(3, dtype=np.float64)[None])
                solver.history.fill(1)
                for viscosity in (0.0, eta):
                    params["viscosity"] = viscosity
                    solver.set_parameters(params)
                    solver._stress(0)
                    stress = viscosity * symmetric_rate
                    transfer_factor = 4 * solver.inv_dx ** 2
                    expected_affine = (solver.config.particle_mass * AFFINE
                                       - solver.dt * solver.config.particle_volume * transfer_factor * stress)
                    np.testing.assert_allclose(solver.affine.to_numpy()[0], expected_affine,
                                               rtol=2e-13, atol=2e-16)
                    solver._clear_grid()
                    solver._p2g(0)
                    solver._normalize_grid()
                    solver._grid_operations()
                    solver._g2p(0)
                    reconstructed = solver.new_C.to_numpy()[0]
                    np.testing.assert_allclose(reconstructed, expected_affine / solver.config.particle_mass,
                                               rtol=2e-12, atol=2e-12)
                    np.testing.assert_allclose(solver.adv_v.to_numpy()[0], TRANSLATION, rtol=2e-12, atol=2e-12)
                    inferred_stress = ((AFFINE - reconstructed) * solver.config.particle_mass
                                       / (solver.dt * solver.config.particle_volume * transfer_factor))
                    np.testing.assert_allclose(inferred_stress, stress, rtol=2e-10, atol=2e-10)
                    inferred_eta = np.sum(inferred_stress * symmetric_rate) / np.sum(symmetric_rate ** 2)
                    self.assertAlmostEqual(float(inferred_eta), viscosity, delta=2e-10)

    def test_corrected_stencil_rejects_unallocated_nodes(self):
        solver, state, _ = self.make([[.42, 0.0, .41]])
        checked_load(solver, state)
        for axis in range(3):
            for value in (-.51 * solver.dx, 1.0):
                with self.subTest(axis=axis, value=value):
                    invalid = state.copy()
                    invalid.x[0, axis] = value
                    with self.assertRaisesRegex(InvalidStateError, "input stencil"):
                        checked_load(solver, invalid)
        # The lower ghost cell is allocated, but there is no extra upper ghost cell.
        edge = state.copy()
        edge.x[0, 1] = -.49 * solver.dx
        checked_load(solver, edge)


if __name__ == "__main__":
    unittest.main()
