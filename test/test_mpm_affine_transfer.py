# pyright: reportMissingImports=false

import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import taichi_viscoelastic_mpm_scene as mpm


class MpmAffineTransferTests(unittest.TestCase):
    def tearDown(self):
        ti.reset()

    def build_affine_particle(self, grid, gradient, velocity, viscosity=0.0):
        ti.reset()
        ti.init(arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False)
        args = SimpleNamespace(
            particles=1,
            grid=grid,
            dt=0.0001,
            density=1000.0,
            youngs_modulus=0.0,
            poisson_ratio=0.3,
            viscosity=viscosity,
            gravity=0.0,
            floor_y=-1.0,
            tool_close_time=1.0,
            tool_motion_start=0.0,
            tool_contact_padding=0.0,
            tool_contact_friction=1.0,
            tool_contact_absorption=0.0,
            tool_stickiness=0.0,
            floor_friction=1.0,
            floor_stickiness=0.0,
            floor_absorption=0.0,
            floor_plastic_damping_band=0.0,
            velocity_damping=1.0,
            pure_viscoelastic=True,
            plastic_min=0.9,
            plastic_max=1.1,
            plastic_velocity_damping=1.0,
            plastic_affine_damping=1.0,
            use_jp=False,
            jp_hardening=0.0,
            jp_min=0.5,
            jp_max=2.0,
            ros_control=True,
            tool_collision="none",
        )
        positions, _, initialize, substep, _, _ = mpm.build_sim(args)
        initialize()
        # The error-reporting wrapper retains the actual solver state fields.
        state = inspect.getclosurevars(substep).nonlocals
        position = (grid // 2 + np.array([0.83, 1.18, 0.94])) / grid
        positions.from_numpy(position[None].astype(np.float32))
        state["v"].from_numpy(np.asarray(velocity, dtype=np.float32)[None])
        state["C"].from_numpy(np.asarray(gradient, dtype=np.float32)[None])
        return args, positions, substep, state

    def test_affine_gradient_and_deformation_are_reproduced_across_grids(self):
        gradient = np.array([[1.2, -0.4, 0.8], [0.3, 3.0, -0.6], [-0.2, 0.5, -1.0]])
        velocity = np.array([0.12, -0.21, 0.06])
        for grid in (24, 48, 96):
            with self.subTest(grid=grid):
                args, positions, substep, state = self.build_affine_particle(grid, gradient, velocity)
                initial_position = positions.to_numpy()[0].copy()
                substep(0.0)
                np.testing.assert_allclose(state["C"].to_numpy()[0], gradient, rtol=3e-5, atol=3e-5)
                np.testing.assert_allclose(state["v"].to_numpy()[0], velocity, rtol=3e-5, atol=3e-6)
                np.testing.assert_allclose(positions.to_numpy()[0], initial_position + args.dt * velocity, atol=1e-7)
                substep(args.dt)
                increment = np.eye(3) + args.dt * gradient
                np.testing.assert_allclose(state["F"].to_numpy()[0], increment @ increment, rtol=3e-6, atol=3e-7)
                np.testing.assert_allclose(state["C"].to_numpy()[0], gradient, rtol=6e-5, atol=6e-5)

    def test_affine_gradient_is_reproduced_next_to_scene_zero(self):
        gradient = np.array([[1.2, -0.4, 0.8], [0.3, 3.0, -0.6], [-0.2, 0.5, -1.0]])
        velocity = np.array([0.12, 0.21, 0.06])
        args, positions, substep, state = self.build_affine_particle(48, gradient, velocity)
        for fraction in (0.0, 0.016, 0.49):
            with self.subTest(fraction=fraction):
                position = np.array([0.51, fraction / args.grid, 0.53], dtype=np.float32)
                positions.from_numpy(position[None])
                state["v"].from_numpy(velocity[None].astype(np.float32))
                state["C"].from_numpy(gradient[None].astype(np.float32))
                substep(0.0)
                np.testing.assert_allclose(state["v"].to_numpy()[0], velocity, rtol=3e-5, atol=3e-6)
                np.testing.assert_allclose(state["C"].to_numpy()[0], gradient, rtol=3e-5, atol=3e-5)
                np.testing.assert_allclose(positions.to_numpy()[0], position + args.dt * velocity, atol=1e-7)

    def test_translation_does_not_create_a_velocity_gradient(self):
        velocity = np.array([0.12, -0.21, 0.06])
        _, _, substep, state = self.build_affine_particle(48, np.zeros((3, 3)), velocity)
        substep(0.0)
        np.testing.assert_allclose(state["v"].to_numpy()[0], velocity, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(state["C"].to_numpy()[0], 0.0, atol=3e-5)

    def test_rigid_rotation_preserves_spin_without_viscous_strain_rate(self):
        gradient = np.array([[0.0, -2.0, 0.5], [2.0, 0.0, -0.3], [-0.5, 0.3, 0.0]])
        _, _, substep, state = self.build_affine_particle(48, gradient, np.zeros(3), viscosity=5.0)
        substep(0.0)
        actual = state["C"].to_numpy()[0]
        np.testing.assert_allclose(actual, gradient, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(actual + actual.T, 0.0, atol=3e-5)

    def test_viscous_stress_uses_the_physical_affine_gradient(self):
        gradient = np.array([[1.2, -0.4, 0.8], [-0.4, 0.6, -0.2], [0.8, -0.2, -0.3]])
        viscosity = 5.0
        for grid in (24, 48, 96):
            with self.subTest(grid=grid):
                args, _, substep, state = self.build_affine_particle(grid, gradient, np.zeros(3), viscosity)
                substep(0.0)
                # A lone particle gives an affine grid field after P2G normalization.
                # With E=0, only the known viscous stress changes its affine part.
                expected = gradient - 4 * args.dt * grid ** 2 * viscosity / args.density * (gradient + gradient.T)
                np.testing.assert_allclose(state["C"].to_numpy()[0], expected, rtol=3e-5, atol=3e-5)
                np.testing.assert_allclose(state["v"].to_numpy()[0], 0.0, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
