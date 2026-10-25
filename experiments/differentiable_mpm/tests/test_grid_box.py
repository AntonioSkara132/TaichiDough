"""CPU checks that a grid box reproduces the full-cube solver where the dough stays inside it."""
from dataclasses import replace
from types import SimpleNamespace
import json
import unittest

import numpy as np
import taichi as ti

from experiments.differentiable_mpm.checkpoint import estimate_memory
from experiments.differentiable_mpm.config import ObservationSettings
from experiments.differentiable_mpm.data import validate_initial_stencils
from experiments.differentiable_mpm.dataset_config import _common_settings
from experiments.differentiable_mpm.reference_adapter import reference_arguments
from experiments.differentiable_mpm.solver import Stepper
from experiments.differentiable_mpm.state import (
    DEFAULT_PARAMETERS, STATE_NAMES, InvalidStateError, ParticleState, SimulationConfig, ToolControl,
)
from experiments.differentiable_mpm.tests.test_coulomb_contact import plane_sdf
from experiments.differentiable_mpm.tests.test_solver import dot_state, terminal_seed


STEPS = 24
BOX = {"grid_box_lower": (9, 0, 10), "grid_box_cells": (14, 12, 13)}


def dough(n_axis=5):
    axis = np.linspace(0.0, 0.07, n_axis)
    offsets = np.stack(np.meshgrid(axis, axis[:3] * 0.4, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    return offsets + [0.45, 0.004, 0.46]


def base_config(n, physics_version="corrected-v1", **changes):
    return SimulationConfig(
        n_particles=n, grid=32, precision="f64", dt=2e-4, particle_mass=2e-4, particle_volume=1.4e-7,
        gravity=-9.81, plasticity="von-mises", von_mises_yield_stress_pa=400.0, tool_collision="sdf",
        tool_contact_model="coulomb-bounded-shear-adhesive-v3", tool_contact_padding=0.003,
        tool_friction_coefficient=0.6, tool_adhesion=5.0, floor_contact_model="coulomb-v1",
        floor_friction_coefficient=0.5, p2g_mode="serial", physics_version=physics_version, **changes)


def pushing_control():
    control = ToolControl.stationary()
    control.poses[0, :3] = [0.44, 0.03, 0.495]
    control.poses[1, :3] = [10.0, 10.0, 10.0]
    control.velocities[0] = [0.3, -0.05, 0.1, 0.0, 0.4, 0.0]
    return control


def parameters():
    return dict(DEFAULT_PARAMETERS, youngs_modulus=8000.0, poisson_ratio=0.4, viscosity=2.0,
                tool_friction_coefficient=0.6, tool_adhesion=5.0, floor_friction_coefficient=0.5,
                von_mises_yield_stress_pa=400.0)


def rollout(config, positions, seed=None):
    solver = Stepper(config, parameters(), capacity=STEPS + 1, sdf=plane_sdf())
    control = pushing_control()
    solver.load_state(0, ParticleState.initial(positions, np.float64))
    for t in range(STEPS):
        solver.advance(t, control)
    states = [solver.state(t) for t in range(STEPS + 1)]
    if seed is None:
        return states, solver.diagnostics(), None, None
    solver.clear_state_gradients()
    solver.clear_parameter_gradients()
    solver.load_adjoint(STEPS, seed)
    for t in reversed(range(STEPS)):
        solver.reverse_step(t, control)
    return states, solver.diagnostics(), solver.adjoint(0), solver.parameter_gradients()


class GridBoxConfigTests(unittest.TestCase):
    def test_defaults_record_and_round_trip(self):
        full = SimulationConfig(n_particles=1)
        self.assertEqual((full.box_lower, full.box_cells), ((0, 0, 0), (48, 48, 48)))
        self.assertFalse(set(BOX) & set(full.record()))
        boxed = SimulationConfig(n_particles=1, grid_box_lower=[9, 0, 10], grid_box_cells=[14, 12, 13])
        self.assertEqual((boxed.box_lower, boxed.box_cells), ((9, 0, 10), (14, 12, 13)))
        record = boxed.record()
        self.assertEqual(record["grid_box_cells"], [14, 12, 13])
        self.assertEqual(SimulationConfig(**json.loads(json.dumps(record))), boxed)

    def test_invalid_boxes_are_rejected(self):
        for box in ({"grid_box_lower": (0, 0, 0)}, {"grid_box_cells": (8, 8, 8)},
                    {"grid_box_lower": (0, 0), "grid_box_cells": (8, 8, 8)},
                    {"grid_box_lower": (0, 0, 0), "grid_box_cells": (8, 7, 8)},
                    {"grid_box_lower": (0, 1, 0), "grid_box_cells": (8, 8, 8)},
                    {"grid_box_lower": (0.0, 0, 0), "grid_box_cells": (8, 8, 8)},
                    {"grid_box_lower": (True, 0, 0), "grid_box_cells": (8, 8, 8)},
                    {"grid_box_lower": "000", "grid_box_cells": (8, 8, 8)},
                    {"grid_box_lower": (-1, 0, 0), "grid_box_cells": (8, 8, 8)},
                    {"grid_box_lower": (0, 0, 0), "grid_box_cells": (8, 8, SimulationConfig(n_particles=1).grid + 1)}):
            with self.subTest(box=box), self.assertRaises(ValueError):
                SimulationConfig(n_particles=1, **box)

    def test_box_is_episode_specific_and_rejected_by_reference(self):
        def episode(**simulation):
            return SimpleNamespace(simulation={"n_particles": 1, **simulation}, backend="cpu",
                                   tool_sdf_builder="legacy-voxel-v1", tool_sdf_qualification="strict-v2",
                                   tool_sdf_resolution=64, tool_mesh_scale=0.001,
                                   observation=ObservationSettings(), loss={}, strict_loss=False, optimizer={})
        self.assertEqual(_common_settings(episode()), _common_settings(episode(**BOX)))
        with self.assertRaisesRegex(ValueError, "full"):
            reference_arguments(SimulationConfig(n_particles=1, **BOX), DEFAULT_PARAMETERS)
        self.assertFalse(set(BOX) & set(vars(reference_arguments(SimulationConfig(n_particles=1), DEFAULT_PARAMETERS))))

    def test_memory_estimate_counts_box_nodes(self):
        full = estimate_memory(10, 10, grid=32)
        boxed = estimate_memory(10, 10, grid=32, grid_cells=BOX["grid_box_cells"])
        self.assertEqual(full["grid_primal_and_adjoint_estimate_bytes"], 33 ** 3 * 20 * 4)
        self.assertEqual(boxed["grid_primal_and_adjoint_estimate_bytes"], 15 * 13 * 14 * 20 * 4)

    def test_initial_stencil_check_uses_box(self):
        config = base_config(1, **BOX)
        validate_initial_stencils([[0.5, 0.05, 0.5]], config)
        for outside in ([0.25, 0.05, 0.5], [0.5, 0.36, 0.5], [0.5, 0.05, 0.74]):
            with self.subTest(position=outside), self.assertRaises(ValueError):
                validate_initial_stencils([outside], config)
            validate_initial_stencils([outside], base_config(1))


class GridBoxSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, debug=True, offline_cache=False)

    def test_box_matches_full_cube_forward_and_adjoint(self):
        positions = dough()
        for physics_version in ("corrected-v1", "legacy-v1"):
            with self.subTest(physics_version=physics_version):
                full_config = base_config(len(positions), physics_version)
                boxed_config = replace(full_config, **BOX)
                seed = terminal_seed(ParticleState.initial(positions, np.float64))
                full, full_counts, full_adjoint, full_gradients = rollout(full_config, positions, seed)
                boxed, boxed_counts, boxed_adjoint, boxed_gradients = rollout(boxed_config, positions, seed)
                self.assertGreater(full_counts["grid_tool0"], 0)
                self.assertGreater(full_counts["grid_floor"], 0)
                self.assertGreater(float(np.abs(full[-1].x - full[0].x).max()), 1e-4)
                self.assertEqual(full_counts, boxed_counts)
                for t, (a, b) in enumerate(zip(full, boxed)):
                    for name in STATE_NAMES:
                        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=f"{name} at {t}")
                for name in STATE_NAMES:
                    np.testing.assert_array_equal(getattr(full_adjoint, name), getattr(boxed_adjoint, name))
                self.assertEqual(full_gradients, boxed_gradients)
                self.assertNotEqual(dot_state(full_adjoint, full_adjoint), 0.0)

    def test_leaving_the_box_is_an_invalid_state(self):
        positions = dough()
        config = base_config(len(positions), grid_box_lower=(9, 0, 10), grid_box_cells=(14, 12, 13))
        solver = Stepper(config, parameters(), capacity=2, sdf=plane_sdf())
        shifted = positions.copy()
        # Base node 7 is below the allocated nodes 8.. of a box starting at 9.
        shifted[0, 0] = 8.3 / 32
        with self.assertRaisesRegex(InvalidStateError, "input stencil"):
            solver.load_state(0, ParticleState.initial(shifted, np.float64))
            solver.advance(0, pushing_control())


if __name__ == "__main__":
    unittest.main()
