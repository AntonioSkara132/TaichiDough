"""Analytic and integration checks for unilateral Coulomb tool contact."""
import unittest

import numpy as np
import taichi as ti

from experiments.differentiable_mpm.solver import Stepper
from experiments.differentiable_mpm.state import DEFAULT_PARAMETERS, SDFData, SimulationConfig, ToolControl


@ti.data_oriented
class ContactProbe:
    def __init__(self):
        self.pos = ti.Vector.field(3, ti.f64, shape=())
        self.velocity = ti.Vector.field(3, ti.f64, shape=())
        self.collider = ti.Vector.field(3, ti.f64, shape=())
        self.normal = ti.Vector.field(3, ti.f64, shape=())
        self.output_pos = ti.Vector.field(3, ti.f64, shape=())
        self.output_velocity = ti.Vector.field(3, ti.f64, shape=())

    @ti.kernel
    def velocity_response(self, solver: ti.template()):
        self.output_velocity[None] = solver._tool_velocity_response(
            self.velocity[None], self.collider[None], self.normal[None])

    @ti.kernel
    def floor_response(self, solver: ti.template()):
        self.output_velocity[None] = solver._floor_coulomb_response(self.velocity[None])

    @ti.kernel
    def grid_response(self, solver: ti.template()):
        self.output_velocity[None] = solver._tool_grid_response(self.pos[None], self.velocity[None])

    @ti.kernel
    def particle_response(self, solver: ti.template()):
        x, v = solver._tool_particle_response(self.pos[None], self.velocity[None])
        self.output_pos[None] = x
        self.output_velocity[None] = v


def plane_sdf():
    resolution = 9
    axis = np.linspace(-0.2, 0.2, resolution)
    coordinates = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    distance = coordinates[..., 0]
    normal = np.zeros(coordinates.shape, dtype=np.float64)
    normal[..., 0] = 1.0
    return SDFData(
        distances=np.stack([distance, distance]),
        gradients=np.stack([normal, normal]),
        minimums=np.full((2, 3), -0.2),
        spacings=np.full((2, 3), 0.4 / (resolution - 1)),
    )


class CoulombContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1,
                debug=True, offline_cache=False)

    def solver(self, coefficient=0.5, sdf=False):
        config = SimulationConfig(
            n_particles=1, grid=12, precision="f64", gravity=0.0, floor_y=-1.0,
            tool_collision="sdf" if sdf else "none", tool_contact_padding=0.004,
            tool_contact_model="coulomb-v1", tool_friction_coefficient=coefficient,
            tool_contact_absorption=0.0, tool_stickiness=0.0,
        )
        legacy = {name: value for name, value in DEFAULT_PARAMETERS.items()
                  if name not in {"tool_friction_coefficient", "tool_stickiness"}}
        return Stepper(config, legacy, capacity=2, sdf=plane_sdf() if sdf else None)

    def response(self, coefficient, velocity, collider=(0, 0, 0), normal=(1, 0, 0)):
        solver = self.solver(coefficient)
        probe = ContactProbe()
        probe.velocity[None] = velocity
        probe.collider[None] = collider
        probe.normal[None] = normal
        probe.velocity_response(solver)
        return np.asarray(probe.output_velocity[None], dtype=float)

    def test_analytic_sliding_static_and_zero_friction(self):
        np.testing.assert_array_equal(self.response(0.0, (-2, 3, 0)), [0, 3, 0])
        np.testing.assert_allclose(self.response(0.5, (-2, 3, 0)), [0, 2, 0], rtol=0, atol=1e-14)
        np.testing.assert_array_equal(self.response(2.0, (-2, 3, 0)), [0, 0, 0])

    def test_kinetic_branch_is_continuous_away_from_switches(self):
        h = 1e-6
        lower = self.response(0.5, (-2, 3 - h, 0))[1]
        center = self.response(0.5, (-2, 3, 0))[1]
        upper = self.response(0.5, (-2, 3 + h, 0))[1]
        self.assertAlmostEqual((upper - lower) / (2 * h), 1.0, places=9)
        self.assertAlmostEqual(center, 2.0, places=13)

    def test_separating_velocity_is_unchanged(self):
        velocity = np.array([1.25, -3.0, 0.7])
        np.testing.assert_array_equal(self.response(10.0, velocity), velocity)
        # A tool withdrawing opposite its outward normal must not pull stationary dough.
        np.testing.assert_array_equal(self.response(10.0, (0, 0.4, 0), collider=(-2, 0, 0)), [0, 0.4, 0])

    def test_pressing_sliding_tool_drags_without_matching_it(self):
        # Relative velocity is [-1,-2,0]. mu*delta_n=0.5 removes 0.5 m/s of slip.
        actual = self.response(0.5, (-1, 0, 0), collider=(0, 2, 0))
        np.testing.assert_allclose(actual, [0, 0.5, 0], rtol=0, atol=1e-14)

    def test_grid_and_projected_particle_paths_use_unilateral_response(self):
        solver = self.solver(0.5, sdf=True)
        control = ToolControl.stationary()
        control.poses[0, :3] = [0.4, 0.4, 0.4]
        control.poses[1, :3] = [10, 10, 10]
        control.velocities[0] = [0.0, 0.3, 0.0, 0.0, 0.0, 2.0]
        solver._set_control(control)
        probe = ContactProbe()
        probe.pos[None] = [0.401, 0.4, 0.4]

        separating = np.array([1.0, -0.6, 0.2])
        probe.velocity[None] = separating
        probe.grid_response(solver)
        np.testing.assert_array_equal(np.asarray(probe.output_velocity[None]), separating)
        probe.particle_response(solver)
        np.testing.assert_array_equal(np.asarray(probe.output_velocity[None]), separating)
        np.testing.assert_allclose(np.asarray(probe.output_pos[None]), [0.4041, 0.4, 0.4], atol=1e-12)

        inward = np.array([-1.0, -0.2, 0.0])
        probe.velocity[None] = inward
        probe.particle_response(solver)
        projected = np.asarray(probe.output_pos[None], dtype=float)
        collider = np.array([0.0, 0.3 + 2.0 * (projected[0] - 0.4), 0.0])
        relative = inward - collider
        tangential = relative - np.array([1.0, 0.0, 0.0]) * relative[0]
        scale = max(0.0, 1.0 - 0.5 * (-relative[0]) / np.linalg.norm(tangential))
        expected = collider + tangential * scale
        np.testing.assert_allclose(np.asarray(probe.output_velocity[None]), expected, rtol=2e-12, atol=2e-12)

    def test_configuration_rejects_adhesive_multipliers_and_invalid_mu(self):
        for value in (-0.1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SimulationConfig(n_particles=1, tool_contact_model="coulomb-v1",
                                 tool_friction_coefficient=value)
        for options in ({"tool_contact_absorption": 0.1}, {"tool_stickiness": 0.1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SimulationConfig(n_particles=1, tool_contact_model="coulomb-v1", **options)
        with self.assertRaises(ValueError):
            SimulationConfig(n_particles=1, tool_contact_model="unknown")


class BoundedAdhesiveContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1,
                debug=True, offline_cache=False)

    def solver(self, model="coulomb-bounded-adhesive-v2", floor="retention-v1", **parameters):
        config = SimulationConfig(
            n_particles=1, grid=12, precision="f64", gravity=0.0, floor_y=-1.0, dt=2e-4,
            tool_contact_model=model, floor_contact_model=floor, tool_contact_padding=0.004)
        return Stepper(config, dict(DEFAULT_PARAMETERS, **parameters), capacity=2)

    def response(self, solver, velocity, collider=(0, 0, 0), normal=(1, 0, 0)):
        probe = ContactProbe()
        probe.velocity[None] = velocity
        probe.collider[None] = collider
        probe.normal[None] = normal
        probe.velocity_response(solver)
        return np.asarray(probe.output_velocity[None], dtype=float)

    def test_approach_matches_coulomb_v1(self):
        for velocity, collider in (((-2, 3, 0), (0, 0, 0)), ((-1, 0, .4), (0, 2, 0)), ((-3, 2, .4), (-2, .3, .1))):
            expected = self.response(self.solver("coulomb-v1", tool_friction_coefficient=.5), velocity, collider)
            actual = self.response(self.solver(tool_friction_coefficient=.5, tool_adhesion=300.0), velocity, collider)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_zero_adhesion_leaves_separation_unchanged(self):
        velocity = np.array([1.25, -3.0, 0.7])
        np.testing.assert_allclose(self.response(self.solver(tool_friction_coefficient=10.0), velocity),
                                   velocity, rtol=0, atol=1e-12)

    def test_adhesion_is_normal_and_bounded(self):
        limit = 300.0 * 2e-4
        solver = self.solver(tool_friction_coefficient=0.0, tool_adhesion=300.0)
        # Slow separation is almost stopped; tangential velocity is untouched without friction.
        slow = self.response(solver, (limit * 1e-3, 0.7, 0))
        self.assertLess(abs(slow[0]), limit * 1e-6)
        self.assertEqual(slow[1], 0.7)
        # Fast separation loses at most limit, so the dough pulls off.
        for speed in (10 * limit, 1000 * limit):
            fast = self.response(solver, (speed, 0, 0))
            self.assertGreater(fast[0], speed - limit)
            self.assertLess(fast[0], speed)
        # Rotating the normal rotates the response: adhesion acts along the normal only.
        normal = np.array([0.6, 0.8, 0.0])
        relative = 5 * limit * normal + np.array([0, 0, 0.3])
        rotated = self.response(solver, relative, normal=normal)
        self.assertAlmostEqual(rotated[2], 0.3, places=14)
        np.testing.assert_allclose(np.cross(rotated - np.array([0, 0, 0.3]), normal), 0, atol=1e-14)

    def test_adhesive_pull_feeds_friction_bound(self):
        limit = 300.0 * 2e-4
        solver = self.solver(tool_friction_coefficient=0.5, tool_adhesion=300.0)
        speed = 10 * limit
        pulled = speed - self.response(self.solver(tool_friction_coefficient=0.0, tool_adhesion=300.0), (speed, 0, 0))[0]
        actual = self.response(solver, (speed, 1.0, 0))
        self.assertAlmostEqual(actual[1], 1.0 - 0.5 * pulled, places=12)

    def test_co_moving_tool_returns_its_velocity(self):
        collider = np.array([0.3, -0.1, 0.05])
        np.testing.assert_allclose(self.response(self.solver(tool_adhesion=300.0), collider, collider), collider,
                                   rtol=0, atol=1e-15)

    def test_floor_coulomb_slides_and_sticks(self):
        solver = self.solver("coulomb-v1", floor="coulomb-v1", floor_friction_coefficient=0.5)
        probe = ContactProbe()
        for velocity, expected in (((1.0, -0.2, 0.0), (0.9, 0, 0)), ((0.05, -0.2, 0.0), (0, 0, 0)),
                                   ((0.3, 0.2, -0.4), (0.3, 0.2, -0.4)), ((0.0, -0.2, 0.0), (0, 0, 0))):
            probe.velocity[None] = velocity
            probe.floor_response(solver)
            np.testing.assert_allclose(np.asarray(probe.output_velocity[None]), expected, rtol=0, atol=1e-7)

    def test_configuration_rules(self):
        with self.assertRaises(ValueError):
            SimulationConfig(n_particles=1, tool_contact_model="coulomb-adhesive-v1", tool_adhesion=1.0)
        for options in ({"tool_stickiness": 0.1}, {"tool_contact_absorption": 0.1}, {"tool_adhesion": -1.0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SimulationConfig(n_particles=1, tool_contact_model="coulomb-bounded-adhesive-v2", **options)
        for options in ({"floor_absorption": 0.1}, {"floor_stickiness": 0.1}, {"floor_friction_coefficient": -0.1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SimulationConfig(n_particles=1, floor_contact_model="coulomb-v1", **options)
        with self.assertRaises(ValueError):
            SimulationConfig(n_particles=1, floor_contact_model="unknown")


if __name__ == "__main__":
    unittest.main()
