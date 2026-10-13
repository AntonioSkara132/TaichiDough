"""CPU checks for grid-only prescribed tool velocities and their adjoints."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import taichi as ti

from experiments.differentiable_mpm.checkpoint import CheckpointedRollout
from experiments.differentiable_mpm.config import SCHEMA, load_config
from experiments.differentiable_mpm.reference_adapter import reference_arguments
from experiments.differentiable_mpm.solver import Stepper
from experiments.differentiable_mpm.state import (
    DEFAULT_PARAMETERS, PARAMETER_NAMES, SimulationConfig, ParticleState, ToolControl,
    validate_tool_parameters,
)
from experiments.differentiable_mpm.tests.test_coulomb_contact import plane_sdf
from experiments.differentiable_mpm.tests.test_solver import dot_state, fixture, terminal_seed
from experiments.differentiable_mpm.tests.test_trajectory import ParticleTargetLoss, Target
from experiments.differentiable_mpm.tests.test_tunable_contact import GradientProbe


MODE = "dirichlet-v1"
MODEL = "coulomb-bounded-adhesive-v2"
SEED = np.array([0.7, -0.4, 0.2])


def moving_control():
    control = ToolControl.stationary()
    control.poses[0, :3] = [.4, .4, .4]
    control.poses[1, :3] = [10., 10., 10.]
    control.velocities[0] = [.02, -.01, .03, .4, -.2, .5]
    return control


def copy_control(control):
    return ToolControl(control.poses.copy(), control.velocities.copy(), control.time)


class GridDirichletConfigTests(unittest.TestCase):
    def test_defaults_and_padding_resolution_round_trip(self):
        config = SimulationConfig(n_particles=1, tool_contact_padding=.004)
        self.assertEqual(config.tool_grid_contact_mode, "inherit")
        self.assertIsNone(config.tool_grid_contact_padding)
        self.assertEqual(config.effective_tool_grid_contact_padding, .004)
        for padding in (None, 0., .002, .008):
            changed = replace(config, tool_grid_contact_padding=padding)
            restored = SimulationConfig(**json.loads(json.dumps(asdict(changed))))
            self.assertEqual(restored, changed)
            self.assertEqual(restored.effective_tool_grid_contact_padding,
                             .004 if padding is None else padding)
            self.assertEqual(restored.tool_contact_padding, .004)

    def test_invalid_mode_padding_and_collision_are_rejected(self):
        for mode in ("sticky", None, True, 1, [], {}):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                SimulationConfig(n_particles=1, tool_collision="sdf", tool_grid_contact_mode=mode)
        for padding in (True, np.bool_(False), "0.004", [], -.001, float("nan"), float("inf")):
            with self.subTest(padding=padding), self.assertRaises(ValueError):
                SimulationConfig(n_particles=1, tool_grid_contact_padding=padding)
        with self.assertRaises(ValueError):
            SimulationConfig(n_particles=1, tool_grid_contact_mode=MODE)

    def test_loader_preserves_fixed_settings_and_particle_parameter_fits(self):
        paths = ("episode", "calibration", "initial_particles", "reconstruction_metadata",
                 "tool_geometry", "collision_manifest", "kinova_collision_mesh", "ur_collision_mesh")
        document = {
            "schema": SCHEMA, "name": "grid-dirichlet-test",
            "paths": {name: "/unused/" + name for name in paths},
            "simulation": {"n_particles": 1, "tool_collision": "sdf", "tool_contact_model": MODEL,
                           "tool_grid_contact_mode": MODE, "tool_grid_contact_padding": .006,
                           "tool_contact_padding": .002},
            "parameters": dict(DEFAULT_PARAMETERS, tool_adhesion=5.),
            "fit_parameters": ["tool_friction_coefficient", "tool_adhesion"],
            "training": {"start_frame": 1, "end_frame": 2},
            "validation": {"start_frame": 3, "end_frame": 4},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(document))
            config = load_config(path)
            simulation = config.as_dict()["simulation"]
            self.assertEqual(simulation["tool_grid_contact_mode"], MODE)
            self.assertEqual(simulation["tool_grid_contact_padding"], .006)
            self.assertEqual(simulation["tool_contact_padding"], .002)
            validate_tool_parameters(config.parameters, SimulationConfig(**simulation), config.fit_parameters)
            fingerprint = config.fingerprint
            config.simulation["tool_grid_contact_mode"] = "inherit"
            self.assertNotEqual(config.fingerprint, fingerprint)
        self.assertNotIn("tool_grid_contact_mode", PARAMETER_NAMES)
        self.assertNotIn("tool_grid_contact_padding", PARAMETER_NAMES)

    def test_reference_accepts_only_matching_inherited_grid_contact(self):
        config = SimulationConfig(n_particles=1, tool_collision="sdf", tool_contact_padding=.004)
        for padding in (None, .004):
            reference_arguments(replace(config, tool_grid_contact_padding=padding), DEFAULT_PARAMETERS)
        for options in ({"tool_grid_contact_mode": MODE}, {"tool_grid_contact_padding": .006}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                reference_arguments(replace(config, **options), DEFAULT_PARAMETERS)


class GridDirichletResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, debug=True, offline_cache=False)
        config = SimulationConfig(
            n_particles=1, grid=12, precision="f64", gravity=0., floor_y=-1.,
            tool_collision="sdf", tool_contact_model=MODEL, tool_contact_padding=.004,
            tool_grid_contact_mode=MODE, tool_grid_contact_padding=.02,
        )
        cls.params = dict(DEFAULT_PARAMETERS, tool_adhesion=5., tool_friction_coefficient=.6)
        cls.prescribed = Stepper(config, cls.params, capacity=2, sdf=plane_sdf(), control_gradients=True)
        cls.inherited = Stepper(replace(config, tool_grid_contact_mode="inherit"), cls.params,
                                capacity=2, sdf=plane_sdf())
        cls.narrow = Stepper(replace(config, tool_grid_contact_mode="inherit", tool_grid_contact_padding=None),
                             cls.params, capacity=2, sdf=plane_sdf())
        cls.fixed_controls = Stepper(replace(config, tool_grid_contact_padding=1. / 64.), cls.params,
                                     capacity=2, sdf=plane_sdf(), control_gradients=False)
        cls.probe = GradientProbe()

    def setUp(self):
        self.control = moving_control()
        for solver in (self.prescribed, self.inherited, self.narrow, self.fixed_controls):
            solver._set_control(self.control)
            solver.set_parameters(self.params)
            solver._reset_diagnostics()
        self.probe.pos[None] = [.407, .43, .38]
        self.probe.velocity[None] = [-.3, .4, -.2]
        self.probe.normal[None] = [1., 0., 0.]

    def response(self, solver, path="grid"):
        getattr(self.probe, path + "_response")(solver)
        return np.asarray(self.probe.output_velocity[None])

    def gradients(self):
        solver, probe = self.prescribed, self.probe
        solver.clear_parameter_gradients()
        for field in (solver.tool_center, solver.tool_rotation, solver.tool_velocity, solver.tool_omega):
            field.grad.fill(0)
        probe.pos.grad[None] = [0., 0., 0.]
        probe.velocity.grad[None] = [0., 0., 0.]
        probe.evaluate(solver, "grid")
        probe.loss.grad[None] = 1.
        probe.evaluate.grad(solver, "grid")
        return solver._control_gradient(self.control)

    def test_matches_translation_and_rotation_for_all_relative_directions(self):
        pos = np.asarray(self.probe.pos[None])
        collider = self.control.velocities[0, :3] + np.cross(
            self.control.velocities[0, 3:], pos - self.control.poses[0, :3])
        for relative in ((-1., .3, .2), (1., -.3, .2), (0., .4, 0.), (0., 0., 0.)):
            self.probe.velocity[None] = collider + relative
            np.testing.assert_allclose(self.response(self.prescribed), collider, rtol=0, atol=1e-14)
        self.control.poses[0, 3:] = [0., 0., np.sin(.3), np.cos(.3)]
        self.prescribed._set_control(self.control)
        self.probe.pos[None] = [.4, .4, .4]
        np.testing.assert_allclose(self.response(self.prescribed), self.control.velocities[0, :3], atol=1e-14)

    def test_release_outside_band_and_invalid_sdf_leave_velocity_unchanged(self):
        velocity = np.asarray(self.probe.velocity[None])
        for position in ((.425, .43, .38), (3., 3., 3.)):
            self.probe.pos[None] = position
            np.testing.assert_array_equal(self.response(self.prescribed), velocity)

    def test_exact_grid_band_boundary_is_excluded(self):
        solver = self.fixed_controls
        band = solver.config.effective_tool_grid_contact_padding
        self.probe.pos[None] = self.control.poses[0, :3]
        velocity = np.asarray(self.probe.velocity[None])
        # Constant SDF at a sample point makes the strict comparison exact.
        try:
            for distance in (band, np.nextafter(band, 0.), np.nextafter(band, np.inf)):
                solver.sdf.fill(distance)
                expected = self.control.velocities[0, :3] if distance < band else velocity
                np.testing.assert_array_equal(self.response(solver), expected)
        finally:
            solver.sdf.from_numpy(plane_sdf().distances)

    def test_fixed_control_mode_supports_forward_and_input_parameter_adjoints(self):
        solver, probe = self.fixed_controls, self.probe
        self.assertFalse(solver.control_gradients)
        velocity = np.asarray(probe.velocity[None])
        omega = self.control.velocities[0, 3:]
        for position, active in (((.407, .43, .38), True), ((.425, .43, .38), False)):
            probe.pos[None] = position
            expected = (self.control.velocities[0, :3]
                        + np.cross(omega, np.asarray(position) - self.control.poses[0, :3])) if active else velocity
            np.testing.assert_allclose(self.response(solver), expected, atol=1e-14)
            solver.clear_parameter_gradients()
            probe.pos.grad[None] = [0., 0., 0.]
            probe.velocity.grad[None] = [0., 0., 0.]
            probe.evaluate(solver, "grid")
            probe.loss.grad[None] = 1.
            probe.evaluate.grad(solver, "grid")
            np.testing.assert_array_equal(probe.velocity.grad[None], np.zeros(3) if active else SEED)
            np.testing.assert_allclose(probe.pos.grad[None], np.cross(SEED, omega) if active else np.zeros(3),
                                       atol=1e-14)
            np.testing.assert_array_equal(list(solver.parameter_gradients().values()), np.zeros(len(PARAMETER_NAMES)))

    def test_closest_tool_and_equal_distance_tie_keep_existing_selection(self):
        self.control.poses[1] = self.control.poses[0]
        self.control.velocities[1] = [-.1, .2, .3, 0., 0., 0.]
        self.prescribed._set_control(self.control)
        pos = np.asarray(self.probe.pos[None])
        first = self.control.velocities[0, :3] + np.cross(
            self.control.velocities[0, 3:], pos - self.control.poses[0, :3])
        np.testing.assert_allclose(self.response(self.prescribed), first, atol=1e-14)
        self.control.poses[1, 0] += .003
        self.prescribed._set_control(self.control)
        np.testing.assert_allclose(self.response(self.prescribed), self.control.velocities[1, :3], atol=1e-14)

    def test_grid_band_is_independent_and_particle_response_is_unchanged(self):
        self.probe.pos[None] = [.41, .4, .4]
        velocity = np.asarray(self.probe.velocity[None])
        np.testing.assert_array_equal(self.response(self.narrow), velocity)
        self.assertGreater(np.linalg.norm(self.response(self.inherited) - velocity), .1)
        for solver in (self.prescribed, self.inherited):
            np.testing.assert_array_equal(self.response(solver, "particle"), velocity)
            np.testing.assert_array_equal(self.probe.output_pos[None], self.probe.pos[None])
        self.probe.pos[None] = [.402, .4, .4]
        expected_velocity = self.response(self.inherited, "particle")
        expected_position = np.asarray(self.probe.output_pos[None])
        np.testing.assert_allclose(expected_position, [.4041, .4, .4], atol=1e-14)
        np.testing.assert_array_equal(self.response(self.prescribed, "particle"), expected_velocity)
        np.testing.assert_array_equal(self.probe.output_pos[None], expected_position)

    def test_default_grid_response_still_uses_selected_particle_law(self):
        self.probe.pos[None] = [.402, .4, .4]
        position = np.asarray(self.probe.pos[None])
        self.probe.collider[None] = self.control.velocities[0, :3] + np.cross(
            self.control.velocities[0, 3:], position - self.control.poses[0, :3])
        expected = self.response(self.narrow, "velocity")
        np.testing.assert_allclose(self.response(self.narrow), expected, atol=1e-14)

    def test_active_response_has_exact_control_and_zero_input_parameter_adjoints(self):
        gradient = self.gradients()
        r = np.asarray(self.probe.pos[None]) - self.control.poses[0, :3]
        omega = self.control.velocities[0, 3:]
        np.testing.assert_allclose(gradient.velocities[0, :3], SEED, atol=1e-14)
        np.testing.assert_allclose(gradient.velocities[0, 3:], np.cross(r, SEED), atol=1e-14)
        np.testing.assert_allclose(gradient.poses[0, :3], np.cross(omega, SEED), atol=1e-14)
        np.testing.assert_allclose(self.probe.pos.grad[None], np.cross(SEED, omega), atol=1e-14)
        np.testing.assert_array_equal(self.probe.velocity.grad[None], np.zeros(3))
        np.testing.assert_array_equal(list(self.prescribed.parameter_gradients().values()), np.zeros(len(PARAMETER_NAMES)))
        np.testing.assert_array_equal(gradient.velocities[1], np.zeros(6))
        np.testing.assert_array_equal(gradient.poses[1], np.zeros(7))
        np.testing.assert_array_equal(gradient.rotation, np.zeros((2, 3)))

        for field, columns, actual in (("poses", range(3), gradient.poses),
                                       ("velocities", range(6), gradient.velocities)):
            for column in columns:
                samples = []
                for sign in (1., -1.):
                    changed = copy_control(self.control)
                    getattr(changed, field)[0, column] += sign * 1e-6
                    self.prescribed._set_control(changed)
                    self.probe.evaluate(self.prescribed, "grid")
                    samples.append(self.probe.loss[None])
                self.assertAlmostEqual(actual[0, column], (samples[0] - samples[1]) / 2e-6,
                                       delta=2e-10, msg=f"{field}[0,{column}]")
        self.prescribed._set_control(self.control)

    def test_inactive_response_passes_input_gradient_and_has_zero_control_gradient(self):
        self.probe.pos[None] = [.425, .43, .38]
        gradient = self.gradients()
        np.testing.assert_array_equal(self.probe.velocity.grad[None], SEED)
        np.testing.assert_array_equal(self.probe.pos.grad[None], np.zeros(3))
        np.testing.assert_array_equal(gradient.poses, np.zeros((2, 7)))
        np.testing.assert_array_equal(gradient.velocities, np.zeros((2, 6)))
        np.testing.assert_array_equal(list(self.prescribed.parameter_gradients().values()), np.zeros(len(PARAMETER_NAMES)))

    def test_g2p_receives_prescribed_rigid_velocity_and_affine_gradient(self):
        solver = self.prescribed
        self.control.poses[0, 0] = .52
        solver._set_control(self.control)
        state = ParticleState.initial([[.415, .41, .41]], np.float64)
        solver.load_state(0, state)
        solver.grid_m.fill(1.)
        solver.grid_u.fill(0.)
        solver._grid_operations()
        solver._g2p(0)
        omega = self.control.velocities[0, 3:]
        expected = self.control.velocities[0, :3] + np.cross(omega, state.x[0] - self.control.poses[0, :3])
        wx, wy, wz = omega
        affine = np.array([[0., -wz, wy], [wz, 0., -wx], [-wy, wx, 0.]])
        np.testing.assert_allclose(solver.adv_v[0], expected, rtol=0, atol=1e-13)
        np.testing.assert_allclose(solver.adv_x[0], state.x[0] + solver.dt * expected, rtol=0, atol=1e-13)
        np.testing.assert_allclose(solver.new_C[0], affine, rtol=0, atol=1e-12)


class GridDirichletRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, debug=True, offline_cache=False)
        config, cls.initial, cls.params, cls.control, sdf = fixture(sdf=True, tool_contact_model=MODEL)
        config = replace(config, tool_grid_contact_mode=MODE, tool_grid_contact_padding=.04, p2g_mode="serial")
        cls.solver = Stepper(config, cls.params, capacity=3, sdf=sdf, control_gradients=True)

    def value(self, state, control, seed):
        self.solver.set_parameters(self.params)
        self.solver.load_state(0, state)
        self.solver.advance(0, control)
        return dot_state(self.solver.state(1), seed)

    def test_one_step_control_and_initial_state_gradients_match_finite_differences(self):
        solver, state, control = self.solver, self.initial, self.control
        seed = terminal_seed(state)
        self.value(state, control, seed)
        counts = solver.diagnostics()
        self.assertGreater(counts["grid_tool0"] + counts["grid_tool1"], 0)
        solver.clear_state_gradients()
        solver.clear_parameter_gradients()
        solver.load_adjoint(1, seed)
        gradient = solver.reverse_step(0, control)
        initial_gradient = solver.adjoint(0)
        rng = np.random.default_rng(7205)
        position = rng.normal(size=(2, 3)) * .01
        velocity = rng.normal(size=(2, 6)) * np.array([.02] * 3 + [.2] * 3)
        initial_position = rng.normal(size=state.x.shape) * .01
        predicted = float(np.sum(gradient.poses[:, :3] * position) + np.sum(gradient.velocities * velocity)
                          + np.sum(initial_gradient.x * initial_position))
        candidates = []
        for h in (1e-5, 3e-6):
            samples = []
            for sign in (1., -1.):
                changed = copy_control(control)
                changed.poses[:, :3] += sign * h * position
                changed.velocities += sign * h * velocity
                perturbed_state = state.copy()
                perturbed_state.x += sign * h * initial_position
                samples.append(self.value(perturbed_state, changed, seed))
                self.assertEqual(solver.diagnostics(), counts, "Finite difference changed active diagnostic counts")
            candidates.append((samples[0] - samples[1]) / (2 * h))
        self.assertTrue(np.isfinite(predicted))
        self.assertGreater(np.abs(gradient.velocities).max(), 0.)
        tolerance = 1e-7 + 5e-4 * max(abs(predicted), max(abs(v) for v in candidates))
        self.assertLessEqual(min(abs(predicted - fd) for fd in candidates), tolerance,
                             f"AD={predicted}, FD={candidates}")

    def test_two_step_checkpoint_segments_preserve_state_parameter_and_control_gradients(self):
        controls = [copy_control(self.control) for _ in range(2)]
        controls[1].poses[:, :3] += self.solver.dt * controls[1].velocities[:, :3]
        targets = [Target(i, i + 1, self.initial.x + .001 * (i + 1)) for i in range(2)]

        def evaluate(segment_length):
            rollout = CheckpointedRollout(
                self.solver, self.initial, controls, targets, ParticleTargetLoss(), segment_length,
                replay_rtol=0., replay_atol={name: 0. for name in self.initial.arrays()},
            )
            return rollout.value_and_gradient(self.params, compute_control_grad=True)

        full, split = evaluate(2), evaluate(1)
        self.assertAlmostEqual(full.value, split.value, places=12)
        np.testing.assert_allclose(list(full.gradient.values()), list(split.gradient.values()), rtol=1e-10, atol=1e-10)
        for name in self.initial.arrays():
            np.testing.assert_allclose(getattr(full.initial_gradient, name), getattr(split.initial_gradient, name),
                                       rtol=1e-10, atol=1e-10)
        for name in ("poses", "velocities", "rotation"):
            self.assertTrue(np.isfinite(full.control_gradient[name]).all())
            np.testing.assert_allclose(full.control_gradient[name], split.control_gradient[name], rtol=1e-10, atol=1e-10)
        self.assertGreater(np.abs(full.control_gradient["velocities"]).max(), 0.)
        self.assertEqual(full.diagnostics["observation_gradient_injections"], [1, 1])
        self.assertEqual(split.diagnostics["observation_gradient_injections"], [1, 1])


if __name__ == "__main__":
    unittest.main()
