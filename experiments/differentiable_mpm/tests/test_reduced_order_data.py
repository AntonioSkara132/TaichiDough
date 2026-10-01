"""Archive validation and exact Taichi stepping for reduced-order training data."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import taichi as ti

from experiments.differentiable_mpm.reduced_order_data import (
    CONTROL_DIM, Trajectory, collect_trajectory, encode_control, load_trajectory,
    pack_state, save_trajectory, scene_identity, unpack_state,
)
from experiments.differentiable_mpm.solver import Stepper
from experiments.differentiable_mpm.state import (
    DEFAULT_PARAMETERS, PARAMETER_NAMES, STATE_NAMES, InvalidStateError,
    ParticleState, SimulationConfig, ToolControl,
)
from experiments.differentiable_mpm.tests.test_solver import fixture


def example_trajectory():
    state = ParticleState.initial([[.4, .4, .4], [.45, .44, .42]], np.float64)
    next_state = state.copy()
    next_state.x[:, 0] += .001
    control = encode_control(ToolControl.stationary())
    return Trajectory(np.stack([pack_state(state), pack_state(next_state)]),
                      control[None], np.array([DEFAULT_PARAMETERS[name] for name in PARAMETER_NAMES]),
                      .0002, "one", "fixed-scene")


class StateEncodingTests(unittest.TestCase):
    def test_all_fields_pack_round_trip_and_order(self):
        state = ParticleState.initial([[.4, .4, .4], [.5, .4, .4]], np.float64)
        state.v[:] = np.arange(6).reshape(2, 3)
        state.C[:] = np.arange(18).reshape(2, 3, 3)
        state.F[:] = 2 * np.eye(3)
        state.Jp[:] = [1.1, .9]
        packed = pack_state(state)
        self.assertEqual(packed.shape, (50,))
        self.assertEqual(packed.dtype, np.float64)
        np.testing.assert_array_equal(packed[:6], state.x.reshape(-1))
        np.testing.assert_array_equal(packed[6:12], state.v.reshape(-1))
        np.testing.assert_array_equal(packed[12:30], state.C.reshape(-1))
        np.testing.assert_array_equal(packed[30:48], state.F.reshape(-1))
        np.testing.assert_array_equal(packed[48:], state.Jp)
        restored = unpack_state(packed, n_particles=2)
        for name in STATE_NAMES:
            np.testing.assert_array_equal(getattr(restored, name), getattr(state, name))
        restored.F[:] = 0
        self.assertFalse(np.all(state.F == 0))

    def test_invalid_packed_arrays(self):
        for values in (np.empty(0), np.zeros(24), np.zeros((1, 25)),
                       np.full(25, np.nan), np.zeros(25, dtype=complex)):
            with self.subTest(values=values.shape), self.assertRaises(ValueError):
                unpack_state(values)
        with self.assertRaises(ValueError):
            unpack_state(np.zeros(25), n_particles=2)
        state = ParticleState.initial([[.4, .4, .4]])
        state.Jp[0] = np.nan
        with self.assertRaises(InvalidStateError):
            pack_state(state)

    def test_quaternion_sign_invariance_and_feature_order(self):
        control = ToolControl.stationary()
        control.poses[0, :3] = [.2, .3, .4]
        control.poses[0, 3:] = [0, np.sin(.35), 0, np.cos(.35)]
        control.velocities[:] = np.arange(12).reshape(2, 6)
        values = encode_control(control)
        flipped = ToolControl(control.poses.copy(), control.velocities.copy())
        flipped.poses[:, 3:] *= -1
        np.testing.assert_array_equal(values, encode_control(flipped))
        np.testing.assert_array_equal(values[:3], control.poses[0, :3])
        np.testing.assert_array_equal(values[12:18], control.velocities[0])
        rotation = values[3:12].reshape(3, 3)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
        self.assertEqual(values.shape, (CONTROL_DIM,))

    def test_bad_tool_quaternion_or_nonfinite_control(self):
        control = ToolControl.stationary()
        control.poses[0, 6] = 2
        with self.assertRaises(ValueError):
            encode_control(control)
        control = ToolControl.stationary()
        control.velocities[0, 1] = np.inf
        with self.assertRaises(ValueError):
            encode_control(control)


class TrajectoryArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary_root = os.environ.get("TMPDIR")
        self.directory = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "trajectory.npz"

    def test_round_trip_without_pickle(self):
        trajectory = example_trajectory()
        trajectory.metadata = {"source": "test", "nested": {"count": 3}}
        save_trajectory(self.path, trajectory)
        restored = load_trajectory(self.path, expected_identity="fixed-scene")
        for name in ("states", "controls", "parameters"):
            np.testing.assert_array_equal(getattr(restored, name), getattr(trajectory, name))
        self.assertEqual(restored.metadata, trajectory.metadata)
        self.assertEqual(restored.n_particles, 2)
        self.assertEqual(restored.n_steps, 1)
        self.assertEqual(restored.dt, trajectory.dt)
        self.assertEqual(restored.trajectory_id, trajectory.trajectory_id)
        with np.load(self.path, allow_pickle=False) as archive:
            self.assertEqual(archive["record"].dtype.kind, "U")
        with self.assertRaises(ValueError):
            Trajectory.load(self.path, expected_identity="another-scene")

    def test_invalid_dimensions_and_nonfinite_fields(self):
        trajectory = example_trajectory()
        changes = ({"states": trajectory.states[:, :-1]},
                   {"states": trajectory.states[:1]},
                   {"states": np.full_like(trajectory.states, np.nan)},
                   {"controls": np.zeros((0, 36))},
                   {"controls": trajectory.controls[:, :-1]},
                   {"controls": np.full_like(trajectory.controls, np.inf)},
                   {"parameters": trajectory.parameters[:-1]},
                   {"parameters": np.full(9, np.nan)},
                   {"dt": 0}, {"dt": -1}, {"dt": np.nan}, {"dt": True},
                   {"trajectory_id": ""}, {"identity": ""},
                   {"metadata": {"bad": np.nan}})
        for change in changes:
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                replace(trajectory, **change).validate()

    def test_invalid_rotation_and_parameters(self):
        trajectory = example_trajectory()
        controls = trajectory.controls.copy()
        controls[0, 3:12] = 0
        with self.assertRaises(ValueError):
            replace(trajectory, controls=controls).validate()
        parameters = trajectory.parameters.copy()
        parameters[0] = -1
        with self.assertRaises(ValueError):
            replace(trajectory, parameters=parameters).validate()

    def test_unknown_version_and_missing_archive_field(self):
        trajectory = example_trajectory()
        trajectory.save(self.path)
        with np.load(self.path, allow_pickle=False) as archive:
            fields = {name: archive[name].copy() for name in archive.files}
        record = json.loads(fields["record"].item())
        record["version"] = 100
        fields["record"] = np.asarray(json.dumps(record))
        np.savez(self.path, **fields)
        with self.assertRaises(ValueError):
            load_trajectory(self.path)
        fields.pop("controls")
        np.savez(self.path, **fields)
        with self.assertRaises(ValueError):
            load_trajectory(self.path)

    def test_pickle_arrays_are_not_loaded(self):
        np.savez(self.path, states=np.asarray([{"code": "not-executed"}], dtype=object),
                 controls=np.zeros((1, 36)), parameters=np.zeros(9), record=np.asarray("{}"))
        with self.assertRaises(ValueError):
            load_trajectory(self.path)

    def test_scene_identity_changes_with_settings_order_and_geometry(self):
        config, state, _, _, sdf = fixture(sdf=True)
        identity = scene_identity(config, state.x, sdf)
        self.assertEqual(identity, scene_identity(config, state.x.copy(), sdf))
        self.assertNotEqual(identity, scene_identity(config, state.x[::-1], sdf))
        self.assertNotEqual(identity, scene_identity(replace(config, dt=config.dt * 2), state.x, sdf))
        changed_sdf = replace(sdf, distances=sdf.distances + .001)
        self.assertNotEqual(identity, scene_identity(config, state.x, changed_sdf))
        with self.assertRaises(ValueError):
            scene_identity(config, state.x[:-1], sdf)
        with self.assertRaises(ValueError):
            scene_identity(config, state.x)
        with self.assertRaises(ValueError):
            scene_identity(replace(config, tool_collision="none"), state.x, sdf)


class CollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1,
                debug=True, offline_cache=False)

    def test_recycled_collection_matches_direct_contact_and_plastic_steps(self):
        config, state, parameters, control, sdf = fixture(plastic=True, sdf=True)
        controls = []
        for step in range(4):
            pose = control.poses.copy()
            pose[:, :3] += control.velocities[:, :3] * step * config.dt
            controls.append(ToolControl(pose, control.velocities.copy(), step * config.dt))
        trajectory = collect_trajectory(config, state, parameters, controls, sdf,
                                        trajectory_id="contact-plastic")
        direct = Stepper(config, parameters, capacity=5, sdf=sdf)
        direct.load_state(0, state)
        for step, step_control in enumerate(controls):
            direct.advance(step, step_control)
        for step in range(5):
            np.testing.assert_array_equal(trajectory.states[step], pack_state(direct.state(step)))
        np.testing.assert_array_equal(trajectory.controls, np.stack([encode_control(c) for c in controls]))
        np.testing.assert_array_equal(trajectory.parameters,
                                      [parameters[name] for name in PARAMETER_NAMES])
        self.assertEqual(trajectory.identity, scene_identity(config, state.x, sdf))
        self.assertEqual(trajectory.metadata["initial_time_s"], 0)
        self.assertGreater(trajectory.metadata["collection_seconds"], 0)
        self.assertIn("first-use kernel compilation", trajectory.metadata["collection_timing"])
        self.assertFalse(np.array_equal(trajectory.states[0], trajectory.states[-1]))
        self.assertFalse(np.array_equal(state.Jp, unpack_state(trajectory.states[1]).Jp))

    def test_shared_reference_identifies_perturbed_initial_state(self):
        config, state, parameters, _, _ = fixture()
        perturbed = state.copy()
        perturbed.x[:, 0] += .001
        controls = [ToolControl.stationary(0)]
        first = collect_trajectory(config, state, parameters, controls, trajectory_id="first")
        second = collect_trajectory(config, perturbed, parameters, controls,
                                    trajectory_id="second", reference_x=state.x)
        self.assertEqual(first.identity, second.identity)
        third = collect_trajectory(config, perturbed, parameters, controls, trajectory_id="third")
        self.assertNotEqual(first.identity, third.identity)
        tampered = dict(second.metadata, reference_positions_sha256="wrong")
        with self.assertRaises(ValueError):
            replace(second, metadata=tampered).validate()

    def test_collection_rejects_empty_or_misaligned_controls(self):
        config, state, parameters, _, _ = fixture()
        with self.assertRaises(ValueError):
            collect_trajectory(config, state, parameters, [])
        with self.assertRaises(ValueError):
            collect_trajectory(config, state, parameters,
                               [ToolControl.stationary(0), ToolControl.stationary(0)])
        with self.assertRaises(ValueError):
            collect_trajectory(replace(config, n_particles=5), state, parameters,
                               [ToolControl.stationary(0)])


if __name__ == "__main__":
    unittest.main()
