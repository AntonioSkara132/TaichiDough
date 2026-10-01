"""Compression, gradients, trajectory splits and learned rollout regression tests."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from experiments.differentiable_mpm.reduced_order import PODCodec, ReducedDynamics, control_features
from experiments.differentiable_mpm.reduced_order_data import Trajectory, encode_control, pack_state
from experiments.differentiable_mpm.state import DEFAULT_PARAMETERS, PARAMETER_NAMES, ParticleState, ToolControl
from experiments.differentiable_mpm.train_reduced_order import split_trajectories, train, evaluate


def trajectory(index, steps=12):
    control = ToolControl.stationary()
    control.poses[:, :3] = .4
    control.velocities[:, 0] = .05 + index * .002
    controls = np.repeat(encode_control(control)[None], steps, axis=0)
    state = ParticleState.initial([[.4, .4, .4], [.41, .4, .4]])
    state.v[:, 0] = control.velocities[0, 0]
    initial = pack_state(state)
    states = np.repeat(initial[None], steps + 1, axis=0)
    for k in range(steps + 1):
        states[k, [0, 3]] += k * .01 * control.velocities[0, 0]
    return Trajectory(states, controls, np.array([DEFAULT_PARAMETERS[n] for n in PARAMETER_NAMES]),
                      .01, f'trajectory-{index}', 'fixed-test-scene')


class ReducedOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_codec_round_trip_and_gradients(self):
        rng = np.random.default_rng(5)
        base = rng.normal(size=(1, 50))
        direction = rng.normal(size=(1, 50))
        snapshots = base + np.arange(8)[:, None] * direction
        codec = PODCodec.fit(snapshots, 2, rank=1)
        states = torch.tensor(snapshots, dtype=torch.float32, requires_grad=True)
        reconstructed = codec.decode(codec.encode(states))
        torch.testing.assert_close(reconstructed, states, atol=3e-6, rtol=3e-6)
        reconstructed.square().sum().backward()
        self.assertTrue(torch.isfinite(states.grad).all())
        self.assertGreater(float(states.grad.abs().sum()), 0)

    def test_constant_fields_and_train_only_mean(self):
        snapshots = np.ones((5, 50))
        codec = PODCodec.fit(snapshots, 2, 2)
        self.assertTrue(torch.isfinite(codec.encode(torch.tensor(snapshots).float())).all())
        torch.testing.assert_close(codec.mean_x, torch.ones(6))
        torch.testing.assert_close(codec.decode(torch.zeros(codec.latent_dim)), torch.ones(50))
        torch.testing.assert_close(codec.decode(torch.randn(codec.latent_dim)), torch.ones(50))
        self.assertEqual(float(codec.latent_active.sum()), 0)
        with self.assertRaises(ValueError):
            PODCodec.fit(np.full((4, 50), np.nan), 2)

    def test_rollout_gradients_and_save_load(self):
        t = trajectory(1)
        codec = PODCodec.fit(t.states, 2, 2)
        model = ReducedDynamics(codec, t.identity, t.dt, 16)
        # Nonzero weights test actual control sensitivity, not just graph presence.
        torch.nn.init.normal_(model.network[-1].weight, std=.02)
        z = codec.encode(torch.tensor(t.states[0]).float())
        controls = torch.tensor(t.controls).float().requires_grad_()
        parameters = torch.tensor(t.parameters).float().requires_grad_()
        result = model.rollout(z, controls, parameters)
        result[-1].sum().backward()
        for gradient in (controls.grad, parameters.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            model.save(path)
            loaded = ReducedDynamics.load(path)
            torch.testing.assert_close(loaded.rollout(z, controls, parameters), result)
        with self.assertRaises(ValueError):
            model.check_scene('other', t.dt, 2)

    def test_control_pose_encoding_matches_collector_and_is_differentiable(self):
        control = ToolControl.stationary()
        control.poses[0, 3:] = np.array([.1, .2, .3, .4]) / np.sqrt(.3)
        poses = torch.tensor(control.poses, dtype=torch.float64, requires_grad=True)
        velocities = torch.tensor(control.velocities, dtype=torch.float64, requires_grad=True)
        features = control_features(poses, velocities)
        torch.testing.assert_close(features, torch.tensor(encode_control(control)))
        flipped = poses.detach().clone()
        flipped[:, 3:] *= -1
        torch.testing.assert_close(features, control_features(flipped, velocities))
        features.square().sum().backward()
        self.assertTrue(torch.isfinite(poses.grad).all())
        self.assertGreater(float(poses.grad[:, :3].abs().sum()), 0)

    def test_split_by_complete_trajectory(self):
        groups = split_trajectories([trajectory(i) for i in range(10)])
        ids = [{t.trajectory_id for t in group} for group in groups]
        self.assertEqual(len(set.union(*ids)), 10)
        self.assertFalse(ids[0] & ids[1] or ids[1] & ids[2] or ids[0] & ids[2])
        with self.assertRaises(ValueError):
            split_trajectories([trajectory(1)] * 3)

    def test_training_improves_independent_linear_rollout(self):
        training = [trajectory(i) for i in (0, 2, 4, 6)]
        validation = [trajectory(3)]
        model, report = train(training, validation, rank=1, hidden_dim=24,
                              epochs=60, horizon=4, learning_rate=.003)
        result = evaluate(model, [trajectory(5)])[0]
        self.assertLess(result['errors_rms']['rollout']['x'], result['errors_rms']['persistence']['x'] * .5)
        self.assertGreater(report['best_epoch'], 0)
        self.assertTrue(result['finite'])
        self.assertEqual(report['train_ids'], [t.trajectory_id for t in training])


if __name__ == '__main__':
    unittest.main()
