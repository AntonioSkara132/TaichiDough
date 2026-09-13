"""Small serial-MPM checks for paper-style position objectives, not real data."""
from dataclasses import replace
import unittest

import numpy as np

from experiments.differentiable_mpm.checkpoint import CheckpointedRollout
from experiments.differentiable_mpm.loss_options import (
    loss_temporal_reduction, make_observation_loss, parse_loss_config,
)
from experiments.differentiable_mpm.optimize import AdamOptions, ObjectiveValue, ProjectedAdam
from experiments.differentiable_mpm.parameters import PhysicalParameterSpace
from experiments.differentiable_mpm.point_set_loss import PointSetObservation
from experiments.differentiable_mpm.renderer import Camera, LocalSplatRenderer
from experiments.differentiable_mpm.runtime import init_runtime
from experiments.differentiable_mpm.solver import Stepper
from experiments.differentiable_mpm.tests.test_trajectory import fixture


VERSIONS = (
    'dpsi-pcd-cd-v1', 'dpsi-prt-cd-v1',
    'dpsi-pcd-emd-v1', 'dpsi-prt-emd-v1',
    'empm-offline-v1', 'empm-mask-inspired-v1',
)


class PaperLossTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_runtime('cpu', 'f64', cpu_threads=1)
        config, cls.initial, cls.parameters, cls.controls, _ = fixture(False)
        cls.config = replace(config, p2g_mode='serial')
        cls.stepper = Stepper(cls.config, cls.parameters, capacity=len(cls.controls) + 1)
        reference = dict(cls.parameters, youngs_modulus=9000.0)
        cls.stepper.set_parameters(reference)
        cls.stepper.load_state(0, cls.initial)
        cls.targets = {}
        for step, control in enumerate(cls.controls):
            cls.stepper.advance(step, control)
            cls.targets[step + 1] = cls.stepper.state(step + 1).x.copy()
        transform = np.eye(4)
        transform[:3, 3] = [-0.43, -0.43, 0.5]
        cls.camera = Camera(32, 32, 80.0, 80.0, 16.0, 16.0, transform)
        renderer = LocalSplatRenderer(cls.camera, len(cls.initial.x), precision='f64')
        cls.masks = {step: renderer.forward(points).coverage > 0.2
                     for step, points in cls.targets.items()}
        if not all(mask.any() for mask in cls.masks.values()):
            raise AssertionError('Synthetic mask targets must have foreground support')
        cls.problems = {}
        for version in VERSIONS:
            options: dict[str, object] = {'version': version}
            if version == 'empm-mask-inspired-v1':
                # Isolate the renderer adjoint instead of letting point error dominate.
                options.update(geometric_weight=0.0, mask_weight=1.0)
            settings = parse_loss_config(options)
            objective = make_observation_loss(cls.camera, settings, cls.initial.x, precision='f64')
            steps = [8] if version.startswith('dpsi-') else [2, 3, 4, 6, 8]
            observations = []
            for step in steps:
                extra = {}
                if version == 'empm-offline-v1':
                    extra.update(track_particle_ids=np.arange(len(cls.initial.x)),
                                 track_positions_scene=cls.targets[step].copy(),
                                 track_valid=np.ones(len(cls.initial.x), dtype=bool))
                elif version == 'empm-mask-inspired-v1':
                    extra.update(foreground_mask=cls.masks[step],
                                 known_mask=np.ones((32, 32), dtype=bool))
                observations.append(PointSetObservation(
                    frame_index=step, step=step, timestamp=step * cls.config.dt,
                    points_scene=cls.targets[step].copy(),
                    target_representation='inferred_volume' if '-prt-' in version else 'partial_observed',
                    metadata={'source_kind': 'synthetic_fixture'}, **extra))
            cls.problems[version] = settings, objective, observations

    def rollout(self, version, length=3, reduction=None):
        settings, objective, observations = self.problems[version]
        return CheckpointedRollout(
            self.stepper, self.initial, self.controls, observations, objective,
            segment_length=length, observation_reduction=reduction or loss_temporal_reduction(settings),
            replay_rtol=0, replay_atol={name: 0 for name in self.initial.arrays()})

    def test_full_and_checkpointed_parameter_gradients(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                full = self.rollout(version, 8).value_and_gradient(self.parameters)
                segmented = self.rollout(version, 3).value_and_gradient(self.parameters)
                self.assertAlmostEqual(full.value, segmented.value, places=13)
                np.testing.assert_allclose(list(full.gradient.values()), list(segmented.gradient.values()),
                                           rtol=2e-9, atol=2e-12)
                for name in self.initial.arrays():
                    np.testing.assert_allclose(getattr(full.initial_gradient, name),
                                               getattr(segmented.initial_gradient, name),
                                               rtol=2e-9, atol=2e-11)
                self.assertEqual(segmented.diagnostics['observation_gradient_injections'],
                                 [1] * len(self.problems[version][2]))
                for name, base_h in [('youngs_modulus', 0.6), ('viscosity', 0.01)]:
                    ad = full.gradient[name]
                    self.assertGreater(abs(ad), 1e-15, f'{version}: inactive {name} fixture')
                    errors = []
                    for scale in (1.0, 0.3):
                        h = base_h * scale
                        plus, minus = dict(self.parameters), dict(self.parameters)
                        plus[name] += h
                        minus[name] -= h
                        rollout = self.rollout(version)
                        fd = (rollout.value_and_gradient(plus, compute_grad=False).value -
                              rollout.value_and_gradient(minus, compute_grad=False).value) / (2 * h)
                        error = abs(ad - fd) / max(abs(ad), abs(fd), 1e-15)
                        errors.append(error)
                        print(f'PAPER_LOSS {version} {name} h={h:g} AD={ad:.12g} '
                              f'FD={fd:.12g} relative={error:.4g}', flush=True)
                    self.assertLess(min(errors), 3e-3, f'{version} {name}: {errors}')

    def test_sequence_sum_scales_value_and_gradient(self):
        for version in ('empm-offline-v1', 'empm-mask-inspired-v1'):
            with self.subTest(version=version):
                mean = self.rollout(version, reduction='mean').value_and_gradient(self.parameters)
                total = self.rollout(version, reduction='sum').value_and_gradient(self.parameters)
                count = len(self.problems[version][2])
                self.assertAlmostEqual(total.value, count * mean.value, places=13)
                np.testing.assert_allclose(list(total.gradient.values()),
                                           np.asarray(list(mean.gradient.values())) * count,
                                           rtol=2e-9, atol=2e-12)

    def test_short_synthetic_fits_reduce_each_objective(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                rollout = self.rollout(version)
                space = PhysicalParameterSpace(self.parameters, ['youngs_modulus'], 'none',
                                               bounds={'youngs_modulus': [3000.0, 12000.0]})

                def objective(parameters):
                    result = rollout.value_and_gradient(parameters)
                    return ObjectiveValue(result.value, result.gradient, result.diagnostics)

                options = AdamOptions(learning_rate=0.1)
                if version == 'empm-offline-v1':
                    # This short fixture has micrometer-scale residuals in raw m².
                    # Keep the objective unscaled and tighten absolute tolerances.
                    options = replace(options, epsilon=1e-14, gradient_tolerance=1e-14,
                                      loss_tolerance=1e-20)
                optimizer = ProjectedAdam(space, objective, options)
                initial = optimizer.initialize().value
                result = optimizer.run(4)
                print(f'PAPER_FIT {version} initial={initial:.12g} final={result.best_value:.12g} '
                      f'accepted={result.accepted_updates} status={result.status}', flush=True)
                self.assertGreater(result.accepted_updates, 0, version)
                self.assertLess(result.best_value, initial, version)

    def test_raw_squared_meter_fixture_needs_explicit_optimizer_tolerances(self):
        rollout = self.rollout('empm-offline-v1')
        space = PhysicalParameterSpace(self.parameters, ['youngs_modulus'], 'none',
                                       bounds={'youngs_modulus': [3000.0, 12000.0]})

        def objective(parameters):
            result = rollout.value_and_gradient(parameters)
            return ObjectiveValue(result.value, result.gradient, result.diagnostics)

        optimizer = ProjectedAdam(space, objective)
        initial = optimizer.initialize()
        gradient = space.pullback(space.coordinates(), initial.gradient)
        self.assertGreater(abs(gradient[0]), 1e-14)
        self.assertLess(abs(gradient[0]), optimizer.options.gradient_tolerance)
        result = optimizer.run(1)
        self.assertEqual(result.status, 'converged_gradient')
        self.assertEqual(result.accepted_updates, 0)
        self.assertEqual(result.evaluations, 1)
        self.assertEqual(result.best_value, initial.value)


if __name__ == '__main__':
    unittest.main()
