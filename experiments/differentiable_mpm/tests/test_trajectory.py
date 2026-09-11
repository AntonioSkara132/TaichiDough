"""Actual Taichi full-trajectory and segmented-adjoint comparisons."""
from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.differentiable_mpm.checkpoint import CheckpointedRollout
from experiments.differentiable_mpm.runtime import init_runtime
from experiments.differentiable_mpm.solver import Stepper
from experiments.differentiable_mpm.state import DEFAULT_PARAMETERS, ParticleState, SimulationConfig, ToolControl


@dataclass
class Target:
    frame_index: int
    step: int
    positions: np.ndarray


class ParticleTargetLoss:
    """An analytic position loss for checking the simulation adjoint, not real data."""
    def value_and_grad_positions(self, positions, observation, compute_grad=True):
        r = (positions - observation.positions) / 0.01
        return SimpleNamespace(value=float(0.5 * np.sum(r * r) / len(r)),
                               gradient=r / (0.01 * len(r)) if compute_grad else None,
                               components={}, diagnostics={})


def fixture(plastic=False):
    positions = np.array([[0.41 + 0.025 * i, 0.43 + 0.025 * j, 0.42 + 0.025 * k]
                          for i in range(2) for j in range(2) for k in range(2)])
    state = ParticleState.initial(positions, np.float64)
    state.v[:] = np.stack((positions[:, 1] * 0.05, positions[:, 0] * -0.03, positions[:, 2] * 0.04), axis=1)
    state.C[:] = [[0.1, 0.03, -0.02], [0.04, -0.07, 0.01], [0.02, 0.01, 0.06]]
    state.F[:] = np.diag([0.88, 1.13, 1.005] if plastic else [0.975, 1.036, 1.012])
    state.Jp[:] = np.linspace(0.95, 1.02, len(positions))
    cfg = SimulationConfig(n_particles=len(positions), grid=12, dt=0.0002,
                           particle_mass=0.001, particle_volume=1e-6, gravity=-1.0,
                           plasticity='stretch-clamp' if plastic else 'none',
                           use_jp=plastic, jp_hardening=0.4, precision='f64')
    params = dict(DEFAULT_PARAMETERS, youngs_modulus=6000, viscosity=3.0,
                  plastic_min=0.94, plastic_max=1.07)
    controls = [ToolControl.stationary(i * cfg.dt) for i in range(8)]
    center = positions.mean(axis=0)
    targets = [Target(i, step, center + (positions - center) * np.array([0.93, 1.08, 0.95])
                      + np.array([0.001, -0.0005, 0.0008]) * (1 + i / 5))
               for i, step in enumerate([0, 2, 3, 4, 6, 8])]
    return cfg, state, params, controls, targets


class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_runtime('cpu', 'f64', cpu_threads=1)

    def check_case(self, plastic):
        cfg, state, params, controls, targets = fixture(plastic)
        stepper = Stepper(cfg, params, capacity=9)
        def rollout(length):
            return CheckpointedRollout(stepper, state, controls, targets, ParticleTargetLoss(), length,
                                       replay_rtol=0, replay_atol={n: 0 for n in state.arrays()})
        full = rollout(8).value_and_gradient(params)
        for length in (1, 3, 4):
            result = rollout(length).value_and_gradient(params)
            self.assertAlmostEqual(full.value, result.value, places=12)
            np.testing.assert_allclose(list(full.gradient.values()), list(result.gradient.values()), rtol=1e-10, atol=1e-11)
            for name in state.arrays():
                np.testing.assert_allclose(getattr(full.initial_gradient, name), getattr(result.initial_gradient, name), rtol=1e-10, atol=1e-11)
            self.assertEqual(result.diagnostics['observation_gradient_injections'], [1] * len(targets))
        names = ['youngs_modulus', 'poisson_ratio', 'viscosity']
        steps = {'youngs_modulus': 0.6, 'poisson_ratio': 1e-4, 'viscosity': 0.01,
                 'plastic_min': 1e-4, 'plastic_max': 1e-4}
        if plastic:
            names += ['plastic_min', 'plastic_max']
        errors = {}
        for name in names:
            best = float('inf')
            for multiplier in (1, 0.3):
                h = steps[name] * multiplier
                plus, minus = dict(params), dict(params)
                plus[name] += h
                minus[name] -= h
                fd = (rollout(3).value_and_gradient(plus, compute_grad=False).value -
                      rollout(3).value_and_gradient(minus, compute_grad=False).value) / (2 * h)
                ad = full.gradient[name]
                error = abs(ad - fd) / max(abs(ad), abs(fd), 1e-9)
                best = min(best, error)
                print(f'TRAJECTORY plastic={plastic} {name} h={h:g} AD={ad:.10g} FD={fd:.10g} relative={error:.4g}', flush=True)
            errors[name] = best
            self.assertLess(best, 1e-3, f'{name} derivative discrepancy: {best}')
        self.assertTrue(all(abs(full.gradient[name]) > 1e-12 for name in names))

    def test_elastic_viscous_full_trajectory(self):
        self.check_case(False)

    def test_plastic_jp_full_trajectory(self):
        self.check_case(True)


if __name__ == '__main__':
    unittest.main()
