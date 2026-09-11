"""Run actual Taichi spectral kernels in separate fp32 and fp64 CPU processes.

From the repository root:
    python3 -m unittest discover -s experiments/differentiable_mpm/tests -p test_spectral.py -v
A single precision can be selected with: python3 <this file> --precision f64
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import taichi as ti

from experiments.differentiable_mpm.spectral import (
    clamp_forward, clamp_vjp, polar_forward, polar_vjp,
)


@ti.data_oriented
class SpectralKernels:
    def __init__(self, dtype):
        self.F = ti.Matrix.field(3, 3, dtype=dtype, shape=())
        self.bar = ti.Matrix.field(3, 3, dtype=dtype, shape=())
        self.bounds = ti.Vector.field(2, dtype=dtype, shape=())
        self.rotation = ti.Matrix.field(3, 3, dtype=dtype, shape=())
        self.projected = ti.Matrix.field(3, 3, dtype=dtype, shape=())
        self.rotation_reference = ti.Matrix.field(3, 3, dtype=dtype, shape=())
        self.projected_reference = ti.Matrix.field(3, 3, dtype=dtype, shape=())
        self.singular_values = ti.Vector.field(3, dtype=dtype, shape=())
        self.rotation_bar = ti.Matrix.field(3, 3, dtype=dtype, shape=())
        self.projected_bar = ti.Matrix.field(3, 3, dtype=dtype, shape=())
        self.bounds_bar = ti.Vector.field(2, dtype=dtype, shape=())
        self.yielded = ti.field(dtype=ti.i32, shape=())
        self.yielded_reference = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def forward(self):
        self.rotation[None] = polar_forward(self.F[None])
        Y, yielded = clamp_forward(self.F[None], self.bounds[None][0], self.bounds[None][1])
        self.projected[None] = Y
        self.yielded[None] = yielded

    @ti.kernel
    def reference(self):
        R, _ = ti.polar_decompose(self.F[None])
        self.rotation_reference[None] = R
        U, sig, V = ti.svd(self.F[None])
        self.singular_values[None] = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])
        yielded = 0
        for i in ti.static(range(3)):
            original = sig[i, i]
            projected = ti.min(ti.max(original, self.bounds[None][0]), self.bounds[None][1])
            if ti.abs(original - projected) > 1e-6:
                yielded = 1
            sig[i, i] = projected
        self.projected_reference[None] = U @ sig @ V.transpose()
        self.yielded_reference[None] = yielded

    @ti.kernel
    def backward(self):
        self.rotation_bar[None] = polar_vjp(self.F[None], self.bar[None])
        F_bar, lower_bar, upper_bar = clamp_vjp(
            self.F[None], self.bounds[None][0], self.bounds[None][1], self.bar[None],
        )
        self.projected_bar[None] = F_bar
        self.bounds_bar[None] = ti.Vector([lower_bar, upper_bar])

    def value(self, F, lower, upper, G, which):
        self.F[None] = F
        self.bounds[None] = [lower, upper]
        self.forward()
        output = self.rotation.to_numpy() if which == "polar" else self.projected.to_numpy()
        # A float64 host reduction limits cancellation in the test oracle only.
        return float(np.sum(output.astype(np.float64) * G))

    def adjoint(self, F, lower, upper, G, which):
        self.F[None] = F
        self.bounds[None] = [lower, upper]
        self.bar[None] = G
        self.backward()
        field = self.rotation_bar if which == "polar" else self.projected_bar
        return field.to_numpy().astype(np.float64), self.bounds_bar.to_numpy().astype(np.float64)


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def _fixtures():
    L = _rotation([1, 2, -1], 0.52)
    R = _rotation([2, -1, 3], -0.37)
    return {
        "identity": np.eye(3),
        "isotropic_compression": 0.75 * np.eye(3),
        "isotropic_expansion": 1.35 * np.eye(3),
        "rigid_rotation": L,
        "repeated_interior": L @ np.diag([1.025, 1.025, 1.025]) @ R.T,
        "repeated_lower": L @ np.diag([0.75, 0.75, 1.025]) @ R.T,
        "repeated_upper": L @ np.diag([1.35, 1.35, 0.75]) @ R.T,
        "distinct": L @ np.diag([1.35, 1.025, 0.75]) @ R.T,
    }


def _numpy_forward(F, lower, upper, which):
    """High-accuracy mathematical-map oracle, used only in gradient tests."""
    U, s, Vh = np.linalg.svd(F)
    return U @ Vh if which == "polar" else (U * np.clip(s, lower, upper)) @ Vh


def _run_precision(precision):
    dtype = ti.f64 if precision == "f64" else ti.f32
    np_dtype = np.float64 if precision == "f64" else np.float32
    ti.init(arch=ti.cpu, default_fp=dtype, cpu_max_num_threads=1, fast_math=False,
            offline_cache=False, debug=True, enable_fallback=False)
    kernels = SpectralKernels(dtype)
    rng = np.random.default_rng(8242)
    G = rng.normal(size=(3, 3)).astype(np_dtype).astype(np.float64)
    steps = (1e-4, 2e-5, 5e-6) if precision == "f64" else (1e-2, 4e-3, 2e-3)
    derivative_atol = 2e-7 if precision == "f64" else 3e-3
    analytic_atol = 2e-13 if precision == "f64" else 3e-6
    maximum_errors = {"matrix_directional": 0.0, "bound_directional": 0.0,
                      "mathematical_map_directional": 0.0, "forward_scalar_approximation": 0.0}

    class KernelTests(unittest.TestCase):
        def test_forward_is_reference_operation(self):
            for name, F in _fixtures().items():
                with self.subTest(fixture=name):
                    kernels.F[None] = F
                    kernels.bounds[None] = [0.9, 1.1]
                    kernels.forward()
                    kernels.reference()
                    np.testing.assert_array_equal(kernels.rotation.to_numpy(), kernels.rotation_reference.to_numpy())
                    np.testing.assert_array_equal(kernels.projected.to_numpy(), kernels.projected_reference.to_numpy())
                    self.assertEqual(kernels.yielded[None], kernels.yielded_reference[None])
                    self.assertTrue(np.isfinite(kernels.rotation.to_numpy()).all())
                    self.assertTrue(np.isfinite(kernels.projected.to_numpy()).all())

        def test_identity_and_repeated_stretch_analytic_limits(self):
            F_bar, bounds_bar = kernels.adjoint(np.eye(3), 0.9, 1.1, G, "clamp")
            np.testing.assert_allclose(F_bar, G, rtol=0, atol=analytic_atol)
            np.testing.assert_allclose(bounds_bar, 0, rtol=0, atol=analytic_atol)
            for stretch in (0.75, 1.0, 1.35):
                F = np.eye(3) * stretch
                F_bar, _ = kernels.adjoint(F, 0.9, 1.1, G, "polar")
                np.testing.assert_allclose(F_bar, (G - G.T) / (2 * stretch), rtol=0, atol=analytic_atol)
            for stretch, bound, expected_bound in ((0.75, 0.9, 0), (1.35, 1.1, 1)):
                F_bar, bounds_bar = kernels.adjoint(np.eye(3) * stretch, 0.9, 1.1, G, "clamp")
                np.testing.assert_allclose(F_bar, (G - G.T) * bound / (2 * stretch), rtol=0, atol=analytic_atol)
                expected = np.zeros(2)
                expected[expected_bound] = np.trace(G)
                np.testing.assert_allclose(bounds_bar, expected, rtol=0, atol=analytic_atol)

        def test_matrix_directional_gradients_at_multiple_steps(self):
            for name, F in _fixtures().items():
                F = F.astype(np_dtype).astype(np.float64)
                directions = [rng.normal(size=(3, 3)) for _ in range(2)]
                if name in {"identity", "distinct"}:
                    directions.extend(np.eye(9).reshape(9, 3, 3))
                for which in ("polar", "clamp"):
                    F_bar, _ = kernels.adjoint(F, 0.9, 1.1, G, which)
                    for index, direction in enumerate(directions):
                        direction /= np.linalg.norm(direction)
                        ad = float(np.sum(F_bar * direction))
                        for h in steps:
                            with self.subTest(fixture=name, operation=which, direction=index, step=h):
                                plus = kernels.value(F + h * direction, 0.9, 1.1, G, which)
                                minus = kernels.value(F - h * direction, 0.9, 1.1, G, which)
                                fd = (plus - minus) / (2 * h)
                                error = abs(fd - ad)
                                maximum_errors["matrix_directional"] = max(maximum_errors["matrix_directional"], error)
                                allowed = derivative_atol * max(1.0, abs(ad))
                                if precision == "f64":
                                    # The unchanged Taichi SVD has an approximately
                                    # 1e-10 forward approximation near repeated
                                    # stretches. Separate its 1/h amplification
                                    # from error in the composite VJP itself.
                                    exact_plus = float(np.sum(_numpy_forward(F + h * direction, 0.9, 1.1, which) * G))
                                    exact_minus = float(np.sum(_numpy_forward(F - h * direction, 0.9, 1.1, which) * G))
                                    map_fd = (exact_plus - exact_minus) / (2 * h)
                                    map_error = abs(map_fd - ad)
                                    approximation = max(abs(plus - exact_plus), abs(minus - exact_minus))
                                    maximum_errors["mathematical_map_directional"] = max(
                                        maximum_errors["mathematical_map_directional"], map_error)
                                    maximum_errors["forward_scalar_approximation"] = max(
                                        maximum_errors["forward_scalar_approximation"], approximation)
                                    self.assertLessEqual(map_error, allowed)
                                    self.assertLessEqual(approximation, 1e-9)
                                    approximation_fd = abs((plus - exact_plus) - (minus - exact_minus)) / (2 * h)
                                    self.assertLessEqual(error, allowed + approximation_fd)
                                else:
                                    self.assertLessEqual(error, allowed)

        def test_bound_gradients_at_multiple_steps(self):
            for name, F in _fixtures().items():
                F = F.astype(np_dtype).astype(np.float64)
                _, bound_bar = kernels.adjoint(F, 0.9, 1.1, G, "clamp")
                for i in range(2):
                    for h in steps:
                        with self.subTest(fixture=name, bound=i, step=h):
                            plus_bounds = [0.9, 1.1]
                            minus_bounds = [0.9, 1.1]
                            plus_bounds[i] += h
                            minus_bounds[i] -= h
                            plus = kernels.value(F, *plus_bounds, G, "clamp")
                            minus = kernels.value(F, *minus_bounds, G, "clamp")
                            fd = (plus - minus) / (2 * h)
                            error = abs(fd - bound_bar[i])
                            maximum_errors["bound_directional"] = max(maximum_errors["bound_directional"], error)
                            self.assertLessEqual(error, derivative_atol * max(1.0, abs(bound_bar[i])))

        def test_taichi_forward_differences_at_usable_steps(self):
            # Larger fp64 perturbations avoid amplifying the reference SVD's
            # approximately 1e-10 forward error. This is a direct finite-
            # difference check of actual Taichi outputs, without a NumPy oracle.
            usable_steps = (2e-3, 1e-3, 5e-4) if precision == "f64" else steps
            tolerance = 1e-5 if precision == "f64" else derivative_atol
            local_rng = np.random.default_rng(8242)
            local_rng.normal(size=(3, 3))
            for name, F in _fixtures().items():
                F = F.astype(np_dtype).astype(np.float64)
                directions = [local_rng.normal(size=(3, 3)) for _ in range(2)]
                if name in {"identity", "distinct"}:
                    directions.extend(np.eye(9).reshape(9, 3, 3))
                for which in ("polar", "clamp"):
                    F_bar, _ = kernels.adjoint(F, 0.9, 1.1, G, which)
                    for direction in directions:
                        direction /= np.linalg.norm(direction)
                        ad = float(np.sum(F_bar * direction))
                        for h in usable_steps:
                            with self.subTest(fixture=name, operation=which, step=h):
                                plus = kernels.value(F + h * direction, 0.9, 1.1, G, which)
                                minus = kernels.value(F - h * direction, 0.9, 1.1, G, which)
                                error = abs((plus - minus) / (2 * h) - ad)
                                maximum_errors["usable_step_directional"] = max(
                                    maximum_errors.get("usable_step_directional", 0.0), error)
                                self.assertLessEqual(error, tolerance * max(1.0, abs(ad)))

        def test_threshold_convention_uses_capped_side(self):
            F = np.eye(3, dtype=np_dtype)
            kernels.F[None] = F
            kernels.bounds[None] = [0.9, 1.1]
            kernels.reference()
            s = kernels.singular_values.to_numpy()
            lower = float(np.min(s))
            F_bar, bounds_bar = kernels.adjoint(F, lower, 1.1, np.eye(3), "clamp")
            capped = int(np.count_nonzero(s <= lower))
            self.assertGreaterEqual(capped, 1)
            np.testing.assert_allclose(bounds_bar, [capped, 0], atol=analytic_atol, rtol=0)
            np.testing.assert_allclose(np.trace(F_bar), 3 - capped, atol=analytic_atol, rtol=0)
            upper = float(np.max(s))
            F_bar, bounds_bar = kernels.adjoint(F, 0.9, upper, np.eye(3), "clamp")
            capped = int(np.count_nonzero(s >= upper))
            np.testing.assert_allclose(bounds_bar, [0, capped], atol=analytic_atol, rtol=0)
            np.testing.assert_allclose(np.trace(F_bar), 3 - capped, atol=analytic_atol, rtol=0)

        def test_yield_flag_tolerance_does_not_disable_projection_derivative(self):
            # The reference's 1e-6 yielded flag is for damping, not for deciding
            # whether projection or its derivative exists.
            F = np.eye(3, dtype=np_dtype)
            kernels.F[None] = F
            kernels.bounds[None] = [0.9, 1.1]
            kernels.reference()
            computed_stretches = kernels.singular_values.to_numpy()
            upper = float(np.min(computed_stretches) - np_dtype(4e-7))
            kernels.bounds[None] = [0.9, upper]
            kernels.forward()
            self.assertEqual(kernels.yielded[None], 0)
            F_bar, bounds_bar = kernels.adjoint(F, 0.9, upper, G, "clamp")
            self.assertGreater(abs(bounds_bar[1]), 0.1)
            np.testing.assert_allclose(F_bar + F_bar.T, 0, atol=analytic_atol, rtol=0)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(KernelTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("SPECTRAL_RESULT " + json.dumps({"precision": precision, "backend": "cpu",
          "tests": result.testsRun, "successful": result.wasSuccessful(),
          "max_absolute_errors": maximum_errors}), flush=True)
    return 0 if result.wasSuccessful() else 1


class TestSpectral(unittest.TestCase):
    def _check_precision(self, precision):
        root = Path(__file__).resolve().parents[3]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--precision", precision],
                                cwd=root, env=env, text=True, capture_output=True, timeout=240)
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
        summary = next(line for line in result.stdout.splitlines() if line.startswith("SPECTRAL_RESULT "))
        print(summary, flush=True)

    def test_cpu_f64(self):
        self._check_precision("f64")

    def test_cpu_f32(self):
        self._check_precision("f32")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("f32", "f64"))
    args, remaining = parser.parse_known_args()
    if args.precision:
        raise SystemExit(_run_precision(args.precision))
    unittest.main(argv=[sys.argv[0], *remaining])
