"""Compare complete forward states in separate deterministic CPU processes."""

from dataclasses import replace
import argparse
import json
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from experiments.differentiable_mpm.reference_adapter import (
    REPOSITORY_ROOT, EXPERIMENT_ROOT, current_reference_policy, reference_policy, reference_identity,
)
from experiments.differentiable_mpm.state import (
    DEFAULT_PARAMETERS, ParticleState, SDFData, SimulationConfig, STATE_NAMES, ToolControl,
)


FIXTURES = ("elastic", "viscous", "compression", "stretch", "jp_hardening", "jp_clipped", "floor", "floor_zero", "sdf")
ATOL = {"x": 1e-6, "v": 1e-5, "C": 1e-4, "F": 1e-5, "Jp": 1e-5}
RTOL = 1e-5


def scratch_root():
    job = os.environ.get("CLAUDE_JOB_DIR")
    root = Path(job) / "tmp" if job else EXPERIMENT_ROOT / "runs" / "test_tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _rotation_quaternion(axis, angle):
    q = np.zeros(4, dtype=np.float64)
    q[axis] = np.sin(angle / 2)
    q[3] = np.cos(angle / 2)
    return q


def box_sdf_fixture() -> SDFData:
    """Two fixed solid SDFs with separately sampled, normalized gradient volumes."""
    r = 24
    minimums = np.full((2, 3), -0.12, dtype=np.float32)
    spacings = np.full((2, 3), 0.24 / (r - 1), dtype=np.float32)
    xyz = np.stack(np.meshgrid(*[np.linspace(-0.12, 0.12, r)] * 3, indexing="ij"), axis=-1)
    distances, gradients = [], []
    for extents in (np.array([0.025, 0.06, 0.07]), np.array([0.07, 0.06, 0.035])):
        q = np.abs(xyz) - extents
        sdf = (np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(np.max(q, axis=-1), 0)).astype(np.float32)
        normal = np.stack(np.gradient(sdf, *spacings[0], edge_order=1), axis=-1).astype(np.float32)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
        distances.append(sdf)
        gradients.append(normal)
    return SDFData(np.stack(distances), np.stack(gradients), minimums, spacings)


def make_fixture(name, steps=12, physics_version="corrected-v1"):
    if name not in FIXTURES:
        raise ValueError(f"Unknown parity fixture: {name}")
    offsets = np.stack(np.meshgrid(*[[-0.003, 0.0, 0.003]] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    state = ParticleState.initial(offsets + [0.445, 0.35, 0.475])
    config = SimulationConfig(n_particles=len(state.x), grid=24, dt=1e-4,
                              particle_mass=1.8e-5, particle_volume=6e-8,
                              floor_y=0.0, precision="f32", physics_version=physics_version)
    parameters = dict(DEFAULT_PARAMETERS, youngs_modulus=12000.0, poisson_ratio=0.27,
                      plastic_min=0.94, plastic_max=1.06, tool_retention=0.23, floor_retention=0.37)
    state.v[:] = [0.12, -0.08, 0.045]
    state.v += offsets * [0.4, -0.6, 0.2]
    state.C[:] = [[-0.8, 0.4, 0.1], [-0.2, 0.55, 0.25], [0.15, -0.1, 0.3]]
    state.F[:] = [[1.025, 0.012, -0.003], [0.0, 0.986, 0.006], [0.0, 0.0, 1.015]]
    sdf = None
    if name == "viscous":
        parameters["viscosity"] = 8.0
        state.F[:] = np.eye(3)
        state.C[:] = [[-4.5, 3.0, 0.7], [-1.1, 2.6, -0.5], [0.2, 1.3, -0.4]]
    if name in {"compression", "stretch", "jp_hardening", "jp_clipped"}:
        config = replace(config, plasticity="stretch-clamp", plastic_velocity_damping=0.97,
                         plastic_affine_damping=0.93)
        if name == "compression":
            state.F[:] = np.diag([0.86, 1.01, 1.02])
        elif name == "stretch":
            state.F[:] = np.diag([1.14, 0.99, 0.98])
        else:
            config = replace(config, use_jp=True, jp_hardening=2.3)
            state.F[:] = np.diag([0.86, 1.12, 1.035])
            state.Jp[:] = 0.97
            if name == "jp_clipped":
                config = replace(config, jp_min=0.6, jp_max=1.2)
                state.Jp[:] = 0.601
                state.F[:] = np.diag([0.80, 0.83, 1.025])
    if name == "floor":
        config = replace(config, floor_y=0.25, floor_absorption=0.12, floor_stickiness=0.18,
                         floor_plastic_damping_band=0.006, plastic_affine_damping=0.86)
        state.x[:, 1] = 0.24997 + offsets[:, 1] * 0.003
        state.v[:] = [0.18, -0.35, 0.12]
        state.F[:] = np.eye(3)
    if name == "floor_zero":
        # The preserved truncating stencil permits fx < 0.5 and negative grid mass.
        config = replace(config, floor_y=0.0, particle_mass=0.0018, particle_volume=6e-6,
                         floor_plastic_damping_band=0.006, plastic_affine_damping=0.86)
        state.x[:, 1] = 5e-5 + offsets[:, 1] / 60.0
        state.v[:] = [0.18, -0.35, 0.12]
        state.F[:] = np.eye(3)
    if name == "sdf":
        config = replace(config, tool_collision="sdf", tool_contact_padding=0.0026,
                         tool_contact_absorption=0.11, tool_stickiness=0.19)
        state.x[:, 0] = 0.445 + offsets[:, 0] * 0.5
        state.x[:, 2] = 0.475 + offsets[:, 2] * 0.5
        state.v[:] = [-0.2, -0.1, -0.15]
        state.F[:] = np.eye(3)
        sdf = box_sdf_fixture()
    controls = []
    for step in range(steps):
        time = step * config.dt
        control = ToolControl.stationary(time)
        if name == "sdf":
            control.velocities[0] = [0.12, -0.015, 0.018, 0.0, 0.0, 0.8]
            control.velocities[1] = [-0.015, 0.008, 0.08, 0.0, -0.7, 0.0]
            control.poses[0, :3] = np.array([0.425, 0.35, 0.48]) + time * control.velocities[0, :3]
            control.poses[1, :3] = np.array([0.47, 0.35, 0.446]) + time * control.velocities[1, :3]
            control.poses[0, 3:] = _rotation_quaternion(2, 0.10 + time * 0.8)
            control.poses[1, 3:] = _rotation_quaternion(1, 0.12 - time * 0.7)
        controls.append(control)
    return config, parameters, state, controls, sdf


def run_worker(kind, fixture, steps, output, physics_version="corrected-v1"):
    import taichi as ti

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Parity worker refuses to replace {output}")
    config, parameters, initial, controls, sdf = make_fixture(fixture, steps, physics_version)
    selected_reference = reference_identity(physics_version)
    ti.init(arch=ti.cpu, enable_fallback=False, default_fp=ti.f32, cpu_max_num_threads=1,
            random_seed=0, offline_cache=False, debug=True, fast_math=False)
    snapshots = [initial.copy()]
    contacts = []
    negative_grid_mass_nodes = []
    if kind == "reference":
        from experiments.differentiable_mpm.reference_adapter import ReferenceStepper

        solver = ReferenceStepper(config, parameters, sdf=sdf)
        solver.load_state(initial)
        loaded = solver.state()
        for name in STATE_NAMES:
            np.testing.assert_array_equal(getattr(loaded, name), getattr(initial, name))
        grid_mass = None
        if fixture == "floor_zero":
            kernel = inspect.getclosurevars(solver._substep).nonlocals["substep_kernel"]
            grid_mass = inspect.getclosurevars(kernel._primal.func).nonlocals["grid_m"]
        for control in controls:
            contacts.append(solver.advance(control))
            snapshots.append(solver.state())
            if grid_mass is not None:
                negative_grid_mass_nodes.append(int(np.count_nonzero(grid_mass.to_numpy() < 0)))
    elif kind == "differentiable":
        from experiments.differentiable_mpm.solver import Stepper

        solver = Stepper(config, parameters, capacity=steps + 1, sdf=sdf)
        solver.load_state(0, initial)
        loaded = solver.state(0)
        for name in STATE_NAMES:
            np.testing.assert_array_equal(getattr(loaded, name), getattr(initial, name))
        for index, control in enumerate(controls):
            solver.advance(index, control)
            snapshots.append(solver.state(index + 1))
            if fixture == "floor_zero":
                negative_grid_mass_nodes.append(int(np.count_nonzero(solver.grid_m.to_numpy() < 0)))
    else:
        raise ValueError(f"Unknown worker: {kind}")
    arrays = {name: np.stack([getattr(state, name) for state in snapshots]) for name in STATE_NAMES}
    arrays["diagnostics"] = np.asarray(json.dumps({"kind": kind, "fixture": fixture, "contacts": contacts,
                                                  "reference_identity": selected_reference,
                                                  "negative_grid_mass_nodes": negative_grid_mass_nodes,
                                                  "actual_arch": str(ti.lang.impl.current_cfg().arch)}))
    np.savez(output, **arrays)


def worker_command(kind, fixture, steps, output, physics_version="corrected-v1"):
    return [sys.executable, "-m", "experiments.differentiable_mpm.tests.test_forward_parity",
            "--reference-policy", current_reference_policy(), "--physics-version", physics_version,
            "--worker", kind, "--fixture", fixture, "--steps", str(steps), "--output", str(output)]


def launch_worker(kind, fixture, steps, output, physics_version="corrected-v1"):
    command = worker_command(kind, fixture, steps, output, physics_version)
    result = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, timeout=240)
    if result.returncode:
        raise AssertionError(f"{' '.join(command)} failed ({result.returncode})\n{result.stdout}\n{result.stderr}")
    with np.load(output, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in STATE_NAMES}
        diagnostics = json.loads(str(saved["diagnostics"]))
    return arrays, diagnostics


class ForwardParityTests(unittest.TestCase):
    def test_corrected_forward_states_match_corrected_reference(self):
        self.check_forward_states("corrected-v1")

    def test_legacy_forward_states_match_original_reference(self):
        self.check_forward_states("legacy-v1")

    def check_forward_states(self, physics_version):
        with tempfile.TemporaryDirectory(dir=scratch_root(), prefix="forward-parity-") as temporary:
            for fixture in FIXTURES:
                with self.subTest(fixture=fixture):
                    output = Path(temporary)
                    original, diagnostics = launch_worker("reference", fixture, 12, output / f"{fixture}-reference.npz",
                                                          physics_version)
                    differentiated, ad_diagnostics = launch_worker("differentiable", fixture, 12,
                                                                   output / f"{fixture}-ad.npz", physics_version)
                    self.assertEqual(diagnostics["reference_identity"], reference_identity(physics_version))
                    self.assertEqual(ad_diagnostics["reference_identity"], diagnostics["reference_identity"])
                    for name in STATE_NAMES:
                        # Checking each state detects the first divergence rather than a final-image coincidence.
                        np.testing.assert_allclose(differentiated[name], original[name], rtol=RTOL, atol=ATOL[name],
                                                   err_msg=f"{fixture}: {name} trajectory differs")
                    if fixture == "sdf":
                        records = diagnostics["contacts"]
                        for tool in range(2):
                            self.assertGreater(sum(row[tool]["particles"]["applied_responses"] for row in records), 0,
                                               f"SDF fixture did not exercise particle contact for tool {tool}")
                        self.assertGreater(sum(row[tool]["grid_nodes"]["applied_responses"]
                                               for row in records for tool in range(2)), 0)
                    elif fixture == "floor_zero":
                        for observed in (diagnostics, ad_diagnostics):
                            if physics_version == "legacy-v1":
                                self.assertGreater(min(observed["negative_grid_mass_nodes"]), 0,
                                                   "Legacy fixture did not exercise negative grid mass")
                            else:
                                self.assertEqual(observed["negative_grid_mass_nodes"], [0] * 12)
                        self.assertTrue(np.all(original["x"][1:, :, 1] >= 0.0))
                    elif fixture == "floor":
                        self.assertTrue(np.all(original["x"][1:, :, 1] >= 0.25))
                        self.assertTrue(np.any(original["x"][1, :, 1] == np.float32(0.25)))
                    elif fixture == "jp_hardening":
                        self.assertGreater(np.max(np.abs(original["Jp"][1] - original["Jp"][0])), 1e-3)
                    elif fixture == "jp_clipped":
                        np.testing.assert_allclose(original["Jp"][1], 0.6, rtol=0, atol=1e-7)
                    elif fixture in {"compression", "stretch"}:
                        singular = np.linalg.svd(original["F"][1], compute_uv=False)
                        self.assertGreaterEqual(float(singular.min()), 0.94 - 2e-6)
                        self.assertLessEqual(float(singular.max()), 1.06 + 2e-6)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--reference-policy", choices=("strict", "frozen"), default="strict")
        parser.add_argument("--physics-version", choices=("corrected-v1", "legacy-v1"), default="corrected-v1")
        parser.add_argument("--worker", choices=("reference", "differentiable"), required=True)
        parser.add_argument("--fixture", choices=FIXTURES, required=True)
        parser.add_argument("--steps", type=int, default=12)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        if args.steps < 1:
            parser.error("--steps must be positive")
        with reference_policy(args.reference_policy):
            run_worker(args.worker, args.fixture, args.steps, args.output, args.physics_version)
    else:
        unittest.main()
