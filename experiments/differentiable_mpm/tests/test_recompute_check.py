"""Analytic/mocked forward diagnostics. No Taichi initialization or simulation launch."""
from collections import Counter
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.differentiable_mpm import recompute_check as diagnostic
from experiments.differentiable_mpm.results import RunStore
from experiments.differentiable_mpm.state import ParticleState, STATE_NAMES, ToolControl


class MockField:
    def __init__(self, array):
        self.array = array.copy()

    def to_numpy(self):
        return self.array.copy()


class MockStepper:
    """Deterministic updates depend on every persistent array and the buffer slot."""
    def __init__(self, capacity, perturb=None):
        self.capacity = capacity
        self.slots = {}
        self.visits = Counter()
        self.advance_calls = []
        self.read_calls = []
        self.loads = []
        self.perturb = perturb
        self.counts = {}

    def load_state(self, slot, state):
        self.slots[slot] = state.copy()
        self.loads.append((slot, state.copy()))

    def state(self, slot):
        self.read_calls.append((len(self.advance_calls), slot))
        return self.slots[slot].copy()

    def advance(self, slot, control):
        assert 0 <= slot < self.capacity - 1
        step = int(control.time)
        self.visits[step] += 1
        before = self.slots[slot].copy()
        self.advance_calls.append((step, slot, control, before))
        coupling = sum(float(value.sum()) for value in before.arrays().values()) / 4096
        after = ParticleState(**{name: value + coupling + (index + 1) / 8 + slot / 64 + step / 128
                                 for index, (name, value) in enumerate(before.arrays().items())})
        scratch = {
            "trial": before.F + before.C / 16,
            "corrected": before.F + before.C / 16,
            "history": before.Jp + coupling,
            "rotation": np.full_like(before.F, coupling),
            "affine": before.F + before.C,
            "grid_m": np.full((2, 2, 2), coupling, dtype=before.x.dtype),
            "grid_p": np.full((2, 2, 2, 3), coupling + 0.125, dtype=before.x.dtype),
            "grid_u": np.full((2, 2, 2, 3), coupling + 0.25, dtype=before.x.dtype),
            "grid_v": np.full((2, 2, 2, 3), coupling + 0.5, dtype=before.x.dtype),
        }
        counts = {"yielded": step % 3, "particle_tool0": 2, "grid_tool0": 1}
        if self.perturb:
            self.perturb(self, step, before, after, scratch, counts)
        self.slots[slot + 1] = after
        self.counts = counts
        for name, value in scratch.items():
            setattr(self, name, MockField(value))

    def diagnostics(self):
        return dict(self.counts)

    def reverse_step(self, *args):
        raise AssertionError("Forward diagnostic must never compute a reverse step")

    def clear_state_gradients(self):
        raise AssertionError("Forward diagnostic must not manipulate gradients")

    def parameter_gradients(self):
        raise AssertionError("Forward diagnostic must not read gradients")


def initial_state():
    return ParticleState.initial(np.arange(6, dtype=np.float64).reshape(2, 3) / 16, np.float64)


def controls(count):
    values = [ToolControl.stationary(time=float(step)) for step in range(count)]
    for value in values:
        value.poses.flags.writeable = False
        value.velocities.flags.writeable = False
    return values


class TemporaryRunTest(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ["CLAUDE_JOB_DIR"]) / "tmp" if "CLAUDE_JOB_DIR" in os.environ else None
        self.temporary = tempfile.TemporaryDirectory(prefix="recompute-test-", dir=base)
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.serial = 0

    def store(self):
        self.serial += 1
        return RunStore(self.root / f"run_{self.serial}", {"mock_test": True}, allowed_root=self.root)

    def execute(self, spec=None, perturb=None, total=20):
        spec = spec or diagnostic.DiagnosticSpec(start_step=4, steps=4, probe_step=6, repeats=3, layout_length=4)
        stepper, recorded = MockStepper(spec.layout_length + 1, perturb), controls(total)
        with self.store() as store:
            result = diagnostic.diagnose_recomputation(stepper, initial_state(), recorded, spec, store)
        return result, stepper, recorded, store


class ArrayComparisonTests(unittest.TestCase):
    def test_known_difference_and_hashes(self):
        a = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = a + np.asarray([[0, 1], [-2, 3]], dtype=np.float32)
        row = diagnostic.array_comparison(a, b)
        self.assertFalse(row["bit_equal"])
        self.assertEqual(row["max_abs_difference"], 3.0)
        self.assertAlmostEqual(row["rms_difference"], np.sqrt(14 / 4))
        self.assertEqual(row["different_elements"], 3)
        self.assertEqual(row["first_different_indices"], [[0, 1], [1, 0], [1, 1]])
        self.assertNotEqual(row["expected"]["sha256"], row["actual"]["sha256"])
        self.assertEqual(diagnostic.array_identity(a), diagnostic.array_identity(np.asfortranarray(a)))

    def test_bit_equality_distinguishes_signed_zero(self):
        a, b = np.asarray([0.0], dtype=np.float32), np.asarray([-0.0], dtype=np.float32)
        self.assertTrue(np.array_equal(a, b))
        row = diagnostic.array_comparison(a, b)
        self.assertFalse(row["bit_equal"])
        self.assertEqual(row["max_abs_difference"], 0.0)
        self.assertEqual(row["different_elements"], 1)

    def test_dimensions_dtype_nonfinite_scalar_and_empty(self):
        a = np.ones((2, 3), dtype=np.float32)
        for b in (a.astype(np.float64), a.T):
            row = diagnostic.array_comparison(a, b)
            self.assertFalse(row["same_dimensions_and_dtype"])
            self.assertFalse(row["bit_equal"])
            self.assertIsNone(row["max_abs_difference"])
        nan = diagnostic.array_comparison(np.array([np.nan]), np.array([np.nan]))
        self.assertFalse(nan["finite"])
        self.assertIsNone(nan["rms_difference"])
        json.dumps(nan, allow_nan=False)
        scalar = diagnostic.array_comparison(np.array(2.0), np.array(3.0))
        self.assertEqual(scalar["expected"]["shape"], [])
        self.assertEqual(scalar["max_abs_difference"], 1.0)
        self.assertTrue(diagnostic.array_comparison(np.array([]), np.array([]))["bit_equal"])

    def test_count_schema_and_missing_counts(self):
        self.assertEqual(diagnostic.count_record({"n": np.int64(4)}), {"n": 4})
        row = diagnostic.count_comparison({"a": 0}, {"b": 0})
        self.assertFalse(row["equal"])
        self.assertEqual(row["differences"], {"a": {"expected": 0, "actual": None},
                                              "b": {"expected": None, "actual": 0}})
        for values in ({"a": 1.5}, {"a": True}, {1: 0}, []):
            with self.assertRaises(ValueError):
                diagnostic.count_record(values)


class RecomputeCoreTests(TemporaryRunTest):
    def test_identical_segments_and_exact_restore_all_arrays(self):
        result, stepper, recorded, store = self.execute()
        self.assertEqual(result["status"], "all_recorded_comparisons_equal")
        self.assertEqual(result["purpose"], diagnostic.PURPOSE)
        self.assertFalse(result["gradient_computed"])
        self.assertIn("not contact-particle identities", result["count_scope"])
        self.assertEqual(result["saved_original_states"], 5)
        self.assertEqual(len(stepper.advance_calls), 8 + 3 * 4 + 3)
        for row in result["segment_repeats"]:
            self.assertTrue(row["restoration"]["bit_equal"])
            self.assertTrue(row["all_state_bit_equal"])
            self.assertEqual(set(row["restoration"]["arrays"]), set(STATE_NAMES))
            self.assertEqual(row["state_max_abs_differences"], dict.fromkeys(STATE_NAMES, 0.0))
        for row in result["probe_repeats"]:
            self.assertIsNone(row["first_differing_stage"])
            self.assertTrue(row["all_intermediates_bit_equal"])
            self.assertEqual(set(row["intermediates"]), set(diagnostic.SCRATCH_FIELDS))
        for step, slot, control, _ in stepper.advance_calls:
            self.assertEqual(slot, step % 4)
            self.assertIs(control, recorded[step])
        manifest = store.read_json("original_segment.json")
        self.assertEqual(set(manifest["states"]), {str(i) for i in range(4, 9)})
        self.assertEqual(set(manifest["counts_by_input_step"]), {str(i) for i in range(4, 8)})
        self.assertEqual(len(list((store.path / "original").glob("state_*.npz"))), 4)
        self.assertFalse(any("gradient" in path.name or "optimizer" in path.name for path in store.path.iterdir()))
        checkpoint = diagnostic.load_state(store, manifest["states"]["4"])
        starts = [call[3] for call in stepper.advance_calls if call[0] == 4]
        self.assertEqual(len(starts), 4)
        for state in starts:
            self.assertTrue(diagnostic.state_comparison(checkpoint, state)["bit_equal"])

    def test_first_count_discrepancy_uses_global_input_step(self):
        def perturb(stepper, step, before, after, scratch, counts):
            if stepper.visits[step] > 1 and step >= 5:
                counts["particle_tool0"] += 1
        result, *_ = self.execute(perturb=perturb)
        self.assertEqual(result["status"], "mismatch_detected")
        for row in result["segment_repeats"]:
            self.assertTrue(row["all_state_bit_equal"])
            self.assertFalse(row["all_counts_equal"])
            first = row["first_count_discrepancy"]
            self.assertEqual(first["input_step"], 5)
            self.assertEqual(first["differences"], {"particle_tool0": {"expected": 2, "actual": 3}})
        self.assertEqual(result["probe_repeats"][0]["first_differing_stage"], "counts_only")

    def test_state_drift_with_equal_counts_and_original_probe_input(self):
        def perturb(stepper, step, before, after, scratch, counts):
            if stepper.visits[step] > 1 and step == 4:
                for value in after.arrays().values():
                    value += 0.125
        result, stepper, _, store = self.execute(perturb=perturb)
        for row in result["segment_repeats"]:
            self.assertTrue(row["all_counts_equal"])
            self.assertFalse(row["all_state_bit_equal"])
            self.assertEqual(row["first_state_bit_discrepancy"]["input_step"], 4)
            self.assertTrue(all(value > 0 for value in row["state_max_abs_differences"].values()))
        original_probe = next(before for step, _, _, before in stepper.advance_calls if step == 6)
        segment_probe_inputs = [before for step, _, _, before in stepper.advance_calls[:-3] if step == 6][1:]
        self.assertTrue(all(not diagnostic.state_comparison(original_probe, state)["bit_equal"]
                            for state in segment_probe_inputs))
        for step, slot, _, state in stepper.advance_calls[-3:]:
            self.assertEqual((step, slot), (6, 2))
            self.assertTrue(diagnostic.state_comparison(original_probe, state)["bit_equal"])
        self.assertTrue(all(row["first_differing_stage"] is None for row in result["probe_repeats"]))
        manifest = store.read_json("original_segment.json")
        self.assertTrue(diagnostic.state_comparison(original_probe, diagnostic.load_state(store, manifest["states"]["6"]))["bit_equal"])

    def test_unaligned_range_and_crossing_checkpoint_boundaries(self):
        spec = diagnostic.DiagnosticSpec(start_step=3, steps=7, probe_step=8, repeats=2, layout_length=4)
        result, stepper, _, _ = self.execute(spec)
        self.assertTrue(result["all_recorded_comparisons_equal"])
        expected = [(step, step % 4) for step in range(10)]
        expected += [(step, step % 4) for _ in range(2) for step in range(3, 10)]
        expected += [(8, 0), (8, 0)]
        self.assertEqual([(step, slot) for step, slot, *_ in stepper.advance_calls], expected)
        # Reads before the target consist only of its start input; prefix states are not retained.
        self.assertEqual([row for row in stepper.read_calls if row[0] < 4], [(3, 3)])
        self.assertEqual(result["saved_original_states"], 8)

    def test_prefix_downloads_only_checkpoint_boundaries(self):
        spec = diagnostic.DiagnosticSpec(start_step=13, steps=3, probe_step=14, repeats=1, layout_length=4)
        result, stepper, _, _ = self.execute(spec)
        prefix_reads = [row for row in stepper.read_calls if row[0] <= 13]
        self.assertEqual(prefix_reads, [(4, 4), (8, 4), (12, 4), (13, 1)])
        self.assertEqual(result["saved_original_states"], 4)

    def test_zero_start_final_probe_and_one_step_segment(self):
        for spec, total in ((diagnostic.DiagnosticSpec(0, 4, 3, 1, 4), 4),
                            (diagnostic.DiagnosticSpec(4, 1, 4, 1, 4), 8)):
            with self.subTest(spec=spec):
                result, _, _, _ = self.execute(spec, total=total)
                self.assertTrue(result["all_recorded_comparisons_equal"])

    def test_probe_distinguishes_material_from_grid_stages(self):
        for field, stage in (("trial", "material"), ("corrected", "material"),
                             ("history", "material"), ("rotation", "material"), ("affine", "material"),
                             ("grid_m", "grid_reduction"), ("grid_p", "grid_reduction"),
                             ("grid_u", "grid_normalization"), ("grid_v", "grid_operations")):
            with self.subTest(field=field):
                def perturb(stepper, step, before, after, scratch, counts):
                    if step == 6 and stepper.visits[step] > 1:
                        scratch[field].flat[0] += 0.25
                        if field in diagnostic.MATERIAL_FIELDS:
                            scratch["grid_p"].flat[0] += 0.5
                result, _, _, store = self.execute(perturb=perturb)
                self.assertTrue(all(row["all_state_bit_equal"] for row in result["segment_repeats"]))
                self.assertFalse(result["all_recorded_comparisons_equal"])
                original = diagnostic.load_arrays(store, store.read_json("original_segment.json")["probe_intermediates"])
                for row in result["probe_repeats"]:
                    self.assertEqual(row["first_differing_stage"], stage)
                    actual = diagnostic.load_arrays(store, row["intermediate_file"])
                    # The stored floating-point difference, not the intended real-number increment.
                    expected_difference = float(np.max(np.abs(actual[field] - original[field])))
                    self.assertEqual(row["intermediates"][field]["max_abs_difference"], expected_difference)
                    self.assertFalse(row["intermediates"][field]["bit_equal"])

    def test_failure_saves_exact_step_and_partial_original_manifest(self):
        def perturb(stepper, step, before, after, scratch, counts):
            if step == 6:
                raise RuntimeError("injected forward failure")
        spec = diagnostic.DiagnosticSpec(4, 4, 6, 1, 4)
        stepper = MockStepper(5, perturb)
        with self.store() as store:
            with self.assertRaisesRegex(RuntimeError, "injected forward failure"):
                diagnostic.diagnose_recomputation(stepper, initial_state(), controls(12), spec, store)
            failure = store.read_json("failure_step.json")
            manifest = store.read_json("original_segment.json")
        self.assertEqual(failure["input_step"], 6)
        self.assertEqual(failure["state_slot"], 2)
        self.assertEqual(failure["phase"], "original")
        self.assertEqual(set(manifest["states"]), {"4", "5", "6"})
        input_state = diagnostic.load_state(store, failure["input"])
        self.assertTrue(diagnostic.state_comparison(input_state, stepper.advance_calls[-1][3])["bit_equal"])
        self.assertIn("stale", failure["note"])

    def test_invalid_requests_do_not_advance_or_clamp(self):
        base = diagnostic.DiagnosticSpec(4, 4, 6, 1, 4)
        cases = [replace(base, start_step=-1), replace(base, start_step=True), replace(base, steps=0),
                 replace(base, repeats=0), replace(base, layout_length=0), replace(base, layout_length=100),
                 replace(base, probe_step=3), replace(base, probe_step=8), replace(base, steps=100),
                 replace(base, steps=1.5)]
        for spec in cases:
            with self.subTest(spec=spec), self.store() as store:
                stepper = MockStepper(5)
                with self.assertRaises(ValueError):
                    diagnostic.diagnose_recomputation(stepper, initial_state(), controls(20), spec, store)
                self.assertFalse(stepper.advance_calls)
        with self.assertRaises(ValueError):
            base.validate(20, capacity=8)

    def test_saved_array_integrity_and_fresh_output_refusal(self):
        with self.store() as store:
            record = diagnostic.save_arrays(store, "values.npz", {"x": np.arange(4)})
            np.testing.assert_array_equal(diagnostic.load_arrays(store, record)["x"], np.arange(4))
            with self.assertRaises(FileExistsError):
                diagnostic.save_arrays(store, "values.npz", {"x": np.zeros(4)})
            with self.assertRaises(ValueError):
                diagnostic.save_arrays(store, "../not-in-run.npz", {})
            with (store.path / "values.npz").open("ab") as stream:
                stream.write(b"modified")
            with self.assertRaisesRegex(ValueError, "changed"):
                diagnostic.load_arrays(store, record)
        with self.assertRaises(FileExistsError):
            RunStore(store.path, {"mock_test": True}, allowed_root=self.root)

    def test_memory_estimate_does_not_grow_with_prefix_length(self):
        a = diagnostic.DiagnosticSpec(4, 4, 6, 3, 4)
        b = diagnostic.DiagnosticSpec(4000, 4, 4002, 3, 4)
        self.assertEqual(diagnostic.diagnostic_memory_estimate(initial_state(), 8, a),
                         diagnostic.diagnostic_memory_estimate(initial_state(), 8, b))
        values = diagnostic.diagnostic_memory_estimate(initial_state(), 8, a)
        self.assertEqual(values["original_segment_state_disk_bytes_uncompressed"],
                         5 * sum(v.nbytes for v in initial_state().arrays().values()))


class RecomputeCliTests(TemporaryRunTest):
    def test_parser_defaults_and_explicit_flags(self):
        args = diagnostic.parse_args([])
        self.assertEqual((args.end_frame, args.start_step, args.steps, args.probe_step, args.repeats),
                         (60, 9728, 64, 9774, 3))
        self.assertEqual(args.reference_policy, "strict")
        args = diagnostic.parse_args(["--backend", "cuda", "--precision", "f32", "--reference-policy", "frozen",
                                      "--path", "episode=/mnt/episode", "--output-dir", "new-run"])
        self.assertEqual(args.path, ["episode=/mnt/episode"])
        self.assertEqual((args.backend, args.precision, args.reference_policy), ("cuda", "f32", "frozen"))
        for flags in (["--steps", "0"], ["--probe-step", "0"], ["--repeats", "0"], ["--cpu-threads", "0"]):
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    diagnostic.parse_args(flags)

    def fake_modules(self, perturb=None, runtime_error=None):
        config = types.SimpleNamespace(backend="cpu", simulation={"precision": "f64"}, seed=19,
                                       segment_length=4, validate=Mock())
        config.as_dict = lambda: {"backend": config.backend, "simulation": config.simulation.copy(),
                                  "seed": config.seed, "segment_length": config.segment_length}
        prepared = types.SimpleNamespace(total_steps=12, initial_state=initial_state(), controls=controls(12),
                                         simulation_config=types.SimpleNamespace(grid=8), parameters={"E": 1}, sdf=None,
                                         provenance={"input_hash": "mock-input-hash"}, fingerprint="mock-fingerprint")
        prepared.summary = lambda: {"mock_test": True, "provenance": prepared.provenance}
        stepper = MockStepper(5, perturb)
        runtime = Mock(side_effect=runtime_error) if runtime_error else Mock(return_value={
            "actual_arch": "mock-cuda", "initialization_verified": True,
            "forward_verified": False, "backward_verified": False})
        modules = {
            "config": {"load_config": Mock(return_value=config), "verify_input_paths": Mock(return_value={"mock": "verified"})},
            "reference_adapter": {"reference_policy": Mock(side_effect=lambda policy: nullcontext()),
                                  "verify_reference": Mock(return_value={"originals_unchanged": False, "mock_test": True})},
            "runtime": {"init_runtime": runtime},
            "data": {"prepare_experiment": Mock(return_value=prepared)},
            "solver": {"Stepper": Mock(return_value=stepper)},
        }
        loaded = {}
        for name, contents in modules.items():
            full_name = "experiments.differentiable_mpm." + name
            module = types.ModuleType(full_name)
            module.__dict__.update(contents)
            loaded[full_name] = module
        return loaded, modules, stepper

    def run_mock_cli(self, perturb=None, runtime_error=None, extra=()):
        self.serial += 1
        output = self.root / f"cli_{self.serial}"
        args = diagnostic.parse_args(["--output-dir", str(output), "--start-step", "4", "--steps", "4",
                                      "--probe-step", "6", "--repeats", "2", "--end-frame", "60",
                                      "--backend", "cuda", "--precision", "f32", "--reference-policy", "frozen", *extra])
        modules, records, stepper = self.fake_modules(perturb, runtime_error)
        source = {"sources": {"solver.py": "mock-stable-hash"}}
        with patch.dict(sys.modules, modules), patch.object(diagnostic, "source_identity", return_value=source), \
                patch.object(diagnostic, "RunStore", side_effect=lambda path, identity: RunStore(path, identity, allowed_root=self.root)), \
                patch.object(diagnostic, "capture_stdout", side_effect=lambda path: nullcontext()), \
                redirect_stdout(io.StringIO()):
            code = diagnostic.run(args)
        return code, output, records, stepper

    def test_cli_records_runtime_settings_hashes_without_gradients(self):
        code, output, records, stepper = self.run_mock_cli(extra=("--path", "episode=/mnt/episode"))
        self.assertEqual(code, 0)
        result = json.loads((output / "result.json").read_text())
        manifest = json.loads((output / "run_manifest.json").read_text())
        self.assertEqual(result["actual_backend"], "mock-cuda")
        self.assertEqual(result["purpose"], diagnostic.PURPOSE)
        self.assertFalse(result["runtime"]["backward_verified"])
        self.assertTrue(result["source_identity_unchanged_at_end"])
        self.assertEqual(manifest["identity"]["requested_runtime"]["seed"], 19)
        records["runtime"]["init_runtime"].assert_called_once_with("cuda", "f32", 1, False, 19)
        records["reference_adapter"]["reference_policy"].assert_called_once_with("frozen")
        records["config"]["load_config"].assert_called_once()
        self.assertEqual(records["config"]["load_config"].call_args.args[1], {"episode": "/mnt/episode"})
        self.assertEqual(records["solver"]["Stepper"].call_args.kwargs["capacity"], 5)
        self.assertEqual(len(stepper.advance_calls), 8 + 8 + 2)
        with np.load(output / "target_controls.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["input_steps"], [4, 5, 6, 7])
            np.testing.assert_array_equal(archive["times"], [4, 5, 6, 7])
        self.assertTrue((output / "input_verification.json").is_file())
        self.assertTrue((output / "prepared_provenance.json").is_file())

    def test_cli_failure_and_mismatch_exit_codes(self):
        code, output, records, stepper = self.run_mock_cli(runtime_error=RuntimeError("mock backend unavailable"))
        self.assertEqual(code, 1)
        result = json.loads((output / "result.json").read_text())
        self.assertEqual(result["status"], "diagnostic_failed")
        self.assertIn("mock backend unavailable", result["message"])
        self.assertFalse(stepper.advance_calls)
        records["data"]["prepare_experiment"].assert_not_called()

        def perturb(stepper, step, before, after, scratch, counts):
            if stepper.visits[step] > 1:
                counts["yielded"] += 1
        code, output, _, _ = self.run_mock_cli(perturb=perturb)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads((output / "result.json").read_text())["status"], "mismatch_detected")

    def test_stdout_tee_captures_python_and_native_descriptors(self):
        # A separate Python process tests descriptor handling, not a simulation/backend.
        output = self.root / "stdout.log"
        code = """
import os
from pathlib import Path
import sys
from experiments.differentiable_mpm.recompute_check import capture_stdout
with capture_stdout(Path(sys.argv[1])):
    print('python-stdout', flush=True)
    print('python-stderr', file=sys.stderr, flush=True)
    os.write(1, b'native-stdout\\n')
    os.write(2, b'native-stderr\\n')
print('after-capture', flush=True)
"""
        root = Path(__file__).resolve().parents[3]
        env = {**os.environ, "PYTHONPATH": str(root)}
        completed = subprocess.run([sys.executable, "-c", code, str(output)], cwd=root, env=env,
                                   capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        text = output.read_text()
        for line in ("python-stdout", "python-stderr", "native-stdout", "native-stderr"):
            self.assertIn(line, text)
            self.assertIn(line, completed.stdout)
        self.assertNotIn("after-capture", text)
        self.assertIn("after-capture", completed.stdout)


if __name__ == "__main__":
    unittest.main()
