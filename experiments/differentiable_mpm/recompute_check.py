"""Forward-only checkpoint/one-step repeatability diagnostic; no gradients or fitting.

Example: python -m experiments.differentiable_mpm.recompute_check \
    --reference-policy frozen --backend cuda --precision f32 --end-frame 60 \
    --start-step 9728 --steps 64 --probe-step 9774 --repeats 3 \
    --output-dir experiments/differentiable_mpm/runs/cuda_recompute_probe

Comparisons use exact bytes and exact aggregate counts, with numerical differences
reported separately. Matching aggregate counts do not establish equal contact IDs.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import traceback
from typing import Any
import uuid

import numpy as np

from .results import EXPERIMENT_ROOT, RUN_ROOT, RunStore, source_identity
from .state import ParticleState, STATE_NAMES


SCHEMA = "taichidough/recompute-diagnostic/v1"
PURPOSE = "forward diagnostic only, not gradient qualification"
MATERIAL_FIELDS = ("trial", "corrected", "history", "rotation", "affine")
GRID_FIELDS = ("grid_m", "grid_p", "grid_u", "grid_v")
SCRATCH_FIELDS = MATERIAL_FIELDS + GRID_FIELDS
COUNTS_NOTE = "Counts are aggregate contact/yield/floor summaries, not contact-particle identities."


@dataclass(frozen=True)
class DiagnosticSpec:
    start_step: int = 9728
    steps: int = 64
    probe_step: int = 9774
    repeats: int = 3
    layout_length: int = 64

    @property
    def stop_step(self):
        return self.start_step + self.steps

    def validate(self, total_steps, capacity):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.start_step < 0 or min(self.steps, self.repeats, self.layout_length) < 1:
            raise ValueError("Require start-step >= 0 and steps/repeats/layout length >= 1")
        if not self.start_step <= self.probe_step < self.stop_step:
            raise ValueError("probe-step must identify an input step inside the requested target segment")
        if self.stop_step > total_steps:
            raise ValueError(f"Requested segment ends at step {self.stop_step}, beyond prepared step {total_steps}")
        if self.layout_length > total_steps:
            raise ValueError("Configured segment length exceeds the prepared trajectory; choose an explicit compatible configuration")
        if capacity != self.layout_length + 1:
            raise ValueError("Stepper capacity must exactly preserve the configured checkpoint state layout")
        return self


def array_identity(array):
    array = np.asarray(array)
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}, sort_keys=True).encode()
    digest = hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()
    return {"dtype": array.dtype.str, "shape": list(array.shape), "sha256": digest}


def array_comparison(expected, actual):
    expected, actual = np.asarray(expected), np.asarray(actual)
    result: dict[str, Any] = {"expected": array_identity(expected), "actual": array_identity(actual)}
    same_dimensions = expected.shape == actual.shape and expected.dtype == actual.dtype
    finite = bool(np.isfinite(expected).all() and np.isfinite(actual).all())
    result.update({"same_dimensions_and_dtype": same_dimensions, "finite": finite,
                   "expected_nonfinite_count": int((~np.isfinite(expected)).sum()),
                   "actual_nonfinite_count": int((~np.isfinite(actual)).sum()),
                   "bit_equal": False, "max_abs_difference": None, "rms_difference": None,
                   "different_elements": None, "first_different_indices": []})
    if same_dimensions:
        a, b = np.ascontiguousarray(expected), np.ascontiguousarray(actual)
        byte_a = a.view(np.uint8).reshape(-1, a.dtype.itemsize)
        byte_b = b.view(np.uint8).reshape(-1, b.dtype.itemsize)
        different = np.any(byte_a != byte_b, axis=1).reshape(expected.shape)
        result["bit_equal"] = bool(not different.any())
        result["different_elements"] = int(different.sum())
        result["first_different_indices"] = np.argwhere(different)[:8].tolist()
        if finite:
            with np.errstate(over="ignore", invalid="ignore"):
                delta = actual.astype(np.float64) - expected.astype(np.float64)
                maximum = float(np.max(np.abs(delta))) if delta.size else 0.0
                if np.isfinite(maximum):
                    result["max_abs_difference"] = maximum
                    result["rms_difference"] = (maximum * float(np.sqrt(np.mean((delta / maximum) ** 2)))
                                                if maximum else 0.0)
    return result


def state_comparison(expected, actual):
    arrays = {name: array_comparison(getattr(expected, name), getattr(actual, name)) for name in STATE_NAMES}
    return {"bit_equal": all(row["bit_equal"] for row in arrays.values()),
            "finite": all(row["finite"] for row in arrays.values()), "arrays": arrays}


def count_record(values):
    if not isinstance(values, dict):
        raise ValueError("Stepper diagnostics must be an integer count dictionary")
    result = {}
    for name, value in values.items():
        if not isinstance(name, str) or isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("Diagnostic counts must have string keys and integer values")
        result[name] = int(value)
    return result


def count_comparison(expected, actual):
    expected, actual = count_record(expected), count_record(actual)
    differences = {name: {"expected": expected.get(name), "actual": actual.get(name)}
                   for name in sorted(set(expected) | set(actual)) if expected.get(name) != actual.get(name)}
    return {"equal": not differences, "differences": differences,
            "expected": expected, "actual": actual, "scope": COUNTS_NOTE}


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_arrays(store, relative_path, arrays):
    path = store.path / relative_path
    if not path.resolve().is_relative_to(store.path):
        raise ValueError("Diagnostic arrays must stay inside the new run")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
    return {"file": relative_path, "file_sha256": _file_hash(path),
            "arrays": {name: array_identity(array) for name, array in arrays.items()}}


def load_arrays(store, record):
    path = store.path / record["file"]
    if _file_hash(path) != record["file_sha256"]:
        raise ValueError(f"Saved diagnostic arrays changed: {record['file']}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if {name: array_identity(array) for name, array in arrays.items()} != record["arrays"]:
        raise ValueError("Saved array contents do not match their recorded identities")
    return arrays


def save_state(store, relative_path, state):
    state.validate()
    return save_arrays(store, relative_path, state.arrays())


def load_state(store, record):
    return ParticleState(**load_arrays(store, record))


def capture_intermediates(stepper):
    return {name: np.asarray(getattr(stepper, name).to_numpy()).copy() for name in SCRATCH_FIELDS}


def _update_maxima(maxima, comparison):
    for name, row in comparison["arrays"].items():
        value = row["max_abs_difference"]
        if value is None:
            maxima[name] = None
        elif maxima[name] is not None:
            maxima[name] = max(maxima[name], value)


def _first_probe_stage(restoration, scratch, output, counts):
    if not restoration["bit_equal"]:
        return "restored_input"
    if any(not scratch[name]["bit_equal"] for name in MATERIAL_FIELDS):
        return "material"
    if any(not scratch[name]["bit_equal"] for name in ("grid_m", "grid_p")):
        return "grid_reduction"
    if not scratch["grid_u"]["bit_equal"]:
        return "grid_normalization"
    if not scratch["grid_v"]["bit_equal"]:
        return "grid_operations"
    if not output["bit_equal"]:
        return "particle_transfer_or_contact"
    if not counts["equal"]:
        return "counts_only"
    return None


def diagnostic_memory_estimate(initial_state, grid, spec):
    state_bytes = sum(array.nbytes for array in initial_state.arrays().values())
    scalar_bytes = initial_state.x.dtype.itemsize
    # Four 3x3 matrices plus Jp history; mass plus three grid vector fields.
    scratch_bytes = (37 * len(initial_state.x) + 10 * grid ** 3) * scalar_bytes
    return {"state_bytes": state_bytes, "probe_scratch_bytes": scratch_bytes,
            "device_state_primal_and_adjoint_bytes": 2 * (spec.layout_length + 1) * state_bytes,
            "original_segment_state_disk_bytes_uncompressed": (spec.steps + 1) * state_bytes,
            "probe_scratch_disk_bytes_uncompressed": (spec.repeats + 1) * scratch_bytes,
            "additional_saved_repeat_states_bytes_upper_bound": 3 * spec.repeats * state_bytes,
            "host_diagnostic_arrays_bytes_estimate": 8 * state_bytes + 3 * scratch_bytes,
            "notes": ["Original target states are streamed to disk; the forward prefix is not retained.",
                      "Prepared observations, controls, SDFs, solver scratch and allocator overhead are additional.",
                      "No gradients are computed, although Stepper allocates its usual adjoint fields."]}


def diagnose_recomputation(stepper, initial_state, controls, spec, store, emit=None):
    """Run the diagnostic using only forward/state methods; also usable with mocks."""
    spec.validate(len(controls), stepper.capacity)
    initial_state.validate()
    emit = emit or (lambda event: None)
    layout = spec.layout_length
    original = {}
    original_counts = {}
    probe_original = None
    execution_phase, repeat_index = "original", 0

    def advance(global_input, read_state=True):
        slot = global_input % layout
        try:
            stepper.advance(slot, controls[global_input])
            counts = count_record(stepper.diagnostics())
            state = stepper.state(slot + 1) if read_state else None
            if state is not None:
                state.validate()
            return state, counts
        except Exception as error:
            failure = {"phase": execution_phase, "repeat": repeat_index, "input_step": global_input,
                       "state_slot": slot, "exception": type(error).__name__, "message": str(error),
                       "purpose": PURPOSE,
                       "note": "Output and scratch may be incomplete or stale after a failed substep."}
            # Preserve whatever can still be downloaded; keep the original exception.
            for name, fetch in (("input", lambda: stepper.state(slot).arrays()),
                                ("output", lambda: stepper.state(slot + 1).arrays()),
                                ("intermediates", lambda: capture_intermediates(stepper))):
                try:
                    failure[name] = save_arrays(store, f"failure/{name}.npz", fetch())
                except Exception as download_error:
                    failure[name + "_save_error"] = str(download_error)
            store.write_json("failure_step.json", failure)
            raise

    def rollover(global_output, state):
        if global_output % layout == 0 and global_output < len(controls):
            stepper.load_state(0, state)

    def note(phase, **values):
        event = {"phase": phase, **values}
        store.append_event(event)
        emit(event)

    stepper.load_state(0, initial_state)
    note("original_prefix_start", stop_step=spec.stop_step, target_start=spec.start_step,
         target_steps=spec.steps, layout_length=layout)
    try:
        for global_input in range(spec.stop_step):
            if global_input == spec.start_step:
                checkpoint = stepper.state(global_input % layout)
                original[global_input] = save_state(store, "start_checkpoint.npz", checkpoint)
            output = global_input + 1
            state, counts = advance(global_input, read_state=(global_input >= spec.start_step or output % layout == 0))
            if global_input >= spec.start_step:
                original[output] = save_state(store, f"original/state_{output:08d}.npz", state)
                original_counts[global_input] = counts
            if global_input == spec.probe_step:
                probe_original = save_arrays(store, "probe/original_intermediates.npz", capture_intermediates(stepper))
            rollover(output, state)
            if output % 512 == 0 or output in (spec.start_step, spec.stop_step):
                note("original_prefix_progress", step=output, stop_step=spec.stop_step)
    finally:
        original_manifest = {"states": {str(k): v for k, v in original.items()},
                             "counts_by_input_step": {str(k): v for k, v in original_counts.items()},
                             "probe_intermediates": probe_original}
        store.write_json("original_segment.json", original_manifest)
    if probe_original is None or set(original) != set(range(spec.start_step, spec.stop_step + 1)):
        raise RuntimeError("Original diagnostic segment is incomplete")
    result = {"schema": SCHEMA, "purpose": PURPOSE, "settings": asdict(spec),
              "original_forward_steps": spec.stop_step, "prepared_control_steps": len(controls),
              "saved_original_states": len(original), "gradient_computed": False,
              "segment_repeats": [], "probe_repeats": [], "count_scope": COUNTS_NOTE,
              "notes": ["Readbacks can affect GPU scheduling; agreement here does not prove general determinism.",
                        "This tool records differences; it does not relax the calibration replay checks.",
                        "Exact bytes, including signed zeros, are compared without tolerances or epsilon.",
                        "First-differing-stage labels identify the earliest recorded difference, not a proven cause.",
                        "Equal input/material arrays with different grid_m/grid_p are compatible with differing P2G reduction order."]}
    checkpoint = load_state(store, original[spec.start_step])
    for repeat in range(spec.repeats):
        execution_phase, repeat_index = "segment_repeat", repeat + 1
        note("segment_repeat_start", repeat=repeat + 1, start_step=spec.start_step)
        stepper.load_state(spec.start_step % layout, checkpoint)
        restored = stepper.state(spec.start_step % layout)
        restoration = state_comparison(checkpoint, restored)
        row = {"repeat": repeat + 1, "restoration": restoration, "steps": [],
               "first_count_discrepancy": None, "first_state_bit_discrepancy": None,
               "all_state_bit_equal": restoration["bit_equal"], "all_counts_equal": True,
               "all_states_finite": restoration["finite"],
               "state_max_abs_differences": {name: 0.0 for name in STATE_NAMES}}
        if not restoration["bit_equal"]:
            row["restored_state_file"] = save_state(store, f"repeats/{repeat + 1:03d}_restored_state.npz", restored)
        for global_input in range(spec.start_step, spec.stop_step):
            state, counts = advance(global_input)
            output = global_input + 1
            comparison = state_comparison(load_state(store, original[output]), state)
            count_diff = count_comparison(original_counts[global_input], counts)
            item = {"input_step": global_input, "output_step": output,
                    "state": comparison, "counts": count_diff}
            row["steps"].append(item)
            _update_maxima(row["state_max_abs_differences"], comparison)
            row["all_state_bit_equal"] &= comparison["bit_equal"]
            row["all_states_finite"] &= comparison["finite"]
            row["all_counts_equal"] &= count_diff["equal"]
            if not count_diff["equal"] and row["first_count_discrepancy"] is None:
                row["first_count_discrepancy"] = {"input_step": global_input, **count_diff}
            if not comparison["bit_equal"] and row["first_state_bit_discrepancy"] is None:
                row["first_state_bit_discrepancy"] = {"input_step": global_input, "output_step": output,
                    "state": comparison,
                    "actual_state_file": save_state(store, f"repeats/{repeat + 1:03d}_first_different_state.npz", state)}
            rollover(output, state)
        result["segment_repeats"].append(row)
        store.write_json("comparison.json", {**result, "status": "running"})
        note("segment_repeat_complete", repeat=repeat + 1, all_state_bit_equal=row["all_state_bit_equal"],
             all_counts_equal=row["all_counts_equal"], first_count_discrepancy=row["first_count_discrepancy"],
             state_max_abs_differences=row["state_max_abs_differences"])

    original_probe_input = load_state(store, original[spec.probe_step])
    original_probe_output = load_state(store, original[spec.probe_step + 1])
    original_probe_scratch = load_arrays(store, probe_original)
    for repeat in range(spec.repeats):
        execution_phase, repeat_index = "probe_repeat", repeat + 1
        note("probe_repeat_start", repeat=repeat + 1, input_step=spec.probe_step)
        slot = spec.probe_step % layout
        stepper.load_state(slot, original_probe_input)
        restoration = state_comparison(original_probe_input, stepper.state(slot))
        state, counts = advance(spec.probe_step)
        scratch_arrays = capture_intermediates(stepper)
        scratch = {name: array_comparison(original_probe_scratch[name], scratch_arrays[name]) for name in SCRATCH_FIELDS}
        output = state_comparison(original_probe_output, state)
        count_diff = count_comparison(original_counts[spec.probe_step], counts)
        first = _first_probe_stage(restoration, scratch, output, count_diff)
        row = {"repeat": repeat + 1, "input_step": spec.probe_step, "state_slot": slot,
               "restoration": restoration, "intermediates": scratch, "output_state": output,
               "counts": count_diff, "first_differing_stage": first,
               "all_intermediates_bit_equal": all(v["bit_equal"] for v in scratch.values()),
               "all_intermediates_finite": all(v["finite"] for v in scratch.values()),
               "intermediate_file": save_arrays(store, f"probe/repeat_{repeat + 1:03d}_intermediates.npz", scratch_arrays),
               "output_state_file": save_state(store, f"probe/repeat_{repeat + 1:03d}_output_state.npz", state)}
        result["probe_repeats"].append(row)
        store.write_json("comparison.json", {**result, "status": "running"})
        note("probe_repeat_complete", repeat=repeat + 1, first_differing_stage=first,
             all_counts_equal=count_diff["equal"], output_bit_equal=output["bit_equal"])
    equal = (all(row["all_state_bit_equal"] and row["all_counts_equal"] and row["all_states_finite"]
                 for row in result["segment_repeats"])
             and all(row["first_differing_stage"] is None and row["all_intermediates_finite"]
                     and row["restoration"]["finite"] and row["output_state"]["finite"]
                     for row in result["probe_repeats"]))
    result["status"] = "all_recorded_comparisons_equal" if equal else "mismatch_detected"
    result["all_recorded_comparisons_equal"] = equal
    store.write_json("comparison.json", result)
    note("diagnostic_complete", status=result["status"], purpose=PURPOSE)
    return result


@contextmanager
def capture_stdout(path):
    """Tee native/Python stdout and stderr to a new log while retaining console output."""
    sys.stdout.flush()
    sys.stderr.flush()
    log = Path(path).open("xb", buffering=0)
    original_stdout, original_stderr = os.dup(1), os.dup(2)
    read_fd, write_fd = os.pipe()
    errors = []

    def copy_output():
        destinations = {"stdout.log": log.fileno(), "console": original_stdout}
        try:
            while True:
                block = os.read(read_fd, 65536)
                if not block:
                    break
                for name, fd in list(destinations.items()):
                    try:
                        view = memoryview(block)
                        while view:
                            written = os.write(fd, view)
                            view = view[written:]
                    except OSError as error:
                        errors.append(f"{name}: {error}")
                        del destinations[name]
                # Even if either destination fails, drain the pipe until writers close.
        except OSError as error:
            errors.append(str(error))
        finally:
            os.close(read_fd)

    worker = threading.Thread(target=copy_output, name="recompute-stdout", daemon=True)
    worker.start()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(original_stdout, 1)
        os.dup2(original_stderr, 2)
        worker.join()
        log.close()
        os.close(original_stdout)
        os.close(original_stderr)
        if errors:
            raise OSError("Diagnostic stdout capture failed: " + "; ".join(errors))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "State capacity is config.segment_length + 1; --steps does not change that setting. "
        "Exit codes: 0 = recorded comparisons equal, 2 = mismatch or source change, 1 = setup/execution failure. "
        "Equality is a forward diagnostic only, not gradient qualification."))
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs/episode18_viscoelastic.json")
    parser.add_argument("--path", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"))
    parser.add_argument("--precision", choices=("f32", "f64"))
    parser.add_argument("--reference-policy", choices=("strict", "frozen"), default="strict")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--end-frame", type=int, default=60)
    parser.add_argument("--start-step", type=int, default=9728)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--probe-step", type=int, default=9774)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if min(args.end_frame, args.steps, args.repeats, args.cpu_threads) < 1 or args.start_step < 0:
        parser.error("Require end-frame/steps/repeats/cpu-threads >= 1 and start-step >= 0")
    if not args.start_step <= args.probe_step < args.start_step + args.steps:
        parser.error("probe-step must be inside [start-step, start-step + steps)")
    return args


def run(args):
    # Keep diagnostic/mocked imports free of runtime initialization and dataset loading.
    from .config import load_config, verify_input_paths
    from .reference_adapter import reference_policy, verify_reference

    overrides = {}
    for item in args.path:
        if "=" not in item:
            raise ValueError("--path requires NAME=PATH")
        name, value = item.split("=", 1)
        if not name or not value or name in overrides:
            raise ValueError("Path overrides require distinct names and nonempty paths")
        overrides[name] = value
    config = load_config(args.config, overrides)
    if args.backend is not None:
        config.backend = args.backend
    if args.precision is not None:
        config.simulation["precision"] = args.precision
    config.validate()
    spec = DiagnosticSpec(args.start_step, args.steps, args.probe_step, args.repeats, config.segment_length)
    source = source_identity()
    identity = {"schema": SCHEMA, "purpose": PURPOSE, "configuration": config.as_dict(),
                "settings": asdict(spec), "prepared_end_frame": args.end_frame,
                "reference_policy": args.reference_policy, "source": source,
                "requested_runtime": {"backend": config.backend, "precision": config.simulation.get("precision", "f32"),
                                      "cpu_threads": args.cpu_threads, "debug": args.debug, "seed": config.seed},
                "actual_runtime_record": "runtime.json", "resume_supported": False,
                "stdout_capture": "Combined file descriptors 1/2 after RunStore creation; earlier config parsing and output buffered beyond the capture scope are excluded."}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output = args.output_dir or RUN_ROOT / f"recompute_{stamp}_{uuid.uuid4().hex[:6]}"
    with RunStore(output, identity) as store, capture_stdout(store.path / "stdout.log"):
        runtime = None
        try:
            print(PURPOSE.upper(), flush=True)
            print(f"Run directory: {store.path}", flush=True)
            print("Settings: " + json.dumps(asdict(spec), sort_keys=True), flush=True)
            with reference_policy(args.reference_policy):
                verification = verify_reference()
                store.write_json("reference_verification.json", verification)
                if verification.get("originals_unchanged") is False:
                    print("Working originals differ; explicit frozen policy uses the preserved physics.", flush=True)
                store.write_json("input_verification.json", verify_input_paths(config))
                from .runtime import init_runtime
                runtime = init_runtime(config.backend, config.simulation.get("precision", "f32"),
                                       args.cpu_threads, args.debug, config.seed)
                store.write_json("runtime.json", runtime)
                from .data import prepare_experiment
                prepared = prepare_experiment(config, end_frame=args.end_frame, build_sdf=True)
                spec.validate(prepared.total_steps, spec.layout_length + 1)
                store.write_json("resolved_inputs.json", prepared.summary())
                store.write_json("prepared_provenance.json", prepared.provenance)
                memory = diagnostic_memory_estimate(prepared.initial_state, prepared.simulation_config.grid, spec)
                store.write_json("memory_estimate.json", memory)
                print("Memory/disk estimate: " + json.dumps(memory, sort_keys=True), flush=True)
                print(f"Actual backend: {runtime['actual_arch']}; prepared steps={prepared.total_steps}; "
                      f"original forward stops at {spec.stop_step}.", flush=True)
                target_controls = {"poses": np.stack([prepared.controls[i].poses for i in range(spec.start_step, spec.stop_step)]),
                                   "velocities": np.stack([prepared.controls[i].velocities for i in range(spec.start_step, spec.stop_step)]),
                                   "times": np.asarray([prepared.controls[i].time for i in range(spec.start_step, spec.stop_step)]),
                                   "input_steps": np.arange(spec.start_step, spec.stop_step, dtype=np.int64)}
                store.write_json("target_controls.json", save_arrays(store, "target_controls.npz", target_controls))
                from .solver import Stepper
                stepper = Stepper(prepared.simulation_config, prepared.parameters, capacity=spec.layout_length + 1, sdf=prepared.sdf)

                def emit(event):
                    print("RECOMPUTE " + json.dumps(event, sort_keys=True), flush=True)

                result = diagnose_recomputation(stepper, prepared.initial_state, prepared.controls, spec, store, emit)
                result["runtime"] = {**runtime, "forward_verified": True, "backward_verified": False,
                                     "execution_scope": "Diagnostic prefix, repeated segment and probe only; no observation loss or backward kernels."}
                result["source_identity_unchanged_at_end"] = source_identity() == source
                result["actual_backend"] = runtime["actual_arch"]
                result["prepared_fingerprint"] = prepared.fingerprint
                store.write_json("runtime.json", result["runtime"])
                store.write_json("result.json", result)
                if not result["source_identity_unchanged_at_end"]:
                    print("WARNING: experiment source changed during the diagnostic; recorded comparisons need this qualification.", flush=True)
                    return 2
                return 0 if result["all_recorded_comparisons_equal"] else 2
        except BaseException as error:
            failure = {"schema": SCHEMA, "purpose": PURPOSE, "status": "diagnostic_failed", "gradient_computed": False,
                       "exception": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(),
                       "runtime": runtime, "settings": asdict(spec)}
            store.write_json("result.json", failure)
            store.append_event({"event": "failure", "exception": type(error).__name__, "message": str(error)})
            print(failure["traceback"], flush=True)
            return 130 if isinstance(error, KeyboardInterrupt) else 1


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        print(f"Recompute diagnostic could not start: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
