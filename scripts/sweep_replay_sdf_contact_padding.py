#!/usr/bin/env python3
"""Run and measure a fixed-parameter Episode 18 SDF contact-padding sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "taichidough/replay-contact-padding-sweep/v1"
CACHE_SCHEMA = "taichidough/replay-contact-padding-sweep-cache/v1"
DEBUG_SCHEMA = "taichidough/replay-sdf-contact-debug/v1"
TOOL_NAMES = ("UR5e_spathla", "gen3_spathla")
COUNTER_NAMES = ("contact_candidates", "applied_responses", "inward_normal_velocity_removed")
BRANCH_NAMES = ("grid_nodes", "particles")
DEFAULT_SOURCE_END_FRAME = 160
DEFAULT_FLOOR_TOLERANCE_M = 1e-6


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"Missing {description}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {description} {path}: {exc}") from exc


def require_object(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def require_argv(path: Path) -> list[str]:
    value = load_json(path, "source command")
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError("Source command must be a nonempty JSON string array")
    return list(value)


def option_values(argv: Sequence[str], option: str) -> list[str]:
    values: list[str] = []
    for index, value in enumerate(argv):
        if value == option:
            if index + 1 >= len(argv):
                raise ValueError(f"{option} has no value")
            values.append(argv[index + 1])
    return values


def required_option(argv: Sequence[str], option: str) -> str:
    values = option_values(argv, option)
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {option}, found {len(values)}")
    return values[0]


def parse_finite(value: str, option: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{option} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{option} must be a finite number")
    return parsed


def replace_option(argv: Sequence[str], option: str, value: str) -> list[str]:
    """Replace a valued option exactly once, retaining all unrelated arguments."""
    result: list[str] = []
    found = 0
    index = 0
    while index < len(argv):
        if argv[index] == option:
            if index + 1 >= len(argv):
                raise ValueError(f"{option} has no value")
            found += 1
            result.extend((option, value))
            index += 2
        else:
            result.append(argv[index])
            index += 1
    if found != 1:
        raise ValueError(f"Expected exactly one {option}, found {found}")
    return result


def ensure_flag(argv: Sequence[str], flag: str) -> list[str]:
    count = sum(item == flag for item in argv)
    if count > 1:
        raise ValueError(f"Expected at most one {flag}, found {count}")
    return list(argv) if count else [*argv, flag]


def padding_cases(grid: int) -> list[dict[str, Any]]:
    if grid <= 0:
        raise ValueError("grid must be positive")
    dx = 1.0 / grid
    cases = (
        ("padding_0dx", "0 dx", 0.0),
        ("padding_dx_8", "dx / 8", dx / 8.0),
        ("padding_dx_4", "dx / 4", dx / 4.0),
        ("padding_3dx_8", "3 dx / 8", 3.0 * dx / 8.0),
        ("padding_dx_2", "dx / 2", dx / 2.0),
    )
    values = [value for _, _, value in cases]
    if len(set(values)) != len(values) or not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("Padding cases must be unique nonnegative finite values")
    return [
        {
            "id": identifier,
            "label": label,
            "padding_scene_m": padding,
            "padding_over_dx": padding / dx,
        }
        for identifier, label, padding in cases
    ]


def validate_baseline(argv: Sequence[str]) -> dict[str, Any]:
    youngs_modulus = parse_finite(required_option(argv, "--youngs-modulus"), "--youngs-modulus")
    grid_value = required_option(argv, "--grid")
    try:
        grid = int(grid_value)
    except ValueError as exc:
        raise ValueError("--grid must be an integer") from exc
    if grid != 48:
        raise ValueError(f"Sweep requires --grid 48, found {grid}")
    if not math.isclose(youngs_modulus, 2000.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Sweep requires --youngs-modulus 2000, found {youngs_modulus:g}")
    if required_option(argv, "--tool-collision") != "sdf":
        raise ValueError("Sweep requires --tool-collision sdf")
    if int(required_option(argv, "--tool-sdf-resolution")) != 64:
        raise ValueError("Sweep requires --tool-sdf-resolution 64")
    if int(required_option(argv, "--replay-start-frame")) != 0:
        raise ValueError("Sweep requires --replay-start-frame 0")
    required_option(argv, "--replay-episode")
    for option in (
        "--initial-particles",
        "--initial-particles-metadata",
        "--initial-particles-calibration",
        "--tool-geometry",
        "--output-dir",
        "--replay-end-frame",
        "--replay-stride",
        "--tool-contact-padding",
    ):
        required_option(argv, option)
    return {"youngs_modulus_pa": youngs_modulus, "grid": grid, "dx_scene_m": 1.0 / grid}


def build_case_argv(baseline: Sequence[str], case: Mapping[str, Any], simulation_dir: Path) -> list[str]:
    argv = replace_option(baseline, "--replay-end-frame", str(DEFAULT_SOURCE_END_FRAME))
    argv = replace_option(argv, "--replay-stride", "1")
    argv = replace_option(argv, "--tool-collision", "sdf")
    argv = replace_option(argv, "--tool-contact-padding", format(float(case["padding_scene_m"]), ".17g"))
    argv = replace_option(argv, "--output-dir", str(simulation_dir))
    return ensure_flag(argv, "--record-sdf-contact-diagnostics")


def run_command(argv: Sequence[str], stdout_path: Path, stderr_path: Path, timeout_s: float) -> dict[str, Any]:
    started = time.monotonic()
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            completed = subprocess.run(list(argv), stdout=stdout, stderr=stderr, timeout=timeout_s, check=False)
        failure = None if completed.returncode == 0 else f"command exited with status {completed.returncode}"
        return {"returncode": completed.returncode, "duration_s": time.monotonic() - started, "failure_reason": failure}
    except subprocess.TimeoutExpired:
        return {"returncode": None, "duration_s": time.monotonic() - started, "failure_reason": f"command exceeded timeout of {timeout_s:g} seconds"}
    except OSError as exc:
        return {"returncode": None, "duration_s": time.monotonic() - started, "failure_reason": f"could not execute command: {exc}"}


def integer_counter(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} must be a nonnegative integer")
    return value


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def validate_and_aggregate_debug(debug: Mapping[str, Any], replay: Mapping[str, Any], expected_padding: float) -> dict[str, Any]:
    if debug.get("schema") != DEBUG_SCHEMA:
        raise ValueError("SDF contact debug schema is unsupported")
    if debug.get("tool_collision") != "sdf" or replay.get("tool_collision") != "sdf":
        raise ValueError("SDF contact debug and replay metadata must use sdf collision")
    names = debug.get("tool_names")
    if names != list(TOOL_NAMES) or replay.get("tool_names") != list(TOOL_NAMES):
        raise ValueError("SDF contact debug tool names do not match the expected tools")
    if debug.get("sequence_fingerprint") != replay.get("sequence_fingerprint"):
        raise ValueError("SDF contact debug sequence fingerprint does not match replay metadata")
    actual_padding = debug.get("tool_contact_padding_scene_m")
    if not isinstance(actual_padding, (int, float)) or not math.isclose(float(actual_padding), expected_padding, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("SDF contact debug padding does not match the requested padding")
    frames = debug.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("SDF contact debug contains no frames")
    seen_frames: set[int] = set()
    totals: dict[str, dict[str, dict[str, Any]]] = {
        name: {branch: {counter: 0 for counter in COUNTER_NAMES} for branch in BRANCH_NAMES}
        for name in TOOL_NAMES
    }
    intervals: list[dict[str, Any]] = []
    first_response: dict[str, dict[str, Any] | None] = {name: None for name in TOOL_NAMES}
    for row in frames:
        if not isinstance(row, dict):
            raise ValueError("SDF contact debug frame must be an object")
        frame = integer_counter(row.get("frame"), "debug frame")
        if frame in seen_frames:
            raise ValueError("SDF contact debug contains duplicate frames")
        seen_frames.add(frame)
        tools = row.get("tools")
        if not isinstance(tools, list) or len(tools) != len(TOOL_NAMES):
            raise ValueError("Every SDF contact debug frame must contain both tools")
        snapshot = row.get("snapshot_without_substep_evaluations")
        if not isinstance(snapshot, bool):
            raise ValueError("SDF contact debug frame must state whether it is a snapshot")
        interval_tools: dict[str, Any] = {}
        for tool_name, tool in zip(TOOL_NAMES, tools):
            if not isinstance(tool, dict):
                raise ValueError("SDF contact debug tool row must be an object")
            tool_branches: dict[str, dict[str, int]] = {}
            for branch in BRANCH_NAMES:
                counts = tool.get(branch)
                if not isinstance(counts, dict):
                    raise ValueError(f"SDF contact debug is missing {branch}")
                tool_branches[branch] = {counter: integer_counter(counts.get(counter), f"{tool_name}.{branch}.{counter}") for counter in COUNTER_NAMES}
                if not snapshot:
                    for counter, value in tool_branches[branch].items():
                        totals[tool_name][branch][counter] += value
            interval_tools[tool_name] = tool_branches
        interval = {
            "frame": frame,
            "source_frame": row.get("source_frame"),
            "sim_time_s": row.get("sim_time_s"),
            "snapshot_without_substep_evaluations": snapshot,
            "tools": interval_tools,
        }
        intervals.append(interval)
        if not snapshot:
            for tool_name in TOOL_NAMES:
                applied = sum(interval_tools[tool_name][branch]["applied_responses"] for branch in BRANCH_NAMES)
                if applied and first_response[tool_name] is None:
                    first_response[tool_name] = {"frame": frame, "source_frame": row.get("source_frame"), "sim_time_s": row.get("sim_time_s")}
    for name in TOOL_NAMES:
        for branch in BRANCH_NAMES:
            branch_total = totals[name][branch]
            branch_total["applied_per_candidate"] = ratio(branch_total["applied_responses"], branch_total["contact_candidates"])
            branch_total["inward_removed_per_applied"] = ratio(branch_total["inward_normal_velocity_removed"], branch_total["applied_responses"])
    return {"per_window": intervals, "totals_by_tool": totals, "first_applied_response": first_response}


def particle_statistics(initial: np.ndarray, current: np.ndarray, floor_y: float, floor_tolerance_m: float) -> dict[str, Any]:
    if initial.ndim != 2 or initial.shape[1:] != (3,) or current.shape != initial.shape:
        raise ValueError("Particle arrays must have matching (N, 3) shape")
    if not np.isfinite(initial).all() or not np.isfinite(current).all():
        raise ValueError("Particle arrays must contain only finite coordinates")
    delta = current - initial
    magnitudes = np.linalg.norm(delta, axis=1)
    min_y = float(np.min(current[:, 1]))
    below_floor = current[:, 1] < floor_y - floor_tolerance_m
    return {
        "particle_count": int(len(current)),
        "mean_displacement_m": float(np.mean(magnitudes)),
        "rms_displacement_m": float(np.sqrt(np.mean(np.square(magnitudes)))),
        "p50_displacement_m": float(np.quantile(magnitudes, 0.50)),
        "p95_displacement_m": float(np.quantile(magnitudes, 0.95)),
        "p99_displacement_m": float(np.quantile(magnitudes, 0.99)),
        "max_displacement_m": float(np.max(magnitudes)),
        "signed_mean_xyz_displacement_m": [float(value) for value in np.mean(delta, axis=0)],
        "centroid_displacement_xyz_m": [float(value) for value in np.mean(current, axis=0) - np.mean(initial, axis=0)],
        "centroid_displacement_m": float(np.linalg.norm(np.mean(current, axis=0) - np.mean(initial, axis=0))),
        "min_y_m": min_y,
        "particles_below_floor": int(np.count_nonzero(below_floor)),
        "maximum_floor_penetration_m": float(max(0.0, floor_y - min_y)),
    }


def collect_particle_metrics(metadata: Mapping[str, Any], floor_tolerance_m: float) -> dict[str, Any]:
    parameters = require_object(metadata.get("parameters"), "replay metadata parameters")
    floor_y = parameters.get("floor_y")
    if not isinstance(floor_y, (int, float)) or not math.isfinite(float(floor_y)):
        raise ValueError("Replay metadata floor_y must be finite")
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Replay metadata contains no frames")
    initial_path = Path(str(require_object(frames[0], "initial replay frame").get("particles")))
    initial = np.load(initial_path)
    per_frame: list[dict[str, Any]] = []
    for frame in frames:
        row = require_object(frame, "replay frame")
        particles = np.load(Path(str(row.get("particles"))))
        metrics = particle_statistics(initial, particles, float(floor_y), floor_tolerance_m)
        per_frame.append({"frame": row.get("frame"), "source_frame": row.get("source_frame"), "sim_time_s": row.get("sim_time_s"), **metrics})
    return {"floor_y_m": float(floor_y), "floor_tolerance_m": floor_tolerance_m, "per_frame": per_frame, "final": per_frame[-1]}


def validate_replay_metadata(metadata: Mapping[str, Any], case: Mapping[str, Any]) -> None:
    parameters = require_object(metadata.get("parameters"), "replay metadata parameters")
    replay = require_object(metadata.get("replay"), "replay metadata replay")
    required = {
        "youngs_modulus": 2000.0,
        "grid": 48,
        "replay_start_frame": 0,
        "replay_end_frame": DEFAULT_SOURCE_END_FRAME,
        "replay_stride": 1,
        "tool_collision": "sdf",
        "record_sdf_contact_diagnostics": True,
    }
    for key, expected in required.items():
        actual = parameters.get(key)
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Replay metadata parameter {key} does not match the sweep")
        elif actual != expected:
            raise ValueError(f"Replay metadata parameter {key} does not match the sweep")
    if not math.isclose(float(parameters.get("tool_contact_padding", math.nan)), float(case["padding_scene_m"]), rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("Replay metadata contact padding does not match the sweep")
    if replay.get("tool_collision") != "sdf" or replay.get("source_start_frame") != 0 or replay.get("source_end_frame") != DEFAULT_SOURCE_END_FRAME:
        raise ValueError("Replay metadata does not match the requested SDF source range")
    assets = require_object(replay.get("solid_collision_assets"), "solid collision assets")
    hashes = assets.get("sha256")
    if not isinstance(hashes, list) or len(hashes) != 2 or not all(isinstance(value, str) and len(value) == 64 for value in hashes):
        raise ValueError("Replay metadata does not contain two solid collision asset hashes")


def aggregate_case(simulation_dir: Path, case: Mapping[str, Any], floor_tolerance_m: float) -> dict[str, Any]:
    metadata_path = simulation_dir / "camera_parameters.json"
    debug_path = simulation_dir / "replay_sdf_contact_debug.json"
    metadata = require_object(load_json(metadata_path, "replay metadata"), "replay metadata")
    validate_replay_metadata(metadata, case)
    replay = require_object(metadata.get("replay"), "replay metadata replay")
    debug = require_object(load_json(debug_path, "SDF contact debug"), "SDF contact debug")
    return {
        "metadata_path": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "debug_path": str(debug_path),
        "debug_sha256": file_sha256(debug_path),
        "sequence_fingerprint": replay.get("sequence_fingerprint"),
        "solid_collision_assets": replay.get("solid_collision_assets"),
        "contact": validate_and_aggregate_debug(debug, replay, float(case["padding_scene_m"])),
        "particles": collect_particle_metrics(metadata, floor_tolerance_m),
    }


def total_particle_responses(case: Mapping[str, Any]) -> int:
    contact = case.get("metrics", {}).get("contact", {})
    return sum(int(contact.get("totals_by_tool", {}).get(name, {}).get("particles", {}).get("applied_responses", 0)) for name in TOOL_NAMES)


def rank_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    for case in cases:
        flags = case.setdefault("flags", [])
        if case.get("status") != "complete":
            flags.append("execution_or_validation_failure")
            continue
        final = case["metrics"]["particles"]["final"]
        if final["particles_below_floor"]:
            flags.append("floor_violation")
    ordered = sorted(
        cases,
        key=lambda item: (
            len(item.get("flags", [])),
            -total_particle_responses(item),
            item.get("metrics", {}).get("particles", {}).get("final", {}).get("p95_displacement_m", math.inf),
            item.get("metrics", {}).get("particles", {}).get("final", {}).get("max_displacement_m", math.inf),
            float(item["padding_scene_m"]),
        ),
    )
    for rank, case in enumerate(ordered, start=1):
        case["rank"] = rank
    return {
        "selection_status": "not_requested",
        "selected_padding_scene_m": None,
        "ranking_key": ["fewest_flags", "most_particle_applied_responses", "lowest_final_p95_displacement", "lowest_final_max_displacement", "smallest_padding"],
        "ordered_case_ids": [case["id"] for case in ordered],
    }


def git_provenance() -> dict[str, Any]:
    def command(*argv: str) -> str | None:
        completed = subprocess.run(argv, capture_output=True, text=True, check=False)
        return completed.stdout.strip() if completed.returncode == 0 else None
    return {"head": command("git", "rev-parse", "HEAD"), "status_porcelain": command("git", "status", "--short")}


def baseline_provenance(argv: Sequence[str], source_command_path: Path) -> dict[str, Any]:
    option_paths = ("--replay-episode", "--initial-particles", "--initial-particles-metadata", "--initial-particles-calibration", "--tool-geometry")
    inputs: dict[str, Any] = {}
    for option in option_paths:
        path = Path(required_option(argv, option))
        inputs[option[2:].replace("-", "_")] = {"path": str(path.resolve()), "sha256": file_sha256(path) if path.is_file() else None}
    manifest = Path(__file__).resolve().parents[1] / "meshes" / "tool_collision_meshes_v1.json"
    inputs["solid_collision_manifest"] = {"path": str(manifest), "sha256": file_sha256(manifest)}
    return {"source_command": str(source_command_path.resolve()), "source_command_sha256": file_sha256(source_command_path), "input_files": inputs, "git": git_provenance(), "python": sys.version, "numpy": np.__version__, "platform": platform.platform()}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def cache_directory(root: Path, payload_hash: str) -> Path:
    return root / "cache" / payload_hash / "attempt_000"


def execute_case(root: Path, baseline: Sequence[str], case: Mapping[str, Any], timeout_s: float, floor_tolerance_m: float, resume: bool, retry_failures: bool) -> dict[str, Any]:
    payload = {"schema": CACHE_SCHEMA, "baseline": list(baseline), "case": dict(case), "source_end_frame": DEFAULT_SOURCE_END_FRAME, "replay_stride": 1}
    key = sha256_bytes(canonical_json_bytes(payload))
    cache_dir = cache_directory(root, key)
    simulation_dir = cache_dir / "simulation"
    case_dir = root / "cases" / str(case["id"])
    argv = build_case_argv(baseline, case, simulation_dir)
    if case_dir.exists() and not resume:
        raise ValueError(f"Case report directory already exists: {case_dir}; use --resume")
    case_dir.mkdir(parents=True, exist_ok=True)
    write_json(case_dir / "command.json", {"argv": argv, "argv_sha256": sha256_bytes(canonical_json_bytes(argv)), "cache_key": key})
    metadata_path = cache_dir / "cache_metadata.json"
    if metadata_path.is_file():
        previous = require_object(load_json(metadata_path, "cache metadata"), "cache metadata")
        if previous.get("payload_hash") == key and previous.get("status") == "complete":
            try:
                metrics = aggregate_case(simulation_dir, case, floor_tolerance_m)
                result = {**case, "status": "complete", "cached": True, "cache_key": key, "cache_dir": str(cache_dir), "metrics": metrics, "flags": []}
                write_json(case_dir / "case_metrics.json", result)
                return result
            except (OSError, ValueError) as exc:
                return {**case, "status": "failed", "cached": True, "cache_key": key, "cache_dir": str(cache_dir), "failure_reason": f"cached output validation failed: {exc}", "flags": []}
        if previous.get("status") == "failed" and not retry_failures:
            return {**case, "status": "failed", "cached": True, "cache_key": key, "cache_dir": str(cache_dir), "failure_reason": previous.get("failure_reason", "previous cached case failed"), "flags": []}
        return {**case, "status": "failed", "cached": True, "cache_key": key, "cache_dir": str(cache_dir), "failure_reason": "cache directory exists but cannot be reused; remove it manually or use a new sweep root", "flags": []}
    if cache_dir.exists():
        return {**case, "status": "failed", "cached": False, "cache_key": key, "cache_dir": str(cache_dir), "failure_reason": "cache directory exists without readable metadata; it was preserved", "flags": []}
    simulation_dir.mkdir(parents=True)
    metadata: dict[str, Any] = {"schema": CACHE_SCHEMA, "payload_hash": key, "status": "running", "case": dict(case), "argv": argv, "started_at_unix_s": time.time()}
    write_json(metadata_path, metadata)
    execution = run_command(argv, cache_dir / "simulator.stdout.log", cache_dir / "simulator.stderr.log", timeout_s)
    failure = execution["failure_reason"]
    metrics = None
    if failure is None:
        try:
            metrics = aggregate_case(simulation_dir, case, floor_tolerance_m)
        except (OSError, ValueError) as exc:
            failure = f"output validation failed: {exc}"
    metadata.update({"status": "complete" if failure is None else "failed", "execution": execution, "failure_reason": failure, "completed_at_unix_s": time.time()})
    write_json(metadata_path, metadata)
    result = {**case, "status": "complete" if failure is None else "failed", "cached": False, "cache_key": key, "cache_dir": str(cache_dir), "failure_reason": failure, "flags": []}
    if metrics is not None:
        result["metrics"] = metrics
    write_json(case_dir / "case_metrics.json", result)
    return result


def write_summary_csv(path: Path, cases: Iterable[Mapping[str, Any]]) -> None:
    headers = ["rank", "id", "label", "padding_scene_m", "status", "cached", "flags", "particle_applied_responses", "final_mean_displacement_m", "final_p95_displacement_m", "final_max_displacement_m", "final_particles_below_floor", "failure_reason"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for case in cases:
            final = case.get("metrics", {}).get("particles", {}).get("final", {})
            writer.writerow({"rank": case.get("rank"), "id": case["id"], "label": case["label"], "padding_scene_m": case["padding_scene_m"], "status": case["status"], "cached": case["cached"], "flags": ";".join(case.get("flags", [])), "particle_applied_responses": total_particle_responses(case), "final_mean_displacement_m": final.get("mean_displacement_m"), "final_p95_displacement_m": final.get("p95_displacement_m"), "final_max_displacement_m": final.get("max_displacement_m"), "final_particles_below_floor": final.get("particles_below_floor"), "failure_reason": case.get("failure_reason")})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-command", type=Path, default=root / "output" / "episode18_e2000_full_replay_command.json")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "episode18_e2000_sdf_padding_sweep")
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--floor-tolerance-m", type=float, default=DEFAULT_FLOOR_TOLERANCE_M)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive and finite")
    if not math.isfinite(args.floor_tolerance_m) or args.floor_tolerance_m < 0:
        raise ValueError("--floor-tolerance-m must be nonnegative and finite")
    baseline = require_argv(args.source_command)
    fixed = validate_baseline(baseline)
    root = args.output_dir.resolve()
    if root.exists() and not args.resume:
        raise ValueError(f"Sweep output directory already exists: {root}; use --resume or choose a new directory")
    root.mkdir(parents=True, exist_ok=True)
    cases = [execute_case(root, baseline, case, args.timeout_s, args.floor_tolerance_m, args.resume, args.retry_failures) for case in padding_cases(fixed["grid"])]
    ranking = rank_cases(cases)
    report = {"schema": SCHEMA, "provenance": baseline_provenance(baseline, args.source_command), "sweep_spec": {**fixed, "source_frame_range_inclusive": [0, DEFAULT_SOURCE_END_FRAME], "tool_collision": "sdf", "replay_stride": 1, "record_sdf_contact_diagnostics": True, "floor_tolerance_m": args.floor_tolerance_m, "padding_cases": padding_cases(fixed["grid"])}, "acceptance_criteria": None, "cases": cases, "ranking": ranking}
    write_json(root / "padding_sweep.json", report)
    write_summary_csv(root / "summary.csv", cases)
    print(f"Wrote {root / 'padding_sweep.json'}")
    return 0 if all(case["status"] == "complete" for case in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
