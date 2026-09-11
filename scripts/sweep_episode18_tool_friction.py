#!/usr/bin/env python3
"""Sweep three tool-contact friction values for the fixed Episode 18 SDF replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

from calibrate_youngs_modulus import compile_cli_arguments, score_evaluation_artifact
# pyright: reportMissingImports=false
from sweep_replay_sdf_contact_padding import replace_option, required_option

FIXED_YOUNGS_MODULUS_PA = 130579.320726
FIXED_TOOL_CONTACT_PADDING_M = 1.0 / 384.0
FRICTION_VALUES = (0.1, 0.2, 0.3)
REPLAY_START_FRAME = 0
REPLAY_END_FRAME = 60
SCORING_START_FRAME = 1
SCORING_END_FRAME = 60
LOSS_SETTINGS = {
    "weights": {
        "depth_change": 1.0,
        "mask_iou": 1.0,
        "observed_to_simulation_distance": 1.0,
        "real_coverage": 1.0,
    },
    "depth_scale_m": 0.01,
    "distance_scale_m": 0.01,
    "huber_delta": 1.0,
    "min_common_pixels": 50,
    "min_observed_pixels": 50,
    "min_simulation_pixels": 50,
    "min_observed_points": 50,
    "min_simulation_points": 50,
    "nearest_chunk_size": 1024,
    "require_all_frames": True,
}


def load_command(path: Path) -> list[str]:
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError("Source command must be a nonempty JSON string array")
    return value


def command_from_material_manifest(path: Path) -> list[str]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "taichidough/material-calibration-manifest/v1":
        raise ValueError("--material-manifest must be a material calibration manifest")
    inputs = manifest.get("inputs", {})
    simulator_arguments = manifest.get("fixed_parameters", {}).get("simulator_arguments")
    commands = manifest.get("commands", {})
    required_inputs = ("simulator", "sequence", "initial_particles", "reconstruction_metadata", "calibration", "geometry")
    if not isinstance(simulator_arguments, dict) or not all(isinstance(inputs.get(name), dict) for name in required_inputs):
        raise ValueError("Material manifest has incomplete simulator inputs")
    return [
        str(commands.get("python", "python3")),
        str(inputs["simulator"].get("path", "")),
        *compile_cli_arguments(simulator_arguments),
        "--output-dir", "/unused",
        "--youngs-modulus", "2000",
        "--replay-episode", str(inputs["sequence"].get("episode_dir", "")),
        "--replay-start-frame", str(manifest.get("reconstruction_frame", 0)),
        "--replay-end-frame", "60",
        "--initial-particles", str(inputs["initial_particles"].get("path", "")),
        "--initial-particles-metadata", str(inputs["reconstruction_metadata"].get("path", "")),
        "--initial-particles-calibration", str(inputs["calibration"].get("path", "")),
        "--tool-geometry", str(inputs["geometry"].get("path", "")),
    ]


def rebase_workspace_paths(argv: Sequence[str], target_repo: Path | None = None) -> list[str]:
    """Map paths under the saved command's workspace onto this checkout's workspace."""
    if len(argv) < 2 or not Path(argv[1]).is_absolute():
        return list(argv)
    source_repo = Path(argv[1]).parent.parent
    source_workspace = source_repo.parent
    target_repo = (target_repo or Path(__file__).resolve().parents[1]).resolve()
    target_workspace = target_repo.parent
    result: list[str] = []
    for value in argv:
        path = Path(value)
        if not path.is_absolute():
            result.append(value)
            continue
        try:
            relative = path.relative_to(source_workspace)
        except ValueError:
            result.append(value)
        else:
            result.append(str(target_workspace / relative))
    return result


def validate_baseline(argv: Sequence[str]) -> None:
    if int(required_option(argv, "--grid")) != 48:
        raise ValueError("Friction sweep requires --grid 48")
    if required_option(argv, "--tool-collision") != "sdf":
        raise ValueError("Friction sweep requires --tool-collision sdf")
    if int(required_option(argv, "--tool-sdf-resolution")) != 64:
        raise ValueError("Friction sweep requires --tool-sdf-resolution 64")
    if int(required_option(argv, "--replay-start-frame")) != REPLAY_START_FRAME:
        raise ValueError("Friction sweep requires --replay-start-frame 0")
    for option in (
        "--youngs-modulus",
        "--tool-contact-padding",
        "--tool-contact-friction",
        "--replay-end-frame",
        "--replay-stride",
        "--output-dir",
        "--replay-episode",
        "--initial-particles-calibration",
    ):
        required_option(argv, option)


def friction_cases() -> list[dict[str, Any]]:
    return [
        {"id": f"friction_{value:.1f}".replace(".", "p"), "tool_contact_friction": value}
        for value in FRICTION_VALUES
    ]


def build_case_argv(baseline: Sequence[str], friction: float, simulation_dir: Path) -> list[str]:
    if friction not in FRICTION_VALUES:
        raise ValueError(f"Unsupported tool friction {friction:g}")
    argv = replace_option(baseline, "--youngs-modulus", format(FIXED_YOUNGS_MODULUS_PA, ".12g"))
    argv = replace_option(argv, "--tool-contact-padding", format(FIXED_TOOL_CONTACT_PADDING_M, ".17g"))
    argv = replace_option(argv, "--tool-contact-friction", format(friction, ".1f"))
    argv = replace_option(argv, "--replay-stride", "1")
    argv = replace_option(argv, "--replay-end-frame", str(REPLAY_END_FRAME))
    return replace_option(argv, "--output-dir", str(simulation_dir))


def evaluator_argv(baseline: Sequence[str], simulation_dir: Path, evaluation_dir: Path) -> list[str]:
    root = Path(__file__).resolve().parents[1]
    return [
        baseline[0],
        str(root / "scripts" / "evaluate_dynamic_topview_match.py"),
        "--episode-dir", required_option(baseline, "--replay-episode"),
        "--taichi-metadata", str(simulation_dir / "camera_parameters.json"),
        "--calibration", required_option(baseline, "--initial-particles-calibration"),
        "--frame-stride", "1",
        "--pair-tolerance", required_option(baseline, "--dt"),
        "--cell-size", "0.003",
        "--trim-quantile", "0.005",
        "--output-dir", str(evaluation_dir),
    ]


def run(argv: Sequence[str], stdout_path: Path, stderr_path: Path, timeout_s: float) -> dict[str, Any]:
    started = time.monotonic()
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            completed = subprocess.run(argv, stdout=stdout, stderr=stderr, check=False, timeout=timeout_s)
        return {
            "returncode": completed.returncode,
            "duration_s": time.monotonic() - started,
            "failure_reason": None if completed.returncode == 0 else f"command exited with status {completed.returncode}",
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "duration_s": time.monotonic() - started, "failure_reason": str(exc)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def execute_case(root: Path, baseline: Sequence[str], case: Mapping[str, Any], timeout_s: float, resume: bool, retry_failures: bool) -> dict[str, Any]:
    case_dir = root / "cases" / str(case["id"])
    simulation_dir = case_dir / "simulation"
    evaluation_dir = case_dir / "evaluation"
    result_path = case_dir / "result.json"
    if result_path.is_file():
        previous = json.loads(result_path.read_text())
        if resume and previous.get("status") == "complete":
            return previous
        if not retry_failures:
            raise ValueError(f"Case result already exists: {result_path}; use --resume for complete cases or --retry-failures to rerun failed cases")
    if case_dir.exists() and not retry_failures:
        raise ValueError(f"Case directory already exists: {case_dir}; use --resume or --retry-failures")
    case_dir.mkdir(parents=True, exist_ok=True)
    simulator = build_case_argv(baseline, float(case["tool_contact_friction"]), simulation_dir)
    evaluator = evaluator_argv(baseline, simulation_dir, evaluation_dir)
    write_json(case_dir / "simulator_command.json", {"argv": simulator})
    write_json(case_dir / "evaluator_command.json", {"argv": evaluator})
    simulation = run(simulator, case_dir / "simulator.stdout.log", case_dir / "simulator.stderr.log", timeout_s)
    result: dict[str, Any] = {**case, "status": "failed", "simulator": simulation, "evaluator": None, "loss": None}
    if simulation["failure_reason"] is None:
        evaluation = run(evaluator, case_dir / "evaluator.stdout.log", case_dir / "evaluator.stderr.log", timeout_s)
        result["evaluator"] = evaluation
        if evaluation["failure_reason"] is None:
            try:
                result["loss"] = score_evaluation_artifact(
                    evaluation_dir / "dynamic_topview_metrics.json",
                    LOSS_SETTINGS,
                    start_frame=SCORING_START_FRAME,
                    end_frame=SCORING_END_FRAME,
                )
                result["status"] = "complete"
            except (OSError, ValueError) as exc:
                result["failure_reason"] = f"could not score evaluation: {exc}"
        else:
            result["failure_reason"] = evaluation["failure_reason"]
    else:
        result["failure_reason"] = simulation["failure_reason"]
    write_json(result_path, result)
    return result


def write_summary(path: Path, cases: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "tool_contact_friction", "status", "weighted_total", "failure_reason"])
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "id": case["id"],
                "tool_contact_friction": case["tool_contact_friction"],
                "status": case["status"],
                "weighted_total": (case.get("loss") or {}).get("weighted_total"),
                "failure_reason": case.get("failure_reason"),
            })


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-command", type=Path, default=root / "output" / "episode18_e2000_full_replay_command.json")
    parser.add_argument("--material-manifest", type=Path, help="Use the fixed simulator inputs recorded by a prior material calibration")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "episode18_e130579_dx8_tool_friction_sweep")
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive and finite")
    baseline = command_from_material_manifest(args.material_manifest) if args.material_manifest else load_command(args.source_command)
    baseline = rebase_workspace_paths(baseline)
    validate_baseline(baseline)
    if args.validate_only:
        print(json.dumps({"status": "validated", "friction_values": FRICTION_VALUES}, indent=2))
        return 0
    root = args.output_dir.resolve()
    if root.exists() and not args.resume:
        raise ValueError(f"Output directory already exists: {root}; use --resume")
    root.mkdir(parents=True, exist_ok=True)
    cases = [execute_case(root, baseline, case, args.timeout_s, args.resume, args.retry_failures) for case in friction_cases()]
    report = {
        "schema": "taichidough/episode18-tool-friction-sweep/v1",
        "fixed_parameters": {
            "youngs_modulus_pa": FIXED_YOUNGS_MODULUS_PA,
            "tool_contact_padding_m": FIXED_TOOL_CONTACT_PADDING_M,
            "grid": 48,
            "tool_collision": "sdf",
            "tool_sdf_resolution": 64,
        },
        "training_source_frame_range_inclusive": [SCORING_START_FRAME, SCORING_END_FRAME],
        "cases": cases,
        "limitation": "This compares effective tool-contact friction for this fixed Episode 18 numerical setup; it is not a universal plastic friction coefficient.",
    }
    write_json(root / "friction_sweep.json", report)
    write_summary(root / "summary.csv", cases)
    print(f"Wrote {root / 'friction_sweep.json'}")
    return 0 if all(case["status"] == "complete" for case in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
