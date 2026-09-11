#!/usr/bin/env python3
"""Sweep viscosity for the fixed Episode 18 SDF replay at one chosen tool friction."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

# pyright: reportMissingImports=false
from calibrate_youngs_modulus import score_evaluation_artifact
from sweep_episode18_tool_friction import (
    FIXED_TOOL_CONTACT_PADDING_M,
    FIXED_YOUNGS_MODULUS_PA,
    LOSS_SETTINGS,
    command_from_material_manifest,
    evaluator_argv,
    load_command,
    rebase_workspace_paths,
    remove_flag,
    replace_option,
    run,
    validate_baseline,
    write_json,
)

VISCOSITY_VALUES_PA_S = (0.0, 1.0, 2.5, 5.0, 10.0)
REPLAY_END_FRAME = 60
SCORING_START_FRAME = 1
SCORING_END_FRAME = 60


def viscosity_cases() -> list[dict[str, Any]]:
    return [
        {"id": f"viscosity_{value:g}".replace(".", "p"), "viscosity_pa_s": value}
        for value in VISCOSITY_VALUES_PA_S
    ]


def build_case_argv(baseline: Sequence[str], viscosity_pa_s: float, tool_contact_friction: float, simulation_dir: Path) -> list[str]:
    if viscosity_pa_s not in VISCOSITY_VALUES_PA_S:
        raise ValueError(f"Unsupported viscosity {viscosity_pa_s:g}")
    if not math.isfinite(tool_contact_friction) or tool_contact_friction < 0:
        raise ValueError("--tool-contact-friction must be finite and nonnegative")
    argv = replace_option(baseline, "--youngs-modulus", format(FIXED_YOUNGS_MODULUS_PA, ".12g"))
    argv = replace_option(argv, "--tool-contact-padding", format(FIXED_TOOL_CONTACT_PADDING_M, ".17g"))
    argv = replace_option(argv, "--tool-contact-friction", format(tool_contact_friction, ".12g"))
    argv = replace_option(argv, "--viscosity", format(viscosity_pa_s, ".12g"))
    argv = replace_option(argv, "--replay-stride", "1")
    argv = replace_option(argv, "--replay-end-frame", str(REPLAY_END_FRAME))
    argv = remove_flag(argv, "--cpu")
    return replace_option(argv, "--output-dir", str(simulation_dir))


def execute_case(root: Path, baseline: Sequence[str], case: Mapping[str, Any], tool_contact_friction: float, timeout_s: float, resume: bool, retry_failures: bool) -> dict[str, Any]:
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
    simulator = build_case_argv(baseline, float(case["viscosity_pa_s"]), tool_contact_friction, simulation_dir)
    evaluator = evaluator_argv(baseline, simulation_dir, evaluation_dir)
    write_json(case_dir / "simulator_command.json", {"argv": simulator})
    write_json(case_dir / "evaluator_command.json", {"argv": evaluator})
    simulation = run(simulator, case_dir / "simulator.stdout.log", case_dir / "simulator.stderr.log", timeout_s)
    result: dict[str, Any] = {**case, "tool_contact_friction": tool_contact_friction, "status": "failed", "simulator": simulation, "evaluator": None, "loss": None}
    if simulation["failure_reason"] is None:
        evaluation = run(evaluator, case_dir / "evaluator.stdout.log", case_dir / "evaluator.stderr.log", timeout_s)
        result["evaluator"] = evaluation
        if evaluation["failure_reason"] is None:
            try:
                result["loss"] = score_evaluation_artifact(
                    evaluation_dir / "dynamic_topview_metrics.json", LOSS_SETTINGS,
                    start_frame=SCORING_START_FRAME, end_frame=SCORING_END_FRAME,
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
        writer = csv.DictWriter(stream, fieldnames=["id", "viscosity_pa_s", "tool_contact_friction", "status", "weighted_total", "failure_reason"])
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "id": case["id"], "viscosity_pa_s": case["viscosity_pa_s"],
                "tool_contact_friction": case["tool_contact_friction"], "status": case["status"],
                "weighted_total": (case.get("loss") or {}).get("weighted_total"), "failure_reason": case.get("failure_reason"),
            })


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-command", type=Path, default=root / "output" / "episode18_e2000_full_replay_command.json")
    parser.add_argument("--material-manifest", type=Path, help="Use fixed simulator inputs from a prior material calibration")
    parser.add_argument("--tool-contact-friction", required=True, type=float, help="Fixed value selected before this viscosity sweep")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "episode18_e130579_dx8_viscosity_sweep")
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive and finite")
    if not math.isfinite(args.tool_contact_friction) or args.tool_contact_friction < 0:
        raise ValueError("--tool-contact-friction must be finite and nonnegative")
    baseline = command_from_material_manifest(args.material_manifest) if args.material_manifest else load_command(args.source_command)
    baseline = rebase_workspace_paths(baseline)
    validate_baseline(baseline)
    if args.validate_only:
        print(json.dumps({"status": "validated", "viscosity_values_pa_s": VISCOSITY_VALUES_PA_S, "tool_contact_friction": args.tool_contact_friction}, indent=2))
        return 0
    root = args.output_dir.resolve()
    if root.exists() and not args.resume:
        raise ValueError(f"Output directory already exists: {root}; use --resume")
    root.mkdir(parents=True, exist_ok=True)
    cases = [execute_case(root, baseline, case, args.tool_contact_friction, args.timeout_s, args.resume, args.retry_failures) for case in viscosity_cases()]
    report = {
        "schema": "taichidough/episode18-viscosity-sweep/v1",
        "fixed_parameters": {
            "youngs_modulus_pa": FIXED_YOUNGS_MODULUS_PA,
            "tool_contact_padding_m": FIXED_TOOL_CONTACT_PADDING_M,
            "tool_contact_friction": args.tool_contact_friction,
            "grid": 48, "tool_collision": "sdf", "tool_sdf_resolution": 64,
            "velocity_damping": 1.0, "plastic_velocity_damping": 1.0,
        },
        "training_source_frame_range_inclusive": [SCORING_START_FRAME, SCORING_END_FRAME],
        "cases": cases,
        "limitation": "This compares effective viscosity for this fixed Episode 18 numerical setup; it is not a universal material viscosity measurement.",
    }
    write_json(root / "viscosity_sweep.json", report)
    write_summary(root / "summary.csv", cases)
    print(f"Wrote {root / 'viscosity_sweep.json'}")
    return 0 if all(case["status"] == "complete" for case in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
