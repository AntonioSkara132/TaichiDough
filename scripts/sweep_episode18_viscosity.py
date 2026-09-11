#!/usr/bin/env python3
"""Sweep viscosity for the fixed Episode 18 SDF replay at a chosen stiffness and tool friction."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

# pyright: reportMissingImports=false
from calibrate_youngs_modulus import score_evaluation_artifact
from material_calibration import canonical_json_hash, file_sha256, sequence_fingerprint
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
from sweep_replay_sdf_contact_padding import required_option

VISCOSITY_VALUES_PA_S = (0.0, 1.0, 2.5, 5.0, 10.0)
REPLAY_END_FRAME = 60
SCORING_START_FRAME = 1
SCORING_END_FRAME = 60


def viscosity_cases() -> list[dict[str, Any]]:
    return [
        {"id": f"viscosity_{value:g}".replace(".", "p"), "viscosity_pa_s": value}
        for value in VISCOSITY_VALUES_PA_S
    ]


def build_case_argv(
    baseline: Sequence[str], viscosity_pa_s: float, tool_contact_friction: float, simulation_dir: Path,
    youngs_modulus_pa: float = FIXED_YOUNGS_MODULUS_PA, cpu: bool | None = None,
) -> list[str]:
    if viscosity_pa_s not in VISCOSITY_VALUES_PA_S:
        raise ValueError(f"Unsupported viscosity {viscosity_pa_s:g}")
    if not math.isfinite(tool_contact_friction) or tool_contact_friction < 0:
        raise ValueError("--tool-contact-friction must be finite and nonnegative")
    if not math.isfinite(youngs_modulus_pa) or youngs_modulus_pa <= 0:
        raise ValueError("--youngs-modulus must be positive and finite")
    argv = replace_option(baseline, "--youngs-modulus", format(youngs_modulus_pa, ".12g"))
    argv = replace_option(argv, "--tool-contact-padding", format(FIXED_TOOL_CONTACT_PADDING_M, ".17g"))
    argv = replace_option(argv, "--tool-contact-friction", format(tool_contact_friction, ".12g"))
    argv = replace_option(argv, "--viscosity", format(viscosity_pa_s, ".12g"))
    argv = replace_option(argv, "--replay-start-frame", "0")
    argv = replace_option(argv, "--replay-stride", "1")
    argv = replace_option(argv, "--replay-end-frame", str(REPLAY_END_FRAME))
    if cpu is not None:
        argv = remove_flag(argv, "--cpu")
        if cpu:
            argv.append("--cpu")
    return replace_option(argv, "--output-dir", str(simulation_dir))


def file_record(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def case_fingerprint(simulator: Sequence[str], evaluator: Sequence[str]) -> dict[str, Any]:
    scripts = Path(__file__).resolve().parent
    source_files = {"simulator": file_record(Path(simulator[1])), "evaluator": file_record(Path(evaluator[1]))}
    for name in (
        "sweep_episode18_viscosity", "sweep_episode18_tool_friction", "sweep_replay_sdf_contact_padding",
        "calibrate_youngs_modulus", "material_calibration", "deformpath_dynamics", "deformpath_topview",
    ):
        source_files[name] = file_record(scripts / f"{name}.py")
    inputs = {
        option[2:].replace("-", "_"): file_record(Path(required_option(simulator, option)))
        for option in ("--initial-particles", "--initial-particles-metadata", "--initial-particles-calibration", "--tool-geometry")
    }
    for option, filename in (
        ("--ur-tool-mesh", "ur_spathla.stl"),
        ("--kinova-tool-mesh", "gen3_spathla.stl"),
        ("--ur-tool-collision-mesh", "ur_spathla_collision_solid.stl"),
        ("--kinova-tool-collision-mesh", "gen3_spathla_collision_solid.stl"),
    ):
        if option in simulator:
            path = Path(required_option(simulator, option))
        else:
            # Resolve sourced ROS assets as the simulator does; keep validation stdout as JSON.
            with contextlib.redirect_stdout(sys.stderr):
                from taichi_viscoelastic_mpm_scene import find_tool_mesh
            path = find_tool_mesh(filename, None)
        inputs[option[2:].replace("-", "_")] = file_record(path)
    inputs["collision_manifest"] = file_record(scripts.parent / "meshes" / "tool_collision_meshes_v1.json")
    episode = Path(required_option(simulator, "--replay-episode")).expanduser().resolve()
    metadata = episode / "sequence_metadata.json"
    sequence = {
        "episode_dir": str(episode), "fingerprint": sequence_fingerprint(episode),
        "metadata": file_record(metadata) if metadata.is_file() else None,
    }
    simulator_template = replace_option(simulator, "--output-dir", "{simulation_dir}")
    evaluator_template = replace_option(evaluator, "--output-dir", "{evaluation_dir}")
    evaluator_template = replace_option(evaluator_template, "--taichi-metadata", "{simulation_dir}/camera_parameters.json")
    return {
        "schema": "taichidough/episode18-viscosity-case-fingerprint/v1",
        "source_files": source_files, "input_files": inputs, "sequence": sequence,
        "simulator_argv": simulator_template, "evaluator_argv": evaluator_template,
        "youngs_modulus_pa": float(required_option(simulator, "--youngs-modulus")),
        "backend_request": "cpu" if "--cpu" in simulator else "gpu",
        "loss_settings": LOSS_SETTINGS,
        "scoring_source_frame_range_inclusive": [SCORING_START_FRAME, SCORING_END_FRAME],
    }


def validate_loss(loss: Mapping[str, Any]) -> None:
    if not loss.get("valid"):
        raise ValueError(f"invalid loss: {loss.get('failure_reason') or 'scored frames are invalid'}")
    total = loss.get("weighted_total")
    if isinstance(total, bool) or not isinstance(total, (int, float)) or not math.isfinite(total):
        raise ValueError("loss weighted_total must be finite")
    frames = [frame.get("source_frame") for frame in loss.get("frames", [])]
    if frames != list(range(SCORING_START_FRAME, SCORING_END_FRAME + 1)):
        raise ValueError("loss must contain every source frame from 1 through 60 exactly once")


def execute_case(
    root: Path, baseline: Sequence[str], case: Mapping[str, Any], tool_contact_friction: float,
    timeout_s: float, resume: bool, retry_failures: bool,
    youngs_modulus_pa: float = FIXED_YOUNGS_MODULUS_PA, cpu: bool | None = None,
) -> dict[str, Any]:
    case_dir = root / "cases" / str(case["id"])
    execution_dir = case_dir
    simulation_dir = execution_dir / "simulation"
    evaluation_dir = execution_dir / "evaluation"
    result_path = case_dir / "result.json"
    simulator = build_case_argv(baseline, float(case["viscosity_pa_s"]), tool_contact_friction, simulation_dir, youngs_modulus_pa, cpu)
    evaluator = evaluator_argv(baseline, simulation_dir, evaluation_dir)
    fingerprint = case_fingerprint(simulator, evaluator)
    cache_key = canonical_json_hash(fingerprint)
    if result_path.is_file():
        previous = json.loads(result_path.read_text())
        if previous.get("cache_key") != cache_key or previous.get("fingerprint") != fingerprint:
            raise ValueError(f"Case fingerprint is missing or differs: {result_path}; preserve it and use a new output directory")
        if previous.get("status") == "complete":
            if not resume:
                raise ValueError(f"Completed case already exists: {result_path}; use --resume, not --retry-failures")
            validate_loss(previous.get("loss") or {})
            output_files = previous.get("output_files") or {}
            if set(output_files) != {"simulation_metadata", "evaluation_metrics"}:
                raise ValueError(f"Completed case has no output fingerprints: {result_path}; use a new output directory")
            for record in output_files.values():
                if file_record(Path(record["path"])) != record:
                    raise ValueError(f"Completed case output differs: {record['path']}; use a new output directory")
            return previous
        if previous.get("status") != "failed" or not retry_failures:
            raise ValueError(f"Case result already exists: {result_path}; use --retry-failures only for failed cases")
        # A failed replay may have written frames; keep them and retry in an empty directory.
        attempt = 1
        while (case_dir / f"attempt_{attempt:03d}").exists():
            attempt += 1
        execution_dir = case_dir / f"attempt_{attempt:03d}"
        simulation_dir = execution_dir / "simulation"
        evaluation_dir = execution_dir / "evaluation"
        simulator = build_case_argv(baseline, float(case["viscosity_pa_s"]), tool_contact_friction, simulation_dir, youngs_modulus_pa, cpu)
        evaluator = evaluator_argv(baseline, simulation_dir, evaluation_dir)
    elif case_dir.exists():
        raise ValueError(f"Case directory exists without a result: {case_dir}; preserve it and use a new output directory")
    execution_dir.mkdir(parents=True, exist_ok=True)
    write_json(execution_dir / "simulator_command.json", {"argv": simulator, "cache_key": cache_key})
    write_json(execution_dir / "evaluator_command.json", {"argv": evaluator, "cache_key": cache_key})
    simulation = run(simulator, execution_dir / "simulator.stdout.log", execution_dir / "simulator.stderr.log", timeout_s)
    result: dict[str, Any] = {
        **case, "youngs_modulus_pa": fingerprint["youngs_modulus_pa"], "backend_request": fingerprint["backend_request"],
        "tool_contact_friction": tool_contact_friction, "status": "failed", "simulator": simulation,
        "evaluator": None, "loss": None, "cache_key": cache_key, "fingerprint": fingerprint,
        "simulation_dir": str(simulation_dir), "evaluation_dir": str(evaluation_dir),
    }
    if simulation["failure_reason"] is None:
        evaluation = run(evaluator, execution_dir / "evaluator.stdout.log", execution_dir / "evaluator.stderr.log", timeout_s)
        result["evaluator"] = evaluation
        if evaluation["failure_reason"] is None:
            try:
                loss = score_evaluation_artifact(
                    evaluation_dir / "dynamic_topview_metrics.json", LOSS_SETTINGS,
                    start_frame=SCORING_START_FRAME, end_frame=SCORING_END_FRAME,
                )
                validate_loss(loss)
                if case_fingerprint(simulator, evaluator) != fingerprint:
                    raise ValueError("source files or inputs changed during execution; use a new output directory")
                result["output_files"] = {
                    "simulation_metadata": file_record(simulation_dir / "camera_parameters.json"),
                    "evaluation_metrics": file_record(evaluation_dir / "dynamic_topview_metrics.json"),
                }
                result["loss"] = loss
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
        writer = csv.DictWriter(stream, fieldnames=["id", "viscosity_pa_s", "youngs_modulus_pa", "backend_request", "tool_contact_friction", "status", "weighted_total", "failure_reason"])
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "id": case["id"], "viscosity_pa_s": case["viscosity_pa_s"],
                "youngs_modulus_pa": case["youngs_modulus_pa"], "backend_request": case["backend_request"],
                "tool_contact_friction": case["tool_contact_friction"], "status": case["status"],
                "weighted_total": (case.get("loss") or {}).get("weighted_total"), "failure_reason": case.get("failure_reason"),
            })


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-command", type=Path, default=root / "output" / "episode18_e2000_full_replay_command.json")
    parser.add_argument("--material-manifest", type=Path, help="Use fixed simulator inputs from a prior material calibration")
    parser.add_argument("--youngs-modulus", type=float, default=FIXED_YOUNGS_MODULUS_PA, help="Fixed Young's modulus in Pa; defaults to the previous Episode 18 fit")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--cpu", dest="cpu", action="store_true", help="Request CPU simulation; otherwise preserve the source command's backend")
    backend.add_argument("--gpu", dest="cpu", action="store_false", help="Request Taichi's GPU selection by removing --cpu from the simulator command")
    parser.set_defaults(cpu=None)
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
    if not math.isfinite(args.youngs_modulus) or args.youngs_modulus <= 0:
        raise ValueError("--youngs-modulus must be positive and finite")
    baseline = command_from_material_manifest(args.material_manifest) if args.material_manifest else load_command(args.source_command)
    baseline = rebase_workspace_paths(baseline)
    validate_baseline(baseline)
    root = args.output_dir.resolve()
    if args.validate_only:
        commands = []
        fingerprint = None
        for case in viscosity_cases():
            simulation_dir = root / "cases" / case["id"] / "simulation"
            evaluation_dir = root / "cases" / case["id"] / "evaluation"
            simulator = build_case_argv(baseline, case["viscosity_pa_s"], args.tool_contact_friction, simulation_dir, args.youngs_modulus, args.cpu)
            evaluator = evaluator_argv(baseline, simulation_dir, evaluation_dir)
            fingerprint = case_fingerprint(simulator, evaluator)
            commands.append({**case, "simulator": simulator, "evaluator": evaluator, "cache_key": canonical_json_hash(fingerprint)})
        if fingerprint is None:
            raise ValueError("Viscosity sweep has no cases")
        print(json.dumps({
            "status": "validated", "viscosity_values_pa_s": VISCOSITY_VALUES_PA_S,
            "youngs_modulus_pa": fingerprint["youngs_modulus_pa"], "backend_request": fingerprint["backend_request"],
            "tool_contact_friction": args.tool_contact_friction, "source_files": fingerprint["source_files"],
            "input_files": fingerprint["input_files"], "sequence": fingerprint["sequence"], "commands": commands,
        }, indent=2))
        return 0
    if root.exists() and not args.resume:
        raise ValueError(f"Output directory already exists: {root}; use --resume")
    root.mkdir(parents=True, exist_ok=True)
    cases = [execute_case(root, baseline, case, args.tool_contact_friction, args.timeout_s, args.resume, args.retry_failures, args.youngs_modulus, args.cpu) for case in viscosity_cases()]
    report = {
        "schema": "taichidough/episode18-viscosity-sweep/v1",
        "fixed_parameters": {
            "youngs_modulus_pa": cases[0]["youngs_modulus_pa"], "backend_request": cases[0]["backend_request"],
            "tool_contact_padding_m": FIXED_TOOL_CONTACT_PADDING_M,
            "tool_contact_friction": args.tool_contact_friction,
            "grid": 48, "tool_collision": "sdf", "tool_sdf_resolution": 64,
            "velocity_damping": 1.0, "plastic_velocity_damping": 1.0,
        },
        "training_source_frame_range_inclusive": [SCORING_START_FRAME, SCORING_END_FRAME],
        "cases": cases,
        "limitation": "This compares effective viscosity for this fixed Episode 18 numerical setup; it is not a universal material viscosity measurement. backend_request records the requested mode, not the device selected by Taichi.",
    }
    write_json(root / "viscosity_sweep.json", report)
    write_summary(root / "summary.csv", cases)
    print(f"Wrote {root / 'viscosity_sweep.json'}")
    return 0 if all(case["status"] == "complete" for case in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
