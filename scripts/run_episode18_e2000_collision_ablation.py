#!/usr/bin/env python3
"""Run a locked Episode 18 E=2000 Pa no-tool versus solid-SDF comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

# pyright: reportMissingImports=false
from sweep_replay_sdf_contact_padding import (
    collect_particle_metrics,
    file_sha256,
    replace_option,
    required_option,
    validate_baseline,
)


SCHEMA = "taichidough/episode18-collision-ablation-result/v1"
PROFILE_SCHEMA = "taichidough/episode18-collision-ablation/v1"
DEFAULT_TIMEOUT_S = 10_800.0


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {description}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def resolve(path_value: str, source: Path) -> Path:
    path = Path(path_value)
    return (path if path.is_absolute() else source.parent / path).resolve()


def validate_profile(profile_path: Path) -> tuple[dict[str, Any], list[str]]:
    profile = load_object(profile_path, "ablation profile")
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"Ablation profile schema must be {PROFILE_SCHEMA}")
    baseline_spec = profile.get("baseline_command")
    if not isinstance(baseline_spec, dict):
        raise ValueError("Ablation profile has no baseline_command object")
    baseline_path = resolve(str(baseline_spec.get("path", "")), profile_path)
    if file_sha256(baseline_path) != baseline_spec.get("sha256"):
        raise ValueError("Baseline command SHA-256 does not match the ablation profile")
    baseline = json.loads(baseline_path.read_text())
    if not isinstance(baseline, list) or not all(isinstance(value, str) for value in baseline):
        raise ValueError("Baseline command must be a JSON string array")
    fixed = validate_baseline(baseline)
    expectations = profile.get("fixed_expectations")
    if not isinstance(expectations, dict):
        raise ValueError("Ablation profile fixed_expectations must be an object")
    checks = {
        "youngs_modulus_pa": fixed["youngs_modulus_pa"],
        "grid": fixed["grid"],
        "particles": int(required_option(baseline, "--particles")),
        "dt_s": float(required_option(baseline, "--dt")),
        "replay_start_frame": int(required_option(baseline, "--replay-start-frame")),
        "replay_end_frame": int(required_option(baseline, "--replay-end-frame")),
        "replay_stride": int(required_option(baseline, "--replay-stride")),
        "tool_sdf_resolution": int(required_option(baseline, "--tool-sdf-resolution")),
        "tool_mesh_scale": float(required_option(baseline, "--tool-mesh-scale")),
        "object_mass_kg": float(required_option(baseline, "--object-mass-kg")),
        "density_kg_m3": float(required_option(baseline, "--density")),
    }
    for name, actual in checks.items():
        expected = expectations.get(name)
        if isinstance(actual, float):
            if not isinstance(expected, (int, float)) or not math.isclose(float(expected), actual, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Baseline {name} does not match the ablation profile")
        elif actual != expected:
            raise ValueError(f"Baseline {name} does not match the ablation profile")
    inputs = profile.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Ablation profile inputs must be an object")
    for name in ("initial_particles", "reconstruction_metadata", "tool_geometry", "solid_collision_manifest"):
        spec = inputs.get(name)
        if not isinstance(spec, dict):
            raise ValueError(f"Ablation profile inputs.{name} is missing")
        path = resolve(str(spec.get("path", "")), profile_path)
        if not path.is_file() or file_sha256(path) != spec.get("sha256"):
            raise ValueError(f"Ablation profile input {name} does not match its SHA-256")
    meshes = inputs.get("collision_meshes")
    if not isinstance(meshes, list) or [item.get("name") for item in meshes if isinstance(item, dict)] != ["UR5e_spathla", "gen3_spathla"]:
        raise ValueError("Ablation profile collision meshes must list UR5e_spathla then gen3_spathla")
    for mesh in meshes:
        path = resolve(str(mesh.get("path", "")), profile_path)
        if not path.is_file() or file_sha256(path) != mesh.get("sha256"):
            raise ValueError(f"Collision mesh {mesh.get('name')} does not match its SHA-256")
    return profile, baseline


def selected_skin(label: str, padding: float) -> dict[str, Any]:
    if not label.strip() or label.strip().lower() in {"auto", "automatic", "sweep-best"}:
        raise ValueError("--skin-label must name an explicit operator decision, not automatic selection")
    if not math.isfinite(padding) or padding < 0:
        raise ValueError("--tool-contact-padding-m must be finite and nonnegative")
    return {"label": label.strip(), "contact_padding_m": padding, "selection": "operator_selected_assumption"}


def remove_flag(argv: Sequence[str], flag: str) -> list[str]:
    return [value for value in argv if value != flag]


def condition_argv(baseline: Sequence[str], condition: str, simulation_dir: Path, skin: Mapping[str, Any]) -> list[str]:
    if condition not in {"no_tool", "sdf"}:
        raise ValueError(f"Unknown ablation condition {condition}")
    argv = replace_option(baseline, "--tool-collision", "none" if condition == "no_tool" else "sdf")
    argv = replace_option(argv, "--output-dir", str(simulation_dir))
    if condition == "no_tool":
        return remove_flag(argv, "--record-sdf-contact-diagnostics")
    argv = replace_option(argv, "--tool-contact-padding", format(float(skin["contact_padding_m"]), ".17g"))
    return argv if "--record-sdf-contact-diagnostics" in argv else [*argv, "--record-sdf-contact-diagnostics"]


def run(argv: Sequence[str], stdout: Path, stderr: Path, timeout_s: float) -> dict[str, Any]:
    started = time.monotonic()
    try:
        with stdout.open("w") as out, stderr.open("w") as err:
            completed = subprocess.run(argv, stdout=out, stderr=err, check=False, timeout=timeout_s)
        return {"returncode": completed.returncode, "duration_s": time.monotonic() - started, "failure_reason": None if completed.returncode == 0 else f"command exited with status {completed.returncode}"}
    except subprocess.TimeoutExpired:
        return {"returncode": None, "duration_s": time.monotonic() - started, "failure_reason": f"command exceeded timeout of {timeout_s:g} seconds"}


def run_evaluator(baseline: Sequence[str], simulation_dir: Path, evaluation_dir: Path, timeout_s: float, stdout: Path, stderr: Path) -> tuple[list[str], dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    argv = [
        baseline[0], str(root / "scripts" / "evaluate_dynamic_topview_match.py"),
        "--episode-dir", required_option(baseline, "--replay-episode"),
        "--taichi-metadata", str(simulation_dir / "camera_parameters.json"),
        "--calibration", required_option(baseline, "--initial-particles-calibration"),
        "--frame-stride", required_option(baseline, "--replay-stride"),
        "--pair-tolerance", required_option(baseline, "--dt"),
        "--cell-size", "0.003", "--trim-quantile", "0.005", "--output-dir", str(evaluation_dir),
    ]
    return argv, run(argv, stdout, stderr, timeout_s)


def summarize_condition(condition: str, simulation_dir: Path, skin: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    metadata_path = simulation_dir / "camera_parameters.json"
    metadata = load_object(metadata_path, f"{condition} simulation metadata")
    parameters = metadata.get("parameters", {})
    replay = metadata.get("replay", {})
    if parameters.get("youngs_modulus") != 2000.0 or replay.get("source_start_frame") != 0 or replay.get("source_end_frame") != 387:
        raise ValueError(f"{condition} runtime metadata does not match locked E=2000 full replay")
    metrics = {"particles": collect_particle_metrics(metadata, 1e-6)}
    if condition == "sdf":
        if parameters.get("tool_collision") != "sdf" or not math.isclose(float(parameters.get("tool_contact_padding", math.nan)), float(skin["contact_padding_m"]), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("SDF runtime collision mode or selected skin does not match")
        solid = replay.get("solid_collision_assets")
        if not isinstance(solid, dict) or solid.get("sha256") != [mesh["sha256"] for mesh in profile["inputs"]["collision_meshes"]]:
            raise ValueError("SDF runtime solid collision assets do not match the locked profile")
        debug_path = simulation_dir / "replay_sdf_contact_debug.json"
        if not debug_path.is_file():
            raise ValueError("SDF condition did not write contact diagnostics")
        metrics["sdf_contact_debug"] = {"path": str(debug_path), "sha256": file_sha256(debug_path)}
    elif parameters.get("tool_collision") != "none":
        raise ValueError("No-tool runtime metadata does not use --tool-collision none")
    return {"metadata_path": str(metadata_path), "metadata_sha256": file_sha256(metadata_path), **metrics}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=root / "configs" / "episode18_e2000_collision_ablation.json")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "episode18_e2000_collision_ablation")
    parser.add_argument("--skin-label", required=True)
    parser.add_argument("--tool-contact-padding-m", required=True, type=float)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        raise ValueError("--timeout-s must be finite and positive")
    profile_path = args.profile.resolve()
    profile, baseline = validate_profile(profile_path)
    skin = selected_skin(args.skin_label, args.tool_contact_padding_m)
    root = args.output_dir.resolve()
    if root.exists():
        raise ValueError(f"Ablation output directory already exists: {root}")
    provenance = {"profile_path": str(profile_path), "profile_sha256": file_sha256(profile_path), "baseline_argv": baseline, "skin": skin}
    if args.validate_only:
        print(json.dumps({"schema": SCHEMA, "status": "validated_only", "provenance": provenance}, indent=2, allow_nan=False))
        return 0
    root.mkdir(parents=True)
    cases: list[dict[str, Any]] = []
    for condition in ("no_tool", "sdf"):
        case_dir = root / condition
        simulation_dir = case_dir / "simulation"
        evaluation_dir = case_dir / "evaluation"
        case_dir.mkdir(parents=True)
        simulator_argv = condition_argv(baseline, condition, simulation_dir, skin)
        write_json(case_dir / "simulator_command.json", {"argv": simulator_argv})
        simulator = run(simulator_argv, case_dir / "simulator.stdout.log", case_dir / "simulator.stderr.log", args.timeout_s)
        case: dict[str, Any] = {"condition": condition, "simulator": simulator, "status": "complete" if simulator["failure_reason"] is None else "failed"}
        if simulator["failure_reason"] is None:
            try:
                evaluator_argv, evaluator = run_evaluator(baseline, simulation_dir, evaluation_dir, args.timeout_s, case_dir / "evaluator.stdout.log", case_dir / "evaluator.stderr.log")
                write_json(case_dir / "evaluator_command.json", {"argv": evaluator_argv})
                case["evaluator"] = evaluator
                if evaluator["failure_reason"] is None:
                    case["metrics"] = summarize_condition(condition, simulation_dir, skin, profile)
                    report_path = evaluation_dir / "dynamic_topview_metrics.json"
                    case["dynamic_evaluation"] = {"path": str(report_path), "sha256": file_sha256(report_path), "summary": load_object(report_path, "dynamic evaluation").get("summary")}
                else:
                    case["status"] = "failed"
            except (OSError, ValueError) as exc:
                case["status"] = "failed"; case["failure_reason"] = str(exc)
        cases.append(case)
    report = {"schema": SCHEMA, "status": "complete" if all(case["status"] == "complete" for case in cases) else "failed", "provenance": provenance, "conditions": cases, "limitations": profile["limitations"]}
    write_json(root / "ablation_result.json", report)
    print(f"Wrote {root / 'ablation_result.json'}")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
