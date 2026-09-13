#!/usr/bin/env python3
"""Replay one dataset episode with saved material parameters and render a video."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.differentiable_mpm.dataset_config import MATERIAL_NAMES, load_dataset
from experiments.differentiable_mpm.parameters import PhysicalParameterSpace
from experiments.differentiable_mpm.reference_adapter import get_reference_modules, reference_policy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def executable(value: str, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty Python executable path")
    candidate = shutil.which(value) if os.sep not in value else value
    if candidate is None:
        raise ValueError(f"{name} executable was not found: {value}")
    path = Path(candidate).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{name} is not an executable file: {path}")
    return path


def path_overrides(values: list[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for text in values:
        try:
            key, raw_path = text.split("=", 1)
            episode_id, input_name = key.rsplit(".", 1)
        except ValueError as error:
            raise ValueError("Path overrides require EPISODE.INPUT=PATH") from error
        if not episode_id or not input_name or not raw_path or input_name in result.get(episode_id, {}):
            raise ValueError("Path overrides must be nonempty and unique")
        result.setdefault(episode_id, {})[input_name] = str(Path(raw_path).expanduser().resolve())
    return result


def material(values) -> dict[str, float]:
    if not isinstance(values, dict) or set(values) != set(MATERIAL_NAMES):
        raise ValueError("Material parameters must contain exactly the five shared material names")
    result = {name: float(values[name]) for name in MATERIAL_NAMES}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Material parameters must be finite")
    return result


def selection_parameters(path: Path, dataset) -> tuple[dict[str, float], dict]:
    record = json.loads(path.read_text())
    if record.get("schema") != "taichidough/dataset-material-selection/v1":
        raise ValueError("Expected a dataset selected_parameters.json file")
    if record.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("Selected parameters belong to a different dataset configuration")
    return material(record.get("shared_material_parameters")), {
        "kind": "selected_parameters", "path": str(path), "sha256": sha256(path),
        "training_value": record.get("training_value"),
        "ignore_recompute_mismatch": record.get("ignore_recompute_mismatch"),
    }


def optimizer_parameters(run_dir: Path, dataset, choice: str) -> tuple[dict[str, float], dict]:
    manifest_path = run_dir / "run_manifest.json"
    state_path = run_dir / "optimizer_state.json"
    if not manifest_path.is_file() or not state_path.is_file():
        raise ValueError("Active run requires run_manifest.json and optimizer_state.json")
    manifest = json.loads(manifest_path.read_text())
    identity = manifest.get("identity", {})
    if identity.get("action") != "dataset-fit":
        raise ValueError("Run directory is not a dataset fit")
    recorded = identity.get("dataset_identity", {})
    if recorded.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("Optimizer checkpoint belongs to a different dataset configuration")
    state_record = json.loads(state_path.read_text())
    if state_record.get("identity_sha256") != manifest.get("identity_sha256"):
        raise ValueError("Optimizer checkpoint identity does not match its run manifest")
    state = state_record.get("state", {})
    settings = state.get("space", {})
    space = PhysicalParameterSpace(settings.get("initial", {}), settings.get("fit", ()),
                                   settings.get("plasticity", "none"), bounds=settings.get("bounds"),
                                   scales=settings.get("scales"))
    coordinates = state.get("best_u") if choice == "best" else state.get("coordinates")
    if coordinates is None:
        raise ValueError(f"Optimizer checkpoint has no {choice} accepted parameter coordinates")
    selected = material({name: space.physical(coordinates)[name] for name in MATERIAL_NAMES})
    return selected, {
        "kind": "optimizer_checkpoint", "choice": choice, "run_dir": str(run_dir),
        "manifest_sha256": sha256(manifest_path), "optimizer_state_sha256": sha256(state_path),
        "saved_at": state_record.get("saved_at"), "iterations": state.get("iterations"),
        "accepted_updates": state.get("accepted_updates"), "evaluations": state.get("evaluations"),
        "objective_value": (state.get("current") or {}).get("value") if choice == "current" else state.get("best_value"),
        "ignore_recompute_mismatch": recorded.get("ignore_recompute_mismatch"),
    }


def parameter_source(args, dataset) -> tuple[dict[str, float], dict]:
    explicit_values = {
        "youngs_modulus": args.youngs_modulus, "poisson_ratio": args.poisson_ratio,
        "viscosity": args.viscosity, "plastic_min": args.plastic_min,
        "plastic_max": args.plastic_max,
    }
    has_explicit = any(value is not None for value in explicit_values.values())
    sources = int(args.run_dir is not None) + int(args.parameters is not None) + int(has_explicit)
    if sources != 1:
        raise ValueError("Choose exactly one parameter source: --run-dir, --parameters, or all five explicit material values")
    if has_explicit:
        if any(value is None for value in explicit_values.values()):
            raise ValueError("Explicit parameters require all five material arguments")
        return material(explicit_values), {"kind": "explicit_cli"}
    if args.parameters is not None:
        return selection_parameters(args.parameters.expanduser().resolve(), dataset)
    run_dir = args.run_dir.expanduser().resolve()
    selected = run_dir / "selected_parameters.json"
    if selected.is_file():
        values, provenance = selection_parameters(selected, dataset)
        provenance["run_dir"] = str(run_dir)
        return values, provenance
    return optimizer_parameters(run_dir, dataset, args.checkpoint_choice)


def execute(command, log_path: Path) -> int:
    print("Running: " + " ".join(map(str, command)), flush=True)
    with log_path.open("x") as log:
        process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--episode-id", required=True)
    result.add_argument("--run-dir", type=Path)
    result.add_argument("--parameters", type=Path)
    result.add_argument("--checkpoint-choice", choices=("best", "current"), default="best")
    result.add_argument("--youngs-modulus", type=float)
    result.add_argument("--poisson-ratio", type=float)
    result.add_argument("--viscosity", type=float)
    result.add_argument("--plastic-min", type=float)
    result.add_argument("--plastic-max", type=float)
    result.add_argument("--path", action="append", default=[], metavar="EPISODE.INPUT=PATH")
    result.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), default="cuda")
    result.add_argument("--precision", choices=("f32", "f64"), default="f32")
    result.add_argument("--cpu-threads", type=int, default=1)
    result.add_argument("--reference-policy", choices=("strict", "frozen"), default="frozen")
    result.add_argument("--end-frame", type=int)
    result.add_argument("--simulation-python", default=sys.executable)
    result.add_argument("--render-python", default=sys.executable)
    result.add_argument("--camera-zoom", type=float, default=1.0)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--prepare-only", action="store_true")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        sim_python = executable(args.simulation_python, "simulation-python")
        render_python = executable(args.render_python, "render-python")
        if args.cpu_threads < 1:
            raise ValueError("cpu-threads must be positive")
        if not math.isfinite(args.camera_zoom) or not 0.25 <= args.camera_zoom <= 4:
            raise ValueError("camera-zoom must be in [0.25,4]")
        overrides = path_overrides(args.path)
        dataset = load_dataset(args.dataset, path_overrides=overrides, backend=args.backend,
                               precision=args.precision)
        matches = [episode for episode in dataset.episodes if episode.id == args.episode_id]
        if len(matches) != 1:
            raise ValueError(f"Dataset does not contain episode {args.episode_id!r}")
        episode = matches[0]
        parameters, provenance = parameter_source(args, dataset)
        effective = episode.parameters_for(parameters)

        with reference_policy(args.reference_policy):
            modules = get_reference_modules(physics_version=episode.config.simulation.get("physics_version", "corrected-v1"))
            sequence = modules.dynamics.load_observation_sequence(episode.config.paths["episode"])
        end_frame = len(sequence.times) - 1 if args.end_frame is None else args.end_frame
        if not isinstance(end_frame, int) or not 1 <= end_frame < len(sequence.times):
            raise ValueError(f"end-frame must be in [1,{len(sequence.times) - 1}]")

        raw = episode.config.as_dict()
        for generated_field in ("config_path", "source_document_sha256", "path_overrides"):
            raw.pop(generated_field, None)
        raw["name"] = f"{episode.id}-forward-video"
        raw["parameters"] = effective
        raw["fit_parameters"] = list(dataset.fit_parameters)
        raw["parameter_bounds"] = {name: list(bounds) for name, bounds in dataset.parameter_bounds.items()}
        raw["backend"] = args.backend
        raw["simulation"]["precision"] = args.precision
        raw["training"] = {"start_frame": 1, "end_frame": end_frame, "stride": 1}
        raw["validation"] = {"start_frame": end_frame, "end_frame": end_frame, "stride": 1}

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = (args.output_dir or ROOT / "runs" / f"forward_video_{episode.id}_{stamp}_{uuid.uuid4().hex[:6]}").expanduser().resolve()
        if output == ROOT or not output.is_relative_to(ROOT):
            raise ValueError(f"output-dir must be inside {ROOT}")
        output.mkdir(parents=True, exist_ok=False)
        config_path = output / "forward_config.json"
        write_new_json(config_path, raw)
        manifest = {
            "schema": "taichidough/dataset-forward-video/v2", "dataset": str(args.dataset.expanduser().resolve()),
            "dataset_sha256": sha256(args.dataset.expanduser().resolve()), "dataset_fingerprint": dataset.fingerprint,
            "episode_id": episode.id, "episode_config": str(episode.config_path),
            "parameter_source": provenance, "parameters": parameters,
            "fixed_parameters": {name: effective[name] for name in effective if name not in MATERIAL_NAMES},
            "mass_kg": episode.config.mass_kg, "density_kg_m3": episode.config.density_kg_m3,
            "end_frame": end_frame, "available_frames": len(sequence.times),
            "backend": args.backend, "precision": args.precision, "reference_policy": args.reference_policy,
            "simulation_python": str(sim_python), "render_python": str(render_python),
            "camera_zoom": args.camera_zoom, "path_overrides": overrides,
            "simulation_requested": not args.prepare_only, "calibration_requested": False,
            "backward_requested": False,
        }
        write_new_json(output / "launcher_manifest.json", manifest)
        print(f"Output: {output}", flush=True)
        print(f"Simulation Python: {sim_python}", flush=True)
        print(f"Rendering Python: {render_python}", flush=True)
        if args.prepare_only:
            print("Prepared only; no simulation or rendering was run.", flush=True)
            return 0

        dependency = [render_python, "-c", "import numpy, scipy, skimage, pyvista; from PIL import Image; "
                      "import sys; sys.path.insert(0, " + repr(str(Path(__file__).resolve().parent)) + "); "
                      "from render_support import video_tools; print('Render dependencies OK:', video_tools())"]
        if execute(dependency, output / "render_dependencies.log"):
            raise RuntimeError("Rendering dependency check failed; no simulation was started")
        simulation = output / "simulation"
        forward = [sim_python, Path(__file__).with_name("forward.py"), "--config", config_path,
                   "--output-dir", simulation, "--backend", args.backend,
                   "--precision", args.precision, "--cpu-threads", str(args.cpu_threads),
                   "--reference-policy", args.reference_policy]
        code = execute(forward, output / "forward_stdout.log")
        if not (simulation / "simulation_result.json").is_file():
            raise RuntimeError(f"Forward process exited {code} without simulation_result.json")
        render = [render_python, Path(__file__).with_name("render.py"), "--run-dir", output,
                  "--camera-zoom", str(args.camera_zoom)]
        render_code = execute(render, output / "render_stdout.log")
        if render_code:
            raise RuntimeError("Rendering failed; saved simulation states can be rendered with recover.py")
        video = output / "perspective" / "requested_material_perspective.mp4"
        print(f"VIDEO: {video}", flush=True)
        return 0 if code == 0 else 2
    except (ValueError, OSError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
