#!/usr/bin/env python3
"""Materialize frame chunks as a differentiable-MPM calibration dataset.

The frame chunker creates aligned observation files only.  This stage adds an
initial reconstruction and an experiment configuration to every chunk, then
writes a dataset manifest whose shared parameters can be optimized across all
chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RECONSTRUCTOR = REPO_ROOT / "scripts" / "reconstruct_voxel_dough_from_deformpath.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run(command: list[str]) -> None:
    print("+ " + " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, check=True)


def existing(path: Path, name: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return path


def config_for_chunk(base: dict[str, Any], chunk_dir: Path, frame_count: int,
                     particles: Path, reconstruction_metadata: Path) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["name"] = f"{chunk_dir.name}-dynamics"
    paths = dict(config["paths"])
    paths.update({
        "episode": str(chunk_dir.resolve()),
        "calibration": str((chunk_dir / "scene_calibration_v2.json").resolve()),
        "initial_particles": str(particles.resolve()),
        "reconstruction_metadata": str(reconstruction_metadata.resolve()),
    })
    config["paths"] = paths
    config["expected_sha256"] = {
        name: sha256(Path(path)) for name, path in paths.items() if name != "episode"
    }
    # A chunk starts from its own observed geometry.  The voxel reconstruction
    # volume therefore belongs to that chunk; derive its mass from the fixed
    # density instead of inheriting the full-episode mass, which would make the
    # simulator reject the configuration as density-inconsistent.
    reconstruction = json.loads(reconstruction_metadata.read_text(encoding="utf-8"))
    volume = float(reconstruction["object_volume_m3"])
    config["mass_kg"] = float(config["density_kg_m3"]) * volume
    if frame_count < 3:
        raise ValueError(f"{chunk_dir.name} has only {frame_count} frames; at least three are required")
    train_end = frame_count - 2
    config["training"] = {"start_frame": 1, "end_frame": train_end, "stride": 1}
    config["validation"] = {"start_frame": frame_count - 1, "end_frame": frame_count - 1, "stride": 1}
    return config


def validation_chunk_names(rows: list[dict[str, Any]], requested: list[str]) -> set[str]:
    available = {Path(str(row["directory"])).name for row in rows}
    names = {name.strip() for item in requested for name in item.split(",") if name.strip()}
    unknown = sorted(names - available)
    if unknown:
        raise ValueError(f"Unknown validation chunks: {unknown}")
    if names and names == available:
        raise ValueError("At least one chunk must remain in training")
    return names


def materialize(args: argparse.Namespace) -> Path:
    chunks_dir = args.chunks_dir.expanduser().resolve()
    manifest_path = existing(chunks_dir / "chunk_manifest.json", "chunk manifest")
    base_config_path = existing(args.base_config, "base experiment config")
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("schema") != "taichidough/deformpath-frame-chunks/v1":
        raise ValueError("Unsupported chunk manifest schema")
    rows = document.get("chunks")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Chunk manifest contains no chunks")
    validation_chunks = validation_chunk_names(rows, args.validation_chunk)
    if args.overwrite and (chunks_dir / "dataset.json").exists():
        (chunks_dir / "dataset.json").unlink()
    elif (chunks_dir / "dataset.json").exists():
        raise FileExistsError(f"Dataset manifest exists; use --overwrite: {chunks_dir / 'dataset.json'}")

    episodes = []
    generated = []
    for row in rows:
        chunk_dir = chunks_dir / row["directory"]
        metadata_path = existing(chunk_dir / "sequence_metadata.json", f"{chunk_dir.name} metadata")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame_count = int(metadata["num_valid_samples"])
        if frame_count < 3:
            raise ValueError(f"{chunk_dir.name} has too few aligned frames: {frame_count}")
        calibration = existing(chunk_dir / "scene_calibration_v2.json", f"{chunk_dir.name} calibration")
        reconstruction_root = chunk_dir / "reconstruction"
        if args.overwrite and reconstruction_root.exists():
            shutil.rmtree(reconstruction_root)
        reconstruction_root.mkdir(parents=True, exist_ok=True)
        run([sys.executable, str(RECONSTRUCTOR), "--episode-dir", str(chunk_dir),
             "--pointclouds-name", "pointclouds_interpolated.pt", "--frame", "0",
             "--output-dir", str(reconstruction_root), "--num-particles", str(args.num_particles),
             "--voxel-size", str(args.voxel_size), "--fill-mode", "floor", "--fill-axis", "y",
             "--fill-direction", "negative", "--floor-clearance", str(args.floor_clearance),
             "--floor-min-thickness", str(args.floor_min_thickness),
             "--floor-max-thickness", str(args.floor_max_thickness),
             "--calibration", str(calibration), "--save-pt"])
        reconstruction_dir = reconstruction_root / chunk_dir.name / "frame_0000"
        particles = existing(reconstruction_dir / "sampled_particles_xyz.npy", f"{chunk_dir.name} particles")
        reconstruction_metadata = existing(reconstruction_dir / "reconstruction_metadata.json",
                                            f"{chunk_dir.name} reconstruction metadata")
        config = config_for_chunk(base_config, chunk_dir, frame_count, particles, reconstruction_metadata)
        config_path = chunk_dir / "differentiable_mpm.json"
        if config_path.exists() and not args.overwrite:
            raise FileExistsError(f"Config exists; use --overwrite: {config_path}")
        write_json(config_path, config)
        generated.append({"chunk": chunk_dir.name, "frames": frame_count,
                          "config": str(config_path.resolve()),
                          "config_sha256": sha256(config_path),
                          "particles": str(particles.resolve()),
                          "reconstruction_metadata": str(reconstruction_metadata.resolve())})
        episodes.append({
            "id": chunk_dir.name,
            "config": str(config_path.relative_to(chunks_dir)),
            "membership": "validation" if chunk_dir.name in validation_chunks else "training",
            "weight": 1.0,
            "scored_window": {"start_frame": 1, "end_frame": frame_count - 2, "stride": 1},
        })

    parameters = base_config["parameters"]
    shared_names = ["youngs_modulus", "poisson_ratio", "viscosity", "plastic_min", "plastic_max"]
    fit_names = [name for name in base_config.get("fit_parameters", shared_names) if name in shared_names]
    if not fit_names:
        raise ValueError("Base config must define at least one fitted material parameter")
    missing_bounds = [name for name in fit_names if name not in base_config.get("parameter_bounds", {})]
    if missing_bounds:
        raise ValueError(f"Base config has no bounds for fitted parameters: {missing_bounds}")
    bounds = {name: list(base_config["parameter_bounds"][name]) for name in fit_names}
    dataset = {
        "schema": "taichidough/differentiable-dataset/v1",
        "name": f"{chunks_dir.parent.name}-chunked",
        "shared_parameters": {
            "initial": {name: float(parameters[name]) for name in shared_names},
            "fit": fit_names,
            "bounds": bounds,
        },
        "tool_contact": {"retention": 1.0, "absorption": 0.0, "stickiness": 0.0},
        "episodes": episodes,
    }
    dataset_path = chunks_dir / "dataset.json"
    write_json(dataset_path, dataset)
    write_json(chunks_dir / "materialization_manifest.json", {
        "schema": "taichidough/deformpath-chunk-materialization/v1",
        "chunk_manifest": str(manifest_path),
        "base_config": str(base_config_path),
        "dataset": str(dataset_path),
        "num_particles": args.num_particles,
        "voxel_size_m": args.voxel_size,
        "validation_chunks": sorted(validation_chunks),
        "generated_chunks": generated,
    })
    return dataset_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True,
                        help="Existing full-episode smoke config used as the parameter/physics template")
    parser.add_argument("--num-particles", type=int, default=24000)
    parser.add_argument("--voxel-size", type=float, default=0.003)
    parser.add_argument("--floor-clearance", type=float, default=0.003)
    parser.add_argument("--floor-min-thickness", type=float, default=0.001)
    parser.add_argument("--floor-max-thickness", type=float, default=0.25)
    parser.add_argument("--validation-chunk", action="append", default=[], metavar="NAME[,NAME...]",
                        help="Chunk directory names to mark as held-out validation episodes. "
                             "Repeat or pass comma-separated names. Unlisted chunks remain training.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_particles < 1 or args.voxel_size <= 0:
        parser.error("num-particles must be positive and voxel-size must be positive")
    dataset = materialize(args)
    print(f"Dataset manifest: {dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
