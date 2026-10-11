"""Reconstruct Episode20 frame-zero particles for all temporal v2 chunks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Sequence

import numpy as np

from experiments.differentiable_mpm.prepare_episode20_kugla_temporal import (
    DEFAULT_PRESERVED_V1,
    DEFAULT_OUTPUT as DEFAULT_TEMPORAL,
    EXPECTED_FRAME_COUNTS,
    canonical_json,
    load_json,
    replace_path_prefix,
    sha256_file,
    tree_sha256,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RECONSTRUCTOR = REPO / "scripts/reconstruct_voxel_dough_from_deformpath.py"
SCHEMA = "taichidough/episode20-reconstructions/v2"
PARTICLE_COUNT = 24000
GRID_SIZE = 48
VOXEL_SIZE_M = 0.003
MIN_VOLUME_M3 = 1.0e-6
MAX_VOLUME_M3 = 1.0e-3


def snapshot_files(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in paths}


def finite_json(path: Path) -> dict[str, Any]:
    value = load_json(path)
    json.dumps(value, allow_nan=False)
    return value


def validate_calibration(path: Path) -> dict[str, Any]:
    calibration = finite_json(path)
    if calibration.get("schema") != "taichidough/scene-calibration/v2":
        raise ValueError("Episode20 reconstruction requires scene-calibration v2")
    if calibration.get("source_frame") != "mocap" or calibration.get("scene_frame") != "table-aligned":
        raise ValueError("Episode20 reconstruction requires mocap input and table-aligned scene frames")
    transform = np.asarray(calibration.get("scene_from_source"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Episode20 reconstruction requires a finite table-alignment transform")
    if np.allclose(transform, np.eye(4), rtol=0, atol=1e-12):
        raise ValueError("Episode20 reconstruction must retain the measured table alignment")
    plane = np.asarray(calibration.get("floor_plane_scene"), dtype=np.float64)
    if plane.shape != (4,) or not np.allclose(plane, [0.0, 1.0, 0.0, 0.0], rtol=0, atol=1e-12):
        raise ValueError("Episode20 reconstruction requires y=0 in the table-aligned scene")
    provenance = calibration.get("provenance")
    alignment = provenance.get("table_alignment") if isinstance(provenance, dict) else None
    if not isinstance(alignment, dict) or alignment.get("previous_scene_frame") != "mocap":
        raise ValueError("Episode20 calibration is missing mocap-to-table alignment provenance")
    if alignment.get("source_tensors_transformed") is not False:
        raise ValueError("Episode20 source tensors must remain in mocap coordinates")
    return calibration


def reconstruction_command(
    python: str,
    reconstructor: Path,
    recording: Path,
    output_root: Path,
    calibration: Path,
) -> list[str]:
    return [
        python,
        str(reconstructor),
        "--episode-dir",
        str(recording),
        "--pointclouds-name",
        "pointclouds_interpolated.pt",
        "--frame",
        "0",
        "--output-dir",
        str(output_root),
        "--num-particles",
        str(PARTICLE_COUNT),
        "--voxel-size",
        str(VOXEL_SIZE_M),
        "--fill-mode",
        "floor",
        "--fill-axis",
        "y",
        "--fill-direction",
        "negative",
        "--floor-clearance",
        "0.003",
        "--floor-min-thickness",
        "0.001",
        "--floor-max-thickness",
        "0.25",
        "--bbox-padding",
        "0.006",
        "--trim-quantile",
        "0.005",
        "--footprint-dilate",
        "1",
        "--seed",
        "0",
        "--calibration",
        str(calibration),
    ]


def validate_reconstruction(
    frame_dir: Path,
    calibration_path: Path,
    final_frame_dir: Path,
) -> dict[str, Any]:
    required_arrays = (
        "source_filtered_points_xyz.npy",
        "scene_transformed_points_xyz.npy",
        "real_points_xyz.npy",
        "voxel_centers_xyz.npy",
        "surface_particles_xyz.npy",
        "sampled_particles_xyz.npy",
    )
    metadata_path = frame_dir / "reconstruction_metadata.json"
    for name in (*required_arrays, metadata_path.name):
        if not (frame_dir / name).is_file():
            raise FileNotFoundError(f"Missing reconstruction output: {frame_dir / name}")
    arrays = {name: np.load(frame_dir / name, allow_pickle=False) for name in required_arrays}
    for name, array in arrays.items():
        if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite [N,3] values")
    particles = arrays["sampled_particles_xyz.npy"]
    if particles.shape != (PARTICLE_COUNT, 3):
        raise ValueError(f"Expected {PARTICLE_COUNT} reconstructed particles")
    horizontal = particles[:, (0, 2)]
    if np.any(horizontal < -1e-6):
        raise ValueError("Reconstructed particles lie outside the horizontal MPM domain")
    stencil_base = np.floor(particles * GRID_SIZE - 0.5).astype(np.int64)
    if np.any(stencil_base < -1) or np.any(stencil_base + 2 >= GRID_SIZE):
        raise ValueError("Reconstructed particles have an unsafe corrected-v1 MPM grid stencil")
    minimum_floor_distance = float(particles[:, 1].min())
    if minimum_floor_distance < -(VOXEL_SIZE_M / 2.0 + 1e-6):
        raise ValueError("Reconstructed particles extend too far below the calibrated floor")

    metadata = finite_json(metadata_path)
    volume = float(metadata.get("object_volume_m3", float("nan")))
    if not np.isfinite(volume) or not MIN_VOLUME_M3 <= volume <= MAX_VOLUME_M3:
        raise ValueError(f"Implausible Episode20 reconstruction volume: {volume}")
    if metadata.get("schema") != "voxel_dough_reconstruction/v2" or metadata.get("frame") != 0:
        raise ValueError("Unexpected reconstruction metadata schema or frame")
    if metadata.get("pointclouds_name") != "pointclouds_interpolated.pt":
        raise ValueError("Reconstruction used the wrong point-cloud source")
    if metadata.get("sampled_particles") != PARTICLE_COUNT:
        raise ValueError("Reconstruction metadata has the wrong particle count")
    if metadata.get("fill_mode") != "floor" or metadata.get("fill_axis") != "y" or metadata.get("fill_direction") != "negative":
        raise ValueError("Reconstruction did not use the requested negative-Y floor fill")
    if metadata.get("calibration_schema") != "taichidough/scene-calibration/v2":
        raise ValueError("Reconstruction metadata does not name metric calibration v2")
    calibration_record = metadata.get("calibration")
    if not isinstance(calibration_record, dict):
        raise ValueError("Reconstruction metadata is missing calibration details")
    if calibration_record.get("source_frame") != "mocap" or calibration_record.get("scene_frame") != "table-aligned":
        raise ValueError("Reconstruction metadata has incorrect coordinate frames")
    frames = metadata.get("array_frames")
    if not isinstance(frames, dict):
        raise ValueError("Reconstruction metadata is missing array coordinate frames")
    if frames.get("source_filtered_points_xyz") != "mocap":
        raise ValueError("Source reconstruction points must remain in mocap coordinates")
    scene_arrays = {
        "scene_transformed_points_xyz",
        "real_points_xyz",
        "voxel_centers_xyz",
        "surface_particles_xyz",
        "sampled_particles_xyz",
    }
    if any(frames.get(name) != "table-aligned" for name in scene_arrays):
        raise ValueError("Reconstructed arrays must be recorded in table-aligned coordinates")
    fill = metadata.get("fill")
    if not isinstance(fill, dict) or fill.get("segmentation_counts", {}).get("kept", 0) < 50:
        raise ValueError("Reconstruction retained too few points above the floor")
    intersection = fill.get("intersection")
    if not isinstance(intersection, dict) or intersection.get("accepted", 0) < 10:
        raise ValueError("Too few visible-surface columns reached the floor")

    array_hash = hashlib.sha256(
        np.ascontiguousarray(particles.astype(np.float32)).tobytes(order="C")
    ).hexdigest()
    if metadata.get("sampled_particles_sha256") != array_hash:
        raise ValueError("Reconstructed particle content hash differs from metadata")
    metadata = replace_path_prefix(metadata, str(frame_dir), str(final_frame_dir))
    metadata = replace_path_prefix(metadata, str(frame_dir.parent.parent), str(final_frame_dir.parent.parent))
    metadata_path.write_bytes(canonical_json(metadata))
    return {
        "object_volume_m3": volume,
        "sampled_particles": PARTICLE_COUNT,
        "minimum_floor_distance_m": minimum_floor_distance,
        "retained_points": int(fill["segmentation_counts"]["kept"]),
        "accepted_surface_columns": int(intersection["accepted"]),
        "calibration_sha256": sha256_file(calibration_path),
        "sampled_particles_file_sha256": sha256_file(frame_dir / "sampled_particles_xyz.npy"),
        "reconstruction_metadata_sha256": sha256_file(metadata_path),
    }


def reconstruct_all(
    temporal: Path,
    reconstructor: Path = DEFAULT_RECONSTRUCTOR,
    preserved_v1: Path = DEFAULT_PRESERVED_V1,
    *,
    python: str = sys.executable,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    temporal = temporal.expanduser().resolve()
    reconstructor = reconstructor.expanduser().resolve()
    preserved_v1 = preserved_v1.expanduser().resolve()
    selection_path = temporal / "range_selection.json"
    validation_path = temporal / "validation.json"
    calibration_path = temporal / "calibration/scene_calibration_v2.json"
    for path in (selection_path, validation_path, calibration_path, reconstructor):
        if not path.is_file():
            raise FileNotFoundError(path)
    if load_json(validation_path).get("status") != "valid":
        raise ValueError("Episode20 temporal package validation did not pass")
    validate_calibration(calibration_path)
    selection = load_json(selection_path)
    ranges = selection.get("ranges")
    if not isinstance(ranges, list) or len(ranges) != 10:
        raise ValueError("Expected exactly ten Episode20 temporal ranges")
    output = temporal / "reconstructions"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Episode20 reconstructions: {output}")

    recordings = temporal / "recordings"
    source_files = [selection_path, validation_path, calibration_path, reconstructor]
    for row in ranges:
        recording = recordings / row["output_name"]
        source_files.extend(
            recording / name
            for name in ("pointclouds_interpolated.pt", "paths_interpolated.pt", "sequence_metadata.json")
        )
    if any(not path.is_file() for path in source_files):
        missing = [str(path) for path in source_files if not path.is_file()]
        raise FileNotFoundError("Missing reconstruction input: " + ", ".join(missing))
    sources_before = snapshot_files(source_files)
    preserved_before = tree_sha256(preserved_v1)

    temp = Path(tempfile.mkdtemp(prefix=".episode20_reconstructions.", dir=temporal))
    final_rows = []
    try:
        for row, expected_count in zip(ranges, EXPECTED_FRAME_COUNTS):
            name = row["output_name"]
            recording = recordings / name
            metadata = load_json(recording / "sequence_metadata.json")
            if metadata.get("frame_count") != expected_count:
                raise ValueError(f"{name} has an unexpected materialized frame count")
            command = reconstruction_command(python, reconstructor, recording, temp, calibration_path)
            completed = runner(
                command,
                cwd=REPO,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"Reconstruction failed for {name}:\n{completed.stdout}")
            frame_dir = temp / name / "frame_0000"
            checked = validate_reconstruction(
                frame_dir,
                calibration_path,
                output / name / "frame_0000",
            )
            final_rows.append(
                {
                    "episode_id": name,
                    "command": command,
                    "stdout": completed.stdout,
                    **checked,
                }
            )

        sources_after = snapshot_files(source_files)
        if sources_after != sources_before:
            raise RuntimeError("Episode20 reconstruction inputs changed during execution")
        preserved_after = tree_sha256(preserved_v1)
        if preserved_after != preserved_before:
            raise RuntimeError("Preserved Episode20 temporal v1 changed during reconstruction")
        manifest = {
            "schema": SCHEMA,
            "status": "valid",
            "settings": {
                "frame": 0,
                "pointclouds_name": "pointclouds_interpolated.pt",
                "num_particles": PARTICLE_COUNT,
                "grid": GRID_SIZE,
                "stencil_validation": "corrected-v1 floor(x * grid - 0.5), base in [-1, grid - 3]",
                "voxel_size_m": VOXEL_SIZE_M,
                "fill_mode": "floor",
                "fill_axis": "y",
                "fill_direction": "negative",
                "floor_clearance_m": 0.003,
                "floor_min_thickness_m": 0.001,
                "floor_max_thickness_m": 0.25,
                "bbox_padding_m": 0.006,
                "trim_quantile": 0.005,
                "footprint_dilate": 1,
                "seed": 0,
            },
            "chunks": final_rows,
            "source_sha256_start": sources_before,
            "source_sha256_end": sources_after,
            "preserved_v1_unchanged": True,
            "calibration_optimization_run": False,
            "simulation_run": False,
        }
        (temp / "reconstructions_manifest.json").write_bytes(canonical_json(manifest))
        hashes = {
            str(path.relative_to(temp)): sha256_file(path)
            for path in sorted(temp.rglob("*"))
            if path.is_file() and path.name != "SHA256SUMS"
        }
        (temp / "SHA256SUMS").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in hashes.items()),
            encoding="utf-8",
        )
        os.rename(temp, output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-dir", type=Path, default=DEFAULT_TEMPORAL)
    parser.add_argument("--reconstructor", type=Path, default=DEFAULT_RECONSTRUCTOR)
    parser.add_argument("--preserved-v1", type=Path, default=DEFAULT_PRESERVED_V1)
    args = parser.parse_args(argv)
    output = reconstruct_all(args.temporal_dir, args.reconstructor, args.preserved_v1)
    print(output)
    print("Reconstructed initial states only; no calibration optimization or simulation was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
