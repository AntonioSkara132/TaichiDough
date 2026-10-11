"""Publish validated Episode20 calibration inputs from temporal v2."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from experiments.differentiable_mpm.config import load_config, verify_input_paths
from experiments.differentiable_mpm.dataset_config import load_dataset
from experiments.differentiable_mpm.prepare_episode20_kugla_temporal import (
    CONSTANT_MASS_KG,
    DEFAULT_OUTPUT as DEFAULT_TEMPORAL,
    DEFAULT_PRESERVED_V1,
    EXPECTED_FRAME_COUNTS,
    INHERITED_BUNDLE,
    INHERITED_FILES,
    NOMINAL_DENSITY_KG_M3,
    REFERENCE_VOLUME_M3,
    SCHEMA as TEMPORAL_SCHEMA,
    canonical_json,
    load_json,
    sha256_file,
    tree_sha256,
)
from experiments.differentiable_mpm.reference_adapter import get_reference_modules, reference_policy

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = EXPERIMENT / "configs/episode18_registered_adhesive_fit.json"
FINALIZATION_DIR = "calibration_inputs_v2"
DATASET_NAME = "dataset_episode20_kugla_temporal_stretch_clamp_v2.json"
SCHEMA = "taichidough/episode20-kugla-calibration-inputs/v2"
TOOL_CONTACT_PADDING_M = (1.0 / 48.0) / 16.0
INITIAL = {
    "youngs_modulus": 17000.0,
    "poisson_ratio": 0.47,
    "viscosity": 1.0,
    "plastic_min": 0.9,
    "plastic_max": 1.1,
    "tool_retention": 1.0,
    "floor_retention": 0.8,
    "tool_friction_coefficient": 0.9,
    "tool_stickiness": 0.05,
}
FIT = (
    "youngs_modulus",
    "poisson_ratio",
    "viscosity",
    "plastic_min",
    "plastic_max",
    "floor_retention",
    "tool_friction_coefficient",
    "tool_stickiness",
)
BOUNDS = {
    "youngs_modulus": [2000.0, 60000.0],
    "poisson_ratio": [0.45, 0.49],
    "viscosity": [0.0, 60.0],
    "plastic_min": [0.7, 0.999],
    "plastic_max": [1.001, 1.3],
    "floor_retention": [0.2, 0.9],
    "tool_friction_coefficient": [0.5, 1.0],
    "tool_stickiness": [0.0, 0.3],
}


def relative(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO):
        raise ValueError(f"Generated calibration input must be inside the repository: {resolved}")
    return str(resolved.relative_to(REPO))


def verify_sha256_manifest(root: Path) -> dict[str, str]:
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing SHA256SUMS: {manifest}")
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not name or name in rows or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Invalid SHA256SUMS entry: {line!r}")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Invalid SHA-256 digest for {name}")
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"SHA256SUMS mismatch for {path}")
        rows[name] = digest
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path != manifest
        and path.relative_to(root).parts[0] not in {"reconstructions", FINALIZATION_DIR}
    }
    if set(rows) != actual:
        raise ValueError("SHA256SUMS does not list every initial temporal-package file exactly once")
    return rows


def validate_temporal(temporal: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = load_json(temporal / "range_selection.json")
    assumptions = load_json(temporal / "assumptions.json")
    validation = load_json(temporal / "validation.json")
    if assumptions.get("schema") != TEMPORAL_SCHEMA + "/assumptions":
        raise ValueError("Finalizer requires Episode20 temporal v2 assumptions")
    if validation.get("status") != "valid":
        raise ValueError("Episode20 temporal v2 validation did not pass")
    frames = validation.get("checks", {}).get("expected_frame_counts")
    if frames != list(EXPECTED_FRAME_COUNTS):
        raise ValueError("Episode20 temporal package has unexpected frame counts")
    physical = assumptions.get("physical")
    if not isinstance(physical, dict):
        raise ValueError("Episode20 temporal package is missing physical assumptions")
    expected_physical = {
        "nominal_density_kg_m3": NOMINAL_DENSITY_KG_M3,
        "reference_volume_m3": REFERENCE_VOLUME_M3,
        "eventual_mass_kg": CONSTANT_MASS_KG,
        "mass_policy": "constant across all chunks, inherited from the Episode18 reference volume and nominal density",
        "measurement_status": "inherited_assumption_not_episode20_measurement",
    }
    if physical != expected_physical:
        raise ValueError("Episode20 physical assumptions differ from the reviewed inherited policy")
    coordinate_frames = assumptions.get("coordinate_frames")
    if coordinate_frames != {
        "pointclouds": "mocap",
        "tool_paths": "mocap",
        "scene": "table-aligned",
        "scene_from_source": "rigid transform from the independently estimated Episode20 table plane",
        "table_plane_validation": "passed",
    }:
        raise ValueError("Episode20 temporal package has unexpected coordinate-frame declarations")
    inherited = assumptions.get("inherited_geometry")
    if not isinstance(inherited, dict) or inherited.get("physical_setup_status") != "assumed_same_UR5e_and_Gen3_spatula_marker_assemblies":
        raise ValueError("Episode20 tool-geometry reuse assumption is missing")
    ranges = selection.get("ranges")
    if not isinstance(ranges, list) or len(ranges) != 10:
        raise ValueError("Expected exactly ten Episode20 temporal ranges")
    verify_sha256_manifest(temporal)
    return selection, assumptions


def validate_calibration(path: Path) -> dict[str, Any]:
    calibration = load_json(path)
    if calibration.get("schema") != "taichidough/scene-calibration/v2":
        raise ValueError("Episode20 calibration must use scene-calibration v2")
    if calibration.get("source_frame") != "mocap" or calibration.get("scene_frame") != "table-aligned":
        raise ValueError("Episode20 calibration must map mocap inputs into table-aligned coordinates")
    transform = np.asarray(calibration.get("scene_from_source"), dtype=np.float64)
    plane = np.asarray(calibration.get("floor_plane_scene"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Episode20 calibration must have a finite table-alignment transform")
    if np.allclose(transform, np.eye(4), rtol=0, atol=1e-12):
        raise ValueError("Episode20 calibration must retain the measured table alignment")
    if plane.shape != (4,) or not np.allclose(plane, [0.0, 1.0, 0.0, 0.0], rtol=0, atol=1e-12):
        raise ValueError("Episode20 calibration must use y=0 in the table-aligned scene")
    provenance = calibration.get("provenance")
    alignment = provenance.get("table_alignment") if isinstance(provenance, dict) else None
    if not isinstance(alignment, dict) or alignment.get("previous_scene_frame") != "mocap":
        raise ValueError("Episode20 calibration is missing mocap-to-table alignment provenance")
    if alignment.get("source_tensors_transformed") is not False:
        raise ValueError("Episode20 source tensors must remain in mocap coordinates")
    return calibration


def build_config(
    base: dict[str, Any],
    name: str,
    recording: Path,
    particles: Path,
    reconstruction_metadata: Path,
    calibration: Path,
    frame_count: int,
    sequence_fingerprint: str,
    volume: float,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["name"] = f"{name}-episode20-shared-eight-parameter-fit-v2"
    config["paths"].update(
        episode=relative(recording),
        calibration=relative(calibration),
        initial_particles=relative(particles),
        reconstruction_metadata=relative(reconstruction_metadata),
    )
    for key, path in {
        "tool_geometry": INHERITED_BUNDLE / "tool_geometry.json",
        "collision_manifest": INHERITED_BUNDLE / "collision_manifest.json",
        "ur_collision_mesh": INHERITED_BUNDLE / "ur_spathla_collision_solid.stl",
        "kinova_collision_mesh": INHERITED_BUNDLE / "gen3_spathla_collision_solid.stl",
    }.items():
        config["paths"][key] = relative(path)
    config["expected_sha256"].update(
        calibration=sha256_file(calibration),
        initial_particles=sha256_file(particles),
        reconstruction_metadata=sha256_file(reconstruction_metadata),
        tool_geometry=sha256_file(INHERITED_BUNDLE / "tool_geometry.json"),
        collision_manifest=sha256_file(INHERITED_BUNDLE / "collision_manifest.json"),
        ur_collision_mesh=sha256_file(INHERITED_BUNDLE / "ur_spathla_collision_solid.stl"),
        kinova_collision_mesh=sha256_file(INHERITED_BUNDLE / "gen3_spathla_collision_solid.stl"),
    )
    config["expected_sequence_fingerprint"] = sequence_fingerprint
    config["mass_kg"] = CONSTANT_MASS_KG
    config["density_kg_m3"] = CONSTANT_MASS_KG / volume
    config["parameters"] = dict(INITIAL)
    config["fit_parameters"] = list(FIT)
    config["parameter_bounds"] = {key: list(value) for key, value in BOUNDS.items()}
    config["simulation"].update(
        physics_version="corrected-v1",
        n_particles=24000,
        grid=48,
        floor_y=0.0,
        plasticity="stretch-clamp",
        use_jp=False,
        jp_hardening=0.0,
        tool_collision="sdf",
        tool_contact_padding=TOOL_CONTACT_PADDING_M,
        tool_contact_model="coulomb-adhesive-v1",
        tool_contact_absorption=0.0,
        tool_friction_coefficient=INITIAL["tool_friction_coefficient"],
        tool_stickiness=INITIAL["tool_stickiness"],
    )
    config["training"] = {"start_frame": 1, "end_frame": frame_count - 2, "stride": 1}
    config["validation"] = {"start_frame": frame_count - 1, "end_frame": frame_count - 1, "stride": 1}
    return config


def finalize(
    temporal: Path,
    base_config_path: Path = DEFAULT_BASE_CONFIG,
    preserved_v1: Path = DEFAULT_PRESERVED_V1,
) -> Path:
    temporal = temporal.expanduser().resolve()
    base_config_path = base_config_path.expanduser().resolve()
    preserved_v1 = preserved_v1.expanduser().resolve()
    selection, assumptions = validate_temporal(temporal)
    base = load_json(base_config_path)
    calibration_path = temporal / "calibration/scene_calibration_v2.json"
    table_report_path = temporal / "calibration/episode20_table_plane_validation.json"
    reconstruction_manifest_path = temporal / "reconstructions/reconstructions_manifest.json"
    reconstructions_hash_path = temporal / "reconstructions/SHA256SUMS"
    for path in (calibration_path, table_report_path, reconstruction_manifest_path, reconstructions_hash_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    validate_calibration(calibration_path)
    table_report = load_json(table_report_path)
    if table_report.get("passed") is not True or table_report.get("calibration_sha256") != sha256_file(calibration_path):
        raise ValueError("Episode20 table-plane report does not validate this calibration")
    reconstruction_manifest = load_json(reconstruction_manifest_path)
    if reconstruction_manifest.get("status") != "valid" or len(reconstruction_manifest.get("chunks", [])) != 10:
        raise ValueError("Episode20 reconstructions did not pass validation")
    for name in INHERITED_FILES:
        if not (INHERITED_BUNDLE / name).is_file():
            raise FileNotFoundError(INHERITED_BUNDLE / name)
    inherited_hashes = {name: sha256_file(INHERITED_BUNDLE / name) for name in INHERITED_FILES}
    if assumptions["inherited_geometry"].get("sha256") != inherited_hashes:
        raise ValueError("Inherited tool assets changed after temporal preparation")

    output = temporal / FINALIZATION_DIR
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite finalized Episode20 inputs: {output}")
    preserved_before = tree_sha256(preserved_v1)
    source_paths = [
        temporal / "range_selection.json",
        temporal / "assumptions.json",
        temporal / "validation.json",
        temporal / "SHA256SUMS",
        calibration_path,
        table_report_path,
        reconstruction_manifest_path,
        reconstructions_hash_path,
        base_config_path,
        *(INHERITED_BUNDLE / name for name in INHERITED_FILES),
    ]
    source_hashes_before = {str(path): sha256_file(path) for path in source_paths}

    temp = Path(tempfile.mkdtemp(prefix=f".{FINALIZATION_DIR}.", dir=temporal))
    try:
        configs = temp / "configs"
        configs.mkdir()
        episode_rows = []
        volume_rows: dict[str, float] = {}
        density_rows: dict[str, float] = {}
        fingerprints: dict[str, str] = {}
        config_hashes: dict[str, str] = {}
        ranges = selection["ranges"]
        with reference_policy("frozen"):
            modules = get_reference_modules(physics_version="corrected-v1")
            for row, expected_count in zip(ranges, EXPECTED_FRAME_COUNTS):
                name = row["output_name"]
                recording = temporal / "recordings" / name
                reconstruction = temporal / "reconstructions" / name / "frame_0000"
                particles = reconstruction / "sampled_particles_xyz.npy"
                metadata_path = reconstruction / "reconstruction_metadata.json"
                for path in (recording, particles, metadata_path):
                    if not path.exists():
                        raise FileNotFoundError(f"Missing Episode20 prepared input: {path}")
                sequence = modules.dynamics.load_observation_sequence(recording)
                frame_count = len(sequence.times)
                if frame_count != expected_count:
                    raise ValueError(f"{name} has {frame_count} frames, expected {expected_count}")
                metadata = load_json(metadata_path)
                volume = float(metadata.get("object_volume_m3", float("nan")))
                if not math.isfinite(volume) or not 1e-6 <= volume <= 1e-3:
                    raise ValueError(f"{name} has an implausible reconstruction volume")
                if metadata.get("array_frames", {}).get("sampled_particles_xyz") != "table-aligned":
                    raise ValueError(f"{name} reconstruction is not in table-aligned coordinates")
                volume_rows[name] = volume
                density_rows[name] = CONSTANT_MASS_KG / volume
                fingerprints[name] = sequence.fingerprint
                config = build_config(
                    base,
                    name,
                    recording,
                    particles,
                    metadata_path,
                    calibration_path,
                    frame_count,
                    sequence.fingerprint,
                    volume,
                )
                config_path = configs / f"{name}_stretch_clamp_v2.json"
                config_path.write_bytes(canonical_json(config))
                loaded = load_config(config_path)
                verify_input_paths(loaded)
                config_hashes[config_path.name] = sha256_file(config_path)
                episode_rows.append(
                    {
                        "id": name,
                        "config": f"configs/{config_path.name}",
                        "membership": "training",
                        "weight": 1.0,
                        "scored_window": {
                            "start_frame": 1,
                            "end_frame": frame_count - 1,
                            "stride": 1,
                        },
                    }
                )

        dataset = {
            "schema": "taichidough/differentiable-dataset/v2",
            "name": "episode20-kugla-temporal-shared-eight-parameter-v2",
            "shared_parameters": {
                "initial": {name: INITIAL[name] for name in FIT},
                "fit": list(FIT),
                "bounds": {key: list(value) for key, value in BOUNDS.items()},
            },
            "episodes": episode_rows,
        }
        dataset_path = temp / DATASET_NAME
        dataset_path.write_bytes(canonical_json(dataset))
        loaded_dataset = load_dataset(dataset_path)
        if len(loaded_dataset.episodes) != 10 or any(
            episode.membership != "training" for episode in loaded_dataset.episodes
        ):
            raise ValueError("Final Episode20 dataset must contain ten training members")

        physical = {
            "schema": "taichidough/episode20-kugla-temporal-physical-inputs/v2",
            "nominal_density_kg_m3": NOMINAL_DENSITY_KG_M3,
            "reference_volume_m3": REFERENCE_VOLUME_M3,
            "constant_mass_kg": CONSTANT_MASS_KG,
            "mass_applied_to_every_chunk": True,
            "mass_measurement_status": "inherited_from_episode18_not_measured_for_episode20",
            "tool_geometry_status": "inherited_assumption_same_UR5e_and_Gen3_spatula_marker_assemblies",
            "tool_geometry_sha256": inherited_hashes,
            "split_provenance": {
                "split_group": selection["source"]["split_group"],
                "all_chunks_from_one_recording": True,
                "dataset_membership": "all_training",
                "independent_dataset_validation_present": False,
                "within_chunk_final_frame_reserved_for_validation": True,
                "omitted_annotation_segments": [1, 12],
            },
            "per_chunk_reconstructed_volume_m3": volume_rows,
            "per_chunk_effective_density_kg_m3": density_rows,
            "effective_density_note": "Each config stores constant_mass_kg / reconstructed_volume_m3; 1200 kg/m3 is used only with the inherited Episode18 reference volume to derive constant mass.",
            "sequence_fingerprints": fingerprints,
            "shared_initial": dataset["shared_parameters"]["initial"],
            "shared_bounds": BOUNDS,
            "calibration_optimization_run": False,
            "fit_run": False,
            "simulation_run": False,
        }
        (temp / "physical_inputs.json").write_bytes(canonical_json(physical))

        source_hashes_after = {str(path): sha256_file(path) for path in source_paths}
        if source_hashes_after != source_hashes_before:
            raise RuntimeError("Episode20 finalization inputs changed during execution")
        preserved_after = tree_sha256(preserved_v1)
        if preserved_after != preserved_before:
            raise RuntimeError("Preserved Episode20 temporal v1 changed during finalization")
        preparation_validation = {
            "schema": SCHEMA,
            "status": "valid",
            "checks": {
                "temporal_package_valid": True,
                "source_coordinate_frame": "mocap",
                "scene_coordinate_frame": "table-aligned",
                "measured_nonidentity_scene_from_source": True,
                "table_plane_validation_passed": True,
                "reconstruction_count": 10,
                "config_count": 10,
                "dataset_episode_count": 10,
                "all_dataset_members_training": True,
                "all_configs_loaded": True,
                "all_input_hashes_verified": True,
                "constant_mass_kg": CONSTANT_MASS_KG,
                "mass_is_inherited_assumption": True,
                "tool_geometry_is_inherited_assumption": True,
                "preserved_v1_unchanged": True,
                "calibration_optimization_run": False,
                "fit_run": False,
                "simulation_run": False,
            },
            "dataset_fingerprint": loaded_dataset.fingerprint,
            "config_sha256": config_hashes,
            "source_sha256_start": source_hashes_before,
            "source_sha256_end": source_hashes_after,
            "preserved_v1_file_sha256": preserved_after,
        }
        (temp / "preparation_validation.json").write_bytes(canonical_json(preparation_validation))
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
    return output / DATASET_NAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-dir", type=Path, default=DEFAULT_TEMPORAL)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--preserved-v1", type=Path, default=DEFAULT_PRESERVED_V1)
    args = parser.parse_args(argv)
    dataset = finalize(args.temporal_dir, args.base_config, args.preserved_v1)
    print(dataset)
    print(f"constant_mass_kg={CONSTANT_MASS_KG:.7f} (inherited Episode18 assumption)")
    print("Prepared inputs only; no calibration optimization, fit, or simulation was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
