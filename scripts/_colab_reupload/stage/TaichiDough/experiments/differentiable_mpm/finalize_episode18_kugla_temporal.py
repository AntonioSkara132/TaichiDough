"""Generate shared-parameter Episode18 configs from materialized chunks and reconstructions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.differentiable_mpm.reference_adapter import get_reference_modules, reference_policy

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
DEFAULT_TEMPORAL = EXPERIMENT / "data" / "episode18_kugla_temporal_v1"
DEFAULT_BASE_CONFIG = EXPERIMENT / "configs" / "episode18_registered_adhesive_fit.json"
DENSITY_KG_M3 = 1200.0
REFERENCE_VOLUME_M3 = 0.000113832
CONSTANT_MASS_KG = DENSITY_KG_M3 * REFERENCE_VOLUME_M3
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO):
        raise ValueError(f"Generated input must be inside the repository: {resolved}")
    return str(resolved.relative_to(REPO))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def finalize(temporal: Path, base_config_path: Path) -> Path:
    temporal = temporal.resolve()
    selection = load_json(temporal / "range_selection.json")
    assumptions = load_json(temporal / "assumptions.json")
    if assumptions["physical"] != {
        "chunk01_volume_m3": REFERENCE_VOLUME_M3,
        "density_kg_m3": DENSITY_KG_M3,
        "eventual_mass_kg": CONSTANT_MASS_KG,
        "mass_policy": "constant across every retained segment, derived from the validated episode18_kugla frame-0 reconstruction volume",
        "status": "ready",
    }:
        raise ValueError("Temporal physical assumptions differ from the accepted constant-mass policy")
    ranges = selection.get("ranges")
    if not isinstance(ranges, list) or len(ranges) != 13:
        raise ValueError("Expected exactly 13 retained Episode18 ranges")

    base = load_json(base_config_path.resolve())
    recordings = temporal / "recordings"
    reconstructions = temporal / "reconstructions"
    configs = temporal / "configs"
    dataset_path = temporal / "dataset_episode18_kugla_temporal_stretch_clamp_v1.json"
    physical_path = temporal / "physical_inputs.json"
    if dataset_path.exists() or physical_path.exists() or configs.exists():
        raise FileExistsError("Refusing to overwrite finalized Episode18 temporal configs")

    episode_rows = []
    volume_rows = {}
    effective_density_rows = {}
    fingerprints = {}
    with reference_policy("frozen"):
        modules = get_reference_modules(physics_version="corrected-v1")
        for row in ranges:
            name = row["output_name"]
            recording = recordings / name
            reconstruction = reconstructions / name / "frame_0000"
            particles = reconstruction / "sampled_particles_xyz.npy"
            metadata_path = reconstruction / "reconstruction_metadata.json"
            for path in (recording, particles, metadata_path):
                if not path.exists():
                    raise FileNotFoundError(f"Missing prepared chunk input: {path}")
            sequence = modules.dynamics.load_observation_sequence(recording)
            frame_count = len(sequence.times)
            if frame_count < 3:
                raise ValueError(f"{name} has fewer than three processed frames")
            metadata = load_json(metadata_path)
            volume = float(metadata["object_volume_m3"])
            if not math.isfinite(volume) or volume <= 0:
                raise ValueError(f"{name} reconstruction volume must be positive and finite")
            volume_rows[name] = volume
            effective_density = CONSTANT_MASS_KG / volume
            effective_density_rows[name] = effective_density
            fingerprints[name] = sequence.fingerprint

            config = json.loads(json.dumps(base))
            config["name"] = f"{name}-shared-eight-parameter-fit"
            config["paths"]["episode"] = relative(recording)
            config["paths"]["initial_particles"] = relative(particles)
            config["paths"]["reconstruction_metadata"] = relative(metadata_path)
            config["expected_sha256"]["initial_particles"] = sha256(particles)
            config["expected_sha256"]["reconstruction_metadata"] = sha256(metadata_path)
            config["expected_sequence_fingerprint"] = sequence.fingerprint
            config["mass_kg"] = CONSTANT_MASS_KG
            # The reference density determines one constant physical mass. The config
            # density must remain algebraically consistent with each reconstructed volume.
            config["density_kg_m3"] = effective_density
            config["parameters"] = dict(INITIAL)
            config["fit_parameters"] = list(FIT)
            config["parameter_bounds"] = {key: list(value) for key, value in BOUNDS.items()}
            config["simulation"]["plasticity"] = "stretch-clamp"
            config["simulation"]["use_jp"] = False
            config["simulation"]["jp_hardening"] = 0.0
            config["simulation"]["tool_collision"] = "sdf"
            config["simulation"]["tool_contact_padding"] = TOOL_CONTACT_PADDING_M
            config["simulation"]["tool_contact_model"] = "coulomb-adhesive-v1"
            config["simulation"]["tool_contact_absorption"] = 0.0
            config["simulation"]["tool_friction_coefficient"] = INITIAL["tool_friction_coefficient"]
            config["simulation"]["tool_stickiness"] = INITIAL["tool_stickiness"]
            config["training"] = {"start_frame": 1, "end_frame": frame_count - 2, "stride": 1}
            config["validation"] = {"start_frame": frame_count - 1, "end_frame": frame_count - 1, "stride": 1}
            filename = f"{name}_stretch_clamp_v1.json"
            write_new(configs / filename, config)
            episode_rows.append({
                "id": name,
                "config": f"configs/{filename}",
                "membership": "training",
                "weight": 1.0,
                "scored_window": {"start_frame": 1, "end_frame": frame_count - 1, "stride": 1},
            })

    dataset = {
        "schema": "taichidough/differentiable-dataset/v2",
        "name": "episode18-kugla-temporal-shared-eight-parameter-v1",
        "shared_parameters": {
            "initial": {name: INITIAL[name] for name in FIT},
            "fit": list(FIT),
            "bounds": {key: list(value) for key, value in BOUNDS.items()},
        },
        "episodes": episode_rows,
    }
    write_new(dataset_path, dataset)
    write_new(physical_path, {
        "schema": "taichidough/episode18-kugla-temporal-physical-inputs/v1",
        "density_kg_m3": DENSITY_KG_M3,
        "reference_volume_m3": REFERENCE_VOLUME_M3,
        "constant_mass_kg": CONSTANT_MASS_KG,
        "mass_applied_to_every_chunk": True,
        "split_provenance": {
            "split_group": selection["source"]["split_group"],
            "all_chunks_from_one_recording": True,
            "independent_validation_present": False,
            "omitted_annotation_segments": [1, 15],
        },
        "per_chunk_reconstructed_volume_m3": volume_rows,
        "per_chunk_effective_density_kg_m3": effective_density_rows,
        "effective_density_note": "Each config stores constant_mass_kg / reconstructed_volume_m3 to satisfy the simulator mass identity; 1200 kg/m3 is used only to derive the shared physical mass from the validated reference volume.",
        "sequence_fingerprints": fingerprints,
        "shared_initial": dataset["shared_parameters"]["initial"],
        "shared_bounds": BOUNDS,
        "calibration_run": False,
        "fit_run": False,
    })
    return dataset_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-dir", type=Path, default=DEFAULT_TEMPORAL)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    args = parser.parse_args(argv)
    dataset = finalize(args.temporal_dir, args.base_config)
    print(dataset)
    print(f"constant_mass_kg={CONSTANT_MASS_KG:.7f}")
    print("Prepared configs only; no calibration or simulation was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
