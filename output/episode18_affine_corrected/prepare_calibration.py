#!/usr/bin/env python3
"""Prepare a local corrected-solver manifest without modifying prior experiments."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

# pyright: reportMissingImports=false
from calibrate_youngs_modulus import validate_manifest


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    manifest_path = directory / "material_calibration_manifest.json"
    if manifest_path.exists() or (directory / "candidate_cache").exists():
        raise ValueError(f"Refusing to replace an existing experiment: {directory}")

    source = ROOT / "data/single_episode_calibration/episode18_kugla_fixed_collision_11_9/material_calibration_manifest.json"
    manifest = json.loads(source.read_text())
    old_inputs = json.loads(json.dumps(manifest["inputs"]))
    episode = ROOT.parent / "data/deformpath_training/DeformPath3/snimanje_23_10/episode18_kugla"
    reconstruction = ROOT / "data/single_episode_calibration/episode18_kugla/reconstruction/episode18_kugla/frame_0000"
    local_inputs = {
        "geometry": ROOT / "configs/tool_geometry_episode18_sdf.json",
        "reconstruction_metadata": reconstruction / "reconstruction_metadata.json",
        "initial_particles": reconstruction / "sampled_particles_xyz.npy",
        "calibration": episode / "scene_calibration_v2.json",
        "simulator": ROOT / "scripts/taichi_viscoelastic_mpm_scene.py",
        "evaluator": ROOT / "scripts/evaluate_dynamic_topview_match.py",
        "ur_collision_mesh": ROOT / "meshes/ur_spathla_collision_solid.stl",
        "kinova_collision_mesh": ROOT / "meshes/gen3_spathla_collision_solid.stl",
        "collision_manifest": ROOT / "meshes/tool_collision_meshes_v1.json",
    }
    for name, path in local_inputs.items():
        digest = sha256(path)
        if name in old_inputs and name not in ("simulator", "reconstruction_metadata"):
            if digest != old_inputs[name]["sha256"]:
                raise ValueError(f"Physical input or evaluator differs from the reference: {name}")
        manifest["inputs"][name] = {"path": str(path), "sha256": digest}
    manifest["inputs"]["sequence"]["episode_dir"] = str(episode)
    fixed = manifest["fixed_parameters"]
    fixed["recorded_setup"]["tool_skin_label"] = "dx/8"
    simulator_args = fixed["simulator_arguments"]
    simulator_args["--ur-tool-mesh"] = str(ROOT / "meshes/ur_spathla.stl")
    simulator_args["--kinova-tool-mesh"] = str(ROOT / "meshes/gen3_spathla.stl")
    # Use the default solids so the simulator verifies their asset manifest.
    simulator_args.pop("--ur-tool-collision-mesh", None)
    simulator_args.pop("--kinova-tool-collision-mesh", None)
    simulator_args["--cpu"] = args.backend == "cpu"
    manifest["name"] = "episode18-affine-corrected"
    manifest["execution"] = {
        "cache_dir": str(directory / "candidate_cache"),
        "result_path": str(directory / "material_calibration_result.json"),
        "subprocess_timeout_s": 7200.0,
    }
    validated = validate_manifest(manifest, manifest_path)
    directory.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x") as stream:
        json.dump(validated, stream, indent=2, allow_nan=False)
        stream.write("\n")
    provenance = {
        "reference_manifest": str(source),
        "reference_manifest_sha256": sha256(source),
        "reference_simulator_sha256": old_inputs["simulator"]["sha256"],
        "corrected_simulator_sha256": validated["inputs"]["simulator"]["sha256"],
        "backend_request": args.backend,
        "physical_inputs_preserved": ["initial_particles", "geometry", "calibration", "sequence"],
        "numerical_changes": [
            "G2P uses 4/dx^2 with physical offsets.",
            "Stencil bases use floor; an extra node at -dx preserves support at scene zero.",
        ],
        "fixed_physical_settings": "Original mass, contact, floor coordinates, grid spacing, time step, damping and constitutive law retained.",
    }
    with (directory / "preparation.json").open("x") as stream:
        json.dump(provenance, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
