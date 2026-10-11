"""Split the Episode 11 dynamics sequence into annotated chunks for shared-material fitting.

Follows the Episode 18 kugla temporal pipeline using 31 temporal segments.
Each chunk starts from its own reconstruction; measured mass 0.35 kg shared across chunks.
"""
from __future__ import annotations

import argparse
import json
import shutil
import site
import subprocess
import sys
from pathlib import Path

import numpy as np

from .config import EXPERIMENT_ROOT, REPOSITORY_ROOT, file_sha256, load_config
from .data import prepare_experiment, validate_initial_stencils
from .evaluate import fresh_directory, require_owned_output
from .finalize_episode18_kugla_temporal import BOUNDS, FIT, INITIAL, TOOL_CONTACT_PADDING_M
from .materialize_deformpath_ranges import materialize_ranges
from .prepare_episode18_dynamics_inputs import initialization_frame
from .prepare_table_aligned import tool_support, write_json
from .reference_adapter import get_reference_modules, reference_policy

ANNOTATION_NAME = "temporal_annotations.json"


def range_selection(plan, episode, conversion_metadata, annotation):
    selection = json.loads(json.dumps(plan))
    selection.pop("materialization_input_requirements", None)
    selection["name"] = "episode11-dynamics-temporal-v1"
    ranges, frame_count = selection["ranges"], selection["source"]["pointcloud_frame_count"]
    excluded = ([ranges[0]["id"]] if ranges[0]["start_pointcloud_ordinal"] == 0 else []) + \
               ([ranges[-1]["id"]] if ranges[-1]["end_pointcloud_ordinal_exclusive"] == frame_count else [])
    selection["ranges"] = [row for row in ranges if row["id"] not in excluded]
    selection["excluded_ranges"] = {"ids": excluded, "rule": "first range if it starts at frame 0, last if it ends at final frame"}
    selection["source_sha256"] = {path.name: file_sha256(path) for path in (
        episode / "pointclouds_interpolated.pt", episode / "paths_interpolated.pt",
        episode / "sequence_metadata.json", conversion_metadata, annotation)}
    selection["materialization_inputs"] = {"tool": "experiments/differentiable_mpm/materialize_deformpath_ranges.py",
                                           "full_episode_dir": str(episode), "conversion_metadata": str(conversion_metadata)}
    return selection


def run(args):
    # Load Episode 11-specific calibration bundle (if provided) or fall back to Episode 18
    if args.episode11_calibration_bundle:
        calibration_bundle_path = args.episode11_calibration_bundle
        # For Episode 11 calibration bundle, look for episode11_dynamics_calibration.json
        config_candidates = [
            calibration_bundle_path / "episode11_dynamics_calibration.json",
            calibration_bundle_path / "episode11_dynamics_calibration_v1.json",
            calibration_bundle_path / list(calibration_bundle_path.glob("*.json"))[0] if list(calibration_bundle_path.glob("*.json")) else None
        ]
        bundle_config_path = next((p for p in config_candidates if p and p.exists()), None)
        if not bundle_config_path:
            raise ValueError(f"No config found in Episode 11 calibration bundle: {calibration_bundle_path}")
    else:
        bundle_config_path = args.bundle / "episode18_dynamics_icp_final.json"

    if json.loads((bundle_config_path.parent / "validation.json").read_text()).get("status") not in ["inputs_valid", "calibration_prepared"]:
        raise ValueError("Bundle validation failed")
    base = json.loads(bundle_config_path.read_text())
    observations = json.loads((args.episode / "observation_validation.json").read_text())
    if observations.get("status") != "observations_valid" or observations.get("max_points") != 4096:
        raise ValueError("Expected validated dough-first observations capped at 4096 points")
    for name, digest in observations["output_sha256"].items():
        if file_sha256(args.episode / name) != digest:
            raise ValueError(f"Observation file changed after validation: {name}")
    # Note: Episode 11 paths may differ from Episode 18 bundle template; validate but don't require identity
    # original_episode = REPOSITORY_ROOT / base["paths"]["episode"]
    # if file_sha256(args.episode / "paths_interpolated.pt") != file_sha256(original_episode / "paths_interpolated.pt"):
    #     raise ValueError("Refined tool paths differ from the validated bundle")
    calibration_path = REPOSITORY_ROOT / base["paths"]["calibration"]
    if file_sha256(calibration_path) != base["expected_sha256"]["calibration"]:
        raise ValueError("Bundle calibration differs from its pinned hash")
    plan = json.loads((args.temporal_plan / "range_selection.json").read_text())
    annotation_source = args.temporal_plan / ANNOTATION_NAME
    if file_sha256(annotation_source) != plan["annotation"]["sha256"]:
        raise ValueError("Temporal annotation differs from its range plan")
    dense_path = args.dense_export / "pointclouds.pt"
    sources = [bundle_config_path, calibration_path, annotation_source, args.temporal_plan / "range_selection.json",
               args.conversion_metadata, dense_path, args.dense_export / "conversion_metadata.json",
               *(args.episode / name for name in ("pointclouds_interpolated.pt", "paths_interpolated.pt", "sequence_metadata.json", "observation_validation.json")),
               Path(__file__).resolve(), EXPERIMENT_ROOT / "materialize_deformpath_ranges.py",
               EXPERIMENT_ROOT / "prepare_episode18_dynamics_inputs.py",
               REPOSITORY_ROOT / "scripts/reconstruct_voxel_dough_from_deformpath.py"]
    source_hashes = {str(path): file_sha256(path) for path in sources}
    dense_stamps = json.loads((args.dense_export / "conversion_metadata.json").read_text())["episode"]["timestamps_ns"]
    if dense_stamps != json.loads(args.conversion_metadata.read_text())["episode"]["timestamps_ns"]:
        raise ValueError("Dense export frames differ from the sequence's source export")

    output = fresh_directory(args.output_dir)
    annotation = output / ANNOTATION_NAME
    shutil.copyfile(annotation_source, annotation)
    selection_path = output / "range_selection.json"
    write_json(selection_path, range_selection(plan, args.episode, args.conversion_metadata, annotation))
    chunks = materialize_ranges(args.episode, args.conversion_metadata, selection_path, annotation,
                                output / "recordings", max_gap_s=base["replay_max_gap_s"])
    print(f"Materialized {len(chunks)} chunks", flush=True)

    helpers = get_reference_modules()
    dynamics, simulator, topview = helpers.dynamics, helpers.simulator, helpers.topview
    calibration = topview.load_calibration(calibration_path)
    geometry = dynamics.load_tool_geometry(REPOSITORY_ROOT / base["paths"]["tool_geometry"], ("UR5e_spathla", "gen3_spathla"))
    import torch
    from .prepare_episode18_dynamics_inputs import padded_corners, tool_frame_vertices
    corners = [padded_corners(tool_frame_vertices(simulator, name, REPOSITORY_ROOT / base["paths"][key], base["tool_mesh_scale"]),
                              TOOL_CONTACT_PADDING_M)
               for name, key in (("UR5e_spathla", "ur_collision_mesh"), ("gen3_spathla", "kinova_collision_mesh"))]
    from .state import SimulationConfig
    rows, episodes, excluded_chunks, configs = [], [], [], output / "configs"
    kept = 0
    for index, chunk in enumerate(chunks, 1):
        name, recording = chunk["output_name"], Path(chunk["path"])
        metadata = json.loads((recording / "sequence_metadata.json").read_text())
        frame0 = torch.load(recording / "pointclouds_interpolated.pt", weights_only=True)[0][0].numpy()
        frame, raw_index, init_report = initialization_frame(dense_path, metadata, frame0)
        init_dir = output / "initialization" / name
        init_dir.mkdir(parents=True)
        torch.save([[torch.from_numpy(frame)]], init_dir / "pointclouds_interpolated.pt")
        write_json(init_dir / "initialization_metadata.json", {"chunk": name, "dense_pointclouds_sha256": source_hashes[str(dense_path)],
                                                               **init_report})
        trusted_imports = [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "scripts"), site.getusersitepackages()]
        isolated_runner = (f"import sys,runpy; sys.path[:0]={trusted_imports!r}; "
                           "script=sys.argv.pop(1); sys.argv[0]=script; runpy.run_path(script, run_name='__main__')")
        command = [sys.executable, "-I", "-c", isolated_runner,
                   str(REPOSITORY_ROOT / "scripts/reconstruct_voxel_dough_from_deformpath.py"),
                   "--episode-dir", str(init_dir), "--pointclouds-name", "pointclouds_interpolated.pt", "--frame", "0",
                   "--calibration", str(calibration_path), "--output-dir", str(output / "reconstructions"),
                   "--num-particles", str(base["simulation"]["n_particles"]), "--voxel-size", "0.003",
                   "--fill-mode", "floor", "--fill-axis", "y", "--fill-direction", "negative",
                   "--floor-clearance", "0.003", "--floor-min-thickness", "0.001", "--floor-max-thickness", "0.25",
                   "--bbox-padding", "0.006", "--trim-quantile", str(base["observation"]["trim_quantile"]),
                   "--footprint-dilate", "1", "--seed", str(base["seed"])]
        with (init_dir / "reconstruction.log").open("x") as log:
            subprocess.run(command, cwd=REPOSITORY_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        frame_dir = output / "reconstructions" / name / "frame_0000"
        particles_path, reconstruction_path = frame_dir / "sampled_particles_xyz.npy", frame_dir / "reconstruction_metadata.json"
        particles = np.load(particles_path, allow_pickle=False)
        volume = float(json.loads(reconstruction_path.read_text())["object_volume_m3"])
        sequence = dynamics.load_observation_sequence(recording)
        frame_count = len(sequence.times)

        density = args.mass_kg / volume
        if not args.min_density_kg_m3 <= density <= args.max_density_kg_m3:
            reason = (f"implied density {density:.0f} kg/m^3 outside [{args.min_density_kg_m3:.0f}, "
                      f"{args.max_density_kg_m3:.0f}]: first-frame segmentation misses part of the dough")
            excluded_chunks.append({"id": name, "volume_m3": volume, "density_kg_m3": density, "reason": reason})
            print(f"{name}: excluded, {reason}", flush=True)
            continue

        config = json.loads(json.dumps(base))
        config["name"] = f"{name}-shared-eight-parameter-fit"
        config["paths"]["episode"] = str(recording.relative_to(REPOSITORY_ROOT))
        config["paths"]["initial_particles"] = str(particles_path.relative_to(REPOSITORY_ROOT))
        config["paths"]["reconstruction_metadata"] = str(reconstruction_path.relative_to(REPOSITORY_ROOT))
        config["expected_sha256"]["initial_particles"] = file_sha256(particles_path)
        config["expected_sha256"]["reconstruction_metadata"] = file_sha256(reconstruction_path)
        config["expected_sequence_fingerprint"] = sequence.fingerprint
        config["mass_kg"] = args.mass_kg
        config["density_kg_m3"] = args.mass_kg / volume
        config["loss"]["max_observed_points"] = 4096
        config["parameters"] = dict(INITIAL)
        config["fit_parameters"] = list(FIT)
        config["parameter_bounds"] = {key: list(value) for key, value in BOUNDS.items()}
        config["simulation"].update(plasticity="stretch-clamp", use_jp=False, jp_hardening=0.0, tool_collision="sdf",
                                    tool_contact_padding=TOOL_CONTACT_PADDING_M, tool_contact_model="coulomb-adhesive-v1",
                                    tool_contact_absorption=0.0, tool_friction_coefficient=INITIAL["tool_friction_coefficient"],
                                    tool_stickiness=INITIAL["tool_stickiness"])
        config["training"] = {"start_frame": 1, "end_frame": frame_count - 2, "stride": 1}
        config["validation"] = {"start_frame": frame_count - 1, "end_frame": frame_count - 1, "stride": 1}
        config_path = configs / f"{name}_stretch_clamp_v1.json"
        configs.mkdir(exist_ok=True)
        write_json(config_path, config)

        loaded = load_config(config_path)
        sim = SimulationConfig(**loaded.simulation)
        validate_initial_stencils(particles, sim)
        if (particles < 0).any() or (particles >= 1).any():
            raise ValueError(f"{name} particles leave the simulation domain")
        replay = dynamics.ToolReplay(sequence, calibration, 0, frame_count - 1, max_gap_s=base["replay_max_gap_s"],
                                     marker_from_tool_frames=geometry.marker_from_mesh)
        support = tool_support(replay, corners, dynamics)
        lower, upper = np.array(support["min"]), np.array(support["max"])
        if np.any(lower[[0, 2]] < 0) or np.any(upper[[0, 2]] >= 1) or upper[1] >= 1:
            raise ValueError(f"{name} tool motion exceeds the simulation domain: {support}")
        try:
            prepared = prepare_experiment(loaded, split="training", build_sdf=(kept == 0))
        except ValueError as error:
            if "valid pixels after filtering" not in str(error):
                raise
            config_path.unlink()
            excluded_chunks.append({"id": name, "volume_m3": volume, "density_kg_m3": density, "reason": str(error)})
            print(f"{name}: excluded, {error}", flush=True)
            continue
        kept += 1
        membership = "validation" if kept % args.validation_every == 0 else "training"
        episodes.append({"id": name, "config": f"configs/{config_path.name}", "membership": membership, "weight": 1.0,
                         "scored_window": {"start_frame": 1, "end_frame": frame_count - 1, "stride": 1}})
        rows.append({"id": name, "membership": membership, "frames": frame_count, "raw_first_frame": raw_index,
                     "initialization_points": init_report["dense_points_after_dbscan"], "volume_m3": volume,
                     "density_kg_m3": config["density_kg_m3"], "sequence_fingerprint": sequence.fingerprint,
                     "prepared_fingerprint": prepared.fingerprint, "tool_continuous_bounds": support})
        del prepared
        print(f"{name}: {frame_count} frames, raw {raw_index}, {len(frame)} init points, volume {volume * 1e6:.1f} cm^3, {membership}", flush=True)

    dataset_path = output / "dataset_episode11_dynamics_temporal_v1.json"
    write_json(dataset_path, {"schema": "taichidough/differentiable-dataset/v2",
                              "name": "episode11-dynamics-temporal-shared-eight-parameter-v1",
                              "shared_parameters": {"initial": {name: INITIAL[name] for name in FIT}, "fit": list(FIT),
                                                    "bounds": {key: list(value) for key, value in BOUNDS.items()}},
                              "episodes": episodes})
    if {path: file_sha256(Path(path)) for path in source_hashes} != source_hashes:
        raise ValueError("An input changed during preparation")
    write_json(output / "validation.json", {
        "schema": "taichidough/episode11-dynamics-temporal/v1/validation", "status": "inputs_valid",
        "dataset": str(dataset_path), "mass_kg": args.mass_kg, "validation_every": args.validation_every,
        "chunks": rows, "excluded_chunks": excluded_chunks, "source_hashes": source_hashes,
        "output_sha256": {str(p.relative_to(output)): file_sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "notes": ["Mass 0.35 kg reused from Episode 18 calibration per user authorization; not independently measured for Episode 11.",
                  "Per-chunk density = 0.35 kg / reconstructed_volume_m3.",
                  "Every chunk restarts from a reconstruction of its own first frame; frame 0 is initialization only.",
                  "Chunks are segments of one recording, so validation chunks are not independent recordings.",
                  "Contact model and initial values follow finalize_episode18_kugla_temporal.py.",
                  "First range (starting at frame 0) and last range (ending at final frame) excluded from fitting per Episode 18 protocol."],
        "simulation_run": False, "calibration_run": False})
    print(dataset_path, flush=True)
    return 0


def main(argv=None):
    data = REPOSITORY_ROOT.parent / "data/deformpath_training"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=data / "episode11_dynamics_sequence_v1")
    parser.add_argument("--conversion-metadata", type=Path, default=data / "episode11_dynamics_dough_hsv_export_v1/conversion_metadata.json")
    parser.add_argument("--dense-export", type=Path, default=data / "episode11_dynamics_dough_hsv_export_v1")
    parser.add_argument("--temporal-plan", type=Path, default=EXPERIMENT_ROOT / "data/episode11_dynamics_temporal_v1")
    parser.add_argument("--bundle", type=Path, default=EXPERIMENT_ROOT / "data/episode18_dynamics_icp_final_v1")
    parser.add_argument("--episode11-calibration-bundle", type=Path, default=None,
                        help="Episode 11-specific calibration bundle directory; overrides --bundle")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "data/episode11_dynamics_temporal_v1")
    parser.add_argument("--mass-kg", type=float, required=True)
    parser.add_argument("--validation-every", type=int, default=5, help="every Nth chunk is held out")
    parser.add_argument("--min-density-kg-m3", type=float, default=1000.0)
    parser.add_argument("--max-density-kg-m3", type=float, default=2000.0)
    parser.add_argument("--reference-policy", choices=("strict", "frozen"), default="strict")
    args = parser.parse_args(argv)
    for name in ("episode", "conversion_metadata", "dense_export", "temporal_plan", "bundle"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.episode11_calibration_bundle:
        args.episode11_calibration_bundle = args.episode11_calibration_bundle.expanduser().resolve()
    args.output_dir = require_owned_output(args.output_dir)
    with reference_policy(args.reference_policy):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
