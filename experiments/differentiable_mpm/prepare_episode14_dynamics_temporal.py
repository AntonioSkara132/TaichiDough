"""Prepare annotated Episode 14 chunks using filtered observations and fixed density.

First/last chunks are excluded. Each reconstruction uses its already processed
first cloud: HSV, dough-region filtering, subsampling, then DBSCAN. Mass is density times filled volume.
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

from .config import EXPERIMENT_ROOT, REPOSITORY_ROOT, FrameWindow, file_sha256, load_config
from .data import prepare_experiment, validate_initial_stencils
from .evaluate import fresh_directory, require_owned_output
from .finalize_episode18_kugla_temporal import BOUNDS, FIT, INITIAL, TOOL_CONTACT_PADDING_M
from .materialize_deformpath_ranges import materialize_ranges
from .prepare_table_aligned import tool_support, write_json
from .reference_adapter import get_reference_modules, reference_policy

ANNOTATION_NAME = "temporal_annotations.json"


def mass_from_density(volume_m3, density_kg_m3):
    if not np.isfinite([volume_m3, density_kg_m3]).all() or min(volume_m3, density_kg_m3) <= 0:
        raise ValueError("Reconstructed volume and density must be finite and positive")
    return float(volume_m3 * density_kg_m3)


def range_selection(plan, episode, conversion_metadata, annotation):
    selection = json.loads(json.dumps(plan))
    selection.pop("materialization_input_requirements", None)
    selection["name"] = "episode14-dynamics-temporal-v1"
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
    bundle_config_path = args.calibration_bundle
    if json.loads((bundle_config_path.parent / "validation.json").read_text()).get("status") not in ["inputs_valid", "calibration_prepared"]:
        raise ValueError("Bundle validation failed")
    base = json.loads(bundle_config_path.read_text())
    # Table bounds use a sparse measurement sequence; training keeps 30Hz replay
    # and must reject gaps over 0.1s instead of inheriting its measurement limit.
    base["replay_max_gap_s"] = 0.1
    observations = json.loads((args.episode / "observation_validation.json").read_text())
    if (observations.get("status") != "observations_valid" or observations.get("max_points") != 4096
            or observations.get("processing_order") != ["hsv", "dough_region", "subsample", "dbscan"]):
        raise ValueError("Expected verified HSV -> dough region -> subsample4096 -> DBSCAN observations")
    for name, digest in observations["output_sha256"].items():
        if file_sha256(args.episode / name) != digest:
            raise ValueError(f"Observation file changed after validation: {name}")
    calibration_path = REPOSITORY_ROOT / base["paths"]["calibration"]
    if file_sha256(calibration_path) != base["expected_sha256"]["calibration"]:
        raise ValueError("Bundle calibration differs from its pinned hash")
    plan = json.loads((args.temporal_plan / "range_selection.json").read_text())
    annotation_source = args.temporal_plan / ANNOTATION_NAME
    if file_sha256(annotation_source) != plan["annotation"]["sha256"]:
        raise ValueError("Temporal annotation differs from its range plan")
    sources = [bundle_config_path, calibration_path, annotation_source, args.temporal_plan / "range_selection.json",
               args.conversion_metadata,
               *(args.episode / name for name in ("pointclouds_interpolated.pt", "paths_interpolated.pt", "sequence_metadata.json", "observation_validation.json")),
               Path(__file__).resolve(), EXPERIMENT_ROOT / "materialize_deformpath_ranges.py",
               EXPERIMENT_ROOT / "prepare_episode18_dynamics_inputs.py",
               REPOSITORY_ROOT / "scripts/reconstruct_voxel_dough_from_deformpath.py"]
    source_hashes = {str(path): file_sha256(path) for path in sources}

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
    rows, episodes, failed_chunks, configs = [], [], [], output / "configs"
    kept = 0
    for index, chunk in enumerate(chunks, 1):
        name, recording = chunk["output_name"], Path(chunk["path"])
        metadata = json.loads((recording / "sequence_metadata.json").read_text())
        frame0 = torch.load(recording / "pointclouds_interpolated.pt", weights_only=True)[0][0].numpy()
        frame = np.ascontiguousarray(frame0, dtype=np.float32)
        raw_index = int(metadata["pointcloud_indices"][0])
        init_report = {"raw_frame": raw_index, "points_after_dbscan": len(frame),
                       "processing_order": ["hsv", "dough_region", "subsample", "dbscan"],
                       "source": "processed chunk first frame; no additional sampling or clustering"}
        init_dir = output / "initialization" / name
        init_dir.mkdir(parents=True)
        torch.save([[torch.from_numpy(frame)]], init_dir / "pointclouds_interpolated.pt")
        write_json(init_dir / "initialization_metadata.json", {"chunk": name, "sequence_pointclouds_sha256": source_hashes[str(args.episode / "pointclouds_interpolated.pt")],
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

        density = args.density_kg_m3
        mass_kg = mass_from_density(volume, density)

        config = json.loads(json.dumps(base))
        config["name"] = f"{name}-shared-eight-parameter-fit"
        config["paths"]["episode"] = str(recording.relative_to(REPOSITORY_ROOT))
        config["paths"]["initial_particles"] = str(particles_path.relative_to(REPOSITORY_ROOT))
        config["paths"]["reconstruction_metadata"] = str(reconstruction_path.relative_to(REPOSITORY_ROOT))
        config["expected_sha256"]["initial_particles"] = file_sha256(particles_path)
        config["expected_sha256"]["reconstruction_metadata"] = file_sha256(reconstruction_path)
        config["expected_sequence_fingerprint"] = sequence.fingerprint
        config["mass_kg"] = mass_kg
        config["density_kg_m3"] = density
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
            prepared = prepare_experiment(loaded, split="training", build_sdf=(kept == 0),
                                          scored_window=FrameWindow(1, frame_count - 1, 1))
        except ValueError as error:
            failed_chunks.append({"id": name, "raw_first_frame": raw_index, "frames": frame_count,
                                    "config": str(config_path), "reason": str(error),
                                    "action": "failed validation; no exclusion authorized"})
            print(f"{name}: validation failed: {error}", flush=True)
            continue
        sim_mass = prepared.simulation_config.particle_mass * len(particles)
        sim_volume = prepared.simulation_config.particle_volume * len(particles)
        np.testing.assert_allclose([sim_mass, sim_volume], [mass_kg, volume], rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(prepared.simulation_config.particle_mass /
                                   prepared.simulation_config.particle_volume, density, rtol=1e-10)
        kept += 1
        membership = "validation" if kept % args.validation_every == 0 else "training"
        episodes.append({"id": name, "config": f"configs/{config_path.name}", "membership": membership, "weight": 1.0,
                         "scored_window": {"start_frame": 1, "end_frame": frame_count - 1, "stride": 1}})
        rows.append({"id": name, "membership": membership, "frames": frame_count, "raw_first_frame": raw_index,
                     "initialization_points": init_report["points_after_dbscan"], "volume_m3": volume,
                     "density_kg_m3": config["density_kg_m3"], "mass_kg": mass_kg,
                     "particle_mass_kg": prepared.simulation_config.particle_mass,
                     "particle_volume_m3": prepared.simulation_config.particle_volume,
                     "mass_volume_density_consistent": True, "sequence_fingerprint": sequence.fingerprint,
                     "prepared_fingerprint": prepared.fingerprint, "tool_continuous_bounds": support})
        del prepared
        print(f"{name}: {frame_count} frames, raw {raw_index}, {len(frame)} init points, volume {volume * 1e6:.1f} cm^3, {membership}", flush=True)

    dataset_path = output / "dataset_episode14_dynamics_temporal_v1.json"
    if not failed_chunks:
        write_json(dataset_path, {"schema": "taichidough/differentiable-dataset/v2",
                                  "name": "episode14-dynamics-temporal-shared-eight-parameter-v1",
                                  "shared_parameters": {"initial": {name: INITIAL[name] for name in FIT}, "fit": list(FIT),
                                                        "bounds": {key: list(value) for key, value in BOUNDS.items()}},
                                  "episodes": episodes})
    if {path: file_sha256(Path(path)) for path in source_hashes} != source_hashes:
        raise ValueError("An input changed during preparation")
    write_json(output / "validation.json", {
        "schema": "taichidough/episode14-dynamics-temporal/v1/validation", "status": "inputs_invalid" if failed_chunks else "inputs_valid",
        "dataset": str(dataset_path), "density_kg_m3": args.density_kg_m3, "validation_every": args.validation_every,
        "chunks": rows, "failed_chunks": failed_chunks, "source_hashes": source_hashes,
        "output_sha256": {str(p.relative_to(output)): file_sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "notes": ["Density is 1200 kg/m3 per user instruction; chunk mass is density times reconstructed volume.",
                  "Each chunk restarts from its already filtered first cloud; frame 0 is initialization only.",
                  "HSV and calibrated dough-region filtering precede random subsampling to at most 4096, then largest-cluster DBSCAN.",
                  "No post-DBSCAN resampling or dense-cloud substitution is applied.",
                  "Validation chunks belong to the same recording, not independent episodes.",
                  f"Tool meshes are reused from episode18; marker-to-mesh transforms come from {base['paths']['tool_geometry']}.",
                  "Initial material/contact parameters are starting values, not fitted episode14 parameters.",
                  "First and last annotated ranges excluded by user instruction."],
        "simulation_run": False, "calibration_run": False})
    print(dataset_path, flush=True)
    return 1 if failed_chunks else 0


def main(argv=None):
    data = REPOSITORY_ROOT.parent / "data/deformpath_training"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=data / "episode14_dynamics_sequence_v4")
    parser.add_argument("--conversion-metadata", type=Path, default=data / "episode14_dynamics_hsv_roi4096_export_v4/conversion_metadata.json")
    parser.add_argument("--temporal-plan", type=Path, default=EXPERIMENT_ROOT / "data/episode14_dynamics_temporal_v1")
    parser.add_argument("--calibration-bundle", type=Path, required=True,
                        help="Episode-specific table-aligned experiment JSON")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "data/episode14_dynamics_density1200_roi_temporal_v1")
    parser.add_argument("--density-kg-m3", type=float, default=1200.0)
    parser.add_argument("--validation-every", type=int, default=5, help="every Nth chunk is held out")
    parser.add_argument("--reference-policy", choices=("strict", "frozen"), default="strict")
    args = parser.parse_args(argv)
    for name in ("episode", "conversion_metadata", "temporal_plan", "calibration_bundle"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not np.isfinite(args.density_kg_m3) or args.density_kg_m3 <= 0:
        parser.error("--density-kg-m3 must be finite and positive")
    args.output_dir = require_owned_output(args.output_dir)
    with reference_policy(args.reference_policy):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
