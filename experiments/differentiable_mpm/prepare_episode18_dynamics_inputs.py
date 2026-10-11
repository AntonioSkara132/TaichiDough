"""Generate table-aligned Episode 18 dynamics inputs for the ICP-refined paths (no simulation).

The sequence (observed dough clouds and refined marker paths) is used unchanged. The
initial particles come from a separate one-frame episode: raw frame ``pointcloud_indices[0]``
of a full-density dough HSV export, filtered with the sequence's own DBSCAN settings,
because the 4096-point sequence clouds are too sparse for the voxel floor fill.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .config import EXPERIMENT_ROOT, REPOSITORY_ROOT, file_sha256, load_config
from .data import prepare_experiment, validate_initial_stencils
from .evaluate import fresh_directory, require_owned_output
from .prepare_registered_tools import topology
from .prepare_table_aligned import bounds, move, tool_support, write_json
from .reference_adapter import get_reference_modules, reference_policy
from .table_alignment import GRAVITY_ASSUMPTION, compose_table_calibration, table_alignment_transform

DEFORMPATH_TRAINING = REPOSITORY_ROOT.parent / "data/deformpath_training"
MESH_SETTINGS = {"UR5e_spathla": ("ur_collision_mesh", "UR_TOOL_VISUAL_ORIGIN", "UR_TOOL_VISUAL_RPY"),
                 "gen3_spathla": ("kinova_collision_mesh", "KINOVA_TOOL_VISUAL_ORIGIN", "KINOVA_TOOL_VISUAL_RPY")}
DENSITY_KG_M3 = 1200.0


def tool_frame_vertices(simulator, name, mesh_path, scale):
    _, origin_key, rpy_key = MESH_SETTINGS[name]
    vertices, _ = simulator.load_binary_stl(mesh_path, scale)
    return simulator.mesh_in_tool_frame(vertices, getattr(simulator, origin_key),
                                        simulator.rpy_to_matrix(getattr(simulator, rpy_key)))


def padded_corners(vertices, padding):
    lower, upper = vertices.min(axis=0) - padding, vertices.max(axis=0) + padding
    return np.array(list(itertools.product(*zip(lower, upper))))


def initialization_frame(dense_path, sequence_metadata, sequence_frame0):
    """Raw dense frame for sequence frame 0, filtered like interpolate_deformpath_pairs.py."""
    sys.path.insert(0, str(DEFORMPATH_TRAINING))
    from interpolate_deformpath_pairs import largest_dbscan_cluster_mask
    import torch
    raw_index = int(sequence_metadata["pointcloud_indices"][0])
    settings = sequence_metadata["pointcloud_filter"]
    frame = torch.load(dense_path, map_location="cpu", weights_only=True, mmap=True)[0][raw_index].numpy()
    xyz = frame[:, :3]
    frame = frame[np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)]
    keep, cluster_count = largest_dbscan_cluster_mask(frame[:, :3], eps=settings["dbscan_eps"], min_samples=settings["dbscan_min_samples"])
    frame = np.ascontiguousarray(frame[keep, :sequence_frame0.shape[1]], dtype=np.float32)
    # Column 4 is the raw frame number and column 6 the relative timestamp of the sequence.
    if not (np.all(frame[:, 4] == sequence_frame0[0, 4]) and np.allclose(frame[:, 6], sequence_frame0[0, 6], atol=1e-6, rtol=0)):
        raise ValueError("Dense initialization frame is not the sequence's frame 0")
    from scipy.spatial import cKDTree
    distance = cKDTree(frame[:, :3]).query(sequence_frame0[:, :3])[0]
    return frame, raw_index, {"raw_frame": raw_index, "dense_points_after_dbscan": len(frame),
                              "sequence_frame0_points": len(sequence_frame0), "dbscan_cluster_count": int(cluster_count),
                              "sequence_to_dense_distance_m": {"max": float(distance.max()), "median": float(np.median(distance))}}


def run(args):
    plane_summary = json.loads(args.plane_summary.read_text())
    verification_path = args.plane_summary.with_name("verification.json")
    verification = json.loads(verification_path.read_text())
    if verification.get("passed") is not True:
        raise ValueError("Table-plane estimate has not passed verification")
    if verification.get("output_sha256", {}).get("summary.json") != file_sha256(args.plane_summary):
        raise ValueError("Plane summary differs from its verified output hash")
    if Path(plane_summary["episode"]).resolve() != args.episode:
        raise ValueError("Plane estimate describes a different sequence")
    for path, expected in plane_summary["input_hashes"].items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f"Plane-estimation input changed: {path}")
    reconstruction_script = REPOSITORY_ROOT / "scripts/reconstruct_voxel_dough_from_deformpath.py"
    calibration_source = args.episode / "scene_calibration_v2.json"
    static_path = args.episode / "spatula_static_calibration.json"
    sequence_metadata_path = args.episode / "sequence_metadata.json"
    dense_path = args.dense_export / "pointclouds.pt"
    dense_metadata_path = args.dense_export / "conversion_metadata.json"
    source_paths = {args.base_config, args.plane_summary, verification_path, calibration_source, static_path,
                    sequence_metadata_path, args.episode / "paths_interpolated.pt",
                    args.episode / "pointclouds_interpolated.pt", dense_path, dense_metadata_path,
                    args.ur_mesh, args.kinova_mesh, Path(__file__).resolve(), EXPERIMENT_ROOT / "table_alignment.py",
                    EXPERIMENT_ROOT / "data.py", reconstruction_script, REPOSITORY_ROOT / "scripts/deformpath_topview.py",
                    DEFORMPATH_TRAINING / "interpolate_deformpath_pairs.py"}
    source_hashes = {str(path): file_sha256(Path(path)) for path in sorted(map(str, source_paths))}
    output = fresh_directory(args.output_dir)
    helpers = get_reference_modules()
    topview, dynamics, simulator = helpers.topview, helpers.dynamics, helpers.simulator
    base_document = json.loads(args.base_config.read_text())
    scale = float(base_document["tool_mesh_scale"])
    padding = float(base_document["simulation"]["tool_contact_padding"])
    sequence = dynamics.load_observation_sequence(args.episode)
    sequence_metadata = json.loads(sequence_metadata_path.read_text())
    if tuple(sequence.names) != tuple(MESH_SETTINGS):
        raise ValueError(f"Unexpected tool stream order {sequence.names}")

    # Camera calibrations must agree, otherwise dense and sequence clouds sit in different frames.
    dense_calibration = json.loads((args.dense_export / "scene_calibration_v2.json").read_text())
    old_document = json.loads(calibration_source.read_text())
    for key in ("scene_from_camera", "scene_from_source"):
        if not np.array_equal(np.asarray(dense_calibration[key]), np.asarray(old_document[key])):
            raise ValueError(f"Dense export {key} differs from the sequence calibration")

    # Collision meshes, manifest and tool geometry.
    static = json.loads(static_path.read_text())["tools"]
    mesh_sources = {"UR5e_spathla": args.ur_mesh, "gen3_spathla": args.kinova_mesh}
    rows, tools, corners, mesh_paths = [], [], [], {}
    for name in sequence.names:
        target = output / mesh_sources[name].name
        shutil.copyfile(mesh_sources[name], target)
        audit, mesh_bounds = topology(target)
        rows.append(dict(name=name, collision_mesh=target.name, collision_sha256=file_sha256(target),
                         units="millimetres", coordinate_frame="raw_stl_visual", topology=audit, bounds_mm=mesh_bounds,
                         source_mesh=str(mesh_sources[name]), source_sha256=source_hashes[str(mesh_sources[name])]))
        marker_from_mesh = np.asarray(static[name]["marker_from_mesh"], dtype=np.float64)
        vertices = tool_frame_vertices(simulator, name, target, scale)
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        center = np.eye(4)
        center[:3, 3] = (lower + upper) / 2
        tools.append({"name": name, "half_extents_m": ((upper - lower) / 2).tolist(),
                      "marker_from_collider": (marker_from_mesh @ center).tolist(),
                      "marker_from_mesh": marker_from_mesh.tolist()})
        corners.append(padded_corners(vertices, padding))
        mesh_paths[MESH_SETTINGS[name][0]] = target
    write_json(output / "collision_manifest.json", {"schema": "taichidough/tool-collision-meshes/v1", "tools": rows})
    write_json(output / "tool_geometry.json", {"schema": "taichidough/tool-geometry/v1", "tools": tools,
        "note": "marker_from_mesh is the ICP static calibration (tool-link frame). The box proxy is the "
                "tool-link-frame AABB of the closed collision mesh and is only used when tool_collision is not sdf."})
    geometry = dynamics.load_tool_geometry(output / "tool_geometry.json", sequence.names)

    # Table-aligned calibration with the complete motion centred at X=Z=0.5.
    old_calibration = topview.load_calibration(calibration_source)
    old_replay = dynamics.ToolReplay(sequence, old_calibration, 0, len(sequence.times) - 1,
                                     max_gap_s=base_document["replay_max_gap_s"],
                                     marker_from_tool_frames=geometry.marker_from_mesh)
    trim = base_document["observation"]["trim_quantile"]
    old_clouds = [topview.apply_calibration(topview.filter_xyz(points, trim), old_calibration) for points in sequence.points]
    plane = np.asarray(plane_summary["estimated_plane_scene"], dtype=float)
    transform = table_alignment_transform(plane)
    old_boxes = np.concatenate([corners[j] @ dynamics.quaternion_matrix(p[j, 3:7]).T + p[j, :3]
                                for p in old_replay.poses for j in range(2)])
    combined = np.concatenate([*(move(points, transform) for points in old_clouds), move(old_boxes, transform)])
    translation_xz = 0.5 - (combined.min(axis=0) + combined.max(axis=0))[[0, 2]] / 2
    transform = table_alignment_transform(plane, translation_xz=translation_xz)
    calibration_path = output / "scene_calibration_v2.json"
    write_json(calibration_path, compose_table_calibration(
        old_document, plane, translation_xz=translation_xz,
        provenance={"plane_summary": str(args.plane_summary), "plane_summary_sha256": file_sha256(args.plane_summary),
                    "source_calibration": str(calibration_source), "source_calibration_sha256": source_hashes[str(calibration_source)],
                    "horizontal_placement": "center complete observed motion and padded tool mesh bounds at X=Z=0.5"}))
    calibration = topview.load_calibration(calibration_path)
    replay = dynamics.ToolReplay(sequence, calibration, 0, len(sequence.times) - 1,
                                 max_gap_s=base_document["replay_max_gap_s"],
                                 marker_from_tool_frames=geometry.marker_from_mesh)
    clouds = [topview.apply_calibration(topview.filter_xyz(points, trim), calibration) for points in sequence.points]
    max_point_error = max(float(np.max(np.abs(new - move(old, transform)))) for new, old in zip(clouds, old_clouds))
    pose_error = float(np.max(np.abs(replay.poses[:, :, :3] - move(old_replay.poses[:, :, :3], transform))))
    if max_point_error > 3e-7 or pose_error > 3e-7:
        raise ValueError("Observation or tool composition differs from the common transform")
    print(f"Table-aligned calibration written; translation_xz={translation_xz.tolist()}", flush=True)

    # One-frame initialization episode from the dense export.
    init_dir = output / "initialization_episode"
    init_dir.mkdir()
    import torch
    sequence_frame0 = torch.load(args.episode / "pointclouds_interpolated.pt", weights_only=True)[0][0].numpy()
    frame, raw_index, init_report = initialization_frame(dense_path, sequence_metadata, sequence_frame0)
    torch.save([[torch.from_numpy(frame)]], init_dir / "pointclouds_interpolated.pt")
    write_json(init_dir / "initialization_metadata.json", {
        "schema": "taichidough/initialization-frame/v1", "sequence": str(args.episode), "dense_export": str(args.dense_export),
        "dense_pointclouds_sha256": source_hashes[str(dense_path)], "dense_conversion_metadata_sha256": source_hashes[str(dense_metadata_path)],
        "filter": {"hsv": {k: sequence_metadata["pointcloud_filter"][k] for k in ("hue_range", "saturation_range", "value_range")},
                   "spatial": "finite and z > 0", "dbscan_eps": sequence_metadata["pointcloud_filter"]["dbscan_eps"],
                   "dbscan_min_samples": sequence_metadata["pointcloud_filter"]["dbscan_min_samples"], "max_points": None},
        "columns": "same 7 columns as the sequence pointclouds_interpolated.pt", **init_report})
    print(f"Initialization frame: raw {raw_index}, {len(frame)} points", flush=True)

    reconstruction = base_document  # settings follow the Episode 18 kugla reconstruction
    command = [sys.executable, str(reconstruction_script), "--episode-dir", str(init_dir),
               "--pointclouds-name", "pointclouds_interpolated.pt", "--frame", "0",
               "--calibration", str(calibration_path), "--output-dir", str(output / "reconstruction"),
               "--num-particles", str(reconstruction["simulation"]["n_particles"]), "--voxel-size", str(args.voxel_size),
               "--fill-mode", "floor", "--fill-axis", "y", "--fill-direction", "negative",
               "--floor-clearance", "0.003", "--floor-min-thickness", "0.001", "--floor-max-thickness", "0.25",
               "--bbox-padding", "0.006", "--trim-quantile", str(trim), "--footprint-dilate", "1",
               "--seed", str(reconstruction["seed"])]
    write_json(output / "reconstruction_command.json", {"argv": command})
    with (output / "reconstruction.log").open("x") as log:
        subprocess.run(command, cwd=REPOSITORY_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    frame_dir = output / "reconstruction" / init_dir.name / "frame_0000"
    particle_path = frame_dir / "sampled_particles_xyz.npy"
    metadata_path = frame_dir / "reconstruction_metadata.json"
    particles = np.load(particle_path, allow_pickle=False)
    reconstructed = simulator.load_reconstruction_metadata(metadata_path, particle_path, particles, calibration)
    mass = simulator.compute_mass_properties(len(particles), reconstruction["simulation"]["grid"], DENSITY_KG_M3,
                                             reconstructed["object_volume_m3"], None)
    write_json(output / "mass_properties.json", mass)

    config_document = json.loads(args.base_config.read_text())
    config_document["name"] = "episode18-dynamics-icp-final-table-aligned"
    config_document["paths"] = {"episode": os.path.relpath(args.episode, REPOSITORY_ROOT)}
    for key, path in [("calibration", calibration_path), ("initial_particles", particle_path),
                      ("reconstruction_metadata", metadata_path), ("tool_geometry", output / "tool_geometry.json"),
                      ("collision_manifest", output / "collision_manifest.json"), *mesh_paths.items()]:
        config_document["paths"][key] = str(path.relative_to(REPOSITORY_ROOT))
    config_document["expected_sha256"] = {key: file_sha256(REPOSITORY_ROOT / value)
                                          for key, value in config_document["paths"].items() if key != "episode"}
    config_document["expected_sequence_fingerprint"] = sequence.fingerprint
    config_document["mass_kg"] = mass["total_mass_kg"]
    config_document["density_kg_m3"] = mass["density_kg_m3"]
    # The UR blade is thinner than two 64-grid voxels, which leaves its SDF without interior voxels.
    config_document["tool_sdf_resolution"] = args.tool_sdf_resolution
    config_path = output / "episode18_dynamics_icp_final.json"
    write_json(config_path, config_document)
    config = load_config(config_path)

    point_filter = dynamics.load_scene_point_filter(
        {"initialization": {"metadata_path": str(metadata_path), "metadata_sha256": file_sha256(metadata_path)}},
        metadata_path, calibration)
    retained, counts = [], []
    for points in clouds:
        selected, count = dynamics.filter_scene_points(points, point_filter)
        retained.append(selected)
        counts.append(count)
    selected_points = np.concatenate(retained)
    from .state import SimulationConfig
    sim = SimulationConfig(**config.simulation)
    validate_initial_stencils(particles, sim)
    validate_initial_stencils(selected_points, sim)
    if (particles < 0).any() or (particles >= 1).any() or (selected_points < 0).any() or (selected_points >= 1).any():
        raise ValueError("Particles or retained observations leave the simulation domain")
    support = tool_support(replay, corners, dynamics)
    lower, upper = np.array(support["min"]), np.array(support["max"])
    if np.any(lower[[0, 2]] < 0) or np.any(upper[[0, 2]] >= 1) or upper[1] >= 1:
        raise ValueError(f"Complete tool motion exceeds the simulation domain: {support}")
    print(f"Reconstruction: {mass['object_volume_m3']:.9g} m^3; mass={config_document['mass_kg']:.6f} kg.", flush=True)
    prepared_reports = {}
    for split in ("training", "validation"):
        prepared = prepare_experiment(config, split=split, build_sdf=(split == "training"))
        prepared_reports[split] = {"fingerprint": prepared.fingerprint, "total_steps": prepared.total_steps,
            "scored_frames": list(prepared.scored_frames), "observation_count": len(prepared.observations),
            "sdf_built": prepared.sdf is not None,
            "controls_finite": bool(np.isfinite(prepared.controls.poses).all() and np.isfinite(prepared.controls.velocities).all())}
        del prepared
    if {path: file_sha256(Path(path)) for path in source_hashes} != source_hashes:
        raise ValueError("An input or generation source changed during reconstruction/validation")
    outputs = {str(path.relative_to(output)): file_sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "validation.json", {
        "schema": "taichidough/table-aligned-inputs/v1", "status": "inputs_valid", "config": str(config_path),
        "source_hashes": source_hashes, "sources_unchanged": True, "output_sha256": outputs,
        "transform": transform.tolist(), "gravity_assumption": GRAVITY_ASSUMPTION,
        "recorded_frame_count": len(sequence.times), "initialization": init_report, "mass": mass,
        "particle_bounds": bounds(particles), "observed_bounds_before_clearance": bounds(np.concatenate(clouds)),
        "observed_bounds_after_clearance": bounds(selected_points), "observation_filter_counts": counts,
        "tool_continuous_bounds": support, "initial_particle_stencils_valid": True,
        "retained_observation_stencils_valid": True, "observation_transform_max_error_m": max_point_error,
        "tool_position_max_error_m": pose_error, "prepared_splits": prepared_reports,
        "simulation_run": False, "gradient_test_run": False, "calibration_run": False,
        "warnings": [(f"Density is the measured {args.mass_kg} kg mass over the visible-top-to-floor reconstructed volume." if args.mass_kg else
                      f"Mass is assumed density {DENSITY_KG_M3} kg/m^3 times the visible-top-to-floor reconstructed volume; no measured mass."),
                     "Tools are near the dough from frame 0, and the dough changes by about 4 mm by frame 3.",
                     "Initial material values are inherited from the Episode 18 kugla configuration, not fitted to this recording."]})
    print(f"Input validation passed. Configuration: {config_path}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True, help="sequence directory (AlternativeDynamics_calibrated_icp_final)")
    parser.add_argument("--dense-export", type=Path, required=True, help="full-density dough HSV export directory")
    parser.add_argument("--plane-summary", type=Path, required=True)
    parser.add_argument("--ur-mesh", type=Path, required=True)
    parser.add_argument("--kinova-mesh", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=EXPERIMENT_ROOT / "configs/episode18_table_aligned_registered_tools.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mass-kg", type=float, help="measured dough mass; density then follows from the reconstructed volume")
    parser.add_argument("--tool-sdf-resolution", type=int, default=128)
    parser.add_argument("--voxel-size", type=float, default=0.003)
    parser.add_argument("--reference-policy", choices=("strict", "frozen"), default="strict")
    args = parser.parse_args(argv)
    for name in ("episode", "dense_export", "plane_summary", "ur_mesh", "kinova_mesh", "base_config"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output_dir = require_owned_output(args.output_dir)
    with reference_policy(args.reference_policy):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
