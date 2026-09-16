#!/usr/bin/env python3
"""Build a reproducible DeformPath episode for differentiable MPM.

The pipeline is intentionally non-destructive.  It can start from a ROS bag
directory or from an existing offline export and writes one self-contained
episode directory containing raw/interpolated observations, calibration,
floor diagnostics, an initial particle reconstruction, a smoke-test config,
and a preprocessing manifest.

Metric mocap output requires the moving AprilTag calibration produced by the
offline exporter.  Missing calibration is an error by default; use
``--allow-missing-calibration`` only to produce an export/interpolation
staging directory that is not ready for optimization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_TOOLS_ROOT = REPO_ROOT.parent / "data" / "deformpath_training"
EXPORTER_NAME = "export_deformpath2_offline.py"
INTERPOLATOR_NAME = "interpolate_deformpath_sequence.py"
HISTOGRAM = REPO_ROOT / "scripts" / "plot_pointcloud_histogram.py"
RECONSTRUCTOR = REPO_ROOT / "scripts" / "reconstruct_voxel_dough_from_deformpath.py"
CHUNKER = REPO_ROOT / "scripts" / "chunk_deformpath_episode.py"
DEFAULT_TOOL_GEOMETRY = REPO_ROOT / "configs" / "tool_geometry_episode18_sdf.json"
REGISTERED_ASSETS = REPO_ROOT / "experiments" / "differentiable_mpm" / "data" / "episode18_registered_tools_v1"
DEFAULT_COLLISION_MANIFEST = REGISTERED_ASSETS / "collision_manifest.json"
DEFAULT_UR_COLLISION = REGISTERED_ASSETS / "ur_spathla_collision_solid.stl"
DEFAULT_KINOVA_COLLISION = REGISTERED_ASSETS / "gen3_spathla_collision_solid.stl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def existing_file(path: Path, name: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def patch_floor_calibration(output_dir: Path, floor_y: float) -> dict[str, Any] | None:
    """Apply the declared floor to the exporter calibration and its metadata."""
    conversion_path = output_dir / "conversion_metadata.json"
    calibration_path = output_dir / "scene_calibration_v2.json"
    if not conversion_path.is_file() or not calibration_path.is_file():
        return None

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("schema") != "taichidough/scene-calibration/v2":
        raise ValueError("Exporter calibration is not scene-calibration/v2")
    if calibration.get("source_frame") != "mocap" or calibration.get("scene_frame") != "mocap":
        raise ValueError("Optimization preprocessing requires mocap source and scene frames")

    calibration["floor_plane_scene"] = [0.0, 1.0, 0.0, -float(floor_y)]
    provenance = dict(calibration.get("provenance", {}))
    provenance["floor_override"] = {
        "floor_y_m": float(floor_y),
        "method": "declared floor estimate applied by preprocess_deformpath_episode.py",
    }
    calibration["provenance"] = provenance
    json_write(calibration_path, calibration)

    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    record = dict(conversion.get("calibration", {}))
    record.update(
        status="available",
        path=str(calibration_path),
        sha256=sha256(calibration_path),
        fingerprint=canonical_hash(calibration),
        source_frame="mocap",
        scene_frame="mocap",
        floor_plane_scene=calibration["floor_plane_scene"],
    )
    conversion["calibration"] = record
    conversion["preprocessing_floor_y_m"] = float(floor_y)
    json_write(conversion_path, conversion)
    return record


def copy_raw_export(input_dir: Path, output_dir: Path) -> None:
    names = ("pointclouds.pt", "paths.pt", "conversion_metadata.json", "camera_info.json", "floor_estimate.json")
    for name in names:
        source = input_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
    calibration = input_dir / "scene_calibration_v2.json"
    if calibration.is_file():
        shutil.copy2(calibration, output_dir / calibration.name)


def build_smoke_config(args: argparse.Namespace, output_dir: Path, calibration: dict[str, Any],
                       reconstruction_metadata: dict[str, Any]) -> Path:
    geometry = existing_file(args.tool_geometry, "tool geometry")
    collision_manifest = existing_file(args.collision_manifest, "collision manifest")
    ur_collision = existing_file(args.ur_collision_mesh, "UR collision mesh")
    kinova_collision = existing_file(args.kinova_collision_mesh, "Kinova collision mesh")
    calibration_path = output_dir / "scene_calibration_v2.json"
    particles = output_dir / "reconstruction" / output_dir.name / f"frame_{args.frame:04d}" / "sampled_particles_xyz.npy"
    metadata = output_dir / "reconstruction" / output_dir.name / f"frame_{args.frame:04d}" / "reconstruction_metadata.json"
    if not particles.is_file() or not metadata.is_file():
        raise FileNotFoundError("Reconstruction outputs are missing")

    volume = float(reconstruction_metadata["object_volume_m3"])
    mass = float(args.mass_kg) if args.mass_kg is not None else float(args.density * volume)
    camera = calibration["camera"]
    frame_count = int(json.loads((output_dir / "sequence_metadata.json").read_text())["num_valid_samples"])
    train_end = max(2, min(60, frame_count - 2))
    validation_start = train_end + 1
    validation_end = max(validation_start, min(validation_start + 36, frame_count - 1))

    paths = {
        "episode": str(output_dir),
        "calibration": str(calibration_path),
        "initial_particles": str(particles),
        "reconstruction_metadata": str(metadata),
        "tool_geometry": str(geometry),
        "collision_manifest": str(collision_manifest),
        "ur_collision_mesh": str(ur_collision),
        "kinova_collision_mesh": str(kinova_collision),
    }
    expected = {name: sha256(Path(path)) for name, path in paths.items() if name != "episode"}
    config = {
        "schema": "taichidough/differentiable-mpm-experiment/v1",
        "name": f"{output_dir.name}-dynamics-smoke",
        "paths": paths,
        "expected_sha256": expected,
        "simulation": {
            "physics_version": "corrected-v1",
            "n_particles": int(args.num_particles),
            "grid": 48,
            "dt": 0.0002,
            "gravity": -9.81,
            "floor_y": float(args.floor_y),
            "plasticity": "none",
            "use_jp": False,
            "jp_hardening": 0.0,
            "jp_min": 0.5,
            "jp_max": 2.0,
            "tool_collision": "sdf",
            "tool_contact_padding": 0.0026041666666666665,
            "tool_contact_model": "coulomb-v1",
            "tool_friction_coefficient": 0.5,
            "tool_contact_absorption": 0.0,
            "tool_stickiness": 0.0,
            "floor_absorption": 0.0,
            "floor_stickiness": 0.0,
            "floor_plastic_damping_band": 0.0,
            "velocity_damping": 1.0,
            "plastic_velocity_damping": 1.0,
            "plastic_affine_damping": 1.0,
            "precision": "f64",
            "p2g_mode": "atomic",
        },
        "mass_kg": mass,
        "density_kg_m3": float(args.density),
        "parameters": {
            "youngs_modulus": 130579.320726,
            "poisson_ratio": 0.3,
            "viscosity": 0.0,
            "plastic_min": 0.9,
            "plastic_max": 1.1,
            "tool_retention": 0.2,
            "floor_retention": 0.4,
            "tool_friction_coefficient": 0.5,
            "tool_stickiness": 0.0,
        },
        "fit_parameters": ["youngs_modulus", "poisson_ratio", "viscosity"],
        "parameter_bounds": {
            "youngs_modulus": [10000.0, 300000.0],
            "poisson_ratio": [0.15, 0.45],
            "viscosity": [0.0, 100.0],
        },
        "training": {"start_frame": 1, "end_frame": train_end, "stride": 1},
        "validation": {"start_frame": validation_start, "end_frame": validation_end, "stride": 1},
        "observation": {"width": 160, "height": 120, "splat_radius": 1, "strict_splat_radius": 1, "trim_quantile": 0.005},
        "loss": {
            "version": "partial-visible-splats-v2",
            "footprint_radius": 2,
            "visibility_temperature_m": 0.005,
            "depth_scale_m": 0.01,
            "distance_scale_m": 0.01,
            "huber_delta": 1.0,
            "depth_weight": 1.0,
            "coverage_weight": 0.5,
            "distance_weight": 0.5,
            "min_observed_pixels": 4,
            "max_observed_points": 1024,
            "min_predicted_pixels": 1,
        },
        "strict_loss": {
            "weights": {"depth_change": 1.0, "mask_iou": 1.0, "observed_to_simulation_distance": 1.0, "real_coverage": 1.0},
            "depth_scale_m": 0.01,
            "distance_scale_m": 0.01,
            "huber_delta": 1.0,
            "min_common_pixels": 50,
            "min_observed_pixels": 50,
            "min_simulation_pixels": 50,
            "min_observed_points": 50,
            "min_simulation_points": 50,
            "nearest_chunk_size": 1024,
            "require_all_frames": True,
        },
        "replay_max_gap_s": 0.1,
        "tool_sdf_resolution": 64,
        "tool_mesh_scale": 0.001,
        "backend": "cpu",
        "segment_length": 64,
        "seed": int(args.seed),
        "optimizer": {},
    }
    config_path = output_dir / "differentiable_mpm_smoke.json"
    json_write(config_path, config)
    return config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bag-dir", type=Path, help="ROS 2 bag directory containing metadata.yaml")
    source.add_argument("--input-dir", type=Path, help="Existing raw exporter directory containing pointclouds.pt and paths.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-tools-dir", type=Path, default=DATA_TOOLS_ROOT)
    parser.add_argument("--calibration", type=Path,
                        help="Explicit metric scene_calibration/v2 JSON. Useful when an existing export lacks attached calibration.")
    parser.add_argument("--pointcloud-topic", default="/camera/camera/depth/color/points")
    parser.add_argument("--camera-info-topic", default="/camera/camera/color/camera_info")
    parser.add_argument("--pose-topic", default="/poses")
    parser.add_argument("--output-frame", choices=("mocap", "camera"), default="mocap")
    parser.add_argument("--camera-tag-parent-frame", default="tag16h5:3")
    parser.add_argument("--camera-tag-frame", default="tag3_real")
    parser.add_argument("--mocap-tag-frame", default="apriltag3")
    parser.add_argument("--mocap-frame", default="mocap")
    parser.add_argument("--camera-frame", default="camera_link")
    parser.add_argument("--floor-y", type=float, default=0.0368)
    parser.add_argument("--density", type=float, default=1200.0)
    parser.add_argument("--mass-kg", type=float, default=None, help="Measured mass; defaults to density * reconstructed volume")
    parser.add_argument("--frame", type=int, default=0, help="Interpolated frame used for initial reconstruction")
    parser.add_argument("--num-particles", type=int, default=24000)
    parser.add_argument("--voxel-size", type=float, default=0.003)
    parser.add_argument("--max-raw-points", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trim-quantile", type=float, default=0.005)
    parser.add_argument("--floor-clearance", type=float, default=0.003)
    parser.add_argument("--floor-min-thickness", type=float, default=0.001)
    parser.add_argument("--floor-max-thickness", type=float, default=0.25)
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Optional aligned-frame chunk size; 0 leaves one canonical sequence")
    parser.add_argument("--ranges-file", type=Path, default=None,
                        help="Optional contiguous raw half-open ranges JSON; overrides chunk-size")
    parser.add_argument("--chunks-dir", type=Path, default=None,
                        help="Optional output directory for frame chunks; defaults to OUTPUT/chunks")
    parser.add_argument("--tool-geometry", type=Path, default=DEFAULT_TOOL_GEOMETRY)
    parser.add_argument("--collision-manifest", type=Path, default=DEFAULT_COLLISION_MANIFEST)
    parser.add_argument("--ur-collision-mesh", type=Path, default=DEFAULT_UR_COLLISION)
    parser.add_argument("--kinova-collision-mesh", type=Path, default=DEFAULT_KINOVA_COLLISION)
    parser.add_argument("--allow-missing-calibration", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.floor_y) or not math.isfinite(args.density) or args.density <= 0 \
            or not math.isfinite(args.voxel_size) or args.voxel_size <= 0:
        raise ValueError("floor-y must be finite; density and voxel-size must be finite and positive")
    if args.frame < 0 or args.num_particles < 1 or args.max_points < 1 or args.max_raw_points < 0:
        raise ValueError("frame, particle count, and point limits are invalid")
    if args.chunk_size == 1 or args.chunk_size < 0:
        raise ValueError("chunk-size must be 0 or an integer >= 2")
    if args.chunk_size and args.ranges_file is not None:
        raise ValueError("Use either chunk-size or ranges-file, not both")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty; use a new directory or --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tools_dir = args.data_tools_dir.expanduser().resolve()
    exporter = existing_file(tools_dir / EXPORTER_NAME, "offline exporter")
    interpolator = existing_file(tools_dir / INTERPOLATOR_NAME, "sequence interpolator")

    if args.bag_dir is not None:
        bag_dir = existing_file(args.bag_dir / "metadata.yaml", "ROS bag metadata").parent
        export_command = [sys.executable, str(exporter), "--bag-dir", str(bag_dir), "--output-dir", str(output_dir),
                          "--pointcloud-topic", args.pointcloud_topic, "--camera-info-topic", args.camera_info_topic,
                          "--pose-topic", args.pose_topic, "--output-frame", args.output_frame,
                          "--camera-tag-parent-frame", args.camera_tag_parent_frame,
                          "--camera-tag-frame", args.camera_tag_frame, "--mocap-tag-frame", args.mocap_tag_frame,
                          "--mocap-frame", args.mocap_frame, "--camera-frame", args.camera_frame,
                          "--max-points", str(args.max_raw_points), "--seed", str(args.seed)]
        run(export_command)
    else:
        input_dir = args.input_dir.expanduser().resolve()
        if not (input_dir / "pointclouds.pt").is_file() or not (input_dir / "paths.pt").is_file():
            raise FileNotFoundError("input-dir must contain pointclouds.pt and paths.pt")
        if input_dir != output_dir:
            copy_raw_export(input_dir, output_dir)

    if args.calibration is not None:
        source_calibration = existing_file(args.calibration, "explicit calibration")
        destination_calibration = output_dir / "scene_calibration_v2.json"
        if source_calibration != destination_calibration.resolve():
            shutil.copy2(source_calibration, destination_calibration)

    calibration_record = patch_floor_calibration(output_dir, args.floor_y)
    if calibration_record is None and not args.allow_missing_calibration:
        raise RuntimeError("No metric scene_calibration_v2.json was produced. Export tagged mocap data or use --allow-missing-calibration for staging only.")

    interpolation_command = [sys.executable, str(interpolator), "--input-dir", str(output_dir), "--output-dir", str(output_dir),
                             "--paths-format", "episode-tensor", "--max-points", str(args.max_points), "--seed", str(args.seed)]
    run(interpolation_command)

    raw_points = output_dir / "pointclouds.pt"
    histogram_path = output_dir / "diagnostics" / "raw_y_histogram.png"
    run([sys.executable, str(HISTOGRAM), "--input", str(raw_points), "--output", str(histogram_path),
         "--bins", "120", "--low-quantile", "0.01", "--value-range", "0", "0.25"])
    floor_estimate = {
        "schema": "taichidough/floor-estimate/v1",
        "status": "declared",
        "coordinate_frame": "mocap" if args.output_frame == "mocap" else "camera_link",
        "floor_y_m": float(args.floor_y),
        "floor_plane_scene": [0.0, 1.0, 0.0, -float(args.floor_y)],
        "method": "declared floor value; raw y histogram retained for audit",
        "source_pointclouds": str(raw_points),
        "source_histogram": str(histogram_path),
    }
    json_write(output_dir / "floor_estimate.json", floor_estimate)

    chunk_manifest_path = None
    if args.chunk_size or args.ranges_file is not None:
        chunks_dir = (args.chunks_dir.expanduser().resolve() if args.chunks_dir is not None
                      else output_dir / "chunks")
        chunk_command = [sys.executable, str(CHUNKER), "--input-dir", str(output_dir),
                         "--output-dir", str(chunks_dir), "--overwrite"]
        if args.ranges_file is not None:
            chunk_command.extend(["--ranges-file", str(args.ranges_file.expanduser().resolve())])
        else:
            chunk_command.extend(["--chunk-size", str(args.chunk_size)])
        run(chunk_command)
        chunk_manifest_path = chunks_dir / "chunk_manifest.json"

    config_path = None
    reconstruction_metadata = None
    if calibration_record is not None:
        calibration_path = output_dir / "scene_calibration_v2.json"
        reconstruction_command = [sys.executable, str(RECONSTRUCTOR), "--episode-dir", str(output_dir),
                                  "--pointclouds-name", "pointclouds_interpolated.pt", "--frame", str(args.frame),
                                  "--output-dir", str(output_dir / "reconstruction"), "--num-particles", str(args.num_particles),
                                  "--voxel-size", str(args.voxel_size), "--fill-mode", "floor", "--fill-axis", "y",
                                  "--fill-direction", "negative", "--floor-clearance", str(args.floor_clearance),
                                  "--floor-min-thickness", str(args.floor_min_thickness),
                                  "--floor-max-thickness", str(args.floor_max_thickness), "--calibration", str(calibration_path), "--save-pt"]
        run(reconstruction_command)
        reconstruction_dir = output_dir / "reconstruction" / output_dir.name / f"frame_{args.frame:04d}"
        reconstruction_metadata = json.loads((reconstruction_dir / "reconstruction_metadata.json").read_text(encoding="utf-8"))
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        config_path = build_smoke_config(args, output_dir, calibration, reconstruction_metadata)

    files = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "preprocessing_manifest.json":
            files[str(path.relative_to(output_dir))] = sha256(path)
    manifest = {
        "schema": "taichidough/preprocessed-deformpath-episode/v1",
        "status": "ready_for_optimization" if config_path else "staging_missing_metric_calibration",
        "source": {"bag_dir": str(args.bag_dir.resolve()) if args.bag_dir else None,
                    "input_dir": str(args.input_dir.resolve()) if args.input_dir else None},
        "output_dir": str(output_dir),
        "floor_y_m": float(args.floor_y),
        "density_kg_m3": float(args.density),
        "initial_frame": int(args.frame),
        "calibration": calibration_record,
        "reconstruction": reconstruction_metadata,
        "smoke_config": str(config_path) if config_path else None,
        "chunk_manifest": str(chunk_manifest_path) if chunk_manifest_path else None,
        "files_sha256": files,
    }
    json_write(output_dir / "preprocessing_manifest.json", manifest)
    print(f"Preprocessing status: {manifest['status']}")
    print(f"Output: {output_dir}")
    if config_path:
        print(f"Smoke config: {config_path}")


if __name__ == "__main__":
    main()
