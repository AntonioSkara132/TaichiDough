"""Inputs for preparing Episode 14 calibration chunks with ICP-refined tool paths.

Writes two new directories and leaves their sources unchanged:
- a sequence directory: the dough clouds, metadata and scene calibration copied from the source sequence,
  with paths_interpolated.pt replaced by the refined marker poses, and a new observation_validation.json
  with updated output hashes;
- a calibration bundle: the table-aligned bundle with tool_geometry pointing to a copy of the manual mount
  used during refinement, with rotations orthonormalized (refined marker pose @ marker_from_mesh is the fitted mesh pose).

Pass both to prepare_episode14_dynamics_temporal.py as --episode and --calibration-bundle.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .config import EXPERIMENT_ROOT, REPOSITORY_ROOT, file_sha256
from .evaluate import fresh_directory, require_owned_output
from .prepare_table_aligned import write_json

UNCHANGED = ("pointclouds_interpolated.pt", "sequence_metadata.json", "scene_calibration_v2.json")


def check_paths(source, refined):
    if source.keys() != refined.keys() or list(source["pose_frames"]) != list(refined["pose_frames"]):
        raise ValueError("Refined paths have different keys or tool order")
    if source["path"].shape != refined["path"].shape or not torch.equal(source["stream_validity"], refined["stream_validity"]):
        raise ValueError("Refined paths have a different shape or validity mask")
    if not torch.equal(source["path"][..., 13], refined["path"][..., 13]):
        raise ValueError("Refined paths have different timestamps")
    if not torch.isfinite(refined["path"]).all():
        raise ValueError("Refined paths contain non-finite values")


def orthonormalized(geometry):
    """Project mount rotations onto the nearest rotation matrix.

    A mount exported from Blender carries float32 rounding (about 2e-7) that the simulator's
    1e-7 orthonormality check rejects.
    """
    geometry = json.loads(json.dumps(geometry))
    change = 0.0
    for tool in geometry["tools"]:
        for key in ("marker_from_collider", "marker_from_mesh"):
            matrix = np.asarray(tool[key], dtype=np.float64)
            u, _, vt = np.linalg.svd(matrix[:3, :3])
            rotation = u @ vt
            if np.linalg.det(rotation) <= 0:
                raise ValueError(f"{tool['name']} {key} is not a proper rotation")
            change = max(change, float(np.abs(rotation - matrix[:3, :3]).max()))
            matrix[:3, :3] = rotation
            tool[key] = matrix.tolist()
    if change > 1e-5:
        raise ValueError(f"Mount rotations are far from orthonormal (entry change {change:.1e})")
    return geometry, change


def run(args):
    source_paths = torch.load(args.episode / "paths_interpolated.pt", map_location="cpu", weights_only=True)
    refined_paths = torch.load(args.refined_paths, map_location="cpu", weights_only=True)
    check_paths(source_paths[0], refined_paths[0])
    geometry = json.loads(args.tool_geometry.read_text())
    refinement_geometry = args.refined_paths.parent / "manual_tool_geometry.json"
    if refinement_geometry.exists() and file_sha256(refinement_geometry) != file_sha256(args.tool_geometry):
        raise ValueError("Tool geometry differs from the one used to refine the paths")
    observations = json.loads((args.episode / "observation_validation.json").read_text())
    for name, digest in observations["output_sha256"].items():
        if file_sha256(args.episode / name) != digest:
            raise ValueError(f"Source observation file changed after validation: {name}")

    sequence = fresh_directory(args.sequence_output)
    for name in UNCHANGED:
        shutil.copyfile(args.episode / name, sequence / name)
    shutil.copyfile(args.refined_paths, sequence / "paths_interpolated.pt")
    shift = (source_paths[0]["path"][..., :3] - refined_paths[0]["path"][..., :3]).norm(dim=-1)
    observations["output_sha256"] = {name: file_sha256(sequence / name) for name in observations["output_sha256"]}
    observations["tool_paths"] = {
        "source": str(args.refined_paths), "source_sha256": file_sha256(args.refined_paths),
        "replaced_sequence": str(args.episode), "replaced_paths_sha256": file_sha256(args.episode / "paths_interpolated.pt"),
        "tool_geometry_sha256": file_sha256(args.tool_geometry),
        "marker_shift_m": {name: {"median": float(shift[:, j].median()), "max": float(shift[:, j].max())}
                           for j, name in enumerate(source_paths[0]["pose_frames"])},
        "note": ("Only paths_interpolated.pt differs from the replaced sequence. Marker poses come from per-frame ICP "
                 "of orange tool points against the manually mounted meshes, fused by an SE3 RTS smoother; velocities "
                 "were recomputed from the refined poses. Dough observations are unchanged.")}
    write_json(sequence / "observation_validation.json", observations)

    bundle_dir = fresh_directory(args.bundle_output)
    bundle = json.loads(args.calibration_bundle.read_text())
    geometry_copy = bundle_dir / "tool_geometry.json"
    geometry, rotation_change = orthonormalized(geometry)
    geometry["orthonormalized_from"] = {"path": str(args.tool_geometry), "sha256": file_sha256(args.tool_geometry),
                                        "max_rotation_entry_change": rotation_change}
    write_json(geometry_copy, geometry)
    bundle["name"] = bundle_dir.name
    bundle["paths"]["tool_geometry"] = str(geometry_copy.relative_to(REPOSITORY_ROOT))
    bundle["expected_sha256"]["tool_geometry"] = file_sha256(geometry_copy)
    bundle_path = bundle_dir / args.calibration_bundle.name
    write_json(bundle_path, bundle)
    calibration = REPOSITORY_ROOT / bundle["paths"]["calibration"]
    if file_sha256(calibration) != bundle["expected_sha256"]["calibration"]:
        raise ValueError("Bundle calibration differs from its pinned hash")
    source_validation = json.loads((args.calibration_bundle.parent / "validation.json").read_text())
    write_json(bundle_dir / "validation.json", {
        **source_validation,
        "source_bundle": str(args.calibration_bundle), "source_bundle_sha256": file_sha256(args.calibration_bundle),
        "note": (f"Copy of {args.calibration_bundle.parent.name} with tool_geometry replaced by the manual mount "
                 f"({geometry['manual_alignment']['status'] if 'manual_alignment' in geometry else 'no alignment record'}) "
                 "used to refine the Episode 14 tool paths. Collision meshes and table calibration are unchanged.")})
    print(sequence, bundle_path, sep="\n", flush=True)
    for name, row in observations["tool_paths"]["marker_shift_m"].items():
        print(f"{name}: marker shift median {row['median'] * 1e3:.1f} mm, max {row['max'] * 1e3:.1f} mm", flush=True)
    return 0


def main(argv=None):
    data = REPOSITORY_ROOT.parent / "data/deformpath_training"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=data / "episode14_dynamics_sequence_v4")
    parser.add_argument("--refined-paths", type=Path, required=True)
    parser.add_argument("--tool-geometry", type=Path, required=True, help="mount used when refining the paths")
    parser.add_argument("--calibration-bundle", type=Path,
                        default=EXPERIMENT_ROOT / "data/episode14_dynamics_table_aligned_v1/episode14_dynamics_calibration.json")
    parser.add_argument("--sequence-output", type=Path, required=True)
    parser.add_argument("--bundle-output", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("episode", "refined_paths", "tool_geometry", "calibration_bundle"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.sequence_output = args.sequence_output.expanduser().resolve()
    args.bundle_output = require_owned_output(args.bundle_output)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
