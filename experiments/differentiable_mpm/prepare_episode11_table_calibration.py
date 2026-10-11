"""Write a table-aligned Episode 11 calibration bundle (same composition as prepare_episode18_dynamics_inputs).

The verified table plane puts the table at y=0; the complete observed motion and padded tool boxes are
centred at X=Z=0.5. Tool geometry and collision meshes are the Episode 18 ones named by the base bundle.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from .config import REPOSITORY_ROOT, file_sha256
from .evaluate import fresh_directory
from .prepare_episode18_dynamics_inputs import padded_corners, tool_frame_vertices
from .prepare_table_aligned import move, write_json
from .reference_adapter import get_reference_modules, reference_policy
from .table_alignment import compose_table_calibration, table_alignment_transform

TOOL_MESHES = {"UR5e_spathla": "ur_collision_mesh", "gen3_spathla": "kinova_collision_mesh"}


def run(args):
    summary = json.loads(args.plane_summary.read_text())
    verification = json.loads(args.plane_summary.with_name("verification.json").read_text())
    if verification.get("passed") is not True:
        raise ValueError("Table-plane estimate has not passed verification")
    if verification.get("output_sha256", {}).get("summary.json") != file_sha256(args.plane_summary):
        raise ValueError("Plane summary differs from its verified output hash")
    if Path(summary["episode"]).resolve() != args.episode:
        raise ValueError("Plane estimate describes a different sequence")
    calibration_source = args.episode / "scene_calibration_v2.json"
    base = json.loads(args.base_bundle.read_text())
    helpers = get_reference_modules()
    topview, dynamics, simulator = helpers.topview, helpers.dynamics, helpers.simulator
    sequence = dynamics.load_observation_sequence(args.episode)
    geometry = dynamics.load_tool_geometry(REPOSITORY_ROOT / base["paths"]["tool_geometry"], sequence.names)
    padding = float(base["simulation"]["tool_contact_padding"])
    corners = [padded_corners(tool_frame_vertices(simulator, name, REPOSITORY_ROOT / base["paths"][TOOL_MESHES[name]],
                                                  float(base["tool_mesh_scale"])), padding) for name in sequence.names]

    old_document = json.loads(calibration_source.read_text())
    old_calibration = topview.load_calibration(calibration_source)
    replay = dynamics.ToolReplay(sequence, old_calibration, 0, len(sequence.times) - 1,
                                 max_gap_s=base["replay_max_gap_s"], marker_from_tool_frames=geometry.marker_from_mesh)
    trim = base["observation"]["trim_quantile"]
    clouds = [topview.apply_calibration(topview.filter_xyz(points, trim), old_calibration) for points in sequence.points]
    plane = np.asarray(summary["estimated_plane_scene"], dtype=float)
    transform = table_alignment_transform(plane)
    boxes = np.concatenate([corners[j] @ dynamics.quaternion_matrix(p[j, 3:7]).T + p[j, :3]
                            for p in replay.poses for j in range(len(corners))])
    combined = np.concatenate([*(move(points, transform) for points in clouds), move(boxes, transform)])
    translation_xz = 0.5 - (combined.min(axis=0) + combined.max(axis=0))[[0, 2]] / 2
    aligned = move(combined, table_alignment_transform(plane, translation_xz=translation_xz))

    output = fresh_directory(args.output_dir)
    calibration_path = output / "scene_calibration_v2.json"
    write_json(calibration_path, compose_table_calibration(
        old_document, plane, translation_xz=translation_xz,
        provenance={"plane_summary": str(args.plane_summary), "plane_summary_sha256": file_sha256(args.plane_summary),
                    "source_calibration": str(calibration_source), "source_calibration_sha256": file_sha256(calibration_source),
                    "horizontal_placement": "center complete observed motion and padded tool mesh bounds at X=Z=0.5"}))
    base["name"] = args.output_dir.name
    base["paths"]["episode"] = os.path.relpath(args.episode, REPOSITORY_ROOT)
    base["paths"]["calibration"] = str(calibration_path.relative_to(REPOSITORY_ROOT))
    base["expected_sha256"]["calibration"] = file_sha256(calibration_path)
    write_json(output / args.base_bundle.name, base)
    shutil.copyfile(args.plane_summary, output / "table_plane_summary.json")
    write_json(output / "alignment_report.json", {
        "translation_xz_m": translation_xz.tolist(), "plane": plane.tolist(),
        "aligned_motion_and_tool_bounds": {"min": aligned.min(axis=0).tolist(), "max": aligned.max(axis=0).tolist()}})
    write_json(output / "validation.json", {
        "schema": "taichidough/episode11-calibration/v1/validation", "status": "calibration_prepared",
        "method": "rigid-table-plane-alignment", "plane_verification_passed": True,
        "plane_summary_sha256": file_sha256(args.plane_summary), "calibration_sha256": file_sha256(calibration_path),
        "note": "Tool geometry and collision meshes are the Episode 18 ones; particles are set per chunk."})
    print(f"translation_xz={translation_xz.tolist()} bounds min={aligned.min(axis=0)} max={aligned.max(axis=0)}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--plane-summary", type=Path, required=True)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-policy", choices=("strict", "frozen"), default="strict")
    args = parser.parse_args(argv)
    for name in ("episode", "plane_summary", "base_bundle", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    with reference_policy(args.reference_policy):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
