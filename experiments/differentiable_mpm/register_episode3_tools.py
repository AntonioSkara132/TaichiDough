#!/usr/bin/env python3
"""Register Episode3 tool meshes after motion-based identities are accepted."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

if __package__:
    from .config import REPOSITORY_ROOT, EXPERIMENT_ROOT, file_sha256
    from .reference_adapter import get_reference_modules, reference_policy
    from .register_tool_clouds import evaluate, fit_icp, sample_mesh, transform, visible_model
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.differentiable_mpm.config import REPOSITORY_ROOT, EXPERIMENT_ROOT, file_sha256
    from experiments.differentiable_mpm.reference_adapter import get_reference_modules, reference_policy
    from experiments.differentiable_mpm.register_tool_clouds import evaluate, fit_icp, sample_mesh, transform, visible_model

TOOL_NAMES = ("UR5e_spathla", "gen3_spathla")
GEOMETRY_PATH = REPOSITORY_ROOT / "configs/tool_geometry_episode18_sdf.json"
MESH_NAMES = ("ur_spathla.stl", "gen3_spathla.stl")
COLLISION_NAMES = ("ur_spathla_collision_solid.stl", "gen3_spathla_collision_solid.stl")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require_identity(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256(path) != expected_sha256:
        raise ValueError("Identity summary SHA-256 differs from the required input")
    value = json.loads(path.read_text())
    if value.get("schema") != "taichidough/episode3-tool-identity/v1":
        raise ValueError("Identity summary schema is not supported")
    if value.get("status") != "accepted" or not value.get("mapping"):
        raise ValueError("Identity is unresolved; registration must not run")
    if set(value["mapping"]) != set(TOOL_NAMES):
        raise ValueError("Identity summary does not map both recorded tools")
    if len(set(value["mapping"].values())) != 2:
        raise ValueError("UR5e and Gen3 must map to distinct tracks")
    return value


def four_seeds(initial: np.ndarray, train: Sequence[Mapping[str, Any]], model: tuple[np.ndarray, np.ndarray], camera: Mapping[str, Any]) -> list[np.ndarray]:
    offsets = []
    for frame in train:
        visible, _ = visible_model(*model, initial, frame["camera_from_marker"], camera)
        offsets.append(np.median(frame["points"], axis=0) - np.median(visible, axis=0))
    shifted = initial.copy()
    shifted[:3, 3] += np.median(np.asarray(offsets), axis=0)
    seeds = [initial.copy(), shifted.copy()]
    for angle in (-20.0, 20.0):
        seed = shifted.copy()
        local_y = Rotation.from_rotvec(np.deg2rad(angle) * np.array([0.0, 1.0, 0.0])).as_matrix()
        seed[:3, :3] = local_y @ initial[:3, :3]
        seeds.append(seed)
    return seeds


def select_training_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(candidates) != 4:
        raise ValueError("Exactly four deterministic registration candidates are required")
    return min(candidates, key=lambda row: (row["train_score_mm"], row["seed_index"]))


def registration_passes(before: Mapping[str, Any], after: Mapping[str, Any], relative_information: Sequence[float]) -> tuple[bool, list[str]]:
    reasons = []
    baseline = float(before["equal_frame_median_distance_mm"])
    fitted = float(after["equal_frame_median_distance_mm"])
    if not fitted <= 0.70 * baseline:
        reasons.append("heldout_median_improvement_below_30_percent")
    if not fitted < 5.0:
        reasons.append("heldout_equal_frame_median_not_below_5_mm")
    if not float(after["equal_frame_within_5mm"]) > 0.75:
        reasons.append("heldout_within_5mm_not_above_75_percent")
    if not float(relative_information[0]) > 1e-4:
        reasons.append("point_to_plane_information_ratio_not_above_1e-4")
    before_frames = {int(row["raw_frame"]): row for row in before["frames"]}
    for row in after["frames"]:
        base = before_frames[int(row["raw_frame"])]
        if float(row["median_distance_mm"]) > max(5.0, 1.25 * float(base["median_distance_mm"])):
            reasons.append(f"heldout_frame_{row['raw_frame']}_regressed")
    return not reasons, reasons


def load_models(seed: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]], list[Path]]:
    mesh_paths = [REPOSITORY_ROOT / "meshes" / name for name in (*MESH_NAMES, *COLLISION_NAMES)]
    with reference_policy("frozen"):
        simulator = get_reference_modules().simulator
    specifications = (
        (simulator.UR_TOOL_VISUAL_ORIGIN, simulator.UR_TOOL_VISUAL_RPY),
        (simulator.KINOVA_TOOL_VISUAL_ORIGIN, simulator.KINOVA_TOOL_VISUAL_RPY),
    )
    visual, collision = [], []
    for index, (origin, rpy) in enumerate(specifications):
        vertices, _ = simulator.load_binary_stl(mesh_paths[index], 0.001)
        triangles = simulator.mesh_in_tool_frame(
            vertices, origin, simulator.rpy_to_matrix(rpy)
        ).reshape(-1, 3, 3).astype(float)
        visual.append(sample_mesh(triangles, 24000, seed + index))
        vertices, _ = simulator.load_binary_stl(mesh_paths[index + 2], 0.001)
        triangles = simulator.mesh_in_tool_frame(
            vertices, origin, simulator.rpy_to_matrix(rpy)
        ).reshape(-1, 3, 3).astype(float)
        collision.append(sample_mesh(triangles, 24000, seed + 100 + index))
    return visual, collision, mesh_paths


def load_track_frames(identity: Mapping[str, Any], tracks_path: Path, points_per_frame: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    record = json.loads(tracks_path.read_text())
    by_id = {track["id"]: track for track in record["tracks"]}
    mocap_from_camera = np.asarray(identity["diagnostics"]["camera_transform"], dtype=float)
    camera_from_mocap = np.linalg.inv(mocap_from_camera)
    result: dict[str, list[dict[str, Any]]] = {}
    for tool_index, tool in enumerate(TOOL_NAMES):
        track_id = identity["mapping"][tool]
        if track_id not in by_id:
            raise ValueError(f"Mapped identity track is absent: {track_id}")
        rows = []
        for observation in by_id[track_id]["observations"]:
            if observation.get("count_change_nearby"):
                continue
            marker = np.asarray(observation["poses"][tool], dtype=float)
            points_mocap = np.asarray(observation["sample_points"], dtype=float)
            if len(points_mocap) < 50:
                continue
            rng = np.random.default_rng(seed + int(observation["ordinal"]) * 31 + tool_index)
            take = rng.choice(len(points_mocap), min(points_per_frame, len(points_mocap)), replace=False)
            points_marker = transform(points_mocap[take], np.linalg.inv(marker))
            rows.append({
                "processed": int(observation["ordinal"]), "raw": int(observation["ordinal"]),
                "points": points_marker, "camera_from_marker": camera_from_mocap @ marker,
                "split": observation["membership"], "window": observation["window"],
            })
        development = [row for row in rows if row["split"] == "development"]
        heldout = [row for row in rows if row["split"] == "heldout"]
        if len(development) < 8 or len(heldout) < 8:
            raise ValueError(f"{tool} has fewer than eight registration frames in development or held-out data")
        result[tool] = rows
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    identity_path = args.identity_summary.resolve()
    identity = require_identity(identity_path, args.expected_identity_sha256)
    tracks_path = args.tracks.resolve() if args.tracks else identity_path.parent / identity["artifacts"]["tracks"]
    output = args.output.resolve()
    if not output.is_relative_to(EXPERIMENT_ROOT) or output == EXPERIMENT_ROOT or output.exists():
        raise ValueError("Choose a new output directory inside experiments/differentiable_mpm")
    sources = [identity_path, tracks_path, GEOMETRY_PATH, Path(__file__).resolve()]
    visual_models, collision_models, mesh_paths = load_models(args.seed)
    sources.extend(mesh_paths)
    hashes = {str(path): file_sha256(path) for path in sources}
    geometry = json.loads(GEOMETRY_PATH.read_text())
    frames = load_track_frames(identity, tracks_path, args.points_per_frame, args.seed)
    camera = json.loads(Path(identity["inputs"]["calibration"]).read_text())["camera"]
    output.mkdir(parents=True, exist_ok=False)
    results = []
    candidate_geometry = copy.deepcopy(geometry)
    for index, tool in enumerate(TOOL_NAMES):
        train = [row for row in frames[tool] if row["split"] == "development"]
        heldout = [row for row in frames[tool] if row["split"] == "heldout"]
        initial = np.asarray(geometry["tools"][index]["marker_from_mesh"], dtype=float)
        before_train = evaluate(train, visual_models[index], initial, camera)
        before_heldout = evaluate(heldout, visual_models[index], initial, camera)
        candidates = []
        for seed_index, initial_seed in enumerate(four_seeds(initial, train, visual_models[index], camera)):
            fitted, history, hessian = fit_icp(train, visual_models[index], camera, initial_seed, iterations=args.iterations)
            training = evaluate(train, visual_models[index], fitted, camera)
            candidates.append({
                "seed_index": seed_index, "train_score_mm": training["equal_frame_median_distance_mm"],
                "transform": fitted, "history": history, "hessian": hessian, "train": training,
            })
        selected = select_training_candidate(candidates)
        fitted = np.asarray(selected["transform"])
        after_heldout = evaluate(heldout, visual_models[index], fitted, camera)
        collision_before = evaluate(heldout, collision_models[index], initial, camera)
        collision_after = evaluate(heldout, collision_models[index], fitted, camera)
        eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(selected["hessian"]))
        relative = eigenvalues / max(float(eigenvalues[-1]), 1e-20)
        accepted, reasons = registration_passes(before_heldout, after_heldout, relative)
        candidate_geometry["tools"][index]["marker_from_mesh"] = fitted.tolist()
        result = {
            "name": tool, "identity_track_id": identity["mapping"][tool],
            "accepted": accepted, "reasons": reasons, "selected_seed_index": selected["seed_index"],
            "initial_marker_from_mesh": initial.tolist(), "candidate_marker_from_mesh": fitted.tolist(),
            "train_before": before_train, "train_after": selected["train"],
            "heldout_before": before_heldout, "heldout_after": after_heldout,
            "collision_evaluation_only": {"heldout_before": collision_before, "heldout_after": collision_after},
            "point_to_plane_information_eigenvalues": eigenvalues.tolist(),
            "relative_information_eigenvalues": relative.tolist(),
            "weakest_information_direction": eigenvectors[:, 0].tolist(),
            "candidates": [{
                "seed_index": row["seed_index"], "train_score_mm": row["train_score_mm"],
                "transform": np.asarray(row["transform"]).tolist(), "history": row["history"],
            } for row in candidates],
        }
        save_new(output / f"{tool}_registration.json", result)
        results.append(result)
    status = "accepted" if all(row["accepted"] for row in results) else "registration_rejected"
    if status == "accepted":
        save_new(output / "tool_geometry_episode3_registered_candidate.json", candidate_geometry)
    if {str(path): file_sha256(path) for path in sources} != hashes:
        raise RuntimeError("A registration input changed during execution")
    summary = {
        "schema": "taichidough/episode3-tool-registration/v1", "status": status,
        "identity_summary": str(identity_path), "identity_summary_sha256": args.expected_identity_sha256,
        "identity_mapping": identity["mapping"], "source_hashes": hashes,
        "settings": {"seed": args.seed, "iterations": args.iterations, "points_per_frame": args.points_per_frame, "seed_selection": "training_only"},
        "results": results, "candidate_geometry_written": status == "accepted",
        "collision_geometry_changes_visual_acceptance": False,
        "operations_not_run": ["calibration", "MPM", "gradient", "material_fit"],
    }
    save_new(output / "registration_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-summary", type=Path, required=True)
    parser.add_argument("--expected-identity-sha256", required=True)
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3003)
    parser.add_argument("--points-per-frame", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=35)
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}")
        return 2
    print(json.dumps({"output": str(args.output), "status": summary["status"]}, indent=2))
    return 0 if summary["status"] == "accepted" else 3


if __name__ == "__main__":
    raise SystemExit(main())
