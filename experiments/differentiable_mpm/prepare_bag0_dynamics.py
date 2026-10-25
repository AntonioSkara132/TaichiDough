"""Prepare bag0 episodes as simulator chunks in the Episode 14 layout (no simulation).

Observations are the grid DeformPath clouds in camera_depth_optical_frame. Tool poses are the
raw OptiTrack samples at cloud time + lag, interpolated (linear position, slerp rotation)
between the two bracketing samples after dropping zero positions and jumps faster than the
speed limit, as in the bag0 joint ICP fit. Each episode is split into chunks in which both
tools are tracked; every chunk restarts from its own first cloud with a fixed dough mass.
Every input is named in one sources file, so tool geometry or poses can be replaced by
writing a new sources file and rerunning into a new output directory. The optional sources
keys "version" (file-name suffix, default v1) and "tool_sdf" (builder, resolution and
qualification profile written into every chunk config) leave older sources files unchanged.

Two more optional keys replace the fixed-length chunks. "parts" lists explicit frame ranges
(inclusive, indices into the episode's kept clouds), each with a motion label, a split
(calibration, validation or test) and a start-quality mark; every part becomes one chunk and
the episodes are the ones the parts name. Calibration parts are training rows of the dataset
files, validation parts are validation rows, and test parts get configs but are listed only
in test_episodes_<variant>_<version>.json, so a calibration run never sees them.
"poses.gap_fill" bridges untracked frames by interpolating between the raw samples on either
side of the gap, up to max_gap_s. Bridged frames are flagged; where the bracketing samples are
more than check_clearance_above_s apart, the placed collision mesh must stay at least
min_clearance_m from that frame's dough cloud, otherwise the part fails.
"poses.correction" names a directory with one <episode>.npz per episode, holding per tool
<tool>_stamps_ns [K] (cloud header stamps) and <tool>_marker_correction [K, 4, 4]. Each pose
becomes raw @ C, with C interpolated between the two nearest stamps (linear translation, slerp
rotation) and held constant outside them; a tool with no stamps keeps its raw pose.
"grid_box" with "slack_cells" gives every part its own simulation grid box: the grid nodes that
the stencils of the initial particles and of every observed dough point of the part touch, plus
slack_cells and the WALL_LAYERS outer layers in which the solver removes outward velocity, so the
walls only act slack_cells beyond anything the observed dough reached. The box starts at the
floor and is clipped to the full cube, where the full-cube walls act in the same place.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import site
import subprocess
import sys
from pathlib import Path

import numpy as np

from .config import EXPERIMENT_ROOT, REPOSITORY_ROOT, FrameWindow, file_sha256, load_config
from .data import prepare_experiment, validate_initial_stencils
from .evaluate import fresh_directory, require_owned_output
from .prepare_episode18_dynamics_inputs import MESH_SETTINGS, padded_corners, tool_frame_vertices
from .prepare_registered_tools import topology
from .prepare_table_aligned import bounds, move, tool_support, write_json
from .reference_adapter import get_reference_modules, reference_policy
from .table_alignment import GRAVITY_ASSUMPTION, compose_table_calibration, table_alignment_transform

TOOL_NAMES = ("UR5e_spathla", "gen3_spathla")  # stream order used by Episode 14
REPLAY_MAX_GAP_S = 0.1
TOOL_SDF_KEYS = {"builder": "tool_sdf_builder", "resolution": "tool_sdf_resolution", "qualification": "tool_sdf_qualification"}
PART_SPLITS = {"calibration": "training", "validation": "validation", "test": None}  # split -> dataset membership
GAP_FILL_KEYS = {"max_gap_s", "check_clearance_above_s", "min_clearance_m"}
WALL_LAYERS = 3  # solver.py removes outward grid velocity within this many nodes of the box faces


def raw_tracks(sources):
    """Filtered raw samples per body: no zero positions, no jumps above the speed limit."""
    settings = sources["poses"]
    raw = np.load(settings["file"])
    tracks = {}
    for name, tool in sources["tools"].items():
        keep = raw["names"] == tool["body"]
        if settings["drop_zero_positions"]:
            keep &= np.abs(raw["values"][:, :3]).sum(axis=1) > 0
        stamps, values = raw["stamp_ns"][keep], raw["values"][keep].astype(np.float64)
        speed = np.r_[0, np.linalg.norm(np.diff(values[:, :3], axis=0), axis=1) / (np.diff(stamps) / 1e9)]
        fast = speed >= settings["max_speed_m_s"]
        tracks[name] = (stamps[~fast], values[~fast], {"samples": int(keep.sum()), "dropped_fast": int(fast.sum())})
    return tracks


def bracket_gap_ns(track, stamp_ns):
    """Time between the two raw samples around stamp_ns, or None at either end of the track."""
    i = int(np.searchsorted(track[0], stamp_ns))
    return None if i <= 0 or i >= len(track[0]) else int(track[0][i] - track[0][i - 1])


def pose_at(track, stamp_ns, max_gap_ns):
    """Position and xyzw quaternion at stamp_ns, or None without close bracketing samples."""
    from scipy.spatial.transform import Rotation, Slerp
    stamps, values = track[0], track[1]
    i = int(np.searchsorted(stamps, stamp_ns))
    if i <= 0 or i >= len(stamps) or stamps[i] - stamps[i - 1] > max_gap_ns:
        return None
    a = (stamp_ns - stamps[i - 1]) / (stamps[i] - stamps[i - 1])
    position = (1 - a) * values[i - 1, :3] + a * values[i, :3]
    quaternion = Slerp([0, 1], Rotation.from_quat(values[i - 1:i + 1, 3:7]))(a).as_quat()
    return np.r_[position, quaternion]


def episode_poses(stamps_ns, tracks, settings):
    """[T, 2, 7] poses in the mocap frame, [T, 2] validity, [T, 2] filled flags and [T, 2] bracket gaps
    in s (NaN where not filled), tools in TOOL_NAMES order. Frames are filled only with settings["gap_fill"]."""
    lag_ns, gap_ns = round(settings["lag_s"] * 1e9), round(settings["max_bracket_gap_s"] * 1e9)
    fill = settings.get("gap_fill")
    fill_ns = round(fill["max_gap_s"] * 1e9) if fill else None
    poses = np.full((len(stamps_ns), 2, 7), np.nan)
    valid = np.zeros((len(stamps_ns), 2), dtype=bool)
    filled = np.zeros((len(stamps_ns), 2), dtype=bool)
    fill_gap_s = np.full((len(stamps_ns), 2), np.nan)
    for j, name in enumerate(TOOL_NAMES):
        previous = None
        for i, stamp in enumerate(stamps_ns):
            pose = pose_at(tracks[name], int(stamp) + lag_ns, gap_ns)
            if pose is None and fill_ns is not None:
                pose = pose_at(tracks[name], int(stamp) + lag_ns, fill_ns)
                if pose is not None:
                    filled[i, j], fill_gap_s[i, j] = True, bracket_gap_ns(tracks[name], int(stamp) + lag_ns) / 1e9
            if pose is None:
                continue
            if previous is not None and pose[3:] @ previous < 0:
                pose[3:] = -pose[3:]  # keep consecutive quaternions in one hemisphere
            poses[i, j], valid[i, j], previous = pose, True, pose[3:]
    return poses, valid, filled, fill_gap_s


def corrected_poses(poses, valid, stamps_ns, correction_file):
    """Poses right-multiplied by the marker correction interpolated at each cloud stamp, and the number
    of correction stamps per tool."""
    from scipy.spatial.transform import Rotation, Slerp
    data = np.load(correction_file)
    poses, used = poses.copy(), {}
    for j, name in enumerate(TOOL_NAMES):
        stamps, corrections = data[f"{name}_stamps_ns"].astype(np.float64), data[f"{name}_marker_correction"]
        used[name] = len(stamps)
        rows = np.flatnonzero(valid[:, j])
        if not len(stamps) or not len(rows):
            continue
        t = np.clip(stamps_ns[rows].astype(np.float64), stamps[0], stamps[-1])
        if len(stamps) == 1:
            rotation, translation = Rotation.from_matrix(np.repeat(corrections[:1, :3, :3], len(rows), axis=0)), np.repeat(corrections[:1, :3, 3], len(rows), axis=0)
        else:
            rotation = Slerp(stamps, Rotation.from_matrix(corrections[:, :3, :3]))(t)
            translation = np.stack([np.interp(t, stamps, corrections[:, axis, 3]) for axis in range(3)], axis=1)
        body = Rotation.from_quat(poses[rows, j, 3:7])
        poses[rows, j, :3] += body.apply(translation)
        quaternion = (body * rotation).as_quat()
        quaternion[np.sum(quaternion * poses[rows, j, 3:7], axis=1) < 0] *= -1  # keep the raw hemisphere
        poses[rows, j, 3:7] = quaternion
    return poses, used


def grid_box(points, grid, slack_cells):
    """grid_box_lower and grid_box_cells around the corrected-v1 stencils of points in [0, 1]^3."""
    base = np.floor(np.asarray(points, dtype=np.float64) * grid - 0.5).astype(np.int64)
    margin = slack_cells + WALL_LAYERS
    lower = np.maximum(base.min(axis=0) - margin, 0)
    upper = np.minimum(base.max(axis=0) + 2 + margin + 1, grid)
    lower[1] = 0
    upper = np.maximum(upper, np.minimum(lower + 8, grid))
    lower = np.minimum(lower, upper - 8)
    return [int(v) for v in lower], [int(v) for v in upper - lower]


def chunk_ranges(valid_both, times, chunking):
    """Split runs of tracked frames into near-equal pieces of at most max_frames."""
    runs, start = [], None
    for i, ok in enumerate(valid_both):
        if ok and start is not None and times[i] - times[i - 1] > REPLAY_MAX_GAP_S:
            runs.append((start, i))
            start = None
        if ok and start is None:
            start = i
        if not ok and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(valid_both)))
    kept, dropped = [], []
    for begin, end in runs:
        pieces = math.ceil((end - begin) / chunking["max_frames"])
        edges = np.linspace(begin, end, pieces + 1).round().astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            (kept if b - a >= chunking["min_frames"] else dropped).append((int(a), int(b)))
    return runs, kept, dropped


def part_ranges(parts, valid_both, times):
    """(first, end) per part, end exclusive. Parts are never split, so every frame needs both tool
    poses and no frame gap may exceed REPLAY_MAX_GAP_S."""
    ranges = []
    for part in parts:
        first, last = part["frames"]
        if not 0 <= first < last < len(times):
            raise ValueError(f"part {part['id']}: frames {part['frames']} outside 0..{len(times) - 1}")
        missing = np.flatnonzero(~valid_both[first:last + 1]) + first
        if len(missing):
            raise ValueError(f"part {part['id']}: no pose for a tool at frames {missing.tolist()}")
        gap = float(np.diff(times[first:last + 1]).max())
        if gap > REPLAY_MAX_GAP_S:
            raise ValueError(f"part {part['id']}: frame gap {gap:.3f} s exceeds {REPLAY_MAX_GAP_S} s")
        ranges.append((first, last + 1))
    return ranges


def filled_clearance(chunk, mesh_samples, mocap_from_camera, settings):
    """Smallest distance in m per tool between the placed collision mesh and the dough cloud, over the
    filled frames whose bracket gap exceeds check_clearance_above_s; None when no frame is checked."""
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    result = {}
    for j, name in enumerate(TOOL_NAMES):
        frames = np.flatnonzero(chunk["filled"][:, j] & (chunk["fill_gap_s"][:, j] > settings["check_clearance_above_s"]))
        closest = None
        for f in frames:
            pose = chunk["poses"][f, j]
            world = np.eye(4)
            world[:3, :3], world[:3, 3] = Rotation.from_quat(pose[3:7]).as_matrix(), pose[:3]
            world = world @ mesh_samples[name][0]
            mesh = mesh_samples[name][1] @ world[:3, :3].T + world[:3, 3]
            cloud = move(chunk["clouds"][f][:, :3].numpy().astype(np.float64), mocap_from_camera)
            distance = float(cKDTree(cloud).query(mesh)[0].min())
            closest = distance if closest is None else min(closest, distance)
        result[name] = {"checked_frames": len(frames), "min_clearance_m": closest}
    return result


def path_tensor(poses, times):
    """[T, 2, 14] rows: xyz, xyzw, finite-difference linear and angular velocity, time in seconds."""
    from scipy.spatial.transform import Rotation
    path = np.zeros((len(times), 2, 14))
    path[:, :, :7] = poses
    path[:, :, 13] = times[:, None]
    for j in range(2):
        rotations = Rotation.from_quat(poses[:, j, 3:7])
        dt = np.diff(times)
        linear = np.diff(poses[:, j, :3], axis=0) / dt[:, None]
        angular = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / dt[:, None]
        path[:-1, j, 7:10], path[:-1, j, 10:13] = linear, angular
        path[-1, j, 7:13] = path[-2, j, 7:13]
    return path


def fit_table_plane(path, mocap_from_camera, settings):
    """RANSAC then least-squares plane n.p + d = 0 in the mocap frame, normal along +Y."""
    points = np.load(path)[:, :3].astype(np.float64)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
    points = move(points, mocap_from_camera)
    rng = np.random.default_rng(settings["seed"])
    sample = points[rng.choice(len(points), min(20000, len(points)), replace=False)]
    best, best_point, best_normal = 0, None, None
    for _ in range(settings["ransac_iterations"]):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) < 1e-9:
            continue
        normal /= np.linalg.norm(normal)
        count = int((np.abs((sample - a) @ normal) < settings["inlier_distance_m"]).sum())
        if count > best:
            best, best_point, best_normal = count, a, normal
    inliers = points[np.abs((points - best_point) @ best_normal) < settings["inlier_distance_m"]]
    center = inliers.mean(axis=0)
    normal = np.linalg.svd(inliers - center, full_matrices=False)[2][2]
    normal = normal if normal[1] > 0 else -normal
    plane = np.r_[normal, -normal @ center]
    residual = inliers @ normal + plane[3]
    return plane, {"cloud": str(path), "cloud_sha256": file_sha256(Path(path)), "points": len(points),
                   "inlier_fraction": len(inliers) / len(points), "inlier_rms_m": float(np.sqrt(np.mean(residual ** 2)))}


def tool_inputs(sources, output):
    """Copy meshes, check topology and compose marker_from_mesh = new_body_from_old_body @ base."""
    rows, tools, reports, mesh_paths = [], [], {}, {}
    lag_ns = round(sources["poses"]["lag_s"] * 1e9)
    (output / "tools").mkdir()
    for name in TOOL_NAMES:
        tool = sources["tools"][name]
        source_mesh = Path(tool["mesh"])
        try:
            audit, mesh_bounds = topology(source_mesh)
        except AssertionError as error:
            raise ValueError(f"{name} mesh {source_mesh} is not closed and manifold: {error}") from None
        target = output / "tools" / source_mesh.name
        shutil.copyfile(source_mesh, target)
        rows.append(dict(name=name, collision_mesh=target.name, collision_sha256=file_sha256(target),
                         units="millimetres", coordinate_frame="raw_stl_visual", topology=audit, bounds_mm=mesh_bounds,
                         source_mesh=str(source_mesh), source_sha256=file_sha256(source_mesh)))
        correction_source = Path(tool["body_correction"])
        correction_copy = output / "tools" / correction_source.name
        shutil.copyfile(correction_source, correction_copy)
        correction = json.loads(correction_source.read_text())
        if correction["lag_ns"] != lag_ns:
            raise ValueError(f"{name} body correction was fitted with lag {correction['lag_ns']} ns, not {lag_ns}")
        fit = correction["tools"][tool["body"]]
        base = np.eye(4)
        if tool.get("base_tool_geometry"):
            base_rows = {row["name"]: row for row in json.loads(Path(tool["base_tool_geometry"]).read_text())["tools"]}
            base = np.asarray(base_rows[name]["marker_from_mesh"], dtype=np.float64)
        marker_from_mesh = np.asarray(fit["new_body_from_old_body"], dtype=np.float64) @ base
        mismatch = float(np.abs(marker_from_mesh - np.asarray(fit["new_body_from_mesh"])).max())
        if mismatch > 1e-9:
            raise ValueError(f"{name}: composed marker_from_mesh differs from the fit's new_body_from_mesh by {mismatch}")
        tools.append({"name": name, "marker_from_mesh": marker_from_mesh.tolist()})
        mesh_paths[MESH_SETTINGS[name][0]] = target
        reports[name] = {"body": tool["body"], "mesh": str(source_mesh), "body_correction": str(correction_source),
                         "body_correction_sha256": file_sha256(correction_source),
                         "base_tool_geometry": tool.get("base_tool_geometry"),
                         "fit_quality": {key: fit[key] for key in ("held_out_icp_median_mm", "held_out_x_median_mm",
                                                                    "even_odd_mesh_displacement_mm", "x_error_by_episode_mm")
                                         if key in fit},
                         "selection_note": tool.get("selection_note")}
    return rows, tools, reports, mesh_paths


def run(args):
    sources_path = args.sources
    sources = json.loads(sources_path.read_text())
    if sources.get("schema") != "taichidough/bag0-dynamics-sources/v1":
        raise ValueError("Unsupported sources schema")
    version = sources.get("version", "v1")
    tool_sdf = sources.get("tool_sdf", {})
    if set(tool_sdf) - set(TOOL_SDF_KEYS):
        raise ValueError(f"Unknown tool_sdf settings: {sorted(set(tool_sdf) - set(TOOL_SDF_KEYS))}")
    spec = {row["id"]: row for row in json.loads(Path(sources["segments_spec"]).read_text())["episodes"]}
    parts = sources.get("parts")
    if parts is None:
        order = sources["episode_order"]
        if sorted(order) != sorted(spec):
            raise ValueError("episode_order must list every episode of the segments spec")
    else:
        order = list(dict.fromkeys(part["episode"] for part in parts))
        if set(order) - set(spec):
            raise ValueError(f"parts name episodes missing from the segments spec: {sorted(set(order) - set(spec))}")
        if len({part["id"] for part in parts}) != len(parts):
            raise ValueError("part ids must be unique")
        if any(part["split"] not in PART_SPLITS for part in parts):
            raise ValueError(f"part split must be one of {sorted(PART_SPLITS)}")
    gap_fill = sources["poses"].get("gap_fill")
    if gap_fill is not None and set(gap_fill) != GAP_FILL_KEYS:
        raise ValueError(f"poses.gap_fill must have exactly the keys {sorted(GAP_FILL_KEYS)}")
    correction_dir = Path(sources["poses"]["correction"]["dir"]) if "correction" in sources["poses"] else None
    if correction_dir is not None and (missing := [ep for ep in order if not (correction_dir / f"{ep}.npz").exists()]):
        raise ValueError(f"poses.correction has no file for {missing}")
    box_settings = sources.get("grid_box")
    if box_settings is not None and set(box_settings) != {"slack_cells"}:
        raise ValueError("grid_box must have exactly the key slack_cells")
    observations_root = Path(sources["observations_root"])
    templates = {key: {part: Path(path) for part, path in value.items()} for key, value in sources["config_templates"].items()}
    reconstruction_script = REPOSITORY_ROOT / "scripts/reconstruct_voxel_dough_from_deformpath.py"
    source_paths = [sources_path, Path(sources["segments_spec"]), Path(sources["mocap_to_camera_calibration"]),
                    Path(sources["poses"]["file"]), Path(sources["table_plane"]["fit_cloud"]),
                    *map(Path, sources["table_plane"]["check_clouds"]),
                    *(Path(tool[key]) for tool in sources["tools"].values() for key in ("mesh", "body_correction", "base_tool_geometry") if tool.get(key)),
                    *(path for value in templates.values() for path in value.values()),
                    *(observations_root / ep / name for ep in order for name in ("pointclouds.pt", "conversion_metadata.json", "camera_parameters.json")),
                    *(path for ep in order if (path := observations_root / ep / "filled_flags.pt").exists()),
                    *((correction_dir / f"{ep}.npz" for ep in order) if correction_dir is not None else ()),
                    Path(__file__).resolve(), EXPERIMENT_ROOT / "table_alignment.py", EXPERIMENT_ROOT / "data.py", reconstruction_script]
    source_hashes = {str(path): file_sha256(path) for path in source_paths}
    output = fresh_directory(args.output_dir)
    shutil.copyfile(sources_path, output / "sources.json")
    helpers = get_reference_modules()
    topview, dynamics, simulator = helpers.topview, helpers.dynamics, helpers.simulator
    import torch

    # Tools: closed meshes, collision manifest and tool geometry with box proxies.
    template_config = json.loads(templates["stretch_clamp_coulomb"]["config"].read_text())
    scale = float(template_config["tool_mesh_scale"])
    padding = float(template_config["simulation"]["tool_contact_padding"])
    rows, tools, tool_reports, mesh_paths = tool_inputs(sources, output)
    corners = []
    for tool in tools:
        vertices = tool_frame_vertices(simulator, tool["name"], mesh_paths[MESH_SETTINGS[tool["name"]][0]], scale)
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        center = np.eye(4)
        center[:3, 3] = (lower + upper) / 2
        tool["half_extents_m"] = ((upper - lower) / 2).tolist()
        tool["marker_from_collider"] = (np.asarray(tool["marker_from_mesh"]) @ center).tolist()
        corners.append(padded_corners(vertices, padding))
    write_json(output / "collision_manifest.json", {"schema": "taichidough/tool-collision-meshes/v1", "tools": rows})
    write_json(output / "tool_geometry.json", {"schema": "taichidough/tool-geometry/v1", "tools": tools,
        "note": "marker_from_mesh = bag0 new_body_from_old_body @ static marker_from_mesh (tool-link frame), "
                "where the OptiTrack body pose is mocap_from_new_body. The box proxy is the tool-link-frame AABB "
                "of the closed collision mesh and is only used when tool_collision is not sdf."})
    geometry = dynamics.load_tool_geometry(output / "tool_geometry.json", TOOL_NAMES)

    # Camera, mocap-frame calibration and table plane.
    cameras = [json.loads((observations_root / ep / "camera_parameters.json").read_text()) for ep in order]
    camera_keys = ("frame_id", "width", "height", "intrinsics", "distortion_coefficients", "distortion_model")
    if any(any(c[key] != cameras[0][key] for key in camera_keys) for c in cameras):
        raise ValueError("Episodes have different camera parameters")
    camera = cameras[0]
    intrinsics = np.asarray(camera["intrinsics"])
    mocap_calibration = json.loads(Path(sources["mocap_to_camera_calibration"]).read_text())
    if (mocap_calibration["source_frame"], mocap_calibration["target_frame"]) != ("mocap", camera["frame_id"]):
        raise ValueError("Mocap calibration does not map mocap to the cloud frame")
    camera_from_mocap = np.asarray(mocap_calibration["transform_matrix_4x4"], dtype=np.float64)
    mocap_from_camera = np.linalg.inv(camera_from_mocap)
    plane, plane_report = fit_table_plane(sources["table_plane"]["fit_cloud"], mocap_from_camera, sources["table_plane"])
    plane_checks = []
    for path in sources["table_plane"]["check_clouds"]:
        other, report = fit_table_plane(path, mocap_from_camera, sources["table_plane"])
        report["angle_to_fit_deg"] = float(np.degrees(np.arccos(np.clip(other[:3] @ plane[:3], -1, 1))))
        plane_checks.append(report)
    shutil.copyfile(sources["table_plane"]["fit_cloud"], output / "table_plane_cloud.npy")
    mocap_document = {
        "schema": "taichidough/scene-calibration/v2", "name": "bag0-mocap", "source_frame": "mocap", "scene_frame": "mocap",
        "scene_from_source": np.eye(4).tolist(), "scene_from_camera": mocap_from_camera.tolist(),
        "floor_plane_scene": plane.tolist(),
        "camera": {"fx": float(intrinsics[0, 0]), "fy": float(intrinsics[1, 1]), "cx": float(intrinsics[0, 2]),
                   "cy": float(intrinsics[1, 2]), "width": int(camera["width"]), "height": int(camera["height"]),
                   "d": list(camera["distortion_coefficients"]), "distortion_model": camera["distortion_model"],
                   "frame_id": camera["frame_id"], "zNear": 0.01, "zFar": 5.0, "timestamp_ns": int(camera["timestamp_ns"])},
        "provenance": {"mocap_to_camera_calibration": sources["mocap_to_camera_calibration"],
                       "mocap_to_camera_sha256": source_hashes[sources["mocap_to_camera_calibration"]],
                       "chosen": mocap_calibration.get("chosen"), "method": mocap_calibration.get("method"),
                       "table_plane": plane_report}}
    mocap_path = output / "mocap_scene_calibration_v2.json"
    write_json(mocap_path, mocap_document)
    mocap_calibration_loaded = topview.load_calibration(mocap_path)

    # Collision-mesh surface samples in the mesh frame of marker_from_mesh, for the filled-pose clearance check.
    from .partial_view_icp import sample_mesh
    mesh_samples = {tool["name"]: (np.asarray(tool["marker_from_mesh"]), sample_mesh(
        tool_frame_vertices(simulator, tool["name"], mesh_paths[MESH_SETTINGS[tool["name"]][0]], scale).reshape(-1, 3, 3), 20000, 0)[0])
        for tool in tools}

    # Poses and chunks for every episode.
    tracks = raw_tracks(sources)
    episodes, chunk_plan = {}, []
    (output / "episode_paths").mkdir()
    for ep in order:
        conversion = json.loads((observations_root / ep / "conversion_metadata.json").read_text())
        stamps = np.asarray(conversion["episode"]["kept_timestamps_ns"], dtype=np.int64)
        clouds = torch.load(observations_root / ep / "pointclouds.pt", weights_only=True)[0]
        if len(clouds) != len(stamps) or np.any(np.diff(stamps) <= 0):
            raise ValueError(f"{ep}: cloud count or timestamps are inconsistent")
        times = (stamps - stamps[0]) / 1e9
        poses, valid, filled, fill_gap_s = episode_poses(stamps, tracks, sources["poses"])
        correction_frames = None
        if correction_dir is not None:
            poses, correction_frames = corrected_poses(poses, valid, stamps, correction_dir / f"{ep}.npz")
        if parts is None:
            episode_parts = None
            runs, kept, dropped = chunk_ranges(valid.all(axis=1), times, sources["chunking"])
        else:
            episode_parts = [part for part in parts if part["episode"] == ep]
            runs, kept, dropped = [], part_ranges(episode_parts, valid.all(axis=1), times), []
        filled_flags_path = observations_root / ep / "filled_flags.pt"
        point_filled = torch.load(filled_flags_path, weights_only=True)[0] if filled_flags_path.exists() else None
        torch.save({"path": torch.from_numpy(np.concatenate([poses, np.repeat(times[:, None, None], 2, axis=1)], axis=2)),
                    "stream_validity": torch.from_numpy(valid), "pose_frames": list(TOOL_NAMES),
                    **({"pose_filled": torch.from_numpy(filled)} if gap_fill is not None else {}),
                    "layout": f"xyz, xyzw quaternion (mocap frame, {'corrected' if correction_dir is not None else 'raw'} OptiTrack body), "
                              "time since first cloud in s; NaN where invalid"},
                   output / "episode_paths" / f"{ep}.pt")
        episodes[ep] = {"frames": len(stamps), "valid_frames": {name: int(valid[:, j].sum()) for j, name in enumerate(TOOL_NAMES)},
                        **({"filled_pose_frames": {name: int(filled[:, j].sum()) for j, name in enumerate(TOOL_NAMES)}}
                           if gap_fill is not None else {}),
                        **({"pose_correction_stamps": correction_frames} if correction_frames is not None else {}),
                        "both_valid_frames": int(valid.all(axis=1).sum()), "tracked_runs": runs,
                        "chunks": kept, "dropped_short_pieces": dropped}
        for k, (a, b) in enumerate(kept):
            part = episode_parts[k] if episode_parts is not None else None
            chunk_plan.append({"id": f"bag0_{part['id']}" if part else f"bag0_{ep}_c{k:02d}", "episode": ep, "start": a, "end": b,
                               "stamps": stamps[a:b], "times": times[a:b], "poses": poses[a:b],
                               "filled": filled[a:b], "fill_gap_s": fill_gap_s[a:b], "part": part,
                               "point_filled": point_filled[a:b] if point_filled is not None else None,
                               "clouds": clouds[a:b], "conversion": conversion})
        print(f"{ep}: {len(stamps)} frames, both tools valid {episodes[ep]['both_valid_frames']}, "
              f"{len(kept)} chunks, dropped {dropped}", flush=True)

    # Write chunk recordings in the interpolated sequence format. Metric calibrations map
    # clouds and tool poses with the same scene_from_source, so clouds are stored in mocap.
    cloud_rotation = torch.from_numpy(mocap_from_camera[:3, :3].astype(np.float32))
    cloud_translation = torch.from_numpy(mocap_from_camera[:3, 3].astype(np.float32))
    for chunk in chunk_plan:
        recording = output / "recordings" / chunk["id"]
        recording.mkdir(parents=True)
        cloud_list = []
        for cloud in chunk["clouds"]:
            cloud = cloud.to(torch.float32).clone()
            cloud[:, :3] = cloud[:, :3] @ cloud_rotation.T + cloud_translation
            cloud_list.append(cloud.contiguous())
        torch.save([cloud_list], recording / "pointclouds_interpolated.pt")
        times = chunk["times"] - chunk["times"][0]
        torch.save([{"path": torch.from_numpy(path_tensor(chunk["poses"], times).astype(np.float32)),
                     "stream_validity": torch.ones((len(times), 2), dtype=torch.bool), "pose_frames": list(TOOL_NAMES)}],
                   recording / "paths_interpolated.pt")
        conversion = chunk["conversion"]
        write_json(recording / "sequence_metadata.json", {
            "schema": "taichidough/bag0-chunk-sequence/v1", "episode": chunk["episode"],
            "source_pointclouds": str(observations_root / chunk["episode"] / "pointclouds.pt"),
            "pointcloud_indices": list(range(chunk["start"], chunk["end"])),
            "pointcloud_stamps_ns": [str(value) for value in chunk["stamps"]],
            "pose_frames": list(TOOL_NAMES), "pose_bodies": [sources["tools"][name]["body"] for name in TOOL_NAMES],
            "pose_frame": "mocap", "pose_source": sources["poses"], "time_zero_ns": str(chunk["stamps"][0]),
            "path_layout": "[T, 2, 14]: xyz, xyzw, linear velocity, angular velocity (forward differences), time in s",
            "pointcloud_frame": "mocap", "pointcloud_source_frame": camera["frame_id"],
            "mocap_from_pointcloud_source": mocap_from_camera.tolist(),
            "pointcloud_columns": "xyz in mocap; columns 3-5 copied unchanged from the grid export",
            "pointcloud_filter": {key: conversion[key] for key in ("subsampling", "hsv_filter", "height_band_m",
                                                                   "height_reference", "cluster_filter", "gap_fill")
                                  if key in conversion},
            **({"part": chunk["part"]} if chunk["part"] is not None else {}),
            **({"pose_filled": chunk["filled"].tolist()} if gap_fill is not None else {})})
        chunk["recording"] = recording
        chunk["sequence"] = dynamics.load_observation_sequence(recording)

    # Table-aligned calibration with all chunk clouds and padded tools centred at X=Z=0.5.
    trim = template_config["observation"]["trim_quantile"]
    transform = table_alignment_transform(plane)
    moved = []
    for chunk in chunk_plan:
        sequence = chunk["sequence"]
        replay = dynamics.ToolReplay(sequence, mocap_calibration_loaded, 0, len(sequence.times) - 1,
                                     max_gap_s=REPLAY_MAX_GAP_S, marker_from_tool_frames=geometry.marker_from_mesh)
        boxes = np.concatenate([corners[j] @ dynamics.quaternion_matrix(p[j, 3:7]).T + p[j, :3]
                                for p in replay.poses for j in range(2)])
        clouds = [topview.apply_calibration(topview.filter_xyz(points, trim), mocap_calibration_loaded) for points in sequence.points]
        moved += [move(np.concatenate(clouds), transform), move(boxes, transform)]
    combined = np.concatenate(moved)
    translation_xz = 0.5 - (combined.min(axis=0) + combined.max(axis=0))[[0, 2]] / 2
    transform = table_alignment_transform(plane, translation_xz=translation_xz)
    calibration_path = output / "scene_calibration_v2.json"
    write_json(calibration_path, compose_table_calibration(
        mocap_document, plane, translation_xz=translation_xz,
        provenance={"mocap_calibration": str(mocap_path), "mocap_calibration_sha256": file_sha256(mocap_path),
                    "horizontal_placement": "center all chunk clouds and padded tool mesh bounds at X=Z=0.5"}))
    calibration = topview.load_calibration(calibration_path)
    print(f"Table-aligned calibration written; translation_xz={translation_xz.tolist()}", flush=True)

    # Reconstruction, configs and input validation per chunk.
    from .state import SimulationConfig
    configs_dir = output / "configs"
    configs_dir.mkdir()
    mass_kg = float(sources["mass_kg"])
    obs_size = sources["observation_size"]
    chunk_rows, failed, dataset_rows, test_rows = [], [], {key: [] for key in templates}, {key: [] for key in templates}
    kept = 0
    for chunk in chunk_plan:
        name, recording, sequence = chunk["id"], chunk["recording"], chunk["sequence"]
        frame_count = len(sequence.times)
        init_dir = output / "initialization" / name
        init_dir.mkdir(parents=True)
        frame = np.ascontiguousarray(sequence.points[0], dtype=np.float32)
        torch.save([[torch.from_numpy(frame)]], init_dir / "pointclouds_interpolated.pt")
        write_json(init_dir / "initialization_metadata.json", {
            "chunk": name, "episode": chunk["episode"], "raw_frame": chunk["start"], "points": len(frame),
            "source": "chunk first cloud, unchanged from the grid export"})
        trusted_imports = [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "scripts"), site.getusersitepackages()]
        isolated_runner = (f"import sys,runpy; sys.path[:0]={trusted_imports!r}; "
                           "script=sys.argv.pop(1); sys.argv[0]=script; runpy.run_path(script, run_name='__main__')")
        command = [sys.executable, "-I", "-c", isolated_runner, str(reconstruction_script),
                   "--episode-dir", str(init_dir), "--pointclouds-name", "pointclouds_interpolated.pt", "--frame", "0",
                   "--calibration", str(calibration_path), "--output-dir", str(output / "reconstructions"),
                   "--num-particles", str(template_config["simulation"]["n_particles"]), "--voxel-size", "0.003",
                   "--fill-mode", "floor", "--fill-axis", "y", "--fill-direction", "negative",
                   "--floor-clearance", "0.003", "--floor-min-thickness", "0.001", "--floor-max-thickness", "0.25",
                   "--bbox-padding", "0.006", "--trim-quantile", str(trim), "--footprint-dilate", "1",
                   "--seed", str(template_config["seed"])]
        with (init_dir / "reconstruction.log").open("x") as log:
            result = subprocess.run(command, cwd=REPOSITORY_ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            failed.append({"id": name, "stage": "reconstruction", "reason": f"exit {result.returncode}; see {init_dir / 'reconstruction.log'}"})
            print(f"{name}: reconstruction failed", flush=True)
            continue
        frame_dir = output / "reconstructions" / name / "frame_0000"
        particles_path, reconstruction_path = frame_dir / "sampled_particles_xyz.npy", frame_dir / "reconstruction_metadata.json"
        particles = np.load(particles_path, allow_pickle=False)
        volume = float(json.loads(reconstruction_path.read_text())["object_volume_m3"])
        density = mass_kg / volume
        clouds = [topview.apply_calibration(topview.filter_xyz(points, trim), calibration) for points in sequence.points]
        observed = np.concatenate(clouds)
        box = None
        if box_settings is not None:
            grid = json.loads(next(iter(templates.values()))["config"].read_text())["simulation"]["grid"]
            box = grid_box(np.concatenate([particles, observed]), grid, box_settings["slack_cells"])

        config_paths = {}
        for variant, template in templates.items():
            config = json.loads(template["config"].read_text())
            config["name"] = f"{name}-{variant.replace('_', '-')}"
            config.update({TOOL_SDF_KEYS[key]: value for key, value in tool_sdf.items()})
            config["paths"] = {"episode": str(recording.relative_to(REPOSITORY_ROOT)),
                               "calibration": str(calibration_path.relative_to(REPOSITORY_ROOT)),
                               "initial_particles": str(particles_path.relative_to(REPOSITORY_ROOT)),
                               "reconstruction_metadata": str(reconstruction_path.relative_to(REPOSITORY_ROOT)),
                               "tool_geometry": str((output / "tool_geometry.json").relative_to(REPOSITORY_ROOT)),
                               "collision_manifest": str((output / "collision_manifest.json").relative_to(REPOSITORY_ROOT)),
                               **{key: str(path.relative_to(REPOSITORY_ROOT)) for key, path in mesh_paths.items()}}
            config["expected_sha256"] = {key: file_sha256(REPOSITORY_ROOT / value)
                                         for key, value in config["paths"].items() if key != "episode"}
            config["expected_sequence_fingerprint"] = sequence.fingerprint
            config["mass_kg"] = mass_kg
            if box is not None:
                config["simulation"]["grid_box_lower"], config["simulation"]["grid_box_cells"] = box
            config["density_kg_m3"] = density
            config["observation"]["width"], config["observation"]["height"] = obs_size["width"], obs_size["height"]
            config["training"] = {"start_frame": 1, "end_frame": frame_count - 2, "stride": 1}
            config["validation"] = {"start_frame": frame_count - 1, "end_frame": frame_count - 1, "stride": 1}
            config_paths[variant] = configs_dir / f"{name}_{variant}_{version}.json"
            write_json(config_paths[variant], config)
            load_config(config_paths[variant])

        loaded = load_config(config_paths["stretch_clamp_coulomb"])
        sim = SimulationConfig(**loaded.simulation)
        replay = dynamics.ToolReplay(sequence, calibration, 0, frame_count - 1, max_gap_s=REPLAY_MAX_GAP_S,
                                     marker_from_tool_frames=geometry.marker_from_mesh)
        support = tool_support(replay, corners, dynamics)
        angular = np.linalg.norm(np.asarray(torch.load(recording / "paths_interpolated.pt", weights_only=False)[0]["path"])[:, :, 10:13], axis=2)
        row = {"id": name, "episode": chunk["episode"], "raw_frames": [chunk["start"], chunk["end"]], "frames": frame_count,
               "duration_s": float(sequence.times[-1] - sequence.times[0]), "initialization_points": len(frame),
               "volume_m3": volume, "mass_kg": mass_kg, "density_kg_m3": density,
               "particle_bounds": bounds(particles), "observed_bounds": bounds(observed),
               "observed_below_floor_fraction": float((observed[:, 1] < 0).mean()),
               "tool_continuous_bounds": support,
               "max_tool_angular_speed_rad_s": dict(zip(TOOL_NAMES, angular.max(axis=0).tolist())),
               "sequence_fingerprint": sequence.fingerprint,
               **({"grid_box": {"lower": box[0], "cells": box[1], "node_fraction": float(np.prod(box[1]) / sim.grid ** 3)}}
                  if box is not None else {}),
               "configs": {key: str(path.relative_to(output)) for key, path in config_paths.items()}}
        if chunk["part"] is not None:
            row.update(part=chunk["part"], split=chunk["part"]["split"])
        if gap_fill is not None:
            row["pose_filled_frames"] = dict(zip(TOOL_NAMES, chunk["filled"].sum(axis=0).tolist()))
        if chunk["point_filled"] is not None:
            row["observation_filled_points"] = {"first_frame": int(chunk["point_filled"][0].sum()),
                                                "all_frames": int(sum(int(flags.sum()) for flags in chunk["point_filled"]))}
        try:
            if gap_fill is not None and chunk["filled"].any():
                row["filled_pose_clearance"] = filled_clearance(chunk, mesh_samples, mocap_from_camera, gap_fill)
                for tool_name, check in row["filled_pose_clearance"].items():
                    if check["min_clearance_m"] is not None and check["min_clearance_m"] < gap_fill["min_clearance_m"]:
                        raise ValueError(f"{tool_name} filled pose comes {check['min_clearance_m'] * 1000:.1f} mm from the dough")
            validate_initial_stencils(particles, sim)
            if (particles < 0).any() or (particles >= 1).any():
                raise ValueError("particles leave the simulation domain")
            lower, upper = np.array(support["min"]), np.array(support["max"])
            if np.any(lower[[0, 2]] < 0) or np.any(upper[[0, 2]] >= 1) or upper[1] >= 1:
                raise ValueError(f"tool motion exceeds the simulation domain: {support}")
            prepared = prepare_experiment(loaded, split="training", build_sdf=(kept == 0),
                                          scored_window=FrameWindow(1, frame_count - 1, 1))
        except ValueError as error:
            failed.append({"id": name, "stage": "validation", "reason": str(error)})
            print(f"{name}: validation failed: {error}", flush=True)
            continue
        sim_mass = prepared.simulation_config.particle_mass * len(particles)
        sim_volume = prepared.simulation_config.particle_volume * len(particles)
        np.testing.assert_allclose([sim_mass, sim_volume], [mass_kg, volume], rtol=1e-10, atol=1e-14)
        row.update(prepared_fingerprint=prepared.fingerprint, total_steps=prepared.total_steps,
                   sdf_built=prepared.sdf is not None, particle_mass_kg=prepared.simulation_config.particle_mass,
                   controls_finite=bool(np.isfinite(prepared.controls.poses).all() and np.isfinite(prepared.controls.velocities).all()))
        del prepared
        kept += 1
        if chunk["part"] is None:
            row["membership"] = "validation" if kept % sources["validation_every"] == 0 else "training"
        else:
            row["membership"] = PART_SPLITS[chunk["part"]["split"]] or "test"
        chunk_rows.append(row)
        for variant, path in config_paths.items():
            entry = {"id": name, "config": f"configs/{path.name}", "membership": row["membership"],
                     "weight": 1.0, "scored_window": {"start_frame": 1, "end_frame": frame_count - 1, "stride": 1}}
            (test_rows if row["membership"] == "test" else dataset_rows)[variant].append(entry)
        print(f"{name}: {frame_count} frames, volume {volume * 1e6:.1f} cm^3, density {density:.0f} kg/m^3, "
              f"tool y min {support['min'][1]:.4f}, {row['membership']}", flush=True)

    from .dataset_config import load_dataset
    datasets = []
    priority = set(sources.get("priority_episodes", []))
    by_id = {row["id"]: row for row in chunk_rows}
    for variant, template in templates.items():
        shared = json.loads(template["dataset"].read_text())["shared_parameters"]
        for subset, rows_subset in (("all", dataset_rows[variant]),
                                    ("priority", [r for r in dataset_rows[variant] if by_id[r["id"]]["episode"] in priority])):
            if not rows_subset or not any(r["membership"] == "training" for r in rows_subset):
                continue
            path = output / f"dataset_bag0_{subset}_{variant}_{version}.json"
            write_json(path, {"schema": "taichidough/differentiable-dataset/v2",
                              "name": f"bag0-{subset}-{variant.replace('_', '-')}-{version}",
                              "shared_parameters": shared, "episodes": rows_subset})
            load_dataset(path)
            datasets.append({"dataset": path.name, "variant": variant, "subset": subset, "chunks": len(rows_subset),
                             "validation_chunks": sum(r["membership"] == "validation" for r in rows_subset)})
        if test_rows[variant]:
            path = output / f"test_episodes_{variant}_{version}.json"
            write_json(path, {"schema": "taichidough/bag0-test-episodes/v1", "name": f"bag0-test-{variant.replace('_', '-')}-{version}",
                              "note": "Held-out test parts. They are in no dataset file, so a calibration run never reads them.",
                              "shared_parameters": shared, "episodes": test_rows[variant]})
            datasets.append({"test_episodes": path.name, "variant": variant, "chunks": len(test_rows[variant])})
    if {path: file_sha256(Path(path)) for path in source_hashes} != source_hashes:
        raise ValueError("An input changed during preparation")
    write_json(output / "validation.json", {
        "schema": "taichidough/bag0-dynamics/v1/validation", "status": "inputs_invalid" if failed else "inputs_valid",
        "sources": str(sources_path), "mass_kg": mass_kg, "datasets": datasets, "episodes": episodes,
        "tools": tool_reports, "pose_tracks": {name: track[2] for name, track in tracks.items()},
        "table_plane": {"plane_mocap": plane.tolist(), "fit": plane_report, "checks": plane_checks},
        "transform": transform.tolist(), "gravity_assumption": GRAVITY_ASSUMPTION,
        "chunks": chunk_rows, "failed_chunks": failed, "source_hashes": source_hashes,
        "output_sha256": {str(p.relative_to(output)): file_sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "notes": [*sources.get("notes", []),
                  f"Mass is {mass_kg} kg for every chunk (user instruction); density is mass over the reconstructed volume.",
                  "Each chunk restarts from its own first cloud; frame 0 is initialization only.",
                  "Validation chunks come from the same recording as the training chunks.",
                  "Tool poses are raw OptiTrack samples at cloud time + lag, not the smoothed or exported paths.pt.",
                  "Material and contact starting values come from the Episode 14 v4 Coulomb templates."],
        "simulation_run": False, "calibration_run": False})
    print(f"Wrote {output / 'validation.json'}; {len(chunk_rows)} chunks valid, {len(failed)} failed", flush=True)
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=EXPERIMENT_ROOT / "configs/bag0_dynamics_sources_v1.json")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "data/bag0_dynamics_v1")
    parser.add_argument("--reference-policy", choices=("strict", "frozen"), default="frozen")
    args = parser.parse_args(argv)
    args.sources = args.sources.expanduser().resolve()
    args.output_dir = require_owned_output(args.output_dir)
    with reference_policy(args.reference_policy):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
