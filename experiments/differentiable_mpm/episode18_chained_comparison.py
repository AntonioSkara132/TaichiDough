#!/usr/bin/env python3
"""Build the corrected Episode18 chained simulation and article comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.differentiable_mpm.config import load_config
from experiments.differentiable_mpm.data import prepare_experiment
from experiments.differentiable_mpm.dataset_config import load_dataset
from experiments.differentiable_mpm.estimate_table_plane import HSV_DEFAULTS, hsv_mask
_rgb_workflow = importlib.import_module(
    "experiments.differentiable_mpm.episode18_temporal_rgb_topview"
)
decode_image = _rgb_workflow.decode_image
directory_identity = _rgb_workflow.directory_identity
read_rgb_messages = _rgb_workflow.read_rgb_messages
sha256 = _rgb_workflow.sha256
from experiments.differentiable_mpm.reference_adapter import reference_policy
from experiments.differentiable_mpm.results import source_identity
from experiments.differentiable_mpm.runtime import init_runtime
from experiments.differentiable_mpm.solver import Stepper
from experiments.differentiable_mpm.state import ToolControl, validate_parameters

SCHEMA = "taichidough/episode18-chained-comparison/v3"
EPISODE_IDS = tuple(f"episode18_kugla_chunk{i:02d}" for i in range(1, 5))
CHAIN_A = "chain_a"
INDEPENDENT_C2 = "independent_from_chunk2"
REPLAY_C1_END = "from_chain_a_chunk1_end"
STATE_FIELDS = ("x", "v", "C", "F", "Jp")
PARAMETERS = (
    "youngs_modulus", "poisson_ratio", "viscosity", "plastic_min", "plastic_max",
    "floor_retention", "tool_friction_coefficient", "tool_stickiness",
)
FLAGS = {
    "youngs_modulus": "--youngs-modulus", "poisson_ratio": "--poisson-ratio",
    "viscosity": "--viscosity", "plastic_min": "--plastic-min",
    "plastic_max": "--plastic-max", "floor_retention": "--floor-retention",
    "tool_friction_coefficient": "--tool-friction-coefficient",
    "tool_stickiness": "--tool-stickiness",
}
PARTICLE_COLOR = (42, 120, 214)
BACKGROUND_COLOR = (255, 255, 255)
SIMULATOR_ROTATION_DEG = -90
RGB_CROP_MARGIN = 18
RGB_MIN_COMPONENT_AREA = 100
REPLAY_ATOL = 1e-5
REPLAY_RTOL = 1e-5
TOOL_VERTICAL_OFFSET_M = -0.002
EXPECTED_TOTAL_MASS_KG = 0.1365984


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(REPO)) if resolved.is_relative_to(REPO) else str(resolved)


def resolve_repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def parameters_from(args) -> dict[str, float]:
    values = {name: float(getattr(args, name)) for name in PARAMETERS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Parameters must be finite")
    validate_parameters({**values, "tool_retention": 1.0})
    return values


def derived_config(episode, dataset, parameters: dict[str, float], output: Path,
                   backend: str, precision: str) -> Path:
    raw = episode.config.as_dict()
    for field in ("config_path", "source_document_sha256", "path_overrides"):
        raw.pop(field, None)
    raw["name"] = f"{episode.id}-chained-forward"
    raw["parameters"] = episode.parameters_for(parameters)
    raw["fit_parameters"] = list(dataset.fit_parameters)
    raw["parameter_bounds"] = {
        name: [min(float(bounds[0]), parameters[name]),
               max(float(bounds[1]), parameters[name])]
        for name, bounds in dataset.parameter_bounds.items()
    }
    raw["backend"] = backend
    raw["simulation"]["precision"] = precision
    end = int(episode.scored_window.end_frame)
    raw["training"] = {"start_frame": 1, "end_frame": end, "stride": 1}
    raw["validation"] = {"start_frame": end + 1, "end_frame": end + 1, "stride": 1}
    write_new_json(output, raw)
    return output


def segment_end_frame(item, episode):
    end = int(episode.scored_window.end_frame)
    matches = [frame for frame in item.frames if int(frame.source_frame) == end]
    if len(matches) != 1:
        raise ValueError(f"Episode {episode.id} must contain exactly one end frame {end}")
    return matches[0]


def save_state_frame(simulation: Path, episode_id: str, chunk: int, role: str,
                     local_frame: int, simulation_time_s: float, state,
                     run_root: Path) -> dict[str, Any]:
    snapshot = simulation / f"particles_{role}_{local_frame:06d}.npy"
    np.save(snapshot, state.x)
    return {
        "episode_id": episode_id,
        "chunk": chunk,
        "role": role,
        "local_frame": local_frame,
        "simulation_time_s": simulation_time_s,
        "snapshot": relative_path(snapshot, run_root),
        "snapshot_sha256": sha256(snapshot),
    }


def save_full_state(path: Path, state, run_root: Path) -> dict[str, Any]:
    np.savez_compressed(path, **state.arrays())
    return {
        "path": relative_path(path, run_root),
        "sha256": sha256(path),
        "fields": list(STATE_FIELDS),
    }


def numerical_context(config) -> dict[str, Any]:
    particle_count = int(config.n_particles)
    particle_mass = getattr(config, "particle_mass", None)
    return {
        "n_particles": particle_count,
        "grid": getattr(config, "grid", None),
        "particle_mass_kg": particle_mass,
        "total_mass_kg": None if particle_mass is None else float(particle_count * particle_mass),
        "particle_volume_m3": getattr(config, "particle_volume", None),
        "plasticity": getattr(config, "plasticity", None),
        "use_jp": getattr(config, "use_jp", None),
        "jp_hardening": getattr(config, "jp_hardening", None),
        "tool_collision": getattr(config, "tool_collision", None),
        "tool_contact_padding_m": getattr(config, "tool_contact_padding", None),
        "tool_contact_model": getattr(config, "tool_contact_model", None),
        "tool_contact_absorption": getattr(config, "tool_contact_absorption", None),
        "p2g_mode": getattr(config, "p2g_mode", None),
    }


def offset_tool_control(control, vertical_offset_m: float = TOOL_VERTICAL_OFFSET_M):
    if not isinstance(control, ToolControl):
        return control
    poses = control.poses.copy()
    poses[:, 1] += vertical_offset_m
    shifted = ToolControl(poses=poses, velocities=control.velocities.copy(), time=control.time)
    shifted.validate()
    return shifted


def run_chain(name: str, start_index: int, prepared: list, episodes: list,
              output: Path, *, dynamics_index: int | None = None, initial_state=None,
              initial_state_provenance: dict[str, Any] | None = None,
              initial_time_s: float = 0.0) -> tuple[list[dict[str, Any]], dict[int, Any]]:
    if dynamics_index is None:
        dynamics_index = start_index
    if not 0 <= start_index < 4 or not 0 <= dynamics_index < 4:
        raise ValueError("Chain indices must refer to chunks 1 through 4")
    chain_dir = output / "chains" / name
    chain_dir.mkdir(parents=True)
    base = prepared[dynamics_index]
    if base.sdf is None:
        raise RuntimeError("SDF preparation did not produce collision data")
    end_frames = [segment_end_frame(item, episode)
                  for item, episode in zip(prepared[start_index:4], episodes[start_index:4], strict=True)]
    capacity = max(2, min(65, max(int(frame.completed_substeps) for frame in end_frames) + 1))
    stepper = Stepper(base.simulation_config, base.parameters, capacity=capacity, sdf=base.sdf)
    carried = (prepared[start_index].initial_state if initial_state is None else initial_state).copy()
    if initial_state_provenance is None:
        initial_state_provenance = {
            "type": "independent_reconstruction",
            "episode_id": episodes[start_index].id,
            "local_frame": 0,
        }
    records: list[dict[str, Any]] = []
    terminal_states: dict[int, Any] = {}
    cumulative_time = float(initial_time_s)
    for index in range(start_index, 4):
        item = prepared[index]
        episode = episodes[index]
        if item.simulation_config.n_particles != base.simulation_config.n_particles:
            raise ValueError("Chained chunks must have equal particle counts")
        if int(carried.x.shape[0]) != int(item.simulation_config.n_particles):
            raise ValueError("Carried state particle count does not match the next chunk")
        segment = chain_dir / f"chunk{index + 1:02d}"
        simulation = segment / "simulation"
        simulation.mkdir(parents=True)
        write_new_json(simulation / "prepared_inputs.json", item.summary())
        segment_initial = save_full_state(simulation / "initial_state.npz", carried, output)
        stepper.load_state(0, carried)
        frames = [save_state_frame(
            simulation, episode.id, index + 1, "start", 0, cumulative_time,
            carried, output,
        )]
        end_frame = segment_end_frame(item, episode)
        end_step = int(end_frame.completed_substeps)
        slot = 0
        for step in range(end_step):
            stepper.advance(slot, offset_tool_control(item.controls[step]))
            slot += 1
            completed = step + 1
            if slot == stepper.capacity - 1 and completed < end_step:
                stepper.load_state(0, stepper.state(slot))
                slot = 0
        carried = stepper.state(slot)
        carried.validate()
        end_time = cumulative_time + float(end_frame.sim_time_s)
        frames.append(save_state_frame(
            simulation, episode.id, index + 1, "end", int(end_frame.source_frame),
            end_time, carried, output,
        ))
        segment_terminal = save_full_state(simulation / "terminal_state.npz", carried, output)
        terminal_states[index + 1] = carried.copy()
        state_source = initial_state_provenance if index == start_index else {
            "type": "previous_segment_terminal_state",
            "source_chain": name,
            "source_chunk": index,
            "source_role": "end",
            "fields": list(STATE_FIELDS),
        }
        records.append({
            "episode_id": episode.id,
            "chunk": index + 1,
            "frames": frames,
            "state_source": state_source,
            "initial_state": segment_initial,
            "terminal_state": segment_terminal,
        })
        cumulative_time = end_time
    write_new_json(chain_dir / "chain_manifest.json", {
        "schema": SCHEMA + "/chain",
        "name": name,
        "initial_state": initial_state_provenance,
        "initial_time_s": float(initial_time_s),
        "control_start_episode": episodes[start_index].id,
        "dynamics_context_episode": episodes[dynamics_index].id,
        "numerical_context": numerical_context(base.simulation_config),
        "sequence": [episode.id for episode in episodes[start_index:4]],
        "controls_applied_in_order": [episode.id for episode in episodes[start_index:4]],
        "state_fields_carried_between_chunks": list(STATE_FIELDS),
        "stepper_capacity": capacity,
        "tool_pose_offset_m": {"axis": "scene_y", "value": TOOL_VERTICAL_OFFSET_M,
                               "applied_to_both_tools": True},
        "boundary_policy": (
            "The terminal full MPM state is carried. Controls switch directly to the "
            "next chunk without interpolation."
        ),
        "records": records,
    })
    return records, terminal_states


def compare_arrays(left: np.ndarray, right: np.ndarray, *, atol: float = REPLAY_ATOL,
                   rtol: float = REPLAY_RTOL) -> dict[str, Any]:
    compatible = left.shape == right.shape and left.dtype == right.dtype
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    result: dict[str, Any] = {
        "left_shape": list(left.shape), "right_shape": list(right.shape),
        "left_dtype": str(left.dtype), "right_dtype": str(right.dtype),
        "compatible": compatible, "finite": finite,
        "exact_equal": bool(compatible and np.array_equal(left, right)),
        "within_tolerance": False,
        "atol": atol, "rtol": rtol,
        "max_abs_difference": None, "mean_abs_difference": None, "rmse": None,
    }
    if compatible and finite:
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        result.update({
            "within_tolerance": bool(np.allclose(left, right, atol=atol, rtol=rtol)),
            "max_abs_difference": float(difference.max(initial=0.0)),
            "mean_abs_difference": float(difference.mean()),
            "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        })
    return result


def replay_comparison(output: Path, chain_a: list[dict[str, Any]],
                      replay: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_chunk = {int(record["chunk"]): record for record in chain_a}
    right_by_chunk = {int(record["chunk"]): record for record in replay}
    comparisons = []
    for chunk in (2, 3, 4):
        for role, key in (("start", "initial_state"), ("end", "terminal_state")):
            left_meta = left_by_chunk[chunk][key]
            right_meta = right_by_chunk[chunk][key]
            left_path = resolve_path(output, left_meta["path"])
            right_path = resolve_path(output, right_meta["path"])
            with np.load(left_path, allow_pickle=False) as left, np.load(right_path, allow_pickle=False) as right:
                fields = {field: compare_arrays(left[field], right[field]) for field in STATE_FIELDS}
            comparisons.append({
                "chunk": chunk, "role": role,
                "chain_a_state": left_meta, "replay_state": right_meta,
                "fields": fields,
                "all_exact": all(value["exact_equal"] for value in fields.values()),
                "all_within_tolerance": all(value["within_tolerance"] for value in fields.values()),
            })
    result = {
        "schema": SCHEMA + "/replay-comparison",
        "reference_chain": CHAIN_A,
        "replay_chain": REPLAY_C1_END,
        "state_fields": list(STATE_FIELDS),
        "comparisons": comparisons,
        "all_exact": all(row["all_exact"] for row in comparisons),
        "all_within_tolerance": all(row["all_within_tolerance"] for row in comparisons),
        "note": "CUDA atomic P2G can prevent bitwise equality; exact and tolerance results are both retained.",
    }
    path = output / "replay_comparison.json"
    write_new_json(path, result)
    return {**result, "path": relative_path(path, output), "sha256": sha256(path)}


def requested_rgb_frames(episodes, ranges) -> list[dict[str, Any]]:
    requests = []
    for episode, range_row in zip(episodes, ranges[:4], strict=True):
        for role, local in (("start", 0), ("end", int(episode.scored_window.end_frame))):
            ordinal = int(range_row["start_pointcloud_ordinal"]) + local
            if ordinal >= int(range_row["end_pointcloud_ordinal_exclusive"]):
                raise ValueError(f"{role} frame lies outside the source range for {episode.id}")
            requests.append({
                "episode_id": episode.id,
                "chunk": int(episode.id[-2:]),
                "role": role,
                "local_frame": local,
                "original_ordinal": ordinal,
            })
    return requests


def extract_rgb(args, episodes, ranges) -> list[dict[str, Any]]:
    metadata = json.loads(args.conversion_metadata.read_text())
    stamps = metadata["episode"]["timestamps_ns"]
    requests = requested_rgb_frames(episodes, ranges)
    targets = {row["original_ordinal"]: int(stamps[row["original_ordinal"]]) for row in requests}
    candidates, reader_name = read_rgb_messages(args, targets)
    if set(candidates) != set(targets):
        raise ValueError("The bag did not provide all requested RGB frames")
    from PIL import Image
    output = args.output_dir / "rgb"
    output.mkdir()
    records = []
    for request in requests:
        ordinal = request["original_ordinal"]
        delta, stamp, message = candidates[ordinal]
        if args.require_exact_rgb and delta != 0:
            raise ValueError(f"No exact RGB timestamp for ordinal {ordinal}")
        path = output / f"{request['episode_id']}_{request['role']}_rgb.png"
        Image.fromarray(decode_image(message)).save(path)
        records.append({
            **request,
            "pointcloud_timestamp_ns": targets[ordinal],
            "image_timestamp_ns": stamp,
            "absolute_time_difference_ns": delta,
            "path": relative_path(path, args.output_dir),
            "sha256": sha256(path),
            "encoding": str(message.encoding),
        })
    write_new_json(output / "rgb_manifest.json", {
        "schema": SCHEMA + "/rgb", "reader": reader_name,
        "topic": args.rgb_topic, "records": records,
    })
    return records


def largest_component(mask: np.ndarray, minimum_area: int = RGB_MIN_COMPONENT_AREA) \
        -> tuple[np.ndarray, int, list[int]]:
    from scipy.ndimage import label
    labeled = cast(Any, label(mask, structure=np.ones((3, 3), dtype=np.uint8)))
    labels, count = cast(tuple[np.ndarray, int], labeled)
    if count == 0:
        raise ValueError("RGB dough segmentation found no connected component")
    areas = np.bincount(labels.ravel())[1:]
    component = int(np.argmax(areas)) + 1
    area = int(areas[component - 1])
    if area < minimum_area:
        raise ValueError(f"Largest RGB dough component is too small: {area} pixels")
    selected = labels == component
    yy, xx = np.nonzero(selected)
    bounds = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    return selected, area, bounds


def square_crop(bounds: list[int], width: int, height: int, margin: int = RGB_CROP_MARGIN) \
        -> list[int]:
    x0, y0, x1, y1 = bounds
    x0, y0, x1, y1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin
    side = min(max(x1 - x0, y1 - y0), width, height)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    left = int(math.floor(cx - side / 2.0))
    top = int(math.floor(cy - side / 2.0))
    left = min(max(left, 0), width - side)
    top = min(max(top, 0), height - side)
    return [left, top, left + side, top + side]


def crop_rgb_frames(output: Path, records: list[dict[str, Any]], *, margin: int = RGB_CROP_MARGIN,
                    minimum_area: int = RGB_MIN_COMPONENT_AREA) \
        -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from PIL import Image
    arrays = []
    components = []
    dimensions = None
    for record in records:
        source = resolve_path(output, record["path"])
        if sha256(source) != record["sha256"]:
            raise ValueError(f"RGB source hash changed: {source}")
        array = np.asarray(Image.open(source).convert("RGB"))
        current = (array.shape[1], array.shape[0])
        if dimensions is None:
            dimensions = current
        elif current != dimensions:
            raise ValueError("All RGB frames must have matching dimensions")
        _, area, bounds = largest_component(hsv_mask(array, HSV_DEFAULTS), minimum_area)
        arrays.append(array)
        components.append({
            "episode_id": record["episode_id"], "role": record["role"],
            "area_pixels": area, "bounds_xyxy": bounds,
        })
    union = [
        min(row["bounds_xyxy"][0] for row in components),
        min(row["bounds_xyxy"][1] for row in components),
        max(row["bounds_xyxy"][2] for row in components),
        max(row["bounds_xyxy"][3] for row in components),
    ]
    assert dimensions is not None
    crop = square_crop(union, *dimensions, margin=margin)
    cropped_dir = output / "rgb_cropped"
    cropped_dir.mkdir()
    cropped_records = []
    for record, array in zip(records, arrays, strict=True):
        x0, y0, x1, y1 = crop
        path = cropped_dir / f"{record['episode_id']}_{record['role']}_rgb_cropped.png"
        Image.fromarray(array[y0:y1, x0:x1]).save(path)
        cropped_records.append({
            **record, "original_path": record["path"], "original_sha256": record["sha256"],
            "path": relative_path(path, output), "sha256": sha256(path),
            "crop_xyxy": crop,
        })
    manifest = {
        "schema": SCHEMA + "/rgb-crop",
        "hsv_thresholds_opencv": list(HSV_DEFAULTS),
        "connectivity": 8, "minimum_component_area_pixels": minimum_area,
        "source_dimensions_wh": list(dimensions), "components": components,
        "union_bounds_xyxy": union, "margin_pixels": margin,
        "square_crop_xyxy": crop, "records": cropped_records,
    }
    path = cropped_dir / "crop_manifest.json"
    write_new_json(path, manifest)
    return cropped_records, {**manifest, "path": relative_path(path, output), "sha256": sha256(path)}


def particle_bounds(chains: dict[str, list[dict[str, Any]]], root: Path | None = None) \
        -> tuple[np.ndarray, np.ndarray]:
    root = Path(".") if root is None else root
    paths = [resolve_path(root, frame["snapshot"])
             for records in chains.values() for record in records for frame in record["frames"]]
    if not paths:
        raise ValueError("No particle snapshots were provided")
    lower = np.full(2, np.inf)
    upper = np.full(2, -np.inf)
    for path in paths:
        points = np.load(path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"Invalid particle snapshot: {path}")
        lower = np.minimum(lower, points[:, (0, 2)].min(axis=0))
        upper = np.maximum(upper, points[:, (0, 2)].max(axis=0))
    center = (lower + upper) / 2.0
    span = max(float(np.max(upper - lower)) * 1.08, 1e-6)
    return center - span / 2.0, center + span / 2.0


def render_particle_image(points: np.ndarray, bounds: tuple[np.ndarray, np.ndarray],
                          size: int = 900, point_radius: int = 2):
    from PIL import Image, ImageFilter, ImageOps
    lower, upper = bounds
    normalized = (points[:, (0, 2)] - lower) / (upper - lower)
    pixels = np.rint(normalized * (size - 1)).astype(np.int64)
    valid = np.all((pixels >= 0) & (pixels < size), axis=1)
    pixels = pixels[valid]
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[size - 1 - pixels[:, 1], pixels[:, 0]] = 255
    particle_mask = Image.fromarray(mask, mode="L")
    if point_radius > 0:
        particle_mask = particle_mask.filter(ImageFilter.MaxFilter(2 * point_radius + 1))
    image = Image.new("RGB", (size, size), BACKGROUND_COLOR)
    image.paste(PARTICLE_COLOR, mask=particle_mask)
    rotated = image.rotate(SIMULATOR_ROTATION_DEG, resample=Image.Resampling.NEAREST,
                           expand=False, fillcolor=BACKGROUND_COLOR)
    return ImageOps.mirror(rotated)


def render_topviews(output: Path, chains: dict[str, list[dict[str, Any]]]) \
        -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    bounds = particle_bounds(chains, output)
    topview = output / "topview"
    topview.mkdir(parents=True)
    rendered: dict[str, list[dict[str, Any]]] = {}
    for chain_name, records in chains.items():
        rendered[chain_name] = []
        for record in records:
            rendered_frames = []
            for frame in record["frames"]:
                snapshot = resolve_path(output, frame["snapshot"])
                if sha256(snapshot) != frame["snapshot_sha256"]:
                    raise ValueError(f"Particle snapshot hash changed: {snapshot}")
                points = np.load(snapshot, allow_pickle=False)
                image = render_particle_image(points, bounds)
                path = topview / f"{chain_name}_chunk{record['chunk']:02d}_{frame['role']}.png"
                image.save(path)
                if sha256(snapshot) != frame["snapshot_sha256"]:
                    raise ValueError(f"Particle snapshot changed while rendering: {snapshot}")
                rendered_frames.append({
                    "episode_id": record["episode_id"], "chunk": record["chunk"],
                    "role": frame["role"], "local_frame": frame["local_frame"],
                    "path": relative_path(path, output), "sha256": sha256(path),
                    "snapshot": frame["snapshot"],
                    "snapshot_sha256": frame["snapshot_sha256"],
                })
            rendered[chain_name].append({
                "episode_id": record["episode_id"], "chunk": record["chunk"],
                "frames": rendered_frames,
            })
    camera = {
        "projection": "orthographic particle raster",
        "source_axes": {"horizontal": "+x", "vertical": "+z"},
        "view_direction": [0, -1, 0],
        "post_render_rotation_deg": SIMULATOR_ROTATION_DEG,
        "rotation_interpretation": "clockwise in image coordinates",
        "post_render_horizontal_mirror": True,
        "transform_order": "rotate_-90_then_mirror_left_right",
        "common_xz_bounds_m": [bounds[0].tolist(), bounds[1].tolist()],
        "bounds_padding_factor": 1.08,
        "particle_color_rgb": list(PARTICLE_COLOR),
        "background_color_rgb": list(BACKGROUND_COLOR),
        "tool_geometry_rendered": False,
    }
    write_new_json(topview / "topview_manifest.json", {
        "schema": SCHEMA + "/topview", "camera": camera, "records": rendered,
    })
    return rendered, camera


def pair_index(records: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for record in records:
        frames = record.get("frames") or [record]
        episode = str(record["episode_id"])
        result.setdefault(episode, {})
        for frame in frames:
            result[episode][str(frame["role"])] = str(frame["path"])
    for episode, roles in result.items():
        if set(roles) != {"start", "end"}:
            raise ValueError(f"Expected start and end images for {episode}")
    return result


def figure_cells(rgb: list[dict[str, Any]], rendered: dict[str, list[dict[str, Any]]]) \
        -> list[list[dict[str, str] | None]]:
    rgb_index = pair_index(rgb)
    chain_a_index = pair_index(rendered[CHAIN_A])
    independent_index = pair_index(rendered[INDEPENDENT_C2])
    replay_index = pair_index(rendered[REPLAY_C1_END])
    return [
        [rgb_index[episode] for episode in EPISODE_IDS],
        [chain_a_index[episode] for episode in EPISODE_IDS],
        [None, *[independent_index[episode] for episode in EPISODE_IDS[1:]]],
        [None, *[replay_index[episode] for episode in EPISODE_IDS[1:]]],
    ]


def make_figure(output: Path, rgb: list[dict[str, Any]], chains: dict[str, list[dict[str, Any]]],
                rendered: dict[str, list[dict[str, Any]]], camera: dict[str, Any],
                crop: dict[str, Any], replay_metrics: dict[str, Any],
                parameters: dict[str, float]) -> dict[str, Any]:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    cells = figure_cells(rgb, rendered)
    figure_dir = output / "figure"
    figure_dir.mkdir()
    pdf = figure_dir / "episode18_chained_comparison.pdf"
    png = figure_dir / "episode18_chained_comparison.png"
    manifest_path = figure_dir / "episode18_chained_comparison.json"
    fig = plt.figure(figsize=(7.05, 3.55), facecolor="white")
    outer = fig.add_gridspec(
        5, 5, width_ratios=[0.82, 1, 1, 1, 1],
        height_ratios=[0.12, 1, 1, 1, 1], wspace=0.09, hspace=0.10,
        left=0.01, right=0.995, bottom=0.015, top=0.985,
    )
    for column in range(4):
        axis: Any = fig.add_subplot(outer[0, column + 1])
        axis.axis("off")
        axis.text(0.5, 0.5, f"C{column + 1}", ha="center", va="center", weight="bold")
    row_labels = (
        "Recorded\nRGB",
        "Chain A\nC1–C4",
        "Independent C2\nC2–C4",
        "C1-end replay\nC2–C4",
    )
    for row, row_cells in enumerate(cells):
        label_axis: Any = fig.add_subplot(outer[row + 1, 0])
        label_axis.axis("off")
        label_axis.text(0.98, 0.5, row_labels[row], ha="right", va="center", weight="bold")
        for column, paths in enumerate(row_cells):
            if paths is None:
                axis: Any = fig.add_subplot(outer[row + 1, column + 1])
                axis.axis("off")
                continue
            pair = outer[row + 1, column + 1].subgridspec(1, 2, wspace=0.035)
            for panel, role in enumerate(("start", "end")):
                axis: Any = fig.add_subplot(pair[0, panel])
                image = Image.open(resolve_path(output, paths[role])).convert("RGB")
                axis.imshow(image, interpolation="nearest")
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_aspect("equal")
                axis.set_title(role.capitalize(), fontsize=5.5, pad=1.0, color="#47515a")
                for spine in axis.spines.values():
                    spine.set_color("#d9dee2")
                    spine.set_linewidth(0.45)
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    with Image.open(png) as image:
        png_dimensions = list(image.size)
    manifest = {
        "schema": SCHEMA + "/figure",
        "parameters": {**parameters, "tool_retention": 1.0},
        "layout": {
            "figure_size_inches": [7.05, 3.55], "logical_grid": [4, 4],
            "rows": ["recorded RGB C1-C4", "connected Chain A C1-C4",
                     "independent C2 reconstruction, C2-C4",
                     "Chain A C1 terminal-state replay, C2-C4"],
            "columns": ["chunk01", "chunk02", "chunk03", "chunk04"],
            "cell_content": "equal square start and end panels",
            "empty_cells_zero_based": [[2, 0], [3, 0]],
            "png_dimensions_pixels": png_dimensions, "png_dpi": 300,
        },
        "chains": chains, "replay_comparison": replay_metrics,
        "rgb": rgb, "rgb_crop": crop, "topviews": rendered, "camera": camera,
        "outputs": {
            "pdf": {"path": relative_path(pdf, output), "size_bytes": pdf.stat().st_size,
                    "sha256": sha256(pdf)},
            "png": {"path": relative_path(png, output), "size_bytes": png.stat().st_size,
                    "sha256": sha256(png)},
        },
        "limitations": [
            "Recorded RGB perspective and simulator orthographic panels are not pixel-registered.",
            "Chunk boundaries carry the complete particle state and switch directly to the next controls.",
            "The replay intentionally repeats Chain A C2-C4 from Chain A's C1 terminal state.",
            "Simulator panels contain particles only; tool geometry is omitted.",
        ],
    }
    write_new_json(manifest_path, manifest)
    return {**manifest, "path": relative_path(manifest_path, output),
            "sha256": sha256(manifest_path)}


def load_run_records(output: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing simulation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA + "/run":
        raise ValueError("Simulation run uses an incompatible schema")
    if manifest.get("stages", {}).get("simulate", {}).get("status") != "completed":
        raise ValueError("Simulation stage is not complete")
    chains = manifest["chains"]
    for records in chains.values():
        for record in records:
            for key in ("initial_state", "terminal_state"):
                path = resolve_path(output, record[key]["path"])
                if not path.is_file() or sha256(path) != record[key]["sha256"]:
                    raise ValueError(f"Simulation state is missing or changed: {path}")
            for frame in record["frames"]:
                path = resolve_path(output, frame["snapshot"])
                if not path.is_file() or sha256(path) != frame["snapshot_sha256"]:
                    raise ValueError(f"Particle snapshot is missing or changed: {path}")
    return chains, manifest


def validate_inputs(args) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.output_dir.is_relative_to(ROOT):
        raise ValueError(f"Output must be inside {ROOT}")
    if args.stage in {"all", "simulate"}:
        if args.dataset is None or args.range_selection is None:
            raise ValueError("Simulation stage requires --dataset and --range-selection")
        args.dataset = args.dataset.expanduser().resolve()
        args.range_selection = args.range_selection.expanduser().resolve()
        if args.output_dir.exists():
            raise FileExistsError(f"Output already exists: {args.output_dir}")
    if args.stage in {"all", "figure"}:
        if args.bag_dir is None or args.conversion_metadata is None:
            raise ValueError("Figure stage requires --bag-dir and --conversion-metadata")
        args.bag_dir = args.bag_dir.expanduser().resolve()
        args.conversion_metadata = args.conversion_metadata.expanduser().resolve()
        if args.stage == "figure" and not args.output_dir.is_dir():
            raise FileNotFoundError(f"Simulation output does not exist: {args.output_dir}")


def validate_episode18_context(prepared: list) -> None:
    expected_padding = 1.0 / 48.0 / 16.0
    for index, item in enumerate(prepared, 1):
        config = item.simulation_config
        total_mass = float(config.n_particles * config.particle_mass)
        if not math.isclose(total_mass, EXPECTED_TOTAL_MASS_KG, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"Chunk {index:02d} total mass is {total_mass}, expected {EXPECTED_TOTAL_MASS_KG}")
        expected = {
            "grid": 48, "plasticity": "stretch-clamp", "use_jp": False,
            "jp_hardening": 0.0, "tool_collision": "sdf",
            "tool_contact_model": "coulomb-adhesive-v1",
            "tool_contact_absorption": 0.0,
        }
        for name, value in expected.items():
            if getattr(config, name) != value:
                raise ValueError(f"Chunk {index:02d} requires {name}={value!r}")
        if not math.isclose(config.tool_contact_padding, expected_padding,
                            rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"Chunk {index:02d} tool contact padding must equal dx/16")
        if not math.isclose(float(item.parameters["tool_retention"]), 1.0,
                            rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"Chunk {index:02d} tool retention must equal 1.0")


def simulation_stage(args, parameters: dict[str, float]) -> tuple[Any, list, list, dict, dict]:
    source_before = source_identity()
    dataset = load_dataset(args.dataset, backend=args.backend, precision=args.precision)
    episodes = [episode for episode in dataset.episodes if episode.id in EPISODE_IDS]
    if tuple(episode.id for episode in episodes) != EPISODE_IDS:
        raise ValueError(f"Dataset must contain ordered episodes {EPISODE_IDS}")
    ranges = json.loads(args.range_selection.read_text())["ranges"]
    if tuple(row["output_name"] for row in ranges[:4]) != EPISODE_IDS:
        raise ValueError("Range selection does not begin with chunks 1 through 4")
    args.output_dir.mkdir(parents=True)
    configs = args.output_dir / "configs"
    configs.mkdir()
    config_paths = [derived_config(
        episode, dataset, parameters, configs / f"chunk{index + 1:02d}.json",
        args.backend, args.precision,
    ) for index, episode in enumerate(episodes)]
    with reference_policy(args.reference_policy):
        init_runtime(args.backend, args.precision, cpu_threads=args.cpu_threads, seed=0)
        prepared = [prepare_experiment(load_config(path), split="training", build_sdf=True)
                    for path in config_paths]
        validate_episode18_context(prepared)
        chain_a, chain_a_terminal = run_chain(
            CHAIN_A, 0, prepared, episodes, args.output_dir, dynamics_index=0,
        )
        independent, _ = run_chain(
            INDEPENDENT_C2, 1, prepared, episodes, args.output_dir, dynamics_index=1,
        )
        c1_terminal = chain_a[0]["terminal_state"]
        replay_provenance = {
            "type": "chain_terminal_full_state", "source_chain": CHAIN_A,
            "source_episode_id": EPISODE_IDS[0], "source_chunk": 1,
            "source_role": "end", "path": c1_terminal["path"],
            "sha256": c1_terminal["sha256"], "fields": list(STATE_FIELDS),
            "next_controls": list(EPISODE_IDS[1:]),
            "description": "Replay C2-C4 from Chain A's complete C1 terminal state.",
        }
        replay, _ = run_chain(
            REPLAY_C1_END, 1, prepared, episodes, args.output_dir,
            dynamics_index=0, initial_state=chain_a_terminal[1],
            initial_state_provenance=replay_provenance,
            initial_time_s=float(chain_a[0]["frames"][1]["simulation_time_s"]),
        )
    source_after = source_identity()
    source_unchanged = source_before == source_after
    chains = {CHAIN_A: chain_a, INDEPENDENT_C2: independent, REPLAY_C1_END: replay}
    metrics = replay_comparison(args.output_dir, chain_a, replay)
    manifest = {
        "schema": SCHEMA + "/run", "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"path": repository_path(args.dataset), "sha256": sha256(args.dataset),
                    "fingerprint": dataset.fingerprint},
        "range_selection": {"path": repository_path(args.range_selection),
                            "sha256": sha256(args.range_selection)},
        "parameters": {**parameters, "tool_retention": 1.0},
        "tool_pose_offset_m": {"axis": "scene_y", "value": TOOL_VERTICAL_OFFSET_M,
                               "applied_to_both_tools": True},
        "runtime": {"backend": args.backend, "precision": args.precision,
                    "cpu_threads": args.cpu_threads, "reference_policy": args.reference_policy},
        "numerical_context": numerical_context(prepared[0].simulation_config),
        "source_identity": {"before": source_before, "after": source_after,
                            "unchanged": source_unchanged},
        "chains": chains, "replay_comparison": metrics,
        "stages": {"simulate": {"status": "completed" if source_unchanged else "invalid_source_changed"}},
        "calibration_run": False,
    }
    write_new_json(args.output_dir / "run_manifest.json", manifest)
    if not source_unchanged:
        raise RuntimeError("Tracked Python source changed during simulation; numerical output is invalid")
    return dataset, episodes, ranges, chains, manifest


def figure_stage(args, parameters: dict[str, float] | None = None) -> dict[str, Any]:
    chains, run_manifest = load_run_records(args.output_dir)
    dataset_path = resolve_repository_path(run_manifest["dataset"]["path"])
    range_path = resolve_repository_path(run_manifest["range_selection"]["path"])
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset input is unavailable: {dataset_path}")
    if not range_path.is_file():
        raise FileNotFoundError(f"Range selection is unavailable: {range_path}")
    dataset = load_dataset(dataset_path, backend=run_manifest["runtime"]["backend"],
                           precision=run_manifest["runtime"]["precision"])
    episodes = [episode for episode in dataset.episodes if episode.id in EPISODE_IDS]
    ranges = json.loads(range_path.read_text())["ranges"]
    args.dataset = dataset_path
    args.range_selection = range_path
    rgb = extract_rgb(args, episodes, ranges)
    cropped_rgb, crop = crop_rgb_frames(args.output_dir, rgb)
    rendered, camera = render_topviews(args.output_dir, chains)
    figure = make_figure(
        args.output_dir, cropped_rgb, chains, rendered, camera, crop,
        run_manifest["replay_comparison"], run_manifest["parameters"] if parameters is None else parameters,
    )
    run_manifest.setdefault("stages", {})["figure"] = {
        "status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_identity": source_identity(), "bag": directory_identity(args.bag_dir),
        "conversion_metadata": {"path": str(args.conversion_metadata),
                                "sha256": sha256(args.conversion_metadata)},
        "figure_manifest": {"path": figure["path"], "sha256": figure["sha256"]},
    }
    write_json(args.output_dir / "run_manifest.json", run_manifest)
    return figure


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=("all", "simulate", "figure"), default="all")
    result.add_argument("--dataset", type=Path)
    result.add_argument("--range-selection", type=Path)
    result.add_argument("--bag-dir", type=Path)
    result.add_argument("--conversion-metadata", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), default="cuda")
    result.add_argument("--precision", choices=("f32", "f64"), default="f32")
    result.add_argument("--cpu-threads", type=int, default=1)
    result.add_argument("--reference-policy", choices=("strict", "frozen"), default="frozen")
    result.add_argument("--rgb-topic", default="/camera/camera/color/image_raw")
    result.add_argument("--allow-nearest-rgb", dest="require_exact_rgb", action="store_false")
    result.set_defaults(require_exact_rgb=True)
    for name, flag in FLAGS.items():
        result.add_argument(flag, dest=name, type=float)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        validate_inputs(args)
        parameters = None
        if args.stage in {"all", "simulate"}:
            if args.dataset is None or args.range_selection is None:
                raise ValueError("Simulation stage requires --dataset and --range-selection")
            if any(getattr(args, name) is None for name in PARAMETERS):
                raise ValueError("Simulation stage requires all eight physical parameters")
            parameters = parameters_from(args)
            simulation_stage(args, parameters)
        if args.stage in {"all", "figure"}:
            figure_stage(args, parameters)
        if args.stage == "simulate":
            print(args.output_dir / "run_manifest.json")
        else:
            print(args.output_dir / "figure/episode18_chained_comparison.png")
            print(args.output_dir / "figure/episode18_chained_comparison.pdf")
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
