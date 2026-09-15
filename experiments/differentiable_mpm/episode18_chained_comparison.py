#!/usr/bin/env python3
"""Compare Episode18 chunk chains with start-to-end camera and simulator views."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.differentiable_mpm.config import load_config
from experiments.differentiable_mpm.data import prepare_experiment
from experiments.differentiable_mpm.dataset_config import load_dataset
_rgb_workflow = importlib.import_module(
    "experiments.differentiable_mpm.episode18_temporal_rgb_topview"
)
decode_image = _rgb_workflow.decode_image
directory_identity = _rgb_workflow.directory_identity
read_rgb_messages = _rgb_workflow.read_rgb_messages
sha256 = _rgb_workflow.sha256
from experiments.differentiable_mpm.reference_adapter import reference_policy
from experiments.differentiable_mpm.runtime import init_runtime
from experiments.differentiable_mpm.solver import Stepper
from experiments.differentiable_mpm.state import validate_parameters

SCHEMA = "taichidough/episode18-chained-comparison/v2"
EPISODE_IDS = tuple(f"episode18_kugla_chunk{i:02d}" for i in range(1, 5))
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


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


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
                     run: Path) -> dict[str, Any]:
    snapshot = simulation / f"particles_{role}_{local_frame:06d}.npy"
    np.save(snapshot, state.x)
    return {
        "episode_id": episode_id,
        "chunk": chunk,
        "role": role,
        "local_frame": local_frame,
        "simulation_time_s": simulation_time_s,
        "snapshot": str(snapshot.resolve()),
        "snapshot_sha256": sha256(snapshot),
        "run": str(run.resolve()),
    }


def save_full_state(path: Path, state) -> dict[str, Any]:
    np.savez_compressed(path, **state.arrays())
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "fields": ["x", "v", "C", "F", "Jp"],
    }


def run_chain(name: str, start_index: int, prepared: list, episodes: list,
              output: Path, initial_state=None,
              initial_state_provenance: dict[str, Any] | None = None) \
        -> tuple[list[dict[str, Any]], dict[int, Any]]:
    chain_dir = output / "chains" / name
    chain_dir.mkdir(parents=True)
    base = prepared[start_index]
    if base.sdf is None:
        raise RuntimeError("SDF preparation did not produce collision data")
    end_frames = [segment_end_frame(item, episode)
                  for item, episode in zip(prepared[start_index:4], episodes[start_index:4], strict=True)]
    capacity = max(2, min(65, max(int(frame.completed_substeps) for frame in end_frames) + 1))
    stepper = Stepper(base.simulation_config, base.parameters, capacity=capacity, sdf=base.sdf)
    carried = (base.initial_state if initial_state is None else initial_state).copy()
    if initial_state_provenance is None:
        initial_state_provenance = {
            "type": "independent_reconstruction",
            "episode_id": episodes[start_index].id,
            "local_frame": 0,
        }
    records: list[dict[str, Any]] = []
    terminal_states: dict[int, Any] = {}
    cumulative_time = 0.0
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
        segment_initial = save_full_state(simulation / "initial_state.npz", carried)
        stepper.load_state(0, carried)
        frames = [save_state_frame(
            simulation, episode.id, index + 1, "start", 0, cumulative_time,
            carried, segment,
        )]
        end_frame = segment_end_frame(item, episode)
        end_step = int(end_frame.completed_substeps)
        slot = 0
        for step in range(end_step):
            stepper.advance(slot, item.controls[step])
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
            end_time, carried, segment,
        ))
        segment_terminal = save_full_state(simulation / "terminal_state.npz", carried)
        terminal_states[index + 1] = carried.copy()
        state_source = (initial_state_provenance if index == start_index else {
            "type": "previous_segment_terminal_state",
            "source_chain": name,
            "source_chunk": index,
            "source_role": "end",
            "fields": ["x", "v", "C", "F", "Jp"],
        })
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
        "sequence": [episode.id for episode in episodes[start_index:4]],
        "controls_applied_in_order": [episode.id for episode in episodes[start_index:4]],
        "state_fields_carried_between_chunks": ["x", "v", "C", "F", "Jp"],
        "boundary_policy": (
            "The terminal full MPM state is carried. At the boundary, controls switch "
            "directly to the next chunk's recorded controls without interpolation."
        ),
        "records": records,
    })
    return records, terminal_states


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
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "encoding": str(message.encoding),
        })
    write_new_json(output / "rgb_manifest.json", {
        "schema": SCHEMA + "/rgb",
        "reader": reader_name,
        "topic": args.rgb_topic,
        "records": records,
    })
    return records


def particle_bounds(chains: dict[str, list[dict[str, Any]]]) -> tuple[np.ndarray, np.ndarray]:
    paths = [Path(frame["snapshot"])
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
    from PIL import Image, ImageFilter

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
    return image.rotate(SIMULATOR_ROTATION_DEG, resample=Image.Resampling.NEAREST,
                        expand=False, fillcolor=BACKGROUND_COLOR)


def render_topviews(output: Path, chains: dict[str, list[dict[str, Any]]]) \
        -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    bounds = particle_bounds(chains)
    topview = output / "topview"
    topview.mkdir(parents=True)
    rendered: dict[str, list[dict[str, Any]]] = {}
    for chain_name, records in chains.items():
        rendered[chain_name] = []
        for record in records:
            rendered_frames = []
            for frame in record["frames"]:
                snapshot = Path(frame["snapshot"])
                points = np.load(snapshot, allow_pickle=False)
                image = render_particle_image(points, bounds)
                path = topview / f"{chain_name}_chunk{record['chunk']:02d}_{frame['role']}.png"
                image.save(path)
                rendered_frames.append({
                    "episode_id": record["episode_id"],
                    "chunk": record["chunk"],
                    "role": frame["role"],
                    "local_frame": frame["local_frame"],
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                    "snapshot": str(snapshot.resolve()),
                    "snapshot_sha256": frame["snapshot_sha256"],
                })
            rendered[chain_name].append({
                "episode_id": record["episode_id"],
                "chunk": record["chunk"],
                "frames": rendered_frames,
            })
    camera = {
        "projection": "orthographic particle raster",
        "source_axes": {"horizontal": "+x", "vertical": "+z"},
        "view_direction": [0, -1, 0],
        "post_render_rotation_deg": SIMULATOR_ROTATION_DEG,
        "rotation_interpretation": "clockwise in image coordinates",
        "common_xz_bounds_m": [bounds[0].tolist(), bounds[1].tolist()],
        "particle_color_rgb": list(PARTICLE_COLOR),
        "background_color_rgb": list(BACKGROUND_COLOR),
        "tool_geometry_rendered": False,
    }
    return rendered, camera


def pair_index(records: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for record in records:
        frames = record.get("frames")
        if frames is None:
            frames = [record]
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
    first_index = pair_index(rendered["from_chunk1"])
    second_index = pair_index(rendered["from_chunk2_replayed"])
    rows: list[list[dict[str, str] | None]] = []
    rows.append([rgb_index[episode] for episode in EPISODE_IDS])
    rows.append([first_index[episode] for episode in EPISODE_IDS])
    rows.append([None, *[rgb_index[episode] for episode in EPISODE_IDS[1:]]])
    rows.append([None, *[second_index[episode] for episode in EPISODE_IDS[1:]]])
    return rows


def load_font(size: int, bold: bool = False):
    from PIL import ImageFont
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def fit_image(path: str, size: tuple[int, int]):
    from PIL import Image, ImageOps
    image = Image.open(path).convert("RGB")
    return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)


def draw_pair(cell, paths: dict[str, str]) -> None:
    from PIL import ImageDraw
    draw = ImageDraw.Draw(cell)
    label_font = load_font(22, bold=True)
    width, height = cell.size
    pad = 12
    label_height = 32
    half = (width - 3 * pad) // 2
    image_size = (half, height - label_height - 2 * pad)
    for index, role in enumerate(("start", "end")):
        image = fit_image(paths[role], image_size)
        x0 = pad + index * (half + pad)
        y0 = label_height + pad + (image_size[1] - image.height) // 2
        cell.paste(image, (x0 + (half - image.width) // 2, y0))
        text = role.upper()
        box = draw.textbbox((0, 0), text, font=label_font)
        draw.text((x0 + (half - (box[2] - box[0])) / 2, 4), text,
                  font=label_font, fill=(51, 59, 66))
    arrow = "→"
    box = draw.textbbox((0, 0), arrow, font=label_font)
    draw.text(((width - (box[2] - box[0])) / 2, 4), arrow,
              font=label_font, fill=(87, 96, 106))
    draw.rectangle((0, 0, width - 1, height - 1), outline=(217, 222, 226), width=2)


def make_figure(output: Path, rgb: list[dict[str, Any]], first: list[dict[str, Any]],
                second: list[dict[str, Any]], rendered: dict[str, list[dict[str, Any]]],
                camera: dict[str, Any], parameters: dict[str, float]) -> None:
    from PIL import Image, ImageDraw

    cells = figure_cells(rgb, rendered)
    cell_size = (720, 300)
    left = 285
    top = 76
    gap_x = 14
    gap_y = 14
    width = left + 4 * cell_size[0] + 3 * gap_x + 24
    height = top + 4 * cell_size[1] + 3 * gap_y + 24
    canvas = Image.new("RGB", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(27, bold=True)
    row_font = load_font(25, bold=True)
    for column in range(4):
        text = f"C{column + 1:02d}"
        box = draw.textbbox((0, 0), text, font=title_font)
        x = left + column * (cell_size[0] + gap_x) + (cell_size[0] - (box[2] - box[0])) / 2
        draw.text((x, 22), text, font=title_font, fill=(33, 39, 44))
    row_labels = (
        "Recorded RGB\nC1 → C4",
        "Simulation\ninitialized at C1",
        "Recorded RGB\nC2 → C4",
        "Simulation\nA:C2 end, then replay\nC2 → C4",
    )
    for row, (label, row_cells) in enumerate(zip(row_labels, cells, strict=True)):
        y = top + row * (cell_size[1] + gap_y)
        line_box = draw.multiline_textbbox((0, 0), label, font=row_font, spacing=8, align="right")
        label_height = line_box[3] - line_box[1]
        draw.multiline_text((left - 22, y + (cell_size[1] - label_height) / 2), label,
                            font=row_font, fill=(33, 39, 44), spacing=8,
                            align="right", anchor="ra")
        for column, paths in enumerate(row_cells):
            if paths is None:
                continue
            cell = Image.new("RGB", cell_size, BACKGROUND_COLOR)
            draw_pair(cell, paths)
            x = left + column * (cell_size[0] + gap_x)
            canvas.paste(cell, (x, y))
    figure = output / "figure"
    figure.mkdir()
    png = figure / "episode18_chained_comparison.png"
    pdf = figure / "episode18_chained_comparison.pdf"
    canvas.save(png, dpi=(300, 300))
    canvas.save(pdf, "PDF", resolution=300.0)
    write_new_json(figure / "episode18_chained_comparison.json", {
        "schema": SCHEMA + "/figure",
        "parameters": parameters,
        "layout": {
            "rows": [
                "camera start-to-end for C1-C4",
                "continuous simulation start-to-end initialized at C1",
                "camera start-to-end for C2-C4 with first cell empty",
                "simulation initialized from chain A chunk2's complete end state, then chunk2-chunk4 controls replayed, with first cell empty",
            ],
            "columns": ["chunk01", "chunk02", "chunk03", "chunk04"],
            "cell_content": "start image followed by end image",
            "empty_cells_zero_based": [[2, 0], [3, 0]],
        },
        "chains": {"from_chunk1": first, "from_chunk2_replayed": second},
        "rgb": rgb,
        "topviews": rendered,
        "camera": camera,
        "outputs": {
            "pdf": {"path": str(pdf.resolve()), "sha256": sha256(pdf)},
            "png": {"path": str(png.resolve()), "sha256": sha256(png)},
        },
        "limitations": [
            "Recorded RGB perspective and simulator orthographic panels are not pixel-registered.",
            "At each chunk boundary the full particle state is retained while controls switch directly to the next chunk trajectory.",
            "No interpolation is applied between one chunk's final control timestamp and the next chunk's frame-0 control timestamp.",
            "The second chain starts from chain A chunk02's complete end state and then applies chunk02, chunk03, and chunk04 controls again.",
            "Simulator panels contain particles only; collision tool geometry is intentionally omitted.",
        ],
    })


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--range-selection", type=Path, required=True)
    result.add_argument("--bag-dir", type=Path, required=True)
    result.add_argument("--conversion-metadata", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), default="cuda")
    result.add_argument("--precision", choices=("f32", "f64"), default="f32")
    result.add_argument("--cpu-threads", type=int, default=1)
    result.add_argument("--reference-policy", choices=("strict", "frozen"), default="frozen")
    result.add_argument("--rgb-topic", default="/camera/camera/color/image_raw")
    result.add_argument("--allow-nearest-rgb", dest="require_exact_rgb", action="store_false")
    result.set_defaults(require_exact_rgb=True)
    for name, flag in FLAGS.items():
        result.add_argument(flag, dest=name, type=float, required=True)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        for name in ("dataset", "range_selection", "bag_dir", "conversion_metadata", "output_dir"):
            setattr(args, name, getattr(args, name).expanduser().resolve())
        if args.output_dir.exists():
            raise FileExistsError(f"Output already exists: {args.output_dir}")
        if not args.output_dir.is_relative_to(ROOT):
            raise ValueError(f"Output must be inside {ROOT}")
        parameters = parameters_from(args)
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
            chain1, chain1_terminal_states = run_chain(
                "from_chunk1", 0, prepared, episodes, args.output_dir,
            )
            chain1_chunk2_terminal = chain1[1]["terminal_state"]
            chain2, _ = run_chain(
                "from_chunk2_replayed", 1, prepared, episodes, args.output_dir,
                initial_state=chain1_terminal_states[2],
                initial_state_provenance={
                    "type": "chain_terminal_full_state",
                    "source_chain": "from_chunk1",
                    "source_episode_id": EPISODE_IDS[1],
                    "source_chunk": 2,
                    "source_role": "end",
                    "path": chain1_chunk2_terminal["path"],
                    "sha256": chain1_chunk2_terminal["sha256"],
                    "fields": ["x", "v", "C", "F", "Jp"],
                    "next_controls": [EPISODE_IDS[1], EPISODE_IDS[2], EPISODE_IDS[3]],
                    "description": (
                        "Initialize from the complete end state of chunk 2 in the C1-C4 "
                        "chain, then apply chunk 2, chunk 3, and chunk 4 controls again."
                    ),
                },
            )
        rgb = extract_rgb(args, episodes, ranges)
        chains = {"from_chunk1": chain1, "from_chunk2_replayed": chain2}
        rendered, camera = render_topviews(args.output_dir, chains)
        make_figure(args.output_dir, rgb, chain1, chain2, rendered, camera, parameters)
        write_new_json(args.output_dir / "run_manifest.json", {
            "schema": SCHEMA + "/run",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(args.dataset),
            "dataset_fingerprint": dataset.fingerprint,
            "range_selection": str(args.range_selection),
            "bag": directory_identity(args.bag_dir),
            "parameters": parameters,
            "runtime": {"backend": args.backend, "precision": args.precision},
            "chains": {
                "from_chunk1": {
                    "initial_state": "chunk01 independent reconstruction",
                    "controls": list(EPISODE_IDS),
                },
                "from_chunk2_replayed": {
                    "initial_state": "from_chunk1 chunk02 complete terminal x,v,C,F,Jp state",
                    "controls": list(EPISODE_IDS[1:]),
                },
            },
            "camera": camera,
            "calibration_run": False,
        })
        print(args.output_dir / "figure/episode18_chained_comparison.png")
        print(args.output_dir / "figure/episode18_chained_comparison.pdf")
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
