#!/usr/bin/env python3
"""Simulate Episode 18 temporal chunks and compare midpoint RGB/top views."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.differentiable_mpm.dataset_config import load_dataset
from experiments.differentiable_mpm.forward_video_v2.run import executable
from experiments.differentiable_mpm.state import validate_parameters

PHYSICAL_NAMES = (
    "youngs_modulus",
    "poisson_ratio",
    "viscosity",
    "plastic_min",
    "plastic_max",
    "floor_retention",
    "tool_friction_coefficient",
    "tool_stickiness",
)
SCHEMA = "taichidough/episode18-temporal-rgb-topview/v1"
EXPECTED_IDS = tuple(f"episode18_kugla_chunk{i:02d}" for i in range(1, 14))
PARAMETER_FLAGS = {
    "youngs_modulus": "--youngs-modulus", "poisson_ratio": "--poisson-ratio",
    "viscosity": "--viscosity", "plastic_min": "--plastic-min",
    "plastic_max": "--plastic-max", "floor_retention": "--floor-retention",
    "tool_friction_coefficient": "--tool-friction-coefficient",
    "tool_stickiness": "--tool-stickiness",
}
COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_file(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def directory_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return {"path": str(path), "files": [{"path": str(item.relative_to(path)), "size": item.stat().st_size,
                                           "sha256": sha256(item)} for item in files]}


def validate_episode_ids(dataset) -> None:
    actual = tuple(episode.id for episode in dataset.episodes)
    if actual != EXPECTED_IDS:
        raise ValueError(f"Dataset episodes must be exactly {list(EXPECTED_IDS)} in order")


def validate_explicit_parameters(values: dict[str, Any], dataset) -> dict[str, float]:
    if set(values) != set(PHYSICAL_NAMES) or any(value is None for value in values.values()):
        raise ValueError("All eight explicit physical parameters are required")
    result = {name: float(values[name]) for name in PHYSICAL_NAMES}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Explicit physical parameters must be finite")
    # Forward replay accepts simulator-valid values even when they are outside fit bounds.
    validate_parameters({**result, "tool_retention": 1.0})
    for episode in dataset.episodes:
        episode.parameters_for(result)
    return result


def choose_midpoint(records: list[dict[str, Any]], start_frame: int, end_frame: int) -> dict[str, Any]:
    eligible = [record for record in records if record.get("source_frame") not in (None, 0)]
    eligible = [record for record in eligible if start_frame <= int(record["source_frame"]) <= end_frame]
    if not eligible:
        raise ValueError("No saved scored frame is available")
    target = (start_frame + end_frame) / 2
    return min(eligible, key=lambda record: (abs(int(record["source_frame"]) - target),
                                             int(record["source_frame"])))


def original_ordinal(local_frame: int, range_record: dict[str, Any]) -> int:
    ordinal = int(range_record["start_pointcloud_ordinal"]) + int(local_frame)
    if not int(range_record["start_pointcloud_ordinal"]) <= ordinal < int(range_record["end_pointcloud_ordinal_exclusive"]):
        raise ValueError("Local frame lies outside its source range")
    return ordinal


def decode_image(message) -> np.ndarray:
    encoding = str(message.encoding).lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(encoding)
    if channels is None:
        raise ValueError(f"Unsupported image encoding: {message.encoding}")
    if int(message.step) < int(message.width) * channels:
        raise ValueError("Image row step is smaller than its pixel data")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    expected = int(message.height) * int(message.step)
    if raw.size < expected:
        raise ValueError("Image data is shorter than height times row step")
    image = raw[:expected].reshape(int(message.height), int(message.step))
    image = image[:, :int(message.width) * channels].reshape(int(message.height), int(message.width), channels)
    if encoding.startswith("bgr"):
        image = image[..., [2, 1, 0] + ([3] if channels == 4 else [])]
    return image[..., :3].copy()


def update_nearest(candidates: dict[int, tuple[int, int, Any]], targets: dict[int, int], stamp: int, payload: Any) -> None:
    for ordinal, target in targets.items():
        candidate = (abs(stamp - target), stamp, payload)
        current = candidates.get(ordinal)
        if current is None or candidate[:2] < current[:2]:
            candidates[ordinal] = candidate


def panel_position(index: int, kind: str) -> tuple[int, int]:
    if not 0 <= index < 13 or kind not in {"rgb", "sim"}:
        raise ValueError("Invalid panel request")
    block = 0 if index < 7 else 2
    return block + (kind == "sim"), index if index < 7 else index - 7


def snapshot_hashes(selections: list[dict[str, Any]]) -> dict[str, str]:
    return {row["episode_id"]: sha256(Path(row["snapshot"])) for row in selections}


def forward_commands(args, parameters: dict[str, float]) -> list[list[str]]:
    runner = ROOT / "forward_video_v2" / "run.py"
    commands = []
    for index, episode_id in enumerate(EXPECTED_IDS, 1):
        command = [str(executable(args.simulation_python, "simulation-python")), str(runner),
                   "--dataset", str(args.dataset.resolve()), "--episode-id", episode_id,
                   "--backend", args.backend, "--precision", args.precision,
                   "--cpu-threads", str(args.cpu_threads), "--reference-policy", args.reference_policy,
                   "--simulation-python", str(executable(args.simulation_python, "simulation-python")),
                   "--render-python", str(executable(args.render_python, "render-python")),
                   "--output-dir", str((args.output_dir / "chunks" / f"chunk{index:02d}").resolve())]
        for name in PHYSICAL_NAMES:
            command.extend([PARAMETER_FLAGS[name], repr(parameters[name])])
        commands.append(command)
    return commands


def root_identity(args, dataset, parameters) -> dict[str, Any]:
    paths = (args.dataset, args.range_selection, args.conversion_metadata)
    return {
        "dataset_fingerprint": dataset.fingerprint, "episode_ids": list(EXPECTED_IDS),
        "parameters": parameters, "backend": args.backend, "precision": args.precision,
        "reference_policy": args.reference_policy, "cpu_threads": args.cpu_threads,
        "simulation_python": str(executable(args.simulation_python, "simulation-python")),
        "render_python": str(executable(args.render_python, "render-python")),
        "inputs": {str(path.resolve()): sha256(path.resolve()) for path in paths},
        "bag": directory_identity(args.bag_dir), "rgb_topic": args.rgb_topic,
        "require_exact_rgb": args.require_exact_rgb,
    }


def prepare_root(args, identity: dict[str, Any]) -> dict[str, Any]:
    manifest_path = args.output_dir / "run_manifest.json"
    if args.output_dir.exists():
        if not args.resume or not manifest_path.is_file():
            raise FileExistsError("Output exists; use --resume only for a matching run")
        manifest = json_file(manifest_path)
        if manifest.get("identity") != identity:
            raise ValueError("Resume identity does not match the existing run")
        return manifest
    args.output_dir.mkdir(parents=True)
    manifest = {"schema": SCHEMA + "/run", "created_at": datetime.now(timezone.utc).isoformat(),
                "invocation": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                "identity": identity, "stages": {}, "calibration_run": False}
    write_json(manifest_path, manifest)
    return manifest


def completed_chunk(path: Path) -> bool:
    result_path = path / "simulation" / "simulation_result.json"
    if not result_path.is_file():
        return False
    result = json_file(result_path)
    if result.get("status") != "completed":
        return False
    return all((path / "simulation" / record["particles"]).is_file() for record in result.get("frames", []))


def simulate(args, parameters) -> None:
    (args.output_dir / "chunks").mkdir(exist_ok=True)
    for command in forward_commands(args, parameters):
        chunk = Path(command[command.index("--output-dir") + 1])
        if args.resume and completed_chunk(chunk):
            continue
        if chunk.exists():
            raise FileExistsError(f"Incomplete chunk output exists: {chunk}")
        subprocess.run(command, check=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def selections_for(args, dataset) -> list[dict[str, Any]]:
    ranges = json_file(args.range_selection)["ranges"]
    if [row["output_name"] for row in ranges] != list(EXPECTED_IDS):
        raise ValueError("Range selection does not match the ordered dataset chunks")
    rows = []
    for index, (episode, range_record) in enumerate(zip(dataset.episodes, ranges, strict=True), 1):
        chunk = args.output_dir / "chunks" / f"chunk{index:02d}"
        result = json_file(chunk / "simulation" / "simulation_result.json")
        selected = choose_midpoint(result["frames"], episode.scored_window.start_frame,
                                   episode.scored_window.end_frame)
        local = int(selected["source_frame"])
        snapshot = (chunk / "simulation" / selected["particles"]).resolve()
        rows.append({"episode_id": episode.id, "chunk": index, "local_frame": local,
                     "original_ordinal": original_ordinal(local, range_record),
                     "simulation_time_s": float(selected["sim_time_s"]), "snapshot": str(snapshot),
                     "snapshot_sha256": sha256(snapshot), "tool_poses": selected["tool_poses"],
                     "run": str(chunk.resolve())})
    return rows


def read_rgb_messages(args, targets: dict[int, int]) -> tuple[dict[int, tuple[int, int, Any]], str]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ModuleNotFoundError:
        try:
            import importlib
            AnyReader = importlib.import_module("rosbags.highlevel").AnyReader
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "RGB extraction requires rosbag2_py or the standalone rosbags package; "
                "install rosbags in the active Python environment"
            ) from error

        candidates: dict[int, tuple[int, int, Any]] = {}
        with AnyReader([args.bag_dir.resolve()]) as reader:
            connections = [connection for connection in reader.connections
                           if connection.topic == args.rgb_topic]
            if not connections:
                raise ValueError(f"Topic not found: {args.rgb_topic}")
            for connection, _, serialized in reader.messages(connections=connections):
                message = reader.deserialize(serialized, connection.msgtype)
                stamp = (int(message.header.stamp.sec) * 1_000_000_000
                         + int(message.header.stamp.nanosec))
                update_nearest(candidates, targets, stamp, message)
        return candidates, "rosbags"

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(args.bag_dir.resolve()), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if args.rgb_topic not in types:
        raise ValueError(f"Topic not found: {args.rgb_topic}")
    message_type = get_message(types[args.rgb_topic])
    candidates = {}
    while reader.has_next():
        topic, serialized, _ = reader.read_next()
        if topic != args.rgb_topic:
            continue
        message = deserialize_message(serialized, message_type)
        stamp = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
        update_nearest(candidates, targets, stamp, message)
    return candidates, "rosbag2_py"


def extract_rgb(args, selections) -> list[dict[str, Any]]:
    from PIL import Image

    metadata = json_file(args.conversion_metadata)
    stamps = metadata["episode"]["timestamps_ns"]
    targets = {row["original_ordinal"]: int(stamps[row["original_ordinal"]]) for row in selections}
    candidates, reader_name = read_rgb_messages(args, targets)
    if set(candidates) != set(targets):
        raise ValueError("The bag did not provide every requested RGB frame")
    output = args.output_dir / "rgb"; output.mkdir(exist_ok=True)
    records = []
    for row in selections:
        ordinal = row["original_ordinal"]
        delta, stamp, message = candidates[ordinal]
        if args.require_exact_rgb and delta != 0:
            raise ValueError(f"No exact RGB timestamp for point-cloud ordinal {ordinal}")
        path = output / f"chunk{row['chunk']:02d}_rgb.png"
        Image.fromarray(decode_image(message)).save(path)
        records.append({"episode_id": row["episode_id"], "original_ordinal": ordinal,
                        "pointcloud_timestamp_ns": targets[ordinal], "image_timestamp_ns": stamp,
                        "absolute_time_difference_ns": delta, "encoding": message.encoding,
                        "width": int(message.width), "height": int(message.height),
                        "path": str(path.resolve()), "sha256": sha256(path)})
    write_json(output / "rgb_manifest.json", {"schema": SCHEMA + "/rgb", "topic": args.rgb_topic,
               "reader": reader_name, "bag": directory_identity(args.bag_dir),
               "conversion_metadata": str(args.conversion_metadata.resolve()), "records": records})
    return records


def relocated_repo_path(path: Path) -> Path:
    if path.exists():
        return path
    parts = path.parts
    if "TaichiDough" in parts:
        candidate = REPO.joinpath(*parts[parts.index("TaichiDough") + 1:])
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def stl(path: Path) -> np.ndarray:
    path = relocated_repo_path(path)
    raw = path.read_bytes(); count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"Expected binary STL: {path}")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.frombuffer(raw, dtype=dtype, offset=84, count=count)["vertices"].astype(float)


def render_topviews(args, selections) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyvista as pv
    from scipy.spatial.transform import Rotation
    from PIL import Image
    from experiments.differentiable_mpm.forward_video_v2.render_support import density_boundary

    prepared = [json_file(Path(row["run"]) / "simulation" / "prepared_inputs.json") for row in selections]
    meshes = []
    collision = prepared[0]["provenance"]["collision"]
    for asset in collision["assets_in_stream_order"]:
        vertices = stl(Path(asset["path"])) * collision["mesh_scale"]
        vertices = vertices @ Rotation.from_euler("xyz", asset["visual_rpy_rad"]).as_matrix().T
        meshes.append(vertices + np.asarray(asset["visual_origin_m"]))
    points_all = [np.load(row["snapshot"], allow_pickle=False) for row in selections]
    lower = np.min(np.concatenate(points_all), axis=0); upper = np.max(np.concatenate(points_all), axis=0)
    for row in selections:
        for vertices, pose in zip(meshes, row["tool_poses"], strict=True):
            pose = np.asarray(pose); transformed = vertices @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3]
            lower = np.minimum(lower, transformed.min(axis=(0, 1))); upper = np.maximum(upper, transformed.max(axis=(0, 1)))
    center = (lower + upper) / 2; span = max(upper[0] - lower[0], upper[2] - lower[2]) * 1.08
    output = args.output_dir / "topview"; output.mkdir(exist_ok=True)
    records = []
    for row, points, prep in zip(selections, points_all, prepared, strict=True):
        triangles, info = density_boundary(points, float(prep["mass"]["particle_volume_m3"]))
        plot = pv.Plotter(off_screen=True, window_size=[900, 900])
        plot.set_background("#ffffff")  # pyright: ignore[reportCallIssue,reportArgumentType]
        def add(tris, color):
            flat = tris.reshape(-1, 3); faces = np.column_stack((np.full(len(tris), 3), np.arange(len(flat)).reshape(-1, 3)))
            plot.add_mesh(pv.PolyData(flat, faces.ravel()), color=color, smooth_shading=False, show_edges=False)
        add(triangles, COLORS[0])
        for vertices, pose, color in zip(meshes, row["tool_poses"], COLORS[1:], strict=True):
            pose = np.asarray(pose); add(vertices @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3], color)
        plot.camera_position = [[center[0], upper[1] + max(span, .1), center[2]], center, [0, 0, 1]]
        plot.camera.parallel_projection = True; plot.camera.parallel_scale = span / 2
        plot.enable_anti_aliasing("ssaa")
        image = plot.screenshot(return_img=True); plot.close()
        if image is None: raise RuntimeError("Offscreen renderer returned no image")
        path = output / f"chunk{row['chunk']:02d}_topview.png"; Image.fromarray(image).save(path)
        records.append({"episode_id": row["episode_id"], "path": str(path.resolve()), "sha256": sha256(path),
                        "density": info, "particle_volume_m3": float(prep["mass"]["particle_volume_m3"])})
    camera = {"projection": "orthographic", "direction": [0, -1, 0], "up": [0, 0, 1],
              "parallel_scale": span / 2, "common_bounds_min_m": lower.tolist(),
              "common_bounds_max_m": upper.tolist(), "colors": list(COLORS)}
    return records, camera


def assemble_figure(args, selections, rgb_records, top_records, camera, identity) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from PIL import Image
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure_dir = args.output_dir / "figure"; figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(4, 7, figsize=(7.05, 4.25), facecolor="white")
    for axis in axes.flat: axis.set_visible(False)
    for index, (selected, rgb, top) in enumerate(zip(selections, rgb_records, top_records, strict=True)):
        for kind, record in (("rgb", rgb), ("sim", top)):
            row, column = panel_position(index, kind); axis = axes[row, column]; axis.set_visible(True)
            axis.imshow(Image.open(record["path"]).convert("RGB"), interpolation="nearest")
            axis.set_xticks([]); axis.set_yticks([])
            for spine in axis.spines.values(): spine.set_color("#d9dee2"); spine.set_linewidth(.55)
            if kind == "rgb": axis.set_title(f"C{index + 1:02d} · {selected['simulation_time_s']:.2f} s", fontsize=6.5)
    axes[0, 0].set_ylabel("Recorded RGB", fontweight="semibold")
    axes[1, 0].set_ylabel("Simulated top view", fontweight="semibold")
    axes[2, 0].set_ylabel("Recorded RGB", fontweight="semibold")
    axes[3, 0].set_ylabel("Simulated top view", fontweight="semibold")
    fig.subplots_adjust(left=.08, right=.995, top=.95, bottom=.03, wspace=.06, hspace=.12)
    pdf = figure_dir / "episode18_temporal_rgb_topview.pdf"
    png = figure_dir / "episode18_temporal_rgb_topview.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor="white"); fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest = {"schema": SCHEMA + "/figure", "identity": identity, "parameters": identity["parameters"],
                "selections": selections, "rgb": rgb_records, "topview": top_records, "camera": camera,
                "layout": {"size_inches": [7.05, 4.25], "rows": 4, "columns": 7,
                           "placement": "RGB/simulator pairs for chunks 1-7, then chunks 8-13; final column blank"},
                "limitations": ["Recorded RGB perspective and simulator orthographic panels are not pixel-registered.",
                                "Simulator rendering does not reproduce camera appearance or observed tool occlusion."],
                "outputs": {"pdf": {"path": str(pdf.resolve()), "sha256": sha256(pdf)},
                            "png": {"path": str(png.resolve()), "sha256": sha256(png)}}}
    write_json(figure_dir / "episode18_temporal_rgb_topview.json", manifest)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=("all", "simulate", "extract", "figure"), default="all")
    result.add_argument("--dataset", type=Path, required=True); result.add_argument("--range-selection", type=Path, required=True)
    result.add_argument("--bag-dir", type=Path, required=True); result.add_argument("--conversion-metadata", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True); result.add_argument("--simulation-python", default=sys.executable)
    result.add_argument("--render-python", default=sys.executable); result.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), default="cuda")
    result.add_argument("--precision", choices=("f32", "f64"), default="f32"); result.add_argument("--reference-policy", choices=("strict", "frozen"), default="frozen")
    result.add_argument("--cpu-threads", type=int, default=1); result.add_argument("--rgb-topic", default="/camera/camera/color/image_raw")
    result.add_argument("--allow-nearest-rgb", dest="require_exact_rgb", action="store_false"); result.set_defaults(require_exact_rgb=True)
    result.add_argument("--resume", action="store_true")
    for name, flag in PARAMETER_FLAGS.items(): result.add_argument(flag, dest=name, type=float, required=True)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        args.dataset = args.dataset.expanduser().resolve(); args.range_selection = args.range_selection.expanduser().resolve()
        args.bag_dir = args.bag_dir.expanduser().resolve(); args.conversion_metadata = args.conversion_metadata.expanduser().resolve()
        args.output_dir = args.output_dir.expanduser().resolve()
        if args.cpu_threads < 1: raise ValueError("cpu-threads must be positive")
        if args.output_dir == ROOT or not args.output_dir.is_relative_to(ROOT): raise ValueError(f"output-dir must be inside {ROOT}")
        for path in (args.dataset, args.range_selection, args.conversion_metadata):
            if not path.is_file(): raise FileNotFoundError(path)
        if not args.bag_dir.is_dir(): raise FileNotFoundError(args.bag_dir)
        dataset = load_dataset(args.dataset, backend=args.backend, precision=args.precision); validate_episode_ids(dataset)
        parameters = validate_explicit_parameters({name: getattr(args, name) for name in PHYSICAL_NAMES}, dataset)
        identity = root_identity(args, dataset, parameters); manifest = prepare_root(args, identity)
        if args.stage in {"all", "simulate"}: simulate(args, parameters)
        selections = selections_for(args, dataset) if args.stage in {"all", "extract", "figure"} else []
        if selections: write_json(args.output_dir / "selected_frames.json", selections)
        if args.stage in {"all", "extract"}: rgb_records = extract_rgb(args, selections)
        else: rgb_records = json_file(args.output_dir / "rgb" / "rgb_manifest.json")["records"] if args.stage == "figure" else []
        if args.stage in {"all", "figure"}:
            before = snapshot_hashes(selections); top_records, camera = render_topviews(args, selections)
            if snapshot_hashes(selections) != before: raise RuntimeError("Particle snapshots changed during rendering")
            assemble_figure(args, selections, rgb_records, top_records, camera, identity)
        manifest["stages"][args.stage] = {"completed_at": datetime.now(timezone.utc).isoformat()}
        write_json(args.output_dir / "run_manifest.json", manifest)
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
