#!/usr/bin/env python3
"""Render a dataset-aware forward replay without running physics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import cast

import numpy as np
from scipy.spatial.transform import Rotation
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_support import camera_zoom, density_boundary, encode_video, video_tools

COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def stl(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"Expected binary STL: {path}")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.frombuffer(raw, dtype=dtype, offset=84, count=count)["vertices"].astype(float)


def font(size: int, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def add_triangles(plot, triangles, color):
    flat = triangles.reshape(-1, 3)
    faces = np.column_stack((np.full(len(triangles), 3), np.arange(len(flat)).reshape(-1, 3)))
    plot.add_mesh(pv.PolyData(flat, faces.ravel()), color=color, smooth_shading=False,
                  ambient=.3, diffuse=.7, specular=.15, specular_power=18, show_edges=False)


def image_for(triangles, meshes, record, points, lower, upper, floor_y, config, result, zoom, episode_id):
    plot = pv.Plotter(off_screen=True, window_size=[1200, 600])
    plot.set_background("#fcfcfb")  # pyright: ignore[reportCallIssue,reportArgumentType]
    add_triangles(plot, triangles, COLORS[0])
    labels = [f"Tool {index + 1}" for index in range(len(meshes))]
    for vertices, pose, color, label in zip(meshes, record["tool_poses"], COLORS[1:], labels):
        pose = np.asarray(pose, dtype=float)
        transformed = vertices @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3]
        add_triangles(plot, transformed, color)
        center = transformed.reshape(-1, 3).mean(axis=0)
        center[1] = transformed[:, :, 1].max() + .012
        plot.add_point_labels([center], [label], font_size=15, text_color="#242522",
                              show_points=False, shape=None, always_visible=True)
    table = np.array([[lower[0], floor_y, lower[2]], [upper[0], floor_y, lower[2]],
                      [upper[0], floor_y, upper[2]], [lower[0], floor_y, upper[2]]])
    plot.add_mesh(pv.PolyData(table, [4, 0, 1, 2, 3]), color="#e1e0d9", lighting=False)
    target = (lower + upper) / 2
    direction = np.array([1.05, .85, -1.65]); direction /= np.linalg.norm(direction)
    right = np.cross([0, 1, 0], direction); right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    corners = np.array([[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1])
                        for z in (lower[2], upper[2])]) - target
    tangent = np.tan(np.deg2rad(17))
    distance = 1.10 * max(np.max(np.abs(corners @ up) / tangent + corners @ direction),
                          np.max(np.abs(corners @ right) / (2 * tangent) + corners @ direction))
    plot.camera_position = [target + direction * distance, target, [0, 1, 0]]
    plot.camera.view_angle = 34
    plot.camera.zoom(zoom)
    plot.camera.clipping_range = (.001, 10)
    plot.enable_anti_aliasing("ssaa")
    rgb = plot.screenshot(return_img=True)
    plot.close()
    if rgb is None:
        raise RuntimeError("Offscreen renderer did not return an image")
    image = Image.new("RGB", (1200, 800), "#fcfcfb")
    image.paste(Image.fromarray(cast(np.ndarray, rgb)), (0, 100))
    draw = ImageDraw.Draw(image)
    parameters = config["parameters"]
    status = " | PARTIAL RUN" if result["status"] != "completed" else ""
    draw.text((32, 18), f"{episode_id} | dataset forward simulation{status}", font=font(24, True), fill="#242522")
    line = (f"E = {parameters['youngs_modulus']/1000:.5g} kPa | nu = {parameters['poisson_ratio']:.5g} | "
            f"viscosity = {parameters['viscosity']:.5g} Pa s | plastic = "
            f"{parameters['plastic_min']:.5g}-{parameters['plastic_max']:.5g}")
    draw.text((32, 55), line, font=font(17), fill="#242522")
    source_frame = record["source_frame"] if record["source_frame"] is not None else "between samples"
    draw.text((32, 80), f"Frame {source_frame} | t = {record['sim_time_s']:.4f} s | "
              f"lowest particle = {(points[:, 1].min() - floor_y) * 1000:.3f} mm above floor",
              font=font(15), fill="#242522")
    draw.text((32, 716), f"mass = {config['mass_kg']:.6g} kg | density = {config['density_kg_m3']:.6g} kg/m3 | "
              f"floor retention = {parameters['floor_retention']:.5g} | "
              f"tool friction = {config['simulation'].get('tool_friction_coefficient', 0):.5g}",
              font=font(15), fill="#242522")
    draw.text((32, 746), "Rendering-only density boundary; saved particle positions and tool poses are unchanged.",
              font=font(13), fill="#242522")
    draw.text((32, 770), f"Camera zoom {zoom:g}x | Y is up | floor y = {floor_y:.6g} m",
              font=font(13), fill="#242522")
    return image


def render(run_dir: Path, output: Path, zoom: float, ffmpeg=None, ffprobe=None) -> Path:
    started = time.perf_counter()
    tools = video_tools(ffmpeg, ffprobe)
    simulation = run_dir / "simulation"
    result_path = simulation / "simulation_result.json"
    prepared_path = simulation / "prepared_inputs.json"
    config_path = run_dir / "forward_config.json"
    launcher_path = run_dir / "launcher_manifest.json"
    for path in (result_path, prepared_path, config_path, launcher_path):
        if not path.is_file():
            raise ValueError(f"Missing forward-video input: {path}")
    result = json.loads(result_path.read_text())
    prepared = json.loads(prepared_path.read_text())
    config = json.loads(config_path.read_text())
    launcher = json.loads(launcher_path.read_text())
    collision = prepared["provenance"]["collision"]
    inputs = [result_path, prepared_path, config_path, launcher_path]
    for record in prepared["provenance"]["input_files"].values():
        if "sha256" in record:
            path = Path(record["path"])
            if sha256(path) != record["sha256"]:
                raise ValueError(f"Prepared input changed: {path}")
            inputs.append(path)
    meshes = []
    for asset in collision["assets_in_stream_order"]:
        path = Path(asset["path"])
        if sha256(path) != asset["sha256"]:
            raise ValueError(f"Collision asset changed: {path}")
        inputs.append(path)
        vertices = stl(path) * collision["mesh_scale"]
        vertices = vertices @ Rotation.from_euler("xyz", asset["visual_rpy_rad"]).as_matrix().T
        vertices += np.asarray(asset["visual_origin_m"])
        meshes.append(vertices)
    records = list(result["frames"])
    if result["completed_steps"] != records[-1]["step"]:
        records.append({"source_frame": None, "step": result["completed_steps"],
                        "sim_time_s": result["sim_time_s"], "particles": "last_valid_particles.npy",
                        "tool_poses": result["last_valid_tool_poses"]})
    count = int(config["simulation"]["n_particles"])
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    for record in records:
        path = simulation / record["particles"]
        inputs.append(path)
        points = np.load(path, allow_pickle=False)
        if points.shape != (count, 3) or not np.isfinite(points).all():
            raise ValueError(f"Invalid saved particles: {path}")
        lower = np.minimum(lower, points.min(axis=0)); upper = np.maximum(upper, points.max(axis=0))
        for vertices, pose in zip(meshes, record["tool_poses"]):
            transformed = vertices @ Rotation.from_quat(np.asarray(pose)[3:]).as_matrix().T + np.asarray(pose)[:3]
            lower = np.minimum(lower, transformed.min(axis=(0, 1)))
            upper = np.maximum(upper, transformed.max(axis=(0, 1)))
    before = {str(path): sha256(path) for path in dict.fromkeys(inputs)}
    output.mkdir(parents=False, exist_ok=False)
    frames_dir = output / "frames"; frames_dir.mkdir()
    lower -= .015; upper += .015
    floor_y = float(config["simulation"]["floor_y"])
    lower[1] = min(lower[1], floor_y - .005)
    particle_volume = float(prepared["mass"]["particle_volume_m3"])
    indices = sorted(set([*range(0, len(records), 3), len(records) // 2, len(records) - 1]))
    details = []
    for number, index in enumerate(indices):
        record = records[index]
        points = np.load(simulation / record["particles"], allow_pickle=False)
        triangles, info = density_boundary(points, particle_volume)
        frame = image_for(triangles, meshes, record, points, lower, upper, floor_y,
                          config, result, zoom, launcher["episode_id"])
        frame.save(frames_dir / f"frame_{number:05d}.png")
        details.append({"movie_frame": number, "source_frame": record["source_frame"],
                        "time_s": record["sim_time_s"], **info})
        if number % 10 == 0 or number == len(indices) - 1:
            print(f"Rendered {number + 1}/{len(indices)}", flush=True)
    times = [records[index]["sim_time_s"] for index in indices]
    tail = float(np.median(np.diff(times))) if len(times) > 1 else .1
    timing = output / "frame_timing.txt"
    with timing.open("x") as stream:
        for number, value in enumerate(times):
            duration = times[number + 1] - value if number + 1 < len(times) else tail
            stream.write(f"file 'frames/frame_{number:05d}.png'\nduration {duration:.9f}\n")
        stream.write(f"file 'frames/frame_{len(times) - 1:05d}.png'\n")
    video = output / "requested_material_perspective.mp4"
    command, probe = encode_video(timing, video, *tools, minimum_frames=len(indices))
    after = {path: sha256(Path(path)) for path in before}
    if before != after:
        raise RuntimeError("Simulation or prepared inputs changed during rendering")
    write_new_json(output / "render_manifest.json", {
        "schema": "taichidough/dataset-forward-render/v2", "source_run": str(run_dir),
        "simulation_status": result["status"], "frames": details, "parameters": config["parameters"],
        "mass_kg": config["mass_kg"], "density_kg_m3": config["density_kg_m3"],
        "camera_zoom": zoom, "particle_volume_m3": particle_volume,
        "input_sha256": before, "inputs_unchanged": True,
        "ffmpeg_command": command, "ffprobe": probe, "video_sha256": sha256(video),
        "elapsed_s": time.perf_counter() - started,
    })
    print(f"VIDEO: {video}", flush=True)
    return video


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--camera-zoom", type=camera_zoom, default=1.0)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        parser.error("run-dir does not exist")
    output = (args.output_dir or run_dir / "perspective").expanduser().resolve()
    if output == run_dir or not output.is_relative_to(run_dir):
        parser.error("output-dir must be a fresh child of run-dir")
    try:
        render(run_dir, output, args.camera_zoom, args.ffmpeg, args.ffprobe)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
