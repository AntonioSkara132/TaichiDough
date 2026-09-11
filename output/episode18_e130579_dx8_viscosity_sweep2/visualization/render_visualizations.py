#!/usr/bin/env python3
"""Visualize saved viscosity-sweep particles using the existing replay renderer.

This does not initialize Taichi or run a material simulation. Shared tool poses
permit rasterizing the tool meshes once per frame. The first and last composite
are checked against render_replay_depth_tools.render_frame before delivery.
Original sweep metadata and particle arrays are never overwritten.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
import render_replay_depth_tools as render

BACKGROUND = (18, 18, 18)
TEXT = (236, 241, 245)
MUTED = (173, 185, 194)
ACCENT = (239, 187, 100)
FPS = 7.572
WIDTH, HEIGHT = 960, 720
PANEL_WIDTH, PANEL_HEIGHT = 640, 480
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(root: Path, output: Path):
    summary = json.loads((root / "viscosity_sweep.json").read_text())
    cases = sorted(summary["cases"], key=lambda case: case["viscosity_pa_s"])
    if len(cases) != 5 or any(case["status"] != "complete" or not case["loss"]["valid"] for case in cases):
        raise ValueError("Expected five complete, valid viscosity cases")
    reference = None
    for case in cases:
        simulation = root / "cases" / case["id"] / "simulation"
        source = simulation / "camera_parameters.json"
        metadata = deepcopy(json.loads(source.read_text()))
        if metadata.get("replay", {}).get("tool_pose_frame") != "mesh_tool_link":
            raise ValueError(f"Unexpected tool-pose frame in {case['id']}")
        sequence = metadata["frames"]
        for frame in sequence:
            frame["particles"] = str(simulation / Path(frame["particles"]).name)
            if not Path(frame["particles"]).is_file():
                raise FileNotFoundError(frame["particles"])
            render.validate_tool_poses(frame)
            for view in frame.get("views", []):
                for key in ("depth_array", "depth_image", "pointcloud_array"):
                    if key in view:
                        view[key] = str(simulation / Path(view[key]).name)
        signature = [(frame["source_frame"], frame["target_time_s"], frame["tool_poses_scene"]) for frame in sequence]
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError("Cases must have identical frame times and tool poses")
        directory = output / "resolved_replays" / case["id"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "camera_parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
        case["frames"] = sequence
        case["metadata_sha256"] = sha256(source)
        case["frames_directory"] = output / f"{case['id']}_frames"
        case["frames_directory"].mkdir(exist_ok=True)
    return summary, cases


def labeled_image(rgb: np.ndarray, case: dict, frame: dict, modulus: float) -> Image.Image:
    image = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 100), fill=BACKGROUND)
    draw.text((22, 14), f"Viscosity = {case['viscosity_pa_s']:g} Pa s", font=font(25, True), fill=ACCENT)
    draw.text((22, 52), f"Training loss {case['loss']['weighted_total']:.9f}", font=font(19), fill=TEXT)
    draw.text((22, 80), "", font=font(12), fill=TEXT)
    draw.rectangle((0, HEIGHT - 54, WIDTH, HEIGHT), fill=BACKGROUND)
    label = f"Source {frame['source_frame']:02d}  |  t = {frame['target_time_s']:.3f} s  |  E = {modulus / 1000:.3f} kPa"
    draw.text((22, HEIGHT - 42), label, font=font(17), fill=TEXT)
    draw.text((WIDTH - 172, 23), "ISOMETRIC", font=font(15), fill=MUTED)
    return image


def information_panel(cases: list[dict], frame: dict) -> Image.Image:
    image = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    best = min(cases, key=lambda case: case["loss"]["weighted_total"])
    baseline = cases[0]["loss"]["weighted_total"]
    improvement = 100 * (baseline - best["loss"]["weighted_total"]) / baseline
    draw.text((30, 24), "EPISODE 18", font=font(28, True), fill=TEXT)
    draw.text((30, 66), "Viscosity sweep · same camera and shading", font=font(18), fill=MUTED)
    draw.text((30, 110), "Viscosity [Pa s]", font=font(19, True), fill=TEXT)
    draw.text((294, 110), "Training loss", font=font(19, True), fill=TEXT)
    for index, case in enumerate(cases):
        y = 146 + index * 34
        color = ACCENT if case is best else TEXT
        draw.text((30, y), f"{case['viscosity_pa_s']:g}", font=font(21), fill=color)
        draw.text((294, y), f"{case['loss']['weighted_total']:.9f}", font=font(21), fill=color)
    draw.text((30, 332), f"Lowest tested loss: eta = {best['viscosity_pa_s']:g}; improvement {improvement:.4f}%", font=font(16), fill=ACCENT)
    draw.text((30, 366), "Scored frames 1–60 · no held-out comparison", font=font(16), fill=MUTED)
    draw.text((30, 394), "Effective coefficients for this fixed numerical setup", font=font(16), fill=MUTED)
    draw.text((30, 428), f"t = {frame['target_time_s']:.3f} s · slow-motion playback (~4x)", font=font(16), fill=MUTED)
    return image


def encode(directory: Path, destination: Path, log_directory: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing video: {destination}")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-n",
        "-framerate", str(FPS), "-start_number", "0",
        "-i", str(directory / "frame_%06d.png"),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-threads", "2",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    (log_directory / f"{destination.stem}.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_read_frames,r_frame_rate,pix_fmt:format=duration",
        "-of", "json", str(destination),
    ], text=True, capture_output=True, check=True)
    return json.loads(probe.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, output = args.sweep_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*_isometric.mp4")):
        raise FileExistsError("Output already contains videos; choose a new output directory")
    start = time.monotonic()
    summary, cases = load_cases(root, output)
    camera = render.make_isometric_camera(np.array([.72, .42, 1.34]), np.array([.23, .04, .73]), WIDTH, HEIGHT, 52)
    meshes = render.load_tool_link_meshes(REPO / "meshes/ur_spathla.stl", REPO / "meshes/gen3_spathla.stl", .001)
    frames = cases[0]["frames"]
    near, far = render.depth_display_range([frame for case in cases for frame in case["frames"]], camera, False)
    print(f"Shared depth shading: {near:.9f}–{far:.9f} m; {len(frames)} synchronized frames", flush=True)
    comparison_frames = output / "comparison_frames"
    comparison_frames.mkdir(exist_ok=True)
    checks = []
    for index, reference_frame in enumerate(frames):
        tool_depth = np.full((HEIGHT, WIDTH), camera["zFar"], dtype=np.float64)
        tool_owner = np.full((HEIGHT, WIDTH), -1, dtype=np.int8)
        poses = render.validate_tool_poses(reference_frame)
        for tool_index, (mesh, pose) in enumerate(zip(meshes, poses)):
            vertices = render.mesh_scene_vertices(mesh, pose)
            render.rasterize_triangles(tool_depth, tool_owner, vertices, camera, tool_index)
        comparison = Image.new("RGB", (3 * PANEL_WIDTH, 2 * PANEL_HEIGHT), BACKGROUND)
        for case_index, case in enumerate(cases):
            frame = case["frames"][index]
            particles = np.load(frame["particles"])
            if particles.shape != (24000, 3) or not np.isfinite(particles).all():
                raise ValueError(f"Invalid saved particles in {case['id']}, frame {index}")
            depth, *_ = render.rasterize_depth(particles, WIDTH, HEIGHT, camera, splat_radius=2)
            depth = depth.astype(np.float64)
            tools_visible = tool_depth < depth
            combined = np.minimum(depth, tool_depth)
            owner = np.where(tools_visible, tool_owner, -1).astype(np.int8)
            rgb = render.colorize_depth(combined, owner, near, far, camera["zFar"])
            if (index, case_index) in [(0, 0), (len(frames) - 1, len(cases) - 1)]:
                expected = render.render_frame(frame, camera, meshes, near, far, False)
                identical = bool(np.array_equal(rgb, expected))
                checks.append({"case": case["id"], "source_frame": frame["source_frame"], "identical_to_existing_renderer": identical})
                if not identical:
                    raise AssertionError("Shared-tool rendering differs from existing renderer")
            image = labeled_image(rgb, case, frame, summary["fixed_parameters"]["youngs_modulus_pa"])
            image.save(case["frames_directory"] / f"frame_{index:06d}.png")
            comparison.paste(image.resize((PANEL_WIDTH, PANEL_HEIGHT), RESAMPLE), ((case_index % 3) * PANEL_WIDTH, (case_index // 3) * PANEL_HEIGHT))
        comparison.paste(information_panel(cases, reference_frame), (2 * PANEL_WIDTH, PANEL_HEIGHT))
        comparison.save(comparison_frames / f"frame_{index:06d}.png")
        if index == len(frames) - 1:
            comparison.save(output / "viscosity_comparison_preview.png")
        if index % 5 == 0 or index == len(frames) - 1:
            print(f"Rendered {index + 1}/{len(frames)} synchronized frames ({time.monotonic() - start:.1f}s)", flush=True)
    logs = output / "encoding_logs"
    logs.mkdir(exist_ok=True)
    videos = {}
    for case in cases:
        destination = output / f"{case['id']}_isometric.mp4"
        videos[destination.name] = encode(case["frames_directory"], destination, logs)
        print(f"Encoded {destination.name}", flush=True)
    destination = output / "viscosity_comparison_isometric.mp4"
    videos[destination.name] = encode(comparison_frames, destination, logs)
    for name, probe in videos.items():
        stream = probe["streams"][0]
        if int(stream["nb_read_frames"]) != len(frames) or stream["pix_fmt"] != "yuv420p":
            raise ValueError(f"Video verification failed: {name}")
    manifest = {
        "source": str(root), "source_summary_sha256": sha256(root / "viscosity_sweep.json"),
        "renderer": str(REPO / "scripts/render_replay_depth_tools.py"),
        "renderer_sha256": sha256(REPO / "scripts/render_replay_depth_tools.py"),
        "no_simulation_rerun": True, "frame_count": len(frames), "fps": FPS,
        "source_frame_range": [frames[0]["source_frame"], frames[-1]["source_frame"]],
        "scored_source_frame_range": summary["training_source_frame_range_inclusive"],
        "camera": camera, "shared_display_depth_range_m": [near, far],
        "fixed_parameters": summary["fixed_parameters"], "renderer_equivalence_checks": checks,
        "cases": [{"id": case["id"], "viscosity_pa_s": case["viscosity_pa_s"],
                   "training_loss": case["loss"]["weighted_total"], "metadata_sha256": case["metadata_sha256"]} for case in cases],
        "videos": videos, "duration_s": time.monotonic() - start,
        "interpretation": "Training-only comparison of effective coefficients; tiny score differences do not establish a unique physical viscosity.",
    }
    (output / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(videos)} videos. Total rendering/encoding time {manifest['duration_s']:.1f}s", flush=True)
    print(destination, flush=True)


if __name__ == "__main__":
    main()
