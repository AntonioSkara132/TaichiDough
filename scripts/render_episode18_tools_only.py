#!/usr/bin/env python3
"""Render an interpolated two-tool trajectory video without MPM geometry."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


TOOL_COLORS = ((31, 119, 180), (214, 39, 40))
PROJECTIONS = (("XY", 0, 1), ("XZ", 0, 2), ("YZ", 1, 2))
AXIS_NAMES = ("X", "Y", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, help="Interpolated paths .pt file.")
    parser.add_argument("output_dir", type=Path, help="Directory for frames and MP4 output.")
    parser.add_argument("--frame-step", type=int, default=4)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--trail", type=int, default=48)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{suffix}", size=size)


def padded_range(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanmin(values), np.nanmax(values)
    padding = max((hi - lo) * 0.08, 1e-3)
    return float(lo - padding), float(hi + padding)


def data_to_pixels(
    values: np.ndarray, x_range: tuple[float, float], y_range: tuple[float, float], box: tuple[int, int, int, int]
) -> np.ndarray:
    left, top, right, bottom = box
    x = left + (values[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * (right - left)
    y = bottom - (values[:, 1] - y_range[0]) / (y_range[1] - y_range[0]) * (bottom - top)
    return np.column_stack((x, y))


def draw_polyline(draw: ImageDraw.ImageDraw, points: np.ndarray, color: tuple[int, int, int], width: int) -> None:
    if len(points) > 1:
        draw.line([tuple(point) for point in points], fill=color, width=width, joint="curve")


def main() -> None:
    args = parse_args()
    payload = torch.load(args.paths, map_location="cpu", weights_only=False)
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ValueError("Expected a one-episode list of path dictionaries.")
    episode = payload[0]
    path = episode["path"].detach().cpu().numpy().astype(np.float64)
    validity = episode.get("stream_validity")
    valid = validity.detach().cpu().numpy().astype(bool) if torch.is_tensor(validity) else np.isfinite(path[:, :, :3]).all(axis=2)
    labels = episode.get("pose_frames", [f"tool_{index}" for index in range(path.shape[1])])
    xyz = path[:, :, :3]
    if xyz.ndim != 3 or xyz.shape[1] != 2:
        raise ValueError(f"Expected [frames, 2, features], got {xyz.shape}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir()
    output_video = args.output_dir / "episode18_kugla_tools_interpolated.mp4"

    ranges = [padded_range(xyz[:, :, axis][np.isfinite(xyz[:, :, axis])]) for axis in range(3)]
    frame_indices = list(range(0, len(xyz), max(args.frame_step, 1)))
    if frame_indices[-1] != len(xyz) - 1:
        frame_indices.append(len(xyz) - 1)

    title_font, label_font, small_font = font(22, True), font(15), font(13)
    panel_width = args.width // 3
    for output_index, frame_index in enumerate(frame_indices):
        image = Image.new("RGB", (args.width, args.height), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, args.width, 42), fill=(246, 247, 249))
        draw.text((16, 11), "Episode 18 (kugla): interpolated tool paths only", font=title_font, fill=(25, 25, 25))
        draw.text((args.width - 242, 14), f"source {frame_index + 1}/{len(xyz)}", font=small_font, fill=(80, 80, 80))

        for panel_index, (name, x_axis, y_axis) in enumerate(PROJECTIONS):
            x0 = panel_index * panel_width
            box = (x0 + 48, 92, x0 + panel_width - 18, args.height - 66)
            left, top, right, bottom = box
            draw.rectangle(box, outline=(155, 155, 155), width=1)
            for fraction in (0.25, 0.5, 0.75):
                x = left + fraction * (right - left)
                y = top + fraction * (bottom - top)
                draw.line((x, top, x, bottom), fill=(232, 232, 232), width=1)
                draw.line((left, y, right, y), fill=(232, 232, 232), width=1)
            draw.text((x0 + panel_width // 2 - 13, 60), name, font=title_font, fill=(25, 25, 25))
            draw.text((x0 + panel_width // 2 - 8, args.height - 42), AXIS_NAMES[x_axis], font=label_font, fill=(65, 65, 65))
            draw.text((x0 + 14, 74), AXIS_NAMES[y_axis], font=label_font, fill=(65, 65, 65))

            for tool_index, label in enumerate(labels):
                mask = valid[:, tool_index] & np.isfinite(xyz[:, tool_index, [x_axis, y_axis]]).all(axis=1)
                full = xyz[mask, tool_index][:, [x_axis, y_axis]]
                full_pixels = data_to_pixels(full, ranges[x_axis], ranges[y_axis], box)
                faded = tuple((np.array(TOOL_COLORS[tool_index]) * 0.35 + 255 * 0.65).astype(int))
                draw_polyline(draw, full_pixels, faded, 1)

                start = max(0, frame_index - args.trail)
                segment_mask = valid[start : frame_index + 1, tool_index] & np.isfinite(xyz[start : frame_index + 1, tool_index, [x_axis, y_axis]]).all(axis=1)
                segment = xyz[start : frame_index + 1, tool_index][:, [x_axis, y_axis]][segment_mask]
                segment_pixels = data_to_pixels(segment, ranges[x_axis], ranges[y_axis], box)
                draw_polyline(draw, segment_pixels, TOOL_COLORS[tool_index], 3)

                current = xyz[frame_index, tool_index, [x_axis, y_axis]]
                if valid[frame_index, tool_index] and np.isfinite(current).all():
                    px, py = data_to_pixels(current[None, :], ranges[x_axis], ranges[y_axis], box)[0]
                    radius = 6
                    draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=TOOL_COLORS[tool_index], outline="white", width=2)

        legend_y = args.height - 19
        legend_x = 18
        for tool_index, label in enumerate(labels):
            draw.ellipse((legend_x, legend_y - 5, legend_x + 10, legend_y + 5), fill=TOOL_COLORS[tool_index])
            draw.text((legend_x + 15, legend_y - 8), label, font=small_font, fill=(35, 35, 35))
            legend_x += 185
        image.save(frames_dir / f"frame_{output_index:04d}.png")

    command = [
        "ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_video),
    ]
    subprocess.run(command, check=True)
    print(f"Wrote {output_video}")
    print(f"Frames: {len(frame_indices)}; FPS: {args.fps}; duration: {len(frame_indices) / args.fps:.3f} s")


if __name__ == "__main__":
    main()
