#!/usr/bin/env python3
"""Visualize saved calibration evaluations without loading a simulation runtime."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

if __package__:
    from .visualization_report import create_report
else:
    from visualization_report import create_report


REPORT_NAME = "dynamic_topview_metrics.json"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_directory(value):
    path = Path(value).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".json":
            raise ValueError("Expected a run directory or a JSON file inside one")
        path = path.parent
    if not path.is_dir():
        raise ValueError(f"Run directory does not exist: {path}")
    return path


def available_reports(run_dir, split):
    return sorted(run_dir.glob(f"{split}_*_strict/evaluation/{REPORT_NAME}"))


def choose_report(run_dir, split):
    matches = available_reports(run_dir, split)
    if not matches:
        raise ValueError(
            f"No saved {split} strict evaluation under {run_dir}. "
            "Parameter JSON alone cannot reconstruct a trajectory. "
            "Copy the strict evaluation directory including arrays/, or select --metrics explicitly."
        )
    if len(matches) != 1:
        raise ValueError("Multiple saved evaluations; choose one with --metrics:\n" +
                         "\n".join(str(path) for path in matches))
    return matches[0].resolve()


def validate_saved_arrays(metrics):
    report = json.loads(metrics.read_text())
    if report.get("benchmark") != "dynamic-topview-proxy-replay/v1" or not report.get("frames"):
        raise ValueError("Expected a nonempty saved dynamic-topview-proxy-replay/v1 evaluation")
    root = metrics.parent.resolve()
    seen = set()
    previous_time = -math.inf
    inputs = {}
    crop = None
    for row in report["frames"]:
        frame = row["source_frame"]
        elapsed = float(row["time_s"])
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or frame in seen:
            raise ValueError("Source frames must be unique nonnegative integers")
        if not math.isfinite(elapsed) or elapsed < previous_time:
            raise ValueError("Saved observation times must be finite and nondecreasing")
        seen.add(frame)
        previous_time = elapsed
        relative = Path(row["arrays"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"Saved array path must stay inside the evaluation directory: {relative}")
        if not path.is_file():
            raise ValueError(f"Missing saved array: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            expected = None
            for prefix in ("real", "sim"):
                depth, valid = arrays[f"{prefix}_depth"], arrays[f"{prefix}_valid"]
                if depth.ndim != 2 or valid.shape != depth.shape or valid.dtype != np.bool_:
                    raise ValueError(f"Invalid depth/mask arrays: {path}")
                if expected is not None and depth.shape != expected:
                    raise ValueError(f"Real and simulated images differ in dimensions: {path}")
                expected = depth.shape
                if not np.isfinite(depth[valid]).all():
                    raise ValueError(f"Nonfinite depth marked valid: {path}")
            camera = report["camera"]
            if expected != (camera["height"], camera["width"]):
                raise ValueError(f"Saved image dimensions disagree with camera metadata: {path}")
            ys, xs = np.nonzero(arrays["real_valid"] | arrays["sim_valid"])
            if len(xs):
                bounds = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                crop = bounds if crop is None else [min(crop[0], bounds[0]), min(crop[1], bounds[1]),
                                                    max(crop[2], bounds[2]), max(crop[3], bounds[3])]
        inputs[str(relative)] = file_hash(path)
    if crop is not None:
        margin = max(8, round(0.08 * max(crop[2] - crop[0], crop[3] - crop[1])))
        crop = [max(0, crop[0] - margin), max(0, crop[1] - margin),
                min(report["camera"]["width"], crop[2] + margin),
                min(report["camera"]["height"], crop[3] + margin)]
    report["comparison_crop_xyxy"] = crop
    return report, inputs


def load_run_context(run_dir):
    context = {"run_name": run_dir.name, "source_run": str(run_dir)}
    for name in ("selected_parameters.json", "result.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        value = json.loads(path.read_text())
        context.update({
            "parameter_record": name,
            "parameter_record_sha256": file_hash(path),
            "parameters": value.get("best_parameters", value.get("parameters", {})),
            "physics_version": value.get("physics_version"),
            "selection_frames": value.get("selection_frames", []),
            "ignore_recompute_mismatch": value.get("ignore_recompute_mismatch"),
        })
        break
    return context


def load_font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def comparison_frame(output_dir, report, row):
    image = Image.new("RGB", (1280, 600), (22, 25, 29))
    draw = ImageDraw.Draw(image)
    title_font, text_font, small_font = load_font(25), load_font(18), load_font(14)
    draw.text((24, 15), "Fitted simulation | saved depth comparison", fill="white", font=title_font)
    role = row.get("scoring_role", "selection membership unavailable")
    draw.text((24, 52), f"Frame {row['source_frame']}  |  t = {row['time_s']:.3f} s  |  {role}",
              fill=(203, 213, 224), font=text_font)
    for index, (key, title) in enumerate((("real", "Observed dough"),
                                        ("sim", "Fitted simulation"),
                                        ("residual", "Depth error: simulation minus real"))):
        left = 20 + index * 420
        draw.text((left, 98), title, fill="white", font=text_font)
        with Image.open(output_dir / row["images"][key]) as source:
            pixels = source.convert("RGB")
            crop = report["display"].get("comparison_crop_xyxy")
            if crop is not None:
                pixels = pixels.crop(tuple(crop))
            panel = ImageOps.contain(pixels, (400, 350), Image.Resampling.NEAREST)
            image.paste(panel, (left + (400 - panel.width) // 2, 130 + (350 - panel.height) // 2))
    display = report["display"]
    crop_note = "one fixed image crop across all frames" if display.get("comparison_crop_xyxy") else "full camera image"
    draw.text((24, 505), f"Shared optical-depth range: {display['depth_min_m']:.4f} to {display['depth_max_m']:.4f} m | {crop_note}",
              fill=(203, 213, 224), font=small_font)
    draw.text((24, 529), f"Error range: +/- {display['residual_limit_m'] * 1000:.2f} mm. Blue: nearer; red: farther. Dark: unavailable, not zero error.",
              fill=(203, 213, 224), font=small_font)
    draw.text((24, 554), "Saved observations only; no simulation rerun. Selection labels do not establish independent geometry calibration.",
              fill=(178, 188, 201), font=small_font)
    return image


def video_indices(rows, fps):
    times = np.asarray([row["time_s"] for row in rows], dtype=float)
    count = max(1, round((times[-1] - times[0]) * fps) + 1)
    timeline = np.linspace(times[0], times[-1], count)
    return np.maximum(0, np.searchsorted(times, timeline, side="right") - 1)


def write_video(output_dir, report, fps, executable):
    target = output_dir / "depth_comparison.mp4"
    command = [executable, "-v", "error", "-n", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", "1280x600", "-framerate", str(fps), "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(target)]
    indices = video_indices(report["frames"], fps)
    error_path = output_dir / "ffmpeg.log"
    with error_path.open("x") as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            for index in indices:
                frame = comparison_frame(output_dir, report, report["frames"][int(index)])
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"ffmpeg failed with exit {code}; see {error_path}")
    return {"file": target.name, "fps": fps, "frame_count": len(indices),
            "duration_s": len(indices) / fps,
            "sampling": "Uniform first-to-last captured-time sampling; hold latest available observation",
            "source_frames": [report["frames"][int(index)]["source_frame"] for index in indices]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Run directory, or selected_parameters.json/result.json inside it")
    source.add_argument("--metrics", type=Path, help="Explicit local strict evaluation metrics JSON")
    parser.add_argument("--split", choices=("training", "validation"), default="validation")
    parser.add_argument("--output-dir", type=Path, help="New output directory; defaults to a unique sibling of the run")
    parser.add_argument("--list", action="store_true", help="List saved reports without writing outputs")
    parser.add_argument("--video", action="store_true", help="Also encode depth_comparison.mp4 using ffmpeg")
    parser.add_argument("--full-frame", action="store_true", help="Disable the shared camera-image crop in comparison stills/video")
    parser.add_argument("--fps", type=float, default=30.0, help="MP4 sampling rate; interactive HTML follows captured timestamps")
    args = parser.parse_args(argv)
    if not math.isfinite(args.fps) or not 0 < args.fps <= 240:
        parser.error("--fps must be finite and in (0, 240]")
    try:
        if args.metrics:
            metrics = args.metrics.expanduser().resolve()
            if not metrics.is_file():
                raise ValueError(f"Metrics JSON does not exist: {metrics}")
            run_dir = metrics.parent.parent.parent
        else:
            run_dir = run_directory(args.run_dir)
            if args.list:
                for split in ("training", "validation"):
                    matches = available_reports(run_dir, split)
                    print(split + ": " + ("\n  ".join(str(path) for path in matches) if matches else "none"))
                return 0
            metrics = choose_report(run_dir, args.split)
        if args.list:
            print(metrics)
            return 0
        ffmpeg = shutil.which("ffmpeg") if args.video else None
        if args.video and not ffmpeg:
            raise ValueError("--video requires ffmpeg on PATH; omit --video for HTML and PNG output")
        input_report, input_hashes = validate_saved_arrays(metrics)
        context = load_run_context(run_dir)
        if args.output_dir:
            output_dir = args.output_dir.expanduser().resolve()
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_dir = run_dir.parent / f"{run_dir.name}_visualization_{stamp}_{uuid.uuid4().hex[:6]}"
        if output_dir.exists():
            raise ValueError(f"Output directory already exists; choose a new directory: {output_dir}")
        if output_dir == run_dir or output_dir.is_relative_to(run_dir):
            raise ValueError("Choose an output directory outside the original run to keep it unchanged")
        html = create_report(metrics, output_dir, run_context=context)
        report = json.loads((output_dir / "visualization_manifest.json").read_text())
        report["display"]["comparison_crop_xyxy"] = None if args.full_frame else input_report["comparison_crop_xyxy"]
        (output_dir / "visualization_manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        stills = []
        for index in sorted({0, len(report["frames"]) // 2, len(report["frames"]) - 1}):
            row = report["frames"][index]
            name = f"comparison_frame_{row['source_frame']:06d}.png"
            comparison_frame(output_dir, report, row).save(output_dir / name)
            stills.append(name)
        video = write_video(output_dir, report, args.fps, ffmpeg) if args.video else None
        evidence = {"schema": "taichidough/saved-run-visualization/v1", "source_run": str(run_dir),
                    "metrics_json": str(metrics), "metrics_sha256": file_hash(metrics),
                    "array_sha256": input_hashes, "run_context": context,
                    "simulation_rerun": False, "calibration_rerun": False,
                    "recorded_frame_count": len(report["frames"]), "stills": stills, "video": video}
        with (output_dir / "saved_run_visualization.json").open("x") as handle:
            json.dump(evidence, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"HTML: {html}")
        print(f"Contact sheet: {output_dir / 'temporal_contact_sheet.png'}")
        if video:
            print(f"Video: {output_dir / video['file']}")
        print(f"Saved frames: {len(report['frames'])}; no simulation or calibration rerun")
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(2, f"Visualization error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
