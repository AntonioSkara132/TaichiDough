#!/usr/bin/env python3
"""Render saved forward-video-v2 states without rerunning simulation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import subprocess
import sys
import uuid


def executable(value: str) -> Path:
    candidate = shutil.which(value) if value and os.sep not in value else value
    if not candidate:
        raise ValueError("render-python must be a nonempty executable path")
    path = Path(os.path.abspath(Path(candidate).expanduser()))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"render-python is not executable: {path}")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--render-python", default=sys.executable)
    parser.add_argument("--camera-zoom", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    args = parser.parse_args(argv)
    try:
        run_dir = args.run_dir.expanduser().resolve()
        if not (run_dir / "simulation" / "simulation_result.json").is_file():
            raise ValueError("run-dir has no saved forward simulation")
        python = executable(args.render_python)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = (args.output_dir or run_dir / f"perspective_recovery_{stamp}_{uuid.uuid4().hex[:6]}").expanduser().resolve()
        if output == run_dir or not output.is_relative_to(run_dir) or output.exists():
            raise ValueError("output-dir must be a fresh child of run-dir")
        command = [python, Path(__file__).with_name("render.py"), "--run-dir", run_dir,
                   "--output-dir", output, "--camera-zoom", str(args.camera_zoom)]
        if args.ffmpeg:
            command.extend(("--ffmpeg", args.ffmpeg))
        if args.ffprobe:
            command.extend(("--ffprobe", args.ffprobe))
        print("Running: " + " ".join(map(str, command)), flush=True)
        completed = subprocess.run(list(map(str, command)), check=False)
        return completed.returncode
    except (ValueError, OSError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
