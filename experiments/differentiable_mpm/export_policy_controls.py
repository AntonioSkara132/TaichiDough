#!/usr/bin/env python3
"""Export simulator controls for the five learned-action comparison conditions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .policy_adapter import CONDITIONS, build_condition_poses, save_conditions


def _matrix(path: Path, shape):
    value = np.asarray(json.loads(path.read_text()), dtype=np.float64)
    if value.shape != shape:
        raise ValueError(f"{path} must contain a matrix with shape {shape}")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-poses", type=Path, required=True,
                        help=".npy [N,2,7] raw source-frame recorded poses")
    parser.add_argument("--recorded-times", type=Path, required=True,
                        help=".npy [N] relative recorded times in seconds")
    parser.add_argument("--policy-export", type=Path, required=True,
                        help="dom_retrieval export directory containing prediction_xyz.npy and phase.npy")
    parser.add_argument("--control-dt", type=float, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--scene-from-source", type=Path,
                        help="JSON 4x4 calibrated scene_from_source matrix")
    parser.add_argument("--marker-from-tool-frames", type=Path,
                        help="JSON [2,4,4] marker_from_tool_frame matrices")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.control_dt <= 0 or args.duration <= 0:
        raise ValueError("control-dt and duration must be positive")
    recorded_poses = np.load(args.recorded_poses, allow_pickle=False)
    recorded_times = np.load(args.recorded_times, allow_pickle=False)
    predicted_positions = np.load(args.policy_export / "prediction_xyz.npy", allow_pickle=False)
    phase = np.load(args.policy_export / "phase.npy", allow_pickle=False)
    scene = np.eye(4) if args.scene_from_source is None else _matrix(args.scene_from_source, (4, 4))
    marker = np.broadcast_to(np.eye(4), (2, 4, 4)).copy() if args.marker_from_tool_frames is None else _matrix(args.marker_from_tool_frames, (2, 4, 4))
    control_times = np.arange(max(1, int(np.ceil(args.duration / args.control_dt))), dtype=np.float64) * args.control_dt
    control_times = np.minimum(control_times, args.duration)
    predicted_times = phase * args.duration
    conditions = {}
    for condition in CONDITIONS:
        conditions[condition] = build_condition_poses(
            condition, control_times, recorded_times, recorded_poses,
            predicted_times, predicted_positions, scene, marker,
        )
    manifest = save_conditions(args.output, conditions)
    manifest.update({"control_dt_s": args.control_dt, "duration_s": args.duration,
                     "control_times_s": control_times.tolist(),
                     "scene_from_source": scene.tolist(),
                     "marker_from_tool_frames": marker.tolist(),
                     "policy_export": str(args.policy_export.resolve()),
                     "recorded_poses": str(args.recorded_poses.resolve()),
                     "recorded_times": str(args.recorded_times.resolve())})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
