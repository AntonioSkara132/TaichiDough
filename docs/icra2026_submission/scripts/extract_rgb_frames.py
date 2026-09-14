#!/usr/bin/env python3
"""Extract RGB frames nearest selected point-cloud timestamps from Episode 18."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-dir", type=Path, required=True)
    parser.add_argument("--conversion-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--topic", default="/camera/camera/color/image_raw")
    return parser.parse_args()


def decode_image(message) -> np.ndarray:
    encoding = message.encoding.lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(encoding)
    if channels is None:
        raise ValueError(f"Unsupported image encoding: {message.encoding}")
    row = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = row[:, : message.width * channels].reshape(message.height, message.width, channels)
    if encoding.startswith("bgr"):
        image = image[..., [2, 1, 0] + ([3] if channels == 4 else [])]
    if channels == 4:
        image = image[..., :3]
    return image.copy()


def main() -> int:
    args = parse_args()
    metadata = json.loads(args.conversion_metadata.read_text())
    point_stamps = metadata["episode"]["timestamps_ns"]
    if any(frame < 0 or frame >= len(point_stamps) for frame in args.frames):
        raise ValueError("Requested frame is outside conversion timestamp range")
    targets = {frame: int(point_stamps[frame]) for frame in args.frames}

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag_dir.resolve()), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if args.topic not in topic_types:
        raise ValueError(f"Topic not found: {args.topic}")
    message_type = get_message(topic_types[args.topic])
    candidates: dict[int, tuple[int, int, bytes]] = {}
    while reader.has_next():
        topic, serialized, _ = reader.read_next()
        if topic != args.topic:
            continue
        message = deserialize_message(serialized, message_type)
        stamp = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
        for frame, target in targets.items():
            delta = abs(stamp - target)
            current = candidates.get(frame)
            if current is None or delta < current[0]:
                candidates[frame] = (delta, stamp, serialized)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for frame in args.frames:
        delta, stamp, serialized = candidates[frame]
        message = deserialize_message(serialized, message_type)
        rgb = decode_image(message)
        output = args.output_dir / f"recorded_rgb_frame_{frame:06d}.png"
        PILImage.fromarray(rgb).save(output)
        records.append({
            "source_frame": frame,
            "pointcloud_timestamp_ns": targets[frame],
            "image_timestamp_ns": stamp,
            "absolute_time_difference_ns": delta,
            "encoding": message.encoding,
            "width": int(message.width),
            "height": int(message.height),
            "output": output.name,
        })
    manifest = {
        "schema": "taichidough/icra-recorded-rgb-extraction/v1",
        "bag_directory": str(args.bag_dir.resolve()),
        "conversion_metadata": str(args.conversion_metadata.resolve()),
        "topic": args.topic,
        "records": records,
    }
    (args.output_dir / "recorded_rgb_frames.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
