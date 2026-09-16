#!/usr/bin/env python3
"""Split an aligned DeformPath episode into frame-addressable chunks.

Chunking is deliberately performed on the interpolated sequence.  Each output
chunk is a small, self-contained observation episode with local frame indices
and timestamps rebased to zero, while ``chunk_manifest.json`` retains the
mapping back to the canonical global sequence.

This tool does not reconstruct particles or create optimizer configs.  Those
steps belong after chunking and can choose either independent initialization
per chunk (optimization) or state carry-over (continuous video replay).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def unwrap_clouds(value: Any) -> list[Any]:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Pointcloud archive must contain a nonempty list of frames")
    return list(value)


def unwrap_paths(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict) or not {"path", "stream_validity", "pose_frames"} <= set(value):
        raise ValueError("Path archive must contain path, stream_validity, and pose_frames")
    return value


def tensor_length(value: Any, name: str) -> int:
    try:
        length = len(value)
    except TypeError as error:
        raise ValueError(f"{name} must be sliceable along its first dimension") from error
    if length < 1:
        raise ValueError(f"{name} must be nonempty")
    return int(length)


def sliced(value: Any, start: int, end: int, name: str) -> Any:
    try:
        result = value[start:end].clone()
    except AttributeError:
        result = value[start:end]
    if len(result) != end - start:
        raise ValueError(f"Could not slice {name} consistently")
    return result


def copy_static_files(input_dir: Path, chunk_dir: Path) -> list[str]:
    names = (
        "camera_info.json",
        "scene_calibration_v2.json",
        "floor_estimate.json",
    )
    copied = []
    for name in names:
        source = input_dir / name
        if source.is_file():
            shutil.copy2(source, chunk_dir / name)
            copied.append(name)
    return copied


def update_metadata(source: dict[str, Any], *, input_dir: Path, chunk_dir: Path,
                    start: int, end: int, total: int, global_indices: list[int],
                    local_times: list[float], frame_stats: list[Any] | None,
                    raw_start: int | None, raw_end: int | None) -> dict[str, Any]:
    metadata = dict(source)
    metadata.update({
        "input_dir": str(input_dir),
        "output_dir": str(chunk_dir),
        "pointclouds_path": str(chunk_dir / "pointclouds_interpolated.pt"),
        "paths_path": str(chunk_dir / "paths_interpolated.pt"),
        "output_pointclouds_path": str(chunk_dir / "pointclouds_interpolated.pt"),
        "output_paths_path": str(chunk_dir / "paths_interpolated.pt"),
        "num_valid_samples": end - start,
        "num_timeline_samples": end - start,
        "sample_time_range_s": [local_times[0], local_times[-1]],
        "valid_sample_time_range_s": [local_times[0], local_times[-1]],
        "pointcloud_indices": global_indices,
        "chunking": {
            "schema": "taichidough/deformpath-frame-chunk/v1",
            "global_start_frame": start,
            "global_end_frame_exclusive": end,
            "global_total_frames": total,
            "local_start_frame": 0,
            "local_end_frame": end - start - 1,
            "timestamp_origin_global_s": float(source.get("sample_time_range_s", [0.0])[0]) +
                                         float(local_times[0]),
        },
    })
    if raw_start is not None and raw_end is not None:
        metadata["chunking"].update({
            "raw_start_frame": raw_start,
            "raw_end_frame_exclusive": raw_end,
            "raw_frame_capacity": raw_end - raw_start,
        })
    if frame_stats is not None:
        metadata.setdefault("pointcloud_filter", {})["frame_stats"] = frame_stats[start:end]
    return metadata


def parse_ranges(path: Path, original_indices: list[int], frame_count: int) -> list[tuple[int, int, int | None, int | None]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    raw_ranges = document.get("ranges") if isinstance(document, dict) else document
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise ValueError("Ranges file must contain a nonempty list or an object with a ranges list")
    result = []
    previous_end = None
    for index, item in enumerate(raw_ranges, 1):
        if isinstance(item, dict):
            start = item.get("start", item.get("start_frame"))
            end = item.get("end", item.get("end_frame_exclusive", item.get("end_frame")))
        elif isinstance(item, list) and len(item) == 2:
            start, end = item
        else:
            raise ValueError(f"Range {index} must be [start,end) or an object with start/end")
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"Range {index} bounds must be integers")
        if start < 0 or end <= start or (previous_end is not None and start != previous_end):
            raise ValueError(f"Ranges must be contiguous half-open intervals; invalid range {index}: [{start},{end})")
        selected = [local for local, original in enumerate(original_indices) if start <= original < end]
        if len(selected) < 2:
            raise ValueError(f"Raw range {index} [{start},{end}) contains fewer than two aligned frames")
        aligned_start, aligned_end = selected[0], selected[-1] + 1
        if selected != list(range(aligned_start, aligned_end)):
            raise ValueError(f"Raw range {index} does not select a contiguous aligned frame interval")
        result.append((aligned_start, aligned_end, start, end))
        previous_end = end
    selected_all = [local for start, end, _, _ in result for local in range(start, end)]
    if selected_all != list(range(frame_count)):
        missing = sorted(set(range(frame_count)) - set(selected_all))
        overlap = len(selected_all) - len(set(selected_all))
        raise ValueError(f"Ranges must cover every aligned frame exactly once; missing={missing[:5]}, overlap={overlap}")
    return result


def chunk_episode(input_dir: Path, output_dir: Path, chunk_size: int | None,
                  ranges_path: Path | None, overwrite: bool) -> Path:
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise RuntimeError("Chunking requires numpy and torch") from error

    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    points_path = input_dir / "pointclouds_interpolated.pt"
    paths_path = input_dir / "paths_interpolated.pt"
    metadata_path = input_dir / "sequence_metadata.json"
    for path, name in ((points_path, "interpolated pointclouds"), (paths_path, "interpolated paths"),
                       (metadata_path, "sequence metadata")):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    if (chunk_size is None) == (ranges_path is None):
        raise ValueError("Choose exactly one of chunk-size or ranges-file")
    if chunk_size is not None and (not isinstance(chunk_size, int) or chunk_size < 2):
        raise ValueError("chunk-size must be an integer >= 2")
    if output_dir == input_dir:
        raise ValueError("Output directory must differ from input directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is non-empty; use --overwrite: {output_dir}")
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    clouds = unwrap_clouds(torch.load(points_path, map_location="cpu", weights_only=False))
    path_record = unwrap_paths(torch.load(paths_path, map_location="cpu", weights_only=False))
    path_tensor = path_record["path"]
    validity = path_record["stream_validity"]
    frame_count = len(clouds)
    if tensor_length(path_tensor, "path") != frame_count or tensor_length(validity, "stream_validity") != frame_count:
        raise ValueError("Pointcloud and path frame counts do not match")
    path_array = np.asarray(path_tensor, dtype=np.float64)
    if path_array.ndim != 3 or path_array.shape[2] < 14:
        raise ValueError("Expected path tensor with shape [frames, tools, >=14]")
    if not np.isfinite(path_array[:, :, 13]).all():
        raise ValueError("Path timestamps must be finite")
    if not np.allclose(path_array[:, :, 13], path_array[:, :1, 13], atol=1e-6, rtol=0):
        raise ValueError("Tool streams must share timestamps before chunking")
    if np.any(np.diff(path_array[:, 0, 13]) <= 0):
        raise ValueError("Path timestamps must be strictly increasing before chunking")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    original_indices = metadata.get("pointcloud_indices", list(range(frame_count)))
    if len(original_indices) != frame_count:
        raise ValueError("sequence_metadata.pointcloud_indices does not match frame count")
    original_indices = [int(value) for value in original_indices]
    frame_stats = metadata.get("pointcloud_filter", {}).get("frame_stats")
    if frame_stats is not None and len(frame_stats) != frame_count:
        frame_stats = None

    if ranges_path is not None:
        ranges = parse_ranges(ranges_path.expanduser().resolve(), original_indices, frame_count)
    else:
        assert chunk_size is not None
        ranges = [(start, min(start + chunk_size, frame_count), None, None)
                  for start in range(0, frame_count, chunk_size)]
        if len(ranges) == 1 and ranges[0][1] - ranges[0][0] < 2:
            raise ValueError("Sequence is too short for the requested chunk-size")
        if len(ranges) > 1 and ranges[-1][1] - ranges[-1][0] < 2:
            previous_start, _ = ranges[-2][:2]
            ranges[-2] = (previous_start, frame_count, None, None)
            ranges.pop()

    rows = []
    for chunk_number, (start, end, raw_start, raw_end) in enumerate(ranges, 1):
        chunk_dir = output_dir / f"chunk{chunk_number:02d}"

        local_times = path_array[start:end, 0, 13] - path_array[start, 0, 13]
        chunk_clouds = clouds[start:end]
        chunk_paths = dict(path_record)
        chunk_paths["path"] = sliced(path_tensor, start, end, "path").clone()
        chunk_paths["path"][:, :, 13] -= path_tensor[start, :, 13].reshape(1, -1)[0, 0]
        chunk_paths["stream_validity"] = sliced(validity, start, end, "stream_validity")
        chunk_dir.mkdir(parents=True, exist_ok=True)
        torch.save([list(chunk_clouds)], chunk_dir / "pointclouds_interpolated.pt")
        torch.save([chunk_paths], chunk_dir / "paths_interpolated.pt")
        write_json(chunk_dir / "sequence_metadata.json", update_metadata(
            metadata, input_dir=input_dir, chunk_dir=chunk_dir, start=start, end=end,
            total=frame_count, global_indices=original_indices[start:end],
            local_times=local_times.tolist(), frame_stats=frame_stats,
            raw_start=raw_start, raw_end=raw_end))
        copied = copy_static_files(input_dir, chunk_dir)
        row = {
            "id": chunk_dir.name,
            "directory": chunk_dir.name,
            "global_start_frame": start,
            "global_end_frame_exclusive": end,
            "global_frame_count": end - start,
            "global_original_indices": original_indices[start:end],
            "local_start_frame": 0,
            "local_end_frame": end - start - 1,
            "timestamp_start_s": float(path_array[start, 0, 13]),
            "timestamp_end_s": float(path_array[end - 1, 0, 13]),
            "duration_s": float(local_times[-1]),
            "static_files": copied,
            "pointclouds": str((chunk_dir / "pointclouds_interpolated.pt").resolve()),
            "paths": str((chunk_dir / "paths_interpolated.pt").resolve()),
        }
        if raw_start is not None and raw_end is not None:
            row.update({"raw_start_frame": raw_start, "raw_end_frame_exclusive": raw_end,
                        "raw_frame_capacity": raw_end - raw_start})
        rows.append(row)

    manifest = {
        "schema": "taichidough/deformpath-frame-chunks/v1",
        "source_episode": str(input_dir),
        "source_files": {
            name: {"path": str((input_dir / name).resolve()), "sha256": sha256(input_dir / name)}
            for name in ("pointclouds_interpolated.pt", "paths_interpolated.pt", "sequence_metadata.json")
        },
        "frame_count": frame_count,
        "chunk_size": chunk_size,
        "ranges_file": str(ranges_path.expanduser().resolve()) if ranges_path is not None else None,
        "chunk_count": len(rows),
        "timestamp_policy": "local chunk timestamps rebased to zero; source timestamps retained in manifest",
        "state_policy": {
            "optimization": "reconstruct or provide an initial state per chunk",
            "continuous_video": "carry final simulator particle state into the next chunk",
        },
        "chunks": rows,
    }
    manifest_path = output_dir / "chunk_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Canonical directory containing interpolated .pt files")
    parser.add_argument("--output-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--chunk-size", type=int,
                           help="Maximum number of aligned frames per chunk")
    selection.add_argument("--ranges-file", type=Path,
                           help="JSON file containing contiguous raw half-open ranges, e.g. [[0,83],[83,102]]")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = chunk_episode(args.input_dir, args.output_dir, args.chunk_size, args.ranges_file, args.overwrite)
    print(f"Wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
