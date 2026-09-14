"""Materialize verified raw pointcloud ranges from one filtered DeformPath sequence."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, cast

import numpy as np
import torch


SCHEMA = "taichidough/deformpath-range-sequence/v1"
_HEX = frozenset("0123456789abcdef")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location="cpu")


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX for c in value)


def _declared_hashes(selection: dict[str, Any]) -> dict[str, str]:
    value = selection.get("source_sha256", selection.get("source_hashes"))
    if isinstance(value, dict) and value:
        result: dict[str, str] = {}
        for name, record in value.items():
            expected = record.get("sha256") if isinstance(record, dict) else record
            if not isinstance(name, str) or Path(name).name != name or not _valid_hash(expected):
                raise ValueError("source_sha256 must map source basenames to lowercase SHA-256 values")
            result[name] = cast(str, expected)
        return result
    artifacts = selection.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("range_selection must declare source hashes")
    result = {}

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            path, expected = node.get("path"), node.get("sha256")
            if isinstance(path, str) and _valid_hash(expected):
                result[Path(path).name] = cast(str, expected)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(artifacts)
    annotation = selection.get("annotation")
    if isinstance(annotation, dict) and isinstance(annotation.get("path"), str) and _valid_hash(annotation.get("sha256")):
        result[Path(annotation["path"]).name] = annotation["sha256"]
    if not result:
        raise ValueError("range_selection artifacts do not declare source hashes")
    return result


def _verify_declared_hashes(selection: dict[str, Any], sources: dict[str, Path]) -> dict[str, str]:
    declared = _declared_hashes(selection)
    required = set(sources)
    missing = required - set(declared)
    if missing:
        raise ValueError(f"Source hashes are missing {sorted(missing)}")
    actual = {name: _sha256(path) for name, path in sources.items()}
    stale = [name for name in sorted(required) if declared[name] != actual[name]]
    if stale:
        raise ValueError(f"Stale source file hash for {', '.join(stale)}")
    return actual


def _source_fingerprint(annotation: dict[str, Any]) -> str:
    source = annotation.get("source")
    if not isinstance(source, dict):
        raise ValueError("Annotation source must be an object")
    fingerprint = source.get("fingerprint")
    if isinstance(fingerprint, dict):
        fingerprint = fingerprint.get("value")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Annotation source fingerprint is missing")
    return fingerprint


def _selection_identity(selection: dict[str, Any]) -> tuple[str, str, str]:
    if selection.get("schema") not in (None, "taichidough/deformpath-range-selection/v1"):
        raise ValueError("Unsupported range_selection schema")
    raw_source = selection.get("source")
    source: dict[str, Any] = raw_source if isinstance(raw_source, dict) else {}
    split_group = selection.get("split_group", source.get("split_group"))
    recording_id = selection.get("source_recording_id", source.get("recording_id"))
    fingerprint = selection.get("source_fingerprint", source.get("fingerprint"))
    if not isinstance(split_group, str) or not split_group:
        raise ValueError("range_selection split_group is missing")
    if not isinstance(recording_id, str) or not recording_id:
        raise ValueError("range_selection source recording_id is missing")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("range_selection source fingerprint is missing")
    return split_group, recording_id, fingerprint


def _annotation_boundaries(annotation: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = annotation.get("boundaries")
    if not isinstance(rows, list):
        raise ValueError("Annotation boundaries must be a list")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("pointcloud_ordinal"), int):
            raise ValueError("Every annotation boundary needs an integer pointcloud_ordinal")
        ordinal = row["pointcloud_ordinal"]
        if ordinal in result:
            raise ValueError(f"Duplicate annotation boundary ordinal {ordinal}")
        result[ordinal] = row
    return result


def _stamp(value: Any, label: str, *, nullable: bool = False) -> int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{label} must be an integer nanosecond stamp")
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer nanosecond stamp") from exc
    if str(result) != str(value):
        raise ValueError(f"{label} must use canonical integer text")
    return result


def _ranges(selection: dict[str, Any], annotation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = selection.get("ranges", selection.get("retained_ranges"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("range_selection must contain nonempty ranges")
    boundaries = _annotation_boundaries(annotation)
    normalized = []
    names = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every retained range must be an object")
        name = row.get("output_name")
        start = row.get("start_pointcloud_ordinal", row.get("start_raw_pointcloud_ordinal"))
        end = row.get("end_pointcloud_ordinal_exclusive", row.get("end_raw_pointcloud_ordinal_exclusive"))
        start_stamp = row.get("start_pointcloud_header_stamp_ns", row.get("start_header_stamp_ns"))
        end_stamp = row.get("end_pointcloud_header_stamp_ns_exclusive", row.get("end_header_stamp_ns_exclusive"))
        if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Each output_name must be a unique safe directory name")
        if name in names:
            raise ValueError(f"Duplicate output_name {name}")
        names.add(name)
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            raise ValueError(f"Range {name} must use a nonempty half-open ordinal interval")
        start_stamp = _stamp(start_stamp, f"{name} start stamp")
        end_stamp = _stamp(end_stamp, f"{name} end stamp", nullable=True)
        if start not in boundaries or _stamp(boundaries[start].get("pointcloud_header_stamp_ns"), "annotation start stamp") != start_stamp:
            raise ValueError(f"Range {name} start does not match an accepted annotation boundary")
        if end in boundaries:
            annotation_end = _stamp(boundaries[end].get("pointcloud_header_stamp_ns"), "annotation end stamp", nullable=True)
            if annotation_end != end_stamp:
                raise ValueError(f"Range {name} end does not match an accepted annotation boundary")
        elif end_stamp is not None:
            raise ValueError(f"Range {name} end ordinal is not an accepted annotation boundary")
        normalized.append({**row, "output_name": name, "start": start, "end": end,
                           "start_stamp": start_stamp, "end_stamp": end_stamp})
    ordered = sorted(normalized, key=lambda row: (row["start"], row["end"]))
    for left, right in zip(ordered, ordered[1:]):
        if right["start"] < left["end"]:
            raise ValueError(f"Ranges {left['output_name']} and {right['output_name']} overlap")
    return normalized


def _validate_payloads(cloud_payload: Any, path_payload: Any) -> tuple[list[Any], dict[str, Any]]:
    if not (isinstance(cloud_payload, list) and len(cloud_payload) == 1 and isinstance(cloud_payload[0], list)):
        raise ValueError("pointcloud payload must be a list containing one episode list")
    if not (isinstance(path_payload, list) and len(path_payload) == 1 and isinstance(path_payload[0], dict)):
        raise ValueError("paths payload must be a list containing one dictionary")
    clouds, paths = cloud_payload[0], path_payload[0]
    required = {"path", "stream_validity", "pose_frames"}
    if not required <= set(paths):
        raise ValueError("Path dictionary is missing path, stream_validity, or pose_frames")
    path = paths["path"]
    validity = paths["stream_validity"]
    names = paths["pose_frames"]
    if not isinstance(path, torch.Tensor) or tuple(path.shape) != (len(clouds), 2, 14):
        raise ValueError("path must be a tensor with layout [T,2,14]")
    if not isinstance(validity, torch.Tensor) or tuple(validity.shape) != (len(clouds), 2):
        raise ValueError("stream_validity must be a tensor with layout [T,2]")
    if not isinstance(names, (list, tuple)) or len(names) != 2 or len(set(names)) != 2:
        raise ValueError("pose_frames must name two distinct tool streams")
    for cloud in clouds:
        if not isinstance(cloud, torch.Tensor) or cloud.ndim != 2 or cloud.shape[1] < 3:
            raise ValueError("Every pointcloud must be a tensor [N,>=3]")
    return clouds, paths


def _validate_selected(clouds: list[Any], paths: dict[str, Any], positions: list[int], max_gap_s: float) -> None:
    if len(positions) < 3:
        raise ValueError("Every retained range must contain at least 3 filtered frames")
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be finite and positive")
    for position in positions:
        if not clouds[position][:, :3].isfinite().all().item():
            raise ValueError("Retained pointcloud XYZ values must be finite")
    selected_path = paths["path"][positions]
    if not selected_path.isfinite().all().item():
        raise ValueError("Retained path values must be finite")
    if not paths["stream_validity"][positions].bool().all().item():
        raise ValueError("Both tool streams must be valid on every retained row")
    timestamps = selected_path[:, :, 13].detach().cpu().numpy().astype(np.float64)
    if not np.allclose(timestamps[:, 0], timestamps[:, 1], rtol=0, atol=1e-6):
        raise ValueError("Tool streams must share timestamps")
    gaps = np.diff(timestamps[:, 0])
    if np.any(gaps <= 0):
        raise ValueError("Retained path timestamps must be strictly increasing")
    if np.any(gaps > max_gap_s):
        raise ValueError(f"Retained path timestamp gap exceeds {max_gap_s} seconds")


def _boundary_mappings(annotation: dict[str, Any], raw_ordinals: list[int],
                       range_start: int, range_end: int, *, manual_only: bool) -> list[dict[str, Any]]:
    result = []
    for boundary in annotation.get("boundaries", []):
        if manual_only and boundary.get("kind") != "manual":
            continue
        ordinal = boundary["pointcloud_ordinal"]
        if ordinal < range_start or ordinal > range_end:
            continue
        common = {"boundary_id": boundary.get("id"), "kind": boundary.get("kind"),
                  "raw_pointcloud_ordinal": ordinal,
                  "pointcloud_header_stamp_ns": boundary.get("pointcloud_header_stamp_ns")}
        if ordinal == range_end:
            result.append({**common, "output_frame_exclusive": len(raw_ordinals),
                           "mapping": "range_end_exclusive"})
            continue
        output_frame = raw_ordinals.index(ordinal) if ordinal in raw_ordinals else next(
            (i for i, value in enumerate(raw_ordinals) if value >= ordinal), None)
        result.append({**common, "output_frame": output_frame,
                       "mapping": "exact" if ordinal in raw_ordinals else
                                  ("first_retained_at_or_after" if output_frame is not None else None)})
    return result


def _chunk_parent_metadata(parent: dict[str, Any], positions: list[int], frame_count: int,
                           times: np.ndarray, destination: Path) -> dict[str, Any]:
    result = {key: copy.deepcopy(value) for key, value in parent.items()
              if key not in {"pointcloud_indices", "sequence_fingerprint"}}
    parent_count = len(parent.get("pointcloud_indices", []))
    for key, value in list(result.items()):
        if key == "num_pose_frames":
            continue
        if (key.startswith("num_") and isinstance(value, int)) or key in {"frame_count", "pointcloud_count"}:
            result[key] = frame_count
    if isinstance(result.get("per_stream_valid_counts"), dict):
        result["per_stream_valid_counts"] = {
            key: frame_count for key in result["per_stream_valid_counts"]}

    def slice_frame_stats(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if key == "frame_stats" and isinstance(value, list):
                    if len(value) != parent_count:
                        raise ValueError("Parent frame_stats count differs from pointcloud_indices")
                    node[key] = [copy.deepcopy(value[position]) for position in positions]
                else:
                    slice_frame_stats(value)
        elif isinstance(node, list):
            for value in node:
                slice_frame_stats(value)

    slice_frame_stats(result)
    for key in ("sample_time_range_s", "valid_sample_time_range_s", "path_time_range_s"):
        if key in result:
            result[key] = [float(times[0]), float(times[-1])]
    if frame_count > 1:
        median_dt = float(np.median(np.diff(times)))
        if "sample_dt_sec" in result:
            result["sample_dt_sec"] = median_dt
        if "sample_hz" in result:
            result["sample_hz"] = float(1.0 / median_dt)
    result["input_dir"] = str(parent.get("output_dir", parent.get("input_dir", "")))
    result["output_dir"] = str(destination)
    replacements = {
        "output_pointclouds_path": destination / "pointclouds_interpolated.pt",
        "output_paths_path": destination / "paths_interpolated.pt",
    }
    for key, path in replacements.items():
        if key in result:
            result[key] = str(path)
    return result


def _save_chunk(temp_dir: Path, clouds: list[Any], paths: dict[str, Any], positions: list[int], metadata: dict[str, Any]) -> str:
    selected_clouds = [clouds[position] for position in positions]
    selected_paths = copy.copy(paths)
    selected_paths["path"] = paths["path"][positions]
    selected_paths["stream_validity"] = paths["stream_validity"][positions]
    points_path = temp_dir / "pointclouds_interpolated.pt"
    paths_path = temp_dir / "paths_interpolated.pt"
    torch.save([selected_clouds], points_path)
    torch.save([selected_paths], paths_path)
    fingerprint = hashlib.sha256((_sha256(points_path) + _sha256(paths_path)).encode()).hexdigest()
    metadata["sequence_fingerprint"] = fingerprint
    (temp_dir / "sequence_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return fingerprint


def materialize_ranges(full_episode_dir: Path, conversion_metadata_path: Path, range_selection_path: Path,
                       annotation_path: Path, output_root: Path, *, max_gap_s: float = 0.1) -> list[dict[str, Any]]:
    full_episode_dir = Path(full_episode_dir).resolve()
    conversion_metadata_path = Path(conversion_metadata_path).resolve()
    range_selection_path = Path(range_selection_path).resolve()
    annotation_path = Path(annotation_path).resolve()
    output_root = Path(output_root).resolve()
    points_path = full_episode_dir / "pointclouds_interpolated.pt"
    paths_path = full_episode_dir / "paths_interpolated.pt"
    metadata_path = full_episode_dir / "sequence_metadata.json"
    sources = {path.name: path for path in (points_path, paths_path, metadata_path, conversion_metadata_path, annotation_path)}
    if len(sources) != 5 or any(not path.is_file() for path in sources.values()):
        raise ValueError("All five source files must exist and have distinct basenames")
    selection = _json(range_selection_path)
    annotation = _json(annotation_path)
    if annotation.get("schema_name") != "deformpath.temporal_annotations" or annotation.get("schema_version") != 1:
        raise ValueError("Unsupported temporal annotation schema")
    if annotation.get("acceptance_status", annotation.get("status", "accepted")) != "accepted":
        raise ValueError("Temporal annotation is not accepted")
    split_group, recording_id, source_fingerprint = _selection_identity(selection)
    if annotation.get("split_group") != split_group or _source_fingerprint(annotation) != source_fingerprint:
        raise ValueError("Annotation split_group/source identity differs from range_selection")
    annotation_record = selection.get("annotation", {})
    if isinstance(annotation_record, dict) and annotation_record.get("sha256") not in (None, _sha256(annotation_path)):
        raise ValueError("range_selection declares a stale annotation hash")
    source_hashes = _verify_declared_hashes(selection, sources)
    ranges = _ranges(selection, annotation)
    parent_metadata = _json(metadata_path)
    conversion = _json(conversion_metadata_path)
    indices = parent_metadata.get("pointcloud_indices")
    stamps = (conversion.get("episode") or {}).get("timestamps_ns")
    if not isinstance(indices, list) or any(isinstance(i, bool) or not isinstance(i, int) for i in indices):
        raise ValueError("sequence_metadata pointcloud_indices must be integer exporter indices")
    if not isinstance(stamps, list):
        raise ValueError("conversion_metadata episode.timestamps_ns must be a list")
    exporter_stamps = [_stamp(value, "exporter timestamp") for value in stamps]
    if len(set(exporter_stamps)) != len(exporter_stamps):
        raise ValueError("conversion_metadata contains duplicate header stamps")
    annotation_rows = annotation.get("source", {}).get("topics", {}).get("pointcloud", {})
    expected_count = annotation_rows.get("message_count")
    if not isinstance(expected_count, int) or expected_count <= 0:
        raise ValueError("Annotation pointcloud message_count is invalid")
    if len(exporter_stamps) != expected_count:
        raise ValueError("Full-export timestamps_ns count differs from the annotated raw pointcloud count")
    # A full export preserves raw pointcloud order. Build the complete mapping from each exact header
    # stamp to its raw ordinal, then verify every accepted mark against it.
    raw_stamp_to_ordinal = {stamp: ordinal for ordinal, stamp in enumerate(exporter_stamps)}
    boundary_stamps = {row.get("pointcloud_header_stamp_ns"): row["pointcloud_ordinal"] for row in annotation.get("boundaries", [])
                       if row.get("pointcloud_header_stamp_ns") is not None}
    # An explicit mapping is optional, but when present it must agree with the full export.
    raw_rows = selection.get("raw_pointclouds", selection.get("raw_timestamp_mapping", []))
    for row in raw_rows:
        if not isinstance(row, dict) or not isinstance(row.get("pointcloud_ordinal", row.get("ordinal")), int):
            raise ValueError("raw_pointclouds entries need ordinal and header stamp")
        ordinal = row.get("pointcloud_ordinal", row.get("ordinal"))
        stamp = _stamp(row.get("pointcloud_header_stamp_ns", row.get("header_stamp_ns")), "raw mapping stamp")
        if raw_stamp_to_ordinal.get(stamp) != ordinal:
            raise ValueError("Explicit raw header stamp/ordinal mapping differs from the full export")
    for stamp_text, ordinal in boundary_stamps.items():
        stamp = _stamp(stamp_text, "boundary stamp")
        previous = raw_stamp_to_ordinal.get(stamp)
        if previous is not None and previous != ordinal:
            raise ValueError("Range and annotation raw timestamp mappings disagree")
        raw_stamp_to_ordinal[stamp] = ordinal
    marks = selection.get("ranges", selection.get("retained_ranges", []))
    for row in marks:
        for ordinal_key, stamp_key in (("start_pointcloud_ordinal", "start_pointcloud_header_stamp_ns"),
                                       ("end_pointcloud_ordinal_exclusive", "end_pointcloud_header_stamp_ns_exclusive")):
            if row.get(stamp_key) is not None:
                ordinal, stamp = row.get(ordinal_key), _stamp(row[stamp_key], "range mark stamp")
                previous = raw_stamp_to_ordinal.get(stamp)
                if not isinstance(ordinal, int) or (previous is not None and previous != ordinal):
                    raise ValueError("Range mark ordinal/header stamp mapping is invalid")
                raw_stamp_to_ordinal[stamp] = ordinal
    # A complete exact map is mandatory; accepting an inferred fixed offset would corrupt filtered sequences.
    filtered_raw = []
    for exporter_index in indices:
        if exporter_index < 0 or exporter_index >= len(exporter_stamps):
            raise ValueError("pointcloud_indices references an exporter index outside timestamps_ns")
        stamp = exporter_stamps[exporter_index]
        if stamp not in raw_stamp_to_ordinal:
            raise ValueError(f"Missing exact raw ordinal mapping for header stamp {stamp}")
        filtered_raw.append(raw_stamp_to_ordinal[stamp])
    if len(set(filtered_raw)) != len(filtered_raw):
        raise ValueError("Filtered frames map to duplicate raw pointcloud ordinals")
    cloud_payload = _torch_load(points_path)
    path_payload = _torch_load(paths_path)
    clouds, paths = _validate_payloads(cloud_payload, path_payload)
    if len(indices) != len(clouds):
        raise ValueError("pointcloud_indices count differs from tensor frame count")
    if any((output_root / row["output_name"]).exists() for row in ranges):
        raise FileExistsError("Refusing to overwrite an existing output directory")
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    created = []
    try:
        for row in ranges:
            positions = [i for i, ordinal in enumerate(filtered_raw) if row["start"] <= ordinal < row["end"]]
            _validate_selected(clouds, paths, positions, max_gap_s)
            raw_ordinals = [filtered_raw[i] for i in positions]
            exporter_indices = [indices[i] for i in positions]
            header_stamps = [exporter_stamps[i] for i in exporter_indices]
            destination = output_root / row["output_name"]
            selected_times = paths["path"][positions, 0, 13].detach().cpu().numpy().astype(np.float64)
            useful_parent = _chunk_parent_metadata(parent_metadata, positions, len(positions),
                                                   selected_times, destination)
            metadata = useful_parent | {
                "schema": SCHEMA, "schema_version": 1, "source_recording_id": recording_id,
                "split_group": split_group, "output_name": row["output_name"],
                "raw_range": {"start_pointcloud_ordinal": row["start"],
                              "end_pointcloud_ordinal_exclusive": row["end"]},
                "boundary_header_stamps_ns": {"start": str(row["start_stamp"]),
                                               "end_exclusive": None if row["end_stamp"] is None else str(row["end_stamp"])},
                "first_retained_header_stamp_ns": str(header_stamps[0]),
                "last_retained_header_stamp_ns": str(header_stamps[-1]),
                "source_processed_positions": positions, "pointcloud_indices": exporter_indices,
                "original_exporter_indices": exporter_indices, "raw_pointcloud_ordinals": raw_ordinals,
                "frame_count": len(positions), "parent": {
                    "episode_dir": str(full_episode_dir), "source_sha256": source_hashes,
                    "sequence_fingerprint": hashlib.sha256((source_hashes[points_path.name] + source_hashes[paths_path.name]).encode()).hexdigest(),
                    "range_selection_sha256": _sha256(range_selection_path)},
                "temporal_policy": {"concatenated": False, "timestamps_rebased": False,
                                    "initial_state": "reset_from_first_retained_frame", "max_gap_s": max_gap_s},
                "accepted_boundary_mappings": _boundary_mappings(
                    annotation, raw_ordinals, row["start"], row["end"], manual_only=False),
                "accepted_manual_mark_mappings": _boundary_mappings(
                    annotation, raw_ordinals, row["start"], row["end"], manual_only=True),
            }
            temp = Path(tempfile.mkdtemp(prefix=f".{row['output_name']}.", dir=output_root))
            try:
                fingerprint = _save_chunk(temp, clouds, paths, positions, metadata)
                # Reload through the byte-preserved production loader before making the directory visible.
                from experiments.differentiable_mpm.reference_adapter import get_reference_modules
                try:
                    loaded = get_reference_modules().dynamics.load_observation_sequence(temp)
                    validation_loader = "verified_reference"
                except ValueError as exc:
                    if "Working source differs from preserved reference" not in str(exc):
                        raise
                    # The repository may deliberately use frozen-reference policy elsewhere. The output
                    # format is still checked here without changing that process-wide policy.
                    check_clouds, check_paths = _validate_payloads(_torch_load(temp / "pointclouds_interpolated.pt"),
                                                                   _torch_load(temp / "paths_interpolated.pt"))
                    if len(check_clouds) != len(positions) or len(check_paths["path"]) != len(positions):
                        raise ValueError("Reloaded chunk frame count differs from the written selection")
                    loaded = None
                    validation_loader = "local_tensor_layout_fallback_due_to_strict_reference_drift"
                if loaded is not None and (loaded.fingerprint != fingerprint or len(loaded.points) != len(positions)):
                    raise ValueError("Reference loader rejected or changed the written chunk identity")
                metadata["output_validation_loader"] = validation_loader
                (temp / "sequence_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                destination = output_root / row["output_name"]
                os.rename(temp, destination)
                created.append(destination)
                results.append({"output_name": row["output_name"], "path": str(destination),
                                "frame_count": len(positions), "sequence_fingerprint": fingerprint})
            except BaseException:
                shutil.rmtree(temp, ignore_errors=True)
                raise
    except BaseException:
        for path in created:
            shutil.rmtree(path, ignore_errors=True)
        raise
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-episode-dir", type=Path, required=True)
    parser.add_argument("--conversion-metadata", type=Path, required=True)
    parser.add_argument("--range-selection", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-gap-s", type=float, default=0.1)
    args = parser.parse_args(argv)
    results = materialize_ranges(args.full_episode_dir, args.conversion_metadata, args.range_selection,
                                 args.annotation, args.output_root, max_gap_s=args.max_gap_s)
    print(json.dumps({"outputs": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
