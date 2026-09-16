"""Build the immutable mocap-coordinate Episode20 temporal package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import torch

from experiments.differentiable_mpm.materialize_deformpath_ranges import materialize_ranges
from experiments.differentiable_mpm.reference_adapter import reference_policy

SCHEMA = "taichidough/episode20-kugla-temporal/v2"
EXPERIMENT = Path(__file__).resolve().parent
WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = WORKSPACE / "data/deformpath_training/DeformPath2/snimanje_23_10/episode20_kugla/temporal_annotations.json"
DEFAULT_FULL_EPISODE = WORKSPACE / "data/deformpath_training/DeformPath3/snimanje_23_10/episode20_kugla"
DEFAULT_OUTPUT = EXPERIMENT / "data/episode20_kugla_temporal_v2"
DEFAULT_PRESERVED_V1 = EXPERIMENT / "data/episode20_kugla_temporal_v1"
INHERITED_BUNDLE = EXPERIMENT / "data/episode18_registered_tools_v1"
INHERITED_FILES = (
    "tool_geometry.json",
    "collision_manifest.json",
    "tool_0_registration.json",
    "tool_1_registration.json",
    "ur_spathla_collision_solid.stl",
    "gen3_spathla_collision_solid.stl",
)
EXPECTED_FINGERPRINT = "e33294d22879a63805703a432548f5cf2e971a22f7b5a984ff52a30f8734fc85"
EXPECTED_INTERVALS = (
    (0, 7),
    (7, 25),
    (25, 43),
    (43, 61),
    (61, 78),
    (78, 95),
    (95, 113),
    (113, 131),
    (131, 148),
    (148, 166),
    (166, 184),
    (184, 250),
)
RETAINED_INTERVALS = EXPECTED_INTERVALS[1:-1]
EXPECTED_FRAME_COUNTS = (18, 18, 18, 17, 17, 18, 18, 17, 18, 18)
POSE_FRAMES = ("UR5e_spathla", "gen3_spathla")
REFERENCE_VOLUME_M3 = 0.000113832
NOMINAL_DENSITY_KG_M3 = 1200.0
CONSTANT_MASS_KG = REFERENCE_VOLUME_M3 * NOMINAL_DENSITY_KG_M3


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def tree_sha256(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(f"Preserved package does not exist: {root}")
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def load_annotation(source: Path) -> tuple[bytes, dict[str, Any]]:
    raw = source.read_bytes()
    annotation = json.loads(raw)
    if (
        annotation.get("schema_name") != "deformpath.temporal_annotations"
        or annotation.get("schema_version") != 1
    ):
        raise ValueError("Unsupported temporal annotation schema")
    if annotation.get("annotation_revision") != 12:
        raise ValueError("Episode20 annotation revision must be 12")
    fingerprint = annotation.get("source", {}).get("fingerprint", {}).get("value")
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError("Episode20 annotation fingerprint differs from the accepted source")
    boundaries = annotation.get("boundaries")
    segments = annotation.get("segments")
    if not isinstance(boundaries, list) or len(boundaries) != 13:
        raise ValueError("Episode20 annotation must contain 13 boundaries")
    if not isinstance(segments, list) or len(segments) != 12:
        raise ValueError("Episode20 annotation must contain 12 segments")
    intervals = tuple(
        (row.get("start_pointcloud_ordinal"), row.get("end_pointcloud_ordinal_exclusive"))
        for row in segments
    )
    if intervals != EXPECTED_INTERVALS:
        raise ValueError("Episode20 annotation intervals differ from the accepted split")
    if annotation.get("source", {}).get("topics", {}).get("pointcloud", {}).get("message_count") != 250:
        raise ValueError("Episode20 annotation must describe 250 point-cloud messages")
    return raw, annotation


def require_mocap_inputs(full_episode: Path) -> dict[str, Any]:
    paths = {
        "pointclouds_interpolated.pt": full_episode / "pointclouds_interpolated.pt",
        "paths_interpolated.pt": full_episode / "paths_interpolated.pt",
        "sequence_metadata.json": full_episode / "sequence_metadata.json",
        "conversion_metadata.json": full_episode / "conversion_metadata.json",
        "scene_calibration_v2.json": full_episode / "scene_calibration_v2.json",
        "episode20_table_plane_validation.json": full_episode / "episode20_table_plane_validation.json",
        "episode20_export_validation.json": full_episode / "episode20_export_validation.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Episode20 mocap inputs: " + ", ".join(missing))

    conversion = load_json(paths["conversion_metadata.json"])
    sequence = load_json(paths["sequence_metadata.json"])
    calibration = load_json(paths["scene_calibration_v2.json"])
    table = load_json(paths["episode20_table_plane_validation.json"])
    export_validation = load_json(paths["episode20_export_validation.json"])
    if conversion.get("output_frame") != "mocap":
        raise ValueError("Episode20 conversion metadata must declare mocap output")
    if tuple(conversion.get("pose_frames", ())) != POSE_FRAMES:
        raise ValueError("Episode20 conversion must contain the expected two tool streams")
    calibration_record = conversion.get("calibration")
    if not isinstance(calibration_record, dict) or calibration_record.get("status") != "available":
        raise ValueError("Episode20 conversion is missing its metric calibration")
    if sequence.get("pose_frames") != list(POSE_FRAMES):
        raise ValueError("Episode20 sequence metadata has unexpected tool streams")
    sequence_calibration = sequence.get("calibration")
    if not isinstance(sequence_calibration, dict) or sequence_calibration.get("status") != "available":
        raise ValueError("Episode20 interpolation did not propagate calibration")
    if sequence_calibration.get("source_frame") != "mocap" or sequence_calibration.get("scene_frame") != "table-aligned":
        raise ValueError("Episode20 interpolation metadata has incorrect calibration frames")
    if calibration.get("schema") != "taichidough/scene-calibration/v2":
        raise ValueError("Episode20 calibration must use scene-calibration v2")
    if calibration.get("source_frame") != "mocap" or calibration.get("scene_frame") != "table-aligned":
        raise ValueError("Episode20 calibration must map mocap inputs into table-aligned coordinates")
    transform = np.asarray(calibration.get("scene_from_source"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Episode20 scene_from_source must be a finite 4x4 table transform")
    if np.allclose(transform, np.eye(4), rtol=0, atol=1e-12):
        raise ValueError("Episode20 table alignment must retain its measured nonidentity transform")
    floor = np.asarray(calibration.get("floor_plane_scene"), dtype=np.float64)
    if floor.shape != (4,) or not np.allclose(
        floor, [0.0, 1.0, 0.0, 0.0], rtol=0, atol=1e-12
    ):
        raise ValueError("Episode20 calibration must use y=0 in the table-aligned scene")
    provenance = calibration.get("provenance")
    alignment = provenance.get("table_alignment") if isinstance(provenance, dict) else None
    if not isinstance(alignment, dict) or alignment.get("previous_scene_frame") != "mocap":
        raise ValueError("Episode20 calibration is missing mocap-to-table alignment provenance")
    if alignment.get("source_tensors_transformed") is not False:
        raise ValueError("Episode20 source tensors must remain in mocap coordinates")
    if table.get("passed") is not True:
        raise ValueError("Episode20 table-plane validation did not pass")
    if export_validation.get("passed") is not True:
        raise ValueError("Episode20 export validation did not pass")
    expected_calibration_hash = sha256_file(paths["scene_calibration_v2.json"])
    for record, label in ((calibration_record, "conversion"), (sequence_calibration, "sequence")):
        if record.get("sha256") != expected_calibration_hash:
            raise ValueError(f"{label} metadata has a stale calibration hash")
    if table.get("calibration_sha256") != expected_calibration_hash:
        raise ValueError("Table-plane report belongs to another calibration")
    return {
        "paths": paths,
        "conversion": conversion,
        "sequence": sequence,
        "calibration": calibration,
        "table": table,
        "export_validation": export_validation,
        "source_sha256": {name: sha256_file(path) for name, path in paths.items()},
    }


def build_ranges(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    segments = annotation["segments"][1:-1]
    rows = []
    for index, (segment, interval, expected_count) in enumerate(
        zip(segments, RETAINED_INTERVALS, EXPECTED_FRAME_COUNTS), start=1
    ):
        start, end = interval
        if (
            segment["start_pointcloud_ordinal"] != start
            or segment["end_pointcloud_ordinal_exclusive"] != end
        ):
            raise ValueError("Retained segment order differs from the accepted intervals")
        rows.append(
            {
                "id": f"chunk{index:02d}",
                "output_name": f"episode20_kugla_chunk{index:02d}",
                "annotation_segment_id": segment["id"],
                "start_pointcloud_ordinal": start,
                "end_pointcloud_ordinal_exclusive": end,
                "raw_pointcloud_frame_capacity": segment["pointcloud_frame_count"],
                "expected_materialized_frame_count": expected_count,
                "start_pointcloud_header_stamp_ns": segment["start_pointcloud_header_stamp_ns"],
                "end_pointcloud_header_stamp_ns_exclusive": segment[
                    "end_pointcloud_header_stamp_ns_exclusive"
                ],
                "last_included_pointcloud_header_stamp_ns": segment[
                    "last_included_pointcloud_header_stamp_ns"
                ],
                "frame_zero_semantics": "initialization_only",
                "initial_activity": "inactive",
            }
        )
    return rows


def build_documents(
    source: Path,
    annotation_bytes: bytes,
    annotation: dict[str, Any],
    full_episode: Path,
    source_info: dict[str, Any],
    inherited_hashes: dict[str, str],
    preserved_v1_hashes: dict[str, str],
) -> dict[str, Any]:
    annotation_hash = sha256_bytes(annotation_bytes)
    split_group = annotation["split_group"]
    ranges = build_ranges(annotation)
    source_hashes = dict(source_info["source_sha256"])
    source_hashes["accepted_temporal_annotations.json"] = annotation_hash
    range_selection = {
        "schema": "taichidough/deformpath-range-selection/v1",
        "name": "episode20-kugla-temporal-v2",
        "source": {
            "recording_id": "snimanje_23_10/episode20_kugla",
            "fingerprint": EXPECTED_FINGERPRINT,
            "split_group": split_group,
            "pointcloud_frame_count": 250,
            "coordinate_frame": "mocap",
        },
        "annotation": {
            "path": "accepted_temporal_annotations.json",
            "sha256": annotation_hash,
            "schema_name": annotation["schema_name"],
            "schema_version": annotation["schema_version"],
            "annotation_revision": 12,
        },
        "ranges": ranges,
        "processing": {
            "range_semantics": "half-open [start_pointcloud_ordinal, end_pointcloud_ordinal_exclusive)",
            "concatenate_ranges": False,
            "rebase_timestamps": False,
            "materialize_as_independent_episodes": True,
            "maximum_replay_gap_s": 0.1,
            "frame_zero_reserved_for_initialization": True,
            "coordinate_frame": "mocap",
        },
        "materialization_inputs": {
            "tool": "experiments/differentiable_mpm/materialize_deformpath_ranges.py",
            "full_episode_dir": str(full_episode),
            "conversion_metadata": str(full_episode / "conversion_metadata.json"),
            "source_sha256_status": "verified",
            "materialization_run": False,
        },
        "source_sha256": source_hashes,
    }
    assumptions = {
        "schema": SCHEMA + "/assumptions",
        "recording_id": "snimanje_23_10/episode20_kugla",
        "experiment_identity": {
            "all_segments_are_one_experiment": True,
            "split_group": split_group,
        },
        "temporal": {
            "annotation_segment_count": 12,
            "retained_segment_count": 10,
            "omitted_annotation_segments": [1, 12],
            "every_segment_starts_inactive": True,
            "independent_initialization": True,
            "frame_zero_use": "initialization_only",
            "parameters_shared_across_all_segments": True,
            "concatenated": False,
            "timestamps_rebased": False,
            "dataset_membership": "all ten chunks are training members; each config reserves its final frame for within-chunk validation",
        },
        "physical": {
            "nominal_density_kg_m3": NOMINAL_DENSITY_KG_M3,
            "reference_volume_m3": REFERENCE_VOLUME_M3,
            "eventual_mass_kg": CONSTANT_MASS_KG,
            "mass_policy": "constant across all chunks, inherited from the Episode18 reference volume and nominal density",
            "measurement_status": "inherited_assumption_not_episode20_measurement",
        },
        "coordinate_frames": {
            "pointclouds": "mocap",
            "tool_paths": "mocap",
            "scene": "table-aligned",
            "scene_from_source": "rigid transform from the independently estimated Episode20 table plane",
            "table_plane_validation": "passed",
        },
        "inherited_geometry": {
            "source_recording": "snimanje_23_10/episode18_kugla",
            "bundle": str(INHERITED_BUNDLE),
            "sha256": inherited_hashes,
            "physical_setup_status": "assumed_same_UR5e_and_Gen3_spatula_marker_assemblies",
            "episode20_icp_registration_run": False,
        },
        "execution": {
            "new_mocap_export_used": True,
            "source_annotation_modified": False,
            "ranges_materialized": False,
            "calibration_optimization_run": False,
            "fit_run": False,
        },
    }
    provenance = {
        "schema": SCHEMA + "/provenance",
        "generator": "experiments/differentiable_mpm/prepare_episode20_kugla_temporal.py",
        "source_annotation": {
            "path": str(source),
            "size_bytes": len(annotation_bytes),
            "sha256": annotation_hash,
        },
        "copy_policy": "accepted_temporal_annotations.json is copied byte-for-byte",
        "source_database": annotation["source"].get("database_file"),
        "source_metadata": annotation["source"].get("metadata_file"),
        "processed_export": {
            "path": str(full_episode),
            "source_sha256": source_info["source_sha256"],
            "output_frame": "mocap",
            "camera_tag_parent_frame": source_info["conversion"]["tag2real"]["parent_frame"],
        },
        "preserved_temporal_v1": {
            "path": str(DEFAULT_PRESERVED_V1),
            "file_sha256": preserved_v1_hashes,
        },
        "calibration_optimization_run": False,
        "fit_run": False,
    }
    return {
        "range_selection": range_selection,
        "assumptions": assumptions,
        "provenance": provenance,
        "ranges": ranges,
    }


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        return torch.load(path, map_location="cpu")


def replace_path_prefix(node: Any, old: str, new: str) -> Any:
    if isinstance(node, dict):
        return {key: replace_path_prefix(value, old, new) for key, value in node.items()}
    if isinstance(node, list):
        return [replace_path_prefix(value, old, new) for value in node]
    if isinstance(node, str) and node.startswith(old):
        return new + node[len(old) :]
    return node


def validate_materialized_chunk(
    chunk: Path,
    row: dict[str, Any],
    expected_count: int,
) -> dict[str, Any]:
    points_path = chunk / "pointclouds_interpolated.pt"
    paths_path = chunk / "paths_interpolated.pt"
    metadata_path = chunk / "sequence_metadata.json"
    for path in (points_path, paths_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing materialized chunk file: {path}")
    points_payload = _torch_load(points_path)
    paths_payload = _torch_load(paths_path)
    if not (
        isinstance(points_payload, list)
        and len(points_payload) == 1
        and isinstance(points_payload[0], list)
        and len(points_payload[0]) == expected_count
    ):
        raise ValueError(f"{chunk.name} has an invalid point-cloud payload or frame count")
    if not (
        isinstance(paths_payload, list)
        and len(paths_payload) == 1
        and isinstance(paths_payload[0], dict)
    ):
        raise ValueError(f"{chunk.name} has an invalid path payload")
    record = paths_payload[0]
    path = record.get("path")
    validity = record.get("stream_validity")
    if tuple(record.get("pose_frames", ())) != POSE_FRAMES:
        raise ValueError(f"{chunk.name} has unexpected tool stream names")
    if not isinstance(path, torch.Tensor) or tuple(path.shape) != (expected_count, 2, 14):
        raise ValueError(f"{chunk.name} path tensor must have shape [{expected_count},2,14]")
    if not isinstance(validity, torch.Tensor) or tuple(validity.shape) != (expected_count, 2):
        raise ValueError(f"{chunk.name} stream validity has an invalid shape")
    if not path.isfinite().all().item() or not validity.bool().all().item():
        raise ValueError(f"{chunk.name} requires finite paths and both valid tool streams")
    for frame in points_payload[0]:
        if not isinstance(frame, torch.Tensor) or frame.ndim != 2 or frame.shape[1] < 3:
            raise ValueError(f"{chunk.name} contains an invalid point-cloud frame")
        if not frame[:, :3].isfinite().all().item():
            raise ValueError(f"{chunk.name} contains nonfinite XYZ values")
    times = path[:, :, 13].detach().cpu().numpy().astype(np.float64)
    if not np.allclose(times[:, 0], times[:, 1], rtol=0, atol=1e-6):
        raise ValueError(f"{chunk.name} tool streams have different timestamps")
    if np.any(np.diff(times[:, 0]) <= 0) or np.any(np.diff(times[:, 0]) > 0.1):
        raise ValueError(f"{chunk.name} timestamps are not strictly increasing within 0.1 s")
    metadata = load_json(metadata_path)
    if metadata.get("frame_count") != expected_count:
        raise ValueError(f"{chunk.name} metadata frame count differs from tensors")
    expected_interval = [row["start_pointcloud_ordinal"], row["end_pointcloud_ordinal_exclusive"]]
    raw_range = metadata.get("raw_range", {})
    if [raw_range.get("start_pointcloud_ordinal"), raw_range.get("end_pointcloud_ordinal_exclusive")] != expected_interval:
        raise ValueError(f"{chunk.name} metadata has the wrong raw range")
    if metadata.get("output_validation_loader") != "verified_reference":
        raise ValueError(f"{chunk.name} was not reloaded through the frozen reference loader")
    metadata["coordinate_frames"] = {
        "pointclouds": "mocap",
        "tool_paths": "mocap",
        "scene_after_calibration": "table-aligned",
    }
    return {
        "frame_count": expected_count,
        "first_time_s": float(times[0, 0]),
        "last_time_s": float(times[-1, 0]),
        "sequence_fingerprint": metadata.get("sequence_fingerprint"),
        "metadata": metadata,
    }


def write_hash_manifest(root: Path) -> dict[str, str]:
    hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }
    payload = "".join(f"{digest}  {name}\n" for name, digest in hashes.items())
    (root / "SHA256SUMS").write_text(payload, encoding="utf-8")
    return hashes


def prepare(
    source: Path,
    full_episode: Path,
    output: Path,
    preserved_v1: Path = DEFAULT_PRESERVED_V1,
) -> Path:
    source = source.expanduser().resolve()
    full_episode = full_episode.expanduser().resolve()
    output = output.expanduser().resolve()
    preserved_v1 = preserved_v1.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Episode20 temporal package: {output}")
    annotation_bytes, annotation = load_annotation(source)
    source_info = require_mocap_inputs(full_episode)
    inherited_paths = {name: INHERITED_BUNDLE / name for name in INHERITED_FILES}
    missing = [str(path) for path in inherited_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inherited tool assets: " + ", ".join(missing))
    inherited_hashes = {name: sha256_file(path) for name, path in inherited_paths.items()}
    preserved_before = tree_sha256(preserved_v1)
    documents = build_documents(
        source,
        annotation_bytes,
        annotation,
        full_episode,
        source_info,
        inherited_hashes,
        preserved_before,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (temp / "accepted_temporal_annotations.json").write_bytes(annotation_bytes)
        (temp / "range_selection.json").write_bytes(canonical_json(documents["range_selection"]))
        (temp / "assumptions.json").write_bytes(canonical_json(documents["assumptions"]))
        (temp / "provenance.json").write_bytes(canonical_json(documents["provenance"]))
        calibration_dir = temp / "calibration"
        calibration_dir.mkdir()
        for name in ("scene_calibration_v2.json", "episode20_table_plane_validation.json"):
            source_path = full_episode / name
            shutil.copyfile(source_path, calibration_dir / name)
            if sha256_file(source_path) != sha256_file(calibration_dir / name):
                raise RuntimeError(f"Copied calibration input changed: {name}")

        recordings = temp / "recordings"
        with reference_policy("frozen"):
            materialized = materialize_ranges(
                full_episode,
                full_episode / "conversion_metadata.json",
                temp / "range_selection.json",
                temp / "accepted_temporal_annotations.json",
                recordings,
                max_gap_s=0.1,
            )
        if len(materialized) != 10:
            raise ValueError("Episode20 materialization must produce exactly ten chunks")

        chunk_rows = []
        for row, expected_count in zip(documents["ranges"], EXPECTED_FRAME_COUNTS):
            chunk = recordings / row["output_name"]
            checked = validate_materialized_chunk(chunk, row, expected_count)
            metadata_path = chunk / "sequence_metadata.json"
            metadata = replace_path_prefix(checked.pop("metadata"), str(temp), str(output))
            metadata_path.write_bytes(canonical_json(metadata))
            chunk_rows.append({"output_name": row["output_name"], **checked})

        selection = documents["range_selection"]
        selection["materialization_inputs"].update(
            materialization_run=True,
            output_root=str(output / "recordings"),
            validation_loader="verified_reference",
        )
        assumptions = documents["assumptions"]
        assumptions["execution"]["ranges_materialized"] = True
        provenance = documents["provenance"]
        provenance["materialization"] = {
            "status": "complete",
            "chunks": chunk_rows,
            "expected_frame_counts": list(EXPECTED_FRAME_COUNTS),
            "total_frames": sum(EXPECTED_FRAME_COUNTS),
            "validation_loader": "verified_reference",
        }
        (temp / "range_selection.json").write_bytes(canonical_json(selection))
        (temp / "assumptions.json").write_bytes(canonical_json(assumptions))
        (temp / "provenance.json").write_bytes(canonical_json(provenance))

        source_hashes_after = {
            name: sha256_file(path) for name, path in source_info["paths"].items()
        }
        if source_hashes_after != source_info["source_sha256"]:
            raise RuntimeError("Episode20 mocap export changed during temporal preparation")
        preserved_after = tree_sha256(preserved_v1)
        if preserved_after != preserved_before:
            raise RuntimeError("Preserved Episode20 temporal v1 changed during preparation")
        validation = {
            "schema": SCHEMA + "/validation",
            "status": "valid",
            "checks": {
                "annotation_revision": 12,
                "annotation_fingerprint": EXPECTED_FINGERPRINT,
                "retained_range_count": 10,
                "covered_raw_interval": [7, 184],
                "expected_frame_counts": list(EXPECTED_FRAME_COUNTS),
                "materialized_total_frames": sum(row["frame_count"] for row in chunk_rows),
                "both_tool_streams_valid": True,
                "timestamps_preserved": True,
                "source_frame": "mocap",
                "scene_frame": "table-aligned",
                "table_alignment_from_episode20_plane": True,
                "table_plane_validation_passed": True,
                "preserved_v1_unchanged": True,
                "calibration_optimization_run": False,
                "fit_run": False,
            },
            "chunks": chunk_rows,
            "source_sha256_start": source_info["source_sha256"],
            "source_sha256_end": source_hashes_after,
            "preserved_v1_file_sha256": preserved_after,
        }
        (temp / "validation.json").write_bytes(canonical_json(validation))
        write_hash_manifest(temp)
        os.rename(temp, output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--full-episode-dir", type=Path, default=DEFAULT_FULL_EPISODE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preserved-v1", type=Path, default=DEFAULT_PRESERVED_V1)
    args = parser.parse_args(argv)
    result = prepare(args.source, args.full_episode_dir, args.output, args.preserved_v1)
    print(result)
    print("Prepared Episode20 temporal v2 only; no calibration optimization or simulation was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
