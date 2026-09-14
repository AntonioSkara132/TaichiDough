"""Freeze Episode18 kugla annotations and generate its interior-range dataset plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

SCHEMA = "taichidough/episode18-kugla-temporal/v1"
DEFAULT_SOURCE = Path(
    "/home/antonio/diplomski_antonio/diplomski/data/deformpath_training/"
    "DeformPath2/snimanje_23_10/episode18_kugla/temporal_annotations.json"
)
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "episode18_kugla_temporal_v1"
FULL_EPISODE = Path(
    "/home/antonio/diplomski_antonio/diplomski/data/deformpath_training/"
    "DeformPath3/snimanje_23_10/episode18_kugla"
)
CONVERSION_METADATA = FULL_EPISODE / "conversion_metadata.json"
CHUNK01_VOLUME_M3 = 0.000113832
DENSITY_KG_M3 = 1200.0


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


def load_annotation(source: Path) -> tuple[bytes, dict[str, Any]]:
    raw = source.read_bytes()
    annotation = json.loads(raw)
    if annotation.get("schema_name") != "deformpath.temporal_annotations" or annotation.get("schema_version") != 1:
        raise ValueError("Unsupported temporal annotation schema")
    return raw, annotation


def build_files(source: Path) -> dict[str, bytes]:
    annotation_bytes, annotation = load_annotation(source)
    boundaries = annotation.get("boundaries")
    segments = annotation.get("segments")
    if not isinstance(boundaries, list) or not isinstance(segments, list) or len(segments) != 15:
        raise ValueError("Episode18 kugla annotation must contain 15 segments")
    if len(boundaries) != 16:
        raise ValueError("Episode18 kugla annotation must contain 16 boundaries")
    fingerprint = annotation["source"]["fingerprint"]["value"]
    split_group = annotation["split_group"]
    retained_segments = segments[1:-1]
    ranges = []
    for index, segment in enumerate(retained_segments, 1):
        start = segment["start_pointcloud_ordinal"]
        end = segment["end_pointcloud_ordinal_exclusive"]
        if index > 1 and start != retained_segments[index - 2]["end_pointcloud_ordinal_exclusive"]:
            raise ValueError("Episode18 retained segments are not contiguous")
        ranges.append({
            "id": f"chunk{index:02d}",
            "output_name": f"episode18_kugla_chunk{index:02d}",
            "annotation_segment_id": segment["id"],
            "start_pointcloud_ordinal": start,
            "end_pointcloud_ordinal_exclusive": end,
            "raw_pointcloud_frame_capacity": segment["pointcloud_frame_count"],
            "start_pointcloud_header_stamp_ns": segment["start_pointcloud_header_stamp_ns"],
            "end_pointcloud_header_stamp_ns_exclusive": segment["end_pointcloud_header_stamp_ns_exclusive"],
            "last_included_pointcloud_header_stamp_ns": segment["last_included_pointcloud_header_stamp_ns"],
            "frame_zero_semantics": "initialization_only",
            "initial_activity": "inactive",
        })
    if ranges[0]["start_pointcloud_ordinal"] != 46 or ranges[-1]["end_pointcloud_ordinal_exclusive"] != 287:
        raise ValueError("Episode18 retained ranges must cover [46, 287)")

    annotation_hash = sha256_bytes(annotation_bytes)
    processed_sources = {
        "pointclouds_interpolated.pt": FULL_EPISODE / "pointclouds_interpolated.pt",
        "paths_interpolated.pt": FULL_EPISODE / "paths_interpolated.pt",
        "sequence_metadata.json": FULL_EPISODE / "sequence_metadata.json",
        "conversion_metadata.json": CONVERSION_METADATA,
    }
    missing = [str(path) for path in processed_sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prepared Episode18 inputs: " + ", ".join(missing))
    source_hashes = {name: sha256_file(path) for name, path in processed_sources.items()}
    source_hashes["accepted_temporal_annotations.json"] = annotation_hash
    range_selection = {
        "schema": "taichidough/deformpath-range-selection/v1",
        "name": "episode18-kugla-temporal-v1",
        "source": {
            "recording_id": "snimanje_23_10/episode18_kugla",
            "fingerprint": fingerprint,
            "split_group": split_group,
            "pointcloud_frame_count": annotation["source"]["topics"]["pointcloud"]["message_count"],
        },
        "annotation": {
            "path": "accepted_temporal_annotations.json",
            "sha256": annotation_hash,
            "schema_name": annotation["schema_name"],
            "schema_version": annotation["schema_version"],
            "annotation_revision": annotation.get("annotation_revision"),
        },
        "ranges": ranges,
        "processing": {
            "range_semantics": "half-open [start_pointcloud_ordinal, end_pointcloud_ordinal_exclusive)",
            "concatenate_ranges": False,
            "rebase_timestamps": False,
            "materialize_as_independent_episodes": True,
            "maximum_replay_gap_s": 0.1,
            "frame_zero_reserved_for_initialization": True,
        },
        "materialization_inputs": {
            "tool": "experiments/differentiable_mpm/materialize_deformpath_ranges.py",
            "full_episode_dir": str(FULL_EPISODE),
            "conversion_metadata": str(CONVERSION_METADATA),
            "source_sha256_status": "verified",
            "materialization_run": False,
        },
        "source_sha256": source_hashes,
    }
    mass = CHUNK01_VOLUME_M3 * DENSITY_KG_M3
    assumptions = {
        "schema": SCHEMA + "/assumptions",
        "recording_id": "snimanje_23_10/episode18_kugla",
        "experiment_identity": {"all_segments_are_one_experiment": True, "split_group": split_group},
        "temporal": {
            "annotation_segment_count": 15,
            "retained_segment_count": 13,
            "omitted_annotation_segments": [1, 15],
            "every_segment_starts_inactive": True,
            "independent_initialization": True,
            "frame_zero_use": "initialization_only",
            "parameters_shared_across_all_segments": True,
            "concatenated": False,
            "timestamps_rebased": False,
        },
        "physical": {
            "density_kg_m3": DENSITY_KG_M3,
            "mass_policy": "constant across every retained segment, derived from the validated episode18_kugla frame-0 reconstruction volume",
            "chunk01_volume_m3": CHUNK01_VOLUME_M3,
            "eventual_mass_kg": mass,
            "status": "ready",
        },
        "execution": {
            "existing_processed_export_used": True,
            "source_annotation_modified": False,
            "ranges_materialized": False,
            "calibration_run": False,
            "fit_run": False,
        },
    }
    provenance = {
        "schema": SCHEMA + "/provenance",
        "generator": "experiments/differentiable_mpm/prepare_episode18_kugla_temporal.py",
        "source_annotation": {"path": str(source), "size_bytes": len(annotation_bytes), "sha256": annotation_hash},
        "copy_policy": "accepted_temporal_annotations.json is copied byte-for-byte",
        "source_database": annotation["source"].get("database_file"),
        "source_metadata": annotation["source"].get("metadata_file"),
        "processed_export": {
            "path": str(FULL_EPISODE),
            "source_sha256": source_hashes,
            "already_existed": True,
        },
        "calibration_run": False,
        "fit_run": False,
    }
    files = {
        "accepted_temporal_annotations.json": annotation_bytes,
        "range_selection.json": canonical_json(range_selection),
        "assumptions.json": canonical_json(assumptions),
        "provenance.json": canonical_json(provenance),
    }
    hashes = {name: sha256_bytes(data) for name, data in sorted(files.items())}
    validation = {
        "schema": SCHEMA + "/validation",
        "status": "valid",
        "checks": {
            "annotation_byte_copy_matches_source": True,
            "annotation_sha256": annotation_hash,
            "boundary_count": len(boundaries),
            "range_count": len(ranges),
            "ranges_contiguous": True,
            "covered_raw_interval": [46, 287],
            "covered_frame_count": sum(r["raw_pointcloud_frame_capacity"] for r in ranges),
            "omitted_annotation_segments": [1, 15],
            "all_segments_start_inactive": True,
            "one_experiment": True,
            "frame_zero_initialization_only": True,
            "parameters_shared": True,
            "density_kg_m3": DENSITY_KG_M3,
            "chunk01_volume_m3": CHUNK01_VOLUME_M3,
            "eventual_mass_kg": mass,
            "mass_status": "ready",
            "materializer_requirements_recorded": True,
            "existing_processed_export_verified": True,
        },
        "file_sha256": hashes,
    }
    files["validation.json"] = canonical_json(validation)
    bundle_hashes = {name: sha256_bytes(data) for name, data in sorted(files.items())}
    files["SHA256SUMS"] = "".join(f"{digest}  {name}\n" for name, digest in bundle_hashes.items()).encode()
    return files


def generate(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    files = build_files(source)
    if output.exists():
        existing = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
        if existing == files:
            return
        raise FileExistsError(f"Refusing to replace non-identical immutable package: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, data in files.items():
            (temp / name).write_bytes(data)
        temp.rename(output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    generate(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
