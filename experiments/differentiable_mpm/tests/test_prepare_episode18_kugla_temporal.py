from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.differentiable_mpm.prepare_episode18_kugla_temporal import (
    CHUNK01_VOLUME_M3,
    DEFAULT_SOURCE,
    DENSITY_KG_M3,
    generate,
)


def load(path: Path):
    return json.loads(path.read_text())


def test_generates_exact_annotation_copy_all_ranges_and_assumptions(tmp_path: Path):
    output = tmp_path / "package"
    generate(DEFAULT_SOURCE, output)

    assert (output / "accepted_temporal_annotations.json").read_bytes() == DEFAULT_SOURCE.read_bytes()
    annotation = load(output / "accepted_temporal_annotations.json")
    selection = load(output / "range_selection.json")
    assumptions = load(output / "assumptions.json")
    validation = load(output / "validation.json")

    ranges = selection["ranges"]
    assert len(ranges) == 13
    assert [row["output_name"] for row in ranges] == [
        f"episode18_kugla_chunk{i:02d}" for i in range(1, 14)
    ]
    assert [(row["start_pointcloud_ordinal"], row["end_pointcloud_ordinal_exclusive"])
            for row in ranges] == [
        (row["start_pointcloud_ordinal"], row["end_pointcloud_ordinal_exclusive"])
        for row in annotation["segments"][1:-1]
    ]
    assert all(left["end_pointcloud_ordinal_exclusive"] == right["start_pointcloud_ordinal"]
               for left, right in zip(ranges, ranges[1:]))
    assert ranges[0]["start_pointcloud_ordinal"] == 46
    assert ranges[-1]["end_pointcloud_ordinal_exclusive"] == 287
    assert all(row["initial_activity"] == "inactive" for row in ranges)
    assert all(row["frame_zero_semantics"] == "initialization_only" for row in ranges)

    assert assumptions["experiment_identity"]["all_segments_are_one_experiment"] is True
    assert assumptions["temporal"]["parameters_shared_across_all_segments"] is True
    assert assumptions["physical"]["density_kg_m3"] == DENSITY_KG_M3
    assert assumptions["physical"]["eventual_mass_kg"] == pytest.approx(
        CHUNK01_VOLUME_M3 * DENSITY_KG_M3
    )
    assert validation["checks"]["covered_frame_count"] == 241
    assert validation["checks"]["omitted_annotation_segments"] == [1, 15]
    assert validation["checks"]["existing_processed_export_verified"] is True


def test_hash_manifest_and_idempotent_immutable_output(tmp_path: Path):
    output = tmp_path / "package"
    generate(DEFAULT_SOURCE, output)
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    generate(DEFAULT_SOURCE, output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == first

    entries = {}
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    assert entries
    for name, digest in entries.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest

    (output / "assumptions.json").write_text("{}\n")
    with pytest.raises(FileExistsError, match="immutable"):
        generate(DEFAULT_SOURCE, output)
