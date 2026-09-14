from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch: Any = importlib.import_module("torch")

from experiments.differentiable_mpm.materialize_deformpath_ranges import materialize_ranges


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixture(tmp_path: Path, *, indices=None, times=None):
    episode = tmp_path / "filtered"
    episode.mkdir(parents=True)
    raw_count = 20
    stamps = [10_000 + i * 1_000 for i in range(raw_count)]
    indices = indices or [0, 2, 3, 5, 6, 8, 10, 11, 13, 15, 16, 18]
    times = times or [0, .03, .07, .11, .17, .21, .25, .29, .33, .37, .41, .45]
    clouds = [torch.from_numpy(np.array([[float(raw), raw + 0.25, raw + 0.5, 100 + raw]], dtype=np.float64))
              for raw in indices]
    path = torch.from_numpy(np.zeros((len(indices), 2, 14), dtype=np.float64))
    for i, time in enumerate(times):
        path[i, :, :13] = torch.from_numpy(i + np.arange(13, dtype=np.float64))
        path[i, :, 13] = time
    paths = {"path": path, "stream_validity": torch.from_numpy(np.ones((len(indices), 2), dtype=np.bool_)),
             "pose_frames": ["left", "right"], "preserved_extra": torch.from_numpy(np.array([7, 8]))}
    torch.save([clouds], episode / "pointclouds_interpolated.pt")
    torch.save([paths], episode / "paths_interpolated.pt")
    write_json(episode / "sequence_metadata.json", {
        "pointcloud_indices": indices, "parent_note": "kept", "num_pointcloud_frames": len(indices),
        "num_timeline_samples": len(indices), "num_valid_samples": len(indices), "num_pose_frames": 2,
        "frame_stats": [{"source_position": i} for i in range(len(indices))],
        "pointcloud_filter": {"frame_stats": [{"filtered_position": i} for i in range(len(indices))]},
        "per_stream_valid_counts": {"left": len(indices), "right": len(indices)},
        "sample_time_range_s": [times[0], times[-1]], "valid_sample_time_range_s": [times[0], times[-1]],
        "sample_dt_sec": 0.04, "sample_hz": 25.0,
        "input_dir": "/parent/export", "output_dir": str(episode),
        "pointclouds_path": "/parent/export/pointclouds.pt", "paths_path": "/parent/export/paths.pt",
        "output_pointclouds_path": str(episode / "pointclouds_interpolated.pt"),
        "output_paths_path": str(episode / "paths_interpolated.pt")})
    conversion = tmp_path / "conversion_metadata.json"
    write_json(conversion, {"episode": {"timestamps_ns": [str(value) for value in stamps]}})
    fingerprint = "a" * 64
    split_group = "rosbag2-sha256:" + fingerprint
    boundaries = []
    for ordinal, kind in ((0, "automatic_start"), (4, "manual"), (7, "manual"),
                          (9, "manual"), (14, "manual"), (17, "manual"), (20, "automatic_end")):
        boundaries.append({"id": f"b{ordinal}", "kind": kind, "pointcloud_ordinal": ordinal,
                           "pointcloud_header_stamp_ns": None if ordinal == raw_count else str(stamps[ordinal])})
    annotation = tmp_path / "accepted.json"
    write_json(annotation, {"schema_name": "deformpath.temporal_annotations", "schema_version": 1,
                            "split_group": split_group,
                            "source": {"fingerprint": {"value": fingerprint},
                                       "topics": {"pointcloud": {"message_count": raw_count}}},
                            "boundaries": boundaries})
    ranges = [
        {"output_name": "one", "start_pointcloud_ordinal": 0, "end_pointcloud_ordinal_exclusive": 7,
         "start_pointcloud_header_stamp_ns": str(stamps[0]), "end_pointcloud_header_stamp_ns_exclusive": str(stamps[7])},
        {"output_name": "two", "start_pointcloud_ordinal": 9, "end_pointcloud_ordinal_exclusive": 14,
         "start_pointcloud_header_stamp_ns": str(stamps[9]), "end_pointcloud_header_stamp_ns_exclusive": str(stamps[14])},
        {"output_name": "three", "start_pointcloud_ordinal": 14, "end_pointcloud_ordinal_exclusive": 20,
         "start_pointcloud_header_stamp_ns": str(stamps[14]), "end_pointcloud_header_stamp_ns_exclusive": None},
    ]
    selection = tmp_path / "ranges.json"
    sources = [episode / "pointclouds_interpolated.pt", episode / "paths_interpolated.pt",
               episode / "sequence_metadata.json", conversion, annotation]
    write_json(selection, {"schema": "taichidough/deformpath-range-selection/v1",
                           "source": {"recording_id": "recording-3", "fingerprint": fingerprint,
                                      "split_group": split_group},
                           "annotation": {"path": str(annotation), "sha256": sha(annotation)},
                           "retained_ranges": ranges,
                           "source_sha256": {p.name: sha(p) for p in sources}})
    return episode, conversion, selection, annotation, tmp_path / "outputs", clouds, paths, indices, stamps


def run(case):
    episode, conversion, selection, annotation, output, *_ = case
    return materialize_ranges(episode, conversion, selection, annotation, output)


def test_materializes_noncontiguous_ranges_without_changing_tensors_or_time(tmp_path):
    case = fixture(tmp_path)
    results = run(case)
    episode, _, _, _, output, clouds, paths, indices, stamps = case
    assert [r["output_name"] for r in results] == ["one", "two", "three"]
    assert len({r["sequence_fingerprint"] for r in results}) == 3
    expected_positions = [[0, 1, 2, 3, 4], [6, 7, 8], [9, 10, 11]]
    for result, positions in zip(results, expected_positions):
        directory = output / result["output_name"]
        saved_clouds = torch.load(directory / "pointclouds_interpolated.pt", weights_only=True)[0]
        saved_paths = torch.load(directory / "paths_interpolated.pt", weights_only=True)[0]
        assert len(saved_clouds) == len(positions)
        for actual, position in zip(saved_clouds, positions):
            assert actual.equal(clouds[position])
        assert saved_paths["path"].equal(paths["path"][positions])
        assert saved_paths["stream_validity"].equal(paths["stream_validity"][positions])
        assert saved_paths["preserved_extra"].equal(paths["preserved_extra"])
        metadata = json.loads((directory / "sequence_metadata.json").read_text())
        assert metadata["split_group"].startswith("rosbag2-sha256:")
        assert metadata["source_recording_id"] == "recording-3"
        assert metadata["parent_note"] == "kept"
        assert metadata["num_pointcloud_frames"] == len(positions)
        assert metadata["num_timeline_samples"] == len(positions)
        assert metadata["num_valid_samples"] == len(positions)
        assert metadata["num_pose_frames"] == 2
        assert metadata["frame_stats"] == [{"source_position": p} for p in positions]
        assert metadata["pointcloud_filter"]["frame_stats"] == [
            {"filtered_position": p} for p in positions]
        assert metadata["per_stream_valid_counts"] == {"left": len(positions), "right": len(positions)}
        selected_times = [float(paths["path"][p, 0, 13]) for p in positions]
        assert metadata["sample_time_range_s"] == [selected_times[0], selected_times[-1]]
        assert metadata["valid_sample_time_range_s"] == [selected_times[0], selected_times[-1]]
        expected_dt = float(np.median(np.diff(selected_times)))
        assert metadata["sample_dt_sec"] == pytest.approx(expected_dt)
        assert metadata["sample_hz"] == pytest.approx(1.0 / expected_dt)
        assert metadata["output_dir"] == str(directory)
        assert metadata["pointclouds_path"] == "/parent/export/pointclouds.pt"
        assert metadata["paths_path"] == "/parent/export/paths.pt"
        assert metadata["output_pointclouds_path"] == str(directory / "pointclouds_interpolated.pt")
        assert metadata["output_paths_path"] == str(directory / "paths_interpolated.pt")
        assert metadata["source_processed_positions"] == positions
        assert metadata["original_exporter_indices"] == [indices[p] for p in positions]
        assert metadata["raw_pointcloud_ordinals"] == [indices[p] for p in positions]
        assert metadata["first_retained_header_stamp_ns"] == str(stamps[indices[positions[0]]])
        assert metadata["temporal_policy"]["timestamps_rebased"] is False
    mapping = json.loads((output / "one" / "sequence_metadata.json").read_text())["accepted_manual_mark_mappings"]
    assert {row["raw_pointcloud_ordinal"]: row["output_frame"] for row in mapping
            if "output_frame" in row} == {4: 3}
    # Boundary 4 was filtered out, so it maps to the first retained raw ordinal at or after it (5).
    assert mapping[0]["mapping"] == "first_retained_at_or_after"
    end_mapping = next(row for row in mapping if row["raw_pointcloud_ordinal"] == 7)
    assert end_mapping["output_frame_exclusive"] == 5
    assert end_mapping["mapping"] == "range_end_exclusive"
    final_boundaries = json.loads((output / "three" / "sequence_metadata.json").read_text())["accepted_boundary_mappings"]
    assert next(row for row in final_boundaries if row["raw_pointcloud_ordinal"] == 20)["output_frame_exclusive"] == 3
    with pytest.raises(FileExistsError, match="overwrite"):
        run(case)


def mutate_json(path: Path, function) -> None:
    value = json.loads(path.read_text())
    function(value)
    write_json(path, value)


def refresh_hash(case, name: str) -> None:
    selection = case[2]
    value = json.loads(selection.read_text())
    paths = {case[0].joinpath("pointclouds_interpolated.pt").name: case[0] / "pointclouds_interpolated.pt",
             case[0].joinpath("paths_interpolated.pt").name: case[0] / "paths_interpolated.pt",
             "sequence_metadata.json": case[0] / "sequence_metadata.json",
             case[1].name: case[1], case[3].name: case[3]}
    value["source_sha256"][name] = sha(paths[name])
    if name == case[3].name:
        value["annotation"]["sha256"] = sha(case[3])
    write_json(selection, value)


def test_rejects_overlapping_and_empty_ranges(tmp_path):
    case = fixture(tmp_path)
    def overlap(value):
        value["retained_ranges"][1]["start_pointcloud_ordinal"] = 4
        value["retained_ranges"][1]["start_pointcloud_header_stamp_ns"] = "14000"
    mutate_json(case[2], overlap)
    with pytest.raises(ValueError, match="overlap"):
        run(case)
    case = fixture(tmp_path / "empty")
    mutate_json(case[2], lambda value: value["retained_ranges"][1].update(
        start_pointcloud_ordinal=7, end_pointcloud_ordinal_exclusive=9,
        start_pointcloud_header_stamp_ns="17000", end_pointcloud_header_stamp_ns_exclusive="19000"))
    with pytest.raises(ValueError, match="at least 3"):
        run(case)


def test_rejects_invalid_tool_validity(tmp_path):
    case = fixture(tmp_path)
    payload = torch.load(case[0] / "paths_interpolated.pt", weights_only=True)
    payload[0]["stream_validity"][1, 0] = False
    torch.save(payload, case[0] / "paths_interpolated.pt")
    refresh_hash(case, "paths_interpolated.pt")
    with pytest.raises(ValueError, match="Both tool streams"):
        run(case)


@pytest.mark.parametrize("times, message", [
    ([0, .04, .08, .08, .16, .20, .24, .28, .32, .36, .40, .44], "strictly increasing"),
    ([0, .04, .08, .20, .24, .28, .32, .36, .40, .44, .48, .52], "gap exceeds"),
])
def test_rejects_bad_path_timestamps(tmp_path, times, message):
    case = fixture(tmp_path, times=times)
    with pytest.raises(ValueError, match=message):
        run(case)


def test_rejects_stale_hash_and_changed_identity(tmp_path):
    case = fixture(tmp_path)
    mutate_json(case[0] / "sequence_metadata.json", lambda value: value.update(changed=True))
    with pytest.raises(ValueError, match="Stale source"):
        run(case)
    case = fixture(tmp_path / "identity")
    mutate_json(case[3], lambda value: value.update(split_group="rosbag2-sha256:" + "b" * 64))
    refresh_hash(case, case[3].name)
    with pytest.raises(ValueError, match="identity"):
        run(case)


def test_rejects_duplicate_or_missing_exact_stamp_mapping(tmp_path):
    case = fixture(tmp_path)
    mutate_json(case[1], lambda value: value["episode"]["timestamps_ns"].__setitem__(2, "10000"))
    refresh_hash(case, case[1].name)
    with pytest.raises(ValueError, match="duplicate"):
        run(case)
    case = fixture(tmp_path / "missing")
    mutate_json(case[1], lambda value: value["episode"]["timestamps_ns"].pop())
    refresh_hash(case, case[1].name)
    with pytest.raises(ValueError, match="count differs"):
        run(case)


def test_slices_raw_frame_stats_through_pointcloud_indices(tmp_path):
    case = fixture(tmp_path)
    metadata_path = case[0] / "sequence_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["pointcloud_filter"]["frame_stats"] = [
        {"raw_position": position} for position in range(20)
    ]
    write_json(metadata_path, metadata)
    refresh_hash(case, "sequence_metadata.json")

    run(case)

    first = json.loads((case[4] / "one" / "sequence_metadata.json").read_text())
    assert first["pointcloud_filter"]["frame_stats"] == [
        {"raw_position": position} for position in [0, 2, 3, 5, 6]
    ]
