from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch: Any = importlib.import_module("torch")

import experiments.differentiable_mpm.prepare_episode20_kugla_temporal as module


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    full = tmp_path / "mocap"
    full.mkdir()
    calibration = {
        "schema": "taichidough/scene-calibration/v2",
        "source_frame": "mocap",
        "scene_frame": "table-aligned",
        "scene_from_source": [
            [1.0, 0.0, 0.0, 0.4],
            [0.0, 1.0, 0.0, -0.05],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "scene_from_camera": np.eye(4).tolist(),
        "floor_plane_scene": [0.0, 1.0, 0.0, 0.0],
        "provenance": {
            "table_alignment": {
                "previous_scene_frame": "mocap",
                "source_tensors_transformed": False,
            }
        },
    }
    write_json(full / "scene_calibration_v2.json", calibration)
    calibration_hash = sha(full / "scene_calibration_v2.json")
    (full / "pointclouds_interpolated.pt").write_bytes(b"points")
    (full / "paths_interpolated.pt").write_bytes(b"paths")
    calibration_record = {
        "status": "available",
        "sha256": calibration_hash,
        "source_frame": "mocap",
        "scene_frame": "table-aligned",
    }
    write_json(
        full / "conversion_metadata.json",
        {
            "output_frame": "mocap",
            "pose_frames": list(module.POSE_FRAMES),
            "tag2real": {
                "parent_frame": "tag16h5:3",
                "child_frame": "tag3_real",
            },
            "calibration": calibration_record,
        },
    )
    write_json(
        full / "sequence_metadata.json",
        {
            "pose_frames": list(module.POSE_FRAMES),
            "calibration": calibration_record,
        },
    )
    write_json(
        full / "episode20_table_plane_validation.json",
        {"passed": True, "calibration_sha256": calibration_hash},
    )
    write_json(full / "episode20_export_validation.json", {"passed": True})

    inherited = tmp_path / "tools"
    inherited.mkdir()
    for name in module.INHERITED_FILES:
        (inherited / name).write_bytes(name.encode())
    monkeypatch.setattr(module, "INHERITED_BUNDLE", inherited)

    preserved = tmp_path / "v1"
    preserved.mkdir()
    (preserved / "kept.txt").write_text("unchanged\n", encoding="utf-8")
    return full, preserved


def fake_materialize(
    full: Path,
    conversion: Path,
    selection_path: Path,
    annotation: Path,
    output: Path,
    *,
    max_gap_s: float,
) -> list[dict[str, Any]]:
    del full, conversion, annotation
    selection = json.loads(selection_path.read_text())
    output.mkdir()
    results = []
    for row, count in zip(selection["ranges"], module.EXPECTED_FRAME_COUNTS):
        directory = output / row["output_name"]
        directory.mkdir()
        clouds = [
            torch.tensor([[0.4, 0.02, 0.5]], dtype=torch.float32)
            for _ in range(count)
        ]
        paths = torch.zeros((count, 2, 14), dtype=torch.float32)
        paths[:, :, 6] = 1.0
        paths[:, :, 13] = torch.arange(count, dtype=torch.float32)[:, None] * 0.03
        torch.save([clouds], directory / "pointclouds_interpolated.pt")
        torch.save(
            [
                {
                    "path": paths,
                    "stream_validity": torch.ones((count, 2), dtype=torch.bool),
                    "pose_frames": list(module.POSE_FRAMES),
                }
            ],
            directory / "paths_interpolated.pt",
        )
        metadata = {
            "frame_count": count,
            "raw_range": {
                "start_pointcloud_ordinal": row["start_pointcloud_ordinal"],
                "end_pointcloud_ordinal_exclusive": row[
                    "end_pointcloud_ordinal_exclusive"
                ],
            },
            "sequence_fingerprint": f"{len(results) + 1:064x}",
            "output_validation_loader": "verified_reference",
            "output_dir": str(directory),
        }
        write_json(directory / "sequence_metadata.json", metadata)
        results.append(
            {
                "output_name": row["output_name"],
                "path": str(directory),
                "frame_count": count,
                "sequence_fingerprint": metadata["sequence_fingerprint"],
            }
        )
    assert max_gap_s == 0.1
    return results


class Policy:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        del args


def fake_reference_policy(policy: str) -> Policy:
    del policy
    return Policy()


def test_prepares_mocap_v2_atomically_and_preserves_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    full, preserved = make_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "materialize_ranges", fake_materialize)
    monkeypatch.setattr(module, "reference_policy", fake_reference_policy)
    output = tmp_path / "temporal_v2"
    before = (preserved / "kept.txt").read_bytes()

    result = module.prepare(module.DEFAULT_SOURCE, full, output, preserved)

    assert result == output
    assert (preserved / "kept.txt").read_bytes() == before
    assert (
        output / "accepted_temporal_annotations.json"
    ).read_bytes() == module.DEFAULT_SOURCE.read_bytes()
    selection = json.loads((output / "range_selection.json").read_text())
    assumptions = json.loads((output / "assumptions.json").read_text())
    validation = json.loads((output / "validation.json").read_text())
    assert selection["name"] == "episode20-kugla-temporal-v2"
    assert selection["materialization_inputs"]["materialization_run"] is True
    assert [
        row["expected_materialized_frame_count"] for row in selection["ranges"]
    ] == list(module.EXPECTED_FRAME_COUNTS)
    assert (
        assumptions["physical"]["measurement_status"]
        == "inherited_assumption_not_episode20_measurement"
    )
    assert assumptions["coordinate_frames"] == {
        "pointclouds": "mocap",
        "tool_paths": "mocap",
        "scene": "table-aligned",
        "scene_from_source": "rigid transform from the independently estimated Episode20 table plane",
        "table_plane_validation": "passed",
    }
    assert validation["status"] == "valid"
    assert validation["checks"]["materialized_total_frames"] == 177
    assert len(list((output / "recordings").iterdir())) == 10
    entries = {}
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    assert "recordings/episode20_kugla_chunk10/sequence_metadata.json" in entries
    assert all(sha(output / name) == digest for name, digest in entries.items())
    chunk_metadata = json.loads(
        (
            output
            / "recordings/episode20_kugla_chunk01/sequence_metadata.json"
        ).read_text()
    )
    assert chunk_metadata["output_dir"].startswith(str(output))
    assert chunk_metadata["coordinate_frames"] == {
        "pointclouds": "mocap",
        "tool_paths": "mocap",
        "scene_after_calibration": "table-aligned",
    }


def test_refuses_existing_output_before_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    full, preserved = make_inputs(tmp_path, monkeypatch)
    output = tmp_path / "temporal_v2"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        module.prepare(module.DEFAULT_SOURCE, full, output, preserved)


def test_rejects_non_mocap_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    full, preserved = make_inputs(tmp_path, monkeypatch)
    calibration_path = full / "scene_calibration_v2.json"
    calibration = json.loads(calibration_path.read_text())
    calibration["source_frame"] = "camera_color_optical_frame"
    write_json(calibration_path, calibration)
    with pytest.raises(ValueError, match="map mocap inputs"):
        module.prepare(module.DEFAULT_SOURCE, full, tmp_path / "out", preserved)


def test_rejects_identity_table_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    full, preserved = make_inputs(tmp_path, monkeypatch)
    calibration_path = full / "scene_calibration_v2.json"
    calibration = json.loads(calibration_path.read_text())
    calibration["scene_from_source"] = np.eye(4).tolist()
    write_json(calibration_path, calibration)
    with pytest.raises(ValueError, match="nonidentity"):
        module.prepare(module.DEFAULT_SOURCE, full, tmp_path / "out", preserved)


def test_materialization_failure_removes_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    full, preserved = make_inputs(tmp_path, monkeypatch)

    def fail(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(module, "materialize_ranges", fail)
    monkeypatch.setattr(module, "reference_policy", fake_reference_policy)
    with pytest.raises(RuntimeError, match="synthetic"):
        module.prepare(module.DEFAULT_SOURCE, full, tmp_path / "out", preserved)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.*"))
