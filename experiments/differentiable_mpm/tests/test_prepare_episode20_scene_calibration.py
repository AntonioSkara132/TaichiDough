"""Tests for safe Episode20 mocap export preparation."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from experiments.differentiable_mpm.estimate_episode20_table_plane import (
    compare_planes,
    write_report,
)
from experiments.differentiable_mpm.prepare_episode20_scene_calibration import (
    POSE_FRAMES,
    build_commands,
    disk_preflight,
    prepare,
    rewrite_generated_metadata,
    scan_tokens,
    sha256_file,
    validate_calibration,
)


def raw_mocap_calibration() -> dict:
    return {
        "schema": "taichidough/scene-calibration/v2",
        "name": "episode20-mocap",
        "source_frame": "mocap",
        "scene_frame": "mocap",
        "scene_from_source": np.eye(4).tolist(),
        "scene_from_camera": np.eye(4).tolist(),
        "floor_plane_scene": [0.0, 1.0, 0.0, 0.0],
        "camera": {"scene_from_camera": np.eye(4).tolist()},
        "provenance": {
            "camera_tag_parent_frame": "tag16h5:3",
            "camera_tag_frame": "tag3_real",
        },
        "diagnostics": {
            "mocap_from_camera_aggregation": {
                "sample_count": 10,
                "inlier_count": 10,
                "translation_limit_m": 0.01,
                "rotation_limit_deg": 1.0,
                "translation_residual_m": [0.001] * 10,
                "rotation_residual_deg": [0.1] * 10,
            }
        },
    }


def table_aligned_calibration() -> dict:
    calibration = raw_mocap_calibration()
    calibration.update(
        name="episode20-table-aligned",
        scene_frame="table-aligned",
        scene_from_source=[
            [1.0, 0.0, 0.0, 0.4],
            [0.0, 1.0, 0.0, -0.05],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ],
        floor_plane_scene=[0.0, 1.0, 0.0, 0.0],
        provenance={
            "method": "rigid-table-plane-alignment",
            "source_calibration_provenance": raw_mocap_calibration()["provenance"],
            "table_alignment": {
                "previous_scene_frame": "mocap",
                "source_tensors_transformed": False,
            },
        },
    )
    return calibration


def test_scan_tokens_handles_block_boundaries(tmp_path: Path):
    database = tmp_path / "episode.db3"
    database.write_bytes(b"x" * ((8 << 20) - 4) + b"tag16h5:3 and tag16h5:4")
    assert scan_tokens([database], ("tag16h5:3", "tag16h5:4")) == {
        "tag16h5:3": True,
        "tag16h5:4": True,
    }


def test_disk_preflight_reports_exact_requirement(tmp_path: Path):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "a").write_bytes(b"12345")
    result = disk_preflight(tmp_path, reference, reserve_bytes=7)
    assert result["reference_export_bytes"] == 5
    assert result["required_available_bytes"] == 12
    assert result["passed"] is True


def test_validate_calibration_requires_table_alignment_and_selected_tag():
    calibration = table_aligned_calibration()
    conversion = {
        "calibration": {
            "status": "available",
            "source_frame": "mocap",
            "scene_frame": "table-aligned",
        }
    }
    assert validate_calibration(calibration, conversion)["inlier_count"] == 10
    calibration["provenance"]["source_calibration_provenance"][
        "camera_tag_parent_frame"
    ] = "tag16h5:4"
    with pytest.raises(ValueError, match="wrong camera AprilTag"):
        validate_calibration(calibration, conversion)


def test_validate_calibration_rejects_identity_table_alignment():
    calibration = table_aligned_calibration()
    calibration["scene_from_source"] = np.eye(4).tolist()
    conversion = {
        "calibration": {
            "status": "available",
            "source_frame": "mocap",
            "scene_frame": "table-aligned",
        }
    }
    with pytest.raises(ValueError, match="nonidentity"):
        validate_calibration(calibration, conversion)


def test_compare_planes_checks_normal_and_height():
    close = compare_planes(
        [0, 1, 0, -0.001],
        [0, 1, 0, 0],
        [0.2, 0.7],
        maximum_angle_deg=0.1,
        maximum_height_difference_m=0.002,
    )
    assert close["passed"] is True
    far = compare_planes(
        [0, 1, 0, -0.02],
        [0, 1, 0, 0],
        [0.2, 0.7],
        maximum_angle_deg=0.1,
        maximum_height_difference_m=0.002,
    )
    assert far["passed"] is False


def test_rewrite_generated_metadata_updates_paths_and_calibration_hash(tmp_path: Path):
    staging = tmp_path / "stage"
    destination = tmp_path / "final"
    staging.mkdir()
    calibration_path = staging / "scene_calibration_v2.json"
    calibration_path.write_text(json.dumps(table_aligned_calibration()) + "\n")
    conversion_path = staging / "conversion_metadata.json"
    conversion_path.write_text(
        json.dumps({"pointclouds_path": str(staging / "pointclouds.pt")}) + "\n"
    )
    sequence_path = staging / "sequence_metadata.json"
    sequence_path.write_text(
        json.dumps(
            {
                "input_dir": str(staging),
                "calibration": {
                    "status": "available",
                    "source_conversion_metadata_path": str(conversion_path),
                    "source_conversion_metadata_sha256": "old",
                },
            }
        )
        + "\n"
    )
    rewrite_generated_metadata(staging, destination)
    conversion = json.loads(conversion_path.read_text())
    sequence = json.loads(sequence_path.read_text())
    assert conversion["pointclouds_path"] == str(destination / "pointclouds.pt")
    assert conversion["calibration"]["scene_frame"] == "table-aligned"
    assert sequence["input_dir"] == str(destination)
    assert sequence["calibration"]["source_conversion_metadata_path"] == str(
        destination / "conversion_metadata.json"
    )
    assert sequence["calibration"]["source_conversion_metadata_sha256"] != "old"


def test_build_commands_pin_mocap_parent_and_both_tools(tmp_path: Path):
    commands = build_commands(
        tmp_path / "bag", tmp_path / "stage", tmp_path / "geometry.json"
    )
    export = commands[0]
    assert export[export.index("--output-frame") + 1] == "mocap"
    assert export[export.index("--camera-tag-parent-frame") + 1] == "tag16h5:3"
    pose_index = export.index("--pose-frames")
    assert tuple(export[pose_index + 1 : pose_index + 3]) == POSE_FRAMES
    assert "--require-all-streams" in commands[1]


def test_prepare_publishes_only_after_all_checks(tmp_path: Path):
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("metadata\n")
    (bag / "episode.db3").write_bytes(b"tag16h5:3 tag16h5:4")
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "sample").write_bytes(b"x")
    geometry = tmp_path / "tool_geometry.json"
    geometry.write_text("{}\n")
    output = tmp_path / "episode20"

    def fake_run(command, cwd, log_path):
        del cwd
        stage = log_path.parent
        log_path.write_text("ok\n")
        if "--output-frame" in command:
            (stage / "pointclouds.pt").write_bytes(b"raw points")
            (stage / "paths.pt").write_bytes(b"raw paths")
            (stage / "conversion_metadata.json").write_text("{}\n")
            (stage / "scene_calibration_v2.json").write_text(
                json.dumps(raw_mocap_calibration()) + "\n"
            )
        elif "--timeline" in command:
            (stage / "pointclouds_interpolated.pt").write_bytes(b"points")
            (stage / "paths_interpolated.pt").write_bytes(b"paths")
            (stage / "sequence_metadata.json").write_text("{}\n")
        else:
            calibration_path = stage / "scene_calibration_v2.json"
            report = {
                "passed": True,
                "source_mocap_calibration_sha256": sha256_file(calibration_path),
                "estimated_plane_scene": [0.08, 0.996, 0.01, -0.05],
                "recommended_translation_xz_m": [0.4, 0.3],
            }
            (stage / "episode20_table_plane_validation.json").write_text(
                json.dumps(report) + "\n"
            )

    with (
        patch(
            "experiments.differentiable_mpm.prepare_episode20_scene_calibration.run_command",
            side_effect=fake_run,
        ),
        patch(
            "experiments.differentiable_mpm.prepare_episode20_scene_calibration.validate_calibration",
            return_value={"inlier_count": 10},
        ),
        patch(
            "experiments.differentiable_mpm.prepare_episode20_scene_calibration.validate_tensors",
            return_value={"raw_pointcloud_frames": 250},
        ),
    ):
        assert (
            prepare(
                bag,
                output,
                reference=reference,
                tool_geometry=geometry,
                reserve_bytes=0,
            )
            == output
        )

    assert output.is_dir()
    assert (output / "scene_calibration_mocap_v2.json").is_file()
    assert (output / "episode20_export_validation.json").is_file()
    assert (output / "SHA256SUMS").is_file()
    assert not list(tmp_path.glob(".episode20.*"))
    with pytest.raises(FileExistsError):
        prepare(
            bag,
            output,
            reference=reference,
            tool_geometry=geometry,
            reserve_bytes=0,
        )


def test_prepare_removes_staging_after_failure(tmp_path: Path):
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("metadata\n")
    (bag / "episode.db3").write_bytes(b"tag16h5:3 tag16h5:4")
    reference = tmp_path / "reference"
    reference.mkdir()
    geometry = tmp_path / "tool_geometry.json"
    geometry.write_text("{}\n")
    output = tmp_path / "episode20"
    with patch(
        "experiments.differentiable_mpm.prepare_episode20_scene_calibration.run_command",
        side_effect=RuntimeError("failed"),
    ):
        with pytest.raises(RuntimeError, match="failed"):
            prepare(
                bag,
                output,
                reference=reference,
                tool_geometry=geometry,
                reserve_bytes=0,
            )
    assert not output.exists()
    assert not list(tmp_path.glob(".episode20.*"))


def test_write_report_is_finite_and_refuses_overwrite(tmp_path: Path):
    path = tmp_path / "report.json"
    write_report(path, {"passed": True, "value": 1.0})
    assert json.loads(path.read_text())["passed"] is True
    with pytest.raises(FileExistsError):
        write_report(path, {"passed": True})
    with pytest.raises(ValueError):
        write_report(tmp_path / "bad.json", {"value": float("nan")})
