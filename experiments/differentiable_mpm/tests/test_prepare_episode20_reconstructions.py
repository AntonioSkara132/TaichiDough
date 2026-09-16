from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import experiments.differentiable_mpm.prepare_episode20_reconstructions as module


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def table_calibration() -> dict[str, Any]:
    return {
        "schema": "taichidough/scene-calibration/v2",
        "source_frame": "mocap",
        "scene_frame": "table-aligned",
        "scene_from_source": [
            [1.0, 0.0, 0.0, 0.4],
            [0.0, 1.0, 0.0, -0.05],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "floor_plane_scene": [0.0, 1.0, 0.0, 0.0],
        "provenance": {
            "table_alignment": {
                "previous_scene_frame": "mocap",
                "source_tensors_transformed": False,
            }
        },
    }


def make_reconstruction(tmp_path: Path, minimum_y: float = -0.001) -> tuple[Path, Path]:
    frame = tmp_path / "temporary" / "chunk" / "frame_0000"
    frame.mkdir(parents=True)
    source = np.asarray([[0.1, 0.05, 0.2], [0.2, 0.06, 0.3]], dtype=np.float64)
    scene = np.asarray([[0.4, 0.02, 0.5], [0.5, 0.03, 0.6]], dtype=np.float64)
    particles = np.full((module.PARTICLE_COUNT, 3), [0.5, 0.01, 0.5], dtype=np.float32)
    particles[0, 1] = minimum_y
    arrays = {
        "source_filtered_points_xyz.npy": source,
        "scene_transformed_points_xyz.npy": scene,
        "real_points_xyz.npy": scene,
        "voxel_centers_xyz.npy": scene,
        "surface_particles_xyz.npy": scene,
        "sampled_particles_xyz.npy": particles,
    }
    for name, value in arrays.items():
        np.save(frame / name, value)
    content_hash = hashlib.sha256(
        np.ascontiguousarray(particles.astype(np.float32)).tobytes(order="C")
    ).hexdigest()
    metadata = {
        "schema": "voxel_dough_reconstruction/v2",
        "frame": 0,
        "pointclouds_name": "pointclouds_interpolated.pt",
        "sampled_particles": module.PARTICLE_COUNT,
        "object_volume_m3": 0.0001,
        "sampled_particles_sha256": content_hash,
        "fill_mode": "floor",
        "fill_axis": "y",
        "fill_direction": "negative",
        "calibration_schema": "taichidough/scene-calibration/v2",
        "calibration": {
            "source_frame": "mocap",
            "scene_frame": "table-aligned",
        },
        "array_frames": {
            "source_filtered_points_xyz": "mocap",
            "scene_transformed_points_xyz": "table-aligned",
            "real_points_xyz": "table-aligned",
            "voxel_centers_xyz": "table-aligned",
            "surface_particles_xyz": "table-aligned",
            "sampled_particles_xyz": "table-aligned",
        },
        "fill": {
            "segmentation_counts": {"kept": 100},
            "intersection": {"accepted": 20},
        },
        "outputs": {"sampled_particles_xyz": str(frame / "sampled_particles_xyz.npy")},
    }
    write_json(frame / "reconstruction_metadata.json", metadata)
    calibration = tmp_path / "calibration.json"
    write_json(calibration, table_calibration())
    return frame, calibration


def test_validate_calibration_requires_nonidentity_table_alignment(tmp_path: Path):
    path = tmp_path / "calibration.json"
    calibration = table_calibration()
    write_json(path, calibration)
    assert module.validate_calibration(path)["scene_frame"] == "table-aligned"
    calibration["scene_from_source"] = np.eye(4).tolist()
    write_json(path, calibration)
    with pytest.raises(ValueError, match="measured table alignment"):
        module.validate_calibration(path)


def test_reconstruction_command_uses_reviewed_settings(tmp_path: Path):
    command = module.reconstruction_command(
        "python",
        tmp_path / "reconstruct.py",
        tmp_path / "recording",
        tmp_path / "output",
        tmp_path / "calibration.json",
    )
    assert command[command.index("--pointclouds-name") + 1] == "pointclouds_interpolated.pt"
    assert command[command.index("--num-particles") + 1] == "24000"
    assert command[command.index("--voxel-size") + 1] == "0.003"
    assert command[command.index("--fill-direction") + 1] == "negative"
    assert command[command.index("--seed") + 1] == "0"


def test_validate_reconstruction_accepts_half_voxel_floor_tolerance(tmp_path: Path):
    frame, calibration = make_reconstruction(tmp_path, minimum_y=-0.001)
    final = tmp_path / "final" / "chunk" / "frame_0000"

    checked = module.validate_reconstruction(frame, calibration, final)

    assert checked["sampled_particles"] == module.PARTICLE_COUNT
    assert checked["minimum_floor_distance_m"] == pytest.approx(-0.001)
    metadata = json.loads((frame / "reconstruction_metadata.json").read_text())
    assert metadata["outputs"]["sampled_particles_xyz"].startswith(str(final))


def test_validate_reconstruction_rejects_excessive_floor_penetration(tmp_path: Path):
    frame, calibration = make_reconstruction(tmp_path, minimum_y=-0.002)
    with pytest.raises(ValueError, match="too far below"):
        module.validate_reconstruction(
            frame,
            calibration,
            tmp_path / "final" / "chunk" / "frame_0000",
        )


def test_validate_reconstruction_rejects_horizontal_domain_escape(tmp_path: Path):
    frame, calibration = make_reconstruction(tmp_path)
    particles_path = frame / "sampled_particles_xyz.npy"
    particles = np.load(particles_path, allow_pickle=False)
    particles[0, 0] = -0.01
    np.save(particles_path, particles)
    with pytest.raises(ValueError, match="horizontal MPM domain"):
        module.validate_reconstruction(
            frame,
            calibration,
            tmp_path / "final" / "chunk" / "frame_0000",
        )
