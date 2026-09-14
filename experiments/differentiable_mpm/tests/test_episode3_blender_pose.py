# pyright: reportMissingImports=false
from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.differentiable_mpm.episode3_tool_identity import hsv_mask
from experiments.differentiable_mpm.prepare_episode3_blender_pose import (
    BLENDER_FROM_SCENE,
    TOOL_HSV,
    deterministic_sample,
    pose_matrix,
    transform_points,
    validate_rigid,
    write_binary_ply,
)


def read_test_ply(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii")
    count = int(next(row.split()[-1] for row in header.splitlines() if row.startswith("element vertex ")))
    return np.asarray(list(struct.iter_unpack("<fff", raw[end:])), dtype=np.float32).reshape(count, 3)


def test_blender_axis_conversion_is_a_proper_rotation():
    validate_rigid(BLENDER_FROM_SCENE, name="blender_from_scene")
    np.testing.assert_allclose(
        transform_points(np.array([[0, 1, 0]]), BLENDER_FROM_SCENE), [[0, 0, 1]]
    )
    np.testing.assert_allclose(
        transform_points(np.array([[0, 0, 1]]), BLENDER_FROM_SCENE), [[0, -1, 0]]
    )
    assert np.linalg.det(BLENDER_FROM_SCENE[:3, :3]) == pytest.approx(1.0)


def test_full_pose_and_marker_relative_candidate_recovery():
    marker_pose = np.r_[np.array([0.2, -0.1, 0.6]), Rotation.from_euler("xyz", [10, 20, -30], degrees=True).as_quat()]
    marker = pose_matrix(marker_pose)
    candidate = np.eye(4)
    candidate[:3, :3] = Rotation.from_euler("xyz", [4, -3, 7], degrees=True).as_matrix()
    candidate[:3, 3] = [0.012, -0.008, 0.021]
    world = marker @ candidate
    recovered = np.linalg.inv(marker) @ world
    validate_rigid(recovered, name="candidate")
    np.testing.assert_allclose(recovered, candidate, atol=1e-12)


def test_requested_tool_hsv_selects_bright_red_only():
    rgb = np.array([[255, 0, 0], [128, 128, 128], [80, 0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(hsv_mask(rgb, TOOL_HSV), [True, False, False])


def test_binary_ply_round_trip(tmp_path):
    points = np.array([[0.1, 0.2, 0.3], [-1.0, 2.0, 4.0]], dtype=np.float32)
    path = tmp_path / "points.ply"
    write_binary_ply(path, points)
    np.testing.assert_array_equal(read_test_ply(path), points)


def test_sampling_is_deterministic_and_rigid_validation_rejects_scale():
    points = np.arange(300, dtype=np.float64).reshape(100, 3)
    first = deterministic_sample(points, 12, 5)
    second = deterministic_sample(points, 12, 5)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 12
    scaled = np.eye(4)
    scaled[0, 0] = 1.1
    with pytest.raises(ValueError, match="orthonormal"):
        validate_rigid(scaled, name="scaled")
