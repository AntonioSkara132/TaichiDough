#!/usr/bin/env python3
"""Tests for the ROS-independent AprilTag calibration calculations."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np

SCRIPT = Path(__file__).with_name("collect_apriltag_scene_calibration.py")
MODULE_SPEC = importlib.util.spec_from_file_location("collect_apriltag_scene_calibration", SCRIPT)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Cannot load {SCRIPT}")
COLLECTOR = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = COLLECTOR
MODULE_SPEC.loader.exec_module(COLLECTOR)

AprilTagObservation = COLLECTOR.AprilTagObservation
UnsupportedDetectionMessageError = COLLECTOR.UnsupportedDetectionMessageError
april_tag_rejection_reason = COLLECTOR.april_tag_rejection_reason
build_calibration_document = COLLECTOR.build_calibration_document
compose_transforms = COLLECTOR.compose_transforms
extract_apriltag_detection = COLLECTOR.extract_apriltag_detection
pose_to_matrix = COLLECTOR.pose_to_matrix
rigid_inverse = COLLECTOR.rigid_inverse
robust_se3_aggregate = COLLECTOR.robust_se3_aggregate


def ns(**values):
    return SimpleNamespace(**values)


def stamp(seconds: float):
    whole = int(seconds)
    return ns(sec=whole, nanosec=int(round((seconds - whole) * 1e9)))


def detection(
    *,
    family: str = "tag36h11",
    tag_id: int = 7,
    hamming: int = 0,
    decision_margin: float = 80.0,
    size: float = 0.04,
    stamp_s: float = 12.0,
    frame_id: str = "camera_optical",
):
    pose = ns(
        position=ns(x=0.1, y=-0.2, z=0.8),
        orientation=ns(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return ns(
        family=family,
        id=tag_id,
        hamming=hamming,
        decision_margin=decision_margin,
        size=size,
        pose=ns(header=ns(stamp=stamp(stamp_s), frame_id=frame_id), pose=ns(pose=pose)),
    )


def yaw_transform(angle_deg: float, translation: tuple[float, float, float]) -> np.ndarray:
    angle = np.radians(angle_deg)
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0, translation[0]],
            [np.sin(angle), np.cos(angle), 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


class AprilTagSceneCalibrationTests(unittest.TestCase):
    def test_transform_composition_matches_scene_camera_equation(self) -> None:
        scene_from_tag = yaw_transform(90.0, (1.0, 0.0, 0.0))
        camera_from_tag = yaw_transform(20.0, (0.1, -0.2, 0.8))
        scene_from_camera = compose_transforms(scene_from_tag, rigid_inverse(camera_from_tag))

        np.testing.assert_allclose(
            compose_transforms(scene_from_camera, camera_from_tag), scene_from_tag, atol=1e-9
        )

    def test_detection_extraction_supports_nested_pose(self) -> None:
        observation = extract_apriltag_detection(detection())

        self.assertEqual(observation.family, "tag36h11")
        self.assertEqual(observation.tag_id, 7)
        self.assertEqual(observation.frame_id, "camera_optical")
        self.assertEqual(observation.stamp_s, 12.0)
        np.testing.assert_allclose(observation.camera_from_tag[:3, 3], [0.1, -0.2, 0.8])

    def test_unsupported_detection_layout_fails_clearly(self) -> None:
        old_layout = detection()
        del old_layout.hamming
        with self.assertRaisesRegex(UnsupportedDetectionMessageError, "hamming"):
            extract_apriltag_detection(old_layout)

    def test_wrong_tag_and_quality_are_rejected(self) -> None:
        camera_from_tag = np.eye(4)
        camera_from_tag[2, 3] = 0.8
        base = AprilTagObservation(
            family="tag36h11",
            tag_id=7,
            hamming=0,
            decision_margin=80.0,
            edge_size_m=0.04,
            stamp_s=12.0,
            frame_id="camera_optical",
            camera_from_tag=camera_from_tag,
        )
        common = dict(
            expected_family="tag36h11",
            expected_id=7,
            expected_edge_size_m=0.04,
            expected_camera_frame="camera_optical",
            max_hamming=0,
            min_decision_margin=30.0,
        )
        wrong_id = AprilTagObservation(**{**base.__dict__, "tag_id": 8})
        bad_hamming = AprilTagObservation(**{**base.__dict__, "hamming": 1})
        low_margin = AprilTagObservation(**{**base.__dict__, "decision_margin": 5.0})

        self.assertIn("tag id", april_tag_rejection_reason(wrong_id, **common))
        self.assertIn("hamming", april_tag_rejection_reason(bad_hamming, **common))
        self.assertIn("decision margin", april_tag_rejection_reason(low_margin, **common))
        self.assertIsNone(april_tag_rejection_reason(base, **common))

    def test_robust_aggregation_removes_translation_and_rotation_outlier(self) -> None:
        transforms = [
            yaw_transform(0.0, (0.100, 0.200, 0.800)),
            yaw_transform(0.2, (0.101, 0.199, 0.801)),
            yaw_transform(-0.2, (0.099, 0.201, 0.799)),
            yaw_transform(0.1, (0.100, 0.201, 0.800)),
            yaw_transform(60.0, (1.5, -1.0, 0.2)),
        ]
        aggregate, diagnostics = robust_se3_aggregate(
            transforms,
            max_translation_residual_m=0.02,
            max_rotation_residual_deg=3.0,
            min_inliers=3,
            return_diagnostics=True,
        )

        self.assertEqual(diagnostics["inlier_count"], 4)
        self.assertNotIn(4, diagnostics["inlier_indices"])
        np.testing.assert_allclose(aggregate[:3, 3], [0.1, 0.20025, 0.8], atol=1e-5)
        np.testing.assert_allclose(aggregate[:3, :3], np.eye(3), atol=2e-3)

    def test_document_computes_scene_from_camera_and_source(self) -> None:
        scene_from_tag = yaw_transform(30.0, (0.5, 0.1, 0.2))
        camera_from_tag = pose_to_matrix([0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0])
        camera_from_source = yaw_transform(0.0, (0.01, 0.02, 0.03))
        camera = {
            "width": 640,
            "height": 480,
            "fx": 600.0,
            "fy": 605.0,
            "cx": 319.5,
            "cy": 239.5,
            "zNear": 0.01,
            "zFar": 4.0,
        }
        document = build_calibration_document(
            name="test",
            source_frame="depth_optical",
            scene_frame="scene",
            camera=camera,
            scene_from_tag=scene_from_tag,
            camera_from_tag_observations=[camera_from_tag] * 3,
            camera_from_source=camera_from_source,
            floor_plane_scene=np.array([0.0, 2.0, 0.0, -0.4]),
            aggregation_options={"min_inliers": 3},
            provenance={"test": True},
            fingerprints={"input": "abc"},
        )
        scene_from_camera = np.asarray(document["scene_from_camera"])
        scene_from_source = np.asarray(document["scene_from_source"])

        np.testing.assert_allclose(scene_from_camera @ camera_from_tag, scene_from_tag, atol=1e-9)
        np.testing.assert_allclose(scene_from_source, scene_from_camera @ camera_from_source, atol=1e-9)
        np.testing.assert_allclose(document["floor_plane_scene"], [0.0, 1.0, 0.0, -0.2])

    def test_help_runs_without_ros(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--tag-family", result.stdout)
        self.assertNotIn("No module named", result.stderr)


if __name__ == "__main__":
    unittest.main()
