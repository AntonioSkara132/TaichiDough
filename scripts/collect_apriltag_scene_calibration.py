#!/usr/bin/env python3
"""Collect a metric scene calibration from one fixed AprilTag and ROS 2 transforms."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from deformpath_topview import (
        SCENE_CALIBRATION_V2,
        compose_rigid_transforms,
        rigid_inverse,
    )
except ImportError:
    from .deformpath_topview import (
        SCENE_CALIBRATION_V2,
        compose_rigid_transforms,
        rigid_inverse,
    )


class UnsupportedDetectionMessageError(ValueError):
    """The AprilTag message does not expose the fields required for calibration."""


@dataclass(frozen=True)
class AprilTagObservation:
    family: str
    tag_id: int
    hamming: int
    decision_margin: float
    edge_size_m: float | None
    stamp_s: float
    frame_id: str
    camera_from_tag: np.ndarray


@dataclass(frozen=True)
class TransformRecord:
    target_frame: str
    source_frame: str
    target_from_source: np.ndarray
    stamp_s: float
    is_static: bool


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalize_frame(frame_id: str) -> str:
    result = str(frame_id).strip().lstrip("/")
    if not result:
        raise ValueError("frame id must be nonempty")
    return result


def _stamp_seconds(stamp: Any) -> float:
    if stamp is None:
        raise UnsupportedDetectionMessageError("AprilTag detection has no timestamp")
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return float(stamp.secs) + float(stamp.nsecs) * 1e-9
    if isinstance(stamp, (int, float, np.number)):
        return float(stamp)
    raise UnsupportedDetectionMessageError("AprilTag timestamp uses an unsupported layout")


def _one_value(value: Any, field: str) -> Any:
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Sequence):
        if len(value) != 1:
            raise UnsupportedDetectionMessageError(f"AprilTag {field} must contain one value")
        return value[0]
    return value


def quaternion_to_matrix(quaternion_xyzw: Any) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("quaternion norm must be nonzero")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def pose_to_matrix(position: Sequence[float], quaternion_xyzw: Sequence[float]) -> np.ndarray:
    translation = np.asarray(position, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("pose position must contain three finite values")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_to_matrix(quaternion_xyzw)
    result[:3, 3] = translation
    return result


def transform_message_to_matrix(transform: Any) -> np.ndarray:
    translation = transform.translation
    rotation = transform.rotation
    return pose_to_matrix(
        [translation.x, translation.y, translation.z],
        [rotation.x, rotation.y, rotation.z, rotation.w],
    )


def compose_transforms(*transforms: np.ndarray) -> np.ndarray:
    """Compose rigid transforms in ordinary matrix multiplication order."""
    return compose_rigid_transforms(*transforms)


def rotation_geodesic_angle(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = np.asarray(rotation_a, dtype=np.float64).T @ np.asarray(rotation_b, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def mean_rotation(rotations: np.ndarray) -> np.ndarray:
    values = np.asarray(rotations, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or len(values) == 0:
        raise ValueError("rotations must have shape [N, 3, 3] with N > 0")
    accumulator = np.zeros((4, 4), dtype=np.float64)
    for rotation in values:
        quaternion = matrix_to_quaternion(rotation)
        accumulator += np.outer(quaternion, quaternion)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    quaternion = eigenvectors[:, int(np.argmax(eigenvalues))]
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion_to_matrix(quaternion)


def _mad_threshold(residuals: np.ndarray, hard_limit: float, scale: float) -> float:
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    robust_limit = median + scale * 1.4826 * mad
    return float(min(float(hard_limit), max(robust_limit, float(np.finfo(np.float64).eps))))


def robust_se3_aggregate(
    transforms: Iterable[np.ndarray],
    max_translation_residual_m: float = 0.03,
    max_rotation_residual_deg: float = 5.0,
    mad_scale: float = 3.5,
    min_inliers: int = 3,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Average rigid transforms after translation and geodesic rotation rejection."""
    values = np.asarray(list(transforms), dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4) or len(values) == 0:
        raise ValueError("transforms must have shape [N, 4, 4] with N > 0")
    if max_translation_residual_m <= 0.0 or max_rotation_residual_deg <= 0.0:
        raise ValueError("residual limits must be positive")
    if mad_scale <= 0.0 or min_inliers <= 0:
        raise ValueError("mad_scale and min_inliers must be positive")
    for transform in values:
        compose_rigid_transforms(transform)

    translations = values[:, :3, 3]
    translation_center = np.median(translations, axis=0)
    translation_residuals = np.linalg.norm(translations - translation_center, axis=1)

    rotations = values[:, :3, :3]
    pairwise = np.zeros((len(values), len(values)), dtype=np.float64)
    for row in range(len(values)):
        for column in range(row + 1, len(values)):
            angle = rotation_geodesic_angle(rotations[row], rotations[column])
            pairwise[row, column] = pairwise[column, row] = angle
    rotation_seed = rotations[int(np.argmin(np.median(pairwise, axis=1)))]
    rotation_residuals = np.array(
        [rotation_geodesic_angle(rotation_seed, rotation) for rotation in rotations], dtype=np.float64
    )

    translation_limit = _mad_threshold(translation_residuals, max_translation_residual_m, mad_scale)
    rotation_hard_limit = math.radians(max_rotation_residual_deg)
    rotation_limit = _mad_threshold(rotation_residuals, rotation_hard_limit, mad_scale)
    inliers = (translation_residuals <= translation_limit) & (rotation_residuals <= rotation_limit)
    if int(inliers.sum()) < min_inliers:
        raise ValueError(
            f"Only {int(inliers.sum())} of {len(values)} transforms pass robust filtering; "
            f"at least {min_inliers} are required"
        )

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = mean_rotation(rotations[inliers])
    result[:3, 3] = np.mean(translations[inliers], axis=0)
    final_translation_residuals = np.linalg.norm(translations - result[:3, 3], axis=1)
    final_rotation_residuals = np.array(
        [rotation_geodesic_angle(result[:3, :3], rotation) for rotation in rotations], dtype=np.float64
    )
    final_inliers = (
        inliers
        & (final_translation_residuals <= max_translation_residual_m)
        & (final_rotation_residuals <= rotation_hard_limit)
    )
    if int(final_inliers.sum()) < min_inliers:
        raise ValueError(
            f"Only {int(final_inliers.sum())} transforms remain after final residual checks; "
            f"at least {min_inliers} are required"
        )
    if not np.array_equal(final_inliers, inliers):
        result[:3, :3] = mean_rotation(rotations[final_inliers])
        result[:3, 3] = np.mean(translations[final_inliers], axis=0)

    diagnostics = {
        "sample_count": int(len(values)),
        "inlier_count": int(final_inliers.sum()),
        "inlier_indices": np.flatnonzero(final_inliers).astype(int).tolist(),
        "translation_limit_m": float(translation_limit),
        "rotation_limit_deg": float(math.degrees(rotation_limit)),
        "translation_residual_m": final_translation_residuals.tolist(),
        "rotation_residual_deg": np.degrees(final_rotation_residuals).tolist(),
    }
    if return_diagnostics:
        return result, diagnostics
    return result


def _pose_and_header(detection: Any, default_header: Any | None) -> tuple[Any, Any]:
    if not hasattr(detection, "pose"):
        raise UnsupportedDetectionMessageError("AprilTag detection has no pose field")
    pose = detection.pose
    header = getattr(pose, "header", None) or default_header
    for _ in range(4):
        if hasattr(pose, "position") and hasattr(pose, "orientation"):
            return pose, header
        if hasattr(pose, "header"):
            header = pose.header
        if not hasattr(pose, "pose"):
            break
        pose = pose.pose
    raise UnsupportedDetectionMessageError("AprilTag pose uses an unsupported layout")


def extract_apriltag_detection(detection: Any, default_header: Any | None = None) -> AprilTagObservation:
    """Extract fields common to supported apriltag_msgs detection versions."""
    required = ("family", "id", "hamming", "decision_margin")
    missing = [field for field in required if not hasattr(detection, field)]
    if missing:
        raise UnsupportedDetectionMessageError(
            "AprilTag detection is missing fields needed for quality checks: " + ", ".join(missing)
        )
    family_value = _one_value(detection.family, "family")
    if isinstance(family_value, bytes):
        family_value = family_value.decode("ascii")
    family = str(family_value)
    tag_id = int(_one_value(detection.id, "id"))
    hamming = int(_one_value(detection.hamming, "hamming"))
    decision_margin = float(_one_value(detection.decision_margin, "decision_margin"))
    if not np.isfinite(decision_margin):
        raise ValueError("AprilTag decision margin must be finite")

    edge_size = None
    if hasattr(detection, "size"):
        edge_size = float(_one_value(detection.size, "size"))
        if not np.isfinite(edge_size) or edge_size <= 0.0:
            raise ValueError("AprilTag size must be positive and finite")

    pose, header = _pose_and_header(detection, default_header)
    if header is None or not hasattr(header, "stamp") or not hasattr(header, "frame_id"):
        raise UnsupportedDetectionMessageError("AprilTag pose has no stamped frame")
    position = pose.position
    orientation = pose.orientation
    camera_from_tag = pose_to_matrix(
        [position.x, position.y, position.z],
        [orientation.x, orientation.y, orientation.z, orientation.w],
    )
    return AprilTagObservation(
        family=family,
        tag_id=tag_id,
        hamming=hamming,
        decision_margin=decision_margin,
        edge_size_m=edge_size,
        stamp_s=_stamp_seconds(header.stamp),
        frame_id=_normalize_frame(header.frame_id),
        camera_from_tag=camera_from_tag,
    )


def april_tag_rejection_reason(
    observation: AprilTagObservation,
    expected_family: str,
    expected_id: int,
    expected_edge_size_m: float,
    expected_camera_frame: str,
    max_hamming: int,
    min_decision_margin: float,
    camera_info_stamp_s: float | None = None,
    pointcloud_stamp_s: float | None = None,
    max_sync_offset_s: float = 0.1,
    now_s: float | None = None,
    max_age_s: float = 1.0,
) -> str | None:
    if observation.family != expected_family:
        return f"family {observation.family!r} does not match {expected_family!r}"
    if observation.tag_id != expected_id:
        return f"tag id {observation.tag_id} does not match {expected_id}"
    if observation.hamming > max_hamming:
        return f"hamming {observation.hamming} exceeds {max_hamming}"
    if observation.decision_margin < min_decision_margin:
        return f"decision margin {observation.decision_margin:.6g} is below {min_decision_margin:.6g}"
    if observation.frame_id != _normalize_frame(expected_camera_frame):
        return f"pose frame {observation.frame_id!r} does not match camera frame {_normalize_frame(expected_camera_frame)!r}"
    if observation.edge_size_m is not None and not math.isclose(
        observation.edge_size_m, expected_edge_size_m, rel_tol=1e-4, abs_tol=1e-6
    ):
        return f"tag size {observation.edge_size_m:.6g} does not match {expected_edge_size_m:.6g} m"
    if observation.camera_from_tag[2, 3] <= 0.0:
        return "tag pose must be in front of the camera"
    for label, stamp in (("CameraInfo", camera_info_stamp_s), ("point cloud", pointcloud_stamp_s)):
        if stamp is not None and abs(observation.stamp_s - stamp) > max_sync_offset_s:
            return f"{label} timestamp differs by more than {max_sync_offset_s:.6g} s"
    if now_s is not None and now_s - observation.stamp_s > max_age_s:
        return f"detection is older than {max_age_s:.6g} s"
    if observation.stamp_s <= 0.0:
        return "detection timestamp must be positive"
    return None


class TransformGraph:
    """Small latest-transform graph for /tf and /tf_static messages."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], TransformRecord] = {}

    def add(self, record: TransformRecord) -> None:
        target = _normalize_frame(record.target_frame)
        source = _normalize_frame(record.source_frame)
        matrix = compose_rigid_transforms(record.target_from_source)
        key = (target, source)
        previous = self._records.get(key)
        if previous is None or record.is_static or record.stamp_s >= previous.stamp_s:
            self._records[key] = TransformRecord(target, source, matrix, record.stamp_s, record.is_static)

    def add_message(self, message: Any, is_static: bool) -> None:
        for stamped in message.transforms:
            self.add(
                TransformRecord(
                    target_frame=stamped.header.frame_id,
                    source_frame=stamped.child_frame_id,
                    target_from_source=transform_message_to_matrix(stamped.transform),
                    stamp_s=_stamp_seconds(stamped.header.stamp),
                    is_static=is_static,
                )
            )

    def lookup(self, target_frame: str, source_frame: str) -> tuple[np.ndarray, list[TransformRecord]]:
        target = _normalize_frame(target_frame)
        source = _normalize_frame(source_frame)
        if target == source:
            return np.eye(4, dtype=np.float64), []
        adjacency: dict[str, list[tuple[str, np.ndarray, TransformRecord]]] = {}
        for (parent, child), record in self._records.items():
            adjacency.setdefault(child, []).append((parent, record.target_from_source, record))
            adjacency.setdefault(parent, []).append((child, rigid_inverse(record.target_from_source), record))
        queue: list[tuple[str, np.ndarray, list[TransformRecord]]] = [(source, np.eye(4), [])]
        visited = {source}
        while queue:
            frame, frame_from_source, path = queue.pop(0)
            for neighbor, neighbor_from_frame, record in adjacency.get(frame, []):
                if neighbor in visited:
                    continue
                neighbor_from_source = neighbor_from_frame @ frame_from_source
                next_path = path + [record]
                if neighbor == target:
                    return neighbor_from_source, next_path
                visited.add(neighbor)
                queue.append((neighbor, neighbor_from_source, next_path))
        raise KeyError(f"No TF path from {source!r} to {target!r}")


def _header_data(header: Any) -> dict[str, Any]:
    return {"frame_id": _normalize_frame(header.frame_id), "stamp_s": _stamp_seconds(header.stamp)}


def camera_info_data(message: Any, z_near: float, z_far: float) -> dict[str, Any]:
    if int(message.width) <= 0 or int(message.height) <= 0 or len(message.k) != 9:
        raise ValueError("CameraInfo must have positive dimensions and a 3x3 K matrix")
    values = {
        "width": int(message.width),
        "height": int(message.height),
        "fx": float(message.k[0]),
        "fy": float(message.k[4]),
        "cx": float(message.k[2]),
        "cy": float(message.k[5]),
        "zNear": float(z_near),
        "zFar": float(z_far),
    }
    if not np.isfinite(list(values.values())).all() or values["fx"] <= 0.0 or values["fy"] <= 0.0:
        raise ValueError("CameraInfo intrinsics must be finite with positive fx and fy")
    if values["zNear"] < 0.0 or values["zFar"] <= values["zNear"]:
        raise ValueError("camera depth range must satisfy 0 <= zNear < zFar")
    return values


def build_calibration_document(
    *,
    name: str,
    source_frame: str,
    scene_frame: str,
    camera: dict[str, Any],
    scene_from_tag: np.ndarray,
    camera_from_tag_observations: Iterable[np.ndarray],
    camera_from_source: np.ndarray,
    floor_plane_scene: np.ndarray | None,
    aggregation_options: dict[str, Any],
    provenance: dict[str, Any],
    fingerprints: dict[str, Any],
) -> dict[str, Any]:
    camera_from_tag, diagnostics = robust_se3_aggregate(
        camera_from_tag_observations, return_diagnostics=True, **aggregation_options
    )
    scene_from_camera = compose_transforms(scene_from_tag, rigid_inverse(camera_from_tag))
    scene_from_source = compose_transforms(scene_from_camera, camera_from_source)
    floor = None
    if floor_plane_scene is not None:
        floor_array = np.asarray(floor_plane_scene, dtype=np.float64)
        if floor_array.shape != (4,) or not np.isfinite(floor_array).all():
            raise ValueError("floor_plane_scene must contain four finite values")
        norm = float(np.linalg.norm(floor_array[:3]))
        if norm <= 0.0:
            raise ValueError("floor_plane_scene normal must be nonzero")
        floor = (floor_array / norm).tolist()
    return {
        "schema": SCENE_CALIBRATION_V2,
        "name": name,
        "source_frame": _normalize_frame(source_frame),
        "scene_frame": _normalize_frame(scene_frame),
        "scene_from_source": scene_from_source.tolist(),
        "scene_from_camera": scene_from_camera.tolist(),
        "floor_plane_scene": floor,
        "camera": camera,
        "provenance": provenance,
        "fingerprints": fingerprints,
        "diagnostics": {"camera_from_tag_aggregation": diagnostics},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output v2 calibration JSON")
    parser.add_argument("--raw-log", type=Path, help="Raw JSON Lines observation log")
    parser.add_argument("--name", default="apriltag-scene-calibration")
    parser.add_argument("--detections-topic", default="/apriltag/detections")
    parser.add_argument("--detection-type", default="apriltag_msgs.msg.AprilTagDetectionArray")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--pointcloud-topic", default="/camera/depth/color/points")
    parser.add_argument("--tf-topic", default="/tf")
    parser.add_argument("--tf-static-topic", default="/tf_static")
    parser.add_argument("--tag-family", required=True)
    parser.add_argument("--tag-id", type=int, required=True)
    parser.add_argument("--tag-edge-size-m", type=float, required=True)
    parser.add_argument(
        "--scene-from-tag",
        type=float,
        nargs=16,
        required=True,
        metavar="M",
        help="Row-major 4x4 transform from tag coordinates to scene coordinates",
    )
    parser.add_argument("--scene-frame", default="scene")
    parser.add_argument("--floor-plane-scene", type=float, nargs=4, metavar=("NX", "NY", "NZ", "D"))
    parser.add_argument("--z-near", type=float, default=0.01)
    parser.add_argument("--z-far", type=float, default=5.0)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--min-inliers", type=int, default=10)
    parser.add_argument("--max-hamming", type=int, default=0)
    parser.add_argument("--min-decision-margin", type=float, default=30.0)
    parser.add_argument("--max-sync-offset-s", type=float, default=0.1)
    parser.add_argument("--max-age-s", type=float, default=1.0)
    parser.add_argument("--max-translation-residual-m", type=float, default=0.03)
    parser.add_argument("--max-rotation-residual-deg", type=float, default=5.0)
    parser.add_argument("--mad-scale", type=float, default=3.5)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    return parser


def _import_ros(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, PointCloud2
        from tf2_msgs.msg import TFMessage
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 collection requires rclpy, sensor_msgs, and tf2_msgs; --help and pure geometry do not"
        ) from exc
    try:
        module_name, class_name = args.detection_type.rsplit(".", 1)
        detection_type = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise RuntimeError(f"Cannot import AprilTag detection type {args.detection_type!r}") from exc
    return {
        "rclpy": rclpy,
        "Node": Node,
        "QoSProfile": QoSProfile,
        "DurabilityPolicy": DurabilityPolicy,
        "ReliabilityPolicy": ReliabilityPolicy,
        "sensor_qos": qos_profile_sensor_data,
        "CameraInfo": CameraInfo,
        "PointCloud2": PointCloud2,
        "TFMessage": TFMessage,
        "DetectionArray": detection_type,
    }


def _observation_log_entry(observation: AprilTagObservation) -> dict[str, Any]:
    return {
        "family": observation.family,
        "tag_id": observation.tag_id,
        "hamming": observation.hamming,
        "decision_margin": observation.decision_margin,
        "edge_size_m": observation.edge_size_m,
        "stamp_s": observation.stamp_s,
        "frame_id": observation.frame_id,
        "camera_from_tag": observation.camera_from_tag.tolist(),
    }


def run_ros_collection(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ros = _import_ros(args)
    scene_from_tag = compose_rigid_transforms(np.asarray(args.scene_from_tag, dtype=np.float64).reshape(4, 4))
    if args.samples <= 0 or args.min_inliers <= 0 or args.min_inliers > args.samples:
        raise ValueError("samples must be positive and min-inliers must be between 1 and samples")
    if args.tag_edge_size_m <= 0.0:
        raise ValueError("tag-edge-size-m must be positive")

    Node = ros["Node"]

    class CollectorNode(Node):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__("apriltag_scene_calibration_collector")
            self.camera_info: Any | None = None
            self.pointcloud_header: Any | None = None
            self.graph = TransformGraph()
            self.observations: list[AprilTagObservation] = []
            self.log: list[dict[str, Any]] = []
            self.seen: set[tuple[float, int]] = set()
            self.fatal_error: str | None = None
            qos = ros["sensor_qos"]
            tf_qos = ros["QoSProfile"](depth=100)
            static_qos = ros["QoSProfile"](
                depth=100,
                durability=ros["DurabilityPolicy"].TRANSIENT_LOCAL,
                reliability=ros["ReliabilityPolicy"].RELIABLE,
            )
            self.create_subscription(ros["CameraInfo"], args.camera_info_topic, self._camera, qos)
            self.create_subscription(ros["PointCloud2"], args.pointcloud_topic, self._pointcloud, qos)
            self.create_subscription(ros["TFMessage"], args.tf_topic, self._tf, tf_qos)
            self.create_subscription(ros["TFMessage"], args.tf_static_topic, self._tf_static, static_qos)
            self.create_subscription(ros["DetectionArray"], args.detections_topic, self._detections, qos)

        def _camera(self, message: Any) -> None:
            camera_info_data(message, args.z_near, args.z_far)
            self.camera_info = message

        def _pointcloud(self, message: Any) -> None:
            _header_data(message.header)
            self.pointcloud_header = message.header

        def _tf(self, message: Any) -> None:
            self.graph.add_message(message, is_static=False)

        def _tf_static(self, message: Any) -> None:
            self.graph.add_message(message, is_static=True)

        def _detections(self, message: Any) -> None:
            detections = getattr(message, "detections", None)
            if detections is None:
                self.log.append({"accepted": False, "reason": "detection array has no detections field"})
                return
            default_header = getattr(message, "header", None)
            for detection in detections:
                entry: dict[str, Any]
                try:
                    observation = extract_apriltag_detection(detection, default_header)
                    entry = _observation_log_entry(observation)
                    if self.camera_info is None:
                        reason = "CameraInfo has not arrived"
                    elif self.pointcloud_header is None:
                        reason = "point-cloud header has not arrived"
                    else:
                        camera_header = _header_data(self.camera_info.header)
                        pointcloud_header = _header_data(self.pointcloud_header)
                        now_s = self.get_clock().now().nanoseconds * 1e-9
                        reason = april_tag_rejection_reason(
                            observation,
                            expected_family=args.tag_family,
                            expected_id=args.tag_id,
                            expected_edge_size_m=args.tag_edge_size_m,
                            expected_camera_frame=camera_header["frame_id"],
                            max_hamming=args.max_hamming,
                            min_decision_margin=args.min_decision_margin,
                            camera_info_stamp_s=camera_header["stamp_s"],
                            pointcloud_stamp_s=pointcloud_header["stamp_s"],
                            max_sync_offset_s=args.max_sync_offset_s,
                            now_s=now_s,
                            max_age_s=args.max_age_s,
                        )
                    key = (observation.stamp_s, observation.tag_id)
                    if reason is None and key in self.seen:
                        reason = "duplicate detection timestamp"
                    entry["accepted"] = reason is None
                    if reason is not None:
                        entry["reason"] = reason
                    else:
                        self.seen.add(key)
                        self.observations.append(observation)
                except UnsupportedDetectionMessageError as exc:
                    self.fatal_error = str(exc)
                    entry = {"accepted": False, "reason": str(exc), "unsupported_layout": True}
                except ValueError as exc:
                    entry = {"accepted": False, "reason": str(exc)}
                self.log.append(entry)

        def ready(self) -> bool:
            if len(self.observations) < args.samples or self.camera_info is None or self.pointcloud_header is None:
                return False
            camera_frame = _normalize_frame(self.camera_info.header.frame_id)
            source_frame = _normalize_frame(self.pointcloud_header.frame_id)
            try:
                self.graph.lookup(camera_frame, source_frame)
            except KeyError:
                return False
            return True

        def document(self) -> dict[str, Any]:
            if not self.ready():
                raise RuntimeError("collector does not yet have the required observations and transforms")
            assert self.camera_info is not None
            assert self.pointcloud_header is not None
            camera_header = _header_data(self.camera_info.header)
            pointcloud_header = _header_data(self.pointcloud_header)
            camera_frame = camera_header["frame_id"]
            source_frame = pointcloud_header["frame_id"]
            camera_from_source, tf_path = self.graph.lookup(camera_frame, source_frame)
            newest_detection_stamp = max(observation.stamp_s for observation in self.observations)
            stale_tf = [
                record
                for record in tf_path
                if not record.is_static and abs(record.stamp_s - newest_detection_stamp) > args.max_sync_offset_s
            ]
            if stale_tf:
                raise RuntimeError("dynamic TF path timestamp differs from the detections")
            camera = camera_info_data(self.camera_info, args.z_near, args.z_far)
            camera_info_values = {
                "header": camera_header,
                "camera": camera,
                "distortion_model": str(getattr(self.camera_info, "distortion_model", "")),
                "d": [float(value) for value in getattr(self.camera_info, "d", [])],
            }
            tf_values = [
                {
                    "target_frame": record.target_frame,
                    "source_frame": record.source_frame,
                    "target_from_source": record.target_from_source.tolist(),
                    "stamp_s": record.stamp_s,
                    "is_static": record.is_static,
                }
                for record in tf_path
            ]
            accepted_values = [_observation_log_entry(observation) for observation in self.observations]
            raw_log_path = args.raw_log or args.output.with_suffix(".observations.jsonl")
            provenance = {
                "collector": "collect_apriltag_scene_calibration.py",
                "topics": {
                    "detections": args.detections_topic,
                    "camera_info": args.camera_info_topic,
                    "pointcloud": args.pointcloud_topic,
                    "tf": args.tf_topic,
                    "tf_static": args.tf_static_topic,
                },
                "detection_type": args.detection_type,
                "tag_family": args.tag_family,
                "tag_id": args.tag_id,
                "tag_edge_size_m": args.tag_edge_size_m,
                "scene_from_tag": scene_from_tag.tolist(),
                "camera_frame": camera_frame,
                "pointcloud_header": pointcloud_header,
                "raw_observation_log": str(raw_log_path),
            }
            fingerprints = {
                "camera_info": fingerprint(camera_info_values),
                "scene_from_tag": fingerprint(scene_from_tag.tolist()),
                "accepted_observations": fingerprint(accepted_values),
                "tf_path": fingerprint(tf_values),
            }
            aggregation_options = {
                "max_translation_residual_m": args.max_translation_residual_m,
                "max_rotation_residual_deg": args.max_rotation_residual_deg,
                "mad_scale": args.mad_scale,
                "min_inliers": args.min_inliers,
            }
            return build_calibration_document(
                name=args.name,
                source_frame=source_frame,
                scene_frame=args.scene_frame,
                camera=camera,
                scene_from_tag=scene_from_tag,
                camera_from_tag_observations=[
                    observation.camera_from_tag for observation in self.observations[: args.samples]
                ],
                camera_from_source=camera_from_source,
                floor_plane_scene=None
                if args.floor_plane_scene is None
                else np.asarray(args.floor_plane_scene, dtype=np.float64),
                aggregation_options=aggregation_options,
                provenance=provenance,
                fingerprints=fingerprints,
            )

    rclpy = ros["rclpy"]
    rclpy.init()
    node = CollectorNode()
    deadline = time.monotonic() + args.timeout_s
    try:
        while (
            rclpy.ok()
            and time.monotonic() < deadline
            and node.fatal_error is None
            and not node.ready()
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.fatal_error is not None:
            raise RuntimeError(f"Unsupported AprilTag detection message: {node.fatal_error}")
        if not node.ready():
            raise RuntimeError(
                f"Timed out after {args.timeout_s:.3g} s with {len(node.observations)} accepted "
                f"detections out of {args.samples} requested"
            )
        return node.document(), node.log
    finally:
        raw_log_path = args.raw_log or args.output.with_suffix(".observations.jsonl")
        write_raw_log(raw_log_path, node.log)
        node.destroy_node()
        rclpy.shutdown()


def write_raw_log(path: Path, entries: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(_canonical_json(entry) + "\n" for entry in entries)
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = run_ros_collection(args)[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
