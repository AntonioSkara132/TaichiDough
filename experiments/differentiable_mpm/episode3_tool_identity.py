#!/usr/bin/env python3
"""Establish Episode3 tool identities from visual motion before registration."""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_BAG = Path(
    "/home/antonio/diplomski_antonio/diplomski/data/deformpath_training/"
    "DeformPathDataset/snimanje_16_6/episode3_dynamics/episode3_dynamics_0.db3"
)
DEFAULT_SELECTION = Path(
    "/home/antonio/diplomski_antonio/diplomski/data/deformpath_training/staging/"
    "episode3_dynamics_dataset_filt_v1/range_selection.json"
)
DEFAULT_CALIBRATION = EXPERIMENT_ROOT / "data/episode3_dynamics_dataset_filt_v1/scene_calibration_table_aligned_v2.json"
EXPECTED_CALIBRATION_SHA256 = "e6e0eb95e6d75a90e09ca9757b471609f5ba8014a3d24d2e353cc4f179fdc1c7"
EXPECTED_SOURCE_FINGERPRINT = "473a99dace623f9e0b5db93528b3985d3748cbac69d00a1b85990b3a99a24708"
POINT_TOPIC = "/camera/camera/depth/color/points"
POSE_TOPIC = "/poses"
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"
TOOL_NAMES = ("UR5e_spathla", "gen3_spathla")
TOOL_HSV = (0.0, 110.0, 55.0, 255.0, 85.0, 255.0)
PALETTE = ("#0072B2", "#D55E00", "#009E73")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def cdr_header_stamp_ns(prefix: bytes) -> int:
    encapsulation = int.from_bytes(prefix[:2], "big")
    byte_order = ">" if encapsulation in (0x0000, 0x0002) else "<"
    seconds, nanoseconds = struct.unpack_from(f"{byte_order}iI", prefix, 4)
    return int(seconds) * 1_000_000_000 + int(nanoseconds)


def message_stamp_ns(message: Any) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def transform_matrix(translation: Any, quaternion: Any) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    ).as_matrix()
    result[:3, 3] = [translation.x, translation.y, translation.z]
    return result


def pose_matrix(pose: Any) -> np.ndarray:
    return transform_matrix(pose.position, pose.orientation)


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    return values @ matrix[:3, :3].T + matrix[:3, 3]


def rgb_to_hsv_opencv(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.uint8).astype(np.float32) / 255.0
    red, green, blue = (values[..., index] for index in range(3))
    maximum = values.max(axis=-1)
    minimum = values.min(axis=-1)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-6
    red_max = (maximum == red) & nonzero
    green_max = (maximum == green) & nonzero
    blue_max = (maximum == blue) & nonzero
    hue[red_max] = ((green[red_max] - blue[red_max]) / delta[red_max]) % 6.0
    hue[green_max] = (blue[green_max] - red[green_max]) / delta[green_max] + 2.0
    hue[blue_max] = (red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0
    hue *= 30.0
    saturation = np.zeros_like(maximum)
    valued = maximum > 1e-6
    saturation[valued] = delta[valued] / maximum[valued] * 255.0
    return np.stack((hue, saturation, maximum * 255.0), axis=-1)


def hsv_mask(rgb: np.ndarray, limits: Sequence[float] = TOOL_HSV) -> np.ndarray:
    h_min, h_max, s_min, s_max, v_min, v_max = limits
    hsv = rgb_to_hsv_opencv(rgb)
    if h_min <= h_max:
        hue = (hsv[:, 0] >= h_min) & (hsv[:, 0] <= h_max)
    else:
        hue = (hsv[:, 0] >= h_min) | (hsv[:, 0] <= h_max)
    return (
        hue
        & (hsv[:, 1] >= s_min)
        & (hsv[:, 1] <= s_max)
        & (hsv[:, 2] >= v_min)
        & (hsv[:, 2] <= v_max)
    )


def unpack_pointcloud(message: Any) -> tuple[np.ndarray, np.ndarray]:
    offsets = {field.name: int(field.offset) for field in message.fields}
    required = {"x", "y", "z", "rgb"}
    if not required.issubset(offsets):
        raise ValueError(f"Point cloud fields are missing: {sorted(required - set(offsets))}")
    count = int(message.width) * int(message.height)
    raw = np.frombuffer(message.data, dtype=np.uint8)
    point_step = int(message.point_step)
    if int(message.row_step) != int(message.width) * point_step:
        rows = np.empty((count, point_step), dtype=np.uint8)
        image = raw.reshape(int(message.height), int(message.row_step))
        for row in range(int(message.height)):
            start = row * int(message.width)
            rows[start:start + int(message.width)] = image[
                row, : int(message.width) * point_step
            ].reshape(-1, point_step)
    else:
        rows = raw.reshape(count, point_step)
    endian = ">" if message.is_bigendian else "<"
    xyz = np.column_stack([
        rows[:, offsets[name]: offsets[name] + 4].copy().view(f"{endian}f4").reshape(-1)
        for name in ("x", "y", "z")
    ]).astype(np.float64)
    packed = rows[:, offsets["rgb"]: offsets["rgb"] + 4].copy().view(f"{endian}u4").reshape(-1)
    rgb = np.column_stack((
        ((packed >> 16) & 255).astype(np.uint8),
        ((packed >> 8) & 255).astype(np.uint8),
        (packed & 255).astype(np.uint8),
    ))
    return xyz, rgb


def interpolate_transform(
    records: Sequence[tuple[int, np.ndarray]],
    stamp_ns: int,
    *,
    max_gap_ns: int,
    boundary_tolerance_ns: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Interpolate a rigid transform at a sensor timestamp without extrapolation."""
    if len(records) < 2:
        raise ValueError("At least two transform records are required")
    stamps = [record[0] for record in records]
    index = bisect.bisect_left(stamps, stamp_ns)
    if index == 0:
        if stamps[0] - stamp_ns <= boundary_tolerance_ns:
            return records[0][1].copy(), {
                "lower_stamp_ns": stamps[0], "upper_stamp_ns": stamps[0], "fraction": 0.0,
                "boundary_clamped": True,
            }
        raise ValueError("Transform interpolation would extrapolate before the first sample")
    if index == len(records):
        if stamp_ns - stamps[-1] <= boundary_tolerance_ns:
            return records[-1][1].copy(), {
                "lower_stamp_ns": stamps[-1], "upper_stamp_ns": stamps[-1], "fraction": 0.0,
                "boundary_clamped": True,
            }
        raise ValueError("Transform interpolation would extrapolate after the last sample")
    lower_stamp, lower = records[index - 1]
    upper_stamp, upper = records[index]
    gap = upper_stamp - lower_stamp
    if gap <= 0 or gap > max_gap_ns:
        raise ValueError(f"Transform interpolation gap is invalid: {gap} ns")
    fraction = (stamp_ns - lower_stamp) / gap
    translation = (1.0 - fraction) * lower[:3, 3] + fraction * upper[:3, 3]
    rotations = Rotation.from_matrix(np.stack((lower[:3, :3], upper[:3, :3])))
    rotation = Slerp([0.0, 1.0], rotations)([fraction]).as_matrix()[0]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result, {
        "lower_stamp_ns": int(lower_stamp), "upper_stamp_ns": int(upper_stamp),
        "fraction": float(fraction), "boundary_clamped": False,
    }


def robust_transform(transforms: Sequence[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(transforms, dtype=np.float64)
    if values.shape[0] < 5 or values.shape[1:] != (4, 4):
        raise ValueError("Require at least five rigid camera transforms")
    translations = values[:, :3, 3]
    center = np.median(translations, axis=0)
    translation_residual = np.linalg.norm(translations - center, axis=1)
    rotations = values[:, :3, :3]
    pairwise = np.zeros((len(values), len(values)), dtype=np.float64)
    for row in range(len(values)):
        relative = rotations @ rotations[row].T
        pairwise[row] = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
    seed = rotations[int(np.argmin(np.median(pairwise, axis=1)))]
    residual = np.arccos(np.clip((np.trace(rotations @ seed.T, axis1=1, axis2=2) - 1) / 2, -1, 1))
    translation_mad = np.median(np.abs(translation_residual - np.median(translation_residual)))
    rotation_mad = np.median(np.abs(residual - np.median(residual)))
    t_limit = min(0.03, max(0.001, float(np.median(translation_residual) + 3.5 * 1.4826 * translation_mad)))
    r_limit = min(math.radians(5), max(math.radians(0.1), float(np.median(residual) + 3.5 * 1.4826 * rotation_mad)))
    inliers = (translation_residual <= t_limit) & (residual <= r_limit)
    if int(inliers.sum()) < 5:
        raise ValueError("Too few camera transforms passed robust filtering")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_matrix(rotations[inliers]).mean().as_matrix()
    result[:3, 3] = translations[inliers].mean(axis=0)
    return result, {
        "sample_count": len(values), "inlier_count": int(inliers.sum()),
        "translation_limit_mm": t_limit * 1000, "rotation_limit_deg": math.degrees(r_limit),
        "inlier_indices": np.flatnonzero(inliers).tolist(),
    }


class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def deterministic_sample_indices(count: int, maximum: int, seed: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(count, maximum, replace=False))


def component_descriptor(
    points: np.ndarray,
    rgb: np.ndarray,
    *,
    ordinal: int,
    component_index: int,
    maximum_sample: int = 512,
) -> dict[str, Any]:
    values = np.asarray(points, dtype=np.float64)
    colors = np.asarray(rgb, dtype=np.uint8)
    center = np.median(values, axis=0)
    centered = values - center
    covariance = centered.T @ centered / max(len(values) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, -1] *= -1
    projected = centered @ eigenvectors
    extents = np.quantile(projected, 0.95, axis=0) - np.quantile(projected, 0.05, axis=0)
    hsv = rgb_to_hsv_opencv(colors)
    histogram, _ = np.histogramdd(
        hsv, bins=(12, 4, 4), range=((0, 180), (0, 256), (0, 256)), density=False
    )
    histogram = histogram.reshape(-1).astype(np.float64)
    histogram /= max(histogram.sum(), 1.0)
    seed = int(ordinal * 1009 + component_index * 9176)
    selected = deterministic_sample_indices(len(values), maximum_sample, seed)
    sample = values[selected]
    sample_hash = hashlib.sha256(np.asarray(sample, dtype="<f8").tobytes()).hexdigest()
    return {
        "count": int(len(values)), "centroid": center.tolist(),
        "covariance": covariance.tolist(), "eigenvalues": eigenvalues.tolist(),
        "eigenvectors": eigenvectors.tolist(), "extents": extents.tolist(),
        "color_histogram": histogram.tolist(), "sample_indices": selected.tolist(),
        "sample_points": sample.tolist(), "sample_sha256": sample_hash,
    }


def cluster_components(
    points: np.ndarray,
    rgb: np.ndarray,
    *,
    ordinal: int,
    voxel_size: float = 0.008,
    connectivity_radius: float = 0.020,
    minimum_points: int = 80,
) -> tuple[list[dict[str, Any]], int]:
    """Find variable-count connected components without tool or mesh labels."""
    values = np.asarray(points, dtype=np.float64)
    colors = np.asarray(rgb, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] != 3 or colors.shape != (len(values), 3):
        raise ValueError("points/rgb must have shapes [N,3] and [N,3]")
    if len(values) == 0:
        return [], 0
    voxels = np.floor(values / voxel_size).astype(np.int64)
    unique, inverse = np.unique(voxels, axis=0, return_inverse=True)
    sums = np.zeros((len(unique), 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(unique))
    np.add.at(sums, inverse, values)
    centers = sums / counts[:, None]
    union = UnionFind(len(centers))
    for first, second in cKDTree(centers).query_pairs(connectivity_radius, output_type="ndarray"):
        union.union(int(first), int(second))
    roots = np.asarray([union.find(index) for index in range(len(centers))])
    root_counts: dict[int, int] = defaultdict(int)
    for index, root in enumerate(roots):
        root_counts[int(root)] += int(counts[index])
    accepted_roots = [root for root, count in root_counts.items() if count >= minimum_points]
    accepted_roots.sort(key=lambda root: tuple(centers[roots == root].mean(axis=0)))
    components = []
    assigned = np.zeros(len(values), dtype=bool)
    for component_index, root in enumerate(accepted_roots):
        voxel_members = roots == root
        point_members = voxel_members[inverse]
        assigned |= point_members
        descriptor = component_descriptor(
            values[point_members], colors[point_members], ordinal=ordinal,
            component_index=component_index,
        )
        descriptor.update({"component_index": component_index})
        components.append(descriptor)
    return components, int((~assigned).sum())


@dataclass
class Track:
    id: str
    chunk: str
    observations: list[dict[str, Any]] = field(default_factory=list)
    missed: int = 0
    active: bool = True
    ambiguous: bool = False

    @property
    def last(self) -> dict[str, Any]:
        return self.observations[-1]

    def predicted_centroids(self, frame: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Predict with world velocity and each recorded marker's rigid transport."""
        last = np.asarray(self.last["centroid"], dtype=np.float64)
        predictions = {"world-static": last}
        if len(self.observations) >= 2:
            previous = np.asarray(self.observations[-2]["centroid"], dtype=np.float64)
            previous_ordinal = int(self.observations[-2]["ordinal"])
            last_ordinal = int(self.last["ordinal"])
            step = max(last_ordinal - previous_ordinal, 1)
            horizon = max(int(frame["ordinal"]) - last_ordinal, 1)
            predictions["world-velocity"] = last + (last - previous) * (horizon / step)
        for name in TOOL_NAMES:
            last_marker = np.asarray(self.last["poses"][name], dtype=np.float64)
            current_marker = np.asarray(frame["poses"][name], dtype=np.float64)
            predictions[name] = transform(
                last[None, :], current_marker @ np.linalg.inv(last_marker)
            )[0]
        return predictions


def histogram_distance(first: Sequence[float], second: Sequence[float]) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    return float(0.5 * np.sum((a - b) ** 2 / np.maximum(a + b, 1e-12)))


def component_assignment_cost(
    track: Track, component: Mapping[str, Any], frame: Mapping[str, Any]
) -> tuple[float, str]:
    centroid = np.asarray(component["centroid"], dtype=np.float64)
    prediction_errors = {
        name: float(np.linalg.norm(prediction - centroid))
        for name, prediction in track.predicted_centroids(frame).items()
    }
    prediction_name = min(prediction_errors, key=prediction_errors.get)
    distance = prediction_errors[prediction_name]
    extent_a = np.asarray(track.last["extents"], dtype=np.float64)
    extent_b = np.asarray(component["extents"], dtype=np.float64)
    extent_cost = float(np.linalg.norm(extent_a - extent_b) / max(np.linalg.norm(extent_a), 1e-3))
    color_cost = histogram_distance(track.last["color_histogram"], component["color_histogram"])
    count_ratio = abs(math.log(max(float(component["count"]), 1.0) / max(float(track.last["count"]), 1.0)))
    cost = distance / 0.06 + 0.30 * min(extent_cost, 4.0)
    cost += 0.30 * min(color_cost, 4.0) + 0.12 * min(count_ratio, 4.0)
    return cost, prediction_name


def track_components(
    frames: Sequence[dict[str, Any]],
    *,
    max_assignment_cost: float = 3.2,
    max_missed: int = 8,
) -> list[Track]:
    tracks: list[Track] = []
    next_id = 0
    active: list[Track] = []
    for frame in frames:
        components = frame["components"]
        candidates = [track for track in active if track.active]
        assignments: list[tuple[int, int]] = []
        costs = np.empty((len(candidates), len(components)), dtype=np.float64)
        prediction_names: list[list[str]] = []
        if candidates and components:
            for row, track in enumerate(candidates):
                names = []
                for column, component in enumerate(components):
                    costs[row, column], prediction_name = component_assignment_cost(
                        track, component, frame
                    )
                    names.append(prediction_name)
                prediction_names.append(names)
            rows, columns = linear_sum_assignment(costs)
            assignments = [
                (int(row), int(column)) for row, column in zip(rows, columns)
                if costs[row, column] <= max_assignment_cost
            ]
        # Split/merge ambiguity requires two close, descriptor-compatible alternatives.
        # The wider assignment limit is only for maintaining identity through motion and occlusion.
        plausible = costs <= min(0.9, max_assignment_cost)
        split_rows = set(np.flatnonzero(plausible.sum(axis=1) > 1)) if plausible.size else set()
        merge_columns = set(np.flatnonzero(plausible.sum(axis=0) > 1)) if plausible.size else set()
        used_tracks = {row for row, _ in assignments}
        used_components = {column for _, column in assignments}
        for row, column in assignments:
            ambiguous = row in split_rows or column in merge_columns
            observation = dict(components[column])
            observation.update({
                "ordinal": frame["ordinal"], "stamp_ns": frame["stamp_ns"],
                "window": frame["window"], "membership": frame["membership"],
                "poses": frame["poses"], "ambiguous_interval": ambiguous,
                "ambiguity_type": (
                    "split_and_merge" if row in split_rows and column in merge_columns
                    else "split" if row in split_rows else "merge" if column in merge_columns else None
                ),
                "assignment_cost": float(costs[row, column]),
                "prediction_model": prediction_names[row][column],
                "occluded_frames_before": candidates[row].missed,
            })
            candidates[row].observations.append(observation)
            candidates[row].missed = 0
            candidates[row].ambiguous |= ambiguous
        for row, track in enumerate(candidates):
            if row not in used_tracks:
                track.missed += 1
                if track.missed > max_missed:
                    track.active = False
        for column, component in enumerate(components):
            if column in used_components:
                continue
            ambiguous = column in merge_columns
            track = Track(id=f"{frame['chunk']}-track-{next_id:04d}", chunk=frame["chunk"])
            next_id += 1
            observation = dict(component)
            observation.update({
                "ordinal": frame["ordinal"], "stamp_ns": frame["stamp_ns"],
                "window": frame["window"], "membership": frame["membership"],
                "poses": frame["poses"], "ambiguous_interval": ambiguous,
                "ambiguity_type": "merge" if ambiguous else None,
                "assignment_cost": None, "prediction_model": None,
                "occluded_frames_before": 0,
            })
            track.observations.append(observation)
            track.ambiguous = ambiguous
            tracks.append(track)
            active.append(track)
        active = [track for track in active if track.active]
    return tracks


def orientation_distance(rotations: np.ndarray) -> float | None:
    if len(rotations) < 3:
        return None
    reference = rotations[0]
    angles = [Rotation.from_matrix(reference.T @ rotation).magnitude() for rotation in rotations[1:]]
    return float(np.median(angles)) if angles else 0.0


def hypothesis_score(observations: Sequence[Mapping[str, Any]], hypothesis: str) -> dict[str, Any]:
    if hypothesis not in {*TOOL_NAMES, "world-static"}:
        raise ValueError(f"Unknown identity hypothesis: {hypothesis}")
    centroids, orientations, extents = [], [], []
    for observation in observations:
        centroid = np.asarray(observation["centroid"], dtype=np.float64)
        eigenvectors = np.asarray(observation["eigenvectors"], dtype=np.float64)
        if hypothesis == "world-static":
            local_centroid = centroid
            local_orientation = eigenvectors
        else:
            marker = np.asarray(observation["poses"][hypothesis], dtype=np.float64)
            local_centroid = transform(centroid[None, :], np.linalg.inv(marker))[0]
            local_orientation = marker[:3, :3].T @ eigenvectors
        centroids.append(local_centroid)
        orientations.append(local_orientation)
        extents.append(np.asarray(observation["extents"], dtype=np.float64))
    centroid_array = np.asarray(centroids)
    center = np.median(centroid_array, axis=0)
    centroid_dispersion = float(np.median(np.linalg.norm(centroid_array - center, axis=1)))
    extent_array = np.asarray(extents)
    extent_center = np.median(extent_array, axis=0)
    extent_drift = float(np.median(np.linalg.norm(extent_array - extent_center, axis=1)))
    eigenvalues = np.median(np.asarray([row["eigenvalues"] for row in observations]), axis=0)
    rotation_available = bool(eigenvalues[0] > 1.5 * max(eigenvalues[1], 1e-12))
    angular = orientation_distance(np.asarray(orientations)) if rotation_available else None
    lever = float(max(np.max(extent_center), 0.02))
    score = centroid_dispersion + 0.20 * extent_drift
    if angular is not None:
        score += 0.15 * lever * angular
    return {
        "score": float(score), "centroid_dispersion_m": centroid_dispersion,
        "extent_drift_m": extent_drift, "rotation_available": rotation_available,
        "orientation_dispersion_rad": angular,
    }


def score_track(track: Track, *, negative_rolls: Sequence[int] = (3, 7, 13, 19, 31)) -> dict[str, Any]:
    usable = [row for row in track.observations if not row.get("ambiguous_interval", False)]
    result: dict[str, Any] = {
        "track_id": track.id, "observation_count": len(track.observations),
        "usable_observation_count": len(usable),
        "ambiguous_observation_count": len(track.observations) - len(usable),
    }
    for membership in ("development", "heldout"):
        observations = [row for row in usable if row["membership"] == membership]
        result[membership] = {"observation_count": len(observations), "hypotheses": {}}
        if len(observations) < 3:
            continue
        for hypothesis in (*TOOL_NAMES, "world-static"):
            result[membership]["hypotheses"][hypothesis] = hypothesis_score(observations, hypothesis)
        for tool in TOOL_NAMES:
            controls = []
            for roll in negative_rolls:
                if len(observations) < 2:
                    continue
                shifted = []
                for index, observation in enumerate(observations):
                    copy = dict(observation)
                    poses = dict(copy["poses"])
                    poses[tool] = observations[(index + roll) % len(observations)]["poses"][tool]
                    copy["poses"] = poses
                    shifted.append(copy)
                controls.append(hypothesis_score(shifted, tool)["score"])
            result[membership]["negative_controls"] = result[membership].get("negative_controls", {})
            result[membership]["negative_controls"][tool] = controls
    return result


def evaluate_identity(
    tracks: Sequence[Track],
    *,
    minimum_observations: int = 8,
    minimum_margin: float = 0.30,
) -> dict[str, Any]:
    scored = [score_track(track) for track in tracks if len(track.observations) >= minimum_observations]
    candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in TOOL_NAMES}
    for track, record in zip(
        [track for track in tracks if len(track.observations) >= minimum_observations], scored
    ):
        usable = [row for row in track.observations if not row.get("ambiguous_interval", False)]
        windows = {row["window"] for row in usable}
        for tool in TOOL_NAMES:
            reasons = []
            margins = []
            negative_ok = True
            for membership in ("development", "heldout"):
                section = record[membership]
                hypotheses = section.get("hypotheses", {})
                if section["observation_count"] < 3 or tool not in hypotheses:
                    reasons.append(f"insufficient_{membership}_support")
                    continue
                winning = hypotheses[tool]["score"]
                alternatives = [value["score"] for name, value in hypotheses.items() if name != tool]
                alternative = min(alternatives)
                margin = (alternative - winning) / max(alternative, 1e-12)
                margins.append(float(margin))
                if margin < minimum_margin:
                    reasons.append(f"{membership}_margin_below_threshold")
                controls = section.get("negative_controls", {}).get(tool, [])
                if controls and winning >= float(np.quantile(controls, 0.05)):
                    negative_ok = False
                    reasons.append(f"{membership}_negative_control_not_beaten")
            if len(windows) < 2:
                reasons.append("fewer_than_two_windows")
            if len(usable) < minimum_observations:
                reasons.append("fewer_than_eight_nonambiguous_observations")
            candidates[tool].append({
                "track_id": track.id, "accepted": not reasons and negative_ok,
                "minimum_margin": min(margins) if margins else None,
                "reasons": sorted(set(reasons)), "windows": sorted(windows),
                "usable_observation_count": len(usable),
                "ambiguous_observation_count": len(track.observations) - len(usable),
            })
    accepted_mapping = None
    for ur in candidates[TOOL_NAMES[0]]:
        for gen in candidates[TOOL_NAMES[1]]:
            if ur["accepted"] and gen["accepted"] and ur["track_id"] != gen["track_id"]:
                score = min(float(ur["minimum_margin"]), float(gen["minimum_margin"]))
                if accepted_mapping is None or score > accepted_mapping[0]:
                    accepted_mapping = (score, {TOOL_NAMES[0]: ur["track_id"], TOOL_NAMES[1]: gen["track_id"]})
    status = "accepted" if accepted_mapping else "identity_unresolved"
    reasons = [] if accepted_mapping else [
        "No pair of distinct persistent tracks passed development, held-out, margin, negative-control, and ambiguity checks."
    ]
    return {
        "status": status, "mapping": accepted_mapping[1] if accepted_mapping else None,
        "pair_minimum_margin": accepted_mapping[0] if accepted_mapping else None,
        "candidates": candidates, "track_scores": scored, "reasons": reasons,
    }


def assign_windows(ordinal: int, retained_ranges: Sequence[Mapping[str, Any]]) -> tuple[str, str, str]:
    for item in retained_ranges:
        start = int(item["start_pointcloud_ordinal"])
        end = int(item["end_pointcloud_ordinal_exclusive"])
        if start <= ordinal < end:
            length = end - start
            section = min(3, int((ordinal - start) * 4 / max(length, 1)))
            membership = "development" if section % 2 == 0 else "heldout"
            return str(item["id"]), f"{item['id']}-window{section + 1}", membership
    raise ValueError(f"Ordinal {ordinal} is outside retained ranges")


def _nearest(records: Sequence[tuple[int, np.ndarray]], stamp_ns: int) -> np.ndarray:
    stamps = [row[0] for row in records]
    index = bisect.bisect_left(stamps, stamp_ns)
    choices = [i for i in (index - 1, index) if 0 <= i < len(records)]
    if not choices:
        raise ValueError("No transform sample is available")
    return records[min(choices, key=lambda i: abs(stamps[i] - stamp_ns))][1]


def load_ros_inputs(database: Path) -> dict[str, Any]:
    """Load point references and transform streams; point clouds remain in SQLite."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    with sqlite3.connect(database) as connection:
        topics = {
            name: {"id": topic_id, "type": type_name}
            for topic_id, name, type_name in connection.execute("SELECT id,name,type FROM topics")
        }
        required = {POINT_TOPIC, POSE_TOPIC, TF_TOPIC, TF_STATIC_TOPIC}
        if not required.issubset(topics):
            raise ValueError(f"ROS bag is missing topics: {sorted(required - set(topics))}")
        point_refs = sorted([
            (cdr_header_stamp_ns(prefix), int(message_id), int(record_stamp))
            for message_id, record_stamp, prefix in connection.execute(
                "SELECT id,timestamp,substr(data,1,12) FROM messages WHERE topic_id=?",
                (topics[POINT_TOPIC]["id"],),
            )
        ])
        tf_class = get_message(topics[TF_TOPIC]["type"])
        tf_sequences: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = defaultdict(list)
        keys = {("mocap", "camera_link"), *(('mocap', name) for name in TOOL_NAMES)}
        for (blob,) in connection.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY id", (topics[TF_TOPIC]["id"],)
        ):
            message = deserialize_message(blob, tf_class)
            for item in message.transforms:
                key = (item.header.frame_id, item.child_frame_id)
                if key in keys:
                    stamp = int(item.header.stamp.sec) * 1_000_000_000 + int(item.header.stamp.nanosec)
                    tf_sequences[key].append((stamp, transform_matrix(item.transform.translation, item.transform.rotation)))
        for records in tf_sequences.values():
            records.sort(key=lambda row: row[0])
        pose_class = get_message(topics[POSE_TOPIC]["type"])
        poses: list[tuple[int, dict[str, np.ndarray]]] = []
        for (blob,) in connection.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY id", (topics[POSE_TOPIC]["id"],)
        ):
            message = deserialize_message(blob, pose_class)
            named = {item.name: pose_matrix(item.pose) for item in message.poses if item.name in TOOL_NAMES}
            if all(name in named for name in TOOL_NAMES):
                poses.append((message_stamp_ns(message), named))
        poses.sort(key=lambda row: row[0])
        static_class = get_message(topics[TF_STATIC_TOPIC]["type"])
        static: dict[tuple[str, str], np.ndarray] = {}
        for (blob,) in connection.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY id", (topics[TF_STATIC_TOPIC]["id"],)
        ):
            message = deserialize_message(blob, static_class)
            for item in message.transforms:
                static[(item.header.frame_id, item.child_frame_id)] = transform_matrix(
                    item.transform.translation, item.transform.rotation
                )
    camera_link_from_optical = (
        static[("camera_link", "camera_depth_frame")]
        @ static[("camera_depth_frame", "camera_depth_optical_frame")]
    )
    return {
        "topics": topics, "point_refs": point_refs, "tf_sequences": tf_sequences,
        "poses": poses, "camera_link_from_optical": camera_link_from_optical,
    }


def interpolate_named_poses(
    records: Sequence[tuple[int, Mapping[str, np.ndarray]]], stamp_ns: int, *, max_gap_ns: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    result, diagnostics = {}, {}
    for name in TOOL_NAMES:
        sequence = [(stamp, np.asarray(named[name])) for stamp, named in records]
        result[name], diagnostics[name] = interpolate_transform(sequence, stamp_ns, max_gap_ns=max_gap_ns)
    return result, diagnostics


def deterministic_frame_subset(
    retained_ranges: Sequence[Mapping[str, Any]], mode: str
) -> list[int]:
    all_ordinals = [
        ordinal for item in retained_ranges
        for ordinal in range(int(item["start_pointcloud_ordinal"]), int(item["end_pointcloud_ordinal_exclusive"]))
    ]
    if mode == "all":
        return all_ordinals
    if mode != "stratified-96":
        raise ValueError(f"Unknown frame subset: {mode}")
    blocks = (
        (40, 48), (220, 228), (428, 436), (700, 708),
        (803, 811), (816, 824), (850, 858), (902, 910),
        (1687, 1695), (1721, 1729), (1755, 1763), (1815, 1823),
    )
    selected = [ordinal for start, end in blocks for ordinal in range(start, end)]
    retained = set(all_ordinals)
    if len(selected) != 96 or not set(selected).issubset(retained):
        raise ValueError("The stratified Episode3 frame profile no longer matches retained ranges")
    return selected


def process_bag(
    database: Path,
    selection: Mapping[str, Any],
    *,
    frame_subset: str,
    voxel_size: float,
    connectivity_radius: float,
    minimum_points: int,
    max_pose_gap_ns: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    loaded = load_ros_inputs(database)
    point_refs = loaded["point_refs"]
    retained_ranges = selection["retained_ranges"]
    expected = sum(
        int(item["end_pointcloud_ordinal_exclusive"]) - int(item["start_pointcloud_ordinal"])
        for item in retained_ranges
    )
    all_retained = {
        ordinal for item in retained_ranges
        for ordinal in range(int(item["start_pointcloud_ordinal"]), int(item["end_pointcloud_ordinal_exclusive"]))
    }
    if len(all_retained) != expected or expected != 1001:
        raise ValueError(f"Expected exactly 1001 retained ordinals, got {expected}")
    selected_ordinals = deterministic_frame_subset(retained_ranges, frame_subset)
    retained = set(selected_ordinals)
    camera_samples = []
    for ordinal in np.linspace(0, len(point_refs) - 1, 32, dtype=int):
        camera_samples.append(_nearest(loaded["tf_sequences"][("mocap", "camera_link")], point_refs[ordinal][0]))
    mocap_from_camera, camera_diagnostics = robust_transform(camera_samples)
    mocap_from_optical = mocap_from_camera @ loaded["camera_link_from_optical"]
    frames = []
    point_class = get_message(loaded["topics"][POINT_TOPIC]["type"])
    with sqlite3.connect(database) as connection:
        for ordinal in sorted(retained):
            stamp_ns, message_id, record_stamp = point_refs[ordinal]
            row = connection.execute("SELECT data FROM messages WHERE id=?", (message_id,)).fetchone()
            if row is None:
                raise ValueError(f"Missing point-cloud message id {message_id}")
            message = deserialize_message(row[0], point_class)
            if message_stamp_ns(message) != stamp_ns:
                raise ValueError("Point-cloud prefix and decoded header stamp differ")
            xyz_optical, rgb = unpack_pointcloud(message)
            valid = np.isfinite(xyz_optical).all(axis=1) & (xyz_optical[:, 2] > 0.05) & (xyz_optical[:, 2] < 3.0)
            xyz_mocap = transform(xyz_optical[valid], mocap_from_optical)
            valid_rgb = rgb[valid]
            selected = hsv_mask(valid_rgb)
            points, colors = xyz_mocap[selected], valid_rgb[selected]
            chunk, window, membership = assign_windows(ordinal, retained_ranges)
            components, noise_count = cluster_components(
                points, colors, ordinal=ordinal, voxel_size=voxel_size,
                connectivity_radius=connectivity_radius, minimum_points=minimum_points,
            )
            poses, pose_diagnostics = interpolate_named_poses(
                loaded["poses"], stamp_ns, max_gap_ns=max_pose_gap_ns
            )
            tf_pose = {}
            for name in TOOL_NAMES:
                tf, tf_diag = interpolate_transform(
                    loaded["tf_sequences"][("mocap", name)], stamp_ns, max_gap_ns=max_pose_gap_ns
                )
                delta = np.linalg.inv(poses[name]) @ tf
                tf_pose[name] = {
                    "translation_difference_mm": float(np.linalg.norm(delta[:3, 3]) * 1000),
                    "rotation_difference_deg": float(np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude())),
                    "tf_interpolation": tf_diag,
                }
            frames.append({
                "ordinal": ordinal, "stamp_ns": stamp_ns, "record_stamp_ns": record_stamp,
                "chunk": chunk, "window": window, "membership": membership,
                "selected_point_count": int(len(points)), "noise_point_count": noise_count,
                "components": components,
                "poses": {name: poses[name].tolist() for name in TOOL_NAMES},
                "pose_interpolation": pose_diagnostics, "tf_pose_consistency": tf_pose,
            })
    return frames, {
        "full_retained_frame_capacity": expected,
        "analyzed_frame_count": len(frames),
        "frame_subset": frame_subset,
        "selected_ordinals": selected_ordinals,
        "camera_transform": mocap_from_optical.tolist(),
        "camera_aggregation": camera_diagnostics,
    }


def draw_timeline(path: Path, tracks: Sequence[Track]) -> None:
    image = Image.new("RGB", (1600, max(420, 80 + 30 * len(tracks))), "#FCFCFB")
    draw = ImageDraw.Draw(image)
    draw.text((40, 22), "Episode3 persistent visual tracks", fill="#111827")
    draw.text((40, 44), "Horizontal position is raw point-cloud ordinal; track IDs are neutral until identity passes.", fill="#4B5563")
    ordinals = [row["ordinal"] for track in tracks for row in track.observations]
    low, high = min(ordinals), max(ordinals)
    for index, track in enumerate(tracks):
        y = 85 + index * 30
        draw.text((15, y - 7), track.id, fill="#111827")
        for observation in track.observations:
            x = 180 + (observation["ordinal"] - low) / max(high - low, 1) * 1370
            color = PALETTE[0] if observation["membership"] == "development" else PALETTE[1]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        if track.ambiguous:
            draw.text((1560, y - 7), "ambiguous", fill="#B42318")
    image.save(path)


def draw_scores(path: Path, identity: Mapping[str, Any]) -> None:
    records = identity["track_scores"]
    rows = []
    for record in records:
        for membership in ("development", "heldout"):
            for hypothesis, value in record[membership].get("hypotheses", {}).items():
                rows.append((record["track_id"], membership, hypothesis, value["score"]))
    image = Image.new("RGB", (1600, max(500, 100 + len(rows) * 20)), "#FCFCFB")
    draw = ImageDraw.Draw(image)
    draw.text((40, 22), "Episode3 identity stability scores", fill="#111827")
    draw.text((40, 44), "Lower is better. Every row is labeled; color is secondary.", fill="#4B5563")
    maximum = max((row[3] for row in rows), default=1.0)
    color = {TOOL_NAMES[0]: PALETTE[0], TOOL_NAMES[1]: PALETTE[1], "world-static": PALETTE[2]}
    for index, (track_id, membership, hypothesis, value) in enumerate(rows):
        y = 80 + index * 20
        label = f"{track_id} {membership} {hypothesis}"
        draw.text((20, y), label, fill="#111827")
        width = 700 * value / max(maximum, 1e-12)
        draw.rectangle((520, y + 2, 520 + width, y + 14), fill=color[hypothesis])
        draw.text((1230, y), f"{value:.6f}", fill="#374151")
    image.save(path)


def serialize_track(track: Track) -> dict[str, Any]:
    return {
        "id": track.id, "chunk": track.chunk, "ambiguous": track.ambiguous,
        "observation_count": len(track.observations), "observations": track.observations,
    }


def run_identity(
    database: Path,
    selection_path: Path,
    calibration: Path,
    output: Path,
    *,
    expected_calibration_sha256: str,
    frame_subset: str = "all",
    voxel_size: float = 0.008,
    connectivity_radius: float = 0.020,
    minimum_points: int = 80,
    max_pose_gap_ms: float = 50.0,
) -> dict[str, Any]:
    database = database.expanduser().resolve()
    selection_path = selection_path.expanduser().resolve()
    calibration = calibration.expanduser().resolve()
    output = output.expanduser().resolve()
    if not output.is_relative_to(EXPERIMENT_ROOT) or output == EXPERIMENT_ROOT:
        raise ValueError(f"Output must be inside {EXPERIMENT_ROOT}")
    if output.exists():
        raise FileExistsError(f"Refuse to replace existing output: {output}")
    selection = json.loads(selection_path.read_text())
    if selection.get("source", {}).get("fingerprint") != EXPECTED_SOURCE_FINGERPRINT:
        raise ValueError("Range selection belongs to a different source recording")
    calibration_hash = sha256(calibration)
    if calibration_hash != expected_calibration_sha256:
        raise ValueError("Calibration SHA-256 differs from the required input")
    before = {
        "database_size": database.stat().st_size, "database_mtime_ns": database.stat().st_mtime_ns,
        "selection_sha256": sha256(selection_path), "calibration_sha256": calibration_hash,
    }
    output.mkdir(parents=True, exist_ok=False)
    frames, diagnostics = process_bag(
        database, selection, frame_subset=frame_subset,
        voxel_size=voxel_size, connectivity_radius=connectivity_radius,
        minimum_points=minimum_points, max_pose_gap_ns=int(max_pose_gap_ms * 1e6),
    )
    tracks = []
    for chunk in (item["id"] for item in selection["retained_ranges"]):
        chunk_frames = [frame for frame in frames if frame["chunk"] == chunk]
        tracks.extend(track_components(chunk_frames))
    identity = evaluate_identity(tracks)
    component_path = output / "component_records.jsonl"
    with component_path.open("x", encoding="utf-8") as stream:
        for frame in frames:
            stream.write(json.dumps(frame, allow_nan=False) + "\n")
    write_new_json(output / "tracks.json", {
        "schema": "taichidough/episode3-visual-tracks/v1",
        "tracks": [serialize_track(track) for track in tracks],
    })
    summary = {
        "schema": "taichidough/episode3-tool-identity/v1", **identity,
        "inputs": {
            "database": str(database), "selection": str(selection_path),
            "calibration": str(calibration), **before,
        },
        "settings": {
            "retained_ranges": selection["retained_ranges"], "voxel_size_m": voxel_size,
            "connectivity_radius_m": connectivity_radius, "minimum_component_points": minimum_points,
            "max_pose_gap_ms": max_pose_gap_ms, "camera_orientation": "recorded_tf",
            "frame_subset": frame_subset,
        },
        "diagnostics": diagnostics,
        "artifacts": {
            "components": "component_records.jsonl", "tracks": "tracks.json",
            "timeline": "identity_timeline.png", "scores": "identity_scores.png",
        },
        "operations_not_run": ["tool_registration", "calibration", "MPM", "gradient", "material_fit"],
    }
    draw_timeline(output / "identity_timeline.png", tracks)
    draw_scores(output / "identity_scores.png", summary)
    write_new_json(output / "identity_summary.json", summary)
    after = {
        "database_size": database.stat().st_size, "database_mtime_ns": database.stat().st_mtime_ns,
        "selection_sha256": sha256(selection_path), "calibration_sha256": sha256(calibration),
    }
    if after != before:
        raise RuntimeError("An immutable Episode3 input changed during identity analysis")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    result.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    result.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    result.add_argument("--expected-calibration-sha256", default=EXPECTED_CALIBRATION_SHA256)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--frame-subset", choices=("all", "stratified-96"), default="all")
    result.add_argument("--voxel-size", type=float, default=0.008)
    result.add_argument("--connectivity-radius", type=float, default=0.020)
    result.add_argument("--minimum-points", type=int, default=80)
    result.add_argument("--max-pose-gap-ms", type=float, default=50.0)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        summary = run_identity(
            args.bag, args.selection, args.calibration, args.output,
            expected_calibration_sha256=args.expected_calibration_sha256,
            frame_subset=args.frame_subset,
            voxel_size=args.voxel_size, connectivity_radius=args.connectivity_radius,
            minimum_points=args.minimum_points, max_pose_gap_ms=args.max_pose_gap_ms,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}")
        return 2
    print(json.dumps({"output": str(args.output), "status": summary["status"]}, indent=2))
    return 0 if summary["status"] == "accepted" else 3


if __name__ == "__main__":
    raise SystemExit(main())
