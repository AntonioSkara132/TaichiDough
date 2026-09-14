from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.differentiable_mpm.episode3_tool_identity import (
    TOOL_NAMES,
    Track,
    cluster_components,
    deterministic_frame_subset,
    evaluate_identity,
    interpolate_transform,
    track_components,
)


def rigid(translation=(0.0, 0.0, 0.0), angle=0.0) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec([0.0, 0.0, angle]).as_matrix()
    value[:3, 3] = translation
    return value


def component(center, *, count=200, extent=(0.08, 0.02, 0.01), color_bin=0):
    histogram = np.zeros(192)
    histogram[color_bin] = 1.0
    return {
        "count": count,
        "centroid": list(center),
        "extents": list(extent),
        "color_histogram": histogram.tolist(),
        "eigenvalues": [0.003, 0.0002, 0.00005],
        "eigenvectors": np.eye(3).tolist(),
        "sample_points": [list(center)],
    }


def frame(ordinal, components, *, membership="development", window="w1"):
    return {
        "ordinal": ordinal,
        "stamp_ns": ordinal * 1_000_000,
        "chunk": "chunk01",
        "window": window,
        "membership": membership,
        "components": components,
        "poses": {name: np.eye(4).tolist() for name in TOOL_NAMES},
    }


def test_stratified_subset_is_exact_and_covers_all_motion_regions():
    ranges = [
        {"start_pointcloud_ordinal": 0, "end_pointcloud_ordinal_exclusive": 734},
        {"start_pointcloud_ordinal": 779, "end_pointcloud_ordinal_exclusive": 910},
        {"start_pointcloud_ordinal": 1687, "end_pointcloud_ordinal_exclusive": 1823},
    ]
    selected = deterministic_frame_subset(ranges, "stratified-96")
    assert len(selected) == len(set(selected)) == 96
    assert {431, 703, 805, 905, 1687, 1822}.issubset(selected)
    assert sum(0 <= value < 734 for value in selected) == 32
    assert sum(779 <= value < 910 for value in selected) == 32
    assert sum(1687 <= value < 1823 for value in selected) == 32
    assert deterministic_frame_subset(ranges, "all")[:2] == [0, 1]


def test_interpolates_translation_and_rotation_and_rejects_bad_time():
    records = [(0, rigid()), (10, rigid((1.0, 0.0, 0.0), math.pi))]
    result, diagnostics = interpolate_transform(records, 5, max_gap_ns=10)
    assert result[:3, 3] == pytest.approx([0.5, 0.0, 0.0])
    assert Rotation.from_matrix(result[:3, :3]).magnitude() == pytest.approx(math.pi / 2)
    assert diagnostics["fraction"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="before"):
        interpolate_transform(records, -1, max_gap_ns=10)
    with pytest.raises(ValueError, match="after"):
        interpolate_transform(records, 11, max_gap_ns=10)
    with pytest.raises(ValueError, match="gap"):
        interpolate_transform(records, 5, max_gap_ns=9)


def test_variable_component_count_and_deterministic_descriptors():
    rng = np.random.default_rng(4)
    first = rng.normal([0.0, 0.0, 0.0], 0.002, size=(160, 3))
    second = rng.normal([0.2, 0.0, 0.0], 0.002, size=(190, 3))
    noise = np.array([[0.5, 0.5, 0.5]])
    points = np.concatenate((first, second, noise))
    rgb = np.tile(np.array([[250, 180, 40]], dtype=np.uint8), (len(points), 1))
    one, noise_count = cluster_components(
        points, rgb, ordinal=7, voxel_size=0.005,
        connectivity_radius=0.018, minimum_points=50,
    )
    two, second_noise_count = cluster_components(
        points, rgb, ordinal=7, voxel_size=0.005,
        connectivity_radius=0.018, minimum_points=50,
    )
    assert len(one) == 2
    assert noise_count == second_noise_count == 1
    assert [row["sample_sha256"] for row in one] == [row["sample_sha256"] for row in two]
    assert [row["centroid"] for row in one] == [row["centroid"] for row in two]


def test_tracking_handles_short_occlusion_without_identity_swap():
    frames = [
        frame(0, [component((0.0, 0, 0), color_bin=1), component((0.5, 0, 0), color_bin=2)]),
        frame(1, [component((0.02, 0, 0), color_bin=1), component((0.48, 0, 0), color_bin=2)]),
        frame(2, [component((0.46, 0, 0), color_bin=2)]),
        frame(3, [component((0.06, 0, 0), color_bin=1), component((0.44, 0, 0), color_bin=2)]),
    ]
    tracks = track_components(frames, max_missed=2)
    long_tracks = sorted((track for track in tracks if len(track.observations) >= 3), key=lambda t: t.last["centroid"][0])
    assert len(long_tracks) == 2
    left = min(tracks, key=lambda track: track.observations[0]["centroid"][0])
    assert [row["ordinal"] for row in left.observations] == [0, 1, 3]


def test_rigid_transport_prevents_fast_tool_motion_fragmentation():
    frames = []
    for ordinal in range(12):
        marker = rigid((0.11 * ordinal, 0.02 * ordinal, 0.0), 0.12 * ordinal)
        center = marker[:3, :3] @ np.array([0.08, 0.01, 0.0]) + marker[:3, 3]
        value = frame(ordinal, [component(center, color_bin=4)])
        value["poses"][TOOL_NAMES[0]] = marker.tolist()
        frames.append(value)
    tracks = track_components(frames)
    assert len(tracks) == 1
    assert len(tracks[0].observations) == 12
    assert {row["prediction_model"] for row in tracks[0].observations[1:]} == {TOOL_NAMES[0]}


def test_records_local_split_ambiguity_without_marking_unrelated_track():
    frames = [
        frame(0, [component((0.0, 0, 0), color_bin=1), component((0.6, 0, 0), color_bin=5)]),
        frame(1, [component((-0.015, 0, 0), color_bin=1), component((0.015, 0, 0), color_bin=1),
                  component((0.6, 0, 0), color_bin=5)]),
    ]
    tracks = track_components(frames)
    stable = next(track for track in tracks if track.observations[0]["centroid"][0] > 0.5)
    assert stable.observations[-1]["ambiguous_interval"] is False
    assert any(
        row.get("ambiguity_type") in {"split", "split_and_merge"}
        for track in tracks for row in track.observations
    )


def attached_track(track_id: str, tool: str, offset: np.ndarray) -> Track:
    track = Track(id=track_id, chunk="chunk01")
    for index in range(24):
        phase = index / 23
        ur = rigid((0.30 * phase, 0.04 * math.sin(phase * 5), 0.0), 0.4 * phase)
        gen = rigid((-0.12 * phase, 0.25 * phase, 0.03 * math.cos(phase * 4)), -0.5 * phase)
        poses = {TOOL_NAMES[0]: ur, TOOL_NAMES[1]: gen}
        marker = poses[tool]
        center = marker[:3, :3] @ offset + marker[:3, 3]
        membership = "development" if index % 4 < 2 else "heldout"
        window = "w1" if index < 12 else "w2"
        observation = component(center)
        observation.update({
            "ordinal": index,
            "stamp_ns": index,
            "window": window,
            "membership": membership,
            "poses": {name: value.tolist() for name, value in poses.items()},
            "count_change_nearby": False,
        })
        track.observations.append(observation)
    return track


def test_motion_identity_accepts_distinct_robot_attached_tracks():
    ur = attached_track("ur-track", TOOL_NAMES[0], np.array([0.10, 0.02, 0.0]))
    gen = attached_track("gen-track", TOOL_NAMES[1], np.array([-0.08, 0.01, 0.03]))
    result = evaluate_identity([ur, gen])
    assert result["status"] == "accepted"
    assert result["mapping"] == {TOOL_NAMES[0]: "ur-track", TOOL_NAMES[1]: "gen-track"}


def test_static_or_ambiguous_tracks_do_not_produce_identity():
    static = Track(id="static", chunk="chunk01")
    for index in range(24):
        observation = component((0.2, 0.3, 0.4))
        observation.update({
            "ordinal": index, "stamp_ns": index,
            "window": "w1" if index < 12 else "w2",
            "membership": "development" if index % 4 < 2 else "heldout",
            "poses": {
                TOOL_NAMES[0]: rigid((index * 0.01, 0, 0)).tolist(),
                TOOL_NAMES[1]: rigid((0, index * 0.01, 0)).tolist(),
            },
            "count_change_nearby": False,
        })
        static.observations.append(observation)
    assert evaluate_identity([static])["status"] == "identity_unresolved"
    ambiguous = attached_track("ambiguous", TOOL_NAMES[0], np.array([0.1, 0, 0]))
    ambiguous.ambiguous = True
    assert evaluate_identity([ambiguous])["status"] == "identity_unresolved"
