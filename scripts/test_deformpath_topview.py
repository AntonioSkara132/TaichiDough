#!/usr/bin/env python3
"""Deterministic checks for static top-view geometry helpers."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from deformpath_topview import (
    SCENE_CALIBRATION_V2,
    apply_calibration,
    calibration_metadata,
    depth_to_pointcloud,
    fixed_grid_metrics,
    load_calibration,
    project_camera_points,
    rasterize_depth,
    rigid_inverse,
    transform_plane,
    transform_points,
    unproject_camera_points,
)
from reconstruct_voxel_dough_from_deformpath import sample_surface_preserving_particles
from visualize_static_topview_benchmark import (
    calibration_reference_transforms,
    floor_column_geometry,
    floor_plane_samples,
    floor_segmentation_diagnostics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def v2_document(
    scene_from_source: np.ndarray | None = None,
    scene_from_camera: np.ndarray | None = None,
) -> dict:
    return {
        "schema": SCENE_CALIBRATION_V2,
        "name": "test-metric",
        "source_frame": "depth_optical",
        "scene_frame": "scene",
        "scene_from_source": np.eye(4).tolist()
        if scene_from_source is None
        else scene_from_source.tolist(),
        "scene_from_camera": np.eye(4).tolist()
        if scene_from_camera is None
        else scene_from_camera.tolist(),
        "floor_plane_scene": [0.0, 1.0, 0.0, -0.2],
        "camera": {
            "width": 80,
            "height": 60,
            "fx": 100.0,
            "fy": 120.0,
            "cx": 39.25,
            "cy": 28.75,
            "zNear": 0.01,
            "zFar": 5.0,
        },
        "provenance": {"method": "unit-test"},
        "fingerprints": {"camera_info": "abc"},
    }


def load_v2(document: dict):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "calibration.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return load_calibration(path)


class StaticTopViewGeometryTests(unittest.TestCase):
    def test_bootstrap_calibration_maps_optical_depth_downward(self) -> None:
        calibration = load_calibration(PROJECT_ROOT / "configs/episode18_topview_calibration.json")
        source = np.array([[0.1, 0.2, 0.4], [0.1, 0.2, 0.41]], dtype=np.float32)
        mapped = apply_calibration(source, calibration)

        np.testing.assert_allclose(mapped[0], [0.7, 0.46, 0.9], atol=1e-6)
        self.assertLess(mapped[1, 1], mapped[0, 1])

    def test_unversioned_v1_metadata_and_mapping_remain_unchanged(self) -> None:
        path = PROJECT_ROOT / "configs/episode18_topview_calibration.json"
        source_data = json.loads(path.read_text(encoding="utf-8"))
        calibration = load_calibration(path)

        self.assertIsNone(calibration.schema)
        self.assertFalse(calibration.is_metric)
        self.assertEqual(calibration_metadata(calibration)["camera"], source_data["camera"])
        self.assertNotIn("schema", calibration_metadata(calibration))
        points = np.array([[0.1, 0.2, 0.4, 99.0]], dtype=np.float64)
        expected = np.empty((1, 3), dtype=np.float32)
        expected[:, 0] = points[:, 0]
        expected[:, 1] = -points[:, 2]
        expected[:, 2] = points[:, 1]
        expected = expected * np.float32(2.0) + np.asarray([0.5, 1.26, 0.5], dtype=np.float32)
        np.testing.assert_array_equal(apply_calibration(points, calibration), expected)

    def test_v2_uses_rigid_scene_from_source_and_records_metadata(self) -> None:
        angle = np.radians(90.0)
        scene_from_source = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0, 0.4],
                [np.sin(angle), np.cos(angle), 0.0, -0.2],
                [0.0, 0.0, 1.0, 0.7],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        calibration = load_v2(v2_document(scene_from_source=scene_from_source))
        point = np.array([[0.2, 0.1, -0.3]], dtype=np.float32)

        np.testing.assert_allclose(apply_calibration(point, calibration), [[0.3, 0.0, 0.4]], atol=1e-7)
        np.testing.assert_allclose(
            transform_points(apply_calibration(point, calibration), rigid_inverse(scene_from_source)),
            point,
            atol=1e-7,
        )
        self.assertEqual(
            calibration.camera["scene_from_camera"], calibration.scene_from_camera.tolist()
        )
        metadata = calibration_metadata(calibration)
        self.assertEqual(metadata["schema"], SCENE_CALIBRATION_V2)
        self.assertEqual(metadata["scene_frame"], "scene")
        self.assertEqual(metadata["provenance"], {"method": "unit-test"})
        self.assertEqual(metadata["fingerprints"], {"camera_info": "abc"})

    def test_plane_transform_preserves_incident_points(self) -> None:
        angle = np.radians(35.0)
        target_from_source = np.array(
            [
                [1.0, 0.0, 0.0, 0.2],
                [0.0, np.cos(angle), -np.sin(angle), -0.4],
                [0.0, np.sin(angle), np.cos(angle), 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        plane_source = np.array([0.0, 0.0, 1.0, -0.3])
        points_source = np.array([[0.0, 0.0, 0.3], [0.4, -0.2, 0.3]])
        plane_target = transform_plane(plane_source, target_from_source)
        points_target = transform_points(points_source, target_from_source)

        self.assertAlmostEqual(float(np.linalg.norm(plane_target[:3])), 1.0)
        np.testing.assert_allclose(points_target @ plane_target[:3] + plane_target[3], 0.0, atol=1e-9)

    def test_exact_intrinsic_projection_unprojection_round_trip(self) -> None:
        camera = v2_document()["camera"]
        points = np.array([[0.12, -0.08, 0.7], [-0.3, 0.25, 1.8]])
        projected = project_camera_points(points, camera)
        recovered = unproject_camera_points(projected[:, :2], projected[:, 2], camera)

        np.testing.assert_allclose(recovered, points, atol=1e-12)
        self.assertAlmostEqual(projected[0, 0], camera["fx"] * points[0, 0] / points[0, 2] + camera["cx"])
        self.assertAlmostEqual(projected[0, 1], camera["fy"] * points[0, 1] / points[0, 2] + camera["cy"])

    def test_v2_rasterization_uses_camera_pose_and_asymmetric_intrinsics(self) -> None:
        scene_from_camera = np.eye(4)
        scene_from_camera[:3, 3] = [0.5, -0.2, 0.1]
        calibration = load_v2(v2_document(scene_from_camera=scene_from_camera))
        camera = dict(calibration.camera)
        optical = np.array([[0.0, 0.0, 1.0]])
        scene = transform_points(optical, scene_from_camera)

        depth, nearest, right, up, forward, position = rasterize_depth(scene, 80, 60, camera)
        expected_x = int(np.floor(camera["cx"]))
        expected_y = int(np.floor(camera["cy"]))
        self.assertEqual(int(nearest[expected_y, expected_x]), 0)
        recovered = depth_to_pointcloud(
            depth,
            nearest,
            right,
            up,
            forward,
            position,
            camera,
            frame="camera_optical",
        )
        expected = unproject_camera_points(
            np.array([[expected_x + 0.5, expected_y + 0.5]]), np.array([1.0]), camera
        )
        np.testing.assert_allclose(recovered, expected, atol=1e-6)
        recovered_scene = depth_to_pointcloud(
            depth,
            nearest,
            right,
            up,
            forward,
            position,
            camera,
            frame="world",
        )
        np.testing.assert_allclose(
            recovered_scene, transform_points(expected, scene_from_camera), atol=1e-6
        )

    def test_floor_visual_diagnostics_reproduce_geometry_and_frames(self) -> None:
        fill = {
            "mode": "floor",
            "axis": "y",
            "direction": "negative",
            "floor_plane_scene": [0.0, 1.0, 0.0, -0.2],
            "floor_clearance_m": 0.01,
            "floor_min_thickness_m": 0.001,
            "floor_max_thickness_m": 0.2,
            "plane_parallel_epsilon": 1e-8,
            "scene_bounds": {"min": [-1.0, 0.0, -1.0], "max": [1.0, 1.0, 1.0]},
        }
        points = np.array([
            [0.0, 0.25, 0.0],
            [0.0, 0.205, 0.0],
            [1.2, 0.3, 0.0],
        ], dtype=np.float32)
        segmentation = floor_segmentation_diagnostics(points, fill)
        np.testing.assert_array_equal(segmentation["kept_mask"], [True, False, False])
        self.assertEqual(segmentation["counts"]["at_or_below_clearance"], 1)
        self.assertEqual(segmentation["counts"]["outside_scene_bounds"], 1)

        surface = np.array([[0.0, 0.25, 0.0], [0.1, 0.3, 0.0]], dtype=np.float32)
        starts, ends, thicknesses = floor_column_geometry(surface, fill)
        np.testing.assert_allclose(starts, surface)
        np.testing.assert_allclose(ends[:, 1], 0.2, atol=1e-7)
        np.testing.assert_allclose(thicknesses, [0.05, 0.1], atol=1e-7)
        samples = floor_plane_samples(np.array(fill["floor_plane_scene"]), np.array([-1.0, 0.0, -1.0]), np.array([1.0, 1.0, 1.0]), 5)
        np.testing.assert_allclose(samples @ np.array([0.0, 1.0, 0.0]) - 0.2, 0.0, atol=1e-7)

        document = v2_document()
        scene_from_tag = np.eye(4)
        scene_from_tag[:3, 3] = [0.1, 0.2, 0.3]
        document["provenance"] = {
            "tag_family": "36h11",
            "tag_id": 4,
            "tag_edge_size_m": 0.04,
            "scene_from_tag": scene_from_tag.tolist(),
        }
        transforms = dict(calibration_reference_transforms(load_v2(document)))
        self.assertIn("scene", transforms)
        self.assertIn("depth_optical", transforms)
        self.assertIn("camera", transforms)
        self.assertIn("tag 36h11:4", transforms)
        np.testing.assert_allclose(transforms["tag 36h11:4"], scene_from_tag)

    def test_v2_rejects_scale_and_reflection(self) -> None:
        scaled = np.eye(4)
        scaled[0, 0] = 1.01
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            load_v2(v2_document(scene_from_source=scaled))

        reflected = np.eye(4)
        reflected[0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "determinant"):
            load_v2(v2_document(scene_from_source=reflected))

    def test_zbuffer_retains_nearest_depth_and_unprojects(self) -> None:
        camera = {
            "position": [0.0, 0.0, 0.0],
            "lookAt": [0.0, 0.0, 1.0],
            "up": [0.0, 1.0, 0.0],
            "fieldOfView": 90.0,
            "zNear": 0.01,
            "zFar": 10.0,
        }
        points = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        depth, nearest, right, up, forward, position = rasterize_depth(points, 100, 100, camera)

        self.assertEqual(int(nearest[50, 50]), 1)
        self.assertAlmostEqual(float(depth[50, 50]), 1.0)
        optical = depth_to_pointcloud(depth, nearest, right, up, forward, position, camera, frame="camera_optical")
        self.assertEqual(optical.shape, (1, 3))
        self.assertAlmostEqual(float(optical[0, 2]), 1.0)

    def test_metrics_detect_depth_perturbation_without_alignment(self) -> None:
        reference = np.array([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0]], dtype=np.float32)
        identical = fixed_grid_metrics(reference, reference.copy(), cell_size=0.005)
        shifted_depth = fixed_grid_metrics(reference, reference + [0.0, 0.0, 0.01], cell_size=0.005)

        self.assertEqual(identical["footprint_iou"], 1.0)
        self.assertEqual(identical["depth_median_absolute_error"], 0.0)
        self.assertAlmostEqual(shifted_depth["depth_bias"], -0.01, places=6)
        self.assertAlmostEqual(shifted_depth["depth_p95_absolute_error"], 0.01, places=6)

    def test_metrics_include_candidate_footprint_beyond_reference_bounds(self) -> None:
        reference = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        candidate = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]], dtype=np.float32)
        metrics = fixed_grid_metrics(reference, candidate, cell_size=0.005)

        self.assertEqual(metrics["reference_cells"], 1)
        self.assertEqual(metrics["candidate_cells"], 2)
        self.assertEqual(metrics["intersection_cells"], 1)
        self.assertEqual(metrics["union_cells"], 2)
        self.assertEqual(metrics["footprint_iou"], 0.5)
        self.assertAlmostEqual(metrics["grid_upper_xy"][0], 0.06, places=6)
        self.assertAlmostEqual(metrics["grid_lower_xy"][0] % 0.005, 0.0, places=6)

    def test_grid_cells_are_origin_anchored_across_candidate_bounds(self) -> None:
        reference = np.array([[0.001, 0.001, 1.0], [0.011, 0.001, 1.0]], dtype=np.float32)
        baseline = fixed_grid_metrics(reference, reference, cell_size=0.005)
        expanded = fixed_grid_metrics(reference, np.vstack([reference, [[0.2, 0.2, 1.0]]]), cell_size=0.005)

        self.assertEqual(baseline["reference_cells"], expanded["reference_cells"])
        self.assertEqual(baseline["grid_lower_xy"], expanded["grid_lower_xy"])

    def test_surface_samples_are_reserved_when_budget_permits(self) -> None:
        surface = np.array([[0.0, 0.0, 0.1], [0.01, 0.0, 0.1]], dtype=np.float32)
        voxels = np.array([[0.0, 0.0, 0.12], [0.01, 0.0, 0.12]], dtype=np.float32)
        sampled = sample_surface_preserving_particles(surface, voxels, count=4, voxel_size=0.001, seed=7)

        np.testing.assert_array_equal(sampled[: len(surface)], surface)
        self.assertEqual(sampled.shape, (4, 3))


if __name__ == "__main__":
    unittest.main()
