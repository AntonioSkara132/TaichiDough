#!/usr/bin/env python3
"""Regression checks for timestamp pairing, proxy replay and honest diagnostics."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import subprocess
import sys
import importlib.util

import numpy as np

try:
    from deformpath_dynamics import ObservationSequence, ToolReplay, depth_comparison, filter_scene_points, load_scene_point_filter, load_tool_geometry, paired_frame_indices, tool_geometry_from_metadata, tool_geometry_metadata, quaternion_matrix, matrix_quaternion, slerp, surface_summary
    from deformpath_topview import load_calibration
    from evaluate_dynamic_topview_match import validate_frames, boundary_distance
    from visualize_dynamic_topview_benchmark import depth_colors, residual_colors, create_report, normalize_report_context
except ImportError:
    from .deformpath_dynamics import ObservationSequence, ToolReplay, depth_comparison, filter_scene_points, load_scene_point_filter, load_tool_geometry, paired_frame_indices, tool_geometry_from_metadata, tool_geometry_metadata, quaternion_matrix, matrix_quaternion, slerp, surface_summary
    from .deformpath_topview import load_calibration
    from .evaluate_dynamic_topview_match import validate_frames, boundary_distance
    from .visualize_dynamic_topview_benchmark import depth_colors, residual_colors, create_report, normalize_report_context


ROOT = Path(__file__).resolve().parents[1]


class DynamicsTests(unittest.TestCase):
    def setUp(self):
        self.calibration = load_calibration(ROOT / "configs/episode18_topview_calibration.json")

    def sequence(self):
        times = np.array([.0667, .10005, .16683])
        poses = np.zeros((3, 2, 14))
        poses[:, :, 6] = 1
        poses[:, :, 13] = times[:, None]
        poses[:, :, 0] = (times - times[0])[:, None]
        return ObservationSequence([np.ones((4, 3))] * 3, poses, np.ones((3, 2), bool), times,
                                   ["one", "two"], [2, 3, 5], Path("."), "fixture")

    def test_real_gaps_are_preserved(self):
        replay = ToolReplay(self.sequence(), self.calibration, 0, 2)
        np.testing.assert_allclose(replay.times, [0, .03335, .10013])
        poses, velocities = replay.at(.06)
        self.assertAlmostEqual(poses[0, 0], .62, places=6)
        self.assertAlmostEqual(velocities[0, 0], 2.0, places=6)
        with self.assertRaises(ValueError):
            ToolReplay(self.sequence(), self.calibration, 0, 2, max_gap_s=.05)
        with self.assertRaises(ValueError):
            replay.at(.11)

    def test_invalid_tool_stream_is_not_silently_replayed(self):
        sequence = self.sequence()
        sequence.valid[1, 1] = False
        with self.assertRaises(ValueError):
            ToolReplay(sequence, self.calibration, 0, 2)

    def test_quaternion_transform_and_slerp(self):
        for q in ([1, 0, 0, 0], [0, 0, 0, 1], [.2, .3, .4, .5]):
            matrix = quaternion_matrix(np.array(q))
            np.testing.assert_allclose(quaternion_matrix(matrix_quaternion(matrix)), matrix, atol=1e-7)
        np.testing.assert_allclose(slerp(np.array([0., 0, 0, 1]), np.array([0., 0, 0, -1]), .5), [0, 0, 0, 1])
        replay = ToolReplay(self.sequence(), self.calibration, 0, 2)
        np.testing.assert_allclose(quaternion_matrix(replay.at(0)[0][0, 3:]), [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=1e-6)

    def test_per_tool_geometry_applies_full_marker_transform(self):
        marker_from_collider = np.eye(4)
        marker_from_collider[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        marker_from_collider[:3, 3] = [0.01, 0.02, 0.03]
        fixture = {
            "schema": "taichidough/tool-geometry/v1",
            "proxy": False,
            "tools": [
                {"name": "two", "half_extents_m": [0.04, 0.01, 0.06], "marker_from_collider": np.eye(4).tolist()},
                {"name": "one", "half_extents_m": [0.03, 0.01, 0.05], "marker_from_collider": marker_from_collider.tolist()},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tools.json"
            path.write_text(json.dumps(fixture))
            geometry = load_tool_geometry(path, self.sequence().names)
            with self.assertRaises(ValueError):
                load_tool_geometry(path, ["one", "missing"])
        self.assertEqual(geometry.names, ("one", "two"))
        np.testing.assert_allclose(geometry.half_extents_m[0], [0.03, 0.01, 0.05])
        restored = tool_geometry_from_metadata(tool_geometry_metadata(geometry), self.sequence().names)
        np.testing.assert_allclose(restored.half_extents_m, geometry.half_extents_m)
        np.testing.assert_allclose(restored.marker_from_collider, geometry.marker_from_collider)
        replay = ToolReplay(
            self.sequence(),
            self.calibration,
            0,
            2,
            max_gap_s=0.2,
            marker_from_colliders=geometry.marker_from_collider,
        )
        poses, _ = replay.at(0)
        np.testing.assert_allclose(poses[0, :3], [0.52, 1.20, 0.54], atol=1e-6)
        expected_scene_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ marker_from_collider[:3, :3]
        np.testing.assert_allclose(quaternion_matrix(poses[0, 3:]), expected_scene_rotation, atol=1e-6)

    def test_v2_tool_replay_uses_metric_rigid_transform(self):
        scene_from_source = np.array([
            [0.0, -1.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, 0.4],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ])
        document = {
            "schema": "taichidough/scene-calibration/v2",
            "name": "synthetic",
            "source_frame": "depth_optical",
            "scene_frame": "taichi_scene",
            "scene_from_source": scene_from_source.tolist(),
            "scene_from_camera": np.eye(4).tolist(),
            "floor_plane_scene": [0.0, 1.0, 0.0, -0.2],
            "camera": {
                "width": 4,
                "height": 3,
                "fx": 4.0,
                "fy": 4.0,
                "cx": 1.5,
                "cy": 1.0,
                "zNear": 0.01,
                "zFar": 2.0,
            },
        }
        marker_from_colliders = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
        marker_from_colliders[0, 0, 3] = 0.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(document))
            calibration = load_calibration(path)
        replay = ToolReplay(
            self.sequence(),
            calibration,
            0,
            2,
            max_gap_s=0.2,
            marker_from_colliders=marker_from_colliders,
        )
        poses, _ = replay.at(0.0)
        np.testing.assert_allclose(poses[0, :3], [0.3, 0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(quaternion_matrix(poses[0, 3:]), scene_from_source[:3, :3], atol=1e-6)

    def test_reconstruction_scene_filter_reuses_floor_and_bounds(self):
        reconstruction = {
            "schema": "voxel_dough_reconstruction/v2",
            "calibration_fingerprint": "calibration-a",
            "calibration": {"scene_frame": "taichi_scene"},
            "fill": {
                "mode": "floor",
                "floor_plane_scene": [0.0, 1.0, 0.0, -0.2],
                "floor_clearance_m": 0.01,
                "scene_bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
            },
        }
        calibration = SimpleNamespace(
            fingerprint="calibration-a",
            floor_plane_scene=np.array([0.0, 1.0, 0.0, -0.2]),
            scene_frame="taichi_scene",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reconstruction_metadata.json"
            path.write_text(json.dumps(reconstruction))
            simulation = {
                "initialization": {
                    "metadata_path": str(path),
                    "metadata_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            }
            point_filter = load_scene_point_filter(simulation, Path(directory) / "camera_parameters.json", calibration)
        points = np.array([
            [0.5, 0.24, 0.5],
            [0.5, 0.205, 0.5],
            [1.2, 0.4, 0.5],
        ], dtype=np.float32)
        kept, counts = filter_scene_points(points, point_filter)
        np.testing.assert_allclose(kept, points[:1])
        self.assertEqual(counts, {"input": 3, "nonfinite": 0, "kept": 1, "rejected": 2})

    def test_pairing_tolerance_and_monotonicity(self):
        self.assertEqual(paired_frame_indices(np.array([0, .0334, .1002]), np.array([0, .03335, .0667, .10013]), .0002), [0, 1, None, 2])
        with self.assertRaises(ValueError):
            paired_frame_indices(np.array([0, 0]), np.array([0]), .1)
        with self.assertRaises(ValueError):
            paired_frame_indices(np.array([0, .1]), np.array([0]), -1)

    def test_positive_depth_offset_sign_and_color(self):
        real = np.ones((3, 3), dtype=np.float32)
        valid = np.ones((3, 3), dtype=bool)
        metrics = depth_comparison(real, valid, real + .01, valid)
        self.assertAlmostEqual(metrics["pixel_bias_m"], .01, places=6)
        self.assertAlmostEqual(metrics["pixel_p95_m"], .01, places=6)
        rgb = residual_colors(real, valid, real + .01, valid, .02)
        self.assertGreater(int(rgb[0, 0, 0]), int(rgb[0, 0, 2]))

    def test_no_overlap_reports_na(self):
        real = np.ones((3, 3))
        a = np.zeros((3, 3), bool)
        b = a.copy()
        a[0, 0], b[2, 2] = True, True
        metrics = depth_comparison(real, a, real, b)
        self.assertIsNone(metrics["pixel_p95_m"])
        self.assertEqual(metrics["pixel_iou"], 0)
        self.assertEqual(metrics["common_visible_pixels"], 0)
        self.assertAlmostEqual(boundary_distance(a, b), np.sqrt(8))
        self.assertIsNone(boundary_distance(a, np.zeros_like(a)))

    def test_frozen_baseline_exposes_change(self):
        real = np.ones((3, 3))
        mask = np.ones((3, 3), bool)
        initial = depth_comparison(real, mask, real, mask)
        later = depth_comparison(real + .02, mask, real, mask)
        self.assertGreater(later["pixel_mae_m"], initial["pixel_mae_m"])

    def test_global_depth_colors_are_comparable(self):
        a = np.array([[.5, .6]])
        b = np.array([[.5, .9]])
        valid = np.ones_like(a, bool)
        np.testing.assert_array_equal(depth_colors(a, valid, .4, 1)[0, 0], depth_colors(b, valid, .4, 1)[0, 0])

    def test_occupied_area_is_not_bbox_area(self):
        summary = surface_summary(np.array([[0, 0, 1], [.3, .3, 1]]), .1)
        self.assertAlmostEqual(summary["area_m2"], .02)
        self.assertAlmostEqual(summary["bbox_area_m2"], .09)

    def test_dynamic_report_preserves_metric_per_tool_geometry(self):
        report = {
            "calibration": {
                "schema": "taichidough/scene-calibration/v2",
                "name": "metric",
                "source_frame": "depth_optical",
                "scene_frame": "taichi_scene",
            },
            "replay": {
                "tool_geometry": {
                    "schema": "taichidough/tool-geometry/v1",
                    "names": ["one", "two"],
                    "half_extents_m": [[.03, .01, .05], [.04, .02, .06]],
                    "proxy": False,
                }
            },
        }
        normalize_report_context(report)
        self.assertIn("no fitted scale", report["calibration"]["display_description"])
        self.assertFalse(report["replay"]["tool_geometry_proxy"])
        self.assertEqual(report["replay"]["tool_half_extents_by_tool_m"][1], [.04, .02, .06])

    def metadata(self):
        return {"calibration": {"fingerprint": self.calibration.fingerprint}, "parameters": {"dt": .001},
                "frames": [{"sim_time_s": 0, "completed_substeps": 0, "initial_state": True,
                            "views": [dict(self.calibration.camera, name="deformpath_top", width=4, height=3, splat_radius=0)]}]}

    def test_camera_and_step_validation(self):
        metadata = self.metadata()
        validate_frames(metadata, self.calibration, "deformpath_top")
        metadata["frames"][0]["views"][0]["fieldOfView"] += 1
        with self.assertRaises(ValueError):
            validate_frames(metadata, self.calibration, "deformpath_top")
        metadata = self.metadata()
        metadata["frames"][0]["completed_substeps"] = 1
        with self.assertRaises(ValueError):
            validate_frames(metadata, self.calibration, "deformpath_top")
        metadata = self.metadata()
        metadata["calibration"]["fingerprint"] = "wrong"
        with self.assertRaises(ValueError):
            validate_frames(metadata, self.calibration, "deformpath_top")

    def test_evaluator_keeps_unpaired_and_missing_observations(self):
        try:
            import evaluate_dynamic_topview_match as evaluator
        except ImportError:
            from . import evaluate_dynamic_topview_match as evaluator
        sequence = self.sequence()
        sequence.points = [np.array([[0, 0, .48], [.01, 0, .48], [0, .01, .48]], np.float32),
                           np.array([[0, 0, .48]], np.float32), np.empty((0, 3), np.float32)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "depth.npy", np.full((3, 4), .5, np.float32))
            view = dict(self.calibration.camera, name="deformpath_top", width=4, height=3, splat_radius=0,
                        depth_array=str(root / "depth.npy"))
            metadata = {"calibration": {"fingerprint": self.calibration.fingerprint}, "parameters": {"dt": .0002},
                        "replay": {"sequence_fingerprint": sequence.fingerprint, "source_start_frame": 0,
                                   "source_end_frame": 2, "source_time_origin_s": float(sequence.times[0])},
                        "frames": [{"sim_time_s": 0, "completed_substeps": 0, "initial_state": True, "views": [view]},
                                   {"sim_time_s": .1002, "completed_substeps": 501, "initial_state": False, "views": [view]}]}
            metadata_path = root / "camera_parameters.json"
            metadata_path.write_text(json.dumps(metadata))
            args = SimpleNamespace(calibration=ROOT / "configs/episode18_topview_calibration.json",
                                   episode_dir=root, taichi_metadata=metadata_path, view="deformpath_top", frame_stride=1,
                                   cell_size=.003, trim_quantile=0, pair_tolerance=None, output_dir=root / "evaluation")
            with patch.object(evaluator, "load_observation_sequence", return_value=sequence):
                report = json.loads(evaluator.evaluate(args).read_text())
            self.assertEqual([r["status"] for r in report["frames"]], ["paired", "unpaired", "observation_unavailable"])
            self.assertIsNone(report["frames"][1]["metrics"])
            self.assertIsNone(report["frames"][2]["metrics"])
            self.assertIsNone(report["frames"][2]["frozen_baseline"]["pixel_iou"])
            self.assertFalse(report["physical_fidelity_validated"])
            self.assertEqual(report["scope"], "initialization_only")

    @unittest.skipUnless(importlib.util.find_spec("taichi"), "Taichi is unavailable")
    def test_headless_export_uses_completed_simulation_time(self):
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.run([
                sys.executable, str(ROOT / "scripts/taichi_viscoelastic_mpm_scene.py"),
                "--cpu", "--particles", "100", "--grid", "16", "--steps", "1", "--save-initial-frame",
                "--save-depth-pointclouds", "--depth-width", "16", "--depth-height", "12",
                "--view", "deformpath_top", "--no-publish-dough-center", "--output-dir", directory,
            ], capture_output=True, text=True, timeout=90)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            metadata = json.loads((Path(directory) / "camera_parameters.json").read_text())
            self.assertEqual(len(metadata["frames"]), 2)
            initial, advanced = metadata["frames"]
            self.assertEqual(initial["sim_time_s"], 0)
            self.assertTrue(initial["initial_state"])
            self.assertEqual(advanced["completed_substeps"], 8)
            self.assertEqual(advanced["step"], 1)
            self.assertFalse(advanced["initial_state"])
            self.assertAlmostEqual(advanced["sim_time_s"], .0016)

    def test_initialization_report_is_not_dynamic_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            depth, mask = np.ones((3, 4), dtype=np.float32), np.ones((3, 4), bool)
            np.savez(root / "frame.npz", real_depth=depth, real_valid=mask, sim_depth=depth, sim_valid=mask)
            report = {"benchmark": "dynamic-topview-proxy-replay/v1", "scope": "initialization_only",
                      "frames": [{"source_frame": 0, "time_s": 0, "status": "paired", "arrays": "frame.npz"}]}
            path = root / "metrics.json"
            path.write_text(json.dumps(report))
            output = create_report(path, root / "report")
            self.assertIn("INITIALIZATION ONLY", output.read_text())
            manifest = json.loads((output.parent / "visualization_manifest.json").read_text())
            self.assertEqual(manifest["scope"], "initialization_only")
            self.assertTrue((output.parent / "temporal_contact_sheet.png").is_file())
            with self.assertRaises(ValueError):
                create_report(path, root / "report")


if __name__ == "__main__":
    unittest.main()
