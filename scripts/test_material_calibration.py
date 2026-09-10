#!/usr/bin/env python3
"""Fast unit tests for effective Young's-modulus calibration utilities."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np

try:
    from material_calibration import (
        LOSS_COMPONENTS,
        aggregate_loss_results,
        aggregate_named_windows,
        boundary_expansion,
        cache_completion_status,
        canonical_json_hash,
        evaluate_scalar_search,
        file_sha256,
        flat_minimum_diagnostics,
        logarithmic_candidates,
        one_sided_nearest_distances,
        partial_view_loss,
        refinement_candidates,
        sequence_fingerprint,
        visible_optical_points,
        whole_window_bootstrap_interval,
    )
    from calibrate_youngs_modulus import (
        MANIFEST_SCHEMA,
        _subprocess_failure_reason,
        _validate_tool_geometry_document,
        execute_candidate_window,
        invalid_state_failure_reason,
        load_invalid_state_diagnostic,
        render_command,
        score_evaluation_artifact,
        validate_manifest,
    )
    from visualize_material_calibration import write_report
except ImportError:
    from .material_calibration import (
        LOSS_COMPONENTS,
        aggregate_loss_results,
        aggregate_named_windows,
        boundary_expansion,
        cache_completion_status,
        canonical_json_hash,
        evaluate_scalar_search,
        file_sha256,
        flat_minimum_diagnostics,
        logarithmic_candidates,
        one_sided_nearest_distances,
        partial_view_loss,
        refinement_candidates,
        sequence_fingerprint,
        visible_optical_points,
        whole_window_bootstrap_interval,
    )
    from .calibrate_youngs_modulus import (
        MANIFEST_SCHEMA,
        _subprocess_failure_reason,
        _validate_tool_geometry_document,
        execute_candidate_window,
        invalid_state_failure_reason,
        load_invalid_state_diagnostic,
        render_command,
        score_evaluation_artifact,
        validate_manifest,
    )
    from .visualize_material_calibration import write_report


class PartialViewLossTests(unittest.TestCase):
    def setUp(self) -> None:
        self.initial = np.ones((4, 4), dtype=np.float64)
        self.valid = np.ones((4, 4), dtype=bool)

    def loss(self, observed, simulation, observed_valid=None, simulation_valid=None, observed_points=None, simulation_points=None):
        observed_valid = self.valid if observed_valid is None else observed_valid
        simulation_valid = self.valid if simulation_valid is None else simulation_valid
        observed_points = visible_optical_points(observed, observed_valid, 60.0) if observed_points is None else observed_points
        simulation_points = visible_optical_points(simulation, simulation_valid, 60.0) if simulation_points is None else simulation_points
        return partial_view_loss(
            observed,
            observed_valid,
            simulation,
            simulation_valid,
            self.initial,
            self.valid,
            self.initial,
            self.valid,
            observed_points,
            simulation_points,
            depth_scale_m=0.1,
            distance_scale_m=0.1,
            min_common_pixels=1,
            min_observed_pixels=1,
            min_simulation_pixels=1,
            min_observed_points=1,
            min_simulation_points=1,
        )

    def test_visible_points_use_exact_v2_intrinsics(self):
        depth = np.full((4, 4), 2.0)
        valid = np.zeros((4, 4), dtype=bool)
        valid[1, 2] = True
        camera = {"width": 4, "height": 4, "fx": 8.0, "fy": 4.0, "cx": 1.5, "cy": 1.5}
        np.testing.assert_allclose(visible_optical_points(depth, valid, camera), [[0.25, 0.0, 2.0]])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            visible_optical_points(depth, valid, {**camera, "width": 5})

    def test_exact_match_is_zero(self):
        result = self.loss(self.initial, self.initial)
        self.assertTrue(result["valid"])
        self.assertEqual(result["weighted_total"], 0.0)
        self.assertEqual(result["components"], {name: 0.0 for name in LOSS_COMPONENTS})
        self.assertEqual(result["support"]["depth_change_pixels"], 16)

    def test_changed_depth_has_normalized_robust_error(self):
        observed = self.initial + 0.1
        simulation = self.initial + 0.2
        result = self.loss(observed, simulation)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["components"]["depth_change"], 0.5, places=12)
        self.assertGreater(result["components"]["observed_to_simulation_distance"], 0.0)

    def test_partial_overlap_reports_iou_and_real_coverage(self):
        observed_valid = np.zeros((4, 4), dtype=bool)
        simulation_valid = np.zeros((4, 4), dtype=bool)
        observed_valid[0, :2] = True
        simulation_valid[0, 1:3] = True
        result = self.loss(self.initial, self.initial, observed_valid, simulation_valid)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["components"]["mask_iou"], 2.0 / 3.0)
        self.assertAlmostEqual(result["components"]["real_coverage"], 0.5)
        self.assertEqual(result["support"]["common_current_pixels"], 1)

    def test_zero_overlap_is_invalid_not_zero(self):
        observed_valid = np.zeros((4, 4), dtype=bool)
        simulation_valid = np.zeros((4, 4), dtype=bool)
        observed_valid[0, 0] = True
        simulation_valid[3, 3] = True
        result = self.loss(self.initial, self.initial, observed_valid, simulation_valid)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["weighted_total"])
        self.assertIsNone(result["components"]["depth_change"])
        self.assertEqual(result["components"]["mask_iou"], 1.0)
        self.assertEqual(result["components"]["real_coverage"], 1.0)
        self.assertIn("depth-change common support 0", result["failure_reason"])

    def test_empty_observation_is_invalid(self):
        empty = np.zeros((4, 4), dtype=bool)
        result = self.loss(self.initial, self.initial, empty, self.valid)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["weighted_total"])
        self.assertIn("observed support 0", result["failure_reason"])

    def test_one_sided_distance_ignores_extra_visible_simulation_points(self):
        observed_points = np.array([[0.0, 0.0, 0.0]])
        simulation_points = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        distances = one_sided_nearest_distances(observed_points, simulation_points)
        np.testing.assert_array_equal(distances, [0.0])
        result = self.loss(
            self.initial,
            self.initial,
            observed_points=observed_points,
            simulation_points=simulation_points,
        )
        self.assertEqual(result["components"]["observed_to_simulation_distance"], 0.0)


class AggregationAndSearchTests(unittest.TestCase):
    @staticmethod
    def valid_result(total: float) -> dict:
        return {
            "valid": True,
            "failure_reason": None,
            "weighted_total": total,
            "components": {name: total for name in LOSS_COMPONENTS},
        }

    def test_aggregation_preserves_invalidity(self):
        invalid = {
            "valid": False,
            "failure_reason": "no support",
            "weighted_total": None,
            "components": {name: None for name in LOSS_COMPONENTS},
        }
        strict = aggregate_loss_results([self.valid_result(1.0), invalid])
        self.assertFalse(strict["valid"])
        self.assertIsNone(strict["weighted_total"])
        permissive = aggregate_loss_results([self.valid_result(1.0), invalid], require_all=False)
        self.assertTrue(permissive["valid"])
        self.assertEqual(permissive["weighted_total"], 1.0)

    def test_named_window_aggregation_uses_manifest_weights(self):
        windows = [
            {"name": "a", "split": "training", "weight": 1.0},
            {"name": "b", "split": "training", "weight": 3.0},
            {"name": "held", "split": "validation", "weight": 1.0},
        ]
        result = aggregate_named_windows(
            {"a": self.valid_result(1.0), "b": self.valid_result(3.0)}, windows, "training"
        )
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["weighted_total"], 2.5)
        self.assertEqual(result["window_names"], ["a", "b"])

    def test_candidate_and_refinement_ordering(self):
        candidates = logarithmic_candidates(100.0, 10_000.0, 3, include=[2000.0, 100.0])
        self.assertEqual(candidates, [100.0, 1000.0, 2000.0, 10000.0])
        refined = refinement_candidates(candidates, 2000.0, subdivisions=2)
        self.assertEqual(refined, sorted(refined))
        self.assertEqual(len(refined), 2)
        self.assertAlmostEqual(refined[0], np.sqrt(1000.0 * 2000.0), places=7)
        self.assertAlmostEqual(refined[1], np.sqrt(2000.0 * 10000.0), places=7)

    def test_boundary_expansion_and_hard_limit(self):
        expanded = boundary_expansion([100.0, 1000.0], 1000.0, [10.0, 10_000.0], 10.0)
        self.assertTrue(expanded["expanded"])
        self.assertEqual(expanded["bounds"], [100.0, 10_000.0])
        stopped = boundary_expansion([100.0, 10_000.0], 10_000.0, [10.0, 10_000.0], 10.0)
        self.assertFalse(stopped["expanded"])
        self.assertTrue(stopped["hit_hard_limit"])
        self.assertIn("hard upper", stopped["reason"])

    def test_flat_minimum_diagnostic(self):
        flat = flat_minimum_diagnostics(
            {100.0: 1.0, 200.0: 1.005, 1000.0: 2.0},
            relative_tolerance=0.01,
            absolute_tolerance=0.0,
            minimum_log10_span=0.2,
        )
        self.assertTrue(flat["flat"])
        sharp = flat_minimum_diagnostics(
            {100.0: 1.0, 200.0: 1.2, 1000.0: 2.0},
            relative_tolerance=0.01,
            absolute_tolerance=0.0,
            minimum_log10_span=0.2,
        )
        self.assertFalse(sharp["flat"])

    def test_synthetic_scalar_search_recovers_known_minimum(self):
        known = 2500.0
        search = evaluate_scalar_search(
            lambda value: float(np.log(value / known) ** 2),
            100.0,
            10_000.0,
            3,
            refinement_rounds=4,
            subdivisions=4,
        )
        recovered = search["best"]["youngs_modulus_pa"]
        self.assertLess(abs(np.log(recovered / known)), 0.03)
        self.assertGreater(len(search["candidates"]), 3)


class CacheAndBootstrapTests(unittest.TestCase):
    def test_canonical_hash_ignores_object_key_order(self):
        first = {"b": [2, 3], "a": {"x": 1}}
        second = {"a": {"x": 1}, "b": [2, 3]}
        self.assertEqual(canonical_json_hash(first), canonical_json_hash(second))
        self.assertNotEqual(canonical_json_hash(first), canonical_json_hash({"a": {"x": 2}, "b": [2, 3]}))

    def test_complete_cache_requires_metadata_and_every_file(self):
        expected = {"schema": "cache/v1", "cache_key": "abc"}
        actual = {"schema": "cache/v1", "cache_key": "abc", "status": "complete"}
        complete = cache_completion_status(expected, actual, ["a.json", "b.json"], ["a.json", "b.json"])
        self.assertTrue(complete["complete"])
        missing = cache_completion_status(expected, actual, ["a.json"], ["a.json", "b.json"])
        self.assertFalse(missing["complete"])
        self.assertEqual(missing["missing_files"], ["b.json"])
        mismatch = cache_completion_status(expected, {**actual, "cache_key": "wrong"}, ["a.json", "b.json"], ["a.json", "b.json"])
        self.assertFalse(mismatch["complete"])
        self.assertIn("cache_key", mismatch["failure_reason"])

    def test_whole_window_bootstrap_is_deterministic(self):
        losses = {
            1000.0: [0.0, 2.0, 2.0],
            2000.0: [1.0, 0.0, 1.0],
            4000.0: [2.0, 1.0, 0.0],
        }
        first = whole_window_bootstrap_interval(losses, samples=250, confidence=0.9, seed=17)
        second = whole_window_bootstrap_interval(losses, samples=250, confidence=0.9, seed=17)
        self.assertEqual(first, second)
        self.assertLessEqual(first["lower_pa"], first["median_pa"])
        self.assertLessEqual(first["median_pa"], first["upper_pa"])
        self.assertEqual(sum(first["selection_counts"].values()), 250)


class CrashDiagnosticTests(unittest.TestCase):
    def test_subprocess_failure_uses_signal_name(self):
        self.assertIsNone(_subprocess_failure_reason(0))
        self.assertEqual(_subprocess_failure_reason(-11), "command terminated by signal SIGSEGV")
        self.assertEqual(_subprocess_failure_reason(7), "command exited with status 7")

    def test_invalid_state_diagnostic_requires_expected_fields(self):
        diagnostic = {
            "schema": "taichidough/mpm-invalid-state/v1",
            "failure_kind": "pre_p2g_stencil_out_of_bounds",
            "particle_index": 12,
            "substep": 34,
            "sim_time_s": 0.0068,
            "grid_size": 48,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_state.json"
            path.write_text(json.dumps(diagnostic), encoding="utf-8")
            self.assertEqual(load_invalid_state_diagnostic(path), diagnostic)
            self.assertEqual(
                invalid_state_failure_reason(diagnostic, "command exited with status 1"),
                "numerical failure pre_p2g_stencil_out_of_bounds for particle 12 "
                "at substep 34 (t=0.0068s); command exited with status 1",
            )
            path.write_text(json.dumps({"schema": diagnostic["schema"]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing"):
                load_invalid_state_diagnostic(path)


class ManifestValidationTests(unittest.TestCase):
    def test_tool_geometry_accepts_optional_mesh_transform(self):
        identity = np.eye(4).tolist()
        document = {
            "schema": "taichidough/tool-geometry/v1",
            "tools": [
                {"name": "left", "half_extents_m": [.1, .1, .1], "marker_from_collider": identity, "marker_from_mesh": identity},
                {"name": "right", "half_extents_m": [.1, .1, .1], "marker_from_collider": identity},
            ],
        }
        _validate_tool_geometry_document(document)
        invalid = json.loads(json.dumps(document))
        invalid["tools"][0]["marker_from_mesh"][0][0] = -1
        with self.assertRaisesRegex(ValueError, "marker_from_mesh"):
            _validate_tool_geometry_document(invalid)

    def make_manifest(self, root: Path) -> tuple[dict, Path]:
        geometry = root / "tool_geometry.json"
        identity = np.eye(4).tolist()
        geometry.write_text(json.dumps({
            "schema": "taichidough/tool-geometry/v1",
            "tools": [
                {"name": "left", "half_extents_m": [0.1, 0.1, 0.1], "marker_from_collider": identity},
                {"name": "right", "half_extents_m": [0.1, 0.1, 0.1], "marker_from_collider": identity},
            ],
        }))
        particles = root / "particles.npy"
        np.save(particles, np.array([[0.2, 0.3, 0.4]], dtype=np.float32))
        reconstruction = root / "reconstruction_metadata.json"
        reconstruction.write_text(json.dumps({
            "schema": "voxel_dough_reconstruction/v2",
            "frame": 0,
            "object_volume_m3": 2.0,
            "voxel_size": 1.0,
            "voxel_count": 2,
            "calibration_schema": "taichidough/scene-calibration/v2",
            "calibration": {"is_metric": True, "scene_frame": "scene_metric"},
            "array_frames": {"sampled_particles_xyz": "scene_metric"},
            "fill": {"mode": "floor", "floor_plane_scene": [0.0, 1.0, 0.0, -0.2]},
            "outputs": {"sampled_particles_xyz": str(particles)},
        }))
        calibration = root / "calibration.json"
        calibration.write_text("{}")
        simulator = root / "simulator.py"
        simulator.write_text("print('simulator placeholder')\n")
        evaluator = root / "evaluator.py"
        evaluator.write_text("print('evaluator placeholder')\n")
        episode = root / "episode"
        episode.mkdir()
        (episode / "pointclouds_interpolated.pt").write_bytes(b"points")
        (episode / "paths_interpolated.pt").write_bytes(b"paths")
        geometry_hash = file_sha256(geometry)
        simulator_arguments = {
            "--particles": 100,
            "--grid": 16,
            "--dt": 0.001,
            "--substeps-per-frame": 1,
            "--replay-stride": 1,
            "--replay-max-gap": 0.1,
            "--depth-width": 32,
            "--depth-height": 24,
            "--depth-splat-radius": 1,
            "--depth-pointcloud-max-points": 0,
            "--poisson-ratio": 0.3,
            "--viscosity": 0.0,
            "--density": 2.0,
            "--object-mass-kg": 4.0,
            "--gravity": -9.81,
            "--floor-y": 0.2,
            "--floor-friction": 0.4,
            "--floor-absorption": 0.0,
            "--tool-contact-padding": 0.0,
            "--tool-contact-friction": 0.0,
            "--tool-contact-absorption": 0.0,
            "--tool-stickiness": 0.0,
            "--floor-stickiness": 0.0,
            "--floor-plastic-damping-band": 0.0,
            "--velocity-damping": 1.0,
            "--pure-viscoelastic": False,
            "--plastic-min": 0.9,
            "--plastic-max": 1.1,
            "--plastic-velocity-damping": 1.0,
            "--plastic-affine-damping": 1.0,
            "--use-jp": False,
            "--jp-hardening": 0.0,
            "--jp-min": 0.5,
            "--jp-max": 2.0,
            "--cpu": True,
            "--initial-particles-fit": "none",
            "--initial-particles-axis-map": "xyz",
            "--initial-particles-scale": 1.0,
            "--initial-particles-offset": [0.0, 0.0, 0.0],
            "--initial-particles-raw-scene-coordinates": False,
            "--initial-particles-seed": 0,
            "--no-publish-dough-center": True,
        }
        evaluator_arguments = {
            "--view": "deformpath_top",
            "--frame-stride": 1,
            "--pair-tolerance": None,
            "--cell-size": 0.01,
            "--trim-quantile": 0.0,
        }
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "name": "synthetic-manifest",
            "inputs": {
                "geometry": {"path": str(geometry), "sha256": geometry_hash},
                "reconstruction_metadata": {"path": str(reconstruction), "sha256": file_sha256(reconstruction)},
                "initial_particles": {"path": str(particles), "sha256": file_sha256(particles)},
                "sequence": {"episode_dir": str(episode), "fingerprint": sequence_fingerprint(episode)},
                "calibration": {"path": str(calibration), "sha256": file_sha256(calibration)},
                "simulator": {"path": str(simulator), "sha256": file_sha256(simulator)},
                "evaluator": {"path": str(evaluator), "sha256": file_sha256(evaluator)},
            },
            "fixed_parameters": {
                "recorded_setup": {
                    "dough_mass_kg": 4.0,
                    "geometry_description": "synthetic test geometry",
                    "tool_geometry_fingerprint": geometry_hash,
                },
                "simulator_arguments": simulator_arguments,
                "evaluator_arguments": evaluator_arguments,
            },
            "windows": [
                {"name": "train", "split": "training", "start_frame": 0, "end_frame": 2},
                {"name": "held", "split": "validation", "start_frame": 3, "end_frame": 5},
            ],
            "loss": {
                "weights": {name: 1.0 for name in LOSS_COMPONENTS},
                "depth_scale_m": 0.01,
                "distance_scale_m": 0.01,
                "huber_delta": 1.0,
                "min_common_pixels": 1,
                "min_observed_pixels": 1,
                "min_simulation_pixels": 1,
                "min_observed_points": 1,
                "min_simulation_points": 1,
                "nearest_chunk_size": 16,
                "require_all_frames": True,
            },
            "search": {
                "default_youngs_modulus_pa": 2000.0,
                "initial_min_pa": 1000.0,
                "initial_max_pa": 4000.0,
                "hard_min_pa": 100.0,
                "hard_max_pa": 10000.0,
                "coarse_count": 3,
                "refinement_rounds": 1,
                "refinement_subdivisions": 2,
                "boundary_expansion_factor": 2.0,
                "max_boundary_expansions": 2,
                "flat_relative_tolerance": 0.01,
                "flat_absolute_tolerance": 0.0,
                "flat_minimum_log10_span": 0.1,
                "bootstrap_samples": 10,
                "bootstrap_confidence": 0.9,
                "bootstrap_seed": 0,
            },
            "commands": {
                "python": sys.executable,
                "simulator": [
                    "{python}", "{inputs.simulator.path}", "{simulator_arguments}",
                    "--output-dir", "{simulation_dir}", "--youngs-modulus", "{youngs_modulus_pa}",
                    "--replay-episode", "{inputs.sequence.episode_dir}",
                    "--replay-start-frame", "{window.replay_start_frame}", "--replay-end-frame", "{window.end_frame}",
                    "--initial-particles", "{inputs.initial_particles.path}",
                    "--initial-particles-metadata", "{inputs.reconstruction_metadata.path}",
                    "--initial-particles-calibration", "{inputs.calibration.path}",
                    "--tool-geometry", "{inputs.geometry.path}",
                ],
                "evaluator": [
                    "{python}", "{inputs.evaluator.path}", "{evaluator_arguments}",
                    "--episode-dir", "{inputs.sequence.episode_dir}",
                    "--taichi-metadata", "{simulation_metadata}",
                    "--calibration", "{inputs.calibration.path}", "--output-dir", "{evaluation_dir}",
                ],
            },
            "execution": {
                "cache_dir": str(root / "cache"),
                "result_path": str(root / "result.json"),
                "subprocess_timeout_s": 10.0,
            },
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return manifest, manifest_path

    def test_manifest_validation_and_command_list_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, path = self.make_manifest(Path(directory))
            normalized = validate_manifest(manifest, path)
            self.assertEqual(normalized["schema"], MANIFEST_SCHEMA)
            self.assertEqual(
                [window["replay_start_frame"] for window in normalized["windows"]],
                [0, 0],
            )
            rendered = render_command(
                ["python", "{simulator_arguments}", "{window.replay_start_frame}"],
                {
                    "simulator_arguments": ["--grid", "16"],
                    "window": {"replay_start_frame": 0},
                },
            )
            self.assertEqual(rendered, ["python", "--grid", "16", "0"])

    def test_failed_simulator_diagnostic_is_cached_without_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, path = self.make_manifest(root)
            simulator = Path(manifest["inputs"]["simulator"]["path"])
            evaluator = Path(manifest["inputs"]["evaluator"]["path"])
            simulator.write_text(
                "import json\n"
                "import sys\n"
                "from pathlib import Path\n"
                "output = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
                "output.mkdir(parents=True, exist_ok=True)\n"
                "(output / 'invalid_state.json').write_text(json.dumps({\n"
                "    'schema': 'taichidough/mpm-invalid-state/v1',\n"
                "    'failure_kind': 'post_g2p_nonfinite',\n"
                "    'particle_index': 3,\n"
                "    'substep': 17,\n"
                "    'sim_time_s': 0.0034,\n"
                "    'grid_size': 16,\n"
                "}))\n"
                "raise SystemExit(17)\n",
                encoding="utf-8",
            )
            evaluator.write_text(
                "from pathlib import Path\n"
                "Path(__file__).with_name('evaluator_was_run').write_text('yes')\n",
                encoding="utf-8",
            )
            manifest["inputs"]["simulator"]["sha256"] = file_sha256(simulator)
            manifest["inputs"]["evaluator"]["sha256"] = file_sha256(evaluator)
            path.write_text(json.dumps(manifest), encoding="utf-8")
            normalized = validate_manifest(manifest, path)
            execution = execute_candidate_window(
                normalized,
                {"synthetic": "fingerprint"},
                normalized["windows"][0],
                2000.0,
            )
            self.assertEqual(execution["status"], "failed")
            self.assertIsNone(execution["loss"])
            self.assertIn("post_g2p_nonfinite", execution["failure_reason"])
            self.assertEqual(execution["invalid_state"]["particle_index"], 3)
            self.assertFalse((root / "evaluator_was_run").exists())
            cache_metadata = json.loads((Path(execution["cache_dir"]) / "cache_metadata.json").read_text())
            self.assertEqual(cache_metadata["status"], "failed")
            self.assertEqual(cache_metadata["simulator_invalid_state"], execution["invalid_state"])
            self.assertNotIn("evaluator", cache_metadata["commands"])

    def test_held_out_scoring_excludes_earlier_replay_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = np.ones((3, 3), dtype=bool)
            initial = np.ones((3, 3), dtype=np.float64)

            def write_arrays(name: str, observed: np.ndarray, simulation: np.ndarray) -> str:
                path = root / f"{name}.npz"
                np.savez(
                    path,
                    real_depth=observed,
                    real_valid=valid,
                    sim_depth=simulation,
                    sim_valid=valid,
                )
                return path.name

            frames = [
                {
                    "source_frame": 0,
                    "time_s": 0.0,
                    "status": "paired",
                    "arrays": write_arrays("initial", initial, initial),
                },
                {
                    "source_frame": 1,
                    "time_s": 0.1,
                    "status": "paired",
                    "arrays": write_arrays("earlier", initial + 0.1, initial + 0.6),
                },
                {
                    "source_frame": 3,
                    "time_s": 0.3,
                    "status": "paired",
                    "arrays": write_arrays("held_3", initial + 0.2, initial + 0.2),
                },
                {
                    "source_frame": 4,
                    "time_s": 0.4,
                    "status": "paired",
                    "arrays": write_arrays("held_4", initial + 0.3, initial + 0.3),
                },
            ]
            evaluation = root / "dynamic_topview_metrics.json"
            evaluation.write_text(
                json.dumps(
                    {
                        "benchmark": "dynamic-topview-proxy-replay/v1",
                        "camera": {
                            "width": 3,
                            "height": 3,
                            "fx": 4.0,
                            "fy": 4.0,
                            "cx": 1.0,
                            "cy": 1.0,
                        },
                        "frames": frames,
                    }
                )
            )
            loss_settings = {
                "weights": {name: 1.0 for name in LOSS_COMPONENTS},
                "depth_scale_m": 0.1,
                "distance_scale_m": 0.1,
                "huber_delta": 1.0,
                "min_common_pixels": 1,
                "min_observed_pixels": 1,
                "min_simulation_pixels": 1,
                "min_observed_points": 1,
                "min_simulation_points": 1,
                "nearest_chunk_size": 16,
                "require_all_frames": True,
            }
            complete_rollout = score_evaluation_artifact(evaluation, loss_settings)
            held_out = score_evaluation_artifact(
                evaluation,
                loss_settings,
                start_frame=3,
                end_frame=4,
            )
            self.assertGreater(complete_rollout["weighted_total"], 0.0)
            self.assertEqual(held_out["weighted_total"], 0.0)
            self.assertEqual(held_out["scored_source_frame_range"], [3, 4])
            self.assertEqual(
                [frame["source_frame"] for frame in held_out["frames"]],
                [3, 4],
            )

    def test_manifest_reports_missing_fields_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, path = self.make_manifest(Path(directory))
            del manifest["inputs"]["geometry"]
            with self.assertRaisesRegex(ValueError, r"Missing required manifest field manifest\.inputs\.geometry"):
                validate_manifest(manifest, path)


class VisualizationTests(unittest.TestCase):
    def test_dependency_light_report_contains_required_comparisons(self):
        def candidate(value, total, status="complete"):
            aggregate = {
                "valid": status == "complete",
                "weighted_total": total,
                "components": {name: total for name in LOSS_COMPONENTS},
                "failure_reason": None if status == "complete" else "synthetic failure",
            }
            return {
                "youngs_modulus_pa": value,
                "status": status,
                "aggregate": aggregate,
                "windows": {},
                "failure_reason": aggregate["failure_reason"],
            }

        candidates = [candidate(1000.0, 1.0), candidate(2000.0, 0.5), candidate(4000.0, None, "failed")]
        best = candidates[1]
        frozen = {
            "valid": True,
            "weighted_total": 1.2,
            "components": {name: 1.2 for name in LOSS_COMPONENTS},
        }
        result = {
            "schema": "taichidough/material-calibration/v1",
            "status": "accepted",
            "interpretation": "Effective value for a recorded setup.",
            "search": {
                "selection": {"youngs_modulus_pa": 2000.0, "accepted": True, "rejection_reasons": []},
                "bootstrap_interval": {"lower_pa": 1000.0, "upper_pa": 2000.0, "confidence": 0.95},
                "candidates": candidates,
            },
            "validation": {"result": candidate(2000.0, 0.6)},
            "baselines": {
                "best": best,
                "default_e_2000_pa": best,
                "softest_successful": candidates[0],
                "stiffest_successful": best,
                "frozen_training": frozen,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result))
            outputs = write_report(result_path, root / "report")
            page = outputs["html"].read_text()
            self.assertIn("Effective Young’s-modulus identification", page)
            self.assertIn("not a universal dough constant", page)
            self.assertIn("Held-out validation", page)
            self.assertIn("Frozen training loss", page)
            self.assertIn("Failed candidate", page)
            self.assertNotIn("plotly", page.lower())
            self.assertTrue(outputs["candidate_csv"].is_file())
            self.assertTrue(outputs["window_csv"].is_file())
            self.assertTrue(outputs["summary_json"].is_file())


if __name__ == "__main__":
    unittest.main()
