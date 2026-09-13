"""Host-only preparation, reduction and dispatch checks for selectable losses."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.differentiable_mpm import calibrate, calibrate_dataset, data, dataset_config
from experiments.differentiable_mpm.checkpoint import CheckpointedRollout
from experiments.differentiable_mpm.config import EXPERIMENT_ROOT, FrameWindow, canonical_hash, load_config
from experiments.differentiable_mpm.loss import LossConfig
from experiments.differentiable_mpm.loss_options import parse_loss_config
from experiments.differentiable_mpm.multi_episode import DatasetObjective, EpisodeExecutionError
from experiments.differentiable_mpm.renderer import Camera
from experiments.differentiable_mpm.replay import observation_schedule
from experiments.differentiable_mpm.run_logging import RunLogger
from experiments.differentiable_mpm.state import DEFAULT_PARAMETERS, ParticleState, SimulationConfig
from experiments.differentiable_mpm.tests import test_checkpoint as recurrence
from experiments.differentiable_mpm.tests import test_calibrate_cli as cli_fixture
from experiments.differentiable_mpm.tests import test_multi_episode as multi_fixture
from experiments.differentiable_mpm.tests.helpers import temporary_directory


class HostOnlyTests(unittest.TestCase):
    def setUp(self):
        guard = patch("taichi.init", side_effect=AssertionError("These tests must not initialize Taichi"))
        guard.start()
        self.addCleanup(guard.stop)


class ConfigDispatchTests(HostOnlyTests):
    def base(self):
        return load_config(EXPERIMENT_ROOT / "configs/episode18_table_aligned_registered_tools.json")

    def test_old_loss_serialization_and_default_are_preserved(self):
        supplied = {"depth_scale_m": 0.01}
        self.assertEqual(parse_loss_config(supplied).as_dict(), LossConfig(**supplied).as_dict())
        config = self.base()
        original = deepcopy(config.loss)
        config.validate()
        self.assertEqual(config.loss, original)

    def test_new_runnable_configs_select_expected_modes(self):
        examples = {"episode18_dpsi_pcd_cd_v1.json": "dpsi-pcd-cd-v1",
                    "episode18_dpsi_pcd_emd_sampled_v1.json": "dpsi-pcd-emd-v1",
                    "episode18_empm_geometry_only_v1.json": "empm-offline-v1"}
        for name, version in examples.items():
            with self.subTest(name=name):
                config = load_config(EXPERIMENT_ROOT / "configs" / name)
                self.assertEqual(parse_loss_config(config.loss).version, version)

    def test_tracking_and_volume_inputs_are_required(self):
        for loss in ({"version": "empm-offline-v1"}, {"version": "dpsi-prt-cd-v1"}):
            with self.subTest(loss=loss):
                config = self.base()
                config.loss = loss
                with self.assertRaisesRegex(ValueError, "Missing loss target inputs"):
                    config.validate()

    def test_supplied_target_files_require_expected_hashes(self):
        config = self.base()
        config.loss = {"version": "dpsi-prt-cd-v1"}
        config.paths["loss_point_targets"] = Path("/not-read/points.npz")
        with self.assertRaisesRegex(ValueError, "requires an expected SHA-256"):
            config.validate()

    def test_current_loss_cannot_silently_ignore_new_targets(self):
        config = self.base()
        config.paths["loss_point_targets"] = Path("/not-read/points.npz")
        config.expected_sha256["loss_point_targets"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "explicit paper loss mode"):
            config.validate()

    def test_common_loss_settings_exclude_per_episode_target_paths(self):
        first, second = self.base(), self.base()
        for index, config in enumerate((first, second)):
            config.loss = {"version": "dpsi-prt-cd-v1"}
            config.paths.update(loss_point_targets=Path(f"/episode{index}/points.npz"),
                                loss_point_targets_metadata=Path(f"/episode{index}/metadata.json"))
            config.expected_sha256.update(loss_point_targets=str(index) * 64,
                                          loss_point_targets_metadata=str(index) * 64)
            config.validate()
        self.assertEqual(dataset_config._common_settings(first), dataset_config._common_settings(second))
        second.loss = {"version": "dpsi-prt-emd-v1"}
        self.assertNotEqual(dataset_config._common_settings(first), dataset_config._common_settings(second))

    def test_saved_single_episode_selection_distinguishes_objective_and_evaluation_frames(self):
        fixture = cli_fixture.CalibrateCliTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        config = deepcopy(fixture.config)
        config.loss = {"version": "dpsi-pcd-cd-v1"}
        output = fixture.root / "paper-selection"
        with fixture.environment(config=config) as environment:
            original = environment.mocks["prepare_experiment"].side_effect
            def prepare(*args, **kwargs):
                prepared = original(*args, **kwargs)
                prepared.provenance.update(objective_frames=[prepared.end_frame],
                                           loss=parse_loss_config(config.loss).as_dict(),
                                           observation_reduction="mean")
                return prepared
            environment.mocks["prepare_experiment"].side_effect = prepare
            status = fixture.run_cli(["fit", "--iterations", "0", "--no-evaluate", "--output-dir", str(output)])
        self.assertEqual(status, 0)
        selected = json.loads((output / "selected_parameters.json").read_text())
        self.assertEqual(selected["selection_frames"], [3])
        self.assertEqual(selected["evaluation_frames"], [1, 2, 3])
        self.assertEqual(selected["loss"]["version"], "dpsi-pcd-cd-v1")

    def test_dataset_selection_preserves_window_and_adds_exact_endpoint(self):
        config = self.base()
        window = FrameWindow(1, 5, 3)
        episode = dataset_config.DatasetEpisode("fixture", config, config.config_path, "training", 1., window)
        shared = {key: config.parameters[key] for key in dataset_config.MATERIAL_NAMES}
        legacy = calibrate_dataset.episode_selection_record(episode, shared)
        self.assertNotIn("objective_frames", legacy)
        config.loss = {"version": "dpsi-pcd-cd-v1"}
        record = calibrate_dataset.episode_selection_record(episode, shared)
        self.assertEqual(record["scored_frames"], [1, 4, 5])
        self.assertEqual(record["objective_frames"], [5])
        self.assertEqual(record["observation_reduction"], "mean")

    def test_make_rollout_retains_legacy_constructor_and_dispatches_paper_sum(self):
        config = self.base()
        prepared = SimpleNamespace(config=config, total_steps=3,
                                   simulation_config=SimulationConfig(**config.simulation),
                                   parameters=config.parameters, sdf=None, camera=None,
                                   initial_state=SimpleNamespace(x=np.zeros((2, 3))), controls=[], observations=[])
        for loss, paper in ((LossConfig(), False),
                            (parse_loss_config({"version": "empm-offline-v1", "tracking_weight": 0}), True)):
            prepared.loss_config = loss
            with patch.object(calibrate, "Stepper"), patch.object(calibrate, "ObservationLoss") as old, \
                    patch.object(calibrate, "make_observation_loss") as new, \
                    patch.object(calibrate, "CheckpointedRollout") as constructor:
                calibrate.make_rollout(prepared)
                self.assertEqual(old.call_count, int(not paper))
                self.assertEqual(new.call_count, int(paper))
                if paper:
                    self.assertEqual(constructor.call_args.kwargs["observation_reduction"], "sum")
                else:
                    self.assertNotIn("observation_reduction", constructor.call_args.kwargs)


class ObservationPreparationTests(HostOnlyTests):
    def setUp(self):
        super().setUp()
        self.directory = temporary_directory("loss-integration-")
        self.addCleanup(self.directory.cleanup)
        self.initial = np.array([[.2, .3, .4], [.25, .3, .4], [.2, .35, .4]])
        self.camera = Camera(8, 6, 10., 10., 4., 3., np.eye(4))
        self.sequence = SimpleNamespace(fingerprint="1" * 64,
                                        times=100. + np.arange(5) * .1,
                                        points=[self.initial.copy() for _ in range(5)])
        self.config = SimpleNamespace(paths={}, expected_sha256={},
                                      simulation={"n_particles": 3, "precision": "f64"},
                                      observation=SimpleNamespace(trim_quantile=0.))
        self.calibration = SimpleNamespace(offset=np.array([.01, .02, .03]))
        self.helpers = SimpleNamespace(
            topview=SimpleNamespace(filter_xyz=lambda points, trim: points,
                                    apply_calibration=lambda points, calibration: points + calibration.offset,
                                    rasterize_depth=Mock(side_effect=AssertionError("No paper target rasterization"))),
            dynamics=SimpleNamespace(filter_scene_points=lambda points, filt: (points, {"kept": len(points)})))
        self.records = {"calibration": {"sha256": "2" * 64}, "initial_particles": {"sha256": "3" * 64}}

    def prepare(self, settings, endpoint=2, scored=(1, 2)):
        return data.prepare_paper_observations(
            self.config, parse_loss_config(settings), sequence=self.sequence,
            calibration=self.calibration, point_filter=None, camera=self.camera,
            frames=observation_schedule(self.sequence.times, np.arange(5), endpoint, .01),
            scored_frames=scored, records=self.records, helpers=self.helpers, initial_positions=self.initial)

    def test_dpsi_scores_explicit_endpoint_and_retains_full_calibrated_cloud(self):
        observations, hashes, filters, metadata = self.prepare({"version": "dpsi-pcd-cd-v1"},
                                                              endpoint=2, scored=(1,))
        self.assertEqual([obs.frame_index for obs in observations], [2])
        self.assertEqual(observations[0].step, 20)
        np.testing.assert_array_equal(observations[0].points_scene, self.initial + self.calibration.offset)
        self.assertAlmostEqual(observations[0].timestamp, .2)
        self.assertEqual(metadata["objective_frames"], [2])
        self.assertEqual(metadata["observation_reduction"], "mean")
        self.assertEqual(set(hashes), {"2"})
        self.assertEqual(filters["2"]["kept"], 3)
        self.helpers.topview.rasterize_depth.assert_not_called()

    def test_empm_scores_all_selected_frames_with_sum(self):
        observations, _, _, metadata = self.prepare({"version": "empm-offline-v1", "tracking_weight": 0})
        self.assertEqual([obs.frame_index for obs in observations], [1, 2])
        self.assertEqual(metadata["observation_reduction"], "sum")
        self.assertEqual(metadata["loss_targets"]["inputs"], {})

    def write_external_volume(self):
        folder = Path(self.directory.name)
        archive, metadata_file = folder / "points.npz", folder / "points.json"
        np.savez_compressed(archive, frame_indices=np.array([1, 4]),
                            timestamps=self.sequence.times[[1, 4]], offsets=np.array([0, 3, 6]),
                            points=np.concatenate([self.initial, self.initial + .02]))
        metadata = {"schema": "taichidough/loss-point-targets/v1", "units": "m",
                    "coordinate_frame": "scene", "timestamp_reference": "sequence",
                    "sequence_fingerprint": self.sequence.fingerprint, "calibration_sha256": "2" * 64,
                    "provenance": {"kind": "synthetic_fixture", "description": "Synthetic integration target"},
                    "target_representation": "inferred_volume"}
        metadata_file.write_text(json.dumps(metadata))
        for name, path in (("loss_point_targets", archive), ("loss_point_targets_metadata", metadata_file)):
            self.config.paths[name] = path
            self.config.expected_sha256[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return archive

    def test_future_target_frames_use_full_sequence_time_map_without_reading_clouds(self):
        self.write_external_volume()
        self.sequence.points = [None] * 5
        observations, _, _, metadata = self.prepare({"version": "dpsi-prt-cd-v1"}, endpoint=1, scored=(1,))
        self.assertEqual(observations[0].frame_index, 1)
        self.assertAlmostEqual(observations[0].timestamp, .1)
        np.testing.assert_array_equal(observations[0].points_scene, self.initial)
        self.assertEqual(metadata["loss_targets"]["inputs"]["points"]["available_frames"], [1, 4])
        self.assertEqual(metadata["loss_targets"]["inputs"]["points"]["scored_frames"], [1])

    def test_target_content_hash_change_is_rejected_before_observation_use(self):
        archive = self.write_external_volume()
        archive.write_bytes(archive.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.prepare({"version": "dpsi-prt-cd-v1"}, endpoint=1, scored=(1,))

    def test_assignment_budget_is_rejected_during_input_preparation(self):
        with self.assertRaisesRegex(ValueError, "Assignment allocation refused"):
            self.prepare({"version": "dpsi-pcd-emd-v1", "max_assignment_pairs": 1})

    def test_prepared_target_or_loss_changes_change_identity(self):
        settings = {"version": "dpsi-pcd-cd-v1"}
        first = self.prepare(settings)
        self.sequence.points[2] = self.sequence.points[2] + .001
        changed = self.prepare(settings)
        self.assertNotEqual(canonical_hash(first[1]), canonical_hash(changed[1]))
        sampled = self.prepare({**settings, "target_sample_count": 2})
        self.assertNotEqual(canonical_hash(changed[3]), canonical_hash(sampled[3]))


class ReductionTests(HostOnlyTests):
    def setUp(self):
        super().setUp()
        self.initial = ParticleState.initial([[.3, .4, .5]], np.float64)
        self.controls = np.linspace(0, .1, 8)
        self.observations = [recurrence.Observation(i, s, np.full((1, 3), .2 + .01 * i))
                             for i, s in enumerate([0, 2, 3, 4, 6, 8])]
        self.parameters = dict(DEFAULT_PARAMETERS, youngs_modulus=3.)

    def rollout(self, length, reduction):
        return CheckpointedRollout(recurrence.LinearStepper(length + 1), self.initial, self.controls,
                                  self.observations, recurrence.QuadraticLoss(), length,
                                  observation_reduction=reduction)

    def test_sum_scales_values_parameter_and_initial_adjoints_once(self):
        mean = self.rollout(3, "mean").value_and_gradient(self.parameters)
        summed = self.rollout(3, "sum").value_and_gradient(self.parameters)
        count = len(self.observations)
        self.assertAlmostEqual(summed.value, count * mean.value, places=13)
        np.testing.assert_allclose(list(summed.gradient.values()), count * np.array(list(mean.gradient.values())), atol=1e-13)
        np.testing.assert_allclose(recurrence.flatten(summed.initial_gradient),
                                   count * recurrence.flatten(mean.initial_gradient), atol=1e-13)
        self.assertEqual(summed.diagnostics["observation_gradient_injections"], [1] * count)
        self.assertEqual(summed.diagnostics["observation_reduction"], "sum")
        self.assertNotIn("observation_reduction", mean.diagnostics)

    def test_segment_boundaries_preserve_summed_full_state_gradient(self):
        full = self.rollout(8, "sum").value_and_gradient(self.parameters)
        for length in (1, 2, 3, 5):
            actual = self.rollout(length, "sum").value_and_gradient(self.parameters)
            self.assertAlmostEqual(actual.value, full.value, places=13)
            np.testing.assert_allclose(list(actual.gradient.values()), list(full.gradient.values()), atol=1e-13)

    def test_invalid_reductions_are_rejected(self):
        for reduction in ("average", "endpoint", None, True, []):
            with self.subTest(reduction=reduction), self.assertRaisesRegex(ValueError, "observation_reduction"):
                self.rollout(3, reduction)


class DatasetAndReportingTests(HostOnlyTests):
    def paper_episodes(self):
        episodes = [multi_fixture.FakeEpisode("short", 2, frames=2),
                    multi_fixture.FakeEpisode("long", 6, frames=20)]
        for ep in episodes:
            ep.config.loss = {"version": "empm-offline-v1", "tracking_weight": 0}
        return episodes

    def test_dataset_weights_sequence_totals_without_extra_frame_scaling(self):
        episodes = self.paper_episodes()
        records = {"short": multi_fixture.evaluation(8, 1, 2, {"observation_reduction": "sum"}),
                   "long": multi_fixture.evaluation(20, 5, 10, {"observation_reduction": "sum"})}
        objective = DatasetObjective(multi_fixture.dataset(*episodes), lambda ep, parameters, grad: records[ep.id])
        result = objective(multi_fixture.SHARED)
        self.assertEqual(result.value, 17.)
        self.assertEqual(result.gradient, {"youngs_modulus": 4., "viscosity": 8.})
        self.assertEqual(result.diagnostics["objective_version"], "weighted-episode-objectives-v1")
        self.assertEqual(result.diagnostics["episode_observation_reduction"], "sum")

    def test_mixed_loss_modes_and_wrong_worker_reduction_are_rejected(self):
        episodes = self.paper_episodes()
        episodes[1].config.loss = {"version": "dpsi-pcd-cd-v1"}
        with self.assertRaisesRegex(ValueError, "common loss definition"):
            DatasetObjective(multi_fixture.dataset(*episodes), lambda *args: None)
        episodes = self.paper_episodes()
        objective = DatasetObjective(multi_fixture.dataset(*episodes), lambda *args: multi_fixture.evaluation())
        with self.assertRaisesRegex(EpisodeExecutionError, "unexpected observation reduction"):
            objective(multi_fixture.SHARED)

    def test_sum_logging_labels_actual_reduction_and_retains_mean_statistic(self):
        logger = object.__new__(RunLogger)
        logger.emit = Mock()
        result = SimpleNamespace(value=6., diagnostics={"observation_reduction": "sum"},
                                 frames=[{"components": {"geometry": 2.}}, {"components": {"geometry": 4.}}])
        logger.objective_result(result)
        self.assertIn("raw component sums", logger.emit.call_args.args[1])
        self.assertEqual(logger.emit.call_args.kwargs["component_sums"], {"geometry": 6.})
        self.assertEqual(logger.emit.call_args.kwargs["component_means"], {"geometry": 3.})


if __name__ == "__main__":
    unittest.main()
