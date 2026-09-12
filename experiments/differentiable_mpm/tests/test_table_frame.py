"""Accepted recorded frames and explicit table calibration in the strict evaluator."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.differentiable_mpm import data
from experiments.differentiable_mpm.config import FrameWindow
from experiments.differentiable_mpm.reference_adapter import get_reference_modules, reference_policy
from experiments.differentiable_mpm.tests.helpers import temporary_directory


class SequenceReached(RuntimeError):
    pass


class TableFrameTests(unittest.TestCase):
    def prepare_until_sequence(self, calibration):
        config = SimpleNamespace(window=lambda split: FrameWindow(1, 2),
                                 paths={"calibration": Path("explicit.json"), "episode": Path("episode")})
        dynamics = SimpleNamespace(load_observation_sequence=lambda path: self.stop_at_sequence())
        helpers = SimpleNamespace(simulator=object(), dynamics=dynamics,
                                  topview=SimpleNamespace(load_calibration=lambda path: calibration))
        with patch.object(data, "verify_input_paths", return_value={}), \
             patch.object(data, "get_reference_modules", return_value=helpers):
            data.prepare_experiment(config, build_sdf=False)

    @staticmethod
    def stop_at_sequence():
        raise SequenceReached("Calibration checks passed")

    def test_recorded_mocap_and_table_aligned_frames_are_accepted(self):
        for frame in ("mocap", "table-aligned"):
            with self.subTest(frame=frame):
                calibration = SimpleNamespace(is_metric=True, schema="taichidough/scene-calibration/v2",
                                              source_frame="mocap", scene_frame=frame, camera={})
                with self.assertRaises(SequenceReached):
                    self.prepare_until_sequence(calibration)

    def test_unrelated_frames_and_nonmetric_calibration_are_rejected(self):
        for source, scene, metric, schema in (
            ("camera", "table-aligned", True, "taichidough/scene-calibration/v2"),
            ("mocap", "unknown", True, "taichidough/scene-calibration/v2"),
            ("mocap", "table-aligned", False, "taichidough/scene-calibration/v2"),
            ("mocap", "table-aligned", True, "taichidough/scene-calibration/v1"),
        ):
            with self.subTest(source=source, scene=scene, metric=metric, schema=schema):
                calibration = SimpleNamespace(is_metric=metric, schema=schema,
                                              source_frame=source, scene_frame=scene, camera={})
                with self.assertRaises(ValueError):
                    self.prepare_until_sequence(calibration)

    def test_strict_evaluator_loads_explicit_table_calibration_unchanged(self):
        with reference_policy("frozen"):
            modules = get_reference_modules()
        evaluator = modules.evaluate_dynamic_topview_match
        self.assertIs(evaluator.load_calibration, modules.topview.load_calibration)
        document = {
            "schema": "taichidough/scene-calibration/v2", "name": "table-test",
            "source_frame": "mocap", "scene_frame": "table-aligned",
            "scene_from_source": np.eye(4).tolist(), "scene_from_camera": np.eye(4).tolist(),
            "floor_plane_scene": [0, 1, 0, 0],
            "camera": {"width": 160, "height": 120, "fx": 100, "fy": 100, "cx": 80, "cy": 60,
                       "zNear": 0.01, "zFar": 5.0},
        }
        with temporary_directory(prefix="table-frame-") as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(document))
            args = SimpleNamespace(calibration=path, episode_dir=Path(directory) / "episode")
            observed = []
            original = evaluator.load_calibration

            def capture_calibration(value):
                result = original(value)
                observed.append(result)
                return result

            with patch.object(evaluator, "load_calibration", side_effect=capture_calibration), \
                 patch.object(evaluator, "load_observation_sequence", side_effect=SequenceReached):
                with self.assertRaises(SequenceReached):
                    evaluator.evaluate(args)
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0].scene_frame, "table-aligned")
            np.testing.assert_array_equal(observed[0].floor_plane_scene, [0, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
