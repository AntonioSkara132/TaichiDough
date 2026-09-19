import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.differentiable_mpm.policy_adapter import load_condition_archive


class PolicyControlArchiveTests(unittest.TestCase):
    def write_archive(self, directory, *, poses=None, velocities=None, **manifest_changes):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        poses = np.zeros((4, 2, 7), dtype=np.float64) if poses is None else poses
        poses[:, :, 6] = 1.
        velocities = np.zeros((len(poses), 2, 6), dtype=np.float64) if velocities is None else velocities
        manifest = {
            "schema": "taichidough/mpm-policy-controls/v1",
            "control_dt_s": 0.1,
            "duration_s": 0.4,
            "control_times_s": [0., 0.1, 0.2, 0.3],
            "conditions": {"hold_position": {
                "file": "hold_position.npz",
                "pose_shape": list(poses.shape),
                "velocity_shape": list(velocities.shape),
            }},
        }
        manifest.update(manifest_changes)
        np.savez_compressed(directory / "hold_position.npz", poses=poses, velocities=velocities)
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_loads_valid_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_archive(directory)
            controls, provenance = load_condition_archive(
                directory, "hold_position", expected_dt=0.1, expected_steps=4,
                expected_duration=0.4)
            self.assertEqual(len(controls), 4)
            self.assertEqual(controls[2].poses.shape, (2, 7))
            self.assertEqual(provenance["condition"], "hold_position")
            self.assertEqual(provenance["control_count"], 4)

    def test_rejects_bad_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_archive(directory, schema="wrong")
            with self.assertRaisesRegex(ValueError, "schema"):
                load_condition_archive(directory, "hold_position")

    def test_rejects_nonunit_quaternion(self):
        with tempfile.TemporaryDirectory() as directory:
            poses = np.zeros((4, 2, 7), dtype=np.float64)
            poses[:, :, 3] = 1.
            velocities = np.zeros((4, 2, 6), dtype=np.float64)
            self.write_archive(directory, poses=poses, velocities=velocities)
            with self.assertRaises(ValueError):
                load_condition_archive(directory, "hold_position")

    def test_rejects_too_short_or_wrong_dt(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_archive(directory)
            with self.assertRaisesRegex(ValueError, "dt"):
                load_condition_archive(directory, "hold_position", expected_dt=0.2)
            with self.assertRaisesRegex(ValueError, "enough controls"):
                load_condition_archive(directory, "hold_position", expected_steps=5)

    def test_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_archive(directory)
            manifest_path = Path(directory) / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["conditions"]["hold_position"]["file"] = "../outside.npz"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inside"):
                load_condition_archive(directory, "hold_position")


if __name__ == "__main__":
    unittest.main()
