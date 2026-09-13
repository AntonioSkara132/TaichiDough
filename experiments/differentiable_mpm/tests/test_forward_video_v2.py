"""Host-only tests for the dataset-aware forward-video launcher."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from experiments.differentiable_mpm.forward_video_v2 import run
from experiments.differentiable_mpm.parameters import PhysicalParameterSpace


INITIAL = {
    "youngs_modulus": 6000.0,
    "poisson_ratio": 0.47,
    "viscosity": 0.0,
    "plastic_min": 0.9,
    "plastic_max": 1.1,
    "tool_retention": 1.0,
    "floor_retention": 0.7,
    "tool_friction_coefficient": 0.9,
    "tool_stickiness": 0.0,
}
BOUNDS = {
    "youngs_modulus": [2000.0, 300000.0],
    "poisson_ratio": [0.45, 0.49],
    "viscosity": [0.0, 100.0],
    "plastic_min": [0.7, 0.999],
    "plastic_max": [1.001, 1.3],
}
FIT = tuple(BOUNDS)


class InputTests(unittest.TestCase):
    def test_path_overrides(self):
        actual = run.path_overrides(["episode18.episode=relative/recording",
                                     "episode18.calibration=calibration.json"])
        self.assertEqual(set(actual), {"episode18"})
        self.assertEqual(set(actual["episode18"]), {"episode", "calibration"})
        self.assertTrue(Path(actual["episode18"]["episode"]).is_absolute())

    def test_bad_path_overrides(self):
        for values in (["missing-equals"], [".episode=x"], ["e.=x"], ["e.episode="],
                       ["e.episode=a", "e.episode=b"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                run.path_overrides(values)

    def test_empty_executable_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            run.executable("", "simulation-python")

    def test_material_requires_exact_finite_values(self):
        values = {name: INITIAL[name] for name in FIT}
        self.assertEqual(run.material(values), values)
        with self.assertRaises(ValueError):
            run.material({**values, "extra": 1.0})
        with self.assertRaises(ValueError):
            run.material({**values, "viscosity": float("nan")})


class ParameterRecordTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = SimpleNamespace(fingerprint="dataset-fingerprint")

    def tearDown(self):
        self.temporary.cleanup()

    def test_completed_selection(self):
        path = self.root / "selected_parameters.json"
        values = {name: INITIAL[name] for name in FIT}
        path.write_text(json.dumps({
            "schema": "taichidough/dataset-material-selection/v1",
            "dataset_fingerprint": self.dataset.fingerprint,
            "shared_material_parameters": values,
            "training_value": 1.25,
        }))
        selected, provenance = run.selection_parameters(path, self.dataset)
        self.assertEqual(selected, values)
        self.assertEqual(provenance["kind"], "selected_parameters")
        self.assertEqual(len(provenance["sha256"]), 64)

    def test_selection_rejects_another_dataset(self):
        path = self.root / "selected_parameters.json"
        path.write_text(json.dumps({
            "schema": "taichidough/dataset-material-selection/v1",
            "dataset_fingerprint": "other",
            "shared_material_parameters": {name: INITIAL[name] for name in FIT},
        }))
        with self.assertRaisesRegex(ValueError, "different dataset"):
            run.selection_parameters(path, self.dataset)

    def test_active_checkpoint_current_and_best(self):
        space = PhysicalParameterSpace(INITIAL, FIT, "stretch-clamp", bounds=BOUNDS)
        current_values = dict(INITIAL, youngs_modulus=8000.0, viscosity=20.0)
        best_values = dict(INITIAL, youngs_modulus=7000.0, viscosity=10.0)
        current_coordinates = [space._encode(name, current_values[name]) for name in FIT]
        best_coordinates = [space._encode(name, best_values[name]) for name in FIT]
        identity = {
            "action": "dataset-fit",
            "dataset_identity": {"dataset_fingerprint": self.dataset.fingerprint,
                                 "ignore_recompute_mismatch": True},
        }
        manifest = {"identity_sha256": "identity", "identity": identity}
        state = {
            "space": space.settings(), "coordinates": current_coordinates, "best_u": best_coordinates,
            "iterations": 8, "accepted_updates": 8, "evaluations": 9,
            "best_value": 0.16, "current": {"value": 0.17},
        }
        (self.root / "run_manifest.json").write_text(json.dumps(manifest))
        (self.root / "optimizer_state.json").write_text(json.dumps({
            "identity_sha256": "identity", "saved_at": "now", "state": state,
        }))
        current, current_provenance = run.optimizer_parameters(self.root, self.dataset, "current")
        best, best_provenance = run.optimizer_parameters(self.root, self.dataset, "best")
        self.assertAlmostEqual(current["youngs_modulus"], 8000.0)
        self.assertAlmostEqual(current["viscosity"], 20.0)
        self.assertAlmostEqual(best["youngs_modulus"], 7000.0)
        self.assertAlmostEqual(best["viscosity"], 10.0)
        self.assertEqual(current_provenance["objective_value"], 0.17)
        self.assertEqual(best_provenance["objective_value"], 0.16)
        self.assertEqual(best_provenance["accepted_updates"], 8)

    def test_checkpoint_identity_must_match(self):
        (self.root / "run_manifest.json").write_text(json.dumps({
            "identity_sha256": "one",
            "identity": {"action": "dataset-fit",
                         "dataset_identity": {"dataset_fingerprint": self.dataset.fingerprint}},
        }))
        (self.root / "optimizer_state.json").write_text(json.dumps({
            "identity_sha256": "two", "state": {},
        }))
        with self.assertRaisesRegex(ValueError, "identity"):
            run.optimizer_parameters(self.root, self.dataset, "best")


if __name__ == "__main__":
    unittest.main()
