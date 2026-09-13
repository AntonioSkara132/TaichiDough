"""Read-only optimizer-history extraction checks; no numerical runtime."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

FILE = Path(__file__).resolve().parents[1] / "plot_parameter_history.py"
SPEC = importlib.util.spec_from_file_location("parameter_history_plotter", FILE)
assert SPEC is not None and SPEC.loader is not None
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        scratch = Path("/home/antonio/.claude/jobs/4c4f1da4/tmp")
        self.directory = tempfile.TemporaryDirectory(dir=scratch if scratch.exists() else None)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.initial = {"type": "evaluation", "stage": "initial", "evaluation": 1,
                        "iteration": 0, "valid": True, "value": 5.0,
                        "parameters": {"youngs_modulus": 100.0}, "elapsed_s": 2.0}
        self.rejected = {"type": "evaluation", "stage": "proposal", "evaluation": 2,
                         "iteration": 1, "valid": True, "value": 6.0,
                         "parameters": {"youngs_modulus": 90.0}, "elapsed_s": 1.0}
        self.accepted = {"type": "evaluation", "stage": "proposal", "evaluation": 3,
                         "iteration": 1, "valid": True, "value": 4.0,
                         "parameters": {"youngs_modulus": 95.0}, "elapsed_s": 1.0}
        self.step = {"type": "step", "iteration": 1, "accepted": True, "accepted_updates": 1,
                     "parameters": {"youngs_modulus": 95.0}, "value_after": 4.0,
                     "attempts": [{"evaluation": 2, "accepted": False, "reason": "insufficient_decrease"},
                                  {"evaluation": 3, "accepted": True}]}
        self.history = [self.initial, self.rejected, self.accepted, self.step]
        self.manifest = {"identity": {"action": "fit", "prepared": {"sequence_fingerprint": "example",
                           "simulation": {"tool_friction_coefficient": 0.3}},
                           "parameter_space": {"fit": ["youngs_modulus"], "bounds": {"youngs_modulus": [10, 1000]}}}}
        self.save()

    def save(self):
        (self.path / "run_manifest.json").write_text(json.dumps(self.manifest))
        (self.path / "result.json").write_text(json.dumps({"optimization": {"history": self.history}}))

    def test_rejected_trial_is_not_retained_state(self):
        run = PLOT.load_run(self.path)
        self.assertEqual([s["parameters"]["youngs_modulus"] for s in run["states"]], [100, 95])
        self.assertEqual([s["decision"] for s in run["evaluations"]], ["initial", "rejected", "accepted"])
        self.assertEqual(run["states"][0]["parameters"]["tool_friction_coefficient"], 0.3)
        self.assertNotIn("tool_friction_coefficient", run["fitted_parameters"])
        self.assertFalse(run["has_wall_time"])

    def test_timestamps_are_not_evaluation_durations(self):
        events = [{"event": "start", "time": "2026-09-13T00:00:00+00:00"},
                  {"type": "initialization", "status": "ready", "time": "2026-09-13T00:00:20+00:00"},
                  {**self.step, "time": "2026-09-13T00:00:50+00:00"}]
        (self.path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
        run = PLOT.load_run(self.path)
        self.assertTrue(run["has_wall_time"])
        self.assertEqual([s["elapsed_wall_s"] for s in run["states"]], [20, 50])

    def test_missing_initial_is_not_fabricated(self):
        self.history.pop(0)
        self.save()
        run = PLOT.load_run(self.path)
        self.assertEqual(len(run["states"]), 1)
        self.assertEqual(run["states"][0]["iteration"], 1)

    def test_mismatched_accepted_evaluation_rejected(self):
        self.step["value_after"] = 3.0
        self.save()
        with self.assertRaisesRegex(ValueError, "disagrees"):
            PLOT.load_run(self.path)

    def test_rejected_update_retains_previous_state(self):
        self.history = [self.initial, self.rejected, {**self.step, "accepted": False,
                        "accepted_updates": 0, "parameters": self.initial["parameters"],
                        "value_after": 5.0, "attempts": [self.step["attempts"][0]]}]
        self.save()
        run = PLOT.load_run(self.path)
        self.assertEqual(run["states"][1]["decision"], "retained_after_rejection")
        self.assertEqual(run["states"][1]["parameters"]["youngs_modulus"], 100)

    def test_duplicate_evaluation_rejected(self):
        self.history.append(copy.deepcopy(self.initial))
        self.save()
        with self.assertRaisesRegex(ValueError, "Duplicate evaluation"):
            PLOT.load_run(self.path)

    def test_saved_state_fallback(self):
        (self.path / "result.json").write_text("{}")
        (self.path / "optimizer_state.json").write_text(json.dumps({"state": {"history": self.history}}))
        run = PLOT.load_run(self.path)
        self.assertEqual(run["history_source"], "optimizer_state.json:state.history")
        self.assertEqual(len(run["states"]), 2)

    def test_synthetic_classification(self):
        self.manifest["identity"]["kind"] = "prestrained-visible-patch-v1"
        self.save()
        self.assertEqual(PLOT.load_run(self.path)["classification"], "synthetic")


if __name__ == "__main__":
    unittest.main()
