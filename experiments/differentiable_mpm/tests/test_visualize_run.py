"""Focused saved-run visualization checks without Taichi or simulation."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.differentiable_mpm import visualize_run


class SavedRunVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "fit"
        self.evaluation = self.run_dir / "validation_a_strict" / "evaluation"
        self.evaluation.mkdir(parents=True)
        self.metrics = self.evaluation / visualize_run.REPORT_NAME
        arrays_dir = self.evaluation / "arrays"
        arrays_dir.mkdir()
        rows = []
        for index in range(3):
            relative = f"arrays/frame_{index:06d}.npz"
            depth = np.full((4, 5), 0.4 + index * 0.01, dtype=np.float32)
            mask = np.ones((4, 5), dtype=bool)
            np.savez(self.evaluation / relative, real_depth=depth, real_valid=mask,
                     sim_depth=depth + 0.001, sim_valid=mask)
            rows.append({"source_frame": index, "original_source_frame": index,
                         "time_s": index / 30, "status": "paired", "arrays": relative})
        self.report = {"benchmark": "dynamic-topview-proxy-replay/v1", "frames": rows,
                       "camera": {"width": 5, "height": 4},
                       "replay": {"tool_collision": "none"}, "limitations": [],
                       "counts": {"observations": 3, "paired": 3, "post_initial_paired": 2}}
        self.metrics.write_text(json.dumps(self.report))
        self.selection = self.run_dir / "selected_parameters.json"
        self.selection.write_text(json.dumps({"best_parameters": {"youngs_modulus": 300000},
                                              "physics_version": "corrected-v1",
                                              "selection_frames": [1]}))

    def test_directory_or_parameter_record_discovery(self):
        for input_path in (self.run_dir, self.selection):
            root = visualize_run.run_directory(input_path)
            self.assertEqual(visualize_run.choose_report(root, "validation"), self.metrics)
        with self.assertRaisesRegex(ValueError, "No saved training"):
            visualize_run.choose_report(self.run_dir, "training")

    def test_ambiguous_reports_require_explicit_choice(self):
        other = self.run_dir / "validation_b_strict" / "evaluation"
        other.mkdir(parents=True)
        (other / visualize_run.REPORT_NAME).write_text("{}")
        with self.assertRaisesRegex(ValueError, "Multiple saved evaluations"):
            visualize_run.choose_report(self.run_dir, "validation")

    def test_report_and_input_hashes(self):
        _, hashes = visualize_run.validate_saved_arrays(self.metrics)
        self.assertEqual(len(hashes), 3)
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_path_escape_rejected(self):
        self.report["frames"][0]["arrays"] = "../outside.npz"
        self.metrics.write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, "stay inside"):
            visualize_run.validate_saved_arrays(self.metrics)

    def test_nonfinite_valid_depth_rejected(self):
        path = self.evaluation / self.report["frames"][0]["arrays"]
        depth = np.full((4, 5), np.nan, dtype=np.float32)
        mask = np.ones((4, 5), dtype=bool)
        np.savez(path, real_depth=depth, real_valid=mask, sim_depth=depth, sim_valid=mask)
        with self.assertRaisesRegex(ValueError, "Nonfinite depth"):
            visualize_run.validate_saved_arrays(self.metrics)

    def test_actual_report_generation_preserves_inputs_and_labels(self):
        originals = {path: path.read_bytes() for path in self.run_dir.rglob("*") if path.is_file()}
        output = self.root / "viewer"
        self.assertEqual(visualize_run.main(["--run-dir", str(self.selection),
                                           "--output-dir", str(output)]), 0)
        self.assertTrue((output / "index.html").is_file())
        self.assertTrue((output / "comparison_frame_000002.png").is_file())
        document = json.loads((output / "visualization_manifest.json").read_text())
        self.assertEqual([row["scoring_role"] for row in document["frames"]],
                         ["initialization", "material fitting", "outside material-selection frames"])
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)
        with self.assertRaises(SystemExit):
            visualize_run.main(["--run-dir", str(self.run_dir), "--output-dir", str(output)])

    def test_video_sampling_reaches_last_observation(self):
        indices = visualize_run.video_indices(self.report["frames"], 30)
        self.assertEqual(indices.tolist(), [0, 1, 2])
        self.assertEqual(visualize_run.video_indices(self.report["frames"][:1], 30).tolist(), [0])


if __name__ == "__main__":
    unittest.main()
