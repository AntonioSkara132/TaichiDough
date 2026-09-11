# pyright: reportMissingImports=false

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sweep_replay_sdf_contact_padding import (
    TOOL_NAMES,
    build_case_argv,
    collect_particle_metrics,
    padding_cases,
    rank_cases,
    validate_and_aggregate_debug,
    validate_baseline,
)


class ContactPaddingSweepTests(unittest.TestCase):
    def baseline(self):
        return [
            "python3", "sim.py", "--youngs-modulus", "2000", "--grid", "48",
            "--tool-collision", "sdf", "--tool-sdf-resolution", "64",
            "--replay-start-frame", "0", "--replay-end-frame", "387", "--replay-stride", "4",
            "--tool-contact-padding", "0", "--output-dir", "/old",
            "--replay-episode", "/episode", "--initial-particles", "/particles.npy",
            "--initial-particles-metadata", "/metadata.json", "--initial-particles-calibration", "/calibration.json",
            "--tool-geometry", "/geometry.json", "--unrelated", "keep-me",
        ]

    def replay(self):
        return {"tool_collision": "sdf", "tool_names": list(TOOL_NAMES), "sequence_fingerprint": "abc"}

    def contact_debug(self, padding=0.0):
        branch = lambda candidates, applied, inward: {"contact_candidates": candidates, "applied_responses": applied, "inward_normal_velocity_removed": inward}
        return {
            "schema": "taichidough/replay-sdf-contact-debug/v1", "tool_collision": "sdf",
            "tool_names": list(TOOL_NAMES), "sequence_fingerprint": "abc", "tool_contact_padding_scene_m": padding,
            "frames": [
                {"frame": 0, "source_frame": 0, "sim_time_s": 0.0, "snapshot_without_substep_evaluations": True, "tools": [
                    {"grid_nodes": branch(99, 99, 99), "particles": branch(99, 99, 99)},
                    {"grid_nodes": branch(99, 99, 99), "particles": branch(99, 99, 99)},
                ]},
                {"frame": 1, "source_frame": 1, "sim_time_s": .1, "snapshot_without_substep_evaluations": False, "tools": [
                    {"grid_nodes": branch(5, 3, 2), "particles": branch(4, 2, 1)},
                    {"grid_nodes": branch(0, 0, 0), "particles": branch(3, 3, 2)},
                ]},
            ],
        }

    def test_exact_padding_values_and_isolated_command_rewrites(self):
        fixed = validate_baseline(self.baseline())
        cases = padding_cases(fixed["grid"])
        self.assertEqual([case["id"] for case in cases], ["padding_0dx", "padding_dx_8", "padding_dx_4", "padding_3dx_8", "padding_dx_2"])
        self.assertEqual([case["padding_scene_m"] for case in cases], [0.0, 1 / 384, 1 / 192, 1 / 128, 1 / 96])
        argv = build_case_argv(self.baseline(), cases[2], Path("/simulation"))
        self.assertEqual(argv.count("--tool-contact-padding"), 1)
        self.assertEqual(argv[argv.index("--tool-contact-padding") + 1], format(1 / 192, ".17g"))
        self.assertEqual(argv[argv.index("--replay-end-frame") + 1], "160")
        self.assertEqual(argv[argv.index("--replay-stride") + 1], "1")
        self.assertEqual(argv[argv.index("--output-dir") + 1], "/simulation")
        self.assertEqual(argv[argv.index("--unrelated") + 1], "keep-me")
        self.assertEqual(argv.count("--record-sdf-contact-diagnostics"), 1)

    def test_baseline_rejects_wrong_fixed_configuration(self):
        baseline = self.baseline()
        baseline[baseline.index("--grid") + 1] = "49"
        with self.assertRaisesRegex(ValueError, "grid 48"):
            validate_baseline(baseline)

    def test_debug_aggregation_excludes_snapshot_and_preserves_tools(self):
        result = validate_and_aggregate_debug(self.contact_debug(), self.replay(), 0.0)
        ur = result["totals_by_tool"]["UR5e_spathla"]
        kinova = result["totals_by_tool"]["gen3_spathla"]
        self.assertEqual(ur["grid_nodes"]["contact_candidates"], 5)
        self.assertEqual(ur["particles"]["applied_responses"], 2)
        self.assertEqual(kinova["particles"]["applied_responses"], 3)
        self.assertIsNone(kinova["grid_nodes"]["applied_per_candidate"])
        self.assertEqual(result["first_applied_response"]["UR5e_spathla"]["source_frame"], 1)
        duplicate = self.contact_debug(); duplicate["frames"].append(duplicate["frames"][1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_and_aggregate_debug(duplicate, self.replay(), 0.0)

    def test_particle_metrics_compute_displacement_and_floor_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = root / "initial.npy"; moved = root / "moved.npy"
            np.save(initial, np.array([[0., .1, 0.], [0., .2, 0.]], dtype=np.float32))
            np.save(moved, np.array([[.3, .1, 0.], [0., -.01, .4]], dtype=np.float32))
            metadata = {"parameters": {"floor_y": 0.0}, "frames": [
                {"frame": 0, "source_frame": 0, "sim_time_s": 0., "particles": str(initial)},
                {"frame": 1, "source_frame": 1, "sim_time_s": .1, "particles": str(moved)},
            ]}
            result = collect_particle_metrics(metadata, 1e-6)["final"]
            expected = np.array([.3, math.sqrt(.21 ** 2 + .16)])
            self.assertAlmostEqual(result["mean_displacement_m"], float(expected.mean()))
            self.assertAlmostEqual(result["rms_displacement_m"], float(np.sqrt(np.mean(expected ** 2))))
            self.assertEqual(result["particles_below_floor"], 1)
            self.assertAlmostEqual(result["maximum_floor_penetration_m"], .01)

    def test_rank_only_does_not_select_padding(self):
        cases = [
            {"id": "none", "label": "none", "padding_scene_m": 0., "status": "complete", "flags": [], "metrics": {"contact": {"totals_by_tool": {name: {"particles": {"applied_responses": 0}} for name in TOOL_NAMES}}, "particles": {"final": {"p95_displacement_m": 0., "max_displacement_m": 0., "particles_below_floor": 0}}}},
            {"id": "contact", "label": "contact", "padding_scene_m": .1, "status": "complete", "flags": [], "metrics": {"contact": {"totals_by_tool": {name: {"particles": {"applied_responses": 1}} for name in TOOL_NAMES}}, "particles": {"final": {"p95_displacement_m": .1, "max_displacement_m": .1, "particles_below_floor": 0}}}},
        ]
        ranking = rank_cases(cases)
        self.assertEqual(ranking["selection_status"], "not_requested")
        self.assertIsNone(ranking["selected_padding_scene_m"])
        self.assertEqual(ranking["ordered_case_ids"][0], "contact")


if __name__ == "__main__":
    unittest.main()
