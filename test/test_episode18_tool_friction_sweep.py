# pyright: reportMissingImports=false

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sweep_episode18_tool_friction import (
    FIXED_TOOL_CONTACT_PADDING_M,
    FIXED_YOUNGS_MODULUS_PA,
    FRICTION_VALUES,
    REPLAY_END_FRAME,
    build_case_argv,
    friction_cases,
    validate_baseline,
)


class Episode18ToolFrictionSweepTests(unittest.TestCase):
    def baseline(self):
        return [
            "python3", "sim.py", "--youngs-modulus", "2000", "--grid", "48",
            "--tool-collision", "sdf", "--tool-sdf-resolution", "64",
            "--replay-start-frame", "0", "--replay-end-frame", "387", "--replay-stride", "4",
            "--tool-contact-padding", "0", "--tool-contact-friction", "0.2", "--output-dir", "/old",
            "--replay-episode", "/episode", "--initial-particles-calibration", "/calibration.json",
            "--dt", "0.0002", "--unrelated", "keep-me",
        ]

    def test_three_requested_values_and_exact_rewrites(self):
        self.assertEqual([case["tool_contact_friction"] for case in friction_cases()], list(FRICTION_VALUES))
        argv = build_case_argv(self.baseline(), 0.3, Path("/simulation"))
        self.assertEqual(argv[argv.index("--youngs-modulus") + 1], format(FIXED_YOUNGS_MODULUS_PA, ".12g"))
        self.assertEqual(argv[argv.index("--tool-contact-padding") + 1], format(FIXED_TOOL_CONTACT_PADDING_M, ".17g"))
        self.assertEqual(argv[argv.index("--tool-contact-friction") + 1], "0.3")
        self.assertEqual(argv[argv.index("--replay-stride") + 1], "1")
        self.assertEqual(argv[argv.index("--replay-end-frame") + 1], str(REPLAY_END_FRAME))
        self.assertEqual(argv[argv.index("--output-dir") + 1], "/simulation")
        self.assertEqual(argv[argv.index("--unrelated") + 1], "keep-me")

    def test_rejects_non_sdf_or_unrequested_friction(self):
        baseline = self.baseline()
        baseline[baseline.index("--tool-collision") + 1] = "none"
        with self.assertRaisesRegex(ValueError, "sdf"):
            validate_baseline(baseline)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            build_case_argv(self.baseline(), 0.15, Path("/simulation"))


if __name__ == "__main__":
    unittest.main()
