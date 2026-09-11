# pyright: reportMissingImports=false

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sweep_episode18_viscosity import (
    FIXED_TOOL_CONTACT_PADDING_M,
    FIXED_YOUNGS_MODULUS_PA,
    REPLAY_END_FRAME,
    VISCOSITY_VALUES_PA_S,
    build_case_argv,
    viscosity_cases,
)


class Episode18ViscositySweepTests(unittest.TestCase):
    def baseline(self):
        return [
            "python3", "sim.py", "--youngs-modulus", "2000", "--grid", "48",
            "--tool-collision", "sdf", "--tool-sdf-resolution", "64",
            "--replay-start-frame", "0", "--replay-end-frame", "387", "--replay-stride", "4",
            "--tool-contact-padding", "0", "--tool-contact-friction", "0.2", "--viscosity", "0",
            "--output-dir", "/old", "--replay-episode", "/episode",
            "--initial-particles-calibration", "/calibration.json", "--dt", "0.0002", "--unrelated", "keep-me",
        ]

    def test_viscosity_values_and_isolated_rewrites(self):
        self.assertEqual([case["viscosity_pa_s"] for case in viscosity_cases()], list(VISCOSITY_VALUES_PA_S))
        argv = build_case_argv(self.baseline(), 2.5, 0.17, Path("/simulation"))
        self.assertEqual(argv[argv.index("--youngs-modulus") + 1], format(FIXED_YOUNGS_MODULUS_PA, ".12g"))
        self.assertEqual(argv[argv.index("--tool-contact-padding") + 1], format(FIXED_TOOL_CONTACT_PADDING_M, ".17g"))
        self.assertEqual(argv[argv.index("--tool-contact-friction") + 1], "0.17")
        self.assertEqual(argv[argv.index("--viscosity") + 1], "2.5")
        self.assertEqual(argv[argv.index("--replay-stride") + 1], "1")
        self.assertEqual(argv[argv.index("--replay-end-frame") + 1], str(REPLAY_END_FRAME))
        self.assertEqual(argv[argv.index("--output-dir") + 1], "/simulation")
        self.assertEqual(argv[argv.index("--unrelated") + 1], "keep-me")

    def test_rejects_unapproved_viscosity_and_negative_friction(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            build_case_argv(self.baseline(), 3.0, 0.2, Path("/simulation"))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            build_case_argv(self.baseline(), 0.0, -0.1, Path("/simulation"))


if __name__ == "__main__":
    unittest.main()
