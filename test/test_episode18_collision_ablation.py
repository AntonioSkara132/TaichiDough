# pyright: reportMissingImports=false

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_episode18_e2000_collision_ablation import (
    condition_argv,
    selected_skin,
    validate_profile,
)


class Episode18CollisionAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.profile, cls.baseline = validate_profile(cls.root / "configs" / "episode18_e2000_collision_ablation.json")

    def test_profile_pins_full_replay_baseline_and_collision_solids(self):
        fixed = self.profile["fixed_expectations"]
        self.assertEqual(fixed["youngs_modulus_pa"], 2000.0)
        self.assertEqual(fixed["replay_end_frame"], 387)
        self.assertEqual(fixed["replay_stride"], 4)
        self.assertEqual([item["name"] for item in self.profile["inputs"]["collision_meshes"]], ["UR5e_spathla", "gen3_spathla"])

    def test_skin_requires_explicit_named_assumption(self):
        self.assertEqual(selected_skin("operator-review", 0.0)["contact_padding_m"], 0.0)
        with self.assertRaisesRegex(ValueError, "explicit"):
            selected_skin("auto", 0.01)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            selected_skin("operator-review", -0.01)

    def test_conditions_differ_only_in_declared_collision_arguments(self):
        skin = selected_skin("operator-review", 1 / 384)
        sdf = condition_argv(self.baseline, "sdf", Path("/tmp/sdf"), skin)
        no_tool = condition_argv(self.baseline, "no_tool", Path("/tmp/none"), skin)
        self.assertEqual(sdf[sdf.index("--tool-collision") + 1], "sdf")
        self.assertEqual(no_tool[no_tool.index("--tool-collision") + 1], "none")
        self.assertEqual(sdf[sdf.index("--tool-contact-padding") + 1], format(1 / 384, ".17g"))
        self.assertEqual(no_tool[no_tool.index("--tool-contact-padding") + 1], "0.0")
        self.assertIn("--record-sdf-contact-diagnostics", sdf)
        self.assertNotIn("--record-sdf-contact-diagnostics", no_tool)


if __name__ == "__main__":
    unittest.main()
