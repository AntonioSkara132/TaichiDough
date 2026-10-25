"""CPU checks for explicit bag0 parts, the pose gap fill and the pose correction in prepare_bag0_dynamics."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scipy.spatial.transform import Rotation

from experiments.differentiable_mpm.prepare_bag0_dynamics import (
    TOOL_NAMES, WALL_LAYERS, corrected_poses, episode_poses, grid_box, part_ranges,
)


def track(stamps_ns, x0=0.0):
    """Raw samples moving 1 m/s along x with the identity rotation."""
    stamps = np.asarray(stamps_ns, dtype=np.int64)
    values = np.zeros((len(stamps), 7))
    values[:, 0] = x0 + stamps / 1e9
    values[:, 6] = 1.0
    return stamps, values, {}


SETTINGS = {"lag_s": 0.0, "max_bracket_gap_s": 0.03}
FILL = {"max_gap_s": 1.0, "check_clearance_above_s": 0.1, "min_clearance_m": 0.02}


class EpisodePosesTests(unittest.TestCase):
    def setUp(self):
        dense = np.arange(0, 2_000_000_000, 10_000_000)
        # The second tool has samples up to 0.49 s and again from 1.21 s.
        sparse = dense[(dense < 500_000_000) | (dense > 1_200_000_000)]
        self.tracks = {TOOL_NAMES[0]: track(dense), TOOL_NAMES[1]: track(sparse, 1.0)}
        self.stamps = np.arange(100_000_000, 1_900_000_000, 33_000_000)

    def test_without_fill_the_gap_stays_invalid(self):
        poses, valid, filled, gaps = episode_poses(self.stamps, self.tracks, SETTINGS)
        inside = (self.stamps > 490_000_000) & (self.stamps < 1_210_000_000)
        self.assertTrue(valid[:, 0].all())
        np.testing.assert_array_equal(valid[:, 1], ~inside)
        self.assertFalse(filled.any())
        self.assertTrue(np.isnan(gaps).all())
        self.assertTrue(np.isnan(poses[inside, 1]).all())

    def test_fill_interpolates_and_flags_the_gap(self):
        poses, valid, filled, gaps = episode_poses(self.stamps, self.tracks, {**SETTINGS, "gap_fill": FILL})
        inside = (self.stamps > 490_000_000) & (self.stamps < 1_210_000_000)
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(filled[:, 1], inside)
        self.assertFalse(filled[:, 0].any())
        np.testing.assert_allclose(gaps[inside, 1], 0.72)
        # Linear motion is reproduced exactly by linear interpolation across the gap.
        np.testing.assert_allclose(poses[:, 1, 0], 1.0 + self.stamps / 1e9)

    def test_fill_respects_max_gap(self):
        _, valid, filled, _ = episode_poses(self.stamps, self.tracks, {**SETTINGS, "gap_fill": {**FILL, "max_gap_s": 0.5}})
        inside = (self.stamps > 490_000_000) & (self.stamps < 1_210_000_000)
        np.testing.assert_array_equal(valid[:, 1], ~inside)
        self.assertFalse(filled.any())


class PartRangesTests(unittest.TestCase):
    def test_ranges_are_inclusive_and_checked(self):
        times = np.arange(20) / 30.0
        valid = np.ones(20, dtype=bool)
        parts = [{"id": "a", "frames": [0, 9]}, {"id": "b", "frames": [10, 19]}]
        self.assertEqual(part_ranges(parts, valid, times), [(0, 10), (10, 20)])
        for frames in ([5, 20], [-1, 3], [4, 4]):
            with self.subTest(frames=frames), self.assertRaisesRegex(ValueError, "outside"):
                part_ranges([{"id": "c", "frames": frames}], valid, times)

    def test_missing_pose_and_frame_gap_fail(self):
        times = np.arange(20) / 30.0
        valid = np.ones(20, dtype=bool)
        valid[12] = False
        with self.assertRaisesRegex(ValueError, r"frames \[12\]"):
            part_ranges([{"id": "a", "frames": [10, 15]}], valid, times)
        times[16:] += 0.2
        with self.assertRaisesRegex(ValueError, "frame gap"):
            part_ranges([{"id": "b", "frames": [13, 18]}], np.ones(20, dtype=bool), times)


class CorrectedPosesTests(unittest.TestCase):
    def setUp(self):
        self.stamps = np.array([0, 100, 150, 200, 300], dtype=np.int64)
        self.poses = np.zeros((5, 2, 7))
        self.poses[:, :, 0] = [[0.0], [1.0], [1.5], [2.0], [3.0]]
        self.poses[:, :, 3:7] = Rotation.from_euler("z", 90, degrees=True).as_quat()
        self.valid = np.ones((5, 2), dtype=bool)
        self.valid[4, 0] = False
        self.poses[4, 0] = np.nan
        correction = np.eye(4)
        correction[:3, 3] = [0.01, 0.0, 0.0]
        rotated = np.eye(4)
        rotated[:3, :3] = Rotation.from_euler("x", 20, degrees=True).as_matrix()
        rotated[:3, 3] = [0.03, 0.0, 0.0]
        self.directory = tempfile.TemporaryDirectory()
        self.file = Path(self.directory.name) / "ep.npz"
        np.savez(self.file, **{f"{TOOL_NAMES[0]}_stamps_ns": np.array([100, 200]),
                               f"{TOOL_NAMES[0]}_marker_correction": np.stack([correction, rotated]),
                               f"{TOOL_NAMES[1]}_stamps_ns": np.zeros(0, dtype=np.int64),
                               f"{TOOL_NAMES[1]}_marker_correction": np.zeros((0, 4, 4))})

    def tearDown(self):
        self.directory.cleanup()

    def test_correction_is_applied_in_the_body_frame_and_interpolated(self):
        poses, used = corrected_poses(self.poses, self.valid, self.stamps, self.file)
        self.assertEqual(used, {TOOL_NAMES[0]: 2, TOOL_NAMES[1]: 0})
        # Held before the first stamp; the body x axis points along mocap +y.
        np.testing.assert_allclose(poses[0, 0, :3], [0.0, 0.01, 0.0], atol=1e-12)
        np.testing.assert_allclose(poses[2, 0, :3], [1.5, 0.02, 0.0], atol=1e-12)
        np.testing.assert_allclose(poses[3, 0, :3], [2.0, 0.03, 0.0], atol=1e-12)
        first = Rotation.from_quat(self.poses[1, 0, 3:7])
        np.testing.assert_allclose(Rotation.from_quat(poses[1, 0, 3:7]).as_matrix(), first.as_matrix(), atol=1e-12)
        midway = Rotation.from_quat(self.poses[2, 0, 3:7]) * Rotation.from_euler("x", 10, degrees=True)
        np.testing.assert_allclose(Rotation.from_quat(poses[2, 0, 3:7]).as_matrix(), midway.as_matrix(), atol=1e-12)
        expected = Rotation.from_quat(self.poses[3, 0, 3:7]) * Rotation.from_euler("x", 20, degrees=True)
        np.testing.assert_allclose(Rotation.from_quat(poses[3, 0, 3:7]).as_matrix(), expected.as_matrix(), atol=1e-12)

    def test_invalid_rows_and_tools_without_stamps_are_unchanged(self):
        poses, _ = corrected_poses(self.poses, self.valid, self.stamps, self.file)
        self.assertTrue(np.isnan(poses[4, 0]).all())
        np.testing.assert_array_equal(poses[:, 1], self.poses[:, 1])


class GridBoxTests(unittest.TestCase):
    def test_walls_stay_slack_cells_beyond_every_stencil(self):
        grid, slack = 48, 3
        points = np.array([[0.40, 0.01, 0.45], [0.55, 0.06, 0.52]])
        lower, cells = grid_box(points, grid, slack)
        upper = np.add(lower, cells)
        base = np.floor(points * grid - 0.5).astype(int)
        self.assertEqual(lower[1], 0)
        # The solver removes outward velocity at nodes I < lower + WALL_LAYERS and I > upper - WALL_LAYERS.
        for axis in (0, 2):
            self.assertLessEqual(lower[axis] + WALL_LAYERS, base[:, axis].min() - slack)
        for axis in range(3):
            self.assertGreaterEqual(upper[axis] - WALL_LAYERS, base[:, axis].max() + 2 + slack)

    def test_box_is_clipped_to_the_cube_and_at_least_eight_cells(self):
        lower, cells = grid_box(np.array([[0.01, 0.01, 0.99]]), 48, 3)
        self.assertEqual(lower[0], 0)
        self.assertEqual(lower[2] + cells[2], 48)
        lower, cells = grid_box(np.array([[0.5, 0.0, 0.5]]), 48, 0)
        self.assertGreaterEqual(min(cells), 8)


if __name__ == "__main__":
    unittest.main()
