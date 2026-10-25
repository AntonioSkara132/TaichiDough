"""Episode 14 selection and density-based mass checks."""
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.differentiable_mpm.prepare_episode14_dynamics_temporal import mass_from_density, range_selection


class Episode14PreparationTests(unittest.TestCase):
    def test_density_sets_mass_from_each_reconstruction(self):
        self.assertAlmostEqual(mass_from_density(0.00025, 1200), 0.3)
        self.assertAlmostEqual(mass_from_density(0.0002, 1200), 0.24)
        for volume, density in [(0, 1200), (-1, 1200), (float('nan'), 1200), (1, float('inf'))]:
            with self.assertRaises(ValueError):
                mass_from_density(volume, density)

    def test_first_and_last_ranges_are_excluded_without_mutating_plan(self):
        plan = {'source': {'pointcloud_frame_count': 1578}, 'ranges': [
            {'id': 'first', 'start_pointcloud_ordinal': 0, 'end_pointcloud_ordinal_exclusive': 700},
            {'id': 'stroke', 'start_pointcloud_ordinal': 700, 'end_pointcloud_ordinal_exclusive': 1346},
            {'id': 'last', 'start_pointcloud_ordinal': 1346, 'end_pointcloud_ordinal_exclusive': 1578}]}
        with patch('experiments.differentiable_mpm.prepare_episode14_dynamics_temporal.file_sha256', return_value='hash'):
            result = range_selection(plan, Path('/episode'), Path('/conversion_metadata.json'), Path('/temporal_annotations.json'))
        self.assertEqual([row['id'] for row in result['ranges']], ['stroke'])
        self.assertEqual(result['excluded_ranges']['ids'], ['first', 'last'])
        self.assertEqual(len(plan['ranges']), 3)


if __name__ == '__main__':
    unittest.main()
