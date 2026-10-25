"""Checks for Episode 14 mask projection, annotation rasterization and frame mapping."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.differentiable_mpm.episode14_foreground_masks import (
    frame_rows, point_mask, project_points, rasterize_polygons,
)


def test_project_observed_points_rejects_behind_camera_and_off_image():
    camera = {'fx': 10., 'fy': 10., 'cx': 5., 'cy': 5., 'width': 12, 'height': 12}
    points = np.array([[0., 0., 1.], [.2, -.1, 1.], [0., 0., -.2],
                       [2., 0., 1.], [np.nan, 0., 1.]])
    pixels = project_points(points, np.eye(4), camera)
    np.testing.assert_array_equal(pixels, [[5, 5], [7, 4]])
    mask = point_mask(pixels, (12, 12), radius=1)
    assert mask[5, 5] == 255 and mask[4, 7] == 255
    assert mask[0, 0] == 0
    assert set(np.unique(mask)) == {0, 255}


def test_projection_uses_inverse_scene_transform():
    camera = {'fx': 10., 'fy': 10., 'cx': 5., 'cy': 5., 'width': 12, 'height': 12}
    camera_from_mocap = np.eye(4)
    camera_from_mocap[0, 3] = -1
    np.testing.assert_array_equal(project_points([[1., 0., 1.]], camera_from_mocap, camera), [[5, 5]])


def test_polygon_add_then_erase_and_invalid_vertices():
    rows = [{'mode': 'add', 'vertices': [[1, 1], [10, 1], [10, 10], [1, 10]]},
            {'mode': 'erase', 'vertices': [[3, 3], [6, 3], [6, 6], [3, 6]]}]
    mask = rasterize_polygons(rows, (12, 12))
    assert mask[2, 2] == 255 and mask[4, 4] == 0 and mask[11, 11] == 0
    assert set(np.unique(mask)) == {0, 255}
    with pytest.raises(ValueError, match='outside'):
        rasterize_polygons([{'mode': 'add', 'vertices': [[0, 0], [12, 1], [1, 2]]}], (12, 12))


def test_raw_header_mapping_marks_unmatched_rgb_unknown():
    pc = [SimpleNamespace(header_stamp_ns=s) for s in (100, 200, 300)]
    rgb = [SimpleNamespace(ordinal=0, header_stamp_ns=100),
           SimpleNamespace(ordinal=1, header_stamp_ns=300)]
    mappings = [SimpleNamespace(rgb_ordinal=0, exact=True),
                SimpleNamespace(rgb_ordinal=0, exact=False),
                SimpleNamespace(rgb_ordinal=1, exact=True)]
    bag = SimpleNamespace(pointcloud_count=3, pointcloud=SimpleNamespace(frames=pc),
                          rgb=SimpleNamespace(frames=rgb), rgb_mappings=mappings)
    meta = {'raw_pointcloud_ordinals': [0, 1, 2], 'frame_count': 3,
            'first_retained_header_stamp_ns': '100', 'last_retained_header_stamp_ns': '300'}
    rows = frame_rows(meta, bag)
    assert [r['rgb_match'] for r in rows] == ['exact', 'missing_exact_rgb', 'exact']
    assert rows[1]['rgb_ordinal'] is None and rows[1]['hand_status'] == 'missing_exact_rgb'
    assert rows[0]['hand_status'] == 'unreviewed'
    meta['last_retained_header_stamp_ns'] = '301'
    with pytest.raises(ValueError, match='timestamps'):
        frame_rows(meta, bag)
