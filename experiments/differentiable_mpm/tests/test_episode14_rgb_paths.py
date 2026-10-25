"""Regression checks for projected mesh silhouettes."""
import json

import numpy as np
import pytest

from experiments.differentiable_mpm.render_episode14_rgb_paths import load_mounts, triangle_silhouette


def test_overlapping_triangles_form_union():
    pixels = np.array([[2., 2.], [18., 2.], [2., 18.]])
    triangles = np.array([[0, 1, 2], [0, 1, 2]])
    mask = triangle_silhouette(pixels, np.ones(3, bool), triangles, (24, 24))
    assert mask[5, 5] == 255
    assert mask[20, 20] == 0


def test_invalid_and_extreme_triangles_are_omitted():
    pixels = np.array([[2., 2.], [18., 2.], [2., 18.]])
    triangles = np.array([[0, 1, 2]])
    assert not triangle_silhouette(pixels, np.array([True, False, True]), triangles, (24, 24)).any()
    pixels[0] = [20000, 20000]
    assert not triangle_silhouette(pixels, np.ones(3, bool), triangles, (24, 24)).any()


def test_empty_mesh_has_empty_silhouette():
    mask = triangle_silhouette(np.empty((0, 2)), np.empty(0, bool), np.empty((0, 3), int), (24, 24))
    assert not mask.any()


def test_candidate_mount_override_preserves_transform(tmp_path):
    matrix = np.eye(4)
    matrix[:3, 3] = [0.01, -0.02, 0.03]
    path = tmp_path / 'geometry.json'
    geometry = {'tools': [{'name': name, 'marker_from_mesh': matrix.tolist()}
                          for name in ['UR5e_spathla', 'gen3_spathla']]}
    path.write_text(json.dumps(geometry))
    assert all(np.array_equal(mount, matrix) for mount in load_mounts(path))
    geometry['tools'].reverse()
    path.write_text(json.dumps(geometry))
    with pytest.raises(ValueError, match='tool order'):
        load_mounts(path)


def test_nonrigid_mount_is_rejected(tmp_path):
    matrix = np.eye(4)
    matrix[0, 0] = 2
    path = tmp_path / 'geometry.json'
    path.write_text(json.dumps({'tools': [{'name': name, 'marker_from_mesh': matrix.tolist()}
                                          for name in ['UR5e_spathla', 'gen3_spathla']]}))
    with pytest.raises(ValueError, match='rigid'):
        load_mounts(path)
