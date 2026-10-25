"""Focused tests for the manual Episode 14 hand annotation state and UI geometry."""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts.annotate_episode14_hands import (
    HandAnnotationSession, fitted_rect, image_pixel, import_reviewed_json,
)


CHUNK = 'episode14_dynamics_chunk06'


def _index(bag_dir):
    pointcloud = SimpleNamespace(frames=[SimpleNamespace(header_stamp_ns=123),
                                         SimpleNamespace(header_stamp_ns=456)])
    rgb = SimpleNamespace(frames=[SimpleNamespace(ordinal=0, header_stamp_ns=123)])
    return SimpleNamespace(
        bag_dir=bag_dir, source_fingerprint='a' * 64,
        pointcloud_count=2, pointcloud=pointcloud, rgb=rgb,
        rgb_mappings=[SimpleNamespace(exact=True, rgb_ordinal=0),
                      SimpleNamespace(exact=False, rgb_ordinal=0)],
    )


@pytest.fixture
def sidecar(tmp_path):
    bag_dir = tmp_path / 'bag'
    bag_dir.mkdir()
    frames = [
        dict(local_frame=0, raw_pointcloud_ordinal=5, pointcloud_header_stamp_ns='123',
             rgb_ordinal=0, rgb_header_stamp_ns='123', rgb_match='exact', hand_status='unreviewed'),
        dict(local_frame=1, raw_pointcloud_ordinal=6, pointcloud_header_stamp_ns='456',
             rgb_ordinal=None, rgb_header_stamp_ns=None,
             rgb_match='missing_exact_rgb', hand_status='missing_exact_rgb'),
    ]
    # Raw pointcloud ordinals are normally much larger than chunk-local ordinals.
    index = _index(bag_dir)
    index.pointcloud.frames = [SimpleNamespace(header_stamp_ns=0)] * 7
    index.pointcloud.frames[5] = SimpleNamespace(header_stamp_ns=123)
    index.pointcloud.frames[6] = SimpleNamespace(header_stamp_ns=456)
    index.pointcloud_count = 7
    index.rgb_mappings = [SimpleNamespace(exact=False, rgb_ordinal=0)] * 7
    index.rgb_mappings[5] = SimpleNamespace(exact=True, rgb_ordinal=0)
    manifest = {'schema_version': 1, 'camera_resolution': [640, 480],
                'source': {'bag_dir': str(bag_dir), 'bag_fingerprint': 'a' * 64,
                           'calibration_sha256': 'b' * 64},
                'chunks': {CHUNK: {'frames': frames}}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return tmp_path, index


def polygon(mode, *points):
    return {'mode': mode, 'vertices': [list(point) for point in points]}


def test_polygon_draft_save_review_undo_and_resume(sidecar):
    root, index = sidecar
    session = HandAnnotationSession(root, CHUNK, index)
    path = session.path
    png = root / 'chunks' / CHUNK / 'hand' / 'pc0005.png'
    add = polygon('add', (10, 10), (30, 10), (30, 30), (10, 30))
    erase = polygon('erase', (15, 15), (24, 15), (24, 24), (15, 24))
    assert session.edit(0, [add])
    assert session.edit(0, [add, erase])
    assert path.exists() and not png.exists()
    assert session.row(0)['status'] == 'unreviewed'
    assert session.edit(0, [add, erase], 'reviewed_mask')
    mask = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    assert mask.shape == (480, 640)
    assert set(np.unique(mask)) == {0, 255}
    assert mask[11, 11] == 255 and mask[20, 20] == 0
    assert session.undo(0)
    assert session.row(0)['status'] == 'unreviewed' and not png.exists()
    assert HandAnnotationSession(root, CHUNK, index).row(0)['polygons'] == [add, erase]
    resumed = HandAnnotationSession(root, CHUNK, index)
    assert resumed.edit(0, resumed.row(0)['polygons'], 'reviewed_mask')
    assert png.exists()
    assert HandAnnotationSession(root, CHUNK, index).row(0)['mask_sha256']
    png.write_bytes(b'changed')
    with pytest.raises(ValueError, match='missing or changed'):
        HandAnnotationSession(root, CHUNK, index)


def test_reviewed_empty_is_explicit_and_missing_rgb_cannot_be_edited(sidecar):
    root, index = sidecar
    session = HandAnnotationSession(root, CHUNK, index)
    assert session.row(0)['status'] == 'unreviewed'
    assert session.edit(0, [], 'reviewed_empty')
    assert session.row(0)['mask_path'] is None
    assert not (root / 'chunks' / CHUNK / 'hand').exists()
    with pytest.raises(ValueError, match='without exact RGB'):
        session.edit(1, [], 'reviewed_empty')
    assert HandAnnotationSession(root, CHUNK, index).row(1)['status'] == 'missing_exact_rgb'


def test_manifest_and_existing_annotation_source_identity_protection(sidecar):
    root, index = sidecar
    session = HandAnnotationSession(root, CHUNK, index)
    session.edit(0, [], 'reviewed_empty')
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['source']['calibration_sha256'] = 'c' * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='another source or calibration'):
        HandAnnotationSession(root, CHUNK, index)
    with pytest.raises(RuntimeError, match='manifest changed'):
        session.edit(0, [], 'unreviewed')
    manifest['source']['bag_fingerprint'] = 'd' * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='fingerprint'):
        HandAnnotationSession(root, CHUNK, index)


def test_second_open_session_cannot_overwrite_first(sidecar):
    root, index = sidecar
    first = HandAnnotationSession(root, CHUNK, index)
    second = HandAnnotationSession(root, CHUNK, index)
    first.edit(0, [], 'reviewed_empty')
    with pytest.raises(RuntimeError, match='changed on disk'):
        second.edit(0, [polygon('add', (0, 0), (4, 0), (4, 4))])
    assert HandAnnotationSession(root, CHUNK, index).row(0)['status'] == 'reviewed_empty'


def test_import_requires_explicit_approval_and_preserves_existing_work(sidecar, tmp_path):
    root, index = sidecar
    session = HandAnnotationSession(root, CHUNK, index)
    source = session.document
    reviewed = {key: source[key] for key in
                ('schema_version', 'chunk', 'source_fingerprint', 'calibration_sha256')}
    frame = {key: session.frames[0][key] for key in
             ('local_frame', 'raw_pointcloud_ordinal', 'pointcloud_header_stamp_ns',
              'rgb_ordinal', 'rgb_header_stamp_ns')}
    frame.update(status='reviewed_mask', polygons=[
        polygon('add', (0, 0), (10, 0), (10, 10))])
    reviewed['frames'] = [frame]
    incoming = tmp_path / 'reviewed.json'
    incoming.write_text(json.dumps(reviewed))
    assert import_reviewed_json(session, incoming) == 1
    assert import_reviewed_json(session, incoming) == 0
    assert session.row(0)['status'] == 'reviewed_mask'
    assert session.row(1)['status'] == 'missing_exact_rgb'
    assert (root / session.row(0)['mask_path']).exists()
    frame['polygons'] = [polygon('add', (20, 20), (30, 20), (30, 30))]
    incoming.write_text(json.dumps(reviewed))
    with pytest.raises(ValueError, match='different existing manual annotations'):
        import_reviewed_json(session, incoming)
    assert session.row(0)['polygons'][0]['vertices'][0] == [0, 0]
    frame['status'] = 'unreviewed'
    incoming.write_text(json.dumps(reviewed))
    with pytest.raises(ValueError, match='explicit reviewed status'):
        import_reviewed_json(session, incoming)


def test_import_prevalidates_all_frames_before_saving(sidecar, tmp_path):
    root, index = sidecar
    session = HandAnnotationSession(root, CHUNK, index)
    reviewed = {key: session.document[key] for key in
                ('schema_version', 'chunk', 'source_fingerprint', 'calibration_sha256')}
    first = {key: session.frames[0][key] for key in
             ('local_frame', 'raw_pointcloud_ordinal', 'pointcloud_header_stamp_ns',
              'rgb_ordinal', 'rgb_header_stamp_ns')}
    first.update(status='reviewed_empty', polygons=[])
    second = {key: session.frames[1][key] for key in
              ('local_frame', 'raw_pointcloud_ordinal', 'pointcloud_header_stamp_ns',
               'rgb_ordinal', 'rgb_header_stamp_ns')}
    second.update(status='reviewed_empty', polygons=[])
    reviewed['frames'] = [first, second]
    incoming = tmp_path / 'reviewed.json'
    incoming.write_text(json.dumps(reviewed))
    with pytest.raises(ValueError, match='RGB/pointcloud identity'):
        import_reviewed_json(session, incoming)
    assert session.document['annotation_revision'] == 0
    assert not session.path.exists()


def test_click_mapping_preserves_rgb_pixel_coordinates():
    rectangle = fitted_rect((20, 40, 960, 600))
    assert rectangle == (100, 40, 800, 600)
    assert image_pixel((100, 40), rectangle) == (0, 0)
    assert image_pixel((899, 639), rectangle) == (639, 479)
    assert image_pixel((99, 40), rectangle) is None
    assert image_pixel((900, 640), rectangle) is None


def test_pygame_add_review_and_missing_rgb_navigation(sidecar, monkeypatch):
    monkeypatch.setenv('SDL_VIDEODRIVER', 'dummy')
    import pygame
    from scripts.annotate_episode14_hands import HandAnnotationApp

    root, index = sidecar
    pygame.display.init()
    pygame.font.init()
    try:
        session = HandAnnotationSession(root, CHUNK, index)
        reader = SimpleNamespace(get=lambda ordinal: np.zeros((480, 640, 3), np.uint8))
        app = HandAnnotationApp(pygame, session, reader)
        app._render()
        x, y, _, _ = app.image_rect
        for position in [(x + 30, y + 30), (x + 130, y + 30), (x + 130, y + 130)]:
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                                 {'button': 1, 'pos': position}))
        app._events()
        app._key(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RETURN, 'mod': 0}))
        assert session.row(0)['status'] == 'unreviewed'
        app._render()
        app._key(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_s, 'mod': 0}))
        assert session.row(0)['status'] == 'reviewed_mask'
        app._key(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RIGHT, 'mod': 0}))
        assert app.local == 1
        app._render()
        app._key(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_e, 'mod': 0}))
        assert session.row(1)['status'] == 'missing_exact_rgb'
    finally:
        pygame.quit()
