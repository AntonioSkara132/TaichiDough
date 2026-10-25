#!/usr/bin/env python3
"""Manually mark visible hands on timestamp-matched Episode 14 raw RGB frames.

Run with --root pointing to an existing tool/hand sidecar. Only reviewed frames
produce hand PNGs; this tool never edits the sidecar manifest or chunk datasets.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from datetime import datetime, timezone


SIZE = (640, 480)  # width, height of the recorded RGB and hand mask
FRAME_KEYS = (
    'local_frame', 'raw_pointcloud_ordinal', 'pointcloud_header_stamp_ns',
    'rgb_ordinal', 'rgb_header_stamp_ns', 'rgb_match',
)
STATUSES = {'unreviewed', 'missing_exact_rgb', 'reviewed_empty', 'reviewed_mask'}


def _stamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode('utf-8')


def fitted_rect(container, width=640, height=480):
    """Return an integer (x, y, width, height) that preserves pixel aspect ratio."""
    x, y, available_width, available_height = container
    scale = min(available_width / width, available_height / height)
    draw_width = max(1, round(width * scale))
    draw_height = max(1, round(height * scale))
    return (x + (available_width - draw_width) // 2,
            y + (available_height - draw_height) // 2, draw_width, draw_height)


def image_pixel(position, rectangle, size=SIZE):
    """Map a display click to a recorded RGB pixel, or None outside the image."""
    x, y, width, height = rectangle
    mx, my = position
    if not (x <= mx < x + width and y <= my < y + height):
        return None
    return (min(size[0] - 1, (mx - x) * size[0] // width),
            min(size[1] - 1, (my - y) * size[1] // height))


class HandAnnotationSession:
    """Revisioned annotation state; one JSON file per chunk, with atomic writes."""

    def __init__(self, root, chunk, bag_index):
        self.root = Path(root).expanduser().resolve()
        if not re.fullmatch(r'episode14_dynamics_chunk\d+', chunk):
            raise ValueError(f'Invalid Episode 14 chunk name: {chunk!r}')
        self.chunk = chunk
        self.index = bag_index
        self.path = self.root / 'chunks' / chunk / 'hand_annotations.json'
        self._last_disk_digest = None
        self.undo_stack = {}
        self.manifest = self._read_manifest()
        self._validate_bag(self.manifest)
        self.frames = self.manifest['chunks'][chunk]['frames']
        self.document = self._load_or_new()

    def _read_manifest(self):
        manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('schema_version') != 1 or manifest.get('camera_resolution') != [640, 480]:
            raise ValueError('Expected sidecar manifest v1 with 640x480 camera resolution')
        source = manifest.get('source', {})
        if not isinstance(source, dict) or not all(
                isinstance(source.get(key), str) and source[key]
                for key in ('bag_dir', 'bag_fingerprint', 'calibration_sha256')):
            raise ValueError('Sidecar manifest lacks source identities')
        if self.chunk not in manifest.get('chunks', {}):
            raise ValueError(f'Chunk {self.chunk} is not in this manifest')
        frames = manifest['chunks'][self.chunk].get('frames')
        if not isinstance(frames, list) or not frames:
            raise ValueError('Selected chunk has no frames')
        return manifest

    def _validate_bag(self, manifest):
        source = manifest['source']
        if source['bag_fingerprint'] != self.index.source_fingerprint:
            raise ValueError('ROS bag fingerprint differs from sidecar manifest')
        if Path(source['bag_dir']).expanduser().resolve() != self.index.bag_dir.resolve():
            raise ValueError('Indexed ROS bag directory differs from sidecar manifest')
        for local, row in enumerate(manifest['chunks'][self.chunk]['frames']):
            raw = row.get('raw_pointcloud_ordinal')
            if (row.get('local_frame') != local or type(raw) is not int or
                    not 0 <= raw < self.index.pointcloud_count):
                raise ValueError(f'Invalid pointcloud ordinal for local frame {local}')
            if str(self.index.pointcloud.frames[raw].header_stamp_ns) != row.get('pointcloud_header_stamp_ns'):
                raise ValueError(f'Pointcloud timestamp mismatch at local frame {local}')
            mapping = self.index.rgb_mappings[raw]
            if mapping.exact:
                rgb = self.index.rgb.frames[mapping.rgb_ordinal]
                expected = ('exact', rgb.ordinal, str(rgb.header_stamp_ns), 'unreviewed')
            else:
                expected = ('missing_exact_rgb', None, None, 'missing_exact_rgb')
            actual = tuple(row.get(key) for key in
                           ('rgb_match', 'rgb_ordinal', 'rgb_header_stamp_ns', 'hand_status'))
            if actual != expected:
                raise ValueError(f'RGB mapping mismatch at local frame {local}: {actual}')

    def _new_document(self):
        source = self.manifest['source']
        return {
            'schema_version': 1, 'chunk': self.chunk,
            'source_fingerprint': source['bag_fingerprint'],
            'calibration_sha256': source['calibration_sha256'],
            'annotation_revision': 0, 'created_at_utc': _stamp(),
            'updated_at_utc': _stamp(),
            'frames': [
                {**{key: row[key] for key in FRAME_KEYS},
                 'status': row['hand_status'], 'polygons': [],
                 'mask_path': None, 'mask_sha256': None}
                for row in self.frames
            ],
        }

    def _load_or_new(self):
        if not self.path.exists():
            return self._new_document()
        data = self.path.read_bytes()
        document = json.loads(data)
        self._validate_document(document)
        self._last_disk_digest = _digest(data)
        return document

    def _validate_document(self, document):
        source = self.manifest['source']
        if (document.get('schema_version') != 1 or document.get('chunk') != self.chunk or
                document.get('source_fingerprint') != source['bag_fingerprint'] or
                document.get('calibration_sha256') != source['calibration_sha256']):
            raise ValueError('Existing hand annotations belong to another source or calibration')
        if (type(document.get('annotation_revision')) is not int or
                document['annotation_revision'] < 1):
            raise ValueError('Invalid hand annotation revision')
        rows = document.get('frames')
        if not isinstance(rows, list) or len(rows) != len(self.frames):
            raise ValueError('Annotation frame count differs from manifest')
        for local, (row, expected) in enumerate(zip(rows, self.frames)):
            if not isinstance(row, dict) or any(row.get(key) != expected[key] for key in FRAME_KEYS):
                raise ValueError(f'Annotation frame identity differs at local frame {local}')
            if row.get('status') not in STATUSES or not isinstance(row.get('polygons'), list):
                raise ValueError(f'Invalid annotation status or polygons at local frame {local}')
            if expected['rgb_match'] != 'exact' and row['status'] != 'missing_exact_rgb':
                raise ValueError(f'Cannot annotate frame {local} without exact RGB')
            if expected['rgb_match'] == 'exact' and row['status'] == 'missing_exact_rgb':
                raise ValueError(f'Exact RGB frame {local} marked missing')
            self._rasterize(row['polygons'])  # Validates geometry even on unreviewed frames.
            if row['status'] == 'reviewed_empty' and row['polygons']:
                raise ValueError(f'Reviewed-empty frame {local} has polygons')
            if row['status'] == 'reviewed_mask':
                relative = self._mask_relative(expected['raw_pointcloud_ordinal'])
                if row.get('mask_path') != relative or not isinstance(row.get('mask_sha256'), str):
                    raise ValueError(f'Reviewed hand mask path/hash invalid at frame {local}')
                path = self.root / relative
                if not path.is_file() or _digest(path.read_bytes()) != row['mask_sha256']:
                    raise ValueError(f'Hand mask missing or changed at frame {local}')
                import cv2
                decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                expected_mask = self._rasterize(row['polygons'])
                if (decoded is None or decoded.shape != (480, 640) or
                        not (decoded == expected_mask).all() or not expected_mask.any()):
                    raise ValueError(f'Hand mask is not the expected 640x480 binary polygon mask at frame {local}')
            elif row.get('mask_path') is not None or row.get('mask_sha256') is not None:
                raise ValueError(f'Unreviewed/empty frame {local} refers to a hand mask')

    def _mask_relative(self, raw):
        return f'chunks/{self.chunk}/hand/pc{raw:04d}.png'

    @staticmethod
    def _rasterize(polygons):
        from experiments.differentiable_mpm.episode14_foreground_masks import rasterize_polygons
        return rasterize_polygons(polygons, shape=(480, 640))

    def _commit(self, local, replacement):
        # Lock both the revision check and the writes across annotator processes.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with (self.path.parent / '.hand_annotations.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                return self._commit_locked(local, replacement)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _commit_locked(self, local, replacement):
        if not 0 <= local < len(self.frames):
            raise IndexError(local)
        # A second open annotator must not overwrite edits made since this session loaded.
        data = self.path.read_bytes() if self.path.exists() else None
        if (None if data is None else _digest(data)) != self._last_disk_digest:
            raise RuntimeError('Annotations changed on disk; reopen before saving')
        current = self._read_manifest()
        if (current['source'] != self.manifest['source'] or
                current['chunks'][self.chunk]['frames'] != self.frames):
            raise RuntimeError('Sidecar manifest changed; reopen before saving')
        if any(replacement.get(key) != self.frames[local][key] for key in FRAME_KEYS):
            raise ValueError('Cannot edit the source frame identity')
        status = replacement['status']
        if status not in STATUSES or self.frames[local]['rgb_match'] != 'exact':
            raise ValueError('Cannot edit a frame without exact RGB')
        mask = self._rasterize(replacement['polygons'])
        relative = self._mask_relative(replacement['raw_pointcloud_ordinal'])
        path = self.root / relative
        if status == 'reviewed_mask':
            if not mask.any():
                raise ValueError('Cannot mark an empty mask reviewed; use reviewed-empty')
            import cv2
            success, encoded = cv2.imencode('.png', mask)
            if not success:
                raise OSError('Could not encode hand mask')
            png = encoded.tobytes()
            replacement['mask_path'] = relative
            replacement['mask_sha256'] = _digest(png)
            _atomic_bytes(path, png)
        else:
            if status == 'reviewed_empty' and replacement['polygons']:
                raise ValueError('Clear polygons before marking reviewed-empty')
            replacement['mask_path'] = None
            replacement['mask_sha256'] = None
        candidate = copy.deepcopy(self.document)
        candidate['frames'][local] = replacement
        candidate['annotation_revision'] += 1
        candidate['updated_at_utc'] = _stamp()
        serialized = _json_bytes(candidate)
        _atomic_bytes(self.path, serialized)
        self.document = candidate
        self._last_disk_digest = _digest(serialized)
        if status != 'reviewed_mask':
            path.unlink(missing_ok=True)
        return candidate['annotation_revision']

    def row(self, local):
        return self.document['frames'][local]

    def edit(self, local, polygons, status='unreviewed'):
        previous = copy.deepcopy(self.row(local))
        replacement = copy.deepcopy(previous)
        replacement['polygons'] = copy.deepcopy(polygons)
        replacement['status'] = status
        if replacement['polygons'] == previous['polygons'] and status == previous['status']:
            return False
        self._commit(local, replacement)
        self.undo_stack.setdefault(local, []).append(previous)
        return True

    def undo(self, local):
        history = self.undo_stack.get(local)
        if not history:
            return False
        previous = history[-1]
        self._commit(local, copy.deepcopy(previous))
        history.pop()
        return True


def import_reviewed_json(session, path):
    """Import explicit human decisions; never infer a hand from image pixels.

    Expected JSON: {"schema_version": 1, "chunk": "...",
    "source_fingerprint": "...", "calibration_sha256": "...", "frames": [
      {"local_frame": 0, "raw_pointcloud_ordinal": 0,
       "pointcloud_header_stamp_ns": "...", "rgb_ordinal": 0,
       "rgb_header_stamp_ns": "...", "status": "reviewed_mask",
       "polygons": [{"mode": "add", "vertices": [[x,y], ...]}]}]}
    A frame can instead have status reviewed_empty and an empty polygon list.
    Frames omitted from the input remain untouched and unreviewed.
    """
    incoming = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(incoming, dict) or any(
            incoming.get(key) != session.document[key]
            for key in ('schema_version', 'chunk', 'source_fingerprint', 'calibration_sha256')):
        raise ValueError('Import source, chunk, or calibration differs from annotation session')
    rows = incoming.get('frames')
    if not isinstance(rows, list):
        raise ValueError('Import frames must be a list')
    approved = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or type(row.get('local_frame')) is not int:
            raise ValueError('Imported frame must identify a local frame number')
        local = row['local_frame']
        if local in seen or not 0 <= local < len(session.frames):
            raise ValueError(f'Invalid or repeated imported local frame: {local}')
        seen.add(local)
        expected = session.frames[local]
        if expected['rgb_match'] != 'exact' or any(
                row.get(key) != expected[key] for key in FRAME_KEYS if key != 'rgb_match'):
            raise ValueError(f'Imported RGB/pointcloud identity differs at frame {local}')
        status, polygons = row.get('status'), row.get('polygons')
        if status not in ('reviewed_mask', 'reviewed_empty') or not isinstance(polygons, list):
            raise ValueError(f'Frame {local} needs explicit reviewed status and polygons')
        if status == 'reviewed_empty' and polygons:
            raise ValueError(f'Reviewed-empty frame {local} cannot have polygons')
        mask = session._rasterize(polygons)
        if status == 'reviewed_mask' and not mask.any():
            raise ValueError(f'Reviewed hand polygon is empty at frame {local}')
        current = session.row(local)
        if current['polygons'] != polygons and (
                current['status'] != 'unreviewed' or current['polygons']):
            raise ValueError(f'Frame {local} has different existing manual annotations')
        approved.append((local, polygons, status))
    # Validate every frame before writing any of them. Existing reviewed decisions
    # and polygon drafts cannot be replaced by a different import.
    return sum(session.edit(local, polygons, status) for local, polygons, status in approved)


def _initialize_pygame(pg):
    try:
        pg.display.init()
    except pg.error:
        if os.environ.get('WAYLAND_DISPLAY') and os.environ.get('DISPLAY'):
            pg.display.quit()
            os.environ['SDL_VIDEODRIVER'] = 'x11'
            pg.display.init()
        else:
            raise
    pg.font.init()


class HandAnnotationApp:
    def __init__(self, pg, session, reader, start=0):
        self.pg, self.session, self.reader = pg, session, reader
        self.local = start
        self.mode = 'add'
        self.vertices = []
        self.show_mask = True
        self.show_outlines = True
        self.show_help = True
        self.goto = None
        self.message = 'Only exact RGB frames can be annotated.'
        self.image_rect = (0, 0, 1, 1)
        self.screen = pg.display.set_mode((1180, 820), pg.RESIZABLE)
        pg.display.set_caption('Episode 14 — manual visible-hand annotation')
        self.clock = pg.time.Clock()
        self.font = pg.font.Font(None, 27)
        self.small = pg.font.Font(None, 22)
        self._rgb_key = None
        self._rgb_surface = None
        self._overlay_key = None
        self._overlay_surface = None

    def seek(self, local):
        if self.vertices:
            self.message = 'Finish polygon (Enter) or cancel it (Esc) before navigating.'
            return
        self.local = max(0, min(local, len(self.session.frames) - 1))
        self.message = ''

    def _frame_image(self):
        row = self.session.row(self.local)
        ordinal = row['rgb_ordinal']
        if ordinal is None:
            return None
        if ordinal != self._rgb_key:
            image = self.reader.get(ordinal)
            if image.shape != (480, 640, 3):
                raise ValueError(f'Raw RGB frame {ordinal} has shape {image.shape}, expected (480,640,3)')
            self._rgb_surface = self.pg.image.fromstring(image.tobytes(), SIZE, 'RGB')
            self._rgb_key = ordinal
        return self._rgb_surface

    def _overlay(self, row):
        key = (self.local, self.session.document['annotation_revision'])
        if key != self._overlay_key:
            mask = self.session._rasterize(row['polygons'])
            overlay = self.pg.Surface(SIZE, self.pg.SRCALPHA)
            # One fill and a per-pixel alpha assignment keep erase polygons visible
            # as erased even when they overlap an earlier add polygon.
            overlay.fill((255, 70, 75, 0))
            alpha = self.pg.surfarray.pixels_alpha(overlay)
            alpha[:, :] = (mask.T != 0) * 105
            del alpha
            self._overlay_surface = overlay
            self._overlay_key = key
        return self._overlay_surface

    def _finish_polygon(self):
        if len(self.vertices) < 3:
            self.message = 'Polygon needs at least three vertices.'
            return
        row = self.session.row(self.local)
        polygons = row['polygons'] + [{'mode': self.mode, 'vertices': self.vertices.copy()}]
        self.session.edit(self.local, polygons)
        self.vertices.clear()
        self.message = 'Polygon saved as draft; press S to mark this frame reviewed.'

    def _key(self, event):
        pg = self.pg
        if self.goto is not None:
            if event.key in (pg.K_RETURN, pg.K_KP_ENTER):
                value = self.goto
                self.goto = None
                if value and 0 <= int(value) < len(self.session.frames):
                    self.seek(int(value))
                else:
                    self.message = 'Local frame number is out of range.'
            elif event.key == pg.K_ESCAPE:
                self.goto = None
            elif event.key == pg.K_BACKSPACE:
                self.goto = self.goto[:-1]
            elif event.unicode.isdecimal():
                self.goto += event.unicode
            return
        if event.key == pg.K_ESCAPE:
            if self.vertices:
                self.vertices.clear()
                self.message = 'Current unsaved polygon cancelled.'
            else:
                self.running = False
        elif event.key in (pg.K_RETURN, pg.K_KP_ENTER):
            self._finish_polygon()
        elif event.key == pg.K_LEFT:
            self.seek(self.local - (10 if event.mod & pg.KMOD_SHIFT else 1))
        elif event.key == pg.K_RIGHT:
            self.seek(self.local + (10 if event.mod & pg.KMOD_SHIFT else 1))
        elif event.key == pg.K_HOME:
            self.seek(0)
        elif event.key == pg.K_END:
            self.seek(len(self.session.frames) - 1)
        elif event.key == pg.K_g:
            if not self.vertices:
                self.goto = str(self.local)
        elif event.key == pg.K_m:
            self.mode = 'erase' if self.mode == 'add' else 'add'
            self.message = f'Drawing mode: {self.mode}'
        elif event.key in (pg.K_u, pg.K_z):
            if self.vertices:
                self.vertices.pop()
            elif not self.session.undo(self.local):
                self.message = 'Nothing to undo on this frame in this session.'
        elif event.key == pg.K_x:
            if not self.vertices and self.session.row(self.local)['rgb_match'] == 'exact':
                self.session.edit(self.local, [])
                self.message = 'All polygons cleared; frame is unreviewed.'
        elif event.key == pg.K_e:
            if not self.vertices and self.session.row(self.local)['rgb_match'] == 'exact':
                if self.session.row(self.local)['polygons']:
                    self.message = 'Press X to clear polygons before marking reviewed-empty.'
                else:
                    self.session.edit(self.local, [], 'reviewed_empty')
                    self.message = 'Reviewed-empty saved (no hand PNG).'
        elif event.key == pg.K_s:
            if not self.vertices and self.session.row(self.local)['rgb_match'] == 'exact':
                polygons = self.session.row(self.local)['polygons']
                if polygons:
                    if self.session._rasterize(polygons).any():
                        self.session.edit(self.local, polygons, 'reviewed_mask')
                        self.message = 'Reviewed hand mask saved.'
                    else:
                        self.message = 'Draw visible hand pixels, or clear and press E.'
                else:
                    self.message = 'No polygons; press E to explicitly mark reviewed-empty.'
        elif event.key == pg.K_o:
            self.show_mask = not self.show_mask
        elif event.key == pg.K_p:
            self.show_outlines = not self.show_outlines
        elif event.key == pg.K_h:
            self.show_help = not self.show_help

    def _events(self):
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            elif event.type == pg.VIDEORESIZE:
                self.screen = pg.display.set_mode((max(760, event.w), max(610, event.h)), pg.RESIZABLE)
            elif event.type == pg.KEYDOWN:
                self._key(event)
            elif event.type == pg.MOUSEBUTTONDOWN and self.goto is None:
                if event.button == 3:
                    if self.vertices:
                        self._finish_polygon()
                elif event.button == 2:
                    self.vertices.clear()
                    self.message = 'Current unsaved polygon cancelled.'
                elif event.button == 1 and self.session.row(self.local)['rgb_match'] == 'exact':
                    pixel = image_pixel(event.pos, self.image_rect)
                    if pixel is not None:
                        self.vertices.append(list(pixel))

    def _render(self):
        pg = self.pg
        self.screen.fill((27, 30, 35))
        width, height = self.screen.get_size()
        container = (22, 78, width - 44, height - 235)
        rect = fitted_rect(container)
        self.image_rect = rect
        row = self.session.row(self.local)
        image = self._frame_image()
        if image is not None:
            self.screen.blit(pg.transform.smoothscale(image, rect[2:]), rect[:2])
            if self.show_mask and row['polygons']:
                self.screen.blit(pg.transform.scale(self._overlay(row), rect[2:]), rect[:2])
            if self.show_outlines:
                for polygon in row['polygons']:
                    points = [self._display_point(v) for v in polygon['vertices']]
                    color = (255, 200, 55) if polygon['mode'] == 'add' else (76, 215, 255)
                    pg.draw.lines(self.screen, color, True, points, 2)
                    for point in points:
                        pg.draw.circle(self.screen, color, point, 3)
            if self.vertices:
                points = [self._display_point(v) for v in self.vertices]
                if len(points) > 1:
                    pg.draw.lines(self.screen, (235, 250, 90), False, points, 2)
                for point in points:
                    pg.draw.circle(self.screen, (235, 250, 90), point, 4)
        else:
            self._text('No exact timestamp-matched RGB: annotation disabled.', 35, height // 2, (255, 185, 85))
        pg.draw.rect(self.screen, (140, 147, 155), rect, 1)
        self._text(f'{self.session.chunk}   local {self.local}/{len(self.session.frames)-1}  '
                   f'raw pc{row["raw_pointcloud_ordinal"]:04d}   RGB {row["rgb_ordinal"]}',
                   22, 19, (245, 247, 249))
        self._text(f'Status: {row["status"]}   Mode: {self.mode}   '
                   f'Polygons: {len(row["polygons"])}   Revision: {self.session.document["annotation_revision"]}',
                   22, height - 148, (255, 204, 119))
        if self.show_help:
            controls = ['Left click: vertices   Enter/right click: finish polygon   Middle/Esc: cancel polygon',
                        'M: add/erase   S: review mask   E: reviewed-empty   X: clear   U/Z: undo',
                        'Left/Right: frame   Shift+arrow: 10 frames   G: jump to local frame',
                        'O: mask overlay   P: polygon outlines   H: hide help   Esc: quit']
            for index, line in enumerate(controls):
                self._text(line, 22, height - 120 + 23 * index, (205, 213, 220), small=True)
        if self.goto is not None:
            self._text(f'Jump to local frame: {self.goto}_', 22, height - 177, (125, 235, 180))
        elif self.message:
            self._text(self.message, 22, height - 177, (125, 235, 180))
        pg.display.flip()

    def _display_point(self, point):
        x, y, w, h = self.image_rect
        return (x + int((point[0] + .5) * w / SIZE[0]),
                y + int((point[1] + .5) * h / SIZE[1]))

    def _text(self, content, x, y, color, small=False):
        font = self.small if small else self.font
        self.screen.blit(font.render(content, True, color), (x, y))

    def run(self):
        self.running = True
        while self.running:
            self._events()
            self._render()
            self.clock.tick(40)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Existing Episode 14 sidecar directory')
    parser.add_argument('--chunk', default='episode14_dynamics_chunk06', help='Chunk to annotate (default: chunk06)')
    parser.add_argument('--frame', type=int, default=0, help='Initial chunk-local frame number')
    parser.add_argument('--import-json', type=Path,
                        help='Import explicitly reviewed polygons from JSON without opening the GUI')
    parser.add_argument('--validate-only', action='store_true', help='Check source, annotations, and saved mask hashes without opening a GUI')
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[2]
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / 'scripts'))
    sys.path.insert(0, str(repo))
    from rosbag_temporal_annotations import index_rosbag, RgbFrameReader

    manifest = json.loads((args.root / 'manifest.json').read_text(encoding='utf-8'))
    bag = index_rosbag(Path(manifest['source']['bag_dir']))
    session = HandAnnotationSession(args.root, args.chunk, bag)
    if not 0 <= args.frame < len(session.frames):
        parser.error(f'--frame must be between 0 and {len(session.frames) - 1}')
    if args.validate_only:
        if args.import_json:
            parser.error('--validate-only cannot be combined with --import-json')
        print(f'Valid: {args.chunk}, revision {session.document["annotation_revision"]}, '
              f'{sum(row["status"].startswith("reviewed_") for row in session.document["frames"])} reviewed frames')
        return
    if args.import_json:
        changed = import_reviewed_json(session, args.import_json)
        print(f'Imported {changed} reviewed frames into {session.path}; '
              f'revision {session.document["annotation_revision"]}')
        return
    import pygame
    _initialize_pygame(pygame)
    try:
        with RgbFrameReader(bag, cache_frames=12) as reader:
            HandAnnotationApp(pygame, session, reader, args.frame).run()
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
