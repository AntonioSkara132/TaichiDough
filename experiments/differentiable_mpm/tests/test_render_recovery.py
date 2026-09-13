"""Rendering-only regressions; no Taichi runtime or simulation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

SUPPORT = Path(__file__).resolve().parents[1] / 'forward_video_v1/render_support.py'
spec = importlib.util.spec_from_file_location('render_support_test', SUPPORT)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


class DensityTests(unittest.TestCase):
    def test_dense_equivalence_across_tile_seams(self):
        rng = np.random.default_rng(18)
        # Deliberately cross positive and negative tile boundaries.
        points = rng.uniform([-.006, -.01, -.006], [.105, .02, .02], (1800, 3))
        voxel, volume = .0015, 1.5e-8
        mesh, info = support.density_boundary(points, volume)
        ids = np.floor(points / voxel).astype(int)
        origin = ids.min(axis=0) - 5
        grid = np.zeros(tuple(ids.max(axis=0) - origin + 6), dtype=np.float32)
        np.add.at(grid, tuple((ids - origin).T), volume / voxel**3)
        gaussian_filter(grid, .8, output=grid, mode='constant', cval=0, truncate=3)
        vertices, faces, _, _ = marching_cubes(grid, .2, spacing=(voxel,) * 3)
        vertices += (origin + .5) * voxel
        self.assertEqual(len(faces), info['triangles'])
        actual = mesh.reshape(-1, 3)
        self.assertLess(cKDTree(vertices).query(actual)[0].max(), 1e-7)
        self.assertLess(cKDTree(actual).query(vertices)[0].max(), 1e-7)
        def area(tris):
            return np.linalg.norm(np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0]), axis=1).sum()/2
        self.assertAlmostEqual(area(mesh), area(vertices[faces]), places=8)

    def test_distant_clusters_do_not_allocate_bounding_box(self):
        rng = np.random.default_rng(7)
        one = rng.uniform(.01, .025, (1000, 3))
        points = np.concatenate([one, one + [1.2, .4, .9]])
        original = points.copy()
        mesh, info = support.density_boundary(points, 1e-8)
        np.testing.assert_array_equal(points, original)
        self.assertLess(info['max_grid_voxels'], 400_000)
        self.assertGreater(mesh[:, :, 0].max(), 1.2)
        self.assertLess(mesh[:, :, 0].min(), .03)
        full_grid = np.ceil(np.ptp(points, axis=0)/.0015).astype(int) + 10
        self.assertGreater(int(np.prod(full_grid)), 30_000_000)

    def test_sparse_threshold_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'No density isosurface'):
            support.density_boundary(np.array([[0., 0., 0.], [1., 1., 1.]]), 1e-12)

    def test_triangle_budget_is_explicit(self):
        points = np.zeros((100, 3))
        with self.assertRaisesRegex(ValueError, 'triangle budget'):
            support.density_boundary(points, 1e-8, max_triangles=1)

    def test_invalid_points(self):
        for points in (np.empty((0, 3)), np.array([[np.nan, 0, 0]]), np.zeros((3, 2))):
            with self.assertRaises(ValueError):
                support.density_boundary(points, 1e-8)


class CameraTests(unittest.TestCase):
    def test_zoom_validation(self):
        import argparse
        self.assertEqual(support.camera_zoom('1.8'), 1.8)
        for value in ('nan', 'inf', '0', '-1', '5', 'invalid'):
            with self.assertRaises(argparse.ArgumentTypeError):
                support.camera_zoom(value)

    def test_zoom_rejects_encoding_only(self):
        completed = subprocess.run([sys.executable, str(SUPPORT.with_name('recover.py')),
            '--run-dir', '.', '--encode-only', '--camera-zoom', '1.8'], capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('cannot change existing images', completed.stderr)


class EncodingTests(unittest.TestCase):
    def setUp(self):
        scratch = os.environ.get('CLAUDE_JOB_DIR')
        self.tmp = tempfile.TemporaryDirectory(dir=str(Path(scratch)/'tmp') if scratch else None)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_failure_includes_real_ffmpeg_error(self):
        concat = self.root / 'frames.txt'
        concat.write_text("file 'missing.png'\n")
        def fail(command, stdout, stderr):
            stdout.write('Permission denied while reading mounted input\n')
            stdout.flush()
            return subprocess.CompletedProcess(command, 1)
        with patch.object(support, 'video_tools', return_value=('ffmpeg', 'ffprobe')):
            with patch.object(support.subprocess, 'run', side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, 'Permission denied while reading mounted input'):
                    support.encode_video(concat, self.root / 'out.mp4')
        self.assertTrue((self.root/'out.ffmpeg.log').exists())

    def test_existing_video_is_preserved(self):
        concat = self.root / 'frames.txt'; concat.write_text('')
        video = self.root / 'out.mp4'; video.write_bytes(b'original')
        with patch.object(support, 'video_tools', return_value=('ffmpeg', 'ffprobe')):
            with self.assertRaises(FileExistsError):
                support.encode_video(concat, video)
        self.assertEqual(video.read_bytes(), b'original')

    def test_missing_executable_message(self):
        with self.assertRaisesRegex(RuntimeError, 'executable not found'):
            support.executable('ffmpeg', '/nonexistent-test/ffmpeg')

    def test_encode_only_cli_real_ffmpeg(self):
        from PIL import Image
        try:
            tools = support.video_tools()
        except RuntimeError:
            self.skipTest('FFmpeg/ffprobe not installed')
        frames = self.root / 'perspective'; frames.mkdir()
        for i, color in enumerate(('red', 'green')):
            Image.new('RGB', (64, 48), color).save(frames/f'{i}.png')
        timing = frames/'frame_timing.txt'
        timing.write_text("file '0.png'\nduration 0.1\nfile '1.png'\nduration 0.1\nfile '1.png'\n")
        before = {p.name: p.read_bytes() for p in frames.iterdir()}
        command = [sys.executable, str(SUPPORT.with_name('recover.py')), '--run-dir', str(self.root),
                   '--encode-only', '--output-dir', str(self.root/'encoded'),
                   '--ffmpeg', tools[0], '--ffprobe', tools[1]]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        manifest = json.loads((self.root/'encoded/encoding_manifest.json').read_text())
        stream = manifest['ffprobe']['streams'][0]
        self.assertEqual(stream['codec_name'], 'h264')
        self.assertGreaterEqual(int(stream['nb_read_frames']), 6)
        self.assertFalse(manifest['simulation_executed'])
        self.assertEqual(before, {p.name: p.read_bytes() for p in frames.iterdir()})
        repeated = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn('Output already exists', repeated.stderr)


if __name__ == '__main__':
    unittest.main()
