"""Persistent SDF cache integrity and independent-process tests."""
import multiprocessing
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.differentiable_mpm.sdf_cache import cached_mesh_sdf, cache_identity


VERTICES = np.arange(27, dtype=np.float64).reshape(9, 3)
SOURCE = {"mesh_sha256": "test-mesh", "builder_sha256": "test-builder"}


def build_volume(vertices, resolution, padding):
    distance = np.ones((resolution,) * 3, dtype=np.float32)
    distance[1, 1, 1] = -1
    return (distance, np.zeros((resolution,) * 3 + (3,), dtype=np.float32),
            np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))


def concurrent_worker(directory, result):
    def build(*args):
        with (Path(directory) / "build_calls.txt").open("a") as stream:
            stream.write("build\n")
        return build_volume(*args)
    try:
        arrays = cached_mesh_sdf(directory, VERTICES, 16, 0.001, SOURCE, build)
        result.put((True, float(arrays[0].sum())))
    except Exception as error:
        result.put((False, str(error)))


class SDFCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def cached(self, builder=build_volume, **kwargs):
        return cached_mesh_sdf(self.directory, VERTICES, 16, 0.001, SOURCE, builder, **kwargs)

    def test_roundtrip_is_bit_identical_and_does_not_rebuild(self):
        original = build_volume(VERTICES, 16, 0.001)
        first = self.cached()
        def fail(*args):
            raise AssertionError("Cache hit must not build")
        second = self.cached(fail, read_only=True)
        for expected, written, loaded in zip(original, first, second):
            np.testing.assert_array_equal(expected, written)
            np.testing.assert_array_equal(expected, loaded)
            self.assertEqual(expected.dtype, loaded.dtype)

    def test_key_changes_for_numerical_inputs_or_source(self):
        base = cache_identity(VERTICES, 16, 0.001, SOURCE)[0]
        variants = [(VERTICES + 1, 16, 0.001, SOURCE), (VERTICES, 17, 0.001, SOURCE),
                    (VERTICES, 16, 0.002, SOURCE),
                    (VERTICES, 16, 0.001, {**SOURCE, "builder_sha256": "new"}),
                    (VERTICES.astype(np.float32), 16, 0.001, SOURCE)]
        self.assertTrue(all(cache_identity(*args)[0] != base for args in variants))
        self.assertEqual(base, cache_identity(VERTICES.copy(), 16, 0.001, dict(reversed(list(SOURCE.items()))))[0])

    def test_read_only_miss_does_not_build(self):
        with self.assertRaises(FileNotFoundError):
            self.cached(read_only=True)
        self.assertEqual(list(self.directory.glob("*.npz")), [])

    def test_corrupt_arrays_are_rejected_without_replacement(self):
        self.cached()
        path = next(self.directory.glob("*.npz"))
        with np.load(path, allow_pickle=False) as archive:
            entries = {name: archive[name] for name in archive.files}
        entries["distances"][2, 2, 2] = 42
        np.savez(path, **entries)
        corrupted = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.cached()
        self.assertEqual(path.read_bytes(), corrupted)

    def test_truncated_file_is_rejected(self):
        self.cached()
        path = next(self.directory.glob("*.npz"))
        path.write_bytes(path.read_bytes()[:100])
        with self.assertRaisesRegex(ValueError, "Invalid SDF cache"):
            self.cached()

    def test_invalid_build_is_not_published(self):
        def invalid(*args):
            arrays = build_volume(*args)
            arrays[3][0] = 0
            return arrays
        with self.assertRaisesRegex(ValueError, "positive spacing"):
            self.cached(invalid)
        self.assertEqual(list(self.directory.glob("*.npz")), [])
        self.assertEqual(list(self.directory.glob("*.tmp")), [])

    def test_four_processes_build_exactly_once(self):
        context = multiprocessing.get_context("spawn")
        result = context.Queue()
        processes = [context.Process(target=concurrent_worker, args=(self.temp.name, result)) for _ in range(4)]
        for process in processes:
            process.start()
        try:
            results = [result.get(timeout=30) for _ in processes]
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)
            self.assertTrue(all(ok for ok, value in results), results)
            self.assertEqual(len({value for ok, value in results}), 1)
            self.assertEqual((self.directory / "build_calls.txt").read_text(), "build\n")
            self.assertEqual(len(list(self.directory.glob("*.npz"))), 1)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
            result.close()


if __name__ == "__main__":
    unittest.main()
