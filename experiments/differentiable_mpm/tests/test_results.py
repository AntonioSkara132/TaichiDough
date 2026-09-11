from pathlib import Path
import unittest

from experiments.differentiable_mpm.results import RunStore
from experiments.differentiable_mpm.tests.helpers import temporary_directory


class RunStoreTests(unittest.TestCase):
    def test_identity_and_snapshots(self):
        with temporary_directory(prefix='result-tests-') as directory:
            root = Path(directory)
            path = root / 'run'
            identity = {'backend': 'cpu', 'parameters': {'E': 3}}
            with RunStore(path, identity, allowed_root=root) as store:
                store.save_optimizer({'event': 'accepted'}, {'iteration': 1})
                self.assertEqual(store.optimizer_state(), {'iteration': 1})
            with RunStore(path, identity, resume=True, allowed_root=root) as store:
                self.assertEqual(store.optimizer_state(), {'iteration': 1})
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                RunStore(path, {'backend': 'vulkan'}, resume=True, allowed_root=root)
            with self.assertRaises(FileExistsError):
                RunStore(path, identity, allowed_root=root)

    def test_unowned_directory_is_not_reused_or_deleted(self):
        with temporary_directory(prefix='result-tests-') as directory:
            root = Path(directory)
            path = root / 'existing'
            path.mkdir()
            marker = path / 'keep.txt'
            marker.write_text('user content')
            with self.assertRaisesRegex(ValueError, 'run_manifest'):
                RunStore(path, {}, resume=True, allowed_root=root)
            self.assertEqual(marker.read_text(), 'user content')

    def test_concurrent_writer_refused(self):
        with temporary_directory(prefix='result-tests-') as directory:
            root = Path(directory)
            with RunStore(root / 'run', {}, allowed_root=root):
                with self.assertRaises(BlockingIOError):
                    RunStore(root / 'run', {}, resume=True, allowed_root=root)

    def test_nonfinite_json_rejected(self):
        with temporary_directory(prefix='result-tests-') as directory:
            root = Path(directory)
            with RunStore(root / 'run', {}, allowed_root=root) as store:
                with self.assertRaises(ValueError):
                    store.write_json('bad.json', {'loss': float('nan')})
                self.assertFalse((store.path / 'bad.json').exists())


if __name__ == '__main__':
    unittest.main()
