"""Archive recovery must never overwrite work or install unverified bytes."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from restore_research import destination, restore


class Recovery(unittest.TestCase):
    def test_restore_and_repeat_without_git_when_already_present(self):
        payload = b'archived experiment\n'
        row = dict(path='artifacts/old/result.json', bytes=len(payload),
                   sha256=hashlib.sha256(payload).hexdigest())
        def git(args, **kwargs):
            if args[1] == 'show':
                kwargs['stdout'].write(payload)
            return subprocess.CompletedProcess(args, 0, stderr=b'')
        with tempfile.TemporaryDirectory() as directory, patch('restore_research.subprocess.run', side_effect=git):
            root = Path(directory)
            self.assertEqual(restore(root, 'a' * 40, [row]), 1)
            self.assertEqual((root / row['path']).read_bytes(), payload)
            with patch('restore_research.subprocess.run', side_effect=AssertionError('unnecessary Git call')):
                self.assertEqual(restore(root, 'a' * 40, [row]), 0)
            (root / row['path']).write_bytes(b'new local work')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                restore(root, 'a' * 40, [row])
            self.assertEqual((root / row['path']).read_bytes(), b'new local work')

    def test_corrupt_archive_never_installs_destination(self):
        def git(args, **kwargs):
            if args[1] == 'show':
                kwargs['stdout'].write(b'bad')
            return subprocess.CompletedProcess(args, 0, stderr=b'')
        with tempfile.TemporaryDirectory() as directory, patch('restore_research.subprocess.run', side_effect=git):
            root = Path(directory)
            row = dict(path='artifacts/old/model.pt', bytes=3, sha256='0' * 64)
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                restore(root, 'a' * 40, [row])
            self.assertFalse((root / row['path']).exists())
            self.assertEqual(list(root.rglob('*.tmp')), [])

    def test_rejects_paths_outside_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in ('../outside', 'artifacts/../../outside', '/absolute', 'C:\\outside', '.git/config', 'src/App.jsx'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    destination(Path(directory), path)


if __name__ == '__main__':
    unittest.main()
