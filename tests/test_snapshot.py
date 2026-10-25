"""Restoration must preserve exact bytes and reject mismatches before writing."""

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.data.snapshot import SNAPSHOT_FILES, restore_snapshot

TEMP_ROOT = Path(__file__).resolve().parents[1] / ".cache" / "snapshot-tests"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.root = Path(self.temp.name)
        snapshots = self.root / "data/snapshots"
        snapshots.mkdir(parents=True)
        archive = snapshots / "test.zip"
        self.content = {name: (name + "\n").encode() for name in SNAPSHOT_FILES}
        with zipfile.ZipFile(archive, "w") as zipped:
            for name, content in self.content.items():
                zipped.writestr(name, content)
        manifest = {"snapshot_id": "test", "archive": "test.zip",
                    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "files": {name: hashlib.sha256(content).hexdigest() for name, content in self.content.items()}}
        (snapshots / "manifest.json").write_text(json.dumps(manifest))

    def tearDown(self):
        self.temp.cleanup()

    def test_restore_is_exact_and_idempotent(self):
        restore_snapshot(self.root)
        restore_snapshot(self.root)
        for name, content in self.content.items():
            self.assertEqual((self.root / name).read_bytes(), content)

    def test_existing_different_data_is_not_overwritten(self):
        target = self.root / SNAPSHOT_FILES[0]
        target.parent.mkdir(parents=True)
        target.write_bytes(b"different snapshot")
        with self.assertRaisesRegex(ValueError, "differs"):
            restore_snapshot(self.root)
        self.assertEqual(target.read_bytes(), b"different snapshot")
        self.assertFalse((self.root / SNAPSHOT_FILES[2]).exists())

    def test_corrupt_archive_is_rejected(self):
        (self.root / "data/snapshots/test.zip").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "archive hash"):
            restore_snapshot(self.root)


if __name__ == "__main__":
    unittest.main()
