"""Restore the exact public FPT research snapshot shared by GitHub and Colab."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

from src.data.contracts import BASE_DIR, file_sha256

SNAPSHOT_FILES = (
    "data/raw/FPT.VN_raw.csv", "data/raw/FPT.VN_raw.metadata.json",
    "data/interim/FPT.VN_features.csv", "data/interim/FPT.VN_features.metadata.json",
    "data/processed/FPT.VN_supervised.csv", "data/processed/FPT.VN_excluded_rows.csv",
)


def restore_snapshot(root: Path = BASE_DIR) -> dict:
    """Validate before writing; never overwrite a different user's dataset."""
    root = Path(root).resolve()
    manifest = json.loads((root / "data/snapshots/manifest.json").read_text(encoding="utf-8"))
    archive = root / "data/snapshots" / manifest["archive"]
    if archive.parent.resolve() != (root / "data/snapshots").resolve():
        raise ValueError("Invalid archive path.")
    if file_sha256(archive) != manifest["archive_sha256"]:
        raise ValueError("Snapshot archive hash mismatch.")
    if set(manifest["files"]) != set(SNAPSHOT_FILES):
        raise ValueError("Unexpected snapshot file list.")
    content = {}
    with zipfile.ZipFile(archive) as zipped:
        if set(zipped.namelist()) != set(SNAPSHOT_FILES):
            raise ValueError("Archive contains unexpected files.")
        if sum(i.file_size for i in zipped.infolist()) > 20_000_000:
            raise ValueError("Snapshot exceeds expected size.")
        for name in SNAPSHOT_FILES:
            data = zipped.read(name)
            if hashlib.sha256(data).hexdigest() != manifest["files"][name]:
                raise ValueError(f"Snapshot content hash mismatch: {name}")
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Snapshot path escapes the project.")
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f"Existing dataset differs: {name}. Preserve it separately before restoring.")
            content[target] = data
    for target, data in content.items():
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".snapshot-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return manifest


if __name__ == "__main__":
    restored = restore_snapshot()
    print("Snapshot ready:", restored["snapshot_id"])
