"""One structured notebook of research notes; only three human-facing Markdown files.

Original UTF-8 bytes are retained inside JSON so historical diagnostic hashes remain
verifiable. Research writers are manual, single-writer operations, like study creation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.data.contracts import BASE_DIR

REGISTRY = Path('reports/training_history/research_notes.json')


def note_key(path: Path, root: Path = BASE_DIR) -> str:
    root = Path(root).resolve()
    path = Path(path)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Research note must stay inside its project root.')
    return path.relative_to(root).as_posix()


def write_note(path: Path, text: str, root: Path = BASE_DIR) -> Path:
    key = note_key(path, root)
    if not key.startswith('reports/'):
        raise ValueError('Research reports must belong to reports/.')
    destination = Path(root) / REGISTRY
    value = json.loads(destination.read_text(encoding='utf-8')) if destination.exists() else {
        'schema': 'fpt-research-notes-v1', 'notes': {}}
    payload = text.encode('utf-8')
    value['notes'][key] = {'sha256': hashlib.sha256(payload).hexdigest(), 'text': text}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    temporary.replace(destination)
    return destination


def note_bytes(path: Path, root: Path = BASE_DIR) -> bytes:
    key = note_key(path, root)
    original = Path(root) / key
    if original.is_file():
        return original.read_bytes()  # Compatibility with older, unconverted studies.
    destination = Path(root) / REGISTRY
    value = json.loads(destination.read_text(encoding='utf-8'))
    if key not in value['notes']:
        raise FileNotFoundError(f'No stored research note: {key}')
    record = value['notes'][key]
    payload = record['text'].encode('utf-8')
    if hashlib.sha256(payload).hexdigest() != record['sha256']:
        raise ValueError(f'Research note hash mismatch: {key}')
    return payload


def note_sha256(path: Path, root: Path = BASE_DIR) -> str:
    return hashlib.sha256(note_bytes(path, root)).hexdigest()
