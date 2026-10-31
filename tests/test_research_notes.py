"""Document consolidation must preserve scientific hashes and future report behavior."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.data.contracts import BASE_DIR
from src.models.research_notes import REGISTRY, note_bytes, note_sha256, write_note


class ResearchNotesTests(unittest.TestCase):
    def test_report_writer_creates_one_registry_and_no_markdown(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);target=root/'reports/example/run/REPORT.md'
            write_note(target,'Báo cáo A\r\n',root=root)
            write_note(root/'reports/another/run/REPORT.md','B',root=root)
            self.assertFalse(target.exists())
            self.assertEqual(list(root.rglob('*.md')),[])
            self.assertEqual(note_bytes(target,root=root),'Báo cáo A\r\n'.encode())
            self.assertEqual(len(json.loads((root/REGISTRY).read_text())['notes']),2)

    def test_historical_diagnostic_hash_remains_verifiable(self):
        study=BASE_DIR/'reports/ridge_refinement/20261030T011109863274Z'
        protocol=json.loads((study/'protocol.json').read_text())
        self.assertEqual(note_sha256(study/'ERROR_ANALYSIS.md'),protocol['diagnostic_report_sha256'])

    def test_all_consolidated_original_notes_match_their_recorded_hash(self):
        registry=json.loads((BASE_DIR/REGISTRY).read_text(encoding='utf-8'))
        for path,record in registry['notes'].items():
            self.assertEqual(hashlib.sha256(record['text'].encode()).hexdigest(),record['sha256'],path)

    def test_modified_note_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);target=root/'reports/run/REPORT.md'
            write_note(target,'Original',root=root)
            value=json.loads((root/REGISTRY).read_text());value['notes']['reports/run/REPORT.md']['text']='Changed'
            (root/REGISTRY).write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):note_bytes(target,root=root)
            with self.assertRaises(ValueError):write_note(root.parent/'REPORT.md','outside',root=root)


if __name__=='__main__':unittest.main()
