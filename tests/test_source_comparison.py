"""Protect matched cohorts, reference-label semantics and temporal context."""

import copy
import json
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import TARGET_COLUMN, TARGET_DATE_COLUMN
from src.models.robustness import fold_masks
from src.models.source_comparison import REFERENCE, align_sources, context_positions, load_inputs, source_part, validate_config


def dataset(returns):
    features = pd.DataFrame({"Date": pd.bdate_range("2021-12-27", periods=6), "Log_Return": np.arange(6) / 100})
    labels = features.iloc[:-1].copy()
    labels[TARGET_DATE_COLUMN] = features.Date.iloc[1:].to_numpy()
    labels[TARGET_COLUMN] = returns
    return features, labels


class SourceComparisonTests(unittest.TestCase):
    def test_archived_features_do_not_need_label_columns(self):
        parent = Path(__file__).resolve().parents[1] / '.cache/source-comparison-tests'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            folder = Path(temporary)
            frames = {}
            for source in ['yahoo_v2', 'vietcap_v3_candidate']:
                features, labels = dataset(np.arange(5))
                frames[f'{source}/features.csv'] = features.to_csv(index=False).encode()
                frames[f'{source}/labels.csv'] = labels.to_csv(index=False).encode()
            with zipfile.ZipFile(folder/'input_snapshot.zip', 'w') as archive:
                for name, content in frames.items():
                    archive.writestr(name, content)
            protocol = {'input_zip_sha256': hashlib.sha256((folder/'input_snapshot.zip').read_bytes()).hexdigest(),
                        'input_files_sha256': {n: hashlib.sha256(b).hexdigest() for n,b in frames.items()},
                        'config': {'common_context': 1}}
            (folder/'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
            _, cohort = load_inputs(folder)
            self.assertEqual(len(cohort), 5)

    def test_same_origins_but_different_target_dates_are_not_compared(self):
        y, v = dataset(np.arange(5)), dataset(np.arange(5) + 10)
        y[1].loc[2, TARGET_DATE_COLUMN] = y[1].loc[3, TARGET_DATE_COLUMN]
        cohort = align_sources({"yahoo_v2": y, "vietcap_v3_candidate": v}, 1)
        self.assertEqual(len(cohort), 4)
        self.assertNotIn(y[0].Date.iloc[2], cohort.Date.to_list())

    def test_native_train_labels_and_fixed_reference_remain_separate(self):
        y, v = dataset(np.arange(5)), dataset(np.arange(5) + 10)
        cohort = align_sources({"yahoo_v2": y, "vietcap_v3_candidate": v}, 1)
        part = source_part(cohort, "yahoo_v2", y[0])
        np.testing.assert_array_equal(part[TARGET_COLUMN], np.arange(5))
        np.testing.assert_array_equal(part[REFERENCE], np.arange(5) + 10)

    def test_context_requirement_applies_to_both_sources(self):
        y, v = dataset(np.arange(5)), dataset(np.arange(5) + 10)
        cohort = align_sources({"yahoo_v2": y, "vietcap_v3_candidate": v}, 4)
        self.assertEqual(len(cohort), 2)
        self.assertTrue(cohort.position_yahoo_v2.ge(3).all())
        self.assertTrue(cohort.position_vietcap_v3_candidate.ge(3).all())

    def test_duplicate_dates_rejected(self):
        y, v = dataset(np.arange(5)), dataset(np.arange(5) + 10)
        y[0].loc[1, "Date"] = y[0].Date.iloc[0]
        with self.assertRaisesRegex(ValueError, "Unique"):
            align_sources({"yahoo_v2": y, "vietcap_v3_candidate": v}, 1)

    def test_context_union_never_adds_rows_after_origin(self):
        result = context_positions(np.array([3, 6]), 3)
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5, 6])
        with self.assertRaises(ValueError):
            context_positions(np.array([1]), 3)

    def test_boundary_labels_are_purged_from_each_partition(self):
        y, v = dataset(np.arange(5)), dataset(np.arange(5) + 10)
        cohort = align_sources({"yahoo_v2": y, "vietcap_v3_candidate": v}, 1)
        masks = fold_masks(cohort, {"train_end": "2021-12-29", "validation_end": "2021-12-31", "evaluation_end": "2022-12-31"})
        self.assertFalse(masks['train'][2])  # origin Dec 29, target Dec 30
        self.assertFalse(masks['validation'][4])  # origin Dec 31, target Jan 3

    def test_protocol_refuses_promotion_and_new_2026_evaluation(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/source_comparison_v1.json").read_text())
        validate_config(config)
        changed = copy.deepcopy(config)
        changed["automatic_promotion"] = True
        with self.assertRaises(ValueError):
            validate_config(changed)
        changed = copy.deepcopy(config)
        changed["folds"][-1]["evaluation_end"] = "2026-12-31"
        with self.assertRaises(ValueError):
            validate_config(changed)


if __name__ == "__main__":
    unittest.main()
