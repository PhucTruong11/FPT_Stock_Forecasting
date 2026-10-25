"""Offline regression checks for data integrity, causal features and label alignment."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.contracts import file_sha256, normalize_ohlcv, write_snapshot
from src.data.download_dataset import download_stock_data
from src.features.build_features import (
    FEATURE_COLUMNS, TARGET_COLUMN, TARGET_DATE_COLUMN, build_dataset, run_pipeline,
)

TEST_TEMP_DIR = Path(__file__).resolve().parents[1] / ".cache" / "pipeline-tests"
TEST_TEMP_DIR.mkdir(parents=True, exist_ok=True)


def sample_ohlcv(rows=120):
    step = np.arange(rows)
    close = 100 + step / 3 + np.sin(step)
    return pd.DataFrame({
        "Date": pd.bdate_range("2020-01-01", periods=rows),
        "Open": close - 0.2, "High": close + 1, "Low": close - 1,
        "Close": close, "Volume": 1000 + (step % 11) * 100,
    })


class ValidationTests(unittest.TestCase):
    def test_sort_deduplicate_and_yfinance_multiindex(self):
        source = sample_ohlcv().set_index("Date")
        source.columns = pd.MultiIndex.from_product([source.columns, ["FPT.VN"]])
        shuffled = pd.concat([source.iloc[::-1], source.iloc[[0]]])
        result = normalize_ohlcv(shuffled)
        self.assertEqual(len(result), 120)
        self.assertTrue(result.Date.is_monotonic_increasing)

    def test_reject_conflicting_duplicate_dates(self):
        source = sample_ohlcv()
        extra = source.iloc[[0]].copy()
        extra["Volume"] += 1
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            normalize_ohlcv(pd.concat([source, extra]))

    def test_reject_missing_price_on_traded_row(self):
        source = sample_ohlcv()
        source.loc[60, "Close"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing/infinite"):
            normalize_ohlcv(source, allow_untraded_missing_prices=True)

    def test_preserve_untraded_missing_candle_without_imputation(self):
        source = sample_ohlcv()
        source.loc[60, ["Open", "High", "Low", "Close"]] = np.nan
        source.loc[60, "Volume"] = 0
        result = normalize_ohlcv(source, allow_untraded_missing_prices=True)
        self.assertTrue(result.loc[60, ["Open", "High", "Low", "Close"]].isna().all())
        _, labeled, excluded, _ = build_dataset(source)
        self.assertEqual(excluded.Reason.iloc[0], "zero_volume_and_missing_prices")
        row = labeled.loc[labeled.Date.eq(source.Date.iloc[59])].iloc[0]
        self.assertEqual(row[TARGET_DATE_COLUMN], source.Date.iloc[61])

    def test_reject_invalid_candle_negative_volume_and_multiple_tickers(self):
        for field, value in (("High", 1.0), ("Volume", -1), ("Volume", 1.5)):
            with self.subTest(field=field, value=value):
                source = sample_ohlcv()
                source[field] = source[field].astype(float)
                source.loc[0, field] = value
                with self.assertRaises(ValueError):
                    normalize_ohlcv(source)
        source = sample_ohlcv().set_index("Date")
        multi = pd.concat({"FPT.VN": source, "OTHER": source}, axis=1).swaplevel(axis=1)
        with self.assertRaisesRegex(ValueError, "one ticker"):
            normalize_ohlcv(multi)


class FeatureTests(unittest.TestCase):
    def test_target_uses_next_eligible_close_and_keeps_latest_inference_row(self):
        source = sample_ohlcv()
        source.loc[60, "Volume"] = 0
        features, labeled, excluded, report = build_dataset(source)
        eligible = source.loc[source.Volume.gt(0)].reset_index(drop=True)
        expected = np.log(eligible.Close.shift(-1) / eligible.Close).iloc[49:-1]
        np.testing.assert_allclose(labeled[TARGET_COLUMN], expected, rtol=1e-12)
        pd.testing.assert_series_equal(
            labeled[TARGET_DATE_COLUMN].reset_index(drop=True),
            eligible.Date.shift(-1).iloc[49:-1].reset_index(drop=True), check_names=False,
        )
        self.assertEqual(features.Date.iloc[-1], source.Date.iloc[-1])
        self.assertEqual(labeled.Date.iloc[-1], source.Date.iloc[-2])
        self.assertEqual(len(features), 120 - 1 - 49)
        self.assertEqual(len(labeled), len(features) - 1)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(report["feature_count"], 16)

    def test_features_do_not_change_when_future_is_modified(self):
        source = sample_ohlcv()
        before, _, _, _ = build_dataset(source)
        changed = source.copy()
        changed.loc[90:, ["Open", "High", "Low", "Close"]] *= 2
        changed.loc[90:, "Volume"] *= 4
        after, _, _, _ = build_dataset(changed)
        cutoff = source.Date.iloc[90]
        pd.testing.assert_frame_equal(before.loc[before.Date.lt(cutoff)], after.loc[after.Date.lt(cutoff)])

    def test_constant_prices_and_volume_stay_finite(self):
        source = sample_ohlcv()
        source[["Open", "High", "Low", "Close"]] = 100.0
        source["Volume"] = 1000
        features, labeled, _, _ = build_dataset(source)
        self.assertTrue(np.isfinite(features[FEATURE_COLUMNS]).all().all())
        self.assertTrue(features.Volume_ZScore_20.eq(0).all())
        self.assertTrue(labeled[TARGET_COLUMN].eq(0).all())

    def test_invalid_candle_is_quarantined_without_bridging_target(self):
        source = sample_ohlcv()
        source.loc[80, "High"] = 1.0
        features, labeled, excluded, report = build_dataset(source)
        self.assertEqual(excluded.Reason.iloc[0], "invalid_ohlc_candle")
        self.assertFalse(features.Date.eq(source.Date.iloc[80]).any())
        self.assertTrue(features.Date.eq(source.Date.iloc[79]).any())
        self.assertFalse(labeled.Date.isin(source.Date.iloc[[79, 80]]).any())
        self.assertEqual(report["unlabeled_feature_rows"], 2)

    def test_model_allowlist_excludes_context_and_targets(self):
        features, labeled, _, _ = build_dataset(sample_ohlcv())
        self.assertEqual(features.columns.tolist(), ["Date", "Close", *FEATURE_COLUMNS])
        self.assertFalse(set(FEATURE_COLUMNS) & {"Date", "Close", TARGET_COLUMN, TARGET_DATE_COLUMN})
        self.assertIn(TARGET_COLUMN, labeled)

    def test_reject_too_short_history(self):
        with self.assertRaisesRegex(ValueError, "51"):
            build_dataset(sample_ohlcv(50))


class SnapshotTests(unittest.TestCase):
    def test_source_failure_preserves_existing_raw(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_DIR) as folder:
            path = Path(folder) / "FPT.VN_raw.csv"
            path.write_text("original snapshot", encoding="utf-8")
            digest = file_sha256(path)
            with patch("src.data.download_dataset.yf.download", return_value=pd.DataFrame()):
                with self.assertRaisesRegex(RuntimeError, "preserved"):
                    download_stock_data(output_dir=Path(folder), retries=1)
            self.assertEqual(file_sha256(path), digest)

    def test_pipeline_manifest_hashes_and_tampering_rejection(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_DIR) as folder:
            root = Path(folder)
            with patch("src.data.download_dataset.yf.download", return_value=sample_ohlcv()):
                path = download_stock_data(start_date="2020-01-01", end_date="2020-07-01",
                                           output_dir=root / "raw", retries=1)
            report = run_pipeline(path, root / "interim", root / "processed")
            manifest = json.loads((root / "interim/FPT.VN_features.metadata.json").read_text())
            feature_path = root / "interim/FPT.VN_features.csv"
            self.assertEqual(manifest["output_sha256"]["features"], file_sha256(feature_path))
            self.assertEqual(report["labeled_rows"], 70)
            saved_hash = file_sha256(feature_path)
            path.write_text(path.read_text().replace("100.0", "100.1"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run_pipeline(path, root / "interim", root / "processed")
            self.assertEqual(file_sha256(feature_path), saved_hash)

    def test_replacement_preserves_previous_snapshot(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_DIR) as folder:
            root = Path(folder)
            path = root / "raw/example.json"
            path.parent.mkdir()
            path.write_text('{"version": 1}', encoding="utf-8")
            with patch("src.data.contracts.BASE_DIR", root):
                write_snapshot(path, {"version": 2})
            backups = list((root / "data/backups").rglob("example.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(json.loads(backups[0].read_text()), {"version": 1})
            self.assertEqual(json.loads(path.read_text()), {"version": 2})


if __name__ == "__main__":
    unittest.main()
