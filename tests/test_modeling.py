"""Regression checks for forecast dates, shared context and honest direction metrics."""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.experiment import make_sequences, metrics, split_masks, _predict
from src.features.build_features import FEATURE_COLUMNS


class ModelingTests(unittest.TestCase):
    def test_native_xgboost_artifact_reloads_with_identical_predictions(self):
        from xgboost import XGBRegressor
        frame = pd.DataFrame(np.random.default_rng(42).normal(size=(24, len(FEATURE_COLUMNS))),
                             columns=FEATURE_COLUMNS)
        model = XGBRegressor(n_estimators=3, max_depth=2, n_jobs=1, random_state=42)
        model.fit(frame, frame.iloc[:, 0])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            model.save_model(path / "xgboost.json")
            actual = _predict({"kind": "xgboost", "file": "xgboost.json"}, frame, frame, path)
        np.testing.assert_array_equal(actual, model.predict(frame))

    def test_labels_crossing_partition_boundary_are_excluded(self):
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2022-12-29", "2022-12-30", "2023-01-03", "2024-12-31", "2025-01-02"]),
            "Target_Date_1d": pd.to_datetime(["2022-12-30", "2023-01-03", "2023-01-04", "2025-01-02", "2025-01-03"]),
        })
        masks = split_masks(frame)
        np.testing.assert_array_equal(masks["train"], [True, False, False, False, False])
        np.testing.assert_array_equal(masks["validation"], [False, False, True, False, False])
        np.testing.assert_array_equal(masks["test"], [False, False, False, False, True])

    def test_sequence_includes_origin_and_retains_history_across_boundary(self):
        values = np.arange(100).reshape(-1, 1)
        sequence = make_sequences(values, np.array([60, 61]), 10)
        np.testing.assert_array_equal(sequence[0, :, 0], np.arange(51, 61))
        np.testing.assert_array_equal(sequence[:, -1, 0], [60, 61])

    def test_sequence_never_wraps_short_context(self):
        with self.assertRaises(ValueError):
            make_sequences(np.arange(10).reshape(-1, 1), np.array([2]), 5)

    def test_zero_return_has_no_directional_accuracy(self):
        score = metrics(np.array([0.01, -0.01, 0]), np.zeros(3))
        self.assertIsNone(score["hit_rate_pct"])
        self.assertEqual(score["direction_coverage_pct"], 0)
        self.assertEqual(score["mse_skill_vs_zero"], 0)
        self.assertEqual(score["neutral_actual_n"], 1)

    def test_abstentions_do_not_inflate_direction_score(self):
        score = metrics(np.array([0.01, -0.01]), np.array([0.01, 0]))
        self.assertEqual(score["hit_rate_pct"], 50)
        self.assertEqual(score["direction_coverage_pct"], 50)

    def test_incorrect_prediction_has_negative_skill(self):
        score = metrics(np.array([0.01, -0.01]), np.array([-0.01, 0.01]))
        self.assertEqual(score["mse_skill_vs_zero"], -3)

    def test_nonfinite_predictions_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics(np.array([0.01]), np.array([np.nan]))


if __name__ == "__main__":
    unittest.main()
