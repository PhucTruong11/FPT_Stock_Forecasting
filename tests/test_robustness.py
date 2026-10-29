"""Research protocol guards against temporal leakage and accidental promotion."""

import copy
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.robustness import fold_masks, score_direction, validate_protocol


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] / "configs/robustness_v1.json").read_text())

    def test_labels_crossing_any_fold_boundary_are_purged(self):
        labels = pd.DataFrame({
            "Date": pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-04", "2021-12-31", "2022-01-04", "2022-12-30"]),
            "Target_Date_1d": pd.to_datetime(["2020-12-31", "2021-01-04", "2021-01-05", "2022-01-04", "2022-01-05", "2023-01-03"]),
        })
        masks = fold_masks(labels, self.config["folds"][0])
        np.testing.assert_array_equal(masks["train"], [True, False, False, False, False, False])
        np.testing.assert_array_equal(masks["validation"], [False, False, True, False, False, False])
        np.testing.assert_array_equal(masks["evaluation"], [False, False, False, False, True, False])
        self.assertTrue((sum(masks.values()) <= 1).all())

    def test_protocol_cannot_enable_automatic_promotion(self):
        config = copy.deepcopy(self.config)
        config["selection"]["automatic_promotion"] = True
        with self.assertRaises(ValueError):
            validate_protocol(config)

    def test_target_cannot_be_added_to_feature_group(self):
        self.config["feature_groups"]["returns"].append("Target_Return_1d")
        with self.assertRaises(ValueError):
            validate_protocol(self.config)

    def test_reused_2025_period_cannot_enter_research_folds(self):
        self.config["folds"][-1]["evaluation_end"] = "2025-12-31"
        with self.assertRaises(ValueError):
            validate_protocol(self.config)

    def test_single_direction_has_balanced_accuracy_fifty(self):
        result = score_direction(np.array([.01, -.01, -.02]), np.array([.01, .01, .01]))
        self.assertEqual(result["balanced_accuracy_pct"], 50)
        self.assertAlmostEqual(result["hit_rate_pct"], 100 / 3)

    def test_zero_return_does_not_invent_directional_accuracy(self):
        result = score_direction(np.array([.01, -.01]), np.zeros(2))
        self.assertIsNone(result["balanced_accuracy_pct"])
        self.assertIsNone(result["hit_rate_pct"])


if __name__ == "__main__":
    unittest.main()
