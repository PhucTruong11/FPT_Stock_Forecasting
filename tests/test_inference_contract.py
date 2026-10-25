"""Inference safety and public output semantics without loading TensorFlow."""

import unittest

import pandas as pd

from src.models.inference import ModelService


class InferenceContractTests(unittest.TestCase):
    def make_service(self):
        service = object.__new__(ModelService)
        service.features = pd.DataFrame({"Date": pd.to_datetime(["2025-12-31"]),
                                         "Close": [100.0], "Log_Return": [0.01]})
        service.manifest = {"run_id": "fixture", "models": {
            "Zero-Return": {"kind": "constant", "value": 0},
            "Explode": {"kind": "constant", "value": 1000}},
            "selection": {"selected_model": "Zero-Return"},
            "splits": {"train": {"target_end": "2022-12-30"}}}
        return service

    def test_zero_return_keeps_price_without_inventing_target_date(self):
        result = self.make_service().predict("Zero-Return")
        self.assertEqual(result["predicted_close_adjusted"], 100)
        self.assertEqual(result["predicted_simple_return_pct"], 0)
        self.assertIsNone(result["target_date"])
        self.assertNotIn("confidence", result)

    def test_nonfinite_price_is_rejected_before_json(self):
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            self.make_service().predict("Explode")

    def test_historical_origin_before_configuration_lock_is_rejected(self):
        service = self.make_service()
        service.features["Date"] = pd.to_datetime(["2024-01-01"])
        with self.assertRaisesRegex(ValueError, "selection period"):
            service.predict("Zero-Return")

    def test_unknown_model_never_falls_back_to_an_alias(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            self.make_service().predict("best_lstm_model.keras")


if __name__ == "__main__":
    unittest.main()
