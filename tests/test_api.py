"""HTTP contract checks, separate from actual model round-trip integration."""

import unittest

import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import create_app


class FakeService:
    default_model = "LSTM"

    def __init__(self, root):
        self.features = pd.DataFrame({"Date": pd.to_datetime(["2025-12-30", "2025-12-31"]),
                                      "Close": [100.0, 101.0], "Log_Return": [0.01, 0.01]})
        self.manifest = {"run_id": "test-run", "models": {"LSTM": {}, "Train Mean": {}},
                         "selection": {"selected_model": "Train Mean"}, "splits": {},
                         "metrics": [{"model": "Zero-Return", "hit_rate_pct": None}], "caveats": []}

    def predict(self, name, observations=None):
        if name not in self.manifest["models"]:
            raise ValueError("Unknown model")
        return {"ticker": "FPT.VN", "model": name, "model_version": "test-run",
                "predicted_log_return": 0.0, "predicted_close_adjusted": 101.0}


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(service_factory=FakeService))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_health_and_model_roles(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ready")
        models = self.client.get("/api/v1/models").json()
        self.assertEqual(models["default_demo_model"], "LSTM")
        self.assertEqual(models["validation_selected_model"], "Train Mean")

    def test_sample_predict_returns_json(self):
        response = self.client.post("/api/v1/predict", json={"model": "LSTM"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_version"], "test-run")

    def test_unknown_model_is_422(self):
        self.assertEqual(self.client.post("/api/v1/predict", json={"model": "missing"}).status_code, 422)

    def test_other_ticker_and_price_basis_are_rejected(self):
        for payload in ({"ticker": "PNJ.VN"}, {"price_basis": "unadjusted"}, {"observations": []}):
            self.assertEqual(self.client.post("/api/v1/predict", json=payload).status_code, 422)

    def test_history_limit_is_enforced(self):
        self.assertEqual(self.client.get("/api/v1/fpt/history?limit=0").status_code, 422)
        rows = self.client.get("/api/v1/fpt/history?limit=1").json()["observations"]
        self.assertEqual(rows[0]["Date"], "2025-12-31")

    def test_na_direction_metric_is_json_null(self):
        self.assertIsNone(self.client.get("/api/v1/metrics").json()["metrics"][0]["hit_rate_pct"])

    def test_openapi_exposes_prediction_endpoint(self):
        self.assertIn("/api/v1/predict", self.client.get("/openapi.json").json()["paths"])


if __name__ == "__main__":
    unittest.main()
