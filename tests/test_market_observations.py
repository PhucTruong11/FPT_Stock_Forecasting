"""Post-snapshot quotes stay separate; dates and adjusted-price basis must agree."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.app.observed_comparison import comparison_chart
from src.data.contracts import BASE_DIR, file_sha256
from src.data.market_observations import (CACHE_PATH, observation_response,
                                          parse_observations, refresh_observations)
from src.models.local_demo import LocalRidgeService

NOW = "2026-11-24T10:00:00Z"


def provider_payload(frame):
    frame = frame.copy()
    times = pd.to_datetime(frame.Date).dt.tz_localize("Asia/Ho_Chi_Minh")
    result = {"symbol": "FPT", "t": [str(int(t.timestamp())) for t in times]}
    result.update({key: frame[col].tolist() for key, col in zip("ohlcv", ["Open", "High", "Low", "Close", "Volume"])})
    return [result]


class MarketObservationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = LocalRidgeService()
        cls.frozen = cls.service.raw.tail(10).copy()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        extra = pd.DataFrame({"Date": ["2026-10-30", "2026-11-03", "2026-11-24", "2026-11-25"],
            "Open": [53000., 50000., 50000., 50000.], "High": [53500., 51000., 51000., 51000.],
            "Low": [52000., 49000., 49000., 49000.], "Close": [52700., 50400., 50500., 50600.],
            "Volume": [5000, 6000, 7000, 8000]})
        self.payload = provider_payload(pd.concat([self.frozen, extra], ignore_index=True))

    def refresh(self, payload=None):
        response = Mock()
        response.json.return_value = self.payload if payload is None else payload
        with patch("src.data.market_observations.requests.post", return_value=response) as request:
            refresh_observations(self.root, now=NOW)
        return request

    def test_vnd_units_exclude_partial_future_and_never_fill_missing_sessions(self):
        frame = parse_observations(self.payload, now=NOW)
        self.assertEqual(frame.Date.max(), pd.Timestamp("2026-11-03"))
        self.assertNotIn(pd.Timestamp("2026-11-02"), frame.Date.tolist())
        self.assertEqual(frame.Close.iloc[-1], 50400.)
        missing_day = self.payload[0]["t"][-2]
        self.assertNotIn(pd.to_datetime(int(missing_day), unit="s").normalize(), frame.Date.tolist())

    def test_wrong_ticker_duplicates_and_invalid_traded_candles_rejected(self):
        invalid = copy.deepcopy(self.payload)
        invalid[0]["symbol"] = "PNJ"
        with self.assertRaises(ValueError):
            parse_observations(invalid, now=NOW)
        invalid = copy.deepcopy(self.payload)
        for key in "tohlcv":
            invalid[0][key].append(invalid[0][key][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_observations(invalid, now=NOW)
        invalid = copy.deepcopy(self.payload)
        invalid[0]["h"][0] = 1.
        with self.assertRaisesRegex(ValueError, "Invalid candle"):
            parse_observations(invalid, now=NOW)

    def test_failed_refresh_preserves_good_cache_and_integrity_is_checked(self):
        self.refresh()
        path = self.root / CACHE_PATH
        before = path.read_bytes()
        invalid = copy.deepcopy(self.payload)
        invalid[0]["c"].pop()
        with self.assertRaises(ValueError):
            self.refresh(invalid)
        self.assertEqual(path.read_bytes(), before)
        path.write_text(before.decode().replace('"FPT"', '"OTHER"'), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integrity"):
            observation_response(self.root, self.frozen, now=NOW)

    def test_revised_adjustment_basis_and_insufficient_overlap_block_comparison(self):
        self.refresh()
        result = observation_response(self.root, self.frozen, now=NOW)
        self.assertTrue(result["basis_check"]["allowed"])
        revised = self.frozen.copy()
        revised["Close"] *= .9
        check = observation_response(self.root, revised, now=NOW)["basis_check"]
        self.assertEqual(check["status"], "changed")
        self.assertFalse(check["allowed"])
        self.assertEqual(observation_response(self.root, self.frozen.tail(2), now=NOW)
                         ["basis_check"]["status"], "insufficient_overlap")

    def test_chart_matches_only_exact_post_snapshot_dates_and_has_no_fake_actuals(self):
        self.refresh()
        observed = observation_response(self.root, self.frozen, now=NOW)
        forecast = {"origin_date": "2026-10-29", "origin_close_adjusted": 52700., "path": [
            {"estimated_date": d, "predicted_close_adjusted": p}
            for d, p in zip(["2026-10-30", "2026-11-02", "2026-11-03"], [52500., 52000., 51500.])]}
        fig, paired = comparison_chart(forecast, observed)
        self.assertEqual(len(paired), 2)
        self.assertAlmostEqual(np.mean(abs(paired.Close - paired.predicted_close_adjusted)), 650.)
        self.assertEqual(pd.Timestamp(max(fig.data[0].x)), pd.Timestamp("2026-11-03"))
        observed["basis_check"]["allowed"] = False
        fig, paired = comparison_chart(forecast, observed)
        self.assertEqual(len(fig.data), 1)
        self.assertTrue(paired.empty)

    def test_http_get_never_fetches_and_refresh_does_not_train_or_change_bundle(self):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        files = [BASE_DIR / "models/local_demo.json", *sorted((BASE_DIR / "models/local_demo/ridge6_20261029").iterdir())]
        before = {str(p): file_sha256(p) for p in files if p.is_file()}
        expected = self.service.predict()["predicted_log_return"]
        response = Mock()
        response.json.return_value = self.payload
        with patch("src.data.market_observations.local_cutoff", return_value=pd.Timestamp("2026-11-24", tz="Asia/Ho_Chi_Minh")), \
             patch("src.data.market_observations.requests.post", return_value=response) as upstream, \
             patch.object(Ridge, "fit", side_effect=AssertionError("Unexpected fit")), \
             patch.object(StandardScaler, "fit", side_effect=AssertionError("Unexpected scaler fit")), \
             TestClient(create_app(root=self.root, service_factory=lambda root: self.service)) as client:
            self.assertEqual(client.get("/api/v1/fpt/observed").json()["status"], "not_loaded")
            upstream.assert_not_called()
            refreshed = client.post("/api/v1/fpt/observed/refresh", json={})
            self.assertEqual(refreshed.status_code, 200, refreshed.text)
            self.assertTrue(refreshed.json()["basis_check"]["allowed"])
            self.assertEqual(upstream.call_count, 1)
            self.assertEqual(client.get("/api/v1/fpt/observed").json()["data_as_of"], "2026-11-03")
            self.assertEqual(upstream.call_count, 1)
            upstream.side_effect = requests.Timeout()
            self.assertEqual(client.post("/api/v1/fpt/observed/refresh", json={}).status_code, 502)
            self.assertEqual(client.get("/api/v1/fpt/observed").status_code, 200)
            self.assertEqual(client.post("/api/v1/predict", json={}).json()["predicted_log_return"], expected)
        self.assertEqual({str(p): file_sha256(p) for p in files if p.is_file()}, before)


if __name__ == "__main__":
    unittest.main()
