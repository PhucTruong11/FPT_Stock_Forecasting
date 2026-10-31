"""Actual packaged Ridge and HTTP contracts; no training or external market calls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.data.contracts import BASE_DIR
from src.models.local_demo import LocalRidgeService, FEATURES, POINTER


class LocalDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = LocalRidgeService()

    def test_prediction_matches_frozen_shadow_math(self):
        service = self.service
        result = service.predict()
        raw = float(service.loaded.predict(service.features[FEATURES].iloc[[-1]])[0])
        expected = raw + np.mean(service.bundle['raw_errors'][-60:])
        self.assertAlmostEqual(result['predicted_log_return'], expected, places=14)
        self.assertAlmostEqual(result['predicted_close_adjusted'], result['latest_close_adjusted'] * np.exp(expected), places=8)
        self.assertIsNone(result['target_date'])
        self.assertFalse(result['forward_ledger_recorded'])
        self.assertEqual(result['training_target_end'], '2026-10-29')

    def test_input_ohlcv_roundtrip_and_zero_baseline(self):
        a = self.service.predict('Ridge', self.service.raw)
        b = self.service.predict('Ridge')
        self.assertEqual(a['predicted_log_return'], b['predicted_log_return'])
        zero = self.service.predict('Zero-Return')
        self.assertEqual(zero['latest_close_adjusted'], zero['predicted_close_adjusted'])
        self.assertIsNone(zero['interval'])

    def test_request_never_fits_or_changes_frozen_state(self):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        before = json.dumps(self.service.bundle, sort_keys=True)
        with patch.object(Ridge, 'fit', side_effect=AssertionError('Unexpected retrain')), \
             patch.object(StandardScaler, 'fit', side_effect=AssertionError('Unexpected scaler fit')):
            self.service.predict()
            self.service.predict('Ridge')
        self.assertEqual(before, json.dumps(self.service.bundle, sort_keys=True))

    def test_current_or_future_bar_rejected(self):
        raw = self.service.raw.copy()
        raw.loc[raw.index[-1], 'Date'] = '2099-01-01'
        with self.assertRaises(ValueError):
            self.service.predict(observations=raw)

    def test_pretraining_origin_and_invalid_latest_bar_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            self.service.predict(observations=self.service.raw.iloc[:-2])
        raw = self.service.raw.copy()
        raw.loc[raw.index[-1], 'High'] = raw.Close.iloc[-1] / 2
        with self.assertRaisesRegex(ValueError, 'invalid'):
            self.service.predict(observations=raw)

    def test_manifest_tamper_rejected_before_loading_pickle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'models/local_demo/test').mkdir(parents=True)
            (root / POINTER).write_text(json.dumps({'artifact_dir': 'models/local_demo/test', 'manifest_sha256': 'bad'}))
            (root / 'models/local_demo/test/manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                LocalRidgeService(root)

    def test_backtest_reproduces_original_metrics(self):
        expected = pd.read_csv(BASE_DIR / 'reports/recency_bias/20261030T022440928507Z/metrics.csv')
        score = next(row for row in self.service.benchmark()['metrics'] if row['strategy'] == 'Ridge_2025+bias60')
        original = expected.loc[expected.strategy.eq('refit_2025_native_bias60') & expected.cohort.eq('vci_full')].iloc[0]
        for key in ['rmse', 'mae', 'hit_rate_pct', 'balanced_accuracy_pct', 'price_mae_adjusted_vnd']:
            self.assertAlmostEqual(score[key], original[key], places=9)
        self.assertEqual(score['n'], 202)
        self.assertEqual(score['nonzero_actual_n'], 197)

    def test_http_default_sample_and_filtered_same_dates(self):
        with TestClient(create_app()) as client:
            self.assertEqual(client.get('/health').json()['default_demo_model'], 'Ridge+bias60')
            result = client.post('/api/v1/predict', json={})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()['model'], 'Ridge+bias60')
            raw = self.service.raw.tail(100).copy()
            raw.columns = [c.lower() for c in raw.columns]
            raw['date'] = pd.to_datetime(raw.date).dt.strftime('%Y-%m-%d')
            custom = client.post('/api/v1/predict', json={'observations': raw.to_dict('records')})
            self.assertEqual(custom.status_code, 200, custom.text)
            self.assertAlmostEqual(custom.json()['predicted_log_return'], result.json()['predicted_log_return'], places=14)
            subset = client.get('/api/v1/backtest?scope=jun_oct_2026&month=2026-10').json()
            self.assertEqual(len(subset['metrics']), 4)
            self.assertEqual({m['n'] for m in subset['metrics']}, {21})
            self.assertTrue(all(p['Target_Date_1d'].startswith('2026-10') for p in subset['predictions']))
            self.assertEqual(client.get('/api/v1/backtest?month=2026-12').status_code, 422)
            self.assertEqual(client.get('/api/v1/backtest?month=bad').status_code, 422)
            self.assertEqual(client.post('/api/v1/predict', json={'model': 'LSTM'}).status_code, 422)


if __name__ == '__main__':
    unittest.main()
