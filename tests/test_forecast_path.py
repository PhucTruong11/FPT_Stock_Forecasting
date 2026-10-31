"""Recursive feature/date semantics and HTTP validation, without any fitting."""
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.models.forecast_path import forecast, return_features, target_window, residual_blocks
from src.models.local_demo import LocalRidgeService, FEATURES


class ForecastPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service=LocalRidgeService()

    def test_first_features_match_canonical_builder_order(self):
        frame=self.service.features
        actual=return_features(np.log(frame.Close.tail(21).to_numpy()))
        self.assertEqual(list(actual.columns),FEATURES)
        np.testing.assert_allclose(actual.to_numpy(),frame[FEATURES].tail(1),atol=2e-15)

    def test_first_step_matches_existing_one_observation_prediction(self):
        for name in ['Ridge','Ridge+bias60','Zero-Return']:
            result=forecast(self.service,'month',name,today='2026-10-30')
            original=self.service.predict(name)
            self.assertAlmostEqual(result['path'][0]['predicted_close_adjusted'],original['predicted_close_adjusted'],places=8)
            self.assertAlmostEqual(result['path'][0]['step_log_return'],original['predicted_log_return'],places=13)

    def test_all_horizons_resolve_real_requested_dates_not_fixed_aliases(self):
        expected={'today':(1,'2026-10-30'),'tomorrow':(2,'2026-11-02'),
                  'week':(5,'2026-11-05'),'month':(21,'2026-11-27')}
        for h,(count,date) in expected.items():
            r=forecast(self.service,h,today='2026-10-30')
            self.assertEqual(r['steps'],count);self.assertEqual(r['effective_target_date'],date)
            self.assertTrue(all(pd.Timestamp(p['estimated_date']).dayofweek<5 for p in r['path']))
        self.assertTrue(forecast(self.service,'tomorrow',today='2026-10-30')['weekend_shifted'])

    def test_weekend_today_moves_forward_but_does_not_claim_exchange_calendar(self):
        r=forecast(self.service,'today',today='2026-10-31')
        self.assertEqual(r['effective_target_date'],'2026-11-02')
        self.assertFalse(r['calendar_verified']);self.assertTrue(r['weekend_shifted'])

    def test_recursive_second_step_rebuilds_features_from_predicted_close(self):
        s=self.service;r=forecast(s,'week','Ridge',today='2026-10-30')
        closes=np.append(s.features.Close.tail(20).to_numpy(),r['path'][0]['predicted_close_adjusted'])
        expected=float(s.loaded.predict(return_features(np.log(closes)))[0])
        self.assertAlmostEqual(expected,r['path'][1]['step_log_return'],places=13)
        self.assertTrue(r['path'][1]['uses_predicted_context'])
        self.assertFalse(r['path'][0]['uses_predicted_context'])

    def test_no_future_ohlcv_or_fit_or_calibration_state_changes(self):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        before=json.dumps(self.service.bundle,sort_keys=True)
        raw_before=self.service.raw.copy(deep=True)
        with patch.object(Ridge,'fit',side_effect=AssertionError('fit forbidden')), \
             patch.object(StandardScaler,'fit',side_effect=AssertionError('fit forbidden')):
            forecast(self.service,'month',today='2026-10-30')
        self.assertEqual(before,json.dumps(self.service.bundle,sort_keys=True))
        pd.testing.assert_frame_equal(raw_before,self.service.raw)

    def test_frozen_seed_scenarios_repeat_and_zero_has_no_ridge_band(self):
        a=forecast(self.service,'week',today='2026-10-30')
        b=forecast(self.service,'week',today='2026-10-30')
        self.assertEqual(a['path'],b['path'])
        z=forecast(self.service,'month','Zero-Return',today='2026-10-30')
        for row in z['path']:
            self.assertAlmostEqual(row['predicted_close_adjusted'],z['origin_close_adjusted'],places=8)
            self.assertIsNone(row['scenario_lower']);self.assertIsNone(row['scenario_upper'])

    def test_scenario_range_is_ordered_finite_without_guaranteed_coverage(self):
        r=forecast(self.service,'month',today='2026-10-30')
        for row in r['path']:
            self.assertLess(row['scenario_lower'],row['scenario_upper'])
            self.assertTrue(np.isfinite(list(v for v in row.values() if isinstance(v,float))).all())
        self.assertFalse(r['multi_step_validated']);self.assertFalse(r['uncertainty']['coverage_guaranteed'])
        self.assertFalse(r['forward_ledger_recorded']);self.assertEqual(r['trained_horizon_observations'],1)

    def test_stale_source_and_nonfuture_origin_are_rejected(self):
        for today in ['2026-10-29','2026-12-31']:
            with self.assertRaises(ValueError):target_window('2026-10-29','month',today)
        with self.assertRaises(ValueError):target_window('2026-10-29','year','2026-10-30')

    def test_bad_feature_history_and_residuals_are_rejected(self):
        for data in [np.zeros(20),np.array([np.nan]*21)]:
            with self.assertRaises(ValueError):return_features(data)
        with self.assertRaises(ValueError):residual_blocks([],2)

    def test_http_horizons_and_validation(self):
        with patch('src.models.forecast_path.local_today',return_value=pd.Timestamp('2026-10-30')),TestClient(create_app()) as client:
            for horizon in ['today','tomorrow','week','month']:
                response=client.post('/api/v1/forecast',json={'horizon':horizon})
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(response.json()['horizon'],horizon)
            for payload in [{'horizon':'year'},{'horizon':'month','steps':100},{'model':'missing'}]:
                self.assertEqual(client.post('/api/v1/forecast',json=payload).status_code,422)
            self.assertEqual(len(client.get('/api/v1/fpt/ohlcv?limit=30').json()['observations']),30)
            self.assertEqual(client.get('/api/v1/fpt/ohlcv?limit=0').status_code,422)
            self.assertEqual(len(client.get('/api/v1/data-quality').json()['feature_columns']),6)


if __name__=='__main__':unittest.main()
