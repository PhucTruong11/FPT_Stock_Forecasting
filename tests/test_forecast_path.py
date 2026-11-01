"""Recursive feature/date semantics and HTTP validation, without any fitting."""
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.models.forecast_path import forecast, return_features, target_window, residual_blocks, recursive_paths, forecast_date_limits
from src.models.recursive_bootstrap_evaluation import mature_pool, interval_scores
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

    def test_custom_date_keeps_snapshot_bridge_and_selected_window_distinct(self):
        r=forecast(self.service,'custom',today='2026-11-01',target_date='2026-11-13')
        self.assertEqual(r['requested_start'],'2026-11-01');self.assertEqual(r['requested_end'],'2026-11-13')
        self.assertEqual(r['effective_target_date'],'2026-11-13')
        self.assertEqual(r['steps'],11);self.assertEqual(r['bridge_steps_before_requested_window'],1)
        self.assertFalse(r['path'][0]['in_requested_window'])
        self.assertTrue(all(x['in_requested_window'] for x in r['path'][1:]))
        weekend=forecast(self.service,'custom',today='2026-11-01',target_date='2026-11-14')
        self.assertTrue(weekend['weekend_shifted']);self.assertEqual(weekend['effective_target_date'],'2026-11-16')

    def test_custom_date_boundaries_and_snapshot_step_cap_are_enforced(self):
        limits=forecast_date_limits('2026-10-29','2026-11-01')
        self.assertEqual(limits['selectable_max_date'],'2026-12-01');self.assertTrue(limits['custom_available'])
        r=forecast(self.service,'custom',today='2026-11-01',target_date='2026-12-01')
        self.assertEqual(r['steps'],23)
        for target in [None,'2026-10-31','2026-12-02','2026-11-03T12:00:00','2026-11-03T00:00:00+07:00']:
            with self.assertRaises(ValueError):target_window('2026-10-29','custom','2026-11-01',target)
        with self.assertRaises(ValueError):target_window('2026-10-29','today','2026-11-01','2026-11-05')
        with self.assertRaises(ValueError):target_window('2026-10-01','custom','2026-11-01','2026-11-20')
        self.assertFalse(forecast_date_limits('2026-10-29','2026-12-31')['custom_available'])
        self.assertEqual(forecast_date_limits('2027-01-29','2027-01-31')['calendar_max_date'],'2027-02-28')

    def test_recursive_second_step_rebuilds_features_from_predicted_close(self):
        s=self.service;r=forecast(s,'week','Ridge',today='2026-10-30')
        closes=np.append(s.features.Close.tail(20).to_numpy(),r['path'][0]['predicted_close_adjusted'])
        expected=float(s.loaded.predict(return_features(np.log(closes)))[0])
        self.assertAlmostEqual(expected,r['path'][1]['step_log_return'],places=13)
        self.assertTrue(r['path'][1]['uses_predicted_context'])
        self.assertFalse(r['path'][0]['uses_predicted_context'])

    def test_bias_only_on_first_step_and_no_compounded_drift(self):
        s=self.service;r=forecast(s,'month','Ridge+bias60',today='2026-10-30')
        self.assertEqual(r['forecast_method_version'],'recursive-ridge6-v4')
        bias=s.predict('Ridge+bias60')['bias_offset'];self.assertNotEqual(bias,0.)
        closes=np.append(s.features.Close.tail(20).to_numpy(),r['path'][0]['predicted_close_adjusted'])
        raw_second=float(s.loaded.predict(return_features(np.log(closes)))[0])
        self.assertAlmostEqual(r['path'][1]['step_log_return'],raw_second,places=13)
        raw=forecast(s,'month','Ridge',today='2026-10-30')
        shift=r['path'][0]['step_log_return']-raw['path'][0]['step_log_return']
        self.assertAlmostEqual(shift,bias,places=13)
        # Step-1 bias may carry into context, but must not be re-added 20 more times.
        self.assertLess(abs(r['path'][-1]['change_from_origin_pct']-raw['path'][-1]['change_from_origin_pct']),
                        abs(bias)*100*5)

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

    def test_centered_bootstrap_is_invariant_to_residual_mean_shift(self):
        errors=np.array([-.03,.01,.02,-.01,.04,-.02])
        before=errors.copy()
        a=residual_blocks(errors,21)
        b=residual_blocks(errors+.25,21)
        np.testing.assert_allclose(a,b,atol=1e-16,rtol=0)
        np.testing.assert_array_equal(errors,before)
        uncentered=residual_blocks(errors,21,center=False)
        np.testing.assert_allclose(uncentered-a,errors.mean(),atol=1e-16,rtol=0)
        # Do not center each sampled path: that would suppress cumulative risk.
        self.assertGreater(np.std(a.sum(axis=1)),.001)
        np.testing.assert_allclose(residual_blocks(np.full(60,-.003),21),0,atol=1e-17)

    def test_zero_innovations_reproduce_point_at_every_step_with_bias(self):
        s=self.service;history=np.log(s.features.Close.tail(21).to_numpy())
        bias=s.predict()['bias_offset']
        points,returns,paths=recursive_paths(s.loaded,history,np.zeros((7,21)),bias=bias)
        np.testing.assert_allclose(paths,np.tile(points,(7,1)),atol=2e-14,rtol=0)
        for step in range(1,21):
            context=np.r_[history,points[:step]][-21:]
            raw=float(s.loaded.predict(return_features(context))[0])
            self.assertAlmostEqual(returns[step],raw,places=13)

    def test_forecast_band_does_not_reintroduce_residual_pool_mean(self):
        import copy
        service=copy.copy(self.service);service.bundle=copy.deepcopy(service.bundle)
        before=forecast(service,'month',today='2026-10-30')
        service.bundle['corrected_errors']=[x+.01 for x in service.bundle['corrected_errors']]
        after=forecast(service,'month',today='2026-10-30')
        for a,b in zip(before['path'],after['path']):
            self.assertEqual(a['predicted_close_adjusted'],b['predicted_close_adjusted'])
            self.assertAlmostEqual(a['scenario_lower'],b['scenario_lower'],places=8)
            self.assertAlmostEqual(a['scenario_upper'],b['scenario_upper'],places=8)
        self.assertAlmostEqual(after['uncertainty']['calibration_residual_mean_removed']-
            before['uncertainty']['calibration_residual_mean_removed'],.01,places=14)

    def test_rolling_pool_cannot_see_unmatured_labels(self):
        dates=pd.bdate_range('2026-01-01',periods=66)
        frame=pd.DataFrame({'Date':dates[:-1],'Target_Date_1d':dates[1:],
            'Reference_Return':np.arange(65)*.0001,'raw_prediction':0.,'prediction':.001})
        origin=dates[60];a=mature_pool(frame,origin)
        changed=frame.copy();future=changed.Target_Date_1d.gt(origin)
        changed.loc[future,['Reference_Return','raw_prediction','prediction']]=999.
        b=mature_pool(changed,origin)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        self.assertEqual(a[2],origin);self.assertEqual(len(a[0]),60)
        with self.assertRaises(ValueError):mature_pool(frame,dates[59])

    def test_interval_score_penalizes_misses_and_excessive_width(self):
        score=interval_scores(np.array([0.,-.2,.3]),np.array([-.1]*3),np.array([.1]*3))
        np.testing.assert_allclose(score,[.2,1.2,2.2])
        narrow=interval_scores([0.],[-.1],[.1])[0]
        wide=interval_scores([0.],[-1.],[1.])[0]
        self.assertLess(narrow,wide)

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

    def test_custom_http_contract_requires_matching_date_and_horizon(self):
        with patch('src.models.forecast_path.local_today',return_value=pd.Timestamp('2026-11-01')),TestClient(create_app()) as client:
            response=client.post('/api/v1/forecast',json={'horizon':'custom','target_date':'2026-11-14'})
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json()['effective_target_date'],'2026-11-16')
            self.assertEqual(client.get('/health').json()['forecast_date_limits']['local_today'],'2026-11-01')
            for payload in [{'horizon':'custom'},{'target_date':'2026-11-14'},{'horizon':'week','target_date':'2026-11-14'},
                    {'horizon':'custom','target_date':'2026-10-31'},{'horizon':'custom','target_date':'2026-12-02'},
                    {'horizon':'custom','target_date':'not-a-date'}]:
                self.assertEqual(client.post('/api/v1/forecast',json=payload).status_code,422,payload)


if __name__=='__main__':unittest.main()
