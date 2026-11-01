"""Causal split, interval scaling and exact forecast-function replay contracts."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.data.contracts import BASE_DIR
from src.models.forecast_path import band_factor, widen_band, forecast
from src.models.recursive_horizon_calibration import calibration_scale, phase_for, historical_service, activation_supported
from src.models.recursive_bootstrap_evaluation import load_inputs


class HorizonCalibrationTests(unittest.TestCase):
    def test_factor_uses_order_statistic_and_never_narrows(self):
        frame=pd.DataFrame({'phase':['calibration']*20,'target_date':['2026-06-01']*20,
            'predicted_log_return':np.zeros(20),'actual_log_return':np.arange(1,21)*.01,
            'lower_log_return':np.full(20,-.1),'upper_log_return':np.full(20,.1)})
        # ceil(21*.8)=17: score=1.7; coverage on fit data is 17/20.
        self.assertAlmostEqual(calibration_scale(frame),1.7)
        frame['actual_log_return']=0.;self.assertEqual(calibration_scale(frame),1.)
        frame.loc[0,'phase']='evaluation'
        with self.assertRaises(ValueError):calibration_scale(frame)
        frame.loc[0,'phase']='calibration';frame.loc[0,'target_date']='2026-07-01'
        with self.assertRaises(ValueError):calibration_scale(frame)
        self.assertEqual(calibration_scale(frame,as_of='2026-07-01'),1.)
        with self.assertRaises(ValueError):calibration_scale(frame,as_of='2026-06-30')

    def test_boundary_uses_target_cutoff_and_embargo(self):
        self.assertEqual(phase_for('2026-06-01','2026-06-30'),'calibration')
        self.assertEqual(phase_for('2026-06-01','2026-07-01'),'embargo')
        self.assertEqual(phase_for('2026-07-01','2026-08-03'),'evaluation')

    def test_unexercised_final_factor_cannot_pass_on_tied_metrics(self):
        before=pd.Series({'n':62,'coverage_pct':67.,'mean_interval_score':.38})
        after=pd.Series({'n':62,'coverage_pct':79.,'mean_interval_score':.37})
        self.assertFalse(activation_supported(before,before,np.ones(62),1.47))
        self.assertFalse(activation_supported(before,after,np.full(62,1.2),1.47))
        self.assertTrue(activation_supported(before,after,np.full(62,1.5),1.47))
        after['mean_interval_score']=.4
        self.assertFalse(activation_supported(before,after,np.full(62,1.5),1.47))

    def test_widening_scales_log_distances_not_price_levels(self):
        p=100.;lo=p*np.exp(-.1);hi=p*np.exp(.2)
        a,b=widen_band(p,lo,hi,1.5)
        self.assertAlmostEqual(np.log(a/p),-.15);self.assertAlmostEqual(np.log(b/p),.3)
        self.assertEqual(widen_band(p,lo,hi,1.),(lo,hi))
        with self.assertRaises(ValueError):widen_band(100.,101.,110.,2.)

    def test_policy_rejects_future_labels_or_uncovered_version(self):
        policy={'calibration_target_end':'2026-06-30','allowed_model_versions':['old'],
            'factors':{'Ridge+bias60':{'5':1.5,'22':2.}}}
        self.assertEqual(band_factor(policy,'Ridge+bias60',5,'2026-07-01','old'),1.5)
        self.assertEqual(band_factor(policy,'Ridge+bias60',21,'2026-07-01','old'),1.)
        self.assertEqual(band_factor(policy,'Ridge+bias60',1,'2026-07-01','old'),1.)
        with self.assertRaises(ValueError):band_factor(policy,'Ridge+bias60',5,'2026-06-29','old')
        with self.assertRaises(ValueError):band_factor(policy,'Ridge+bias60',5,'2026-07-01','new')

    def test_forecast_scales_only_explicit_steps_without_changing_points(self):
        from src.models.local_demo import LocalRidgeService
        service=LocalRidgeService();dates=pd.bdate_range('2026-10-30',periods=22)
        policy={'calibration_target_end':'2026-10-29','allowed_model_versions':['ridge6_20261029'],
            'factors':{'Ridge+bias60':{'5':1.2,'22':1.5}}}
        raw=forecast(service,today='2026-10-30',observation_dates=dates,band_calibration={})
        scaled=forecast(service,today='2026-10-30',observation_dates=dates,band_calibration=policy)
        for a,b in zip(raw['path'],scaled['path']):
            self.assertEqual(a['predicted_close_adjusted'],b['predicted_close_adjusted'])
            factor={5:1.2,22:1.5}.get(a['step'],1.)
            self.assertEqual(b['band_scale_factor'],factor)
            lo,hi=widen_band(a['predicted_close_adjusted'],a['scenario_lower'],a['scenario_upper'],factor)
            self.assertEqual((b['scenario_lower'],b['scenario_upper']),(lo,hi))
        self.assertIsNone(scaled['uncertainty']['displayed_quantiles'])
        self.assertEqual(scaled['uncertainty']['base_quantiles'],[.1,.9])

    def test_historical_service_and_forecast_do_not_see_future_values(self):
        import io,zipfile
        from src.models.recursive_bootstrap_evaluation import EVALUATION
        from sklearn.linear_model import Ridge
        estimator,features,calibration,spec,_=load_inputs(BASE_DIR)
        with zipfile.ZipFile(BASE_DIR/EVALUATION/'data_snapshot.zip') as z:
            raw=pd.read_csv(io.BytesIO(z.read('vci.csv')),parse_dates=['Date'])
        i=int(features.index[features.Date.eq(pd.Timestamp('2026-07-01'))][0])
        dates=features.Date.iloc[i+1:i+23]
        a=historical_service(BASE_DIR,estimator,raw,features,calibration,spec,i)
        altered=raw.copy();altered.loc[altered.Date>features.iloc[i].Date,['Open','High','Low','Close']]=999999.
        future_features=features.copy();future_features.loc[future_features.Date>features.iloc[i].Date,'Close']=999999.
        future_calibration=calibration.copy()
        future_calibration.loc[future_calibration.Target_Date_1d>features.iloc[i].Date,['Reference_Return','raw_prediction','prediction']]=999.
        b=historical_service(BASE_DIR,estimator,altered,future_features,future_calibration,spec,i)
        with patch.object(Ridge,'fit',side_effect=AssertionError('No fitting')):
            x=forecast(a,'month',today='2026-07-02',observation_dates=dates,band_calibration={})
            y=forecast(b,'month',today='2026-07-02',observation_dates=dates,band_calibration={})
        self.assertEqual(x['path'],y['path']);self.assertEqual(len(x['path']),22)
        self.assertEqual(x['effective_target_date'],str(dates.iloc[-1].date()))
        self.assertEqual(x['calendar_method'],'observed_dates_research_only')
        with self.assertRaises(ValueError):forecast(a,observation_dates=list(reversed(dates.to_list())),today='2026-07-02',band_calibration={})


if __name__=='__main__':unittest.main()
