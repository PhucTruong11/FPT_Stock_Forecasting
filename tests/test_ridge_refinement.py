import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.ridge_refinement import fit_thresholds,assign_regimes,select_config,prediction_export


class RidgeRefinementTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((Path(__file__).resolve().parents[1]/'configs/ridge_refinement_v1.json').read_text())

    def test_prediction_export_preserves_origin_target_and_requires_matching_shape(self):
        frame=pd.DataFrame({'Date':pd.to_datetime(['2022-01-03']), 'Target_Date_1d':pd.to_datetime(['2022-01-04']), 'Reference_Return':[.01]})
        output=prediction_export(frame,np.array([.002]),{'candidate':'test'})
        self.assertEqual(output.Target_Date_1d.iloc[0],frame.Target_Date_1d.iloc[0])
        self.assertEqual(output.Reference_Return.iloc[0],.01)
        with self.assertRaises(ValueError):prediction_export(frame,np.array([.01,.02]),{})

    def test_future_extremes_do_not_enter_training_thresholds(self):
        train=pd.DataFrame({'Volatility_20':[.01,.02,.03,.04]})
        thresholds=fit_thresholds(train,self.config)
        future=pd.DataFrame({'Volatility_20':[1000.], 'SMA_Distance_50':[0.]})
        self.assertEqual(assign_regimes(future,thresholds).vol_regime.iloc[0],'high')
        self.assertEqual(thresholds,fit_thresholds(train,self.config))

    def test_threshold_and_sma_band_boundaries(self):
        frame=pd.DataFrame({'Volatility_20':[1.,2.,3.],'SMA_Distance_50':[-.01,0.,.01]})
        result=assign_regimes(frame,{'vol_low':1.,'vol_high':2.,'sma_band':.01})
        self.assertEqual(result.vol_regime.to_list(),['low','medium','high'])
        self.assertEqual(result.trend_proxy.to_list(),['near_SMA50']*3)

    def test_evaluation_winner_cannot_change_validation_selection(self):
        frame=pd.DataFrame([{'window':w,'alpha':a,'phase':p,'rmse':r} for w,a,p,r in
            [('expanding',10.,'validation',.02),('2y',100.,'validation',.01),('expanding',10.,'evaluation',.001),('2y',100.,'evaluation',.1)]])
        selected=select_config(frame,['expanding','2y'],[10.,100.])
        self.assertEqual((selected.window,selected.alpha),('2y',100.))

    def test_tie_prefers_alpha10_then_declared_window(self):
        frame=pd.DataFrame([{'window':w,'alpha':a,'phase':'validation','rmse':.01} for w in ['2y','expanding'] for a in [100.,10.]])
        selected=select_config(frame,['expanding','2y'],[100.,10.])
        self.assertEqual((selected.window,selected.alpha),('expanding',10.))

    def test_alpha_only_cannot_change_previous_window(self):
        frame=pd.DataFrame([{'window':w,'alpha':a,'phase':'validation','rmse':r} for w,a,r in [('expanding',100.,.001),('2y',10.,.02),('2y',100.,.01)]])
        selected=select_config(frame,['expanding','2y'],[10.,100.],window='2y')
        self.assertEqual((selected.window,selected.alpha),('2y',100.))
