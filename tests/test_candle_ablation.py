"""Causal FPT candle features, exact decomposition and validation-only selection."""
import unittest
import numpy as np
import pandas as pd
from tests.test_data_pipeline import sample_ohlcv
from src.features.build_features import build_dataset,FEATURE_COLUMNS
from src.features.candle_features import build_candle_dataset,RETURNS6,GAP_BODY7
from src.models.candle_ablation import choose_case
from src.models.candle_ablation_report import paired_effect


class CandleFeatureTests(unittest.TestCase):
    def test_formula_and_input_allowlist(self):
        raw=sample_ohlcv();features,_,_,_=build_candle_dataset(raw)
        row=features.loc[features.Date.eq(raw.Date.iloc[65])].iloc[0]
        self.assertAlmostEqual(row.Gap_Log,np.log(raw.Open.iloc[65]/raw.Close.iloc[64]))
        self.assertAlmostEqual(row.Body_Log,np.log(raw.Close.iloc[65]/raw.Open.iloc[65]))
        np.testing.assert_allclose(features.Gap_Log+features.Body_Log,features.Log_Return,atol=1e-12)
        self.assertEqual(len(GAP_BODY7),7)
        self.assertNotIn('Log_Return',GAP_BODY7)
        self.assertEqual(GAP_BODY7[2:],RETURNS6[1:])

    def test_future_candle_mutation_does_not_change_origin_features(self):
        raw=sample_ohlcv();changed=raw.copy();changed.loc[85:,['Open','High','Low','Close']]*=1.3
        # Change the future intraday ratio as well; uniform rescaling alone leaves body invariant.
        changed.loc[85:,'Open']=changed.loc[85:,'Close']*.99
        changed.loc[85:,'Low']=np.minimum(changed.loc[85:,'Low'],changed.loc[85:,'Open']-1)
        changed.loc[85:,'High']=np.maximum(changed.loc[85:,'High'],changed.loc[85:,'Close']+1)
        before=build_candle_dataset(raw)[0];after=build_candle_dataset(changed)[0]
        cutoff=raw.Date.iloc[84]
        np.testing.assert_allclose(before.loc[before.Date.le(cutoff),GAP_BODY7],after.loc[after.Date.le(cutoff),GAP_BODY7],atol=1e-12)

    def test_previous_bar_skips_untraded_invalid_and_quarantines_existing_labels(self):
        raw=sample_ohlcv();raw.loc[69,'Volume']=0;raw.loc[70,'High']=1.
        features,labels,_,_=build_candle_dataset(raw)
        row=features.loc[features.Date.eq(raw.Date.iloc[71])].iloc[0]
        self.assertEqual(row.Previous_Eligible_Date,raw.Date.iloc[68])
        self.assertAlmostEqual(row.Gap_Log,np.log(raw.Open.iloc[71]/raw.Close.iloc[68]))
        self.assertNotIn(raw.Date.iloc[68],labels.Date.tolist())

    def test_canonical_features_and_labels_remain_exact(self):
        raw=sample_ohlcv();f,y,_,_=build_dataset(raw);extended_f,extended_y,_,report=build_candle_dataset(raw)
        pd.testing.assert_frame_equal(f,extended_f[f.columns])
        pd.testing.assert_frame_equal(y,extended_y[y.columns])
        self.assertEqual(len(FEATURE_COLUMNS),16)
        self.assertFalse(report['canonical_schema_modified'])

    def test_case_selection_uses_registered_validation_metric(self):
        rows=[{'case':'A_returns6','model':'Ridge_shared','return_rmse':.01,'evaluation_rmse':1.},
              {'case':'B_gap_body7','model':'Ridge_shared','return_rmse':.02,'evaluation_rmse':0.}]
        self.assertEqual(choose_case(rows,'Ridge_shared'),'A_returns6')
        rows=[{'case':'A_returns6','model':'Logistic_raw','balanced_called_nonzero_pct':51.,'hit':90.},
              {'case':'B_gap_body7','model':'Logistic_raw','balanced_called_nonzero_pct':56.,'hit':40.}]
        self.assertEqual(choose_case(rows,'Logistic_raw'),'B_gap_body7')
        with self.assertRaises(ValueError):choose_case(rows,'Ridge_recent')

    def test_case_ties_prefer_A(self):
        rows=[{'case':case,'model':'Ridge_shared','return_rmse':.02} for case in ['B_gap_body7','A_returns6']]
        self.assertEqual(choose_case(rows,'Ridge_shared'),'A_returns6')

    def test_paired_rmse_gain_is_positive_for_better_B_and_reproducible(self):
        y=np.array([.02,-.01,0.,.01,-.03]);a=pd.DataFrame({'actual_log_return':y,'predicted_log_return':np.full(5,.005),'direction':np.ones(5)})
        b=a.copy();b['predicted_log_return']=y;b['direction']=np.sign(y)
        cfg={'seed':1,'block_length':2,'resamples':100}
        result=paired_effect(a,b,cfg)
        self.assertAlmostEqual(result['rmse_gain_pct'],100.)
        self.assertEqual(result,paired_effect(a,b,cfg))
