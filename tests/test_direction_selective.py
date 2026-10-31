"""Non-neutral learning, honest abstention accounting and temporal threshold guards."""
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.models.direction_selective import decide,direction_metrics,probability_metrics,select_margin,phase_mask,log_odds,paired_bootstrap


class DirectionSelectiveTests(unittest.TestCase):
    def spec(self):
        return {'margins':[0.,.025,.05,.2],'minimum_coverage_pct':60.,'minimum_nonzero_calls':30,'minimum_calls_per_actual_direction':10}

    def test_boundary_and_full_coverage(self):
        np.testing.assert_array_equal(decide([0.,.4,.5,.6,1.],.1),[-1,0,0,1,1])
        np.testing.assert_array_equal(decide([0.,.5,1.]),[-1,1,1])
        for values in [[np.nan],[1.1],[-.1]]:
            with self.assertRaises(ValueError):decide(values)

    def test_neutral_is_not_a_correct_abstention(self):
        result=direction_metrics([.01,-.01,0.,0.],[1,-1,0,1])
        self.assertEqual(result['correct_n'],2)
        self.assertEqual(result['calls_n'],3)
        self.assertEqual(result['neutral_calls_n'],1)
        self.assertAlmostEqual(result['called_accuracy_all_pct'],200/3)
        self.assertEqual(result['hit_nonzero_called_pct'],100.)
        self.assertEqual(result['correct_over_all_origins_pct'],50.)

    def test_no_calls_preserves_missing_metrics(self):
        r=direction_metrics([.01,-.01,0],[0,0,0])
        self.assertEqual(r['coverage_pct'],0.)
        self.assertIsNone(r['called_accuracy_all_pct'])
        self.assertIsNone(r['balanced_called_nonzero_pct'])
        self.assertEqual(r['correct_n'],0)

    def test_balanced_needs_both_actual_directions_in_calls(self):
        r=direction_metrics([.01,-.01],[1,0])
        self.assertEqual(r['up_coverage_pct'],100.)
        self.assertEqual(r['down_coverage_pct'],0.)
        self.assertIsNone(r['balanced_called_nonzero_pct'])

    def test_selects_validation_filter_subject_to_coverage_and_tie_break(self):
        y=np.tile([-1.,1.],50)
        probability=np.where(y>0,.8,.2);probability[60:]=np.where(y[60:]>0,.49,.51)
        margin,reason,rows=select_margin(y,probability,self.spec())
        self.assertEqual(margin,.025)
        selected=next(r for r in rows if r['margin']==margin)
        self.assertEqual(selected['coverage_pct'],60.)
        self.assertEqual(selected['balanced_called_nonzero_pct'],100.)
        # A changed future distribution must not trigger test-time threshold retuning/fallback.
        evaluation=direction_metrics(y,decide(np.full(100,.5),margin))
        self.assertEqual(evaluation['calls_n'],0)

    def test_ties_keep_full_coverage_and_no_eligible_margin_falls_back(self):
        y=np.tile([-1.,1.],50);probability=np.where(y>0,.95,.05)
        self.assertEqual(select_margin(y,probability,self.spec())[0],0.)
        margin,reason,_=select_margin(y[:10],probability[:10],self.spec())
        self.assertEqual(margin,0.)
        self.assertIn('fallback',reason)

    def test_probability_scores_use_whole_nonzero_cohort_not_only_calls(self):
        r=probability_metrics([.01,-.01,0],[.8,.2,.99])
        self.assertEqual(r['probability_n'],2)
        self.assertAlmostEqual(r['brier_nonzero'],.04)
        self.assertAlmostEqual(r['logloss_nonzero'],-np.log(.8))
        self.assertTrue(np.isfinite(log_odds([0,1])).all())

    def test_phase_purges_targets_crossing_boundary(self):
        frame=pd.DataFrame({'Date':pd.to_datetime(['2025-06-30','2025-07-01','2025-09-30']),
            'Target_Date_1d':pd.to_datetime(['2025-07-01','2025-07-02','2025-10-01'])})
        self.assertEqual(phase_mask(frame,'2025-07-01','2025-09-30').tolist(),[False,True,False])

    def test_paired_bootstrap_same_majority_control_is_exact_zero(self):
        cfg={'seed':42,'resamples':100,'block_length':3}
        y=np.array([.01,-.01,0,-.01]);d=np.full(4,-1)
        result=paired_bootstrap(y,d,-1,cfg)
        self.assertEqual(result['hit_delta_majority_ci_low_pp'],0.)
        self.assertEqual(result['hit_delta_majority_ci_high_pp'],0.)
        self.assertEqual(result,paired_bootstrap(y,d,-1,cfg))
        missing=paired_bootstrap(y,np.zeros(4,dtype=int),-1,cfg)
        self.assertIsNone(missing['hit_delta_majority_ci_low_pp'])
        self.assertEqual(missing['bootstrap_valid_resamples'],0)

    def test_offline_inference_returns_three_states_and_verifies_artifacts(self):
        from src.models.predict_direction import predict_research_signal
        from src.data.contracts import file_sha256
        class Raw:
            def predict_proba(self,x):
                p=x.iloc[:,0].to_numpy();return np.column_stack([1-p,p])
        class Sigmoid:
            def predict_proba(self,x):
                p=1/(1+np.exp(-x[:,0]));return np.column_stack([1-p,p])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw.pkl';cal=root/'cal.pkl';raw.write_bytes(b'raw');cal.write_bytes(b'cal')
            bundle=root/'bundle.json';bundle.write_text(json.dumps({'automatic_promotion':False,'features':['Log_Return'],
                'threshold_validation_target_cutoff':'2025-12-31','thresholds':[{'model':'Logistic','margin':.1}],
                'models':[{'name':'Logistic','path':str(raw),'sha256':file_sha256(raw)},
                          {'name':'Logistic_sigmoid','path':str(cal),'sha256':file_sha256(cal)}]}))
            features=pd.DataFrame({'Date':pd.date_range('2026-01-01',periods=3),'Log_Return':[.8,.5,.2]})
            with patch('src.models.predict_direction.joblib.load',side_effect=[Raw(),Sigmoid()]):
                output=predict_research_signal(bundle,features)
            self.assertEqual(output.status.tolist(),['tang','chua_du_tin_hieu','giam'])
            raw.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):predict_research_signal(bundle,features)

    def test_offline_inference_rejects_pre_validation_and_invalid_origins(self):
        from src.models.predict_direction import predict_research_signal
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bundle.json';path.write_text(json.dumps({'automatic_promotion':False,'features':['Log_Return'],
                'threshold_validation_target_cutoff':'2025-12-31'}))
            for date,value in [('2025-12-30',.01),(None,.01),('2026-01-02',np.nan)]:
                with self.assertRaises(ValueError):predict_research_signal(path,pd.DataFrame({'Date':[date],'Log_Return':[value]}))
