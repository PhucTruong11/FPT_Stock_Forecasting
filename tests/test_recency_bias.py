"""Temporal leakage, calibration and append-only forward evidence checks."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.models.recency_bias import training_mask,causal_calibration
from src.models.source_comparison import REFERENCE
from src.models.forward_ledger import append_event,read_ledger,validate_issue_dates,mature_outcome,score,issue


class TemporalCalibrationTests(unittest.TestCase):
    def frame(self,n=85):
        dates=pd.bdate_range('2026-01-01',periods=n+1)
        return pd.DataFrame({'Date':dates[:-1],'Target_Date_1d':dates[1:],REFERENCE:np.sin(np.arange(n))*.02,'raw_prediction':np.full(n,.005)})

    def calibrate(self,f):
        return causal_calibration(f,{'window':60,'minimum_matured':20},{'window':60,'minimum_matured':60,'quantiles':[.1,.9]})

    def test_training_window_purges_label_at_boundary(self):
        f=pd.DataFrame({'Date':pd.to_datetime(['2022-12-30','2023-01-02','2025-12-31']),
                        'Target_Date_1d':pd.to_datetime(['2023-01-03','2023-01-03','2026-01-02'])})
        self.assertEqual(training_mask(f,'2025-12-31',3).tolist(),[False,True,False])

    def test_future_and_current_label_changes_do_not_change_forecasts(self):
        f=self.frame();original=self.calibrate(f)
        changed=f.copy();changed.loc[45:,REFERENCE]=123.
        updated=self.calibrate(changed)
        for col in ['prediction','bias_offset','lower_log_return','upper_log_return']:
            np.testing.assert_allclose(original.loc[:45,col],updated.loc[:45,col],equal_nan=True)
        self.assertNotEqual(original.loc[46,'prediction'],updated.loc[46,'prediction'])

    def test_warmup_and_interval_use_previous_corrected_residuals(self):
        f=self.frame();result=self.calibrate(f)
        self.assertTrue(result.loc[:19,'bias_offset'].eq(0).all())
        self.assertAlmostEqual(result.loc[20,'bias_offset'],(f.loc[:19,REFERENCE]-.005).mean())
        self.assertTrue(result.loc[:59,'lower_log_return'].isna().all())
        errors=f.loc[:59,REFERENCE]-result.loc[:59,'prediction']
        self.assertAlmostEqual(result.loc[60,'lower_log_return'],result.loc[60,'prediction']+errors.quantile(.1))
        self.assertEqual(result.loc[84,'calibration_n'],60)

    def test_delayed_labels_are_not_available_even_with_earlier_origin(self):
        f=self.frame();f.loc[0,'Target_Date_1d']=pd.Timestamp('2027-01-01')
        result=self.calibrate(f)
        self.assertEqual(result.loc[20,'calibration_n'],19)
        self.assertEqual(result.loc[20,'bias_offset'],0.)


class ForwardEvidenceTests(unittest.TestCase):
    def test_hash_chain_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'events.jsonl'
            append_event(path,{'kind':'forecast','value':1})
            append_event(path,{'kind':'outcome','value':2})
            self.assertEqual(len(read_ledger(path)),2)
            path.write_text(path.read_text().replace('"value": 1','"value": 9'),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'integrity'):read_ledger(path)

    def test_issue_cannot_use_future_training_origin_or_intraday_bar(self):
        now=pd.Timestamp('2026-10-30T02:00:00Z')
        validate_issue_dates('2026-10-29','2026-10-29','2026-10-30T01:00:00Z',now)
        cases=[('2026-10-31','2026-10-29','2026-10-30T01:00:00Z'),
               ('2026-10-29','2026-10-30','2026-10-30T01:00:00Z'),
               ('2026-10-29','2026-10-29','2026-10-30T03:00:00Z'),
               ('2026-10-30','2026-10-29','2026-10-30T01:00:00Z')]
        for origin,train,created in cases:
            with self.assertRaises(ValueError):validate_issue_dates(origin,train,created,now)

    def forecast(self):
        return {'origin_date':'2026-10-29','issued_local_date':'2026-10-30','origin_close_adjusted':100.,
                'forecasts':{'Ridge_raw':.01,'Zero-Return':0.},'event_sha256':'abc','interval':None,'kind':'forecast'}

    def source(self):
        return pd.DataFrame({'Date':pd.to_datetime(['2026-10-29','2026-10-30','2026-11-02']),
            'Open':[50,55,57],'High':[51,56,58],'Low':[49,54,56],'Close':[50,55,57],'Volume':[100,100,100]})

    def test_maturity_requires_completed_target_and_consistent_adjustment_basis(self):
        raw=self.source();f=self.forecast()
        self.assertIsNone(mature_outcome(f,raw.iloc[:1],pd.Timestamp('2026-10-31T02:00Z')))
        self.assertIsNone(mature_outcome(f,raw,pd.Timestamp('2026-10-30T12:00Z')))
        result=mature_outcome(f,raw,pd.Timestamp('2026-10-31T02:00Z'))
        self.assertEqual(result['actual_target_date'],'2026-10-30')
        self.assertAlmostEqual(result['actual_log_return'],np.log(55/50))
        self.assertEqual(result['origin_adjustment_ratio'],.5)
        self.assertFalse(result['full_session_eligible'])
        f['issued_local_date']='2026-10-29'
        self.assertTrue(mature_outcome(f,raw,pd.Timestamp('2026-10-31T02:00Z'))['full_session_eligible'])

    def test_scoring_is_idempotent_and_preserves_forecast(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=Path(tmp)/'events.jsonl';source=Path(tmp)/'data.csv'
            self.source().iloc[:2].to_csv(source,index=False)
            payload=self.forecast();payload.pop('event_sha256');original=append_event(ledger,payload)
            self.assertEqual(score(source,ledger,pd.Timestamp('2026-10-31T02:00Z')),1)
            self.assertEqual(score(source,ledger,pd.Timestamp('2026-10-31T02:00Z')),0)
            self.assertEqual(read_ledger(ledger)[0],original)

    def test_duplicate_issue_rejected(self):
        # Exercise the full issuance path with real feature generation and a mocked fixed model.
        from tests.test_data_pipeline import sample_ohlcv
        from src.data.contracts import file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model.pkl';model.write_bytes(b'fixed-test-model')
            source=root/'data.csv';sample_ohlcv().to_csv(source,index=False)
            bundle=root/'bundle.json';bundle.write_text(json.dumps({'automatic_promotion':False,'model_path':str(model),
                'model_sha256':file_sha256(model),'training_target_end':'2019-12-31','created_at_utc':'2020-01-01T00:00Z',
                'calibration_targets':[],'feature_columns':['Log_Return'],'raw_errors':[],'corrected_errors':[],
                'bias_spec':{'minimum_matured':20},'interval_spec':{'minimum_matured':60},'train_mean':0.}))
            class Model:
                def predict(self,x):return np.zeros(len(x))
            with patch('src.models.forward_ledger.joblib.load',return_value=Model()):
                issue(bundle,source,root/'events.jsonl',pd.Timestamp('2020-07-01T00:00Z'))
                with self.assertRaisesRegex(ValueError,'already logged'):
                    issue(bundle,source,root/'events.jsonl',pd.Timestamp('2020-07-01T00:00Z'))
