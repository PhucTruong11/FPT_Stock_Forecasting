import unittest
import numpy as np
import pandas as pd
from src.models.evaluation_2026 import price_metrics,block_bootstrap,aligned_cohort
from src.models.evaluation_2026_report import psi_from_training


class Frozen2026Tests(unittest.TestCase):
    def test_price_scores_share_reference_anchor(self):
        result=price_metrics(np.log([1.1,1.2]),[0.,0.],[100.,100.])
        self.assertAlmostEqual(result['price_mae_adjusted_vnd'],15.)
        self.assertAlmostEqual(result['price_mape_pct'],100*np.mean([10/110,20/120]))

    def test_bootstrap_zero_matches_zero_benchmark_exactly(self):
        result=block_bootstrap(np.array([.01,-.02,.03,-.01]),np.zeros(4),{'seed':1,'block_length':2,'resamples':100})
        self.assertAlmostEqual(result['mse_skill_ci_low'],0.)
        self.assertAlmostEqual(result['mse_skill_ci_high'],0.)

    def test_bootstrap_is_reproducible(self):
        cfg={'seed':12,'block_length':2,'resamples':100}
        a=block_bootstrap(np.array([.01,-.02,.03,-.01]),np.full(4,.001),cfg)
        self.assertEqual(a,block_bootstrap(np.array([.01,-.02,.03,-.01]),np.full(4,.001),cfg))

    def test_psi_identical_distribution_is_zero(self):
        a=np.linspace(-1,1,100)
        self.assertAlmostEqual(psi_from_training(a,a),0.)

    def test_common_cohort_matches_both_origin_and_target_and_cutoff(self):
        features=pd.DataFrame({'Date':pd.bdate_range('2025-11-01',periods=100)})
        labels=pd.DataFrame({'Date':features.Date.iloc[60:65].to_numpy(),'Target_Date_1d':features.Date.iloc[61:66].to_numpy(),'Target_Return_1d':[.01]*5,'Close':[100]*5})
        other=labels.copy();other.loc[0,'Target_Date_1d']=features.Date.iloc[62]
        end=str(labels.Target_Date_1d.iloc[-1].date())
        cohort=aligned_cohort({'yahoo':(features,labels),'vci':(features,other)},'2026-01-01',end)
        self.assertEqual(len(cohort),3)
        self.assertNotIn(labels.Date.iloc[0],cohort.Date.to_list())
        self.assertTrue(cohort.Target_Date_1d.lt(end).all())
