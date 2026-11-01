"""Training cutoffs, comparable windows and validation-only final selection."""
import unittest
import numpy as np
import pandas as pd
from src.models.final_window_comparison import training_rows, select_window, metric_table


class FinalWindowComparisonTests(unittest.TestCase):
    def test_window_boundary_and_unmatured_target_exclusion(self):
        dates=pd.to_datetime(['2018-03-21','2020-12-31','2021-01-04','2023-12-28','2023-12-29'])
        target=pd.to_datetime(['2018-03-22','2021-01-04','2021-01-05','2023-12-29','2024-01-02'])
        f=pd.DataFrame({'Date':dates,'Target_Date_1d':target,'value':np.arange(5)})
        small=training_rows(f,'2023-12-31',3)
        self.assertEqual(small.value.to_list(),[2,3])
        self.assertEqual(training_rows(f,'2023-12-31',5).value.to_list(),[1,2,3])
        self.assertEqual(training_rows(f,'2023-12-31',None).value.to_list(),[0,1,2,3])
        f.loc[4,'value']=99999
        self.assertEqual(training_rows(f,'2023-12-31',None).value.to_list(),[0,1,2,3])

    def test_five_year_and_full_history_can_be_identical_early(self):
        d=pd.bdate_range('2018-03-21','2022-12-30')
        f=pd.DataFrame({'Date':d[:-1],'Target_Date_1d':d[1:]})
        pd.testing.assert_frame_equal(training_rows(f,'2022-12-31',5),training_rows(f,'2022-12-31',None))

    def test_2026_scores_cannot_change_validation_selection(self):
        cfg={'validation_years':[2022,2023,2024,2025],'windows':{'3y':3,'5y':5,'all_2018':None}}
        rows=[{'phase':'validation','year':y,'horizon':1,'model':'Ridge','window':w,'rmse_log_return':s}
            for y in cfg['validation_years'] for w,s in [('3y',.02),('5y',.019),('all_2018',.021)]]
        rows.extend({'phase':'diagnostic_2026','year':2026,'horizon':1,'model':'Ridge','window':w,'rmse_log_return':s}
            for w,s in [('3y',.001),('5y',1.),('all_2018',0.)])
        f=pd.DataFrame(rows)
        self.assertEqual(select_window(f,cfg)[0],'5y')
        f.loc[f.phase.eq('diagnostic_2026'),'rmse_log_return']=99999.
        self.assertEqual(select_window(f,cfg)[0],'5y')
        f.loc[f.phase.eq('validation'),'rmse_log_return']=.02
        self.assertEqual(select_window(f,cfg)[0],'3y')
        with self.assertRaises(ValueError):select_window(f.drop(index=0),cfg)

    def test_metric_reports_target_price_error_and_zero_direction_as_na(self):
        f=pd.DataFrame({'phase':['validation']*2,'year':[2025]*2,'horizon':[1]*2,
            'window':['3y']*2,'model':['Zero-Return']*2,'origin_date':['2025-01-02','2025-01-03'],
            'target_date':['2025-01-03','2025-01-06'],'actual_log_return':[.01,-.02],
            'predicted_log_return':[0.,0.],'predicted_close':[100.,100.],'actual_close':[101.,98.],
            'lower_log_return':[None,None],'upper_log_return':[None,None]})
        row=metric_table(f).iloc[0]
        self.assertAlmostEqual(row.rmse_log_return,np.sqrt(.00025))
        self.assertAlmostEqual(row.rmse_price_vnd,np.sqrt(2.5))
        self.assertIsNone(row.hit_rate_pct);self.assertIsNone(row.coverage_pct)


if __name__=='__main__':unittest.main()
