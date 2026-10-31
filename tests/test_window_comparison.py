import unittest
import pandas as pd
from src.models.window_comparison import window_masks,select_windows


class WindowComparisonTests(unittest.TestCase):
    def test_window_changes_only_training_and_purges_boundary(self):
        frame=pd.DataFrame({'Date':pd.to_datetime(['2018-12-28','2019-01-02','2020-12-31','2021-02-01','2022-02-01']),
                            'Target_Date_1d':pd.to_datetime(['2018-12-31','2019-01-03','2021-01-04','2021-02-02','2022-02-02'])})
        fold={'train_end':'2020-12-31','validation_end':'2021-12-31','evaluation_end':'2022-12-31'}
        full=window_masks(frame,fold,None);recent=window_masks(frame,fold,2)
        self.assertEqual(recent['train'].tolist(),[False,True,False,False,False])
        self.assertEqual(full['validation'].tolist(),recent['validation'].tolist())
        self.assertEqual(full['evaluation'].tolist(),recent['evaluation'].tolist())
        with self.assertRaises(ValueError):window_masks(frame,fold,0)

    def test_window_is_selected_by_validation_even_when_evaluation_is_worse(self):
        data=pd.DataFrame([{'fold':'f','model':'LSTM','window':w,'phase':p,'rmse_mean':r}
            for w,p,r in [('expanding','validation',.02),('2y','validation',.01),('expanding','evaluation',.01),('2y','evaluation',.03)]])
        result=select_windows(data,['expanding','2y']).iloc[0]
        self.assertEqual(result.selected_window,'2y')
        self.assertLess(result.evaluation_rmse_gain_pct,0)

    def test_ties_prefer_predeclared_expanding_control(self):
        data=pd.DataFrame([{'fold':'f','model':'Ridge','window':w,'phase':p,'rmse_mean':.01}
            for w in ['2y','expanding'] for p in ['validation','evaluation']])
        self.assertEqual(select_windows(data,['expanding','2y']).iloc[0].selected_window,'expanding')
