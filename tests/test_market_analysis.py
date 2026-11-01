"""Independent descriptive-math checks; no model fitting or network access."""
import unittest
import numpy as np
import pandas as pd
from src.app.market_analysis import prepare_market_frame, window_statistics, monthly_returns


def candles(closes):
    closes=np.asarray(closes,float)
    return pd.DataFrame({'Date':pd.bdate_range('2024-01-01',periods=len(closes)),
        'Open':closes,'High':closes+1,'Low':closes-1,'Close':closes,'Volume':np.full(len(closes),100.)})


class MarketAnalysisTests(unittest.TestCase):
    def test_statistics_use_selected_close_path_and_do_not_confuse_up_rate_with_accuracy(self):
        stats=window_statistics(candles([100,110,99]))
        self.assertAlmostEqual(stats['return_pct'],-1.)
        self.assertAlmostEqual(stats['max_drawdown_pct'],-10.)
        self.assertEqual(stats['up_pct'],50.);self.assertEqual(stats['down_pct'],50.)
        self.assertEqual(stats['daily_return_count'],2)

    def test_invalid_and_untraded_candles_never_distort_indicators(self):
        raw=candles(np.linspace(100,140,45));raw.loc[10,['Close','High']]=999999.
        raw.loc[10,'Volume']=0;raw.loc[20,'High']=1.
        data,quality=prepare_market_frame(raw)
        self.assertEqual(len(data),43);self.assertEqual(quality['untraded_rows'],1)
        self.assertEqual(quality['invalid_traded_rows'],1)
        self.assertLess(data.SMA_20.dropna().max(),141)
        with self.assertRaises(ValueError):prepare_market_frame(pd.concat([raw,raw.iloc[[0]]]))

    def test_indicators_use_full_past_context_and_never_future_prices(self):
        raw=candles(np.linspace(100,160,260));a,_=prepare_market_frame(raw)
        self.assertAlmostEqual(a.SMA_200.iloc[-1],raw.Close.tail(200).mean())
        self.assertTrue(a.tail(22).SMA_200.notna().all())
        changed=raw.copy();changed.loc[200:,['Open','High','Low','Close']]*=10
        b,_=prepare_market_frame(changed)
        columns=['SMA_20','SMA_50','RSI_14','MACD','MACD_signal','Volume_ratio20','ATR14_pct']
        np.testing.assert_allclose(a.loc[100,columns].to_numpy(float),b.loc[100,columns].to_numpy(float),atol=0,rtol=0)

    def test_flat_rsi_neutral_and_gap_body_definitions(self):
        data,_=prepare_market_frame(candles(np.full(40,100.)))
        self.assertTrue(data.RSI_14.dropna().eq(50.).all())
        raw=candles([100,110]);raw.loc[1,'Open']=105.;raw.loc[1,'Low']=104.
        d,_=prepare_market_frame(raw)
        self.assertAlmostEqual(d.Gap_pct.iloc[-1],5.)
        self.assertAlmostEqual(d.Body_pct.iloc[-1],(110/105-1)*100)

    def test_missing_months_are_not_forward_filled(self):
        raw=pd.DataFrame({'Date':pd.to_datetime(['2026-01-30','2026-03-31','2026-04-30']), 'Close':[100.,120.,132.]})
        result=monthly_returns(raw,'2026-01-01').set_index('Month')
        self.assertTrue(pd.isna(result.loc[2,'Return_pct']))
        self.assertTrue(pd.isna(result.loc[3,'Return_pct']))
        self.assertAlmostEqual(result.loc[4,'Return_pct'],10.)


if __name__=='__main__':unittest.main()
