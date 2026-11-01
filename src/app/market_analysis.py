"""Descriptive snapshot analytics only: no forecasts, training, or market fetches."""
from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_market_frame(observations):
    data=pd.DataFrame(observations).copy()
    required=['Date','Open','High','Low','Close','Volume']
    if not set(required).issubset(data):
        raise ValueError('OHLCV snapshot is missing required columns.')
    data['Date']=pd.to_datetime(data.Date,errors='raise')
    if data.Date.isna().any() or data.Date.duplicated().any():
        raise ValueError('Snapshot dates must be present and unique.')
    data=data.sort_values('Date').reset_index(drop=True)
    values=data[required[1:]].apply(pd.to_numeric,errors='coerce')
    traded=values.Volume.gt(0)
    valid=(np.isfinite(values).all(axis=1)&values[['Open','High','Low','Close']].gt(0).all(axis=1)&
        values.High.ge(values[['Open','Close','Low']].max(axis=1))&values.Low.le(values[['Open','Close','High']].min(axis=1)))
    quality={'source_rows':len(data),'untraded_rows':int((~traded).sum()),
        'invalid_traded_rows':int((traded&~valid).sum())}
    data[required[1:]]=values
    data=data.loc[traded&valid].reset_index(drop=True)
    if len(data)<2:
        raise ValueError('Need at least two valid traded observations for analysis.')
    close=data.Close
    data['Return_pct']=close.pct_change(fill_method=None)*100
    data['Log_Return']=np.log(close).diff()
    data['Gap_pct']=(data.Open/close.shift(1)-1)*100
    data['Body_pct']=(close/data.Open-1)*100
    for n in [20,50,200]:data[f'SMA_{n}']=close.rolling(n,min_periods=n).mean()
    std=close.rolling(20,min_periods=20).std(ddof=0)
    data['BB_lower']=data.SMA_20-2*std;data['BB_upper']=data.SMA_20+2*std
    change=close.diff().fillna(0)
    gain=change.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    loss=(-change.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rsi=100-100/(1+gain/loss.replace(0,np.nan))
    rsi=rsi.mask(loss.eq(0)&gain.gt(0),100).mask(loss.gt(0)&gain.eq(0),0).mask(loss.eq(0)&gain.eq(0),50)
    data['RSI_14']=rsi
    data['MACD']=close.ewm(span=12,adjust=False,min_periods=12).mean()-close.ewm(span=26,adjust=False,min_periods=26).mean()
    data['MACD_signal']=data.MACD.ewm(span=9,adjust=False,min_periods=9).mean()
    data['MACD_hist']=data.MACD-data.MACD_signal
    data['Volume_MA20_prior']=data.Volume.shift(1).rolling(20,min_periods=20).mean()
    data['Volume_ratio20']=data.Volume/data.Volume_MA20_prior.replace(0,np.nan)
    data['Volatility20_annual_pct']=data.Log_Return.rolling(20,min_periods=20).std(ddof=1)*np.sqrt(252)*100
    true_range=pd.concat([data.High-data.Low,(data.High-close.shift(1)).abs(),(data.Low-close.shift(1)).abs()],axis=1).max(axis=1)
    data['ATR14_pct']=true_range.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/close*100
    data['Prior20_low']=data.Low.shift(1).rolling(20,min_periods=20).min()
    data['Prior20_high']=data.High.shift(1).rolling(20,min_periods=20).max()
    quality['eligible_rows']=len(data)
    return data,quality


def window_statistics(part):
    close=part.Close.to_numpy(float)
    if len(close)<2:
        raise ValueError('Need at least two closes in the selected window.')
    daily=pd.Series(close).pct_change(fill_method=None).dropna()
    drawdown=(part.Close/part.Close.cummax()-1)*100
    return {'observations':len(part),'return_pct':float((close[-1]/close[0]-1)*100),
        'max_drawdown_pct':float(drawdown.min()),'up_pct':float(daily.gt(0).mean()*100),
        'down_pct':float(daily.lt(0).mean()*100),'flat_pct':float(daily.eq(0).mean()*100),
        'best_day_pct':float(daily.max()*100),'worst_day_pct':float(daily.min()*100),
        'annual_volatility_pct':float(np.log(pd.Series(close)).diff().std(ddof=1)*np.sqrt(252)*100) if len(close)>2 else None,
        'daily_return_count':len(daily)}


def monthly_returns(data, start_date):
    """Full-snapshot month-end observed closes; blank missing months, never ffill."""
    ends=data.set_index('Date').Close.resample('ME').last()
    result=ends.pct_change(fill_method=None)*100
    # A complete first month may be absent; do not invent a previous month close.
    result=result.loc[result.index.to_period('M')>=pd.Timestamp(start_date).to_period('M')]
    frame=pd.DataFrame({'Date':result.index,'Return_pct':result.to_numpy()})
    frame['Year']=frame.Date.dt.year;frame['Month']=frame.Date.dt.month
    return frame


def snapshot_insights(data):
    last=data.iloc[-1];trend=[]
    for n in [20,50,200]:
        average=last[f'SMA_{n}']
        if pd.notna(average):
            distance=(last.Close/average-1)*100
            trend.append(f"Giá {'trên' if distance>=0 else 'dưới'} SMA{n} {abs(distance):.2f}%")
    rsi=last.RSI_14
    momentum='Chưa đủ quan sát cho RSI14.' if pd.isna(rsi) else f"RSI14 {rsi:.1f}: {'vùng cao (≥70)' if rsi>=70 else 'vùng thấp (≤30)' if rsi<=30 else 'giữa hai ngưỡng 30–70'}."
    if pd.notna(last.MACD_hist):momentum+=f" MACD {'trên' if last.MACD_hist>=0 else 'dưới'} đường tín hiệu."
    ratio=last.Volume_ratio20
    volume='Chưa đủ 20 quan sát trước để đối chiếu khối lượng.' if pd.isna(ratio) else f"Khối lượng phiên cuối bằng {ratio:.2f}× trung bình 20 quan sát trước đó."
    return {'trend':'; '.join(trend)+'.' if trend else 'Chưa đủ quan sát cho SMA.',
        'momentum':momentum,'volume':volume}
