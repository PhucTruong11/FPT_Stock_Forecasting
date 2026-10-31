"""Research-only gap/body decomposition; canonical feature schema stays unchanged."""
import numpy as np
from src.data.contracts import normalize_ohlcv,invalid_candle_mask
from src.features.build_features import build_dataset

RETURNS6=['Log_Return','Return_Lag_1','Return_Lag_2','Return_Lag_5','Return_5','Return_20']
GAP_BODY7=['Gap_Log','Body_Log',*RETURNS6[1:]]


def build_candle_dataset(frame):
    raw=normalize_ohlcv(frame,allow_untraded_missing_prices=True,allow_invalid_candles=True)
    valid=raw.loc[raw.Volume.gt(0)&~invalid_candle_mask(raw)].copy().reset_index(drop=True)
    valid['Previous_Eligible_Date']=valid.Date.shift(1)
    valid['Previous_Eligible_Close']=valid.Close.shift(1)
    valid['Gap_Log']=np.log(valid.Open/valid.Previous_Eligible_Close)
    valid['Body_Log']=np.log(valid.Close/valid.Open)
    features,labels,excluded,report=build_dataset(raw)
    context=valid[['Date','Open','Previous_Eligible_Date','Previous_Eligible_Close','Gap_Log','Body_Log']]
    features=features.merge(context,on='Date',how='left',validate='one_to_one')
    labels=labels.merge(context,on='Date',how='left',validate='one_to_one')
    for part in [features,labels]:
        if not np.isfinite(part[GAP_BODY7].to_numpy(dtype=float)).all():raise ValueError('Nonfinite candle features; no silent row dropping.')
        if not part.Previous_Eligible_Date.lt(part.Date).all():raise ValueError('Gap must use a prior eligible bar.')
        if not np.allclose(part.Gap_Log+part.Body_Log,part.Log_Return,atol=1e-12,rtol=1e-10):raise ValueError('Gap/body identity failed.')
    return features,labels,excluded,{**report,'research_extension':'gap/body decomposition; FPT only',
        'previous_basis':'same valid positive-volume sequence as canonical builder; not independently certified exchange calendar',
        'A_columns':RETURNS6,'B_columns':GAP_BODY7,'canonical_schema_modified':False}
