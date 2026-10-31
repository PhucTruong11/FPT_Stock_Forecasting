"""Offline inference for a fixed research direction bundle; no serving promotion."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from src.data.contracts import BASE_DIR,file_sha256
from src.models.direction_selective import log_odds,decide


def predict_research_signal(bundle_path,features,model_name='Logistic'):
    """Accept causal features, not raw OHLCV; caller owns observation/EOD validity."""
    bundle=json.loads(Path(bundle_path).read_text(encoding='utf-8'))
    if bundle['automatic_promotion'] is not False:raise ValueError('Research-only bundle required.')
    columns=bundle['features']
    if 'Date' not in features or not set(columns).issubset(features):raise ValueError('Date and all six feature columns required.')
    dates=pd.to_datetime(features.Date)
    if dates.isna().any():raise ValueError('Valid origin dates required.')
    if dates.le(bundle['threshold_validation_target_cutoff']).any():raise ValueError('Origin must be after threshold-validation labels were available.')
    if features.empty or not np.isfinite(features[columns].to_numpy(dtype=float)).all():raise ValueError('Finite, nonempty feature rows required.')
    if model_name not in ['Logistic','XGBoost']:raise ValueError('Choose Logistic or XGBoost.')
    artifacts={r['name']:r for r in bundle['models']};selected=artifacts[model_name];sigmoid=artifacts[model_name+'_sigmoid']
    for item in [selected,sigmoid]:
        if file_sha256(BASE_DIR/item['path'])!=item['sha256']:raise ValueError('Research artifact hash mismatch.')
    path=BASE_DIR/selected['path']
    if path.suffix=='.json':model=XGBClassifier();model.load_model(path)
    else:model=joblib.load(path)
    calibrator=joblib.load(BASE_DIR/sigmoid['path']);raw=model.predict_proba(features[columns])[:,1].astype(float)
    p=calibrator.predict_proba(log_odds(raw))[:,1]
    margin=float(next(r['margin'] for r in bundle['thresholds'] if r['model']==model_name));direction=decide(p,margin)
    return pd.DataFrame({'Date':dates.to_numpy(),'model':model_name,'raw_probability_up':raw,
        'estimated_probability_up_nonzero':p,'direction':direction,
        'status':np.where(direction>0,'tang',np.where(direction<0,'giam','chua_du_tin_hieu')),
        'margin':margin,'role':'offline research output; not a timestamped live forecast or serving recommendation'})
