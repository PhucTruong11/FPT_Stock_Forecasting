"""Post-result diagnosis only: inspect probability collapse, no fitting or selection."""
from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/data/contracts.py').exists())
sys.path.insert(0,str(ROOT))
from src.data.contracts import file_sha256
from src.data.source_audit import save_json
from src.models.direction_selective import direction_metrics
STUDY=Path(__file__).resolve().parent
manifest=json.loads((STUDY/'manifest.json').read_text())
p=pd.read_csv(STUDY/'predictions.csv.gz',parse_dates=['Date','Target_Date_1d'])
rows=[]
for r in manifest['artifacts']:
    if r['role']!='calibrator':continue
    assert file_sha256(ROOT/r['path'])==r['sha256']
    model=r['name'].removesuffix('_sigmoid');cal=joblib.load(ROOT/r['path'])
    g=p[(p.fold==r['fold'])&(p.phase=='calibration')&(p.strategy==model)]
    h=p[(p.fold==r['fold'])&(p.phase=='evaluation')&(p.strategy==model+'_calibrated')]
    rows.append({'fold':r['fold'],'model':model,'sigmoid_coef':float(cal.coef_[0,0]),'sigmoid_intercept':float(cal.intercept_[0]),
        'calibration_up_pct':float(100*(g.actual_log_return>0).mean()),'evaluation_probability_min':float(h.probability_up.min()),
        'evaluation_probability_max':float(h.probability_up.max()),'evaluation_up_calls':int((h.direction>0).sum()),
        'evaluation_down_calls':int((h.direction<0).sum()),'evaluation_n':len(h)})
pd.DataFrame(rows).to_csv(STUDY/'probability_diagnostics.csv',index=False)
# Existing bias-control reference on the same 104 pairs, disclosed after training, never used for tuning.
reference=ROOT/'reports/period_comparison/20261030T024048531444Z/predictions.csv.gz'
previous=pd.read_csv(reference,parse_dates=['Date','Target_Date_1d'])
old=previous[previous.strategy=='refit_2025_native_bias60']
keys=old[['Date','Target_Date_1d']];out=[]
for strategy,g in p[(p.fold=='eval_2026')&(p.phase=='evaluation')].groupby('strategy'):
    group=g.merge(keys,on=['Date','Target_Date_1d'],validate='one_to_one')
    out.append({'strategy':strategy,**direction_metrics(group.actual_log_return,group.direction)})
out.append({'strategy':'Existing_Ridge_bias_reference',**direction_metrics(old.actual_log_return,np.sign(old.predicted_log_return).astype(int))})
pd.DataFrame(out).to_csv(STUDY/'comparison_existing_jun_oct.csv',index=False)
save_json(STUDY/'diagnostic_provenance.json',{'role':'post-result descriptive comparison only; no refit, threshold/model selection or promotion',
    'existing_reference_file':reference.relative_to(ROOT).as_posix(),'existing_reference_sha256':file_sha256(reference),
    'script_sha256':file_sha256(Path(__file__))})
print(pd.DataFrame(rows).query("fold=='eval_2026'").to_string(index=False))
print(pd.DataFrame(out)[['strategy','calls_n','coverage_pct','hit_nonzero_called_pct','balanced_called_nonzero_pct']].round(3).to_string(index=False))
