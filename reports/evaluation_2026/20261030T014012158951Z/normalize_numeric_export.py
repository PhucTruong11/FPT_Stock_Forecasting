"""Repair derived float32-exp price exports from unchanged saved return forecasts.

No data download, model fitting, model selection or new return forecast is performed.
"""
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

STUDY=Path(__file__).resolve().parent
ROOT=STUDY.parents[2]
sys.path.insert(0,str(ROOT))
from src.data.source_audit import save_json
from src.data.contracts import file_sha256,utc_now
from src.models.evaluation_2026 import price_metrics,block_bootstrap
from src.models.robustness import score_direction


def main():
    if (STUDY/'numeric_export_correction.json').exists():
        raise ValueError('Correction already recorded; do not overwrite original evidence.')
    original=STUDY/'initial_export.zip'
    with zipfile.ZipFile(original,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in ['metrics.csv','predictions.csv.gz','status.json']:archive.write(STUDY/name,name)
    config=json.loads((STUDY/'protocol.json').read_text())['config']
    d=pd.read_csv(STUDY/'predictions.csv.gz',parse_dates=['Date','Target_Date_1d'])
    original_returns=d.predicted_log_return.to_numpy().copy()
    original_reference=d.reference_log_return.to_numpy().copy()
    d['predicted_close_adjusted']=d.reference_origin_close.to_numpy(dtype=float)*np.exp(original_returns)
    d['native_predicted_close_adjusted']=d.native_origin_close.to_numpy(dtype=float)*np.exp(original_returns)
    d['actual_close_adjusted']=d.reference_origin_close.to_numpy(dtype=float)*np.exp(original_reference)
    rows=[]
    for (model,cohort),part in d.groupby(['model','cohort'],sort=False):
        t,q=part.reference_log_return.to_numpy(),part.predicted_log_return.to_numpy()
        rows.append({'model':model,'cohort':cohort,**score_direction(t,q),**price_metrics(t,q,part.reference_origin_close),
                     **block_bootstrap(t,q,config['bootstrap'])})
    assert np.array_equal(d.predicted_log_return.to_numpy(),original_returns)
    assert np.array_equal(d.reference_log_return.to_numpy(),original_reference)
    d.to_csv(STUDY/'predictions.csv.gz',index=False,compression='gzip',date_format='%Y-%m-%d',lineterminator='\n')
    pd.DataFrame(rows).to_csv(STUDY/'metrics.csv',index=False,lineterminator='\n')
    save_json(STUDY/'numeric_export_correction.json',{'corrected_at_utc':utc_now(),
        'reason':'LSTM np.exp(float32) price export differed from float64 metric arithmetic by at most ~0.011 adjusted VND.',
        'method':'normalize derived prices and all scores from unchanged persisted return forecasts using float64',
        'no_download_or_refit_or_model_selection':True,'saved_return_and_reference_values_unchanged_before_reserialization':True,
        'original_export_zip_sha256':file_sha256(original),'script_sha256':file_sha256(Path(__file__)),
        'scoring_source_sha256':{n:file_sha256(ROOT/n) for n in ['src/models/evaluation_2026.py','src/models/evaluation_2026_report.py']}})
    print('Derived numeric exports normalized; frozen models/return forecasts unchanged.')


if __name__=='__main__':main()
