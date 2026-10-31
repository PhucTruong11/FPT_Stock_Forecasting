"""Describe frozen-model directional bias and exact linear Ridge contributions."""
import io
import json
import sys
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

STUDY=Path(__file__).resolve().parent;ROOT=STUDY.parents[2]
sys.path.insert(0,str(ROOT))
from src.data.source_audit import save_json
from src.features.build_features import build_dataset


def main():
    p=pd.read_csv(STUDY/'predictions.csv.gz',parse_dates=['Date'])
    p=p[p.cohort=='common'];rows=[]
    for model,frame in p.groupby('model'):
        t,q=frame.reference_log_return.to_numpy(),frame.predicted_log_return.to_numpy();nonzero=t!=0
        rows.append({'model':model,'n':len(t),'actual_mean_log_return':float(t.mean()),'predicted_mean_log_return':float(q.mean()),
            'predicted_up_pct_all_origins':float(100*np.mean(q>0)),'actual_up_pct_all_origins':float(100*np.mean(t>0)),
            'actual_up_pct_nonzero':float(100*np.mean(t[nonzero]>0)),
            'up_recall_pct':float(100*np.mean(q[t>0]>0)), 'down_recall_pct':float(100*np.mean(q[t<0]<0)),
            'true_up_pred_up':int(((t>0)&(q>0)).sum()),'true_down_pred_down':int(((t<0)&(q<0)).sum())})
    pd.DataFrame(rows).to_csv(STUDY/'bias_diagnostics.csv',index=False,lineterminator='\n')
    protocol=json.loads((STUDY/'protocol.json').read_text());spec=next(s for s in protocol['models'] if s['name']=='Ridge_research_new')
    model=joblib.load(ROOT/spec['file'])
    with zipfile.ZipFile(STUDY/'data_snapshot.zip') as archive:raw=pd.read_csv(io.BytesIO(archive.read('vci.csv')),parse_dates=['Date'])
    features=build_dataset(raw)[0].set_index('Date');forecast=p[p.model=='Ridge_research_new']
    inputs=features.loc[forecast.Date,spec['columns']]
    z=model.steps[0][1].transform(inputs);ridge=model.steps[-1][1];contributions=z*ridge.coef_
    reconstructed=ridge.intercept_+contributions.sum(axis=1)
    assert np.allclose(reconstructed,forecast.predicted_log_return,atol=1e-12,rtol=1e-10)
    effects=[{'feature':name,'coefficient':float(ridge.coef_[i]),'mean_contribution_log_return':float(contributions[:,i].mean()),
              'mean_absolute_contribution_log_return':float(np.abs(contributions[:,i]).mean())} for i,name in enumerate(spec['columns'])]
    pd.DataFrame(effects).to_csv(STUDY/'ridge_feature_contributions.csv',index=False,lineterminator='\n')
    save_json(STUDY/'bias_verification.json',{'no_refit_or_selection':True,'ridge_prediction_equals_intercept_plus_contributions':True,
        'intercept':float(ridge.intercept_),'mean_feature_contribution':float(contributions.sum(axis=1).mean()),
        'common_origins':len(forecast),'interpretation':'algebraic contribution, not causal attribution'})
    print('Bias and contribution diagnostics saved without refitting.')


if __name__=='__main__':main()
