"""Post-fit descriptive comparisons only. Does not change selection or models."""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

STUDY=Path(__file__).resolve().parent
ROOT=STUDY.parents[2]
sys.path.insert(0,str(ROOT))
from src.models.ridge_refinement import inputs,assign_regimes
from src.data.source_audit import save_json


def main():
    proto=json.loads((STUDY/'protocol.json').read_text());config=proto['config']
    _,_,_,features,_=inputs(config)
    predictions=pd.read_csv(STUDY/'strategy_predictions.csv.gz',parse_dates=['Date'])
    thresholds=json.loads((STUDY/'regime_thresholds.json').read_text())
    parts=[]
    for fold in proto['fixed_folds']:
        part=predictions[(predictions.fold==fold['id'])&(predictions.phase=='evaluation')].merge(
            features[['Date','Volatility_20','SMA_Distance_50']],on='Date',validate='many_to_one')
        part=assign_regimes(part,thresholds[fold['id']]);part['scope']='primary' if fold['id'] in config['primary_folds'] else 'stress'
        parts.append(part)
    combined=pd.concat(parts,ignore_index=True);rows=[]
    for (scope,strategy,regime),part in combined.groupby(['scope','strategy','vol_regime']):
        truth=part.Reference_Return.to_numpy();pred=part.predicted_log_return.to_numpy()
        assert len(part)==part.Date.nunique()
        error=np.mean((truth-pred)**2);zero=np.mean(truth**2)
        rows.append({'scope':scope,'strategy':strategy,'vol_regime':regime,'n':len(part),'rmse':float(np.sqrt(error)),
                     'mae':float(np.mean(abs(truth-pred))),'mse_skill_vs_zero':float(1-error/zero)})
    pd.DataFrame(rows).to_csv(STUDY/'selected_regime_comparison.csv',index=False,lineterminator='\n')
    selection=pd.read_csv(STUDY/'validation_selection.csv')
    manifest=json.loads((STUDY/'manifest.json').read_text());effects=[]
    for _,chosen in selection[selection.strategy=='Ridge_joint'].iterrows():
        specs=[c for c in manifest['conditions'] if c['fold']==chosen.fold and c['window']==chosen.window]
        old=next(c for c in specs if c['alpha']==10.)
        new=next(c for c in specs if c['alpha']==chosen.alpha)
        a=joblib.load(ROOT/old['model_path']).steps[-1][1]
        b=joblib.load(ROOT/new['model_path']).steps[-1][1]
        oldpred=predictions[(predictions.fold==chosen.fold)&(predictions.phase=='evaluation')&(predictions.strategy=='Ridge_window_only')].predicted_log_return
        newpred=predictions[(predictions.fold==chosen.fold)&(predictions.phase=='evaluation')&(predictions.strategy=='Ridge_joint')].predicted_log_return
        effects.append({'fold':chosen.fold,'window':chosen.window,'old_alpha':10.,'selected_alpha':float(chosen.alpha),
            'old_coefficient_l2_norm':float(np.linalg.norm(a.coef_)),'new_coefficient_l2_norm':float(np.linalg.norm(b.coef_)),
            'old_forecast_std':float(oldpred.std(ddof=0)),'new_forecast_std':float(newpred.std(ddof=0)),
            'old_intercept':float(a.intercept_),'new_intercept':float(b.intercept_)})
    pd.DataFrame(effects).to_csv(STUDY/'regularization_effects.csv',index=False,lineterminator='\n')
    save_json(STUDY/'postfit_diagnostic_checks.json',{'no_training_or_selection_performed':True,
        'unique_origins_per_selected_strategy_verified':True,'segment_rows':len(rows),'coefficient_comparisons':len(effects)})
    print('Post-fit effects/regimes saved; no models or selections changed.')


if __name__=='__main__':main()
