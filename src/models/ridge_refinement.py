"""Causal error diagnostics and preregistered Ridge alpha/window contrasts."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.data.contracts import BASE_DIR,file_sha256,utc_now
from src.data.source_audit import save_json,guarded_hashes
from src.models.source_comparison import load_inputs,source_part,REFERENCE
from src.models.robustness import fold_masks,score_direction
from src.models.window_comparison import window_masks,SOURCE
from src.features.build_features import TARGET_DATE_COLUMN


def fit_thresholds(train,config):
    spec=config['diagnostics']
    values=train[spec['volatility_feature']].to_numpy()
    if not len(values) or not np.isfinite(values).all():raise ValueError('Finite training volatility required.')
    lo,hi=np.quantile(values,spec['training_quantiles'])
    return {'vol_low':float(lo),'vol_high':float(hi),'sma_band':spec['near_sma_band'],'train_n':len(train)}


def assign_regimes(frame,thresholds):
    result=frame.copy();vol=result.Volatility_20
    result['vol_regime']=np.select([vol.le(thresholds['vol_low']),vol.le(thresholds['vol_high'])],['low','medium'],default='high')
    band=thresholds['sma_band'];distance=result.SMA_Distance_50
    result['trend_proxy']=np.select([distance.lt(-band),distance.gt(band)],['below_SMA50','above_SMA50'],default='near_SMA50')
    return result


def select_config(metrics,windows,alphas,window=None):
    candidates=metrics.loc[metrics.phase.eq('validation')].copy()
    if window is not None:candidates=candidates.loc[candidates.window.eq(window)]
    if candidates.empty:raise ValueError('No validation candidates.')
    candidates['control_priority']=candidates.alpha.ne(10.).astype(int)
    candidates['window_priority']=candidates.window.map({w:i for i,w in enumerate(windows)})
    candidates['alpha_priority']=candidates.alpha.map({a:i for i,a in enumerate(alphas)})
    return candidates.sort_values(['rmse','control_priority','window_priority','alpha_priority'],kind='stable').iloc[0]


def prediction_export(part,prediction,identity):
    prediction=np.asarray(prediction)
    if prediction.shape!=(len(part),):raise ValueError('One scalar forecast per dated origin is required.')
    output=part[['Date',TARGET_DATE_COLUMN,REFERENCE]].copy()
    output['predicted_log_return']=prediction
    for key,value in identity.items():output[key]=value
    return output


def inputs(config):
    base=BASE_DIR/'reports/source_comparison'/config['source_study']
    previous=BASE_DIR/'reports/window_comparison'/config['window_study']
    base_proto=json.loads((base/'protocol.json').read_text());cfg=base_proto['config']
    datasets,cohort=load_inputs(base)
    return base,previous,cfg,datasets[SOURCE][0],cohort


def diagnose(study,config,cfg,features,cohort,previous):
    pred=pd.read_csv(previous/'predictions.csv.gz',parse_dates=['Date','Target_Date_1d'])
    chosen=pd.read_csv(previous/'validation_selection.csv')
    thresholds={};exports=[]
    for fold in cfg['folds']:
        masks=fold_masks(cohort,fold)
        train=source_part(cohort.loc[masks['train']].reset_index(drop=True),SOURCE,features)
        thresholds[fold['id']]=fit_thresholds(train,config)
        for model in config['diagnostics']['models']:
            selected=chosen[(chosen.fold==fold['id'])&(chosen.model==model)].iloc[0].selected_window
            rows=pred[(pred.fold==fold['id'])&(pred.model==model)&(pred.window==selected)&(pred.phase=='evaluation')].copy()
            if rows.empty:raise ValueError('Missing previously selected predictions.')
            rows=rows.merge(features[['Date','Volatility_20','SMA_Distance_50']],on='Date',validate='many_to_one')
            rows=assign_regimes(rows,thresholds[fold['id']])
            rows['seed_id']=rows.seed.fillna(-1).astype(int)
            rows['squared_error']=(rows[REFERENCE]-rows.predicted_log_return)**2
            rows['zero_squared_error']=rows[REFERENCE]**2
            exports.append(rows)
    data=pd.concat(exports,ignore_index=True)
    data.to_csv(study/'diagnostic_predictions.csv.gz',index=False,compression='gzip',date_format='%Y-%m-%d',lineterminator='\n')
    save_json(study/'regime_thresholds.json',thresholds)
    year_rows=[];segments=[]
    for (fold,model,seed),part in data.groupby(['fold','model','seed_id']):
        year_rows.append({'fold':fold,'model':model,'seed_id':seed,**score_direction(part[REFERENCE],part.predicted_log_return)})
    pd.DataFrame(year_rows).to_csv(study/'error_by_fold_seed.csv',index=False,lineterminator='\n')
    for scope,folds in [('primary',config['primary_folds']),('stress',[config['stress_fold']])]:
        sample=data[data.fold.isin(folds)]
        for column in ['vol_regime','trend_proxy']:
            for (model,segment),part in sample.groupby(['model',column]):
                scores=[score_direction(p[REFERENCE],p.predicted_log_return) for _,p in part.groupby('seed_id')]
                # Per-origin seed-mean squared losses, so 3 seeds do not triple a day's weight.
                loss=part.groupby(['fold','Date']).squared_error.mean().sum()
                total=sample[sample.model==model].groupby(['fold','Date']).squared_error.mean().sum()
                n=part[['fold','Date']].drop_duplicates().shape[0]
                segments.append({'scope':scope,'segmentation':column,'model':model,'segment':segment,'unique_origins':n,
                    'seed_runs':len(scores),'mean_seed_rmse':float(np.mean([s['rmse'] for s in scores])),
                    'mean_seed_mae':float(np.mean([s['mae'] for s in scores])),
                    'mean_seed_mse_skill_vs_zero':float(np.mean([s['mse_skill_vs_zero'] for s in scores])),
                    'squared_loss_share_pct':float(100*loss/total),'small_segment':n<config['diagnostics']['minimum_segment_n_for_interpretation']})
    pd.DataFrame(segments).to_csv(study/'error_segments.csv',index=False,lineterminator='\n')
    rolling=data.groupby(['Date','model'],as_index=False).agg(mean_seed_squared_error=('squared_error','mean'),reference=(REFERENCE,'first'))
    rolling=rolling.sort_values(['model','Date'])
    rolling['rolling_60_rms_error']=rolling.groupby('model').mean_seed_squared_error.transform(lambda s:np.sqrt(s.rolling(60,min_periods=60).mean()))
    rolling.to_csv(study/'rolling_errors.csv',index=False,date_format='%Y-%m-%d',lineterminator='\n')
    from src.models.ridge_refinement_report import diagnostic_report
    diagnostic_report(study)


def run(config_path):
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    if config['automatic_promotion'] is not False or config['source']!=SOURCE or config['feature_group']!='returns':raise ValueError('Research only; source/features fixed.')
    if 10. not in config['alphas'] or any(a<=0 for a in config['alphas']):raise ValueError('Positive alphas and alpha10 control required.')
    base,previous,cfg,features,cohort=inputs(config)
    old_proto=json.loads((previous/'protocol.json').read_text());old_manifest=json.loads((previous/'manifest.json').read_text())
    versions={p:importlib.metadata.version(p) for p in old_proto['versions']}
    if versions!=old_proto['versions']:raise ValueError('Dependencies changed relative to reused controls.')
    assert json.loads((previous/'reload_verification.json').read_text())['unique_models']==84
    study=BASE_DIR/'reports/ridge_refinement'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ');study.mkdir(parents=True)
    artifacts=BASE_DIR/'models/research'/('ridge-refinement_'+study.name);artifacts.mkdir(parents=True)
    before=guarded_hashes()
    code=['src/models/ridge_refinement.py','src/models/ridge_refinement_report.py','src/models/window_comparison.py',
          'src/models/source_comparison.py','src/models/robustness.py','src/data/contracts.py']
    frozen={'schema':'fpt-ridge-refinement-v1','study_id':study.name,'registered_before_diagnostics_utc':utc_now(),
        'config':config,'config_sha256':file_sha256(config_path),'python':platform.python_version(),'versions':versions,
        'source_sha256':{name:file_sha256(BASE_DIR/name) for name in code},'protected_before':before,
        'source_study_protocol_sha256':file_sha256(base/'protocol.json'),'window_study_protocol_sha256':file_sha256(previous/'protocol.json'),
        'window_predictions_sha256':file_sha256(previous/'predictions.csv.gz'),'window_selection_sha256':file_sha256(previous/'validation_selection.csv'),
        'input_zip_sha256':file_sha256(base/'input_snapshot.zip'),'fixed_folds':cfg['folds'],'feature_columns':cfg['feature_groups']['returns']}
    save_json(study/'diagnostic_protocol.json',frozen)
    with zipfile.ZipFile(study/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in code:archive.write(BASE_DIR/name,name)
        archive.write(config_path,'config.json')
    save_json(study/'status.json',{'status':'diagnosing','new_fits':0})
    diagnose(study,config,cfg,features,cohort,previous)
    protocol={**frozen,'registered_before_fit_utc':utc_now(),'diagnostic_protocol_sha256':file_sha256(study/'diagnostic_protocol.json'),
        'diagnostic_report_sha256':file_sha256(study/'ERROR_ANALYSIS.md')}
    save_json(study/'protocol.json',protocol)
    print(f'Diagnosis complete and grid registered: {study}',flush=True)
    columns=cfg['feature_groups']['returns'];specs=[];records=[];exports=[];counts=[];new_fits=0
    controls={(c['fold'],c['window']):c for c in old_manifest['conditions'] if c['model']=='Ridge'}
    for fold in cfg['folds']:
        cache={}
        for window,years in config['windows'].items():
            masks=window_masks(cohort,fold,years)
            parts={phase:source_part(cohort.loc[mask].reset_index(drop=True),SOURCE,features) for phase,mask in masks.items()}
            train=parts['train'];y=train[REFERENCE].to_numpy();keymask=hashlib.sha256(masks['train'].tobytes()).hexdigest()
            counts.append({'fold':fold['id'],'window':window,'train_n':len(train),'train_first':str(train.Date.min().date()),'train_last':str(train.Date.max().date())})
            for alpha in config['alphas']:
                identity=f"{fold['id']}_{window}_a{alpha:g}";key=(keymask,alpha)
                if alpha==10.:
                    old=controls[(fold['id'],window)];path=BASE_DIR/old['model_path']
                    if file_sha256(path)!=old_manifest['artifact_sha256'][old['model_path']]:raise ValueError('Old Ridge bytes changed.')
                    model=joblib.load(path);reused=old['candidate'];fitted=False
                elif key in cache:
                    model,path,reused=cache[key];fitted=False
                else:
                    model=make_pipeline(StandardScaler(),Ridge(alpha=alpha));model.fit(train[columns],y)
                    path=artifacts/(identity+'.pkl');joblib.dump(model,path);new_fits+=1;fitted=True;reused=None
                specs.append({'candidate':identity,'fold':fold['id'],'window':window,'alpha':alpha,'alpha_per_training_n':alpha/len(train),
                    'train_n':len(train),'model_path':path.relative_to(BASE_DIR).as_posix(),'new_fit':fitted,'reused_from':reused})
                cache[key]=(model,path,identity)
                for phase in ['validation','evaluation']:
                    part=parts[phase];pred=model.predict(part[columns]);ident={'candidate':identity,'fold':fold['id'],'window':window,'alpha':alpha,'phase':phase}
                    records.append({**ident,**score_direction(part[REFERENCE],pred)})
                    exports.append(prediction_export(part,pred,ident))
                save_json(study/'status.json',{'status':'training','conditions':len(specs),'new_fits':new_fits})
        print(f"{fold['id']} complete: {len(specs)} conditions, {new_fits} new fits",flush=True)
    pd.DataFrame(records).to_csv(study/'metrics.csv',index=False,lineterminator='\n')
    pd.concat(exports,ignore_index=True).to_csv(study/'predictions.csv.gz',index=False,compression='gzip',date_format='%Y-%m-%d',lineterminator='\n')
    pd.DataFrame(counts).to_csv(study/'cohort_counts.csv',index=False,lineterminator='\n')
    assert guarded_hashes()==before
    paths={c['model_path'] for c in specs}
    save_json(study/'manifest.json',{'protocol_sha256':file_sha256(study/'protocol.json'),'conditions':specs,'new_fits':new_fits,
        'artifact_sha256':{p:file_sha256(BASE_DIR/p) for p in paths},'protected_after':guarded_hashes(),'automatic_promotion':False})
    save_json(study/'status.json',{'status':'complete','conditions':len(specs),'new_fits':new_fits,'finished_at_utc':utc_now()})
    from src.models.ridge_refinement_report import report
    report(study)
    print(f'Complete: {study}',flush=True)
    return study


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=BASE_DIR/'configs/ridge_refinement_v1.json')
    run(parser.parse_args().config)
