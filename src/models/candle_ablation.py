"""Paired FPT feature experiment; reuse A controls, fit only gap/body B models."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import sys
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.data.contracts import BASE_DIR,file_sha256,utc_now
from src.data.source_audit import save_json,guarded_hashes
from src.features.build_features import TARGET_COLUMN,TARGET_DATE_COLUMN
from src.features.candle_features import build_candle_dataset,RETURNS6,GAP_BODY7
from src.models.recency_bias import training_mask
from src.models.direction_selective import load_snapshot,phase_mask,decide,log_odds,select_margin,direction_metrics,probability_metrics
from src.models.evaluation_2026 import price_metrics


def choose_case(rows,model):
    if model not in ['Ridge_shared','Logistic_raw']:raise ValueError('Only untouched validation predictions may select the feature case.')
    key='return_rmse' if model=='Ridge_shared' else 'balanced_called_nonzero_pct'
    clean=[r for r in rows if r['model']==model and r.get(key) is not None and np.isfinite(r[key])]
    if len(clean)!=2:raise ValueError('Both A/B validation scores are required.')
    return sorted(clean,key=lambda r:((r[key] if model=='Ridge_shared' else -r[key]),r['case']))[0]['case']


def combined_metrics(frame):
    y=frame.actual_log_return.to_numpy();d=frame.direction.to_numpy(dtype=int)
    result=direction_metrics(y,d)
    if frame.probability_up.notna().all():result.update(probability_metrics(y,frame.probability_up))
    if frame.predicted_log_return.notna().all():
        q=frame.predicted_log_return.to_numpy();zero=np.mean(y*y)
        result.update(return_rmse=float(np.sqrt(np.mean((q-y)**2))),return_mae=float(np.mean(abs(q-y))),
            mse_skill_vs_zero=float(1-np.mean((q-y)**2)/zero),**price_metrics(y,q,frame.Close))
    return result


def run(config_path):
    config_path=Path(config_path);cfg=json.loads(config_path.read_text(encoding='utf-8'))
    if cfg['automatic_promotion'] is not False or cfg['network_download'] is not False:raise ValueError('Frozen research scope required.')
    if cfg['feature_cases']!={'A_returns6':RETURNS6,'B_gap_body7':GAP_BODY7}:raise ValueError('Exact feature replacement required.')
    control=BASE_DIR/'reports/direction_selective'/cfg['control_study'];cp=json.loads((control/'protocol.json').read_text());dcfg=cp['config']
    cm=json.loads((control/'manifest.json').read_text());cv=json.loads((control/'verification.json').read_text())
    if cv['artifacts_reloaded']!=30:raise ValueError('Control is not fully verified.')
    versions={p:importlib.metadata.version(p) for p in cp['versions']}
    if versions!=cp['versions']:raise ValueError('Environment changed; do not silently mix controls.')
    parent,raw,original=load_snapshot(dcfg);features,labels,_,build_report=build_candle_dataset(raw)
    assert np.array_equal(labels.Date,original.Date)
    assert np.allclose(labels[RETURNS6],original[RETURNS6]) and np.allclose(labels[TARGET_COLUMN],original[TARGET_COLUMN])
    assert file_sha256(parent/'data_snapshot.zip')==cp['input_zip_sha256']
    study=BASE_DIR/'reports/candle_ablation'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ');study.mkdir(parents=True)
    artifacts=BASE_DIR/'models/research'/('candle-ablation_'+study.name);artifacts.mkdir(parents=True)
    code=['src/features/candle_features.py','src/features/build_features.py','src/models/candle_ablation.py',
          'src/models/candle_ablation_report.py','src/models/candle_ablation_notebook.py','src/models/direction_selective.py',
          'src/models/recency_bias.py','src/data/contracts.py']
    protocol={'study_id':study.name,'registered_before_fit_utc':utc_now(),'config':cfg,'direction_config':dcfg,'folds':cp['folds'],
        'python':sys.version,'versions':versions,'protected_before':guarded_hashes(),'input_zip_sha256':cp['input_zip_sha256'],
        'control_protocol_sha256':file_sha256(control/'protocol.json'),'control_manifest_sha256':file_sha256(control/'manifest.json'),
        'source_sha256':{p:file_sha256(BASE_DIR/p) for p in code},'config_sha256':file_sha256(config_path)}
    save_json(study/'protocol.json',protocol)
    with zipfile.ZipFile(study/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in code:z.write(BASE_DIR/p,p)
        z.write(config_path,'config.json')
    save_json(study/'build_audit.json',build_report)
    labels.to_csv(study/'feature_label_snapshot.csv.gz',index=False,compression='gzip',date_format='%Y-%m-%d')
    registry=[];exports=[];margins=[];curves=[];validation_scores=[];counts=[];fits=0
    control_artifacts={(r['fold'],r['name']):r for r in cm['artifacts']}
    save_json(study/'status.json',{'status':'running','new_fits':0})

    def save_artifact(model,fold,name,part):
        nonlocal fits
        path=artifacts/(fold+'_B_'+name+'.pkl');joblib.dump(model,path);fits+=1
        record={'fold':fold,'name':name,'case':'B_gap_body7','path':path.relative_to(BASE_DIR).as_posix(),'sha256':file_sha256(path),
            'reused':False,'train_n':len(part),'dates':part[['Date',TARGET_DATE_COLUMN]].astype(str).to_dict('records')}
        registry.append(record)
        return record

    for fold in cp['folds']:
        name=fold['id'];train=labels.loc[training_mask(labels,fold['train_end'],dcfg['training_years'])&labels[TARGET_COLUMN].ne(0)]
        recent=labels.loc[training_mask(labels,fold['validation_end'],dcfg['training_years'])]
        cal=labels.loc[phase_mask(labels,fold['calibration_start'],fold['calibration_end'])&labels[TARGET_COLUMN].ne(0)]
        val=labels.loc[phase_mask(labels,fold['validation_start'],fold['validation_end'])]
        evaluation=labels.loc[phase_mask(labels,fold['evaluation_start'],fold['evaluation_end'])]
        for phase,part in [('train_direction',train),('train_recent',recent),('calibration',cal),('validation',val),('evaluation',evaluation)]:
            counts.append({'fold':name,'phase':phase,'n':len(part),'origin_start':str(part.Date.min().date()),'target_end':str(part[TARGET_DATE_COLUMN].max().date())})
        majority=1 if train[TARGET_COLUMN].gt(0).mean()>=.5 else -1
        for case,columns in cfg['feature_cases'].items():
            models={}
            for model_name,part in [('Ridge_shared',train),('Ridge_recent',recent),('Logistic',train)]:
                if case=='A_returns6':
                    old=control_artifacts[(name,model_name)]
                    assert file_sha256(BASE_DIR/old['path'])==old['sha256']
                    assert old['dates']==part[['Date',TARGET_DATE_COLUMN]].astype(str).to_dict('records')
                    model=joblib.load(BASE_DIR/old['path']);registry.append({**old,'case':case,'reused':True})
                else:
                    if model_name=='Logistic':model=make_pipeline(StandardScaler(),LogisticRegression(**dcfg['logistic'])).fit(part[columns],part[TARGET_COLUMN].gt(0).astype(int))
                    else:model=make_pipeline(StandardScaler(),Ridge(alpha=dcfg['ridge_alpha'])).fit(part[columns],part[TARGET_COLUMN])
                    save_artifact(model,name,model_name,part)
                models[model_name]=model
            raw_cal=models['Logistic'].predict_proba(cal[columns])[:,1].astype(float)
            if case=='A_returns6':
                old=control_artifacts[(name,'Logistic_sigmoid')];assert file_sha256(BASE_DIR/old['path'])==old['sha256']
                assert old['dates']==cal[['Date',TARGET_DATE_COLUMN]].astype(str).to_dict('records')
                calibrator=joblib.load(BASE_DIR/old['path']);registry.append({**old,'case':case,'reused':True})
            else:
                calibrator=LogisticRegression(C=dcfg['calibration']['C'],max_iter=2000,solver='lbfgs').fit(log_odds(raw_cal),cal[TARGET_COLUMN].gt(0).astype(int))
                save_artifact(calibrator,name,'Logistic_sigmoid',cal)
            raw_val=models['Logistic'].predict_proba(val[columns])[:,1].astype(float)
            cal_val=calibrator.predict_proba(log_odds(raw_val))[:,1]
            margin,reason,curve=select_margin(val[TARGET_COLUMN],cal_val,dcfg['selection'])
            margins.append({'fold':name,'case':case,'margin':margin,'reason':reason})
            curves.extend({'fold':name,'case':case,**r} for r in curve)

            def append(part,phase,strategy,direction,probability=None,q=None):
                frame=part[['Date',TARGET_DATE_COLUMN,'Close',TARGET_COLUMN]].copy().rename(columns={TARGET_COLUMN:'actual_log_return'})
                frame['fold']=name;frame['year']=fold['year'];frame['case']=case;frame['phase']=phase;frame['strategy']=strategy
                frame['direction']=direction;frame['probability_up']=probability if probability is not None else np.nan
                frame['predicted_log_return']=q if q is not None else np.nan
                frame['majority_direction']=majority
                exports.append(frame)
                return frame

            for phase,part in [('validation',val),('evaluation',evaluation)]:
                for model_name in ['Ridge_shared','Ridge_recent']:
                    if phase=='validation' and model_name=='Ridge_recent':continue  # Never score this model on data it trained on.
                    q=models[model_name].predict(part[columns]).astype(float)
                    frame=append(part,phase,model_name,np.sign(q).astype(int),q=q)
                    if phase=='validation':validation_scores.append({'fold':name,'case':case,'model':model_name,**combined_metrics(frame)})
                raw_p=models['Logistic'].predict_proba(part[columns])[:,1].astype(float);cal_p=calibrator.predict_proba(log_odds(raw_p))[:,1]
                for strategy,p,m in [('Logistic_raw',raw_p,0.),('Logistic_calibrated',cal_p,0.),('Logistic_selective',cal_p,margin)]:
                    frame=append(part,phase,strategy,decide(p,m),probability=p)
                    if phase=='validation' and strategy=='Logistic_raw':validation_scores.append({'fold':name,'case':case,'model':strategy,**combined_metrics(frame)})
            frame=append(cal,'calibration','Logistic',decide(calibrator.predict_proba(log_odds(raw_cal))[:,1]),probability=calibrator.predict_proba(log_odds(raw_cal))[:,1])
            frame['raw_probability']=raw_cal
        print(f'{name}: same train={len(train)}, cal={len(cal)}, val={len(val)}, eval={len(evaluation)}; new B fits={fits}',flush=True)
        save_json(study/'status.json',{'status':'running','new_fits':fits,'finished_fold':name})
    pd.concat(exports,ignore_index=True).to_csv(study/'predictions.csv.gz',index=False,compression='gzip',date_format='%Y-%m-%d')
    pd.DataFrame(validation_scores).to_csv(study/'validation_scores.csv',index=False)
    selections=[]
    for fold in cp['folds']:
        for model in ['Ridge_shared','Logistic_raw']:
            rows=[r for r in validation_scores if r['fold']==fold['id']]
            selections.append({'fold':fold['id'],'model':model,'selected_case':choose_case(rows,model)})
    pd.DataFrame(selections).to_csv(study/'feature_selection.csv',index=False)
    pd.DataFrame(margins).to_csv(study/'threshold_selection.csv',index=False)
    pd.DataFrame(curves).to_csv(study/'validation_thresholds.csv',index=False)
    pd.DataFrame(counts).to_csv(study/'phase_counts.csv',index=False)
    assert guarded_hashes()==protocol['protected_before']
    save_json(study/'manifest.json',{'new_fits':fits,'reused_artifacts':sum(r['reused'] for r in registry),'artifacts':registry,
        'protected_after':guarded_hashes(),'feature_label_snapshot_sha256':file_sha256(study/'feature_label_snapshot.csv.gz')})
    save_json(study/'status.json',{'status':'trained_pending_verification','new_fits':fits})
    from src.models.candle_ablation_report import report
    report(study)
    return study


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=BASE_DIR/'configs/candle_ablation_v1.json')
    print(run(p.parse_args().config),flush=True)
