"""Temporal direction classifiers, probability calibration and validation-only abstention."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import io
import json
import sys
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from src.data.contracts import BASE_DIR,file_sha256,utc_now
from src.data.source_audit import save_json,guarded_hashes
from src.features.build_features import build_dataset,TARGET_COLUMN,TARGET_DATE_COLUMN
from src.models.recency_bias import training_mask


def phase_mask(labels,start,end):
    return labels.Date.ge(pd.Timestamp(start)) & labels[TARGET_DATE_COLUMN].le(pd.Timestamp(end))


def log_odds(probability):
    p=np.clip(np.asarray(probability,dtype=float),1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)


def decide(probability,margin=0.):
    p=np.asarray(probability,dtype=float)
    if not 0<=margin<.5 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probabilities/margin.')
    return np.where(p>=.5+margin,1,np.where(p<.5-margin,-1,0)).astype(int)


def direction_metrics(returns,directions):
    y=np.sign(np.asarray(returns,dtype=float));d=np.asarray(directions,dtype=int)
    if len(y)!=len(d) or not len(y) or not np.isin(d,[-1,0,1]).all():raise ValueError('Invalid direction cohort.')
    called=d!=0;nonzero=y!=0;correct=called&(d==y);calls=int(called.sum());nz=int((called&nonzero).sum())
    up=called&(y>0);down=called&(y<0)
    up_recall=float(100*np.mean(d[up]==1)) if up.any() else None
    down_recall=float(100*np.mean(d[down]==-1)) if down.any() else None
    return {'n':len(y),'calls_n':calls,'abstain_n':int((~called).sum()),'coverage_pct':100*calls/len(y),
        'actual_up_n':int((y>0).sum()),'actual_down_n':int((y<0).sum()),'actual_neutral_n':int((y==0).sum()),
        'nonzero_calls_n':nz,'neutral_calls_n':int((called&~nonzero).sum()),'correct_n':int(correct.sum()),
        'called_accuracy_all_pct':float(100*correct.sum()/calls) if calls else None,
        'hit_nonzero_called_pct':float(100*correct.sum()/nz) if nz else None,
        'balanced_called_nonzero_pct':(up_recall+down_recall)/2 if up_recall is not None and down_recall is not None else None,
        'up_recall_called_pct':up_recall,'down_recall_called_pct':down_recall,
        'up_coverage_pct':float(100*up.sum()/(y>0).sum()) if (y>0).any() else None,
        'down_coverage_pct':float(100*down.sum()/(y<0).sum()) if (y<0).any() else None,
        'correct_over_all_origins_pct':float(100*correct.mean())}


def probability_metrics(returns,probability):
    r=np.asarray(returns,dtype=float);p=np.asarray(probability,dtype=float);mask=r!=0
    y=(r[mask]>0).astype(int);p=p[mask]
    return {'probability_n':len(y),'brier_nonzero':float(np.mean((p-y)**2)),
            'logloss_nonzero':float(log_loss(y,p,labels=[0,1]))} if len(y) else {'probability_n':0,'brier_nonzero':None,'logloss_nonzero':None}


def select_margin(returns,probability,spec):
    rows=[]
    for margin in spec['margins']:
        d=decide(probability,margin);m=direction_metrics(returns,d);y=np.sign(np.asarray(returns))
        up=int(((d!=0)&(y>0)).sum());down=int(((d!=0)&(y<0)).sum())
        eligible=(m['coverage_pct']>=spec['minimum_coverage_pct'] and m['nonzero_calls_n']>=spec['minimum_nonzero_calls']
                  and min(up,down)>=spec['minimum_calls_per_actual_direction'] and m['balanced_called_nonzero_pct'] is not None)
        rows.append({'margin':margin,'eligible':bool(eligible),**m})
    candidates=[r for r in rows if r['eligible']]
    if candidates:
        chosen=sorted(candidates,key=lambda r:(-r['balanced_called_nonzero_pct'],-r['coverage_pct'],r['margin']))[0]
        reason='validation_balanced_accuracy_with_coverage_constraints'
    else:chosen=next(r for r in rows if r['margin']==0);reason='fallback_full_coverage_no_eligible_validation_margin'
    return float(chosen['margin']),reason,rows


def paired_bootstrap(returns,directions,majority,spec):
    y=np.sign(np.asarray(returns));d=np.asarray(directions);n=len(y);rng=np.random.default_rng(spec['seed'])
    differences=[]
    for _ in range(spec['resamples']):
        starts=rng.integers(0,n,size=int(np.ceil(n/spec['block_length'])))
        idx=((starts[:,None]+np.arange(spec['block_length']))%n).ravel()[:n];a=y[idx];b=d[idx];called=b!=0
        if called.any():differences.append(100*(np.mean(b[called]==a[called])-np.mean(majority==a[called])))
    lo,hi=np.quantile(differences,[.025,.975]) if differences else (None,None)
    return {'hit_delta_majority_ci_low_pp':None if lo is None else float(lo),
        'hit_delta_majority_ci_high_pp':None if hi is None else float(hi),'bootstrap_valid_resamples':len(differences)}


def load_snapshot(config):
    parent=BASE_DIR/'reports/evaluation_2026'/config['evaluation_snapshot']
    acquisition=json.loads((parent/'acquisition.json').read_text())
    archive=parent/'data_snapshot.zip'
    if file_sha256(archive)!=acquisition['zip_sha256']:raise ValueError('Frozen source ZIP changed.')
    with zipfile.ZipFile(archive) as z:
        content=z.read('vci.csv')
        if hashlib.sha256(content).hexdigest()!=acquisition['files']['vci.csv']:raise ValueError('Vietcap member changed.')
    raw=pd.read_csv(io.BytesIO(content),parse_dates=['Date'])
    return parent,raw,build_dataset(raw)[1]


def run(config_path):
    config_path=Path(config_path);cfg=json.loads(config_path.read_text(encoding='utf-8'))
    if cfg['automatic_promotion'] is not False or cfg['network_download'] is not False:raise ValueError('Research-only frozen-source study.')
    if cfg['forced_threshold']!=.5 or 0. not in cfg['selection']['margins']:raise ValueError('Full coverage 0.5 control required.')
    parent,raw,labels=load_snapshot(cfg);columns=cfg['features']
    study=BASE_DIR/'reports/direction_selective'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ');study.mkdir(parents=True)
    directory=BASE_DIR/'models/research'/('direction-selective_'+study.name);directory.mkdir(parents=True)
    code=['src/models/direction_selective.py','src/models/direction_selective_report.py','src/models/direction_selective_notebook.py',
          'src/models/recency_bias.py','src/features/build_features.py','src/data/contracts.py']
    folds=[]
    for year in cfg['evaluation_years']:
        previous=year-1
        folds.append({'id':f'eval_{year}','year':year,'train_end':f'{previous}-06-30',
            'calibration_start':f'{previous}-07-01','calibration_end':f'{previous}-09-30',
            'validation_start':f'{previous}-10-01','validation_end':f'{previous}-12-31',
            'evaluation_start':f'{year}-01-01','evaluation_end':str(min(pd.Timestamp(f'{year}-12-31'),raw.Date.max()).date())})
    protocol={'study_id':study.name,'registered_before_fit_utc':utc_now(),'config':cfg,'folds':folds,
        'config_sha256':file_sha256(config_path),'input_zip_sha256':file_sha256(parent/'data_snapshot.zip'),
        'source_sha256':{p:file_sha256(BASE_DIR/p) for p in code},'protected_before':guarded_hashes(),
        'python':sys.version,'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','xgboost','joblib']}}
    save_json(study/'protocol.json',protocol)
    with zipfile.ZipFile(study/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in code:z.write(BASE_DIR/p,p)
        z.write(config_path,'config.json')
    save_json(study/'status.json',{'status':'running','base_fits':0,'calibrator_fits':0})
    registry=[];exports=[];selection=[];curves=[];phase_info=[];scores=[];matched=[];base_fits=0;cal_fits=0

    def save_model(model,fold,name,part,role):
        path=directory/(fold+'_'+name+('.json' if isinstance(model,XGBClassifier) else '.pkl'))
        if isinstance(model,XGBClassifier):model.save_model(path)
        else:joblib.dump(model,path)
        registry.append({'fold':fold,'name':name,'role':role,'path':path.relative_to(BASE_DIR).as_posix(),'sha256':file_sha256(path),
            'train_n':len(part),'origin_start':str(part.Date.min().date()),'origin_end':str(part.Date.max().date()),
            'target_end':str(part[TARGET_DATE_COLUMN].max().date()),
            'dates':part[['Date',TARGET_DATE_COLUMN]].astype(str).to_dict('records')})
        return path

    for fold in folds:
        name=fold['id'];year=fold['year'];train_all=labels.loc[training_mask(labels,fold['train_end'],cfg['training_years'])].copy()
        train=train_all.loc[train_all[TARGET_COLUMN].ne(0)].copy()
        cal_all=labels.loc[phase_mask(labels,fold['calibration_start'],fold['calibration_end'])].copy()
        cal=cal_all.loc[cal_all[TARGET_COLUMN].ne(0)].copy()
        val=labels.loc[phase_mask(labels,fold['validation_start'],fold['validation_end'])].copy()
        evaluation=labels.loc[phase_mask(labels,fold['evaluation_start'],fold['evaluation_end'])].copy()
        recent=labels.loc[training_mask(labels,fold['validation_end'],cfg['training_years'])].copy()
        for phase,part in [('train_all',train_all),('train_direction',train),('calibration_all',cal_all),('calibration_direction',cal),('validation',val),('evaluation',evaluation),('ridge_recent_train',recent)]:
            if part.empty:raise ValueError('Empty temporal phase.')
            phase_info.append({'fold':name,'phase':phase,'n':len(part),'up_n':int(part[TARGET_COLUMN].gt(0).sum()),
                'down_n':int(part[TARGET_COLUMN].lt(0).sum()),'neutral_n':int(part[TARGET_COLUMN].eq(0).sum()),
                'origin_start':str(part.Date.min().date()),'origin_end':str(part.Date.max().date()),'target_end':str(part[TARGET_DATE_COLUMN].max().date())})
        assert train[TARGET_DATE_COLUMN].max()<cal.Date.min() and cal[TARGET_DATE_COLUMN].max()<val.Date.min()
        assert val[TARGET_DATE_COLUMN].max()<evaluation.Date.min() and recent[TARGET_DATE_COLUMN].max()<evaluation.Date.min()
        ytrain=train[TARGET_COLUMN].gt(0).astype(int);ycal=cal[TARGET_COLUMN].gt(0).astype(int)
        if ytrain.nunique()!=2 or ycal.nunique()!=2:raise ValueError('Both directions required in train/calibration.')
        majority=1 if ytrain.mean()>=.5 else -1
        cases={};probs={};raw_returns={}
        for model_name,part in [('Ridge_shared',train),('Ridge_recent',recent)]:
            ridge=make_pipeline(StandardScaler(),Ridge(alpha=cfg['ridge_alpha'])).fit(part[columns],part[TARGET_COLUMN]);base_fits+=1
            save_model(ridge,name,model_name,part,'base');q=np.asarray(ridge.predict(evaluation[columns]),dtype=float)
            cases[model_name]=np.sign(q).astype(int);raw_returns[model_name]=q
        cases['Majority_train']=np.full(len(evaluation),majority);probs['Majority_train']=np.full(len(evaluation),float(ytrain.mean()))
        cases['Always_up']=np.ones(len(evaluation),dtype=int);cases['Always_down']=-np.ones(len(evaluation),dtype=int)
        for model_name,model in [('Logistic',make_pipeline(StandardScaler(),LogisticRegression(**cfg['logistic']))),('XGBoost',XGBClassifier(**cfg['xgboost']))]:
            model.fit(train[columns],ytrain);base_fits+=1;save_model(model,name,model_name,train,'base')
            pcal=np.asarray(model.predict_proba(cal[columns])[:,1],dtype=float)
            calibrator=LogisticRegression(C=cfg['calibration']['C'],max_iter=2000,solver='lbfgs').fit(log_odds(pcal),ycal);cal_fits+=1
            save_model(calibrator,name,model_name+'_sigmoid',cal,'calibrator')
            pval_raw=np.asarray(model.predict_proba(val[columns])[:,1],dtype=float)
            pval=calibrator.predict_proba(log_odds(pval_raw))[:,1]
            margin,reason,curve=select_margin(val[TARGET_COLUMN],pval,cfg['selection'])
            selection.append({'fold':name,'model':model_name,'margin':margin,'lower_threshold':.5-margin,'upper_threshold':.5+margin,
                'reason':reason,'validation_n':len(val),'calibration_target_end':str(cal[TARGET_DATE_COLUMN].max().date()),
                'validation_target_end':str(val[TARGET_DATE_COLUMN].max().date())})
            curves.extend({'fold':name,'model':model_name,**r} for r in curve)
            raw_p=np.asarray(model.predict_proba(evaluation[columns])[:,1],dtype=float);p=calibrator.predict_proba(log_odds(raw_p))[:,1]
            for suffix,probability,m in [('raw',raw_p,0.),('calibrated',p,0.),('selective',p,margin)]:
                strategy=model_name+'_'+suffix;cases[strategy]=decide(probability,m);probs[strategy]=probability
            # Save calibration/validation predictions so later audit does not rely on labels of evaluation.
            for phase,part,raw_prob in [('calibration',cal,pcal),('validation',val,pval_raw)]:
                frame=part[['Date',TARGET_DATE_COLUMN,TARGET_COLUMN]].copy().rename(columns={TARGET_COLUMN:'actual_log_return'})
                frame['fold']=name;frame['phase']=phase;frame['strategy']=model_name
                frame['raw_probability']=raw_prob;frame['probability_up']=calibrator.predict_proba(log_odds(raw_prob))[:,1]
                exports.append(frame)
        for strategy,d in cases.items():
            truth=evaluation[TARGET_COLUMN].to_numpy();probability=probs.get(strategy)
            row={'fold':name,'year':year,'strategy':strategy,**direction_metrics(truth,d),**paired_bootstrap(truth,d,majority,cfg['bootstrap'])}
            row.update(probability_metrics(truth,probability) if probability is not None else {'probability_n':0,'brier_nonzero':None,'logloss_nonzero':None})
            scores.append(row)
            frame=evaluation[['Date',TARGET_DATE_COLUMN,TARGET_COLUMN]].copy().rename(columns={TARGET_COLUMN:'actual_log_return'})
            frame['fold']=name;frame['phase']='evaluation';frame['strategy']=strategy;frame['direction']=d
            frame['status']=np.where(d>0,'tang',np.where(d<0,'giam','chua_du_tin_hieu'))
            frame['probability_up']=probability if probability is not None else np.nan
            frame['predicted_log_return']=raw_returns.get(strategy,np.full(len(evaluation),np.nan))
            exports.append(frame)
            if strategy.endswith('_selective'):
                called=d!=0
                if called.any():
                    for control in ['Ridge_shared','Ridge_recent','Majority_train','Always_up','Always_down']:
                        matched.append({'fold':name,'selection_strategy':strategy,'control':control,
                            **direction_metrics(truth[called],cases[control][called])})
                    matched.append({'fold':name,'selection_strategy':strategy,'control':strategy,**direction_metrics(truth[called],d[called])})
        pd.concat(exports,ignore_index=True).to_csv(study/'predictions.csv.gz',index=False,compression='gzip',date_format='%Y-%m-%d')
        save_json(study/'status.json',{'status':'running','base_fits':base_fits,'calibrator_fits':cal_fits,'finished_fold':name})
        print(f'{name}: train={len(train)} cal={len(cal)} val={len(val)} eval={len(evaluation)}; base fits={base_fits}, calibrators={cal_fits}',flush=True)
    pd.DataFrame(scores).to_csv(study/'metrics.csv',index=False)
    pd.DataFrame(selection).to_csv(study/'threshold_selection.csv',index=False)
    pd.DataFrame(curves).to_csv(study/'validation_thresholds.csv',index=False)
    pd.DataFrame(phase_info).to_csv(study/'phase_counts.csv',index=False)
    pd.DataFrame(matched).to_csv(study/'matched_subset_controls.csv',index=False)
    assert guarded_hashes()==protocol['protected_before']
    save_json(study/'manifest.json',{'base_fits':base_fits,'calibrator_fits':cal_fits,'artifacts':registry,'protected_after':guarded_hashes()})
    save_json(study/'status.json',{'status':'trained_pending_verification','base_fits':base_fits,'calibrator_fits':cal_fits})
    from src.models.direction_selective_report import report
    report(study)
    return study


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=BASE_DIR/'configs/direction_selective_v1.json')
    print(run(parser.parse_args().config),flush=True)
