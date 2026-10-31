"""Fixed-parameter recency contrasts and strictly matured-label bias calibration."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import zipfile
import sys
import importlib.metadata
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.data.contracts import BASE_DIR,file_sha256,utc_now
from src.data.source_audit import guarded_hashes,save_json
from src.features.build_features import build_dataset,TARGET_DATE_COLUMN,TARGET_COLUMN
from src.models.source_comparison import load_inputs,REFERENCE
from src.models.window_comparison import window_masks
from src.models.evaluation_2026 import aligned_cohort,price_metrics
from src.models.robustness import score_direction


def training_mask(labels,cutoff,years):
    cutoff=pd.Timestamp(cutoff);start=cutoff-pd.DateOffset(years=years)+pd.Timedelta(days=1)
    return labels.Date.ge(start)&labels[TARGET_DATE_COLUMN].le(cutoff)


def causal_calibration(frame,bias_spec,interval_spec):
    """Use only previous forecast labels with target <= the current origin (after close)."""
    frame=frame.sort_values('Date').reset_index(drop=True).copy()
    rows=[]
    for i,row in frame.iterrows():
        prior=[j for j in range(i) if frame.loc[j,TARGET_DATE_COLUMN]<=row.Date]
        history=prior[-bias_spec['window']:]
        residual=[float(frame.loc[j,REFERENCE]-frame.loc[j,'raw_prediction']) for j in history]
        offset=float(np.mean(residual)) if len(residual)>=bias_spec['minimum_matured'] else 0.
        prediction=float(row.raw_prediction+offset)
        interval_history=prior[-interval_spec['window']:]
        errors=[float(frame.loc[j,REFERENCE]-rows[j]['prediction']) for j in interval_history]
        if len(errors)>=interval_spec['minimum_matured']:
            lo,hi=np.quantile(errors,interval_spec['quantiles']);lower,upper=prediction+lo,prediction+hi
        else:lower=upper=np.nan
        rows.append({'prediction':prediction,'bias_offset':offset,'calibration_n':len(residual),
            'last_calibration_target':str(frame.loc[history[-1],TARGET_DATE_COLUMN].date()) if history else None,
            'lower_log_return':lower,'upper_log_return':upper,'interval_n':len(errors)})
    return pd.concat([frame,pd.DataFrame(rows)],axis=1)


def run(config_path):
    config=json.loads(Path(config_path).read_text())
    if config['selection_on_2026'] is not False or config['automatic_promotion'] is not False:raise ValueError('No selection/promotion in inspected 2026.')
    evaluation=BASE_DIR/'reports/evaluation_2026'/config['evaluation_study']
    evaluation_proto=json.loads((evaluation/'protocol.json').read_text())
    assert json.loads((evaluation/'verification.json').read_text())['refits']==0
    acquisition=json.loads((evaluation/'acquisition.json').read_text())
    assert file_sha256(evaluation/'data_snapshot.zip')==acquisition['zip_sha256']
    with zipfile.ZipFile(evaluation/'data_snapshot.zip') as archive:
        content=archive.read('vci.csv');assert hashlib.sha256(content).hexdigest()==acquisition['files']['vci.csv']
    raw=pd.read_csv(io.BytesIO(content),parse_dates=['Date'])
    features,labels,_,_=build_dataset(raw)
    start=evaluation_proto['config']['origin_start'];end=evaluation_proto['config']['end_exclusive']
    cohort=aligned_cohort({'vci':(features,labels)},start,end)
    old_spec=next(s for s in evaluation_proto['models'] if s['name']=='Ridge_research_new')
    columns=old_spec['columns'];assert old_spec['alpha']==config['alpha']
    assert file_sha256(BASE_DIR/old_spec['file'])==evaluation_proto['model_sha256'][old_spec['file']]
    study=BASE_DIR/'reports/recency_bias'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ');study.mkdir(parents=True)
    artifacts=BASE_DIR/'models/research'/('recency-bias_'+study.name);artifacts.mkdir(parents=True)
    code=['src/models/recency_bias.py','src/models/recency_bias_report.py', 'src/models/research_notes.py','src/models/forward_ledger.py','src/features/build_features.py','src/models/evaluation_2026.py']
    protocol={'schema':'fpt-recency-bias-v1','study_id':study.name,'registered_before_fit_utc':utc_now(),'config':config,
        'python':sys.version,'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','joblib']},
        'feature_columns':columns,'alpha':config['alpha'],'evaluation_data_sha256':file_sha256(evaluation/'data_snapshot.zip'),
        'evaluation_protocol_sha256':file_sha256(evaluation/'protocol.json'),'protected_before':guarded_hashes(),
        'source_sha256':{p:file_sha256(BASE_DIR/p) for p in code},'frozen_control':old_spec,'frozen_control_sha256':file_sha256(BASE_DIR/old_spec['file']),
        'evaluation_role':'already inspected 2026, retrospective time-order simulation, no fresh holdout claim'}
    save_json(study/'protocol.json',protocol)
    with zipfile.ZipFile(study/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in code:archive.write(BASE_DIR/p,p)
        archive.write(config_path,'config.json')
    save_json(study/'status.json',{'status':'running','new_fits':0})
    # The saved evaluation cohort supplies dates/labels; no future label is a feature.
    X=features.iloc[cohort.position_vci.to_numpy()][columns]
    registry=[];fit_cache={};fit_count=0

    def fit(name,part,cutoff):
        nonlocal fit_count
        if len(part)<100:raise ValueError('Too few training rows.')
        assert part[TARGET_DATE_COLUMN].le(pd.Timestamp(cutoff)).all()
        signature=hashlib.sha256(pd.util.hash_pandas_object(part[['Date',TARGET_DATE_COLUMN,*columns,TARGET_COLUMN]],index=False).values.tobytes()).hexdigest()
        if signature in fit_cache:return fit_cache[signature]
        model=make_pipeline(StandardScaler(),Ridge(alpha=config['alpha']));model.fit(part[columns],part[TARGET_COLUMN])
        path=artifacts/(name+'.pkl');joblib.dump(model,path);fit_count+=1
        registry.append({'id':name,'path':path.relative_to(BASE_DIR).as_posix(),'training_cutoff':str(pd.Timestamp(cutoff).date()),
            'train_origin_start':str(part.Date.min().date()),'train_origin_end':str(part.Date.max().date()),
            'train_target_end':str(part[TARGET_DATE_COLUMN].max().date()),'train_n':len(part),
            'training_dates':part[['Date',TARGET_DATE_COLUMN]].astype(str).to_dict('records'),'signature':signature})
        fit_cache[signature]=(model,path)
        return model,path

    frozen=joblib.load(BASE_DIR/old_spec['file']);predictions={'frozen_2023':frozen.predict(X)}
    source=BASE_DIR/'reports/source_comparison/20261029T142732805710Z';sp=json.loads((source/'protocol.json').read_text())
    _,old_cohort=load_inputs(source);old_fold=sp['config']['folds'][-1]
    dates=old_cohort.loc[window_masks(old_cohort,old_fold,3)['train'],['Date',TARGET_DATE_COLUMN]]
    same=labels.merge(dates,on=['Date',TARGET_DATE_COLUMN],validate='one_to_one');assert len(same)==693
    rebuilt,_=fit('rebuilt_2023_same_rows',same,config['old_train_end']);predictions['rebuilt_2023_same_rows']=rebuilt.predict(X)
    old_native=labels.loc[training_mask(labels,config['old_train_end'],config['training_window_years'])]
    model,_=fit('refit_2023_native',old_native,config['old_train_end']);predictions['refit_2023_native']=model.predict(X)
    train2025=labels.loc[training_mask(labels,config['static_refresh_end'],config['training_window_years'])]
    updated,_=fit('refit_2025_native',train2025,config['static_refresh_end']);predictions['refit_2025_native']=updated.predict(X)
    monthly=np.empty(len(cohort));assignments=[]
    for month,group in cohort.groupby(cohort.Date.dt.to_period('M'),sort=True):
        cutoff=month.start_time-pd.Timedelta(days=1)
        part=labels.loc[training_mask(labels,cutoff,config['training_window_years'])]
        model,path=fit('monthly_'+str(month),part,cutoff)
        assert part[TARGET_DATE_COLUMN].max()<group.Date.min()
        monthly[group.index]=model.predict(features.iloc[group.position_vci.to_numpy()][columns])
        for idx in group.index:assignments.append({'Date':cohort.loc[idx,'Date'],'Target_Date_1d':cohort.loc[idx,TARGET_DATE_COLUMN],
            'training_cutoff':str(cutoff.date()),'train_target_end':str(part[TARGET_DATE_COLUMN].max().date()),'model_path':path.relative_to(BASE_DIR).as_posix()})
    predictions['monthly_refit_native']=monthly
    calibration_frames={}
    for strategy in ['frozen_2023','refit_2025_native','monthly_refit_native']:
        frame=cohort[['Date',TARGET_DATE_COLUMN]].copy();frame[REFERENCE]=cohort.return_vci.to_numpy();frame['raw_prediction']=predictions[strategy]
        calibrated=causal_calibration(frame,config['bias'],config['interval'])
        predictions[strategy+'_bias60']=calibrated.prediction.to_numpy();calibration_frames[strategy+'_bias60']=calibrated
    predictions['Zero-Return']=np.zeros(len(cohort));predictions['TrainMean_2025']=np.full(len(cohort),train2025[TARGET_COLUMN].mean())
    # A causal rolling-return baseline may use pre-2026 observed returns from day one.
    rolling=[]
    for origin in cohort.Date:
        mature=labels.loc[labels[TARGET_DATE_COLUMN].le(origin)].tail(config['bias']['window'])
        rolling.append(float(mature[TARGET_COLUMN].mean()))
    predictions['RollingMean60']=np.asarray(rolling)
    common=pd.read_csv(evaluation/'common_cohort.csv',parse_dates=['Date',TARGET_DATE_COLUMN])
    common_keys=set(zip(common.Date,common[TARGET_DATE_COLUMN]))
    masks={'vci_full':np.ones(len(cohort),dtype=bool),'previous_common':np.array([(d,t) in common_keys for d,t in zip(cohort.Date,cohort[TARGET_DATE_COLUMN])])}
    rows=[];exports=[]
    for strategy,pred in predictions.items():
        detail=calibration_frames.get(strategy)
        for cohort_name,mask in masks.items():
            truth=cohort.return_vci.to_numpy()[mask];q=np.asarray(pred,dtype=float)[mask];close=cohort.close_vci.to_numpy()[mask]
            extra={'interval_n':0,'interval_coverage_pct':None,'mean_interval_width_log_return':None,'warmup_bias_n':0}
            if detail is not None:
                part=detail.loc[mask];valid=part.lower_log_return.notna().to_numpy()
                extra['warmup_bias_n']=int(part.calibration_n.lt(config['bias']['minimum_matured']).sum())
                if valid.any():
                    lo,hi=part.lower_log_return.to_numpy()[valid],part.upper_log_return.to_numpy()[valid]
                    extra.update(interval_n=int(valid.sum()),interval_coverage_pct=float(100*np.mean((truth[valid]>=lo)&(truth[valid]<=hi))),mean_interval_width_log_return=float(np.mean(hi-lo)))
            rows.append({'strategy':strategy,'cohort':cohort_name,**score_direction(truth,q),**price_metrics(truth,q,close),**extra})
            frame=cohort.loc[mask,['Date',TARGET_DATE_COLUMN]].copy();frame[REFERENCE]=truth;frame['predicted_log_return']=q
            frame['origin_close_adjusted']=close;frame['predicted_close_adjusted']=close*np.exp(q);frame['actual_close_adjusted']=close*np.exp(truth)
            frame['strategy']=strategy;frame['cohort']=cohort_name
            if detail is not None:
                for col in ['bias_offset','calibration_n','last_calibration_target','lower_log_return','upper_log_return','interval_n']:frame[col]=detail.loc[mask,col].to_numpy()
            exports.append(frame)
    pd.DataFrame(rows).to_csv(study/'metrics.csv',index=False,lineterminator='\n')
    pd.concat(exports,ignore_index=True).to_csv(study/'predictions.csv.gz',index=False,compression='gzip',date_format='%Y-%m-%d',lineterminator='\n')
    pd.DataFrame(assignments).to_csv(study/'monthly_model_assignments.csv',index=False,date_format='%Y-%m-%d',lineterminator='\n')
    for strategy,frame in calibration_frames.items():frame.to_csv(study/(strategy+'_calibration.csv'),index=False,date_format='%Y-%m-%d',lineterminator='\n')
    # A separate shadow snapshot as of the latest mature target, not a serving promotion.
    latest=raw.loc[raw.Volume.gt(0),'Date'].max()
    final_train=labels.loc[training_mask(labels,latest,config['training_window_years'])]
    final_model,final_path=fit('shadow_asof_'+str(latest.date()),final_train,latest)
    history=calibration_frames['monthly_refit_native_bias60']
    known=history.loc[history[TARGET_DATE_COLUMN].le(latest)].tail(config['bias']['window'])
    raw_errors=(known[REFERENCE]-known.raw_prediction).to_numpy();corrected_errors=(known[REFERENCE]-known.prediction).to_numpy()
    bundle={'schema':'fpt-shadow-bundle-v1','created_at_utc':utc_now(),'study_id':study.name,'model_path':final_path.relative_to(BASE_DIR).as_posix(),
        'model_sha256':file_sha256(final_path),'feature_columns':columns,'alpha':config['alpha'],'training_years':config['training_window_years'],
        'training_target_end':str(final_train[TARGET_DATE_COLUMN].max().date()),'data_completed_through':str(latest.date()),
        'calibration_targets':known[TARGET_DATE_COLUMN].dt.strftime('%Y-%m-%d').to_list(),
        'raw_errors':raw_errors.tolist(),'corrected_errors':corrected_errors.tolist(),'bias_spec':config['bias'],'interval_spec':config['interval'],
        'train_mean':float(final_train[TARGET_COLUMN].mean()),'automatic_promotion':False,
        'role':'fixed shadow snapshot; calibration frozen at bundle creation until explicit rebuild, not automatic monthly/daily retraining'}
    save_json(study/'shadow_bundle.json',bundle)
    assert guarded_hashes()==protocol['protected_before']
    paths={r['path'] for r in registry}
    save_json(study/'manifest.json',{'protocol_sha256':file_sha256(study/'protocol.json'),'new_fits':fit_count,'models':registry,
        'artifact_sha256':{p:file_sha256(BASE_DIR/p) for p in paths},'protected_after':guarded_hashes(),'automatic_promotion':False})
    save_json(study/'status.json',{'status':'trained_pending_verification','new_fits':fit_count})
    from src.models.recency_bias_report import report
    report(study)
    print('Complete:',study,flush=True)
    return study


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=BASE_DIR/'configs/recency_bias_v1.json');run(parser.parse_args().config)
