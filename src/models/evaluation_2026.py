"""Freeze artifacts before data acquisition; one-step 2026 retrospective evaluation."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import requests
from src.data.contracts import BASE_DIR,file_sha256,utc_now
from src.data.source_audit import guarded_hashes,save_json,parse_vci,parse_kbs,VCI_URL,KBS_URL,profile,compare_pair,compare_returns
from src.data.download_dataset import download_stock_data
from src.features.build_features import build_dataset,FEATURE_COLUMNS,TARGET_DATE_COLUMN,TARGET_COLUMN
from src.models.source_comparison import load_inputs,REFERENCE
from src.models.robustness import fold_masks,score_direction
from src.models.window_comparison import window_masks
from src.models.experiment import make_sequences


def aligned_cohort(datasets,start,end):
    cohort=None
    for source,(features,labels) in datasets.items():
        if labels.Date.duplicated().any() or features.Date.duplicated().any():raise ValueError('Unique dates required.')
        p=pd.Index(features.Date).get_indexer(labels.Date)
        native=labels.loc[p>=59,['Date',TARGET_DATE_COLUMN,TARGET_COLUMN,'Close']].copy()
        native[f'position_{source}']=p[p>=59]
        native=native.rename(columns={TARGET_COLUMN:f'return_{source}','Close':f'close_{source}'})
        cohort=native if cohort is None else cohort.merge(native,on=['Date',TARGET_DATE_COLUMN],validate='one_to_one')
    return cohort.loc[cohort.Date.ge(start)&cohort[TARGET_DATE_COLUMN].lt(end)].sort_values('Date').reset_index(drop=True)


def price_metrics(truth,prediction,origin_close):
    truth,prediction,origin_close=map(lambda v:np.asarray(v,dtype=float),(truth,prediction,origin_close))
    actual=origin_close*np.exp(truth);forecast=origin_close*np.exp(prediction)
    return {'price_rmse_adjusted_vnd':float(np.sqrt(np.mean((forecast-actual)**2))),
            'price_mae_adjusted_vnd':float(np.mean(abs(forecast-actual))),
            'price_mape_pct':float(100*np.mean(abs(forecast/actual-1))),
            'mean_bias_log_return':float(np.mean(prediction-truth)),'prediction_std':float(np.std(prediction))}


def block_bootstrap(truth,prediction,config):
    truth,prediction=np.asarray(truth),np.asarray(prediction);n=len(truth)
    if n<2:return {'mse_skill_ci_low':None,'mse_skill_ci_high':None}
    rng=np.random.default_rng(config['seed']);length=config['block_length']
    starts=rng.integers(0,n,size=(config['resamples'],int(np.ceil(n/length))))
    indices=((starts[:,:,None]+np.arange(length))%n).reshape(config['resamples'],-1)[:,:n]
    mse=np.mean((prediction[indices]-truth[indices])**2,axis=1);zero=np.mean(truth[indices]**2,axis=1)
    skill=1-mse/zero
    return {'mse_skill_ci_low':float(np.quantile(skill,.025)),'mse_skill_ci_high':float(np.quantile(skill,.975))}


def freeze(config_path):
    config=json.loads(Path(config_path).read_text())
    if config['refit_or_select_on_2026'] is not False or config['automatic_promotion'] is not False:raise ValueError('Frozen evaluation only.')
    if config['end_exclusive']>config['client_as_of_date']:raise ValueError('Do not request incomplete/future days.')
    serving=BASE_DIR/'models/experiments'/config['serving_run'];manifest=json.loads((serving/'manifest.json').read_text())
    research=BASE_DIR/'reports/ridge_refinement'/config['research_study'];research_manifest=json.loads((research/'manifest.json').read_text())
    selection=pd.read_csv(research/'validation_selection.csv')
    selected=selection[(selection.fold==config['research_selection_fold'])&(selection.strategy=='Ridge_joint')].iloc[0]
    if selected.window!=config['research_window'] or selected.alpha!=config['research_alpha']:raise ValueError('Research choice is not the existing validation selection.')
    models=[]
    for name in ['LSTM','XGBoost','Ridge']:
        spec=manifest['models'][name]
        models.append({'name':name+'_serving','source':'yahoo','kind':spec['kind'],'file':(serving/spec['file']).relative_to(BASE_DIR).as_posix(),
            'scalers':(serving/spec['scalers']).relative_to(BASE_DIR).as_posix() if spec.get('scalers') else None,
            'lookback':spec.get('lookback'),'columns':FEATURE_COLUMNS,'train_end':'2022-12-31','validation_end':'2024-12-31'})
    for alpha,name in [(config['research_alpha'],'Ridge_research_new'),(config['research_control_alpha'],'Ridge_research_old')]:
        spec=next(c for c in research_manifest['conditions'] if c['fold']==config['research_selection_fold'] and c['window']==config['research_window'] and c['alpha']==alpha)
        models.append({'name':name,'source':'vci','kind':'sklearn','file':spec['model_path'],'scalers':None,
            'columns':json.loads((research/'protocol.json').read_text())['feature_columns'],
            'alpha':alpha,'window':config['research_window'],'train_end':'2023-12-31','validation_end':'2024-12-31'})
    old_source=BASE_DIR/'reports/source_comparison/20261029T142732805710Z';p=json.loads((old_source/'protocol.json').read_text());_,c=load_inputs(old_source)
    mean=float(c.loc[fold_masks(c,p['config']['folds'][-1])['train'],REFERENCE].mean())
    window_mean=float(c.loc[window_masks(c,p['config']['folds'][-1],int(config['research_window'][:-1]))['train'],REFERENCE].mean())
    for name,value,train_end in [('Zero-Return',0.,None),('TrainMean_serving',manifest['models']['Train Mean']['value'],'2022-12-31'),('TrainMean_research',mean,'2023-12-31'),('TrainMean_same_window',window_mean,'2023-12-31')]:
        models.append({'name':name,'source':'vci','kind':'constant','value':value,'train_end':train_end,'validation_end':None})
    files={s[k] for s in models for k in ['file','scalers'] if s.get(k)}
    # Verify identities against the already recorded training manifests, not just current bytes.
    for spec in models:
        if spec['name'].endswith('_serving') and spec.get('file'):
            for key in ['file','scalers']:
                if spec.get(key):assert file_sha256(BASE_DIR/spec[key])==manifest['artifact_sha256'][Path(spec[key]).name]
        elif spec.get('file'):assert file_sha256(BASE_DIR/spec['file'])==research_manifest['artifact_sha256'][spec['file']]
    study=BASE_DIR/'reports/evaluation_2026'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ');study.mkdir(parents=True)
    protocol={'schema':'fpt-frozen-2026-v1','study_id':study.name,'frozen_before_2026_download_utc':utc_now(),'config':config,'config_sha256':file_sha256(config_path),
        'models':models,'model_sha256':{f:file_sha256(BASE_DIR/f) for f in files},'protected_before':guarded_hashes(),
        'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','tensorflow','xgboost','yfinance']},
        'source_sha256':{n:file_sha256(BASE_DIR/n) for n in ['src/models/evaluation_2026.py','src/models/evaluation_2026_report.py',
            'src/features/build_features.py','src/data/contracts.py','src/data/source_audit.py']}}
    save_json(study/'protocol.json',protocol)
    with zipfile.ZipFile(study/'model_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for f in files:archive.write(BASE_DIR/f,f)
    with zipfile.ZipFile(study/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for f in protocol['source_sha256']:archive.write(BASE_DIR/f,f)
        archive.write(config_path,'config.json')
    save_json(study/'status.json',{'status':'frozen_before_download'})
    print('Frozen before inspecting 2026:',study,flush=True)
    return study


def acquire(study,protocol):
    config=protocol['config'];start,end=config['history_start'],config['end_exclusive']
    work=BASE_DIR/'.cache/evaluation-2026'/study.name;work.mkdir(parents=True,exist_ok=False)
    index=BASE_DIR/'reports/holdout_exposures.json';entries=json.loads(index.read_text()) if index.exists() else []
    exposure={'study_id':study.name,'opened_at_utc':utc_now(),'origin_start':config['origin_start'],'end_exclusive':end,
              'role':'retrospective 2026 range requested: consider inspected even if acquisition later fails','no_refit':True}
    entries.append(exposure);save_json(index,entries)
    # All providers refresh the entire context window in one adjustment basis; do not append mismatched price levels.
    ypath=download_stock_data(start_date=start,end_date=end,output_dir=work/'yahoo')
    y=pd.read_csv(ypath,parse_dates=['Date'])
    request={'timeFrame':'ONE_DAY','symbols':['FPT'],'to':int(pd.Timestamp(end,tz='UTC').timestamp()),'countBack':len(pd.bdate_range(start,end))+15}
    headers={'User-Agent':'FPT-University-Frozen-Evaluation/1.0','Accept':'application/json'}
    response=requests.post(VCI_URL,json=request,headers=headers,timeout=40);response.raise_for_status()
    (work/'vci_response.json').write_bytes(response.content);v=parse_vci(response.json(),start,end)
    params={'sdate':pd.Timestamp(start).strftime('%d-%m-%Y'),'edate':(pd.Timestamp(end)-pd.Timedelta(days=1)).strftime('%d-%m-%Y')}
    second=requests.get(KBS_URL,params=params,headers=headers,timeout=40);second.raise_for_status()
    (work/'kbs_response.json').write_bytes(second.content);k=parse_kbs(second.json(),start,end)
    if v.Date.max()<pd.Timestamp(end)-pd.Timedelta(days=7) or k.Date.max()<pd.Timestamp(end)-pd.Timedelta(days=7):raise ValueError('Stale reference data.')
    v.to_csv(work/'vci.csv',index=False,date_format='%Y-%m-%d');k.to_csv(work/'kbs.csv',index=False,date_format='%Y-%m-%d')
    with zipfile.ZipFile(study/'data_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in work.rglob('*'):
            if path.is_file():archive.write(path,path.relative_to(work).as_posix())
    files={path.relative_to(work).as_posix():file_sha256(path) for path in work.rglob('*') if path.is_file()}
    save_json(study/'acquisition.json',{'retrieved_at_utc':utc_now(),'vci_url':VCI_URL,'vci_request':request,'kbs_url':KBS_URL,'kbs_params':params,
        'price_currency':'VND','price_basis':'provider_adjusted_as_of_retrieval_not_point_in_time_certified','files':files,'zip_sha256':file_sha256(study/'data_snapshot.zip')})
    # Make the exposure durable: later tuning may not call this range untouched again.
    exposure['role']='retrospective 2026 evaluation now inspected';exposure['acquisition_complete']=True
    save_json(index,entries)
    return {'yahoo':y,'vci':v,'kbs':k}


def run(study):
    protocol=json.loads((study/'protocol.json').read_text());config=protocol['config']
    for f,sha in protocol['source_sha256'].items():assert file_sha256(BASE_DIR/f)==sha
    raw=acquire(study,protocol);datasets={};quality={}
    for source,frame in raw.items():
        features,labels,excluded,info=build_dataset(frame);datasets[source]=(features,labels)
        quality[source]={'profile':profile(frame[frame.Date.ge(config['origin_start'])]),'build':info}
        excluded.to_csv(study/(source+'_excluded.csv'),index=False,date_format='%Y-%m-%d',lineterminator='\n')
    for a,b in [('yahoo','vci'),('vci','kbs')]:
        left,right=[raw[s].loc[raw[s].Date.ge(config['origin_start'])] for s in [a,b]]
        difference,stats=compare_pair(left,right);returns,rstats=compare_returns(left,right)
        difference.to_csv(study/f'{a}_{b}_source_comparison.csv.gz',index=False,compression='gzip',date_format='%Y-%m-%d',lineterminator='\n')
        returns.to_csv(study/f'{a}_{b}_return_comparison.csv',index=False,date_format='%Y-%m-%d',lineterminator='\n')
        quality[a+'_'+b]={'levels':stats,'returns':rstats}
    cohorts={'common':aligned_cohort({s:datasets[s] for s in ['yahoo','vci']},config['origin_start'],config['end_exclusive']),
             'vci_full':aligned_cohort({'vci':datasets['vci']},config['origin_start'],config['end_exclusive'])}
    if len(cohorts['common'])<100:raise ValueError('Too little common 2026 coverage; inspect sources before scoring.')
    quality['coverage']={name:len(c) for name,c in cohorts.items()};save_json(study/'data_quality.json',quality)
    for name,cohort in cohorts.items():cohort.to_csv(study/(name+'_cohort.csv'),index=False,date_format='%Y-%m-%d',lineterminator='\n')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    import tensorflow as tf
    from xgboost import XGBRegressor
    tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
    rows=[];exports=[]
    for spec in protocol['models']:
        model=None;scalers=None
        if spec.get('file'):
            assert file_sha256(BASE_DIR/spec['file'])==protocol['model_sha256'][spec['file']]
            if spec['kind']=='lstm':model=tf.keras.models.load_model(BASE_DIR/spec['file'],compile=False);scalers=joblib.load(BASE_DIR/spec['scalers'])
            elif spec['kind']=='xgboost':model=XGBRegressor();model.load_model(BASE_DIR/spec['file'])
            else:model=joblib.load(BASE_DIR/spec['file'])
        for cohort_name,cohort in cohorts.items():
            if cohort_name=='vci_full' and spec['source']=='yahoo':continue
            source=spec['source'];features=datasets[source][0];positions=cohort['position_'+source].to_numpy(dtype=int)
            if spec['kind']=='constant':pred=np.full(len(cohort),spec['value'])
            elif spec['kind']=='lstm':
                values=scalers['features'].transform(features[spec['columns']])
                pred=scalers['target'].inverse_transform(model(make_sequences(values,positions,spec['lookback']),training=False).numpy()).ravel()
            else:pred=model.predict(features.iloc[positions][spec['columns']])
            pred=np.asarray(pred,dtype=float)  # Keep exp/export/metric arithmetic on the same float64 basis.
            truth=cohort.return_vci.to_numpy();close=cohort.close_vci.to_numpy()
            rows.append({'model':spec['name'],'cohort':cohort_name,**score_direction(truth,pred),**price_metrics(truth,pred,close),**block_bootstrap(truth,pred,config['bootstrap'])})
            frame=cohort[['Date',TARGET_DATE_COLUMN]].copy();frame['reference_log_return']=truth;frame['reference_origin_close']=close
            frame['actual_close_adjusted']=close*np.exp(truth);frame['predicted_log_return']=pred
            frame['predicted_close_adjusted']=close*np.exp(pred);frame['native_origin_close']=cohort['close_'+source].to_numpy()
            frame['native_predicted_close_adjusted']=frame.native_origin_close*np.exp(pred)
            frame['model']=spec['name'];frame['cohort']=cohort_name;exports.append(frame)
    pd.DataFrame(rows).to_csv(study/'metrics.csv',index=False,lineterminator='\n')
    pd.concat(exports,ignore_index=True).to_csv(study/'predictions.csv.gz',index=False,compression='gzip',date_format='%Y-%m-%d',lineterminator='\n')
    assert guarded_hashes()==protocol['protected_before']
    save_json(study/'status.json',{'status':'complete','evaluated_at_utc':utc_now(),'refits':0,'protected_files_unchanged':True})
    from src.models.evaluation_2026_report import report
    report(study)
    print('Evaluation complete:',study,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=BASE_DIR/'configs/evaluation_2026_v1.json');args=parser.parse_args()
    run(freeze(args.config))
