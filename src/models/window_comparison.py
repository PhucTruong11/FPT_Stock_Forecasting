"""Training-window research: fixed Vietcap inputs/cohorts, validation-only selection."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from src.data.contracts import BASE_DIR, file_sha256, utc_now
from src.data.source_audit import guarded_hashes, save_json
from src.models.source_comparison import REFERENCE, load_inputs, source_part, context_positions
from src.models.robustness import fold_masks, score_direction
from src.models.experiment import make_sequences
from src.features.build_features import TARGET_COLUMN, TARGET_DATE_COLUMN

SOURCE = 'vietcap_v3_candidate'


def window_masks(cohort, fold, years):
    masks = fold_masks(cohort, fold)
    if years is not None:
        if years < 1:
            raise ValueError('Training window must be positive.')
        start = pd.Timestamp(fold['train_end']) - pd.DateOffset(years=years) + pd.Timedelta(days=1)
        masks['train'] = masks['train'] & cohort.Date.ge(start).to_numpy()
    return masks


def select_windows(summary, order):
    rows = []
    for (fold, model), part in summary[summary.phase.eq('validation')].groupby(['fold','model']):
        part = part.assign(priority=part.window.map({w:i for i,w in enumerate(order)})).sort_values(['rmse_mean','priority'],kind='stable')
        chosen = part.iloc[0]
        evaluation = summary[(summary.fold==fold)&(summary.model==model)&(summary.window==chosen.window)&(summary.phase=='evaluation')].iloc[0]
        control = summary[(summary.fold==fold)&(summary.model==model)&(summary.window=='expanding')&(summary.phase=='evaluation')].iloc[0]
        rows.append({'fold':fold,'model':model,'selected_window':chosen.window,'validation_rmse':chosen.rmse_mean,
                     'evaluation_rmse':evaluation.rmse_mean,'expanding_evaluation_rmse':control.rmse_mean,
                     'evaluation_rmse_gain_pct':100*(1-evaluation.rmse_mean/control.rmse_mean)})
    return pd.DataFrame(rows)


def fit_one(model_name, seed, cfg, features, train, val, columns, directory, identity):
    y = train[TARGET_COLUMN].to_numpy(); spec = cfg['models'][model_name]
    scaler_file = None
    if model_name=='Ridge':
        model = make_pipeline(StandardScaler(),Ridge(**spec)); model.fit(train[columns],y)
        path = directory/(identity+'.pkl'); joblib.dump(model,path); scalers = None; values=None; history=None
    elif model_name=='XGBoost':
        model=XGBRegressor(**spec,random_state=seed); model.fit(train[columns],y)
        path=directory/(identity+'.json'); model.save_model(path); scalers=None; values=None; history=None
    else:
        import tensorflow as tf
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
        context=context_positions(train.Feature_Position.to_numpy(),cfg['lookback'])
        sx=StandardScaler().fit(features.iloc[context][columns]); sy=StandardScaler().fit(y.reshape(-1,1))
        scalers={'features':sx,'target':sy,'context_positions':context}; values=sx.transform(features[columns])
        model=tf.keras.Sequential([tf.keras.layers.Input((cfg['lookback'],len(columns))),tf.keras.layers.LSTM(spec['units']),
            tf.keras.layers.Dropout(spec['dropout']),tf.keras.layers.Dense(spec['dense_units'],activation='relu'),tf.keras.layers.Dense(1)])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=spec['learning_rate'],clipnorm=spec['clipnorm']),loss=spec['loss'])
        history=model.fit(make_sequences(values,train.Feature_Position.to_numpy(),cfg['lookback']),sy.transform(y.reshape(-1,1)),
            validation_data=(make_sequences(values,val.Feature_Position.to_numpy(),cfg['lookback']),sy.transform(val[REFERENCE].to_numpy().reshape(-1,1))),
            epochs=spec['epochs'],batch_size=spec['batch_size'],shuffle=False,verbose=0,callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=spec['early_stopping_patience'],restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=spec['reduce_lr_patience'],factor=.5,min_lr=1e-5)])
        history={k:[float(v) for v in vals] for k,vals in history.history.items()}
        path=directory/(identity+'.keras'); model.save(path)
        scaler_file=directory/(identity+'_scalers.pkl'); joblib.dump(scalers,scaler_file)
    return model,path,scaler_file,scalers,values,history


def run(config_path):
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    if config['automatic_promotion'] is not False or config['windows'].get('expanding','missing') is not None:
        raise ValueError('Research only, with expanding control required.')
    if config['reference_source']!=SOURCE or config['feature_group']!='returns':
        raise ValueError('This protocol fixes source and returns-only features.')
    base=BASE_DIR/'reports/source_comparison'/config['control_study']
    base_proto=json.loads((base/'protocol.json').read_text()); cfg=base_proto['config']
    versions={p:importlib.metadata.version(p) for p in base_proto['versions']}
    if versions!=base_proto['versions']:raise ValueError('Environment changed; do not mix old control with new fits.')
    base_manifest=json.loads((base/'manifest.json').read_text())
    if json.loads((base/'reload_verification.json').read_text())['candidates_reloaded']!=112:
        raise ValueError('Control must be independently reloaded before reuse.')
    datasets,cohort=load_inputs(base); features=datasets[SOURCE][0]; columns=cfg['feature_groups']['returns']
    base_predictions=pd.read_csv(base/'predictions.csv.gz',parse_dates=['Date',TARGET_DATE_COLUMN])
    controls=[c for c in base_manifest['candidates'] if c['source']==SOURCE and c['feature_group']=='returns']
    controls={(c['fold'],c['model'],c['seed']):c for c in controls}
    study=BASE_DIR/'reports/window_comparison'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ');study.mkdir(parents=True)
    artifacts=BASE_DIR/'models/research'/('window-comparison_'+study.name);artifacts.mkdir(parents=True)
    before=guarded_hashes()
    protocol={'schema':'fpt-window-comparison-v1','study_id':study.name,'registered_before_fit_utc':utc_now(),'config':config,
        'config_sha256':file_sha256(config_path),'python':platform.python_version(),'versions':versions,
        'fixed_model_and_fold_config':cfg,'base_protocol_sha256':file_sha256(base/'protocol.json'),
        'input_zip_sha256':file_sha256(base/'input_snapshot.zip'),'protected_before':before,
        'source_sha256':{name:file_sha256(BASE_DIR/name) for name in ['src/models/window_comparison.py','src/models/window_comparison_report.py',
            'src/models/source_comparison.py','src/models/robustness.py','src/models/experiment.py']},
        'models_selected_on':'validation mean RMSE over seeds, never evaluation','serving_promotion':False}
    save_json(study/'protocol.json',protocol)
    with zipfile.ZipFile(study/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in protocol['source_sha256']:archive.write(BASE_DIR/name,name)
        archive.write(config_path,'config.json')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    records=[];predictions=[];specs=[];histories={};counts=[];new_fits=0
    save_json(study/'status.json',{'status':'running','conditions':0,'new_fits':0})
    for fold in cfg['folds']:
        cache={}; full_masks=fold_masks(cohort,fold)
        for window,years in config['windows'].items():
            masks=window_masks(cohort,fold,years)
            parts={phase:source_part(cohort.loc[mask].reset_index(drop=True),SOURCE,features) for phase,mask in masks.items()}
            training=parts['train'];validation=parts['validation']
            if len(training)<100:raise ValueError('Insufficient training samples.')
            context=context_positions(training.Feature_Position.to_numpy(),cfg['lookback'])
            assert features.iloc[context].Date.le(pd.Timestamp(fold['train_end'])).all()
            counts.append({'fold':fold['id'],'window':window,'train_n':len(training),'validation_n':len(validation),
                'evaluation_n':len(parts['evaluation']),'train_first':str(training.Date.min().date()),'train_last':str(training.Date.max().date()),
                'context_first':str(features.iloc[context].Date.min().date())})
            def record(model_name,seed,phase,pred,candidate):
                part=parts[phase];actual=part[REFERENCE].to_numpy()
                identity={'fold':fold['id'],'window':window,'model':model_name,'seed':seed,'phase':phase,'candidate':candidate}
                records.append({**identity,**score_direction(actual,pred)})
                output=part[['Date',TARGET_DATE_COLUMN,REFERENCE]].copy();output['predicted_log_return']=pred
                for k,v in identity.items():output[k]=v
                predictions.append(output)
            for baseline in ['Zero-Return','Train Mean','Return Persistence']:
                for phase in ['validation','evaluation']:
                    part=parts[phase];pred=np.zeros(len(part)) if baseline=='Zero-Return' else (np.full(len(part),training[TARGET_COLUMN].mean()) if baseline=='Train Mean' else part.Log_Return.to_numpy())
                    record(baseline,None,phase,pred,f"{fold['id']}_{window}_{baseline}")
            keymask=hashlib.sha256(masks['train'].tobytes()).hexdigest()
            for model_name in ['Ridge','XGBoost','LSTM']:
                for seed in ([None] if model_name=='Ridge' else cfg['seeds']):
                    name=f"{fold['id']}_{window}_{model_name}_s{seed}";key=(keymask,model_name,seed);started=time.perf_counter()
                    if np.array_equal(masks['train'],full_masks['train']):
                        old=controls[(fold['id'],model_name,seed)];old_dir=BASE_DIR/base_manifest['artifact_dir']
                        for f in [old['file'],old['scalers']]:
                            if f and file_sha256(old_dir/f)!=base_manifest['artifact_sha256'][f]:raise ValueError('Control weights changed.')
                        pred_by_phase={}
                        for phase in ['validation','evaluation']:
                            saved=base_predictions[(base_predictions.candidate==old['candidate'])&(base_predictions.phase==phase)]
                            assert np.array_equal(saved.Date.to_numpy(),parts[phase].Date.to_numpy())
                            assert np.allclose(saved[REFERENCE],parts[phase][REFERENCE],atol=1e-12)
                            pred_by_phase[phase]=saved.predicted_log_return.to_numpy()
                        item={'model_path':(old_dir/old['file']).relative_to(BASE_DIR).as_posix(),
                            'scalers_path':(old_dir/old['scalers']).relative_to(BASE_DIR).as_posix() if old['scalers'] else None,
                            'reused_from':old['candidate'],'new_fit':False}
                    elif key in cache:
                        item,pred_by_phase=cache[key];item={**item,'new_fit':False,'reused_from':item['candidate']}
                    else:
                        model,path,scaler_path,scalers,values,history=fit_one(model_name,seed,cfg,features,training,validation,columns,artifacts,name)
                        pred_by_phase={}
                        for phase in ['validation','evaluation']:
                            part=parts[phase]
                            pred_by_phase[phase]=scalers['target'].inverse_transform(model(make_sequences(values,part.Feature_Position.to_numpy(),cfg['lookback']),training=False).numpy()).ravel() if model_name=='LSTM' else model.predict(part[columns])
                        item={'model_path':path.relative_to(BASE_DIR).as_posix(),'scalers_path':scaler_path.relative_to(BASE_DIR).as_posix() if scaler_path else None,
                              'reused_from':None,'new_fit':True}
                        if history:histories[name]=history
                        new_fits+=1
                    item={**item,'candidate':name,'fold':fold['id'],'window':window,'model':model_name,'seed':seed,'seconds':time.perf_counter()-started}
                    specs.append(item);cache[key]=(item,pred_by_phase)
                    for phase,pred in pred_by_phase.items():record(model_name,seed,phase,pred,name)
                    pd.DataFrame(records).to_csv(study/'metrics.csv',index=False,lineterminator='\n')
                    save_json(study/'status.json',{'status':'running','conditions':len(specs),'new_fits':new_fits})
                    print(f"{len(specs)}/112 {name}: {'fit' if item['new_fit'] else 'reuse'}; val={records[-2]['rmse']:.7f} eval={records[-1]['rmse']:.7f}",flush=True)
    pd.concat(predictions,ignore_index=True).to_csv(study/'predictions.csv.gz',index=False,compression='gzip',date_format='%Y-%m-%d',lineterminator='\n')
    pd.DataFrame(counts).to_csv(study/'cohort_counts.csv',index=False,lineterminator='\n');save_json(study/'lstm_history.json',histories)
    paths={i[k] for i in specs for k in ['model_path','scalers_path'] if i[k]}
    assert guarded_hashes()==before
    save_json(study/'manifest.json',{'conditions':specs,'new_fits':new_fits,'reused_conditions':len(specs)-new_fits,
        'artifact_sha256':{p:file_sha256(BASE_DIR/p) for p in paths},'protected_after':guarded_hashes(),
        'protocol_sha256':file_sha256(study/'protocol.json'),'automatic_promotion':False})
    save_json(study/'status.json',{'status':'complete','conditions':len(specs),'new_fits':new_fits,'finished_at_utc':utc_now()})
    print(f'Complete {study}',flush=True)
    return study


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=BASE_DIR/'configs/window_comparison_v1.json')
    run(parser.parse_args().config)
