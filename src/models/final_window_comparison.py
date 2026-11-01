"""Bounded final 3y/5y/all-history Ridge contrast, without serving promotion."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
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

from src.data.contracts import BASE_DIR, file_sha256, utc_now
from src.features.build_features import build_dataset, TARGET_COLUMN, TARGET_DATE_COLUMN
from src.models.local_demo import FEATURES, LocalRidgeService, save_json
from src.models.recency_bias import causal_calibration
from src.models.recursive_bootstrap_evaluation import EVALUATION, protected_hashes
from src.models.recursive_horizon_calibration import historical_service, forecast_record


def training_rows(labels, cutoff, years, history_start='2018-01-01'):
    cutoff=pd.Timestamp(cutoff)
    start=pd.Timestamp(history_start) if years is None else max(pd.Timestamp(history_start),cutoff-pd.DateOffset(years=years)+pd.Timedelta(days=1))
    return labels.loc[labels.Date.ge(start)&labels.Date.le(cutoff)&labels[TARGET_DATE_COLUMN].le(cutoff)].copy()


def select_window(metrics, config):
    part=metrics.loc[metrics.phase.eq('validation')&metrics.model.eq('Ridge')&metrics.horizon.eq(1)]
    expected=set(config['validation_years']);scores=[]
    for priority,window in enumerate(config['windows']):
        rows=part.loc[part.window.eq(window)]
        if set(rows.year)!=expected or len(rows)!=len(expected):
            raise ValueError('Every candidate must cover all registered validation years once.')
        scores.append({'window':window,'validation_mean_rmse':float(rows.rmse_log_return.mean()),
            'validation_worst_rmse':float(rows.rmse_log_return.max()),'priority':priority})
    table=pd.DataFrame(scores).sort_values(['validation_mean_rmse','priority'],kind='stable')
    return str(table.iloc[0].window),table


def metric_table(predictions):
    rows=[]
    for keys,part in predictions.groupby(['phase','year','horizon','window','model'],sort=False):
        phase,year,h,window,model=keys
        y,p=part.actual_log_return.to_numpy(float),part.predicted_log_return.to_numpy(float)
        nonzero=y!=0;active=np.any(p!=0) and model!='Zero-Return'
        recalls=[float((np.sign(p[y*s>0])==s).mean()) for s in [1,-1] if (y*s>0).any()]
        err=p-y;zero=float(np.mean(y*y))
        row={'phase':phase,'year':int(year),'horizon':int(h),'window':window,'model':model,'n':len(part),
            'nonzero_actual_n':int(nonzero.sum()),'origin_start':part.origin_date.min(),'origin_end':part.origin_date.max(),
            'target_start':part.target_date.min(),'target_end':part.target_date.max(),
            'rmse_log_return':float(np.sqrt(np.mean(err*err))),'mae_log_return':float(np.mean(abs(err))),
            'rmse_price_vnd':float(np.sqrt(np.mean((part.predicted_close-part.actual_close)**2))),
            'mae_price_vnd':float(np.mean(abs(part.predicted_close-part.actual_close))),
            'hit_rate_pct':float(100*(np.sign(p[nonzero])==np.sign(y[nonzero])).mean()) if active and nonzero.any() else None,
            'balanced_accuracy_pct':float(100*np.mean(recalls)) if active and len(recalls)==2 else None,
            'mse_skill_vs_zero':float(1-np.mean(err*err)/zero) if zero else None,
            'coverage_pct':None,'mean_width_log_return':None,'mean_interval_score':None}
        valid=part.lower_log_return.notna()&part.upper_log_return.notna()
        if valid.any():
            from src.models.recursive_bootstrap_evaluation import interval_scores
            a=part.loc[valid];truth=a.actual_log_return.to_numpy();lo=a.lower_log_return.to_numpy();hi=a.upper_log_return.to_numpy()
            row.update(interval_n=int(valid.sum()),coverage_pct=float(100*np.mean((truth>=lo)&(truth<=hi))),
                mean_width_log_return=float(np.mean(hi-lo)),mean_interval_score=float(interval_scores(truth,lo,hi).mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_loss_ci(predictions, config):
    """Pair same dates; resample 20-origin blocks within each validation year."""
    table=[];cfg=config['paired_ci']
    for window in ['5y','all_2018']:
        differences=[];replicates=[];rng=np.random.default_rng(cfg['seed'])
        for year in config['validation_years']:
            part=predictions.loc[predictions.phase.eq('validation')&predictions.year.eq(year)&predictions.model.eq('Ridge')]
            a=part.loc[part.window.eq('3y')].sort_values('origin_date');b=part.loc[part.window.eq(window)].sort_values('origin_date')
            assert a.origin_date.to_list()==b.origin_date.to_list()
            np.testing.assert_array_equal(a.actual_log_return.to_numpy(),b.actual_log_return.to_numpy())
            y=a.actual_log_return.to_numpy();pa=a.predicted_log_return.to_numpy();pb=b.predicted_log_return.to_numpy()
            delta=(pb-y)**2-(pa-y)**2;differences.append(float(delta.mean()))
            n=len(delta);block=cfg['block_origins'];starts=rng.integers(0,n,(cfg['resamples'],(n+block-1)//block))
            idx=((starts[:,:,None]+np.arange(block))%n).reshape(cfg['resamples'],-1)[:,:n]
            replicates.append(delta[idx].mean(axis=1))
        low,high=np.quantile(np.mean(replicates,axis=0),[.025,.975])
        table.append({'window':window,'control':'3y','mean_year_mse_delta':float(np.mean(differences)),
            'descriptive_ci95_lower':float(low),'descriptive_ci95_upper':float(high),
            'block_origins':cfg['block_origins'],'resamples':cfg['resamples'],
            'meaning':'negative favors candidate; adaptive inspected validation, not independent evidence of future superiority'})
    return pd.DataFrame(table)


def plot_results(metrics, selection, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors=['#22a98d','#5085b4','#b9824f']
    windows=['3y','5y','all_2018']
    for window,color in zip(windows,colors):
        part=metrics.loc[metrics.phase.eq('validation')&metrics.model.eq('Ridge')&metrics.window.eq(window)].sort_values('year')
        axes[0,0].plot(part.year,part.rmse_log_return,'o-',label=window,color=color)
    axes[0,0].set_xticks([2022,2023,2024,2025]);axes[0,0].set_title('Raw Ridge / validation-year RMSE');axes[0,0].legend()
    s=selection.set_index('window').loc[windows];bars=axes[0,1].bar(windows,s.validation_mean_rmse,color=colors)
    axes[0,1].bar_label(bars,fmt='%.6f',padding=3);axes[0,1].margins(y=.15);axes[0,1].set_title('Selection: equal-year mean RMSE / lower is better')
    for ax,h in zip(axes[1],[1,22]):
        data=metrics.loc[metrics.phase.eq('diagnostic_2026')&metrics.horizon.eq(h)&metrics.model.eq('Ridge')].set_index('window').loc[windows]
        bars=ax.bar(windows,data.rmse_log_return,color=colors);ax.bar_label(bars,fmt='%.6f',padding=3);ax.margins(y=.2)
        zero=metrics.loc[metrics.phase.eq('diagnostic_2026')&metrics.horizon.eq(h)&metrics.model.eq('Zero-Return')].iloc[0]
        ax.axhline(zero.rmse_log_return,color='#7e8794',ls='--',label='Zero Return')
        ax.set_title(f'2026 already-inspected diagnostic / h={h} / n={int(data.n.iloc[0])}');ax.legend(fontsize=9)
    fig.suptitle('FPT / final bounded training-window contrast\nSame alpha100, features, source and evaluation dates; no serving promotion',fontsize=13)
    fig.savefig(path,dpi=160);plt.close(fig)


def run(config_path=BASE_DIR/'configs/final_window_comparison_v1.json'):
    config_path=Path(config_path);config=json.loads(config_path.read_text())
    if (config['windows']!={'3y':3,'5y':5,'all_2018':None} or config['alpha']!=100.
            or config['features']!=FEATURES or config['automatic_promotion'] is not False):
        raise ValueError('Only the registered three-window fixed-alpha contrast is supported.')
    before=protected_hashes(BASE_DIR);source=BASE_DIR/EVALUATION
    acquisition=json.loads((source/'acquisition.json').read_text())
    assert file_sha256(source/'data_snapshot.zip')==acquisition['zip_sha256']
    with zipfile.ZipFile(source/'data_snapshot.zip') as archive:content=archive.read('vci.csv')
    assert hashlib.sha256(content).hexdigest()==acquisition['files']['vci.csv']
    raw=pd.read_csv(io.BytesIO(content),parse_dates=['Date'])
    if raw.Date.min()<pd.Timestamp(config['history_start']):raw=raw.loc[raw.Date.ge(config['history_start'])].copy()
    features,labels,_,quality=build_dataset(raw)
    assert str(features.Date.max().date())==config['asof_candidate_cutoff']
    indexed=raw.set_index('Date')
    truth=np.log(indexed.loc[labels[TARGET_DATE_COLUMN],'Close'].to_numpy()/labels.Close.to_numpy())
    np.testing.assert_allclose(truth,labels[TARGET_COLUMN],atol=2e-15,rtol=0)
    study=BASE_DIR/'reports/final_window_comparison'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ');study.mkdir(parents=True)
    artifact_dir=BASE_DIR/'models/research'/('final-window_'+study.name);artifact_dir.mkdir()
    sources=['src/models/final_window_comparison.py','src/models/recency_bias.py','src/models/recursive_horizon_calibration.py',
        'src/models/recursive_bootstrap_evaluation.py','src/models/forecast_path.py','src/models/local_demo.py',
        'src/features/build_features.py','src/data/contracts.py','tests/test_final_window_comparison.py']
    protocol={'registered_before_training_utc':utc_now(),'config':config,'config_sha256':file_sha256(config_path),
        'input_zip_sha256':file_sha256(source/'data_snapshot.zip'),'input_vci_sha256':acquisition['files']['vci.csv'],
        'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','joblib']},
        'source_sha256':{p:file_sha256(BASE_DIR/p) for p in sources},'data_quality':quality,
        'selection_role':'historical validation2022-25 already inspected; 2026 never chooses window; no untouched holdout claim'}
    save_json(study/'protocol.json',protocol)
    with zipfile.ZipFile(study/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(BASE_DIR/p,p)
        z.write(config_path,'config.json')
    registry=[];cache={};rows=[];calibrations={};models_2025={};fit_count=0;train_rows=[]

    def fit(window,cutoff):
        nonlocal fit_count
        train=training_rows(labels,cutoff,config['windows'][window],config['history_start'])
        assert len(train)>=100 and train[TARGET_DATE_COLUMN].max()<=pd.Timestamp(cutoff)
        signature=hashlib.sha256(pd.util.hash_pandas_object(train[['Date',TARGET_DATE_COLUMN,*FEATURES,TARGET_COLUMN]],index=False).values.tobytes()).hexdigest()
        identity=f'{cutoff}_{window}'
        if signature in cache:
            model,path,reused=cache[signature];fresh=False
        else:
            model=make_pipeline(StandardScaler(),Ridge(alpha=config['alpha']));model.fit(train[FEATURES],train[TARGET_COLUMN])
            np.testing.assert_allclose(model[0].mean_,train[FEATURES].mean().to_numpy(),atol=2e-15,rtol=1e-12)
            path=artifact_dir/(identity+'.pkl');joblib.dump(model,path);fresh=True;reused=None;fit_count+=1
            cache[signature]=(model,path,identity)
        item={'id':identity,'window':window,'cutoff':cutoff,'train_target_end':str(train[TARGET_DATE_COLUMN].max().date()),
            'train_origin_start':str(train.Date.min().date()),'train_origin_end':str(train.Date.max().date()),'train_n':len(train),
            'new_fit':fresh,'reused_from':reused,'path':path.relative_to(BASE_DIR).as_posix(),'sha256':file_sha256(path),'training_signature':signature}
        registry.append(item)
        train_rows.extend({'candidate':identity,'Date':str(r.Date.date()),TARGET_DATE_COLUMN:str(getattr(r,TARGET_DATE_COLUMN).date())} for r in train[['Date',TARGET_DATE_COLUMN]].itertuples(index=False))
        return model,item,train

    for year in [*config['validation_years'],config['diagnostic_year']]:
        cutoff=f'{year-1}-12-31';phase='validation' if year in config['validation_years'] else 'diagnostic_2026'
        part=labels.loc[labels.Date.ge(f'{year}-01-01')&labels[TARGET_DATE_COLUMN].le(f'{year}-12-31')].copy().reset_index(drop=True)
        assert not part.empty and part.Date.min()>pd.Timestamp(cutoff)
        common_keys=None
        for window in config['windows']:
            model,spec,train=fit(window,cutoff);loaded=joblib.load(BASE_DIR/spec['path'])
            prediction=model.predict(part[FEATURES]);np.testing.assert_allclose(loaded.predict(part[FEATURES]),prediction,atol=0,rtol=0)
            keys=list(zip(part.Date,part[TARGET_DATE_COLUMN]));assert common_keys is None or common_keys==keys;common_keys=keys
            predictions={'Ridge':prediction,'Zero-Return':np.zeros(len(part)),'TrainMean':np.full(len(part),train[TARGET_COLUMN].mean())}
            if year==config['diagnostic_year']:
                cal=part[['Date',TARGET_DATE_COLUMN]].copy();cal['Reference_Return']=part[TARGET_COLUMN].to_numpy();cal['raw_prediction']=prediction
                cal=causal_calibration(cal,config['bias'],config['interval']);calibrations[window]=cal;models_2025[window]=(loaded,spec)
                predictions['Ridge+bias60']=cal.prediction.to_numpy()
            for name,pred in predictions.items():
                for j,r in enumerate(part.itertuples(index=False)):
                    rows.append({'phase':phase,'year':year,'horizon':1,'window':window,'model':name,
                        'origin_date':str(r.Date.date()),'target_date':str(getattr(r,TARGET_DATE_COLUMN).date()),
                        'actual_log_return':float(getattr(r,TARGET_COLUMN)),'predicted_log_return':float(pred[j]),
                        'origin_close':float(r.Close),'actual_close':float(r.Close*np.exp(getattr(r,TARGET_COLUMN))),
                        'predicted_close':float(r.Close*np.exp(pred[j])), 'lower_log_return':None,'upper_log_return':None,
                        'train_cutoff':cutoff,'train_n':len(train)})
            print(f'{year} / {window}: train={len(train)}, score={len(part)}, fit={spec["new_fit"]}',flush=True)
    initial=pd.DataFrame(rows);initial_metrics=metric_table(initial)
    selected,selection=select_window(initial_metrics,config)
    # Selection is frozen before any multistep diagnostics or as-of candidate fit.
    save_json(study/'validation_selection.json',{'selected_window':selected,'rule':config['selection'],
        'selected_before_multistep_diagnostics_utc':utc_now(),'scores':selection.to_dict('records')})
    positions=[i for i,d in enumerate(features.Date) if d.year==2026 and i+22<len(features) and i>=20 and
        len(calibrations['3y'].loc[(calibrations['3y'].Date<d)&(calibrations['3y'][TARGET_DATE_COLUMN]<=d)])>=60]
    for ordinal,i in enumerate(positions):
        for window in config['windows']:
            model,spec=models_2025[window]
            service=historical_service(BASE_DIR,model,raw,features,calibrations[window],spec,i)
            for name in ['Ridge','Ridge+bias60']:
                for h in config['multistep_diagnostic_horizons']:
                    record=forecast_record(service,features.Date.iloc[i+1:i+h+1],features.Close.iloc[i+h],ordinal,name,'unscaled',{})
                    rows.append({**record,'phase':'diagnostic_2026','year':2026,'window':window,'train_cutoff':'2025-12-31','train_n':spec['train_n']})
            for h in config['multistep_diagnostic_horizons']:
                origin=features.iloc[i];target=features.iloc[i+h]
                for name,pred in [('Zero-Return',0.),('TrainMean',float(training_rows(labels,'2025-12-31',config['windows'][window])[TARGET_COLUMN].mean()*h))]:
                    rows.append({'phase':'diagnostic_2026','year':2026,'horizon':h,'window':window,'model':name,
                        'origin_date':str(origin.Date.date()),'target_date':str(target.Date.date()),
                        'actual_log_return':float(np.log(target.Close/origin.Close)),'predicted_log_return':pred,
                        'origin_close':float(origin.Close),'actual_close':float(target.Close),'predicted_close':float(origin.Close*np.exp(pred)),
                        'lower_log_return':None,'upper_log_return':None,'train_cutoff':'2025-12-31','train_n':spec['train_n']})
        if (ordinal+1)%20==0:print(f'Paired h5/h22 origins {ordinal+1}/{len(positions)}',flush=True)
    candidates={};serving=LocalRidgeService()
    for window in config['windows']:
        model,spec,_=fit(window,config['asof_candidate_cutoff'])
        check=joblib.load(BASE_DIR/spec['path']);np.testing.assert_allclose(check.predict(features[FEATURES]),model.predict(features[FEATURES]),atol=0,rtol=0)
        candidates[window]={**spec,'selected_by_validation':window==selected,'role':'unscored as-of candidate; raw-only research artifact, not a serving bundle'}
        if window=='3y':np.testing.assert_allclose(model.predict(features[FEATURES]),serving.loaded.predict(features[FEATURES]),atol=2e-14,rtol=0)
    predictions=pd.DataFrame(rows);metrics=metric_table(predictions);ci=paired_loss_ci(predictions,config)
    predictions.to_csv(study/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    metrics.to_csv(study/'metrics.csv',index=False);selection.to_csv(study/'validation_summary.csv',index=False);ci.to_csv(study/'paired_validation_ci.csv',index=False)
    pd.DataFrame(train_rows).to_csv(study/'training_dates.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    for window,cal in calibrations.items():cal.to_csv(study/(window+'_2026_calibration.csv'),index=False,date_format='%Y-%m-%d')
    plot_results(metrics,selection,study/'comparison.png')
    save_json(study/'manifest.json',{'conditions':registry,'new_fits':fit_count,'condition_count':len(registry),
        'reused_conditions':len(registry)-fit_count,'asof_candidates':candidates,'automatic_promotion':False,
        'artifact_sha256':{r['path']:r['sha256'] for r in registry}})
    with zipfile.ZipFile(study/'weights_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted({r['path'] for r in registry}):archive.write(BASE_DIR/path,path)
    save_json(study/'decision.json',{'selected_window_from_validation':selected,'selection_rule':config['selection'],
        'validation_scores':selection.to_dict('records'),'diagnostic2026_not_used_for_selection':True,
        'asof_candidates_have_no_scored_future_labels':True,'current_serving_model_unchanged':True,
        'serving_3y_refit_predictions_identical':True,'future_model_choice_requires_user_review':True,
        'no_additional_grid_or_training_after_this_study':True})
    # Check source inputs, existing artifacts/reports and serving/reference pointers stayed fixed.
    after=protected_hashes(BASE_DIR)
    prior={k:v for k,v in after.items() if not k.startswith(study.relative_to(BASE_DIR).as_posix()+'/') and not k.startswith(artifact_dir.relative_to(BASE_DIR).as_posix()+'/')}
    assert before==prior,'Existing data/models/evidence changed.'
    save_json(study/'verification.json',{'completed_at_utc':utc_now(),'new_model_fits':fit_count,'conditions':len(registry),
        'protected_files_unchanged':len(before),'protected_tree_sha256':hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
        'all_models_independently_reloaded':True,'scalers_fit_train_only':True,'same_origin_target_pairs_for_all_windows':True,
        'serving_3y_refit_predictions_identical':True,'automatic_promotion':False,'downloads':0,
        'protocol_sha256':file_sha256(study/'protocol.json'),
        'outputs_sha256':{p.name:file_sha256(p) for p in study.iterdir() if p.is_file() and p.name!='protocol.json'}})
    print(selection.to_string(index=False));print(metrics.loc[metrics.phase.eq('diagnostic_2026')].to_string(index=False));print('Evidence:',study,flush=True)
    return study


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=BASE_DIR/'configs/final_window_comparison_v1.json')
    run(parser.parse_args().config)
