"""Run the serving forecast function at historical origins; fit interval scale only."""
from __future__ import annotations

import argparse
import json
import math
import zipfile
import io
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.contracts import BASE_DIR, file_sha256, utc_now
from src.models.forecast_path import forecast, widen_band, BAND_POLICY
from src.models.local_demo import LocalRidgeService, save_json
from src.models.recursive_bootstrap_evaluation import load_inputs, mature_pool, protected_hashes, summarize, EVALUATION

CONFIG = {
    'schema': 'fpt-recursive-horizon-calibration-protocol-v1', 'horizons': [5, 22],
    'calibration_target_end': '2026-06-30', 'evaluation_origin_start': '2026-07-01',
    'minimum_calibration_n': 20, 'nominal_coverage': .8,
    'scale_rule': 'max(1, kth ordered asymmetric normalized error), k=ceil((n+1)*0.8); no refitting on evaluation',
    'activation_gate': 'candidate fixed on calibration only; activate per model/h if evaluation coverage distance to 80% and mean interval score do not worsen; at least 30 evaluation origins',
    'nonoverlap': 'every h-th evaluation origin anchored at first evaluation origin',
    'origin_cohort': '2026 origins with 60 matured one-step errors and all 22 future valid observed sessions',
    'forecast_function': 'src.models.forecast_path.forecast; observation_dates supplies dates only, no future prices',
    'historical_service': 'LocalRidgeService.predict reused with frozen end-2025 estimator and raw/features truncated at origin',
    'model_fits': 0, 'downloads': 0, 'automatic_model_promotion': False,
    'role': 'adaptive retrospective study on already inspected 2026; later period is time-separated, not a fresh untouched test',
    'serving_transfer': 'historical scale applied to October bundle only as a disclosed demo policy; no validation of that bundle',
    'references': ['https://otexts.com/fpp3/tscv.html', 'https://statgrades.berkeley.edu/~ryantibs/statlearn-s24/lectures/conformal.pdf'],
}


def phase_for(origin, target):
    if pd.Timestamp(target) <= pd.Timestamp(CONFIG['calibration_target_end']):
        return 'calibration'
    if pd.Timestamp(origin) >= pd.Timestamp(CONFIG['evaluation_origin_start']):
        return 'evaluation'
    return 'embargo'


def calibration_scale(frame, *, as_of=None):
    """A split-calibration scale, not an exchangeability/coverage guarantee for stocks."""
    cutoff=CONFIG['calibration_target_end'] if as_of is None else as_of
    if (len(frame) < CONFIG['minimum_calibration_n'] or (as_of is None and not frame.phase.eq('calibration').all())
            or pd.to_datetime(frame.target_date).max() > pd.Timestamp(cutoff)):
        raise ValueError('Only sufficiently many pre-cutoff calibration labels may set the scale.')
    point = frame.predicted_log_return.to_numpy(float)
    y = frame.actual_log_return.to_numpy(float)
    left = point - frame.lower_log_return.to_numpy(float)
    right = frame.upper_log_return.to_numpy(float) - point
    if not np.isfinite([point, y, left, right]).all() or (left <= 0).any() or (right <= 0).any():
        raise ValueError('Calibration band must strictly surround its point.')
    scores = np.maximum.reduce([(point-y)/left, (y-point)/right, np.zeros(len(frame))])
    rank = math.ceil((len(scores)+1)*CONFIG['nominal_coverage'])
    if rank > len(scores):
        raise ValueError('Too few scores for the requested coverage.')
    return max(1., float(np.sort(scores)[rank-1]))


def activation_supported(before, after, evaluated_factors, latest_factor):
    """Do not activate a newly available scale never exercised by scored forecasts."""
    expanded=np.asarray(evaluated_factors,dtype=float)
    expanded=expanded[expanded>1.+1e-12]
    supported=(latest_factor==1. or (len(expanded)>=20 and latest_factor<=expanded.max()+1e-12))
    improves=(latest_factor==1. or abs(after.coverage_pct-80)<abs(before.coverage_pct-80)-1e-10
        or after.mean_interval_score<before.mean_interval_score-1e-12)
    return bool(supported and improves and before.n>=30 and
        abs(after.coverage_pct-80)<=abs(before.coverage_pct-80)+1e-10 and
        after.mean_interval_score<=before.mean_interval_score+1e-12)


def historical_service(root, estimator, raw, features, calibration, spec, position):
    """No serving cutoff bypass: construct a distinct, fully truncated historical service."""
    origin = features.iloc[position].Date
    errors, corrected, _ = mature_pool(calibration, origin)
    prior = calibration.loc[(calibration.Date < origin) & (calibration.Target_Date_1d <= origin)].tail(60)
    service = object.__new__(LocalRidgeService)
    service.root = root
    service.loaded = estimator
    service.raw = raw.loc[raw.Date <= origin].copy()
    service.features = features.iloc[:position+1].copy()
    service.manifest = {'run_id': spec['id'], 'default_model': 'Ridge+bias60', 'caveats': [CONFIG['role']]}
    service.bundle = {'training_target_end': spec['train_target_end'],
        'calibration_targets': prior.Target_Date_1d.dt.strftime('%Y-%m-%d').to_list(),
        'raw_errors': errors.tolist(), 'corrected_errors': corrected.tolist(),
        'bias_spec': {'window': 60, 'minimum_matured': 20},
        'interval_spec': {'window': 60, 'minimum_matured': 60, 'quantiles': [.1, .9]}}
    assert service.raw.Date.max() == origin and prior.Target_Date_1d.max() <= origin
    assert pd.Timestamp(spec['train_target_end']) < origin
    return service


def forecast_record(service, dates, actual_close, ordinal, model, variant, policy):
    dates = pd.DatetimeIndex(dates)
    origin = service.features.iloc[-1]
    result = forecast(service, 'month' if len(dates) == 22 else 'week', model,
        today=origin.Date + pd.Timedelta(days=1), observation_dates=dates, band_calibration=policy)
    assert result['origin_date'] == str(origin.Date.date()) and result['steps'] == len(dates)
    final = result['path'][-1]; close = float(origin.Close)
    return {'origin_ordinal': ordinal, 'origin_date': result['origin_date'],
        'target_date': final['estimated_date'], 'horizon': len(dates), 'model': model, 'variant': variant,
        'phase': phase_for(origin.Date, dates[-1]), 'origin_close': close, 'actual_close': float(actual_close),
        'predicted_close': final['predicted_close_adjusted'],
        'actual_log_return': float(np.log(actual_close/close)),
        'predicted_log_return': float(np.log(final['predicted_close_adjusted']/close)),
        'lower_log_return': float(np.log(final['scenario_lower']/close)) if final['scenario_lower'] is not None else None,
        'upper_log_return': float(np.log(final['scenario_upper']/close)) if final['scenario_upper'] is not None else None,
        'band_scale_factor': final['band_scale_factor'],
        'last_calibration_target': max(service.bundle['calibration_targets']),
        'model_train_target_end': service.bundle['training_target_end'], 'forecast_method_version': result['forecast_method_version']}


def metrics_for(predictions):
    parts = [predictions.loc[predictions.phase.ne('embargo')].assign(sample=lambda x:x.phase)]
    evaluation = predictions.loc[predictions.phase.eq('evaluation')].copy()
    first = evaluation.origin_ordinal.min()
    sparse = evaluation.loc[(evaluation.origin_ordinal-first).mod(evaluation.horizon).eq(0)]
    parts.append(sparse.assign(sample='evaluation_nonoverlap'))
    parts.append(predictions.loc[predictions.variant.eq('unscaled')].assign(sample='full_history_unscaled'))
    frame = pd.concat(parts, ignore_index=True)
    metrics = summarize(frame)
    extra = frame.groupby(['sample','horizon','model','variant'],sort=False).apply(
        lambda x: pd.Series({'rmse_price_vnd': float(np.sqrt(np.mean((x.actual_close-x.predicted_close)**2))),
            'nonzero_actual_n': int(x.actual_log_return.ne(0).sum())}), include_groups=False).reset_index()
    return metrics.merge(extra,on=['sample','horizon','model','variant'],validate='one_to_one')


def plot_results(metrics, path, *, adaptive=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for j,h in enumerate(CONFIG['horizons']):
        part = metrics.loc[metrics['sample'].eq('evaluation') & metrics.horizon.eq(h) & metrics.model.ne('Zero-Return')]
        part = part.set_index(['model','variant']).loc[[(m,v) for m in ['Ridge','Ridge+bias60'] for v in ['unscaled','candidate_scaled']]]
        labels=['Ridge / before','Ridge / scaled','Ridge+bias / before','Ridge+bias / scaled']
        for ax,col,title,mult in [(axes[0,j],'coverage_pct','Later-period coverage (%)',1),
                (axes[1,j],'mean_interval_score','Later-period interval score x100 (lower is better)',100)]:
            bars=ax.barh(labels,part[col]*mult,color=['#8293a8','#1ca88d']*2)
            ax.bar_label(bars,fmt='%.2f',padding=3);ax.invert_yaxis();ax.margins(x=.25)
            ax.set_title(f'h={h} observed sessions, n={int(part.n.iloc[0])}\n{title}',fontsize=10)
            ax.spines[['top','right']].set_visible(False)
        axes[0,j].axvline(80,color='#bf753c',ls='--');axes[0,j].set_xlim(0,105)
    title='rolling scale from matured past multi-step errors' if adaptive else 'band scale fixed on earlier labels'
    fig.suptitle(f'FPT / exact forecast() replay + {title}\nEnd-2025 Ridge; retrospective 2026; no model refit or point forecast change',fontsize=13)
    fig.savefig(path,dpi=160);plt.close(fig)


def run(root=BASE_DIR, *, install_policy=False):
    root=Path(root).resolve();before=protected_hashes(root)
    if install_policy and (root/BAND_POLICY).exists():
        raise ValueError('A policy already exists; replay without --install-policy to preserve it.')
    estimator,features,calibration,spec,inputs=load_inputs(root)
    with zipfile.ZipFile(root/EVALUATION/'data_snapshot.zip') as archive:
        raw=pd.read_csv(io.BytesIO(archive.read('vci.csv')),parse_dates=['Date'])
    positions=[i for i,d in enumerate(features.Date) if d.year==2026 and i>=20 and i+22<len(features)
        and len(calibration.loc[(calibration.Date<d)&(calibration.Target_Date_1d<=d)])>=60]
    destination=root/'reports/horizon_calibration'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    destination.mkdir(parents=True,exist_ok=False)
    source_files=['src/models/forecast_path.py','src/models/recursive_horizon_calibration.py',
        'src/models/recursive_bootstrap_evaluation.py','src/models/local_demo.py','src/features/build_features.py']
    protocol={'registered_before_evaluation_utc':utc_now(),'config':CONFIG,'inputs_sha256':inputs,
        'historical_model':{k:v for k,v in spec.items() if k!='training_dates'},
        'source_sha256':{p:file_sha256(root/p) for p in source_files},
        'origins':features.Date.iloc[positions].dt.strftime('%Y-%m-%d').to_list()}
    save_json(destination/'protocol.json',protocol)
    with zipfile.ZipFile(destination/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in source_files:archive.write(root/p,p)
    rows=[]
    for ordinal,i in enumerate(positions):
        service=historical_service(root,estimator,raw,features,calibration,spec,i)
        for model in ['Ridge','Ridge+bias60','Zero-Return']:
            for h in CONFIG['horizons']:
                rows.append(forecast_record(service,features.Date.iloc[i+1:i+h+1],features.Close.iloc[i+h],ordinal,model,'unscaled',{}))
        if (ordinal+1)%25==0:print(f'Exact forecast replay {ordinal+1}/{len(positions)} origins',flush=True)
    unscaled=pd.DataFrame(rows)
    candidate={'schema':'fpt-recursive-band-calibration-v1','study':destination.relative_to(root).as_posix(),
        'calibration_target_end':CONFIG['calibration_target_end'],
        'allowed_model_versions':[spec['id'],'ridge6_20261029'], 'evaluated_horizons_observations':CONFIG['horizons'],
        'source_model':spec['id'],'scope':CONFIG['serving_transfer'],'coverage_guaranteed':False,'factors':{},'calibration_n':{}}
    for model in ['Ridge','Ridge+bias60']:
        candidate['factors'][model]={};candidate['calibration_n'][model]={}
        for h in CONFIG['horizons']:
            part=unscaled.loc[unscaled.phase.eq('calibration')&unscaled.model.eq(model)&unscaled.horizon.eq(h)]
            candidate['factors'][model][str(h)]=calibration_scale(part)
            candidate['calibration_n'][model][str(h)]=len(part)
    save_json(destination/'candidate_policy.json',candidate)
    # Re-run forecast itself on later origins with locked earlier-period factors.
    for ordinal,i in enumerate(positions):
        if features.Date.iloc[i]<pd.Timestamp(CONFIG['evaluation_origin_start']):continue
        service=historical_service(root,estimator,raw,features,calibration,spec,i)
        for model in ['Ridge','Ridge+bias60']:
            for h in CONFIG['horizons']:
                row=forecast_record(service,features.Date.iloc[i+1:i+h+1],features.Close.iloc[i+h],ordinal,model,'candidate_scaled',candidate)
                prior=unscaled.loc[unscaled.origin_ordinal.eq(ordinal)&unscaled.horizon.eq(h)&unscaled.model.eq(model)].iloc[0]
                assert row['predicted_close']==prior.predicted_close
                rows.append(row)
        if (ordinal+1)%25==0:print(f'Locked-scale later-period replay {ordinal+1}/{len(positions)}',flush=True)
    # Fit-period calibration diagnostics are labelled as such, never evaluation scores.
    for row in unscaled.loc[unscaled.phase.eq('calibration')&unscaled.model.ne('Zero-Return')].to_dict('records'):
        factor=candidate['factors'][row['model']][str(row['horizon'])]
        close=row['origin_close'];lo,hi=widen_band(row['predicted_close'],close*np.exp(row['lower_log_return']),close*np.exp(row['upper_log_return']),factor)
        rows.append({**row,'variant':'candidate_scaled','band_scale_factor':factor,
            'lower_log_return':float(np.log(lo/close)),'upper_log_return':float(np.log(hi/close))})
    predictions=pd.DataFrame(rows);metrics=metrics_for(predictions);decisions=[]
    active={**candidate,'factors':{m:dict(v) for m,v in candidate['factors'].items()},'candidate_factors':candidate['factors']}
    for model in ['Ridge','Ridge+bias60']:
        for h in CONFIG['horizons']:
            part=metrics.loc[metrics['sample'].eq('evaluation')&metrics.model.eq(model)&metrics.horizon.eq(h)].set_index('variant')
            a,b=part.loc['unscaled'],part.loc['candidate_scaled'];factor=candidate['factors'][model][str(h)]
            passed=bool(a.n>=30 and abs(b.coverage_pct-80)<=abs(a.coverage_pct-80)+1e-10 and b.mean_interval_score<=a.mean_interval_score+1e-12)
            selected=factor if passed else 1.;active['factors'][model][str(h)]=selected
            decisions.append({'model':model,'horizon':h,'candidate_factor':factor,'active_factor':selected,'passed_gate':passed,
                'coverage_before':a.coverage_pct,'coverage_after':b.coverage_pct,
                'interval_score_before':a.mean_interval_score,'interval_score_after':b.mean_interval_score,
                'reason':'no widening needed on calibration' if factor==1 else 'passed predeclared gate' if passed else 'failed gate; retain unscaled band'})
    active['activation_decisions']=decisions;active['activation_gate']=CONFIG['activation_gate']
    save_json(destination/'active_policy.json',active)
    predictions.to_csv(destination/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    metrics.to_csv(destination/'metrics.csv',index=False);pd.DataFrame(decisions).to_csv(destination/'decisions.csv',index=False)
    plot_results(metrics,destination/'comparison.png')
    assert before==protected_hashes(root,exclude=destination),'Historical protected files changed.'
    # Explicit user-authorized configuration update, not model promotion or refitting.
    policy_path=root/BAND_POLICY
    if install_policy:save_json(policy_path,active)
    verification={'completed_at_utc':utc_now(),'new_model_fits':0,'new_downloads':0,'protected_files_unchanged':len(before),
        'forecast_function_reused':True,'historical_predict_uses_LocalRidgeService':True,
        'later_period_points_identical':True,'calibration_target_cutoff_before_evaluation_origins':True,
        'serving_bundle_has_no_scored_future_labels':True,'model_train_target_end':spec['train_target_end'],
        'common_origins':len(positions),'policy_installed':install_policy,
        'proposed_policy_sha256':file_sha256(destination/'active_policy.json'),
        'protocol_sha256':file_sha256(destination/'protocol.json'),
        'outputs_sha256':{p.name:file_sha256(p) for p in destination.iterdir() if p.is_file() and p.name!='protocol.json'}}
    save_json(destination/'verification.json',verification)
    print(metrics.to_string(index=False));print(pd.DataFrame(decisions).to_string(index=False));print('Evidence:',destination,flush=True)
    return destination


def run_adaptive(previous_study, root=BASE_DIR, *, install_policy=False):
    """Post-fixed-study rolling alternative; explicitly not a new untouched test."""
    root=Path(root).resolve();previous_study=Path(previous_study).resolve()
    if not previous_study.is_relative_to(root/'reports/horizon_calibration'):
        raise ValueError('Expected a previous horizon-calibration study.')
    before=protected_hashes(root);estimator,features,calibration,spec,inputs=load_inputs(root)
    prior_verification=json.loads((previous_study/'verification.json').read_text())
    prediction_path=previous_study/'predictions.csv.gz'
    assert file_sha256(prediction_path)==prior_verification['outputs_sha256']['predictions.csv.gz']
    unscaled=pd.read_csv(prediction_path);unscaled=unscaled.loc[unscaled.variant.eq('unscaled')].copy()
    with zipfile.ZipFile(root/EVALUATION/'data_snapshot.zip') as archive:
        raw=pd.read_csv(io.BytesIO(archive.read('vci.csv')),parse_dates=['Date'])
    destination=previous_study/'adaptive';destination.mkdir(exist_ok=False)
    config={**CONFIG,'variant':'rolling normalized-score calibration', 'score_window':60,'minimum_matured_scores':20,
        'role':'post-fixed-study adaptive retrospective analysis on the same already-inspected later period; no new holdout',
        'scale_rule':'at each origin use up to 60 previous same-model same-horizon errors with target <= origin, minimum20, same order statistic and floor1',
        'serving_policy':'freeze the latest available causal scale; model-specific transfer to current October bundle remains unvalidated'}
    sources=['src/models/recursive_horizon_calibration.py','src/models/forecast_path.py','tests/test_horizon_calibration.py']
    protocol={'registered_before_adaptive_evaluation_utc':utc_now(),'config':config,
        'previous_predictions_sha256':file_sha256(prediction_path),'inputs_sha256':inputs,
        'source_sha256':{p:file_sha256(root/p) for p in sources}}
    save_json(destination/'protocol.json',protocol)
    with zipfile.ZipFile(destination/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(root/p,p)
    new=[]
    for ordinal,origin_rows in unscaled.loc[unscaled.phase.eq('evaluation')].groupby('origin_ordinal',sort=True):
        origin=pd.Timestamp(origin_rows.origin_date.iloc[0]);i=int(features.index[features.Date.eq(origin)][0])
        service=historical_service(root,estimator,raw,features,calibration,spec,i)
        for _,old in origin_rows.loc[origin_rows.model.ne('Zero-Return')].iterrows():
            pool=unscaled.loc[unscaled.model.eq(old.model)&unscaled.horizon.eq(old.horizon)&
                (unscaled.target_date<=str(origin.date()))&(unscaled.origin_date<str(origin.date()))].sort_values('origin_date').tail(60)
            factor=calibration_scale(pool,as_of=origin)
            policy={'calibration_target_end':pool.target_date.max(),'allowed_model_versions':[spec['id']],
                'factors':{old.model:{str(old.horizon):factor}}}
            h=int(old.horizon);row=forecast_record(service,features.Date.iloc[i+1:i+h+1],features.Close.iloc[i+h],int(ordinal),old.model,'rolling_scaled',policy)
            assert abs(row['predicted_close']-old.predicted_close)<1e-9
            new.append({**row,'score_calibration_n':len(pool),'last_score_target':pool.target_date.max()})
        if len(new)%80==0:print(f'Adaptive replay {len(new)//4}/62 later origins',flush=True)
    predictions=pd.concat([unscaled,pd.DataFrame(new)],ignore_index=True);metrics=metrics_for(predictions)
    latest=str(features.Date.max().date());policy={'schema':'fpt-recursive-band-calibration-v1',
        'study':destination.relative_to(root).as_posix(),'calibration_target_end':latest,
        'allowed_model_versions':[spec['id'],'ridge6_20261029'],'evaluated_horizons_observations':[5,22],
        'source_model':spec['id'],'scope':config['serving_policy'],'coverage_guaranteed':False,
        'factors':{},'candidate_factors':{},'activation_decisions':[]}
    for model in ['Ridge','Ridge+bias60']:
        policy['factors'][model]={};policy['candidate_factors'][model]={}
        for h in CONFIG['horizons']:
            pool=unscaled.loc[unscaled.model.eq(model)&unscaled.horizon.eq(h)&unscaled.target_date.le(latest)].sort_values('origin_date').tail(60)
            final_factor=calibration_scale(pool,as_of=latest)
            part=metrics.loc[metrics['sample'].eq('evaluation')&metrics.model.eq(model)&metrics.horizon.eq(h)].set_index('variant')
            a,b=part.loc['unscaled'],part.loc['rolling_scaled']
            evaluated=predictions.loc[predictions.phase.eq('evaluation')&predictions.model.eq(model)&predictions.horizon.eq(h)&predictions.variant.eq('rolling_scaled'),'band_scale_factor']
            passed=activation_supported(a,b,evaluated,final_factor)
            selected=final_factor if passed else 1.
            policy['factors'][model][str(h)]=selected;policy['candidate_factors'][model][str(h)]=final_factor
            policy['activation_decisions'].append({'model':model,'horizon':h,'passed_gate':passed,
                'latest_candidate_factor':final_factor,'active_frozen_factor':selected,
                'coverage_before':a.coverage_pct,'coverage_rolling':b.coverage_pct,
                'interval_score_before':a.mean_interval_score,'interval_score_rolling':b.mean_interval_score,
                'score_calibration_n':len(pool),'last_score_target':pool.target_date.max(),
                'scope':'Gate evaluates rolling policy on historical model; does not validate this frozen factor on current model.'})
    predictions.to_csv(destination/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    metrics.to_csv(destination/'metrics.csv',index=False)
    pd.DataFrame(policy['activation_decisions']).to_csv(destination/'decisions.csv',index=False)
    save_json(destination/'active_policy.json',policy)
    plot_results(metrics.assign(variant=metrics.variant.replace({'rolling_scaled':'candidate_scaled'})),destination/'comparison.png',adaptive=True)
    # Parent's files remain historical evidence; only add this new adaptive sub-study.
    assert before==protected_hashes(root,exclude=destination)
    if install_policy:save_json(root/BAND_POLICY,policy)
    verification={'completed_at_utc':utc_now(),'new_model_fits':0,'downloads':0,'protected_files_unchanged':len(before),
        'later_period_origins':int(unscaled.loc[unscaled.phase.eq('evaluation'),'origin_ordinal'].nunique()),
        'same_forecast_function':True,'points_unchanged':True,'score_labels_matured_before_each_origin':True,
        'fresh_holdout':False,'activation_is_post_evaluation_demo_decision':True,
        'policy_installed':install_policy,'proposed_policy_sha256':file_sha256(destination/'active_policy.json'),
        'protocol_sha256':file_sha256(destination/'protocol.json'),
        'outputs_sha256':{p.name:file_sha256(p) for p in destination.iterdir() if p.is_file() and p.name!='protocol.json'}}
    save_json(destination/'verification.json',verification)
    print(metrics.loc[metrics['sample'].eq('evaluation')].to_string(index=False));print(json.dumps(policy,indent=2));print('Adaptive evidence:',destination,flush=True)
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--adaptive-from',type=Path)
    parser.add_argument('--install-policy',action='store_true',help='Explicitly install the proposed demo policy; default is evidence only.')
    args=parser.parse_args()
    if args.adaptive_from:run_adaptive(args.adaptive_from,install_policy=args.install_policy)
    else:run(install_policy=args.install_policy)
