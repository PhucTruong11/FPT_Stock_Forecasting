"""Verify the paired feature experiment and preserve all A/B outcomes."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import joblib
import numpy as np
import pandas as pd
from src.models.research_notes import write_note
from src.data.contracts import BASE_DIR,file_sha256,utc_now
from src.data.source_audit import guarded_hashes,save_json
from src.features.build_features import TARGET_COLUMN,TARGET_DATE_COLUMN
from src.features.candle_features import build_candle_dataset
from src.models.direction_selective import load_snapshot,phase_mask,decide,log_odds,select_margin,direction_metrics
from src.models.recency_bias import training_mask
from src.models.candle_ablation import combined_metrics,choose_case


def paired_effect(a,b,spec):
    """Forced-case paired bootstrap; selective rows deliberately have no causal-gain CI."""
    def stats(idx):
        y=a.actual_log_return.to_numpy()[idx]
        da=a.direction.to_numpy()[idx];db=b.direction.to_numpy()[idx]
        nonzero=y!=0;up=y>0;down=y<0
        result={'hit_change_pp':float(100*np.mean(db[nonzero]==np.sign(y[nonzero]))-100*np.mean(da[nonzero]==np.sign(y[nonzero])))}
        result['balanced_change_pp']=float(50*((np.mean(db[up]>0)+np.mean(db[down]<0))-(np.mean(da[up]>0)+np.mean(da[down]<0)))) if up.any() and down.any() else np.nan
        if a.predicted_log_return.notna().all():
            qa=a.predicted_log_return.to_numpy()[idx];qb=b.predicted_log_return.to_numpy()[idx]
            result['rmse_gain_pct']=float(100*(1-np.sqrt(np.mean((qb-y)**2))/np.sqrt(np.mean((qa-y)**2))))
        return result
    n=len(a);point=stats(np.arange(n));rng=np.random.default_rng(spec['seed']);draws=[]
    for _ in range(spec['resamples']):
        starts=rng.integers(0,n,size=int(np.ceil(n/spec['block_length'])))
        idx=((starts[:,None]+np.arange(spec['block_length']))%n).ravel()[:n];draws.append(stats(idx))
    for key in list(point):
        values=np.array([r[key] for r in draws]);values=values[np.isfinite(values)]
        lo,hi=np.quantile(values,[.025,.975]) if len(values) else (np.nan,np.nan)
        point[key+'_ci_low']=float(lo);point[key+'_ci_high']=float(hi)
    return point


def report(study):
    study=Path(study);p=json.loads((study/'protocol.json').read_text());cfg=p['config'];dcfg=p['direction_config']
    m=json.loads((study/'manifest.json').read_text());control=BASE_DIR/'reports/direction_selective'/cfg['control_study']
    assert file_sha256(control/'protocol.json')==p['control_protocol_sha256']
    assert file_sha256(control/'manifest.json')==p['control_manifest_sha256']
    parent,raw,_=load_snapshot(dcfg);assert file_sha256(parent/'data_snapshot.zip')==p['input_zip_sha256']
    _,labels,_,_=build_candle_dataset(raw)
    assert file_sha256(study/'feature_label_snapshot.csv.gz')==m['feature_label_snapshot_sha256']
    snapshot=pd.read_csv(study/'feature_label_snapshot.csv.gz',parse_dates=['Date',TARGET_DATE_COLUMN,'Previous_Eligible_Date'])
    assert np.array_equal(snapshot.Date,labels.Date)
    assert np.array_equal(snapshot[TARGET_DATE_COLUMN],labels[TARGET_DATE_COLUMN])
    for col in [*cfg['feature_cases']['A_returns6'],*cfg['feature_cases']['B_gap_body7'],'Open','Previous_Eligible_Close',TARGET_COLUMN]:
        assert np.allclose(snapshot[col],labels[col],rtol=1e-10,atol=1e-12)
    assert labels.Previous_Eligible_Date.lt(labels.Date).all()
    assert np.allclose(np.log(labels.Open/labels.Previous_Eligible_Close),labels.Gap_Log,atol=1e-12)
    assert np.allclose(np.log(labels.Close/labels.Open),labels.Body_Log,atol=1e-12)
    with zipfile.ZipFile(study/'source_snapshot.zip') as archive:
        for path,sha in p['source_sha256'].items():assert hashlib.sha256(archive.read(path)).hexdigest()==sha
    predictions=pd.read_csv(study/'predictions.csv.gz',parse_dates=['Date',TARGET_DATE_COLUMN])
    choices=pd.read_csv(study/'threshold_selection.csv');selections=pd.read_csv(study/'feature_selection.csv')
    models={};folds={f['id']:f for f in p['folds']};checks=[]
    for r in m['artifacts']:
        fold=folds[r['fold']];columns=cfg['feature_cases'][r['case']]
        if r['name']=='Ridge_recent':part=labels.loc[training_mask(labels,fold['validation_end'],dcfg['training_years'])]
        elif r['name']=='Logistic_sigmoid':part=labels.loc[phase_mask(labels,fold['calibration_start'],fold['calibration_end'])&labels[TARGET_COLUMN].ne(0)]
        else:part=labels.loc[training_mask(labels,fold['train_end'],dcfg['training_years'])&labels[TARGET_COLUMN].ne(0)]
        assert r['dates']==part[['Date',TARGET_DATE_COLUMN]].astype(str).to_dict('records')
        assert part[TARGET_DATE_COLUMN].max()<pd.Timestamp(fold['evaluation_start'])
        path=BASE_DIR/r['path'];assert file_sha256(path)==r['sha256'];model=joblib.load(path)
        if hasattr(model,'steps'):
            sx=model.steps[0][1]
            assert np.allclose(sx.mean_,part[columns].mean(),atol=1e-12,rtol=1e-10)
            assert np.allclose(sx.var_,part[columns].var(ddof=0),atol=1e-12,rtol=1e-10)
            if r['name'].startswith('Ridge'):
                ridge=model.steps[1][1];residual=part[TARGET_COLUMN].to_numpy()-model.predict(part[columns])
                assert np.allclose(sx.transform(part[columns]).T@residual,dcfg['ridge_alpha']*ridge.coef_,atol=1e-10,rtol=1e-7)
        models[(r['fold'],r['case'],r['name'])]=model
        checks.append({'fold':r['fold'],'case':r['case'],'model':r['name'],'n':len(part),'reused':r['reused'],'verified':True})
    reload_groups=0;score_rows=[];val_rows=[]
    for (fold,case,phase,strategy),g in predictions.groupby(['fold','case','phase','strategy'],sort=False):
        f=folds[fold];columns=cfg['feature_cases'][case]
        if phase=='evaluation':part=labels.loc[phase_mask(labels,f['evaluation_start'],f['evaluation_end'])]
        elif phase=='validation':part=labels.loc[phase_mask(labels,f['validation_start'],f['validation_end'])]
        else:part=labels.loc[phase_mask(labels,f['calibration_start'],f['calibration_end'])&labels[TARGET_COLUMN].ne(0)]
        assert np.array_equal(g.Date,part.Date) and np.array_equal(g[TARGET_DATE_COLUMN],part[TARGET_DATE_COLUMN])
        assert np.allclose(g.actual_log_return,part[TARGET_COLUMN],atol=1e-12,rtol=1e-10)
        assert np.allclose(g.Close,part.Close)
        if strategy.startswith('Ridge'):
            assert not (phase=='validation' and strategy=='Ridge_recent')
            q=models[(fold,case,strategy)].predict(part[columns]);assert np.allclose(q,g.predicted_log_return,atol=1e-12,rtol=1e-9)
            assert np.array_equal(np.sign(q).astype(int),g.direction)
        else:
            raw_p=models[(fold,case,'Logistic')].predict_proba(part[columns])[:,1].astype(float)
            calibrated=models[(fold,case,'Logistic_sigmoid')].predict_proba(log_odds(raw_p))[:,1]
            probability=raw_p if strategy=='Logistic_raw' else calibrated
            assert np.allclose(probability,g.probability_up,atol=1e-12,rtol=1e-9)
            margin=float(choices[(choices.fold==fold)&(choices.case==case)].margin.iloc[0]) if strategy=='Logistic_selective' else 0.
            assert np.array_equal(decide(probability,margin),g.direction)
            if phase=='calibration':assert np.allclose(raw_p,g.raw_probability,atol=1e-12,rtol=1e-9)
        if phase=='evaluation':score_rows.append({'fold':fold,'year':f['year'],'case':case,'strategy':strategy,**combined_metrics(g)})
        if phase=='validation' and strategy in ['Ridge_shared','Logistic_raw']:val_rows.append({'fold':fold,'case':case,'model':strategy,**combined_metrics(g)})
        reload_groups+=1
    # A must reproduce every prior selected strategy on every evaluation origin, not just aggregate scores.
    old_pred=pd.read_csv(control/'predictions.csv.gz',parse_dates=['Date',TARGET_DATE_COLUMN])
    a=predictions[(predictions.case=='A_returns6')&(predictions.phase=='evaluation')]
    old=old_pred[(old_pred.phase=='evaluation')&old_pred.strategy.isin(a.strategy.unique())]
    joined=a.merge(old,on=['fold','strategy','Date',TARGET_DATE_COLUMN],validate='one_to_one',suffixes=('_new','_old'))
    assert len(joined)==len(a)==len(old)
    for col in ['actual_log_return','predicted_log_return','probability_up','direction']:
        assert np.allclose(joined[col+'_new'],joined[col+'_old'],equal_nan=True,atol=1e-12,rtol=1e-9)
    saved_val=pd.read_csv(study/'validation_scores.csv')
    for actual in val_rows:
        saved=saved_val[(saved_val.fold==actual['fold'])&(saved_val.case==actual['case'])&(saved_val.model==actual['model'])].iloc[0]
        for col,value in actual.items():
            if col in ['fold','case','model']:continue
            assert (value is None and pd.isna(saved[col])) or np.isclose(value,saved[col],atol=1e-8,rtol=1e-8)
    for f in p['folds']:
        rows=[r for r in val_rows if r['fold']==f['id']]
        for strategy in ['Ridge_shared','Logistic_raw']:
            chosen=choose_case(rows,strategy);saved=selections[(selections.fold==f['id'])&(selections.model==strategy)].iloc[0]
            assert chosen==saved.selected_case
        for case in cfg['feature_cases']:
            val=predictions[(predictions.fold==f['id'])&(predictions.case==case)&(predictions.phase=='validation')&(predictions.strategy=='Logistic_calibrated')]
            margin,reason,_=select_margin(val.actual_log_return,val.probability_up,dcfg['selection'])
            choice=choices[(choices.fold==f['id'])&(choices.case==case)].iloc[0]
            assert np.isclose(margin,choice.margin) and reason==choice.reason
    assert guarded_hashes()==p['protected_before']==m['protected_after']
    metrics=pd.DataFrame(score_rows);metrics.to_csv(study/'metrics.csv',index=False)
    ev=predictions[predictions.phase=='evaluation'];effects=[]
    for (fold,strategy),g in ev.groupby(['fold','strategy']):
        a=g[g.case=='A_returns6'];b=g[g.case=='B_gap_body7']
        assert list(zip(a.Date,a[TARGET_DATE_COLUMN]))==list(zip(b.Date,b[TARGET_DATE_COLUMN]))
        row={'fold':fold,'year':folds[fold]['year'],'strategy':strategy,'n':len(a),
            'A_calls':int(a.direction.ne(0).sum()),'B_calls':int(b.direction.ne(0).sum())}
        if strategy!='Logistic_selective':row.update(paired_effect(a,b,cfg['paired_bootstrap']))
        effects.append(row)
    pd.DataFrame(effects).to_csv(study/'paired_effects.csv',index=False)
    selected=[]
    for _,r in selections.iterrows():
        chosen=metrics[(metrics.fold==r.fold)&(metrics.case==r.selected_case)&(metrics.strategy==r.model)].iloc[0]
        selected.append({'selected_case':r.selected_case,**chosen.to_dict()})
    pd.DataFrame(selected).to_csv(study/'validation_selected_evaluation.csv',index=False)
    primary=metrics[metrics.year.isin(cfg['primary_years'])].groupby(['case','strategy']).agg(
        mean_fold_balanced=('balanced_called_nonzero_pct','mean'),mean_fold_hit_nonzero=('hit_nonzero_called_pct','mean'),
        mean_fold_called_accuracy=('called_accuracy_all_pct','mean'),mean_fold_coverage=('coverage_pct','mean'),
        mean_fold_return_rmse=('return_rmse','mean'),worst_fold_return_rmse=('return_rmse','max')).reset_index()
    primary.to_csv(study/'primary_summary.csv',index=False)
    # June–October subset matches Notebook06 exactly. No CI transferred from the full cohort.
    prior=BASE_DIR/'reports/period_comparison/20261030T024048531444Z/predictions.csv.gz'
    keys=pd.read_csv(prior,parse_dates=['Date',TARGET_DATE_COLUMN])[['Date',TARGET_DATE_COLUMN]].drop_duplicates();slices=[]
    for (case,strategy),g in ev[ev.fold=='eval_2026'].groupby(['case','strategy']):
        h=g.merge(keys,on=['Date',TARGET_DATE_COLUMN],validate='one_to_one');assert len(h)==104
        slices.append({'scope':'jun_oct_common','case':case,'strategy':strategy,**combined_metrics(h)})
    pd.DataFrame(slices).to_csv(study/'jun_oct_metrics.csv',index=False)
    matched=[]
    for fold,g in ev.groupby('fold'):
        target=g[(g.case=='B_gap_body7')&(g.strategy=='Logistic_selective')];mask=target.direction.ne(0).to_numpy()
        if not mask.any():continue
        for case in cfg['feature_cases']:
            for strategy in ['Ridge_recent','Logistic_raw','Logistic_calibrated']:
                h=g[(g.case==case)&(g.strategy==strategy)].iloc[np.flatnonzero(mask)]
                matched.append({'fold':fold,'selector':'B_Logistic_selective','control':case+'_'+strategy,**direction_metrics(h.actual_log_return,h.direction)})
        h=target.iloc[np.flatnonzero(mask)]
        for control,d in [('B_Logistic_selective',h.direction),('Majority_train',h.majority_direction),('Always_down',np.full(len(h),-1)),('Always_up',np.ones(len(h)))]:
            matched.append({'fold':fold,'selector':'B_Logistic_selective','control':control,**direction_metrics(h.actual_log_return,d)})
    pd.DataFrame(matched).to_csv(study/'same_selected_days.csv',index=False)
    save_json(study/'verification.json',{'artifacts_reloaded':len(checks),'artifact_checks':checks,'prediction_groups_reloaded':reload_groups,
        'A_control_prediction_rows_matched':len(joined),'evaluation_metric_groups':len(metrics),'feature_choices_verified':len(selections),
        'threshold_choices_verified':len(choices),'gap_body_identity_verified':True,'source_zip_and_feature_snapshot_verified':True,
        'canonical_unchanged':True,'verification_refits':0})
    write_report(study,p,m,primary)
    save_json(study/'status.json',{'status':'complete','new_fits':m['new_fits'],'reused_artifacts':m['reused_artifacts'],'completed_at_utc':utc_now()})
    index=study.parent/'studies.json';items=json.loads(index.read_text()) if index.exists() else []
    items=[s for s in items if s['study_id']!=study.name];items.append({'study_id':study.name,'status':'complete'})
    save_json(index,items)


def write_report(study,p,m,primary):
    def fmt(v):return '—' if pd.isna(v) else f'{v:.6f}'
    rows=['| Case | Strategy | Mean-fold RMSE return | Mean-fold Hit nonzero % | Mean-fold balanced % | Mean coverage % |',
          '|---|---|---:|---:|---:|---:|']
    for _,r in primary.iterrows():rows.append(f'| {r.case} | {r.strategy} | {fmt(r.mean_fold_return_rmse)} | {r.mean_fold_hit_nonzero:.2f} | {r.mean_fold_balanced:.2f} | {r.mean_fold_coverage:.2f} |')
    text=f'''# FPT: A/B gap mở cửa và thân nến — {study.name}

## Giao thức trước fit
A giữ6 returns. B **thay** Log_Return bằng Gap_Log/Body_Log, thành7 feature. Gap=log(Open_t/Close_previous_eligible); Body=log(Close_t/Open_t). Cộng hai phần bằng Log_Return; B thêm một chiều để phân biệt chuyển động qua đêm/trong phiên, không giữ cả3 cột phụ thuộc tuyến tính. Chỉ dùng FPT, không chỉ số/mã khác hoặc target/horizon mới. Cả hai phần đã biết khi origin t đóng cửa, dùng dự đoán quan sát giao dịch kế tiếp.

Prev close lấy đúng chuỗi positive-volume/valid candle của feature builder cũ. Không lấy hôm qua theo lịch, không dùng Open của phiên cần đoán, không fill. Nếu gặp candle invalid thì giữ chính sách cũ (loại candle cho feature, quarantine nhãn chạm candle invalid); chưa được chứng nhận lịch sàn. Snapshot Vietcap khóa đến29/10/2026, không download hoặc sửa schema/canonical16feature.

{m['new_fits']} fit mới cho B, {m['reused_artifacts']} artifact A dùng lại sau kiểm tra hash, exact dates, scaler, forecast và versions. Cùng fold/labels/dates/hyperparameters với study direction trước: Ridge_shared và Logistic train nonzero đến30/6; Q3 sigmoid calibration; Q4 threshold validation; evaluation năm sau. Ridge_recent train cả window3y đến31/12 năm trước (gồm return0), giữ thêm làm đối chứng giá. A/B trong từng strategy có đúng cùng dòng train/cal/val/eval. Không đổi C, alpha, grid margin hoặc lớp đứng giá để tìm điểm đẹp.

## Primary2022–2024: các case cố định, không chọn bằng evaluation

{chr(10).join(rows)}

`feature_selection.csv`: Ridge_shared chọn case bằng RMSE validation; Logistic_raw chọn bằng balanced forced validation; tie chọn A. `validation_selected_evaluation.csv` chấm đúng lựa chọn đó. Ridge_recent không được chọn bằng in-sample validation vì nó đã train đến cuối năm; A/B của model này là contrast hồi cứu cố định, chưa phải champion selection. 2025/2026 tách riêng là diagnostic. Mọi giai đoạn đã xem, chưa phải holdout mới.

## Sai số, hướng và selective
Ridge có RMSE/MAE return và price MAE/MAPE cùng anchor Close Vietcap adjusted. Classifier chỉ có xác suất/hướng, không bịa giá dự báo từ p. Hit nonzero và accuracy trên mọi calls giữ riêng; actual0 vẫn trong evaluation, call vào actual0 tính sai. Abstain không được tính đúng nhãn đứng giá. Selective phải đọc coverage từng lớp/calls/neutral counts; không trừ bỏ ngày khó rồi gọi accuracy tăng toàn bộ.

`paired_effects.csv`: B−A Hit/balanced theo điểm%, positive tốt hơn; RMSE gain=100×(1−RMSE_B/RMSE_A), positive tốt hơn. CI95% mô tả từ paired block-bootstrap10 origin/2.000 resamples seed20261030, chỉ các forced strategies. Không gán CI cải thiện cho selective với hai masks khác nhau; `same_selected_days.csv` so control trên đúng ngày B selective giữ lại. CI không sửa hết adaptive research/multiple testing. Fixed alpha/C nhưng đổi cơ sở feature/scaling làm thay regularization hiệu dụng; không quy mọi chênh lệch thành tác động nhân quả của gap.

`jun_oct_metrics.csv` là cùng104 origin/target của Notebook06; không chuyển CI full2026 sang subset. `feature_label_snapshot.csv.gz` giữ raw context, previous date/close, gap/body và nhãn để kiểm tra. `source_snapshot.zip`, `protocol.json`, `manifest.json` giữ nguồn trước fit, config/versions/hashes/dates. `verification.json` reload cả40 artifacts/100 forecast groups, kiểm tra identity/cutoff/scaler, tái lập A và lựa chọn; không fit khi verify.

Không tự promote hoặc sửa serving/API/Streamlit/ledger shadow. Weights nghiên cứu dưới models/research bị gitignore. Phần presentation/decision được viết sau khi xem kết quả, không thay giao thức. Chưa commit/push.
'''
    write_note(study / 'REPORT.md', text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);report(p.parse_args().study)
