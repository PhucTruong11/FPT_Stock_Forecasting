"""Reload artifacts, audit temporal constraints, and report recency/bias contrasts."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.models.research_notes import write_note
from src.data.contracts import BASE_DIR,file_sha256,utc_now
from src.data.source_audit import save_json,guarded_hashes
from src.features.build_features import build_dataset,TARGET_COLUMN,TARGET_DATE_COLUMN
from src.models.evaluation_2026 import aligned_cohort,price_metrics,block_bootstrap
from src.models.robustness import score_direction
from src.models.source_comparison import REFERENCE
from src.models.recency_bias import causal_calibration,training_mask


def metrics(truth,pred,close):
    return {**score_direction(np.asarray(truth),np.asarray(pred)),**price_metrics(truth,pred,close)}


def report(study):
    study=Path(study);proto=json.loads((study/'protocol.json').read_text());cfg=proto['config']
    manifest=json.loads((study/'manifest.json').read_text());evaluation=BASE_DIR/'reports/evaluation_2026'/cfg['evaluation_study']
    assert file_sha256(study/'protocol.json')==manifest['protocol_sha256']
    assert file_sha256(evaluation/'protocol.json')==proto['evaluation_protocol_sha256']
    assert file_sha256(evaluation/'data_snapshot.zip')==proto['evaluation_data_sha256']
    with zipfile.ZipFile(study/'source_snapshot.zip') as archive:
        for p,sha in proto['source_sha256'].items():assert hashlib.sha256(archive.read(p)).hexdigest()==sha
    with zipfile.ZipFile(evaluation/'data_snapshot.zip') as archive:raw=pd.read_csv(io.BytesIO(archive.read('vci.csv')),parse_dates=['Date'])
    features,labels,_,_=build_dataset(raw)
    ep=json.loads((evaluation/'protocol.json').read_text())
    cohort=aligned_cohort({'vci':(features,labels)},ep['config']['origin_start'],ep['config']['end_exclusive'])
    cols=proto['feature_columns'];X=features.iloc[cohort.position_vci.to_numpy()][cols]
    pred=pd.read_csv(study/'predictions.csv.gz',parse_dates=['Date',TARGET_DATE_COLUMN])
    table=pd.read_csv(study/'metrics.csv');full=pred[pred.cohort=='vci_full'].copy()
    registry={r['path']:r for r in manifest['models']};models={};train_checks=[]
    for p,r in registry.items():
        assert file_sha256(BASE_DIR/p)==manifest['artifact_sha256'][p]
        dates=pd.DataFrame(r['training_dates']);dates['Date']=pd.to_datetime(dates.Date);dates[TARGET_DATE_COLUMN]=pd.to_datetime(dates[TARGET_DATE_COLUMN])
        part=labels.merge(dates,on=['Date',TARGET_DATE_COLUMN],validate='one_to_one')
        assert len(part)==r['train_n'] and part[TARGET_DATE_COLUMN].le(r['training_cutoff']).all()
        assert part.Date.ge(pd.Timestamp(r['training_cutoff'])-pd.DateOffset(years=cfg['training_window_years'])+pd.Timedelta(days=1)).all()
        signature=hashlib.sha256(pd.util.hash_pandas_object(part[['Date',TARGET_DATE_COLUMN,*cols,TARGET_COLUMN]],index=False).values.tobytes()).hexdigest()
        assert signature==r['signature']
        # Check fitted scaler statistics and closed-form Ridge normal equations; no new fit.
        model=joblib.load(BASE_DIR/p);scaler=model.steps[0][1];ridge=model.steps[1][1]
        assert np.allclose(scaler.mean_,part[cols].mean(),rtol=1e-10,atol=1e-12)
        assert np.allclose(scaler.var_,part[cols].var(ddof=0),rtol=1e-10,atol=1e-12)
        z=scaler.transform(part[cols]);y=part[TARGET_COLUMN].to_numpy();residual=y-model.predict(part[cols])
        assert np.allclose(z.T@residual,cfg['alpha']*ridge.coef_,rtol=1e-7,atol=1e-10)
        assert abs(residual.mean())<1e-10
        models[r['id']]=model;train_checks.append({'id':r['id'],'train_n':len(part),'target_cutoff_verified':True,'normal_equations_verified':True})
    frozen=proto['frozen_control'];assert file_sha256(BASE_DIR/frozen['file'])==proto['frozen_control_sha256']
    raw_predictions={'frozen_2023':joblib.load(BASE_DIR/frozen['file']).predict(X)}
    for name in ['rebuilt_2023_same_rows','refit_2023_native','refit_2025_native']:raw_predictions[name]=models[name].predict(X)
    assignments=pd.read_csv(study/'monthly_model_assignments.csv',parse_dates=['Date',TARGET_DATE_COLUMN,'training_cutoff','train_target_end'])
    assert len(assignments)==len(cohort) and not assignments.Date.duplicated().any()
    monthly=np.empty(len(cohort))
    for path,g in assignments.groupby('model_path'):
        r=registry[path];assert g.train_target_end.le(g.training_cutoff).all() and g.training_cutoff.lt(g.Date).all()
        expected=g.Date.dt.to_period('M').dt.start_time-pd.Timedelta(days=1)
        assert g.training_cutoff.eq(expected).all()
        mask=cohort.Date.isin(g.Date).to_numpy();monthly[mask]=models[r['id']].predict(X.loc[mask])
    raw_predictions['monthly_refit_native']=monthly
    for name,q in raw_predictions.items():
        saved=full[full.strategy==name].sort_values('Date')
        assert np.array_equal(saved.Date.to_numpy(),cohort.Date.to_numpy())
        assert np.allclose(q,saved.predicted_log_return,rtol=1e-10,atol=1e-12)
    old_predictions=pd.read_csv(evaluation/'predictions.csv.gz',parse_dates=['Date',TARGET_DATE_COLUMN])
    old=old_predictions[(old_predictions.model=='Ridge_research_new')&(old_predictions.cohort=='vci_full')]
    parity=full[full.strategy=='frozen_2023'].merge(old,on=['Date',TARGET_DATE_COLUMN],suffixes=('_new','_old'))
    assert len(parity)==len(cohort) and np.allclose(parity.predicted_log_return_new,parity.predicted_log_return_old,atol=1e-12,rtol=1e-10)
    for base in ['frozen_2023','refit_2025_native','monthly_refit_native']:
        frame=cohort[['Date',TARGET_DATE_COLUMN]].copy();frame[REFERENCE]=cohort.return_vci.to_numpy();frame['raw_prediction']=raw_predictions[base]
        calibrated=causal_calibration(frame,cfg['bias'],cfg['interval'])
        saved=full[full.strategy==base+'_bias60'].sort_values('Date')
        for col in ['bias_offset','calibration_n','lower_log_return','upper_log_return','interval_n']:
            assert np.allclose(calibrated[col],saved[col],equal_nan=True,rtol=1e-10,atol=1e-12)
        assert np.allclose(calibrated.prediction,saved.predicted_log_return,rtol=1e-10,atol=1e-12)
        matured=pd.to_datetime(calibrated.last_calibration_target)
        assert (matured.isna()|matured.le(calibrated.Date)).all()
    # Independent reconstruction of causal baselines.
    train2025=labels.loc[training_mask(labels,cfg['static_refresh_end'],cfg['training_window_years'])]
    expected_baselines={'Zero-Return':np.zeros(len(cohort)),'TrainMean_2025':np.full(len(cohort),train2025[TARGET_COLUMN].mean()),
        'RollingMean60':np.array([labels.loc[labels[TARGET_DATE_COLUMN].le(d)].tail(cfg['bias']['window'])[TARGET_COLUMN].mean() for d in cohort.Date])}
    for name,q in expected_baselines.items():assert np.allclose(q,full[full.strategy==name].sort_values('Date').predicted_log_return,atol=1e-12,rtol=1e-10)
    checks=[]
    for (strategy,scope),g in pred.groupby(['strategy','cohort']):
        truth=g[REFERENCE].to_numpy();q=g.predicted_log_return.to_numpy();row=table[(table.strategy==strategy)&(table.cohort==scope)].iloc[0]
        for key,value in metrics(truth,q,g.origin_close_adjusted).items():
            assert (pd.isna(value) and pd.isna(row[key])) or np.isclose(value,row[key],rtol=1e-9,atol=1e-10),(strategy,key,value,row[key])
        assert np.allclose(g.predicted_close_adjusted,g.origin_close_adjusted*np.exp(q),rtol=1e-12,atol=1e-8)
        assert np.allclose(g.actual_close_adjusted,g.origin_close_adjusted*np.exp(truth),rtol=1e-12,atol=1e-8)
        valid=g.lower_log_return.notna();assert valid.sum()==row.interval_n
        if valid.any():
            covered=g.loc[valid,REFERENCE].between(g.loc[valid,'lower_log_return'],g.loc[valid,'upper_log_return'])
            assert np.isclose(100*covered.mean(),row.interval_coverage_pct)
            assert np.isclose((g.loc[valid,'upper_log_return']-g.loc[valid,'lower_log_return']).mean(),row.mean_interval_width_log_return)
        checks.append({'strategy':strategy,'cohort':scope,'n':len(g),'verified':True})
    assert guarded_hashes()==proto['protected_before']==manifest['protected_after']
    warmup=[];monthly_rows=[];bias_rows=[];uncertainty=[]
    for name,g in full.groupby('strategy'):
        g=g.sort_values('Date');truth=g[REFERENCE].to_numpy();q=g.predicted_log_return.to_numpy()
        bias_rows.append({'strategy':name,'mean_prediction':float(q.mean()),'mean_actual':float(truth.mean()),'mean_error':float(np.mean(q-truth)),
                         'predicted_up_pct':float(100*np.mean(q>0))})
        for month,h in g.groupby(g[TARGET_DATE_COLUMN].dt.to_period('M')):
            y=h[REFERENCE].to_numpy();p=h.predicted_log_return.to_numpy()
            monthly_rows.append({'strategy':name,'target_month':str(month),**score_direction(y,p),
                                 'mse_skill_vs_zero':float(1-np.mean((y-p)**2)/np.mean(y*y))})
        uncertainty.append({'strategy':name,**block_bootstrap(truth,q,{'seed':20261030,'block_length':10,'resamples':2000})})
        if name.endswith('_bias60'):
            base=name.removesuffix('_bias60');raw_group=full[full.strategy==base].sort_values('Date')
            for scope,valid in [('bias_ready',g.calibration_n.ge(cfg['bias']['minimum_matured'])),('interval_ready',g.lower_log_return.notna())]:
                for label,h in [(name,g.loc[valid.to_numpy()]),(base,raw_group.loc[valid.to_numpy()])]:
                    warmup.append({'scope':scope,'comparison':name,'strategy':label,**metrics(h[REFERENCE],h.predicted_log_return,h.origin_close_adjusted)})
    pd.DataFrame(warmup).to_csv(study/'matched_warmup_comparison.csv',index=False)
    months=pd.DataFrame(monthly_rows);months.to_csv(study/'monthly_metrics.csv',index=False)
    bias=pd.DataFrame(bias_rows);bias.to_csv(study/'bias_diagnostics.csv',index=False)
    pd.DataFrame(uncertainty).to_csv(study/'uncertainty_descriptive.csv',index=False)
    verification={'verified_at_utc':utc_now(),'metric_groups':checks,'new_training_checks':train_checks,
        'prediction_groups_reloaded':len(raw_predictions),'calibration_groups_recomputed':3,'frozen_control_matches_previous_evaluation':True,
        'source_and_artifact_hashes_verified':True,'all_metric_fields_verified':True,'canonical_files_unchanged':True,'verification_refits':0}
    save_json(study/'verification.json',verification)
    primary=table[table.cohort=='vci_full'].copy();zero=float(primary.loc[primary.strategy=='Zero-Return','rmse'].iloc[0])
    primary['mse_skill_vs_zero']=1-(primary.rmse/zero)**2
    primary.to_csv(study/'primary_summary.csv',index=False)
    _plots(study,primary,months,full)
    rows=['| Strategy | N | RMSE return | MAE giá adj. (VND) | Hit (%) | Balanced (%) | MSE skill vs Zero (%) |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for _,r in primary.sort_values('rmse').iterrows():
        rows.append(f'| {r.strategy} | {int(r.n)} | {r.rmse:.7f} | {r.price_mae_adjusted_vnd:.2f} | {r.hit_rate_pct:.2f} | {r.balanced_accuracy_pct:.2f} | {100*r.mse_skill_vs_zero:+.2f} |')
    intervals=primary[primary.interval_n.gt(0)]
    report_text=f'''# FPT: recency và hiệu chỉnh bias — {study.name}

## Câu hỏi và giao thức
So cùng 6 return features, StandardScaler fit riêng train, Ridge alpha=100 và window 3 năm. Giao thức được ghi **trước fit**; không chọn alpha/window bằng điểm 2026. Tất cả nhãn 2026 đã xem ở study trước: đây là **chẩn đoán hồi cứu thích nghi**, không phải holdout độc lập mới hay forecast live. Nguồn Vietcap adjusted snapshot đến {raw.Date.max().date()}, tái dùng đúng ZIP/hash; không tải lại hoặc nối hai mặt bằng giá.

`frozen_2023` là artifact cũ, train 693 cặp common origin/target đến 2023. `rebuilt_2023_same_rows` giữ đúng 693 ngày nhưng tái tạo X/y từ snapshot mới: kiểm tra ảnh hưởng revision. `refit_2023_native` bổ sung coverage native cùng cutoff. `refit_2025_native` chuyển window đến hết 2025. `monthly_refit_native` mô phỏng refit vào đầu tháng, chỉ dùng target ≤ cuối tháng trước; không phải đã chạy live vào tháng đó. Giá lịch sử adjusted hiện tại chưa được chứng nhận point-in-time.

`*_bias60` cộng mean của tối đa 60 residual **đã có nhãn trước/sát origin sau đóng cửa**, cần ít nhất 20; không trừ mean lỗi cả năm rồi sửa ngược quá khứ. 20 origin đầu giữ raw forecast. `RollingMean60` là baseline mean return của 60 nhãn đã biết, warm-start từ 2025. 2026 prefix dùng để cập nhật theo quy tắc định trước; các ngày sau vẫn chỉ được gọi là mô phỏng theo thời gian.

## Kết quả chính — cùng {len(cohort)} cặp Vietcap 2026
Origin {cohort.Date.min().date()}–{cohort.Date.max().date()}, target đến {cohort[TARGET_DATE_COLUMN].max().date()}. Một quan sát tiếp theo, không dự báo toàn bộ đường giá năm. Cohort phụ 200 cặp khớp study trước ở `metrics.csv`; không chọn cohort có điểm đẹp.

{chr(10).join(rows)}

Hit chỉ tính nhãn thực tế khác 0; Zero Return là abstention nên Hit/Balanced N/A. MAPE không phải xác suất dự báo đúng. Price MAE cùng anchor adjusted Vietcap; không phải giá chưa điều chỉnh hiển thị ngày lịch sử.

## Bias, độ ổn định và bất định
`matched_warmup_comparison.csv` so raw/bias trên cùng tập ngày bias-ready (≥20 residual) và interval-ready (≥60), tránh đổi mẫu tạo cải thiện giả. `monthly_metrics.csv` lưu N/RMSE/skill từng tháng; không chọn model theo tháng thắng nhiều nhất. `bias_diagnostics.csv` ghi mean prediction, actual và error. `uncertainty_descriptive.csv` dùng block bootstrap 10 origin, 2.000 resamples, seed 20261030: CI mô tả phụ thuộc một giai đoạn, không hiệu chỉnh cho việc xem và thử nhiều phương án.

Khoảng dự báo danh nghĩa 80% = quantile 10–90% của tối đa 60 lỗi **calibrated trước đó**; đủ 60 mới phát interval. Đây là residual interval thực nghiệm, **không bảo đảm coverage**, không confidence interval của mean forecast. N/coverage thực tế:
'''
    for _,r in intervals.iterrows():report_text+=f'- {r.strategy}: N={int(r.interval_n)}, coverage={r.interval_coverage_pct:.2f}%, mean width log-return={r.mean_interval_width_log_return:.5f}.\n'
    report_text+='''
## Artifact và kiểm tra
`protocol.json` + `source_snapshot.zip`: quy tắc, source/package versions và hash trước fit. `manifest.json`: ngày train từng dòng, cutoff, hash model và số fit mới. `verification.json`: reload model, kiểm tra thống kê scaler, normal equations Ridge, nhãn/cutoff, forecast, calibration và mọi metric; không fit lại khi verify. Model production/canonical input/pointer giữ nguyên. Weights nghiên cứu ở `models/research/recency-bias_<study_id>/` bị gitignore: khi tái lập trên máy khác cần chuyển đúng artifact kèm hash hoặc chạy study mới, không giả rằng ZIP source đã chứa weights.

## Shadow phía trước
`shadow_bundle.json` refit riêng đến target đã chín mới nhất, kèm calibration snapshot. Bundle **cố định** đến khi chủ động rebuild; không tự refit mỗi tháng hoặc tự cập nhật bias. Đây là ứng viên để ghi forecast tương lai, không promote serving, không tối ưu trên 2026 rồi tuyên bố lợi thế tương lai.

Sổ `reports/forward_shadow/forecasts.jsonl` chỉ append forecast/outcome với hash-chain và ảnh chụp input. Forecast phải được phát thật trước khi có target, không backdate. Chỉ `issued_local_date < actual_target_date` mới đủ điều kiện báo điểm full-session; phát sáng cùng ngày target chỉ là pre-close shadow và bị loại khỏi nhóm này. Hash-chain local không chứng nhận thời gian bên thứ ba; single writer. Chưa có outcome chín thì không có kết luận cải thiện forward. Hướng dẫn: `ARCHITECTURE.md#forward-ledger`.
'''
    write_note(study / 'REPORT.md', report_text)
    save_json(study/'status.json',{'status':'complete','new_fits':manifest['new_fits'],'completed_at_utc':utc_now()})
    index=study.parent/'studies.json';items=json.loads(index.read_text()) if index.exists() else []
    items=[s for s in items if s['study_id']!=study.name];items.append({'study_id':study.name,'status':'complete','new_fits':manifest['new_fits']})
    save_json(index,items)
    return primary


def _plots(study,primary,monthly,pred):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=study/'plots';plots.mkdir(exist_ok=True)
    names=['frozen_2023','refit_2023_native','refit_2025_native','monthly_refit_native','monthly_refit_native_bias60','RollingMean60','Zero-Return']
    p=primary.set_index('strategy').loc[names]
    fig,ax=plt.subplots(figsize=(11,4.5));values=100*p.mse_skill_vs_zero
    ax.barh(p.index,values,color=['#2474a6' if x>=0 else '#b24c4c' for x in values]);ax.axvline(0,color='black',lw=1)
    ax.set_xlabel('MSE skill vs Zero Return (%)');ax.set_title('2026 retrospective · same 202 Vietcap pairs · positive is better')
    fig.tight_layout();fig.savefig(plots/'recency_skill.png',dpi=140);plt.close(fig)
    names=['frozen_2023','refit_2025_native','monthly_refit_native','monthly_refit_native_bias60','Zero-Return']
    grid=monthly[monthly.strategy.isin(names)].pivot(index='target_month',columns='strategy',values='mse_skill_vs_zero').reindex(columns=names)*100
    fig,ax=plt.subplots(figsize=(11,4.5));grid.plot(marker='o',ax=ax);ax.axhline(0,color='black',lw=1)
    ax.set_ylabel('MSE skill vs Zero Return (%)');ax.set_xlabel('Realized target month');ax.set_title('Stability by month · 15–23 observations per month')
    ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(plots/'monthly_skill.png',dpi=140);plt.close(fig)
    g=pred[pred.strategy=='monthly_refit_native_bias60'].sort_values('Date');valid=g.lower_log_return.notna()
    fig,ax=plt.subplots(figsize=(11,4.5));ax.plot(g[TARGET_DATE_COLUMN],100*g[REFERENCE],color='#444',lw=.8,label='Actual')
    ax.plot(g[TARGET_DATE_COLUMN],100*g.predicted_log_return,color='#2474a6',label='Calibrated prediction')
    ax.fill_between(g.loc[valid,TARGET_DATE_COLUMN],100*g.loc[valid,'lower_log_return'],100*g.loc[valid,'upper_log_return'],alpha=.2,color='#2474a6',label='80% empirical residual interval (after 60 labels)')
    ax.set_ylabel('Log-return (%)');ax.set_title('Past-label calibration · interval has no coverage guarantee');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(plots/'empirical_interval.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('study',type=Path);report(parser.parse_args().study)
