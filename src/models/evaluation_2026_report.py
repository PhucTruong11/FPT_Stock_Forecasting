"""Verify and explain a frozen, non-refitted 2026 temporal evaluation."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.contracts import BASE_DIR,file_sha256
from src.data.source_audit import save_json,guarded_hashes
from src.models.evaluation_2026 import aligned_cohort,price_metrics,block_bootstrap
from src.models.robustness import score_direction
from src.models.source_comparison import load_inputs,source_part
from src.models.window_comparison import window_masks,SOURCE
from src.features.build_features import build_dataset,TARGET_DATE_COLUMN


def psi_from_training(train,current,bins=10):
    edges=np.unique(np.quantile(train,np.linspace(0,1,bins+1)))
    edges=np.r_[-np.inf,edges[1:-1],np.inf]
    a=np.histogram(train,bins=edges)[0].astype(float);b=np.histogram(current,bins=edges)[0].astype(float)
    a=(a+1e-6)/(a.sum()+1e-6*len(a));b=(b+1e-6)/(b.sum()+1e-6*len(b))
    return float(np.sum((b-a)*np.log(b/a)))


def report(study):
    proto=json.loads((study/'protocol.json').read_text());config=proto['config']
    acquisition=json.loads((study/'acquisition.json').read_text())
    assert file_sha256(study/'data_snapshot.zip')==acquisition['zip_sha256']
    with zipfile.ZipFile(study/'model_snapshot.zip') as archive:
        for path,sha in proto['model_sha256'].items():assert hashlib.sha256(archive.read(path)).hexdigest()==sha
    with zipfile.ZipFile(study/'source_snapshot.zip') as archive:
        for path,sha in proto['source_sha256'].items():assert hashlib.sha256(archive.read(path)).hexdigest()==sha
        assert hashlib.sha256(archive.read('config.json')).hexdigest()==proto['config_sha256']
    raw={}
    with zipfile.ZipFile(study/'data_snapshot.zip') as archive:
        for path,sha in acquisition['files'].items():assert hashlib.sha256(archive.read(path)).hexdigest()==sha
        for source,name in [('yahoo','yahoo/FPT.VN_raw.csv'),('vci','vci.csv'),('kbs','kbs.csv')]:
            raw[source]=pd.read_csv(io.BytesIO(archive.read(name)),parse_dates=['Date'])
    datasets={source:build_dataset(frame)[:2] for source,frame in raw.items()}
    cohorts={'common':aligned_cohort({s:datasets[s] for s in ['yahoo','vci']},config['origin_start'],config['end_exclusive']),
             'vci_full':aligned_cohort({'vci':datasets['vci']},config['origin_start'],config['end_exclusive'])}
    pred=pd.read_csv(study/'predictions.csv.gz',parse_dates=['Date',TARGET_DATE_COLUMN]);metrics=pd.read_csv(study/'metrics.csv')
    month_rows=[];checks=[]
    for (name,cohort_name),p in pred.groupby(['model','cohort'],sort=False):
        expected=cohorts[cohort_name]
        assert np.array_equal(p.Date.to_numpy(),expected.Date.to_numpy()) and np.array_equal(p[TARGET_DATE_COLUMN].to_numpy(),expected[TARGET_DATE_COLUMN].to_numpy())
        assert np.allclose(p.reference_log_return,expected.return_vci,atol=1e-12)
        actual=raw['vci'].set_index('Date').Close.reindex(p[TARGET_DATE_COLUMN]).to_numpy()
        assert np.allclose(p.actual_close_adjusted,actual,atol=.02,rtol=1e-6)
        assert np.allclose(p.predicted_close_adjusted,p.reference_origin_close*np.exp(p.predicted_log_return),atol=1e-6)
        y,q=p.reference_log_return.to_numpy(),p.predicted_log_return.to_numpy();nz=y!=0;issued=nz&(q!=0)
        mse=np.mean((y-q)**2);actual_price=p.actual_close_adjusted.to_numpy();forecast=p.predicted_close_adjusted.to_numpy()
        computed={'n':len(y),'rmse':np.sqrt(mse),'mae':np.mean(abs(y-q)),'nonzero_actual_n':nz.sum(),'neutral_actual_n':(~nz).sum(),
            'hit_rate_pct':100*np.mean(np.sign(y[nz])==np.sign(q[nz])) if issued.any() else np.nan,
            'balanced_accuracy_pct':50*((q[y>0]>0).mean()+(q[y<0]<0).mean()) if np.any(q!=0) else np.nan,
            'direction_coverage_pct':100*issued.sum()/nz.sum(),'mse_skill_vs_zero':1-mse/np.mean(y*y),
            'price_rmse_adjusted_vnd':np.sqrt(np.mean((forecast-actual_price)**2)),'price_mae_adjusted_vnd':np.mean(abs(forecast-actual_price)),
            'price_mape_pct':100*np.mean(abs(forecast/actual_price-1)),'mean_bias_log_return':np.mean(q-y),'prediction_std':np.std(q)}
        saved=metrics[(metrics.model==name)&(metrics.cohort==cohort_name)];assert len(saved)==1;saved=saved.iloc[0]
        for field,value in computed.items():assert (pd.isna(value) and pd.isna(saved[field])) or np.isclose(value,saved[field],atol=1e-9,rtol=1e-10),(name,field)
        ci=block_bootstrap(y,q,config['bootstrap'])
        for field,value in ci.items():assert np.isclose(value,saved[field],atol=1e-12)
        for month,part in p.groupby(p[TARGET_DATE_COLUMN].dt.to_period('M')):
            month_rows.append({'model':name,'cohort':cohort_name,'target_month':str(month),
                **score_direction(part.reference_log_return,part.predicted_log_return),
                **price_metrics(part.reference_log_return,part.predicted_log_return,part.reference_origin_close)})
        checks.append({'model':name,'cohort':cohort_name,'n':len(y),'metrics_verified':True})
    pd.DataFrame(month_rows).to_csv(study/'monthly_metrics.csv',index=False,lineterminator='\n')
    # Independently reload every selected artifact from its immutable frozen snapshot.
    import joblib
    import tensorflow as tf
    from xgboost import XGBRegressor
    from src.models.experiment import make_sequences
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    # TensorFlow may already be configured by the evaluator in the same process.
    for spec in proto['models']:
        if spec.get('file'):
            if spec['kind']=='lstm':model=tf.keras.models.load_model(BASE_DIR/spec['file'],compile=False);scalers=joblib.load(BASE_DIR/spec['scalers'])
            elif spec['kind']=='xgboost':model=XGBRegressor();model.load_model(BASE_DIR/spec['file'])
            else:model=joblib.load(BASE_DIR/spec['file'])
            assert file_sha256(BASE_DIR/spec['file'])==proto['model_sha256'][spec['file']]
        for cohort_name,cohort in cohorts.items():
            if cohort_name=='vci_full' and spec['source']=='yahoo':continue
            features=datasets[spec['source']][0];positions=cohort['position_'+spec['source']].to_numpy()
            if spec['kind']=='constant':q=np.full(len(cohort),spec['value'])
            elif spec['kind']=='lstm':
                values=scalers['features'].transform(features[spec['columns']])
                q=scalers['target'].inverse_transform(model(make_sequences(values,positions,spec['lookback']),training=False).numpy()).ravel()
            else:q=model.predict(features.iloc[positions][spec['columns']])
            expected=pred[(pred.model==spec['name'])&(pred.cohort==cohort_name)]
            assert np.allclose(q,expected.predicted_log_return,atol=1e-8,rtol=1e-5)
    assert guarded_hashes()==proto['protected_before']
    save_json(study/'verification.json',{'immutable_model_input_source_archives_verified':True,'metric_groups':len(checks),
        'all_metric_fields_verified':True,'prediction_groups_reloaded':len(checks),'refits':0,'model_choices_fixed_before_2026_download':True,
        'canonical_files_unchanged':True,'checks':checks})
    # Descriptive input shift for the research Ridge, using old training-only quantiles.
    research_proto=json.loads((BASE_DIR/'reports/ridge_refinement'/config['research_study']/'protocol.json').read_text())
    old=BASE_DIR/'reports/source_comparison'/config['source_study'] if 'source_study' in config else BASE_DIR/'reports/source_comparison/20261029T142732805710Z'
    olddata,oldcohort=load_inputs(old);fold=next(f for f in research_proto['fixed_folds'] if f['id']==config['research_selection_fold'])
    oldmask=window_masks(oldcohort,fold,3)['train'];train=source_part(oldcohort.loc[oldmask].reset_index(drop=True),SOURCE,olddata[SOURCE][0])
    current=datasets['vci'][0].iloc[cohorts['vci_full'].position_vci.to_numpy()]
    shifts=[]
    for column in research_proto['feature_columns']:
        a,b=train[column].to_numpy(),current[column].to_numpy();lo,hi=np.quantile(a,[.01,.99])
        shifts.append({'feature':column,'train_mean':float(a.mean()),'current_mean':float(b.mean()),'train_std':float(a.std()),
            'mean_shift_in_train_std':float((b.mean()-a.mean())/a.std()),'current_outside_train_1_99pct_pct':float(100*np.mean((b<lo)|(b>hi))),
            'psi_descriptive':psi_from_training(a,b)})
    pd.DataFrame(shifts).to_csv(study/'input_shift.csv',index=False,lineterminator='\n')
    plots(study,pred,metrics)
    common=metrics[metrics.cohort.eq('common')];cohort=cohorts['common'];truth=cohort.return_vci
    nonzero=int(truth.ne(0).sum());up=int(truth.gt(0).sum());down=int(truth.lt(0).sum())
    rows=['| Model | Train đến | Return RMSE | Return MAE | Giá MAE (VND adj.) | MAPE (%) | Hit (%) | Balanced (%) | MSE skill vs Zero |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for _,r in common.sort_values('rmse').iterrows():
        spec=next(s for s in proto['models'] if s['name']==r.model)
        hit='N/A' if pd.isna(r.hit_rate_pct) else f'{r.hit_rate_pct:.2f}'
        balanced='N/A' if pd.isna(r.balanced_accuracy_pct) else f'{r.balanced_accuracy_pct:.2f}'
        rows.append(f"| {r.model} | {spec['train_end'] or 'baseline'} | {r.rmse:.7f} | {r.mae:.7f} | {r.price_mae_adjusted_vnd:.1f} | {r.price_mape_pct:.2f} | {hit} | {balanced} | {100*r.mse_skill_vs_zero:+.2f}% |")
    text=f'''# Đánh giá model đóng băng trên 2026 — {study.name}

## Phạm vi
Weights, scaler, cấu hình và baseline constants được khóa lúc **{proto['frozen_before_2026_download_utc']}**, trước request dữ liệu 2026. Không refit, không chọn model theo điểm 2026. Dùng origin từ {config['origin_start']} và target trước {config['end_exclusive']}; chỉ dữ liệu phiên đã kết thúc tới 29/10/2026. Cohort chung có **{len(cohort)}** origin–target pairs; từ {cohort.Date.min().date()} tới {cohort.Date.max().date()}, target cuối {cohort.Target_Date_1d.max().date()}.

Đây là **temporal out-of-sample được mở mới trong workflow nhưng đánh giá hồi cứu**, không phải các forecast đã ghi trước khi thị trường diễn ra. Model được train/đánh giá các giai đoạn cũ vào tháng 10/2026; source adjusted history hiện tại không được chứng nhận point-in-time. Sau lượt này, khoảng 2026 này đã được xem và không được gọi là holdout chưa mở cho các lần tune tiếp.

Model dự báo một quan sát dương-volume kế tiếp, sau khi có OHLCV của origin; rolling context được cập nhật bằng giá đã quan sát tại mỗi origin. Không phải dự báo cả đường giá năm 2026 từ cuối 2025, cũng không chứng nhận phiên HOSE kế tiếp.

## So sánh cùng ngày và cùng nhãn Vietcap

{chr(10).join(rows)}

MAE/RMSE giá dùng **cùng Close Vietcap tại origin** để quy đổi lợi suất mọi model, tránh lẫn khác thang adjusted price của Yahoo và Vietcap. Là VND đã điều chỉnh theo provider, **không phải giá khớp lệnh lịch sử**. MAPE là sai số phần trăm giá; không được gọi 100%-MAPE là xác suất dự báo đúng. Các trường native_predicted_close_adjusted giữ forecast theo nguồn gốc để tra cứu, không dùng xếp hạng giá giữa nguồn khác basis.

Mẫu số hit rate: {nonzero} nhãn khác 0, {len(cohort)-nonzero} nhãn trung tính; {up} tăng và {down} giảm. Luôn đoán tăng đạt {100*up/nonzero:.2f}%, luôn giảm {100*down/nonzero:.2f}%, balanced accuracy 50%. Zero Return không phát hướng nên hit N/A. Đọc cùng độ phủ hướng, balanced accuracy và baseline.

## Ổn định và độ bất định
monthly_metrics.csv báo từng tháng và số mẫu; không suy ra ổn định từ một hit rate gộp. CI 95% của MSE skill trong metrics.csv dùng circular moving-block bootstrap, block {config['bootstrap']['block_length']} origin, {config['bootstrap']['resamples']} lần, seed cố định. Đây là khoảng bất định mô tả với giả định phụ thuộc trong block, không bảo đảm thị trường tương lai và không điều chỉnh mọi bias phát triển.

![Lỗi theo tháng](monthly_errors.png)
![Giá quy đổi và lợi suất](price_and_returns.png)
![So với Zero Return](skill_intervals.png)

## Kiểm tra nguồn và giới hạn so sánh
data_quality.json và các file source_comparison lưu độ phủ, nến lỗi, missing sessions và bất đồng Yahoo/Vietcap/KBS. Full-Vietcap cohort trong metrics.csv bổ sung kết quả research/baseline trên toàn bộ ngày nguồn, tránh chỉ nhìn giao của các nguồn và che ngày thiếu của Yahoo.

LSTM/Ridge/XGBoost serving dùng Yahoo, 16 features, train tới 2022; Ridge research dùng Vietcap, 6 features, train 2021–2023, validation2024 chọn window3y alpha100. Vì khác nguồn/train cutoff/features, chênh lệch cross-family **không cô lập riêng thuật toán**. So alpha100 với alpha10 research mới là đối chứng gần về regularization. TrainMean_research là mean expanding train cũ, không lấy mean của năm 2026.

input_shift.csv so phân bố sáu inputs Ridge với training gốc bằng quantile/PSI mô tả; không đặt ngưỡng drift phổ quát hoặc gọi sai số thay đổi là “quên”. Model OHLCV thiếu tin tức/kết quả kinh doanh, thanh khoản và biến cố mới; chưa có mô phỏng phí/slippage/khớp lệnh hoặc bảo đảm lợi nhuận. Không tự thay API/Streamlit/model phục vụ, không commit/push.

## Bằng chứng
protocol.json, model_snapshot.zip, source_snapshot.zip khóa trước data; acquisition.json/data_snapshot.zip lưu raw sources và hash; predictions.csv.gz, metrics.csv, monthly_metrics.csv và verification.json giữ forecast/metric có thể kiểm tra lại. ../holdout_exposures.json (tại reports/) ghi khoảng thời gian đã mở. Xem DECISION.md cho diễn giải ưu/nhược và hướng cải thiện.
'''
    (study/'REPORT.md').write_text(text,encoding='utf-8',newline='\n')
    index=BASE_DIR/'reports/evaluation_2026/studies.json';items=json.loads(index.read_text()) if index.exists() else []
    items=[s for s in items if s['study_id']!=study.name]+[{'study_id':study.name,'status':'complete','common_n':len(cohort),'refits':0,'automatic_promotion':False}]
    save_json(index,items)


def plots(study,pred,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    monthly=pd.read_csv(study/'monthly_metrics.csv');names=['LSTM_serving','Ridge_research_new','Ridge_research_old','Zero-Return','TrainMean_research']
    p=monthly[monthly.cohort.eq('common')&monthly.model.isin(names)].pivot(index='target_month',columns='model',values='rmse')
    fig,ax=plt.subplots(figsize=(12,4));p.plot(ax=ax,marker='o');ax.set_ylabel('RMSE log-return');ax.set_xlabel('Tháng target 2026');ax.set_title('Frozen models · rolling one-step evaluation · tháng ít mẫu cần đọc thận trọng');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(study/'monthly_errors.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7))
    q=pred[(pred.cohort=='common')&(pred.model=='Ridge_research_new')]
    axes[0].plot(q.Target_Date_1d,q.actual_close_adjusted,label='Vietcap actual adj.');axes[0].plot(q.Target_Date_1d,q.predicted_close_adjusted,label='Ridge forecast, same anchor',alpha=.8)
    axes[0].set_ylabel('VND adjusted');axes[0].legend();axes[0].set_title('Giá gần nhau chưa chứng minh tốt hơn last-close baseline')
    axes[1].scatter(q.reference_log_return,q.predicted_log_return,s=12,alpha=.7);axes[1].axhline(0,color='black',lw=.7);axes[1].axvline(0,color='black',lw=.7)
    axes[1].set_xlabel('Lợi suất log thực tế');axes[1].set_ylabel('Lợi suất log dự báo');axes[1].set_title('Ridge mới · từng ngày trong cohort chung')
    fig.tight_layout();fig.savefig(study/'price_and_returns.png',dpi=160);plt.close(fig)
    m=metrics[metrics.cohort.eq('common')].sort_values('mse_skill_vs_zero')
    fig,ax=plt.subplots(figsize=(12,5));x=m.mse_skill_vs_zero.to_numpy()*100
    ax.hlines(range(len(m)),m.mse_skill_ci_low*100,m.mse_skill_ci_high*100,color='#2474a6')
    ax.scatter(x,range(len(m)),color='#2474a6')
    ax.axvline(0,color='black',ls='--');ax.set_yticks(range(len(m)),m.model);ax.set_xlabel('MSE skill vs Zero Return (%) · block bootstrap 95% CI')
    ax.set_title('Đọc độ bất định trước khi kết luận lợi thế');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(study/'skill_intervals.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);args=parser.parse_args();report(args.study.resolve())
