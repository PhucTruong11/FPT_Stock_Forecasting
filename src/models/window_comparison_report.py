"""Validate window research, reload unique models, select windows only on validation."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.research_notes import write_note
from src.data.contracts import BASE_DIR,file_sha256
from src.data.source_audit import save_json,guarded_hashes
from src.models.source_comparison import REFERENCE,load_inputs,source_part,context_positions
from src.models.window_comparison import window_masks,select_windows,SOURCE


def report(study,reload=False):
    proto=json.loads((study/'protocol.json').read_text());manifest=json.loads((study/'manifest.json').read_text())
    if json.loads((study/'status.json').read_text())['status']!='complete':raise ValueError('Incomplete study.')
    assert file_sha256(study/'protocol.json')==manifest['protocol_sha256']
    with zipfile.ZipFile(study/'source_snapshot.zip') as archive:
        for name,sha in proto['source_sha256'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==sha
        assert hashlib.sha256(archive.read('config.json')).hexdigest()==proto['config_sha256']
    config=proto['config'];cfg=proto['fixed_model_and_fold_config']
    base=BASE_DIR/'reports/source_comparison'/config['control_study']
    assert file_sha256(base/'protocol.json')==proto['base_protocol_sha256']
    assert file_sha256(base/'input_snapshot.zip')==proto['input_zip_sha256']
    datasets,cohort=load_inputs(base);features=datasets[SOURCE][0];columns=cfg['feature_groups']['returns']
    scores=pd.read_csv(study/'metrics.csv');predictions=pd.read_csv(study/'predictions.csv.gz',parse_dates=['Date','Target_Date_1d'])
    folds={f['id']:f for f in cfg['folds']};verified=0
    for (candidate,phase),p in predictions.groupby(['candidate','phase']):
        first=p.iloc[0];masks=window_masks(cohort,folds[first.fold],config['windows'][first.window]);expected=cohort.loc[masks[phase]]
        assert np.array_equal(p.Date.to_numpy(),expected.Date.to_numpy())
        assert np.array_equal(p.Target_Date_1d.to_numpy(),expected.Target_Date_1d.to_numpy())
        assert np.allclose(p[REFERENCE],expected[REFERENCE],atol=1e-12)
        t,q=p[REFERENCE].to_numpy(),p.predicted_log_return.to_numpy();assert np.isfinite(q).all()
        mse=np.mean((t-q)**2);nz=t!=0;issued=nz&(q!=0)
        metrics={'rmse':np.sqrt(mse),'mae':np.mean(np.abs(t-q)),'n':len(t),'nonzero_actual_n':nz.sum(),'neutral_actual_n':(~nz).sum(),
            'mse_skill_vs_zero':1-mse/np.mean(t*t),'direction_coverage_pct':100*issued.sum()/nz.sum(),
            'hit_rate_pct':100*np.mean(np.sign(t[nz])==np.sign(q[nz])) if issued.any() else np.nan,
            'balanced_accuracy_pct':50*((q[t>0]>0).mean()+(q[t<0]<0).mean()) if np.any(q!=0) else np.nan}
        saved=scores[(scores.candidate==candidate)&(scores.phase==phase)];assert len(saved)==1;saved=saved.iloc[0]
        for name,value in metrics.items():assert (pd.isna(value) and pd.isna(saved[name])) or abs(value-saved[name])<1e-10,(candidate,phase,name)
        if first.model in ['Zero-Return','Train Mean','Return Persistence']:
            native=source_part(expected.reset_index(drop=True),SOURCE,features)
            baseline=np.zeros(len(p)) if first.model=='Zero-Return' else (np.full(len(p),cohort.loc[masks['train'],REFERENCE].mean()) if first.model=='Train Mean' else native.Log_Return.to_numpy())
            assert np.allclose(q,baseline,atol=1e-12)
        verified+=1
    assert verified==len(scores)==320
    assert proto['protected_before']==manifest['protected_after']==guarded_hashes()
    save_json(study/'verification.json',{'all_metric_fields_verified':True,'metric_rows':verified,'same_fixed_evaluation_cohort':True,
        'training_only_baselines':True,'protected_files_unchanged':True,'protocol_input_and_source_hashes_verified':True})
    if reload:
        import joblib
        import tensorflow as tf
        from xgboost import XGBRegressor
        from src.models.experiment import make_sequences
        tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
        for name,sha in manifest['artifact_sha256'].items():assert file_sha256(BASE_DIR/name)==sha
        unique={c['model_path']:c for c in manifest['conditions']};checks=[]
        for i,c in enumerate(unique.values(),1):
            masks=window_masks(cohort,folds[c['fold']],config['windows'][c['window']])
            train=source_part(cohort.loc[masks['train']].reset_index(drop=True),SOURCE,features)
            if c['model']=='LSTM':
                tf.keras.backend.clear_session();model=tf.keras.models.load_model(BASE_DIR/c['model_path'],compile=False)
                scalers=joblib.load(BASE_DIR/c['scalers_path']);context=context_positions(train.Feature_Position.to_numpy(),cfg['lookback'])
                assert np.array_equal(context,scalers['context_positions'])
                for scaler,samples in [(scalers['features'],features.iloc[context][columns].to_numpy()),(scalers['target'],train[REFERENCE].to_numpy().reshape(-1,1))]:
                    assert int(scaler.n_samples_seen_)==len(samples)
                    assert np.allclose(scaler.mean_,samples.mean(axis=0),atol=1e-12)
                    assert np.allclose(scaler.var_,samples.var(axis=0),atol=1e-12)
                values=scalers['features'].transform(features[columns])
            elif c['model']=='XGBoost':model=XGBRegressor();model.load_model(BASE_DIR/c['model_path'])
            else:
                model=joblib.load(BASE_DIR/c['model_path']);assert np.allclose(model.steps[0][1].mean_,train[columns].mean(),atol=1e-12)
            for phase in ['validation','evaluation']:
                native=source_part(cohort.loc[masks[phase]].reset_index(drop=True),SOURCE,features)
                pred=scalers['target'].inverse_transform(model(make_sequences(values,native.Feature_Position.to_numpy(),cfg['lookback']),training=False).numpy()).ravel() if c['model']=='LSTM' else model.predict(native[columns])
                exported=predictions[(predictions.candidate==c['candidate'])&(predictions.phase==phase)]
                assert np.allclose(pred,exported.predicted_log_return,atol=1e-8,rtol=1e-5)
                checks.append({'candidate':c['candidate'],'phase':phase,'verified':True})
            # Every duplicate condition must have precisely the same saved forecast.
            for duplicate in [x for x in manifest['conditions'] if x['model_path']==c['model_path']]:
                for phase in ['validation','evaluation']:
                    a=predictions[(predictions.candidate==c['candidate'])&(predictions.phase==phase)].predicted_log_return.to_numpy()
                    b=predictions[(predictions.candidate==duplicate['candidate'])&(predictions.phase==phase)].predicted_log_return.to_numpy()
                    assert np.array_equal(a,b)
            if i%14==0:print(f'Reload verified {i}/{len(unique)}',flush=True)
        save_json(study/'reload_verification.json',{'unique_models':len(unique),'phase_checks':len(checks),'all_duplicate_predictions_identical':True,'train_only_scaler_statistics_verified':True,'checks':checks})
    summary=scores.groupby(['fold','window','model','phase']).agg(runs=('rmse','size'),rmse_mean=('rmse','mean'),rmse_std=('rmse','std'),mae_mean=('mae','mean'),skill_mean=('mse_skill_vs_zero','mean')).reset_index()
    summary.to_csv(study/'summary.csv',index=False,lineterminator='\n')
    selected=select_windows(summary,list(config['windows']));selected.to_csv(study/'validation_selection.csv',index=False,lineterminator='\n')
    primary=selected[~selected.fold.eq('eval_2025')].groupby('model').agg(selected_rmse=('evaluation_rmse','mean'),control_rmse=('expanding_evaluation_rmse','mean'))
    primary['gain_pct']=100*(1-primary.selected_rmse/primary.control_rmse);primary.to_csv(study/'selected_primary_summary.csv',lineterminator='\n')
    baseline=summary[(summary.model=='Zero-Return')&(summary.window=='expanding')&(summary.phase=='evaluation')&~summary.fold.eq('eval_2025')].rmse_mean.mean()
    table=['| Model | Window chọn bằng validation: RMSE | Expanding: RMSE | RMSE giảm |','|---|---:|---:|---:|']
    for model,row in primary.iterrows():table.append(f'| {model} | {row.selected_rmse:.7f} | {row.control_rmse:.7f} | {row.gain_pct:+.2f}% |')
    text=f'''# Đối chứng training window FPT — {study.name}

## Thiết kế đã khóa trước fit
Nguồn Vietcap, 6 features lợi suất, cấu hình Ridge/XGBoost/LSTM và seed giữ đúng control study {config['control_study']}. Chỉ đổi thời gian của origin train: expanding, 2y, 3y, 5y. Warm-up và context của scaler có thể có ngày trước đầu window; không có context sau origin/cutoff. Evaluation/validation giữ nguyên ngày và nhãn. Chỉ chọn window bằng **mean validation RMSE qua seed**, tie-break theo thứ tự cấu hình, không chọn bằng evaluation.

Có 112 điều kiện window/model/fold/seed, **{manifest['new_fits']} lượt fit mới**, {manifest['reused_conditions']} điều kiện tái sử dụng vì cùng train cohort với expanding đã kiểm chứng. Đây không phải 112 lần training mới. Mọi window thiếu đủ số năm từ nguồn vẫn giữ đúng các dòng sẵn có; các window trùng train cohort không được coi là bằng chứng độc lập.

## Kết quả chọn bằng validation, evaluation 2022–2024
Mean-fold không trọng số, mean qua seed trong từng fold. Đây là chẩn đoán hồi cứu đã xem, không phải holdout tương lai.

{chr(10).join(table)}

Zero Return cùng nhãn/cohort có mean-fold RMSE **{baseline:.7f}**. Giảm so với expanding chưa có nghĩa vượt baseline. Xem `validation_selection.csv` để biết window được chọn trong từng fold, gồm cả trường hợp kém control. Không chọn window thắng hậu nghiệm trên evaluation để thay kết quả validation selection.

## Năm 2025 và quyết định
2025 báo cáo riêng trong `summary.csv` và `validation_selection.csv`, không gộp vào bảng chính vì đã được xem ở audit/model development. Không có dữ liệu giá 2026 được tải hoặc đánh giá.

Không tự promote model. Nghiên cứu này nhằm kiểm tra liệu giảm lịch sử train có giúp thích nghi hơn; kết quả không chứng minh một window tốt nhất cho tương lai. Tiếp tục cần forward period chưa dùng để chọn hyperparameter. API/Streamlit và tám hash canonical data/model giữ nguyên. Không commit/push theo yêu cầu người dùng.

## Bằng chứng
`protocol.json` và `source_snapshot.zip` được lưu trước training; input tham chiếu hash gói snapshot ở control study. `cohort_counts.csv` có số mẫu và ngày đầu/cuối train/context; `metrics.csv`, `predictions.csv.gz`, `lstm_history.json`, `manifest.json` lưu cả lượt tốt/xấu và provenance reused/new-fit. `verification.json` tính lại toàn bộ trường metric và baselines; `reload_verification.json` nạp lại unique models, kiểm tra scaler và duplicate forecasts.

```powershell
python -m src.models.window_comparison --config configs/window_comparison_v1.json
python -m src.models.window_comparison_report --study reports/window_comparison/{study.name} --reload-artifacts
```
Lần train mới luôn có ID mới. Không chạy lại để ghi đè metric của lượt hiện tại.
'''
    write_note(study / 'REPORT.md', text)
    index=BASE_DIR/'reports/window_comparison/studies.json';studies=json.loads(index.read_text()) if index.exists() else []
    studies=[s for s in studies if s['study_id']!=study.name]+[{'study_id':study.name,'status':'complete','new_fits':manifest['new_fits'],'conditions':112,'automatic_promotion':False}]
    save_json(index,studies);print(f'Verified/report: {study}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--reload-artifacts',action='store_true');args=parser.parse_args()
    report(args.study.resolve(),args.reload_artifacts)
