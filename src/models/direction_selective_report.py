"""Reload every direction artifact and report probability/coverage trade-offs."""
from __future__ import annotations
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from src.models.research_notes import write_note
from src.data.contracts import BASE_DIR,file_sha256,utc_now
from src.data.source_audit import save_json,guarded_hashes
from src.features.build_features import TARGET_COLUMN,TARGET_DATE_COLUMN
from src.models.recency_bias import training_mask
from src.models.direction_selective import load_snapshot,phase_mask,log_odds,decide,direction_metrics,probability_metrics,select_margin,paired_bootstrap


def assert_metrics(actual,saved):
    for key,value in actual.items():
        if value is None:assert pd.isna(saved[key]),(key,value,saved[key])
        else:assert np.isclose(value,saved[key],atol=1e-8,rtol=1e-8),(key,value,saved[key])


def report(study):
    study=Path(study);proto=json.loads((study/'protocol.json').read_text());cfg=proto['config']
    manifest=json.loads((study/'manifest.json').read_text());_,_,labels=load_snapshot(cfg);columns=cfg['features']
    assert file_sha256(BASE_DIR/'reports/evaluation_2026'/cfg['evaluation_snapshot']/'data_snapshot.zip')==proto['input_zip_sha256']
    with zipfile.ZipFile(study/'source_snapshot.zip') as archive:
        for p,sha in proto['source_sha256'].items():assert hashlib.sha256(archive.read(p)).hexdigest()==sha
    predictions=pd.read_csv(study/'predictions.csv.gz',parse_dates=['Date',TARGET_DATE_COLUMN])
    metrics=pd.read_csv(study/'metrics.csv');choices=pd.read_csv(study/'threshold_selection.csv');curves=pd.read_csv(study/'validation_thresholds.csv')
    matched=pd.read_csv(study/'matched_subset_controls.csv');models={};checks=[]
    folds={f['id']:f for f in proto['folds']}
    for item in manifest['artifacts']:
        fold=folds[item['fold']];path=BASE_DIR/item['path'];assert file_sha256(path)==item['sha256']
        if item['name']=='Ridge_recent':part=labels.loc[training_mask(labels,fold['validation_end'],cfg['training_years'])]
        elif item['role']=='calibrator':part=labels.loc[phase_mask(labels,fold['calibration_start'],fold['calibration_end'])&labels[TARGET_COLUMN].ne(0)]
        else:part=labels.loc[training_mask(labels,fold['train_end'],cfg['training_years'])&labels[TARGET_COLUMN].ne(0)]
        assert part[['Date',TARGET_DATE_COLUMN]].astype(str).to_dict('records')==item['dates']
        cutoff=fold['validation_end'] if item['name']=='Ridge_recent' else fold['calibration_end'] if item['role']=='calibrator' else fold['train_end']
        assert part[TARGET_DATE_COLUMN].le(cutoff).all() and part[TARGET_DATE_COLUMN].max()<pd.Timestamp(fold['evaluation_start'])
        if path.suffix=='.json':model=XGBClassifier();model.load_model(path)
        else:model=joblib.load(path)
        if item['role']=='base' and hasattr(model,'steps'):
            scaler=model.steps[0][1]
            assert np.allclose(scaler.mean_,part[columns].mean(),atol=1e-12,rtol=1e-10)
            assert np.allclose(scaler.var_,part[columns].var(ddof=0),atol=1e-12,rtol=1e-10)
        models[(item['fold'],item['name'])]=model
        checks.append({'fold':item['fold'],'name':item['name'],'train_n':len(part),'dates_cutoff_scaler_hash_verified':True})
    for fold in proto['folds']:
        name=fold['id'];evaluation=labels.loc[phase_mask(labels,fold['evaluation_start'],fold['evaluation_end'])]
        val=labels.loc[phase_mask(labels,fold['validation_start'],fold['validation_end'])]
        train=labels.loc[training_mask(labels,fold['train_end'],cfg['training_years'])&labels[TARGET_COLUMN].ne(0)]
        majority=1 if train[TARGET_COLUMN].gt(0).mean()>=.5 else -1
        for model_name in ['Ridge_shared','Ridge_recent']:
            g=predictions[(predictions.fold==name)&(predictions.phase=='evaluation')&(predictions.strategy==model_name)]
            q=models[(name,model_name)].predict(evaluation[columns]);assert np.allclose(q,g.predicted_log_return,rtol=1e-9,atol=1e-12)
            assert np.array_equal(np.sign(q).astype(int),g.direction)
        for model_name in ['Logistic','XGBoost']:
            model=models[(name,model_name)];calibrator=models[(name,model_name+'_sigmoid')]
            vp=model.predict_proba(val[columns])[:,1].astype(float);vcal=calibrator.predict_proba(log_odds(vp))[:,1]
            margin,reason,recomputed=select_margin(val[TARGET_COLUMN],vcal,cfg['selection'])
            choice=choices[(choices.fold==name)&(choices.model==model_name)].iloc[0]
            assert np.isclose(margin,choice.margin) and reason==choice.reason
            saved_curve=curves[(curves.fold==name)&(curves.model==model_name)]
            for actual,(_,saved) in zip(recomputed,saved_curve.iterrows()):
                assert_metrics({k:v for k,v in actual.items() if k!='eligible'},saved)
                assert bool(actual['eligible'])==bool(saved.eligible)
            for phase in ['calibration','validation']:
                g=predictions[(predictions.fold==name)&(predictions.phase==phase)&(predictions.strategy==model_name)]
                part=labels.merge(g[['Date',TARGET_DATE_COLUMN]],on=['Date',TARGET_DATE_COLUMN],validate='one_to_one')
                raw=model.predict_proba(part[columns])[:,1].astype(float)
                assert np.allclose(raw,g.raw_probability,atol=1e-10,rtol=1e-8)
                assert np.allclose(calibrator.predict_proba(log_odds(raw))[:,1],g.probability_up,atol=1e-10,rtol=1e-8)
            raw=model.predict_proba(evaluation[columns])[:,1].astype(float);cal=calibrator.predict_proba(log_odds(raw))[:,1]
            for suffix,p,m in [('raw',raw,0.),('calibrated',cal,0.),('selective',cal,margin)]:
                g=predictions[(predictions.fold==name)&(predictions.phase=='evaluation')&(predictions.strategy==model_name+'_'+suffix)]
                assert np.allclose(p,g.probability_up,rtol=1e-8,atol=1e-10)
                assert np.array_equal(decide(p,m),g.direction)
        evaluation_pred=predictions[(predictions.fold==name)&(predictions.phase=='evaluation')]
        keys=list(zip(evaluation.Date,evaluation[TARGET_DATE_COLUMN]))
        for strategy,g in evaluation_pred.groupby('strategy',sort=False):
            assert list(zip(g.Date,g[TARGET_DATE_COLUMN]))==keys
            assert np.allclose(g.actual_log_return,evaluation[TARGET_COLUMN],rtol=1e-10,atol=1e-12)
            d=g.direction.to_numpy(dtype=int);y=g.actual_log_return.to_numpy();saved=metrics[(metrics.fold==name)&(metrics.strategy==strategy)].iloc[0]
            # Independent counts: neutral + abstain never count as a correct direction.
            called=d!=0;correct=(d==np.sign(y))&called
            assert correct.sum()==saved.correct_n and called.sum()==saved.calls_n
            assert_metrics(direction_metrics(y,d),saved)
            assert_metrics(paired_bootstrap(y,d,majority,cfg['bootstrap']),saved)
            if g.probability_up.notna().all():assert_metrics(probability_metrics(y,g.probability_up),saved)
            if strategy=='Majority_train':assert np.all(d==majority)
            if strategy=='Always_up':assert np.all(d==1)
            if strategy=='Always_down':assert np.all(d==-1)
        for _,row in matched[matched.fold==name].iterrows():
            selection_pred=evaluation_pred[evaluation_pred.strategy==row.selection_strategy];called=selection_pred.direction.ne(0).to_numpy()
            control=evaluation_pred[evaluation_pred.strategy==row.control]
            assert_metrics(direction_metrics(control.actual_log_return.to_numpy()[called],control.direction.to_numpy(dtype=int)[called]),row)
    # Existing latest Ridge control must reproduce the previously published frozen 2026 forecasts.
    old=BASE_DIR/'reports/recency_bias/20261030T022440928507Z'
    old_pred=pd.read_csv(old/'predictions.csv.gz',parse_dates=['Date',TARGET_DATE_COLUMN])
    old_pred=old_pred[(old_pred.strategy=='refit_2025_native')&(old_pred.cohort=='vci_full')]
    current=predictions[(predictions.fold=='eval_2026')&(predictions.phase=='evaluation')&(predictions.strategy=='Ridge_recent')]
    parity=current.merge(old_pred,on=['Date',TARGET_DATE_COLUMN],suffixes=('_new','_old'),validate='one_to_one')
    assert len(parity)==202 and np.allclose(parity.predicted_log_return_new,parity.predicted_log_return_old,atol=1e-12,rtol=1e-9)
    assert guarded_hashes()==proto['protected_before']==manifest['protected_after']
    save_json(study/'verification.json',{'verified_at_utc':utc_now(),'artifact_checks':checks,'artifacts_reloaded':len(checks),
        'metric_groups':len(metrics),'threshold_choices_recomputed':len(choices),'validation_curve_groups':len(curves),
        'matched_control_groups':len(matched),'frozen_2026_ridge_control_parity_n':len(parity),
        'source_archives_and_hashes_verified':True,'canonical_unchanged':True,'verification_refits':0})
    primary=metrics[metrics.year.isin(cfg['primary_years'])].groupby('strategy').agg(
        mean_fold_called_accuracy=('called_accuracy_all_pct','mean'),mean_fold_balanced=('balanced_called_nonzero_pct','mean'),
        mean_fold_coverage=('coverage_pct','mean'),worst_fold_balanced=('balanced_called_nonzero_pct','min'),
        mean_fold_brier=('brier_nonzero','mean'),folds=('fold','nunique')).reset_index()
    primary.to_csv(study/'primary_summary.csv',index=False)
    # Same-date secondary slices preserve the denominators used by Notebook 06.
    slices=[];eval26=predictions[(predictions.fold=='eval_2026')&(predictions.phase=='evaluation')]
    for source,scope in [(BASE_DIR/'reports/evaluation_2026'/cfg['evaluation_snapshot']/'common_cohort.csv','common_2026'),
                         (BASE_DIR/'reports/period_comparison/20261030T024048531444Z/predictions.csv.gz','jun_oct_common')]:
        keys=pd.read_csv(source,parse_dates=['Date',TARGET_DATE_COLUMN])[['Date',TARGET_DATE_COLUMN]].drop_duplicates()
        for strategy,g in eval26.groupby('strategy'):
            h=g.merge(keys,on=['Date',TARGET_DATE_COLUMN],validate='one_to_one')
            slices.append({'scope':scope,'strategy':strategy,**direction_metrics(h.actual_log_return,h.direction)})
    pd.DataFrame(slices).to_csv(study/'diagnostic_2026_slices.csv',index=False)
    _write_report(study,proto,manifest,metrics,choices,primary)
    final=[r for r in manifest['artifacts'] if r['fold']=='eval_2026' and r['name'] in ['Logistic','XGBoost','Logistic_sigmoid','XGBoost_sigmoid']]
    save_json(study/'research_bundle.json',{'role':'fixed research direction branch; not serving promotion or a live forecast',
        'features':columns,'model_train_target_cutoff':'2025-06-30','calibration_target_cutoff':'2025-09-30',
        'threshold_validation_target_cutoff':'2025-12-31','neutral_policy':cfg['neutral_policy'],
        'models':final,'thresholds':choices[choices.fold=='eval_2026'].to_dict('records'),'automatic_promotion':False})
    save_json(study/'status.json',{'status':'complete','base_fits':manifest['base_fits'],'calibrator_fits':manifest['calibrator_fits'],'completed_at_utc':utc_now()})
    index=study.parent/'studies.json';items=json.loads(index.read_text()) if index.exists() else []
    items=[r for r in items if r['study_id']!=study.name];items.append({'study_id':study.name,'status':'complete'})
    save_json(index,items)


def _write_report(study,proto,manifest,metrics,choices,primary):
    def fmt(v,digits=2):return '—' if pd.isna(v) else f'{v:.{digits}f}'
    rows=['| Phương án | Mean-fold accuracy trên calls (%) | Balanced trên calls khác0 (%) | Coverage (%) | Worst-fold balanced (%) |',
          '|---|---:|---:|---:|---:|']
    for _,r in primary.iterrows():rows.append(f'| {r.strategy} | {fmt(r.mean_fold_called_accuracy)} | {fmt(r.mean_fold_balanced)} | {fmt(r.mean_fold_coverage)} | {fmt(r.worst_fold_balanced)} |')
    text=f'''# FPT: phân loại tăng/giảm và “chưa đủ tín hiệu” — {study.name}

## Thiết kế đã khóa trước fit
Hai classifier cấu hình cố định: Logistic Regression và XGBoost Classifier, cùng6 return features. Mỗi fold giữ 3 năm train tới30/6 năm trước; July–September hiệu chỉnh sigmoid độc lập; October–December chọn margin; cả năm tiếp theo để chấm (2026 chỉ đến29/10). Không random split, không refit sau chọn margin. Không thêm VNINDEX, không download mới hoặc sửa API/Streamlit.

Classifier và Ridge_shared dùng đúng cùng các dòng train có return khác0. Ridge_recent là control hiện tại, dùng cả return0 và toàn bộ 3 năm tới31/12 năm trước, có nhiều thông tin gần evaluation hơn; không dùng nó để chọn ngưỡng trên validation mà nó đã học. Riêng2026 tái lập đúng202 forecast Ridge refit_2025_native cũ. Các nhóm khác ngày train không cô lập riêng kiến trúc; so raw/calibrated/selective trong một classifier giữ nguyên weights.

Nhãn tăng/giảm là supervised classification; hồi quy giá vẫn giữ riêng. Raw xác suất là ước lượng P(tăng|không đứng giá) từ tập train nonzero, chưa bảo đảm đáng tin hoặc tốt hơn khi sigmoid. Train/calibration loại return đúng0; **mọi origin evaluation vẫn giữ**. Call trên actual0 tính sai; abstain không được tính là đoán đúng đứng giá. Không chuyển probability thành giá hoặc diễn giải return RMSE cho classifier.

Chọn margin trong {proto['config']['selection']['margins']}: p≥0.5+m ⇒ tăng, p<0.5−m ⇒ giảm, còn lại ⇒ chưa đủ tín hiệu. Tối đa balanced accuracy validation trên calls nonzero, cần coverage≥60%, ít nhất30 calls nonzero, ít nhất10 actual mỗi hướng trong nhóm called. Tie ưu tiên coverage cao/margin nhỏ; không đủ ứng viên thì về margin0, không ép model phải bỏ qua. Validation dùng chọn ngưỡng nên điểm ở đó không phải test. Probability calibration/threshold tuning dùng các nhãn trước evaluation, nhưng **mọi giai đoạn2022–2026 đã được xem trước trong workflow**: đây là nghiên cứu hồi cứu thích nghi, không holdout độc lập mới. Adjusted history hiện tại chưa point-in-time certified.

## Primary 2022–2024 — mean từng fold, không trộn2025/2026 vào lựa chọn

{chr(10).join(rows)}

Coverage tính trên mọi origin. Accuracy trên calls gồm actual0 là sai; Hit nonzero trên calls giữ riêng để đối chiếu metric lịch sử. Balanced chỉ giữa actual tăng/giảm **đã được gọi**, vì vậy phải đọc thêm coverage từng lớp, số calls và correct/all-origins. Không so accuracy trên60% ngày với forced trên100% rồi gọi là cải thiện toàn bộ.

## Đối chứng và bằng chứng
`metrics.csv`: từng fold/strategy, N/calls/abstain, neutral calls, Hit, balanced, coverage tăng/giảm, Brier/logloss trên toàn bộ nhãn nonzero (không chỉ các calls được giữ). `matched_subset_controls.csv`: Ridge/majority/always-up/down trên đúng tập ngày selective gọi tín hiệu. Majority chọn từ train, không lấy lớp chiếm ưu thế evaluation để chọn model. Always-up/down là control cố định.

Block-bootstrap10 origin/1.000 resamples, seed20261030, lưu CI95% mô tả của chênh accuracy selective/forced so majority **trên cùng calls**. Giữ mask khi resample toàn bộ origin. Không xử lý hết việc thích nghi/thử nhiều phương án; calls nhỏ hoặc thiếu một lớp không đủ để diễn giải ổn định. `diagnostic_2026_slices.csv` giữ riêng200 cặp common và104 cặp tháng6–10; không áp CI của full cohort cho các subset.

`phase_counts.csv`: N/cutoff từng phase; `threshold_selection.csv`/`validation_thresholds.csv`: toàn bộ ứng viên và lý do chọn, không tune trên evaluation; `predictions.csv.gz`: calibration/validation/evaluation với trạng thái tăng/giảm/chưa đủ tín hiệu. `manifest.json` lưu {manifest['base_fits']} base fits và {manifest['calibrator_fits']} sigmoid fits, exact training dates/hash của mọi artifact. `verification.json` reload tất cả, đối soát phase, forecast, metric và lựa chọn; không fit khi verify. Weights dưới `models/research/` bị gitignore; source/protocol/input hash đã lưu. Không tự promote.

`research_bundle.json` ghi các artifact/classifier của fold2026 và cutoff/margin cố định để tham khảo tích hợp sau. Không thay model serving, ledger shadow cũ hoặc endpoint API. Đây không phải model đã được cập nhật tới29/10/2026.

## Tài liệu phương pháp
- [Scikit-learn: tách chọn ngưỡng khỏi fitting](https://scikit-learn.org/stable/modules/classification_threshold.html).
- [Scikit-learn: probability calibration và dữ liệu độc lập](https://scikit-learn.org/stable/modules/calibration.html).
- [XGBoost: binary logistic và regularization](https://xgboost.readthedocs.io/en/stable/parameter.html).

Các giá trị margin cụ thể và stress2025/2026 được đọc trong Notebook07, không chỉ nêu model có điểm đẹp. Chưa commit/push.
'''
    write_note(study / 'REPORT.md', text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);report(p.parse_args().study)
