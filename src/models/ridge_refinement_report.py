"""Independent evidence checks and reports for error diagnosis / Ridge refinement."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.data.contracts import BASE_DIR,file_sha256
from src.data.source_audit import save_json,guarded_hashes
from src.models.robustness import fold_masks,score_direction
from src.models.source_comparison import source_part,REFERENCE
from src.models.window_comparison import SOURCE,window_masks
from src.models.ridge_refinement import inputs,fit_thresholds,assign_regimes,select_config


def diagnostic_report(study):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    proto=json.loads((study/'diagnostic_protocol.json').read_text());config=proto['config']
    base,previous,cfg,features,cohort=inputs(config)
    thresholds=json.loads((study/'regime_thresholds.json').read_text())
    d=pd.read_csv(study/'diagnostic_predictions.csv.gz',parse_dates=['Date','Target_Date_1d'])
    chosen=pd.read_csv(previous/'validation_selection.csv');old=pd.read_csv(previous/'predictions.csv.gz',parse_dates=['Date','Target_Date_1d'])
    for fold in cfg['folds']:
        masks=fold_masks(cohort,fold);train=source_part(cohort.loc[masks['train']].reset_index(drop=True),SOURCE,features)
        assert fit_thresholds(train,config)==thresholds[fold['id']]
        assert train.Date.le(pd.Timestamp(fold['train_end'])).all()
        for model in config['diagnostics']['models']:
            window=chosen[(chosen.fold==fold['id'])&(chosen.model==model)].selected_window.iloc[0]
            original=old[(old.fold==fold['id'])&(old.model==model)&(old.window==window)&(old.phase=='evaluation')]
            part=d[(d.fold==fold['id'])&(d.model==model)]
            assert len(original)==len(part)
            for seed,p in part.groupby('seed_id'):
                expected=original[original.seed.fillna(-1).astype(int).eq(seed)]
                assert np.array_equal(p.Date.to_numpy(),expected.Date.to_numpy())
                assert np.allclose(p.predicted_log_return,expected.predicted_log_return,atol=1e-12)
    segments=pd.read_csv(study/'error_segments.csv');year=pd.read_csv(study/'error_by_fold_seed.csv')
    # Check loss attribution without counting each stochastic seed as another date.
    for (scope,segmentation,model),p in segments.groupby(['scope','segmentation','model']):
        assert abs(p.squared_loss_share_pct.sum()-100)<1e-9
        expected_n=d[(d.model==model)&d.fold.isin(config['primary_folds'] if scope=='primary' else [config['stress_fold']])][['fold','Date']].drop_duplicates().shape[0]
        assert p.unique_origins.sum()==expected_n
    save_json(study/'diagnostic_verification.json',{'fixed_previous_validation_selection_verified':True,
        'thresholds_recomputed_from_training_only':True,'no_seed_duplication_in_loss_shares':True,
        'source_forecast_rows':len(d),'all_loss_shares_sum_to_100':True})
    models=['Zero-Return','Train Mean','Ridge','XGBoost','LSTM']
    annual=year.groupby(['fold','model']).rmse.mean().unstack().reindex(columns=models)
    fig,ax=plt.subplots(figsize=(11,4.5));annual.plot(ax=ax,marker='o')
    ax.set_ylabel('Mean-seed RMSE · nhãn Vietcap');ax.set_xlabel('Fold evaluation')
    ax.set_title('Sai số theo thời gian · model/window chọn bằng validation trước đó');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(study/'errors_by_year.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,segmentation,order,title in [(axes[0],'vol_regime',['low','medium','high'],'Biến động: ngưỡng fit train'),
        (axes[1],'trend_proxy',['below_SMA50','near_SMA50','above_SMA50'],'Giá so với SMA50 ±1% (proxy)')]:
        p=segments[(segments.scope=='primary')&(segments.segmentation==segmentation)&segments.model.isin(['Train Mean','Ridge','XGBoost','LSTM'])]
        grid=p.pivot(index='model',columns='segment',values='mean_seed_mse_skill_vs_zero').reindex(index=['Train Mean','Ridge','XGBoost','LSTM'],columns=order)*100
        limit=max(1.,float(np.nanmax(np.abs(grid.to_numpy()))));im=ax.imshow(grid,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
        ax.set_xticks(range(3),grid.columns,rotation=15);ax.set_yticks(range(4),grid.index);ax.set_title(title)
        for i in range(4):
            for j in range(3):ax.text(j,i,f'{grid.iloc[i,j]:+.1f}%',ha='center',va='center')
        fig.colorbar(im,ax=ax,label='MSE skill so với Zero Return (%)')
    fig.suptitle('Chẩn đoán 2022–2024: skill > 0 tốt hơn Zero · không dùng chọn hyperparameter')
    fig.tight_layout();fig.savefig(study/'error_regimes.png',dpi=160);plt.close(fig)
    rolling=pd.read_csv(study/'rolling_errors.csv',parse_dates=['Date'])
    fig,ax=plt.subplots(figsize=(12,4))
    for model in models:
        p=rolling[rolling.model==model];ax.plot(p.Date,p.rolling_60_rms_error,label=model,linewidth=1.2)
    ax.set_title('RMS lỗi gộp seed, rolling 60 origin · forecast ghép từ từng fold');ax.set_ylabel('RMS log-return error')
    ax.legend(ncol=5,fontsize=9);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(study/'rolling_errors.png',dpi=160);plt.close(fig)
    ridge=segments[(segments.scope=='primary')&(segments.segmentation=='vol_regime')&(segments.model=='Ridge')]
    high=ridge[ridge.segment=='high'].iloc[0];total=int(ridge.unique_origins.sum())
    rows=['| Mức biến động | N origin | Mean-seed RMSE Ridge | MSE skill vs Zero | Tỷ trọng squared loss Ridge |',
          '|---|---:|---:|---:|---:|']
    for level in ['low','medium','high']:
        r=ridge[ridge.segment==level].iloc[0]
        rows.append(f'| {level} | {r.unique_origins} | {r.mean_seed_rmse:.7f} | {100*r.mean_seed_mse_skill_vs_zero:+.2f}% | {r.squared_loss_share_pct:.1f}% |')
    text=f'''# Phân tích lỗi trước tối ưu Ridge — {study.name}

Forecast được lấy từ study window trước, lựa chọn model/window đã khóa bằng validation. Chưa có model mới được fit khi tạo báo cáo này. Chỉ chẩn đoán; không loại các ngày sai nhiều hoặc đưa nhãn evaluation vào chọn alpha.

## Theo thời gian
![Sai số theo năm](errors_by_year.png)
Sai số tuyệt đối thay đổi theo mức biến động của nhãn; RMSE cao không tự chứng minh model kém hơn baseline. 2022–2024 là phần chính; 2025 đã xem, chỉ báo riêng như chẩn đoán.

## Theo mức biến động và vị trí so với SMA50
Ngưỡng 1/3 và 2/3 của Volatility_20 được fit bằng **expanding common train origins của từng fold**, rồi áp cố định cho evaluation. Volatility tại origin chỉ dùng giá hiện tại/quá khứ. Trend là proxy Close/SMA50, band ±1%; không phải nhãn thị trường chính thức hoặc bằng chứng “đi ngang” thực sự. Các features này chỉ dùng phân tích, model vẫn giữ sáu features lợi suất.

{chr(10).join(rows)}

Nhóm high-vol có {int(high.unique_origins)}/{total} origin ({100*high.unique_origins/total:.1f}%) nhưng chiếm {high.squared_loss_share_pct:.1f}% squared loss của Ridge. Đây là mô tả phân bố lỗi; chưa chứng minh nguyên nhân hoặc khả năng dự báo trước các ngày khó.

![Skill theo nhóm](error_regimes.png)
![Rolling errors](rolling_errors.png)

## Cách tính và giới hạn
RMSE trong bảng nhóm lấy mean qua các seed sau khi tính trong từng nhóm; N là số origin duy nhất. Tỷ trọng loss dùng mean squared loss qua seed tại mỗi origin, tránh đếm một ngày ba lần. Rolling plot dùng sqrt của mean loss gộp seed qua 60 origin, khác mean-seed RMSE trong bảng. Giữ cả nhóm nhỏ và đánh dấu small_segment nếu N<20; không điền nhóm thiếu thành 0. Các fold đã được xem nhiều lần, không phải ước lượng ngoài mẫu không thiên lệch hoặc phân tích nhân quả.

`error_segments.csv`, `error_by_fold_seed.csv`, `regime_thresholds.json`, `diagnostic_predictions.csv.gz` giữ số liệu gốc; `diagnostic_verification.json` kiểm tra threshold train-only, forecast nguồn và loss shares. Grid alpha/window trong config đã định trước khi tính các nhóm lỗi; không thêm feature hoặc nới grid theo điểm evaluation.
'''
    (study/'ERROR_ANALYSIS.md').write_text(text,encoding='utf-8',newline='\n')


def report(study):
    proto=json.loads((study/'protocol.json').read_text());manifest=json.loads((study/'manifest.json').read_text());config=proto['config']
    assert json.loads((study/'status.json').read_text())['status']=='complete'
    assert file_sha256(study/'protocol.json')==manifest['protocol_sha256']
    assert file_sha256(study/'diagnostic_protocol.json')==proto['diagnostic_protocol_sha256']
    assert file_sha256(study/'ERROR_ANALYSIS.md')==proto['diagnostic_report_sha256']
    with zipfile.ZipFile(study/'source_snapshot.zip') as archive:
        for name,sha in proto['source_sha256'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==sha
        assert hashlib.sha256(archive.read('config.json')).hexdigest()==proto['config_sha256']
    base,previous,cfg,features,cohort=inputs(config);columns=proto['feature_columns']
    for p,sha in [(base/'protocol.json',proto['source_study_protocol_sha256']),(base/'input_snapshot.zip',proto['input_zip_sha256']),
        (previous/'protocol.json',proto['window_study_protocol_sha256']),(previous/'predictions.csv.gz',proto['window_predictions_sha256']),
        (previous/'validation_selection.csv',proto['window_selection_sha256'])]:assert file_sha256(p)==sha
    metrics=pd.read_csv(study/'metrics.csv');pred=pd.read_csv(study/'predictions.csv.gz',parse_dates=['Date','Target_Date_1d'])
    folds={f['id']:f for f in cfg['folds']};unique={};checks=[]
    for name,sha in manifest['artifact_sha256'].items():assert file_sha256(BASE_DIR/name)==sha
    for spec in manifest['conditions']:
        key=(spec['fold'],spec['window'],spec['alpha']);masks=window_masks(cohort,folds[spec['fold']],config['windows'][spec['window']])
        train=source_part(cohort.loc[masks['train']].reset_index(drop=True),SOURCE,features)
        model=joblib.load(BASE_DIR/spec['model_path']);scaler=model.steps[0][1];ridge=model.steps[-1][1]
        assert ridge.alpha==spec['alpha'] and int(scaler.n_samples_seen_)==len(train)
        assert np.allclose(scaler.mean_,train[columns].mean().to_numpy(),atol=1e-12)
        assert np.allclose(scaler.var_,train[columns].var(ddof=0).to_numpy(),atol=1e-12)
        unique[spec['model_path']]=True
        for phase in ['validation','evaluation']:
            expected=cohort.loc[masks[phase]].reset_index(drop=True);native=source_part(expected,SOURCE,features)
            p=pred[(pred.candidate==spec['candidate'])&(pred.phase==phase)]
            assert np.array_equal(p.Date.to_numpy(),expected.Date.to_numpy()) and np.array_equal(p.Target_Date_1d.to_numpy(),expected.Target_Date_1d.to_numpy())
            assert np.allclose(p[REFERENCE],expected[REFERENCE],atol=1e-12)
            q=model.predict(native[columns]);assert np.allclose(q,p.predicted_log_return,atol=1e-12,rtol=1e-10)
            truth=p[REFERENCE].to_numpy();q=p.predicted_log_return.to_numpy();nz=truth!=0;issued=nz&(q!=0);mse=np.mean((truth-q)**2)
            calculated={'n':len(truth),'rmse':np.sqrt(mse),'mae':np.mean(abs(truth-q)),'nonzero_actual_n':nz.sum(),'neutral_actual_n':(~nz).sum(),
                'hit_rate_pct':100*np.mean(np.sign(truth[nz])==np.sign(q[nz])) if issued.any() else np.nan,
                'direction_coverage_pct':100*issued.sum()/nz.sum(),'mse_skill_vs_zero':1-mse/np.mean(truth**2),
                'balanced_accuracy_pct':50*((q[truth>0]>0).mean()+(q[truth<0]<0).mean()) if np.any(q!=0) else np.nan}
            saved=metrics[(metrics.candidate==spec['candidate'])&(metrics.phase==phase)];assert len(saved)==1;saved=saved.iloc[0]
            for name,value in calculated.items():assert (pd.isna(value) and pd.isna(saved[name])) or abs(value-saved[name])<1e-10
            checks.append({'candidate':spec['candidate'],'phase':phase,'verified':True})
    assert len(checks)==len(metrics)==160
    # Alpha10 must reproduce the previous Ridge for every window/fold/phase.
    oldmetrics=pd.read_csv(previous/'metrics.csv');oldselection=pd.read_csv(previous/'validation_selection.csv')
    for _,r in metrics[metrics.alpha.eq(10)].iterrows():
        expected=oldmetrics[(oldmetrics.fold==r.fold)&(oldmetrics.window==r.window)&(oldmetrics.model=='Ridge')&(oldmetrics.phase==r.phase)].iloc[0]
        assert abs(r.rmse-expected.rmse)<1e-12
    selected=[];strategy_exports=[]
    oldpred=pd.read_csv(previous/'predictions.csv.gz',parse_dates=['Date','Target_Date_1d'])
    for fold in cfg['folds']:
        fm=metrics[metrics.fold.eq(fold['id'])]
        old_window=oldselection[(oldselection.fold==fold['id'])&(oldselection.model=='Ridge')].selected_window.iloc[0]
        joint=select_config(fm,list(config['windows']),config['alphas'])
        alpha_only=select_config(fm,list(config['windows']),config['alphas'],old_window)
        strategies={'Ridge_expanding':('expanding',10.),'Ridge_window_only':(old_window,10.),
                    'Ridge_alpha_only':(old_window,float(alpha_only.alpha)),'Ridge_joint':(joint.window,float(joint.alpha))}
        for strategy,(window,alpha) in strategies.items():
            for phase in ['validation','evaluation']:
                p=pred[(pred.fold==fold['id'])&(pred.window==window)&pred.alpha.eq(alpha)&(pred.phase==phase)].copy()
                p['strategy']=strategy;strategy_exports.append(p)
                selected.append({'fold':fold['id'],'strategy':strategy,'window':window,'alpha':alpha,'phase':phase,
                    'grid_boundary':alpha in [min(config['alphas']),max(config['alphas'])],**score_direction(p[REFERENCE],p.predicted_log_return)})
        mean_window=oldselection[(oldselection.fold==fold['id'])&(oldselection.model=='Train Mean')].selected_window.iloc[0]
        for strategy,model,window in [('Zero-Return','Zero-Return','expanding'),('TrainMean_expanding','Train Mean','expanding'),('TrainMean_window_selected','Train Mean',mean_window)]:
            for phase in ['validation','evaluation']:
                p=oldpred[(oldpred.fold==fold['id'])&(oldpred.model==model)&(oldpred.window==window)&(oldpred.phase==phase)].copy()
                p['strategy']=strategy;strategy_exports.append(p)
                selected.append({'fold':fold['id'],'strategy':strategy,'window':window,'alpha':None,'phase':phase,
                    'grid_boundary':False,**score_direction(p[REFERENCE],p.predicted_log_return)})
    chosen=pd.DataFrame(selected);exports=pd.concat(strategy_exports,ignore_index=True)
    chosen.to_csv(study/'strategy_metrics.csv',index=False,lineterminator='\n');exports.to_csv(study/'strategy_predictions.csv.gz',index=False,compression='gzip',date_format='%Y-%m-%d',lineterminator='\n')
    chosen[chosen.phase.eq('validation')].to_csv(study/'validation_selection.csv',index=False,lineterminator='\n')
    primary=chosen[chosen.phase.eq('evaluation')&chosen.fold.isin(config['primary_folds'])].groupby('strategy').agg(
        mean_fold_rmse=('rmse','mean'),mean_fold_mae=('mae','mean'),worst_fold_rmse=('rmse','max'),positive_skill_folds=('mse_skill_vs_zero',lambda x:int((x>0).sum()))).reset_index()
    control=float(primary[primary.strategy.eq('Ridge_window_only')].mean_fold_rmse.iloc[0]);primary['rmse_gain_vs_previous_ridge_pct']=100*(1-primary.mean_fold_rmse/control)
    primary.to_csv(study/'primary_summary.csv',index=False,lineterminator='\n')
    # Post-fit regime comparison is diagnostic only; it cannot feed back into selection.
    thresholds=json.loads((study/'regime_thresholds.json').read_text());segment_rows=[]
    for fold in cfg['folds']:
        p=exports[(exports.fold==fold['id'])&(exports.phase=='evaluation')].merge(features[['Date','Volatility_20','SMA_Distance_50']],on='Date',validate='many_to_one')
        p=assign_regimes(p,thresholds[fold['id']])
        for (strategy,regime),segment in p.groupby(['strategy','vol_regime']):
            segment_rows.append({'fold':fold['id'],'strategy':strategy,'vol_regime':regime,**score_direction(segment[REFERENCE],segment.predicted_log_return)})
    pd.DataFrame(segment_rows).to_csv(study/'selected_error_segments.csv',index=False,lineterminator='\n')
    assert guarded_hashes()==proto['protected_before']==manifest['protected_after']
    save_json(study/'verification.json',{'all_metric_fields_verified':True,'grid_metric_rows':len(checks),'unique_models_reloaded':len(unique),
        'condition_prediction_checks':len(checks),'train_only_scalers_verified':True,'alpha10_matches_previous_control':True,
        'selection_uses_validation_only':True,'canonical_files_unchanged':True,'checks':checks})
    make_plots(study,metrics,primary,chosen)
    table=['| Strategy | Mean-fold RMSE | MAE | RMSE giảm so với Ridge window-only | Fold vượt Zero |','|---|---:|---:|---:|---:|']
    for _,r in primary.iterrows():table.append(f'| {r.strategy} | {r.mean_fold_rmse:.7f} | {r.mean_fold_mae:.7f} | {r.rmse_gain_vs_previous_ridge_pct:+.2f}% | {r.positive_skill_folds}/3 |')
    text=f'''# Ridge regularization × training window — {study.name}

## Thiết kế và đối chứng
Chỉ đổi alpha trong [0.1, 1, 10, 100, 1000] và window expanding/2y/3y/5y; giữ nguồn Vietcap, sáu features lợi suất, target và ngày validation/evaluation. Không tải giá 2026. Protocol chẩn đoán/grid được lưu trước phân tích; protocol fit được lưu sau ERROR_ANALYSIS.md và trước fit đầu tiên. Evaluation 2022–2025 đã được xem nhiều lần: nghiên cứu thích nghi hồi cứu, không phải kiểm định ngoài mẫu mới.

80 điều kiện, **{manifest['new_fits']} fit mới**, {len(manifest['conditions'])-manifest['new_fits']} điều kiện reused. 60 models duy nhất gồm 12 alpha10 cũ và 48 mới. Alpha10 tái lập đúng mọi control cũ. Scaler chỉ fit training vectors. Ridge là thuật toán xác định; không gán seed SD=0 hay dùng seed giả để tạo confidence interval.

Alpha-only giữ window đã chọn ở lượt trước rồi chọn alpha bằng validation. Joint chọn cả alpha/window bằng validation. Tie ưu tiên alpha10, rồi thứ tự window/config; tuyệt đối không dùng evaluation chọn cấu hình. Alpha/n được lưu vì objective Ridge dùng tổng squared error; cùng alpha có hiệu lực khác khi N train đổi. Đây là lựa chọn regularization/window, không phải thêm features.

## Evaluation chính 2022–2024
Mean-fold không trọng số trên cùng nhãn/cohort. Dương trong cột giảm là tốt hơn control; không suy ra significance.

{chr(10).join(table)}

![So sánh các chiến lược](strategy_comparison.png)
![Validation grid](validation_grid.png)

## Năm 2025 và giới hạn
2025 tách riêng trong strategy_metrics.csv; không gộp vào bảng chính, không chọn lại cấu hình sau khi thấy điểm. Grid boundary được gắn cờ trong validation_selection.csv; chọn alpha ở biên grid không chứng minh alpha đó là tối ưu toàn cục. Error segmentation chỉ là mô tả, không chứng minh nhân quả, không dùng xóa ngày biến động mạnh.

Không tự promote model hoặc sửa API/Streamlit. Xem DECISION.md cho diễn giải sau khi kiểm chứng. Giữ nguyên canonical data/model và không commit/push.

## Bằng chứng
ERROR_ANALYSIS.md, protocol.json, diagnostic_protocol.json, source_snapshot.zip, cohort_counts.csv, metrics.csv, predictions.csv.gz, strategy_metrics.csv, strategy_predictions.csv.gz, validation_selection.csv, primary_summary.csv, selected_error_segments.csv, diagnostic_verification.json và verification.json.

```powershell
python -m src.models.ridge_refinement --config configs/ridge_refinement_v1.json
python -m src.models.ridge_refinement_report --study reports/ridge_refinement/{study.name}
```
Mỗi lượt fit mới tạo ID mới; không ghi đè nghiên cứu cũ. Notebook 05 tổng hợp các giai đoạn riêng theo đúng nhãn và cohort.
'''
    (study/'REPORT.md').write_text(text,encoding='utf-8',newline='\n')
    index=BASE_DIR/'reports/ridge_refinement/studies.json';items=json.loads(index.read_text()) if index.exists() else []
    items=[s for s in items if s['study_id']!=study.name]+[{'study_id':study.name,'status':'complete','new_fits':manifest['new_fits'],'conditions':80,'automatic_promotion':False}]
    save_json(index,items)


def make_plots(study,metrics,primary,chosen):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,4.5));p=primary.sort_values('mean_fold_rmse')
    ax.barh(p.strategy,p.rmse_gain_vs_previous_ridge_pct,color=['#2474a6' if x>=0 else '#b24c4c' for x in p.rmse_gain_vs_previous_ridge_pct])
    ax.axvline(0,color='black',lw=1);ax.set_xlabel('RMSE giảm so với Ridge window-only (%) · 2022–2024');ax.set_title('Các lựa chọn bằng validation · giữ cả baseline và kết quả kém hơn')
    fig.tight_layout();fig.savefig(study/'strategy_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(15,4));alphas=sorted(metrics.alpha.unique());windows=['expanding','2y','3y','5y']
    for ax,(fold,part) in zip(axes,metrics[metrics.phase.eq('validation')].groupby('fold')):
        grid=part.pivot(index='alpha',columns='window',values='rmse').reindex(index=alphas,columns=windows)
        ax.imshow(grid,aspect='auto',cmap='viridis_r');ax.set_xticks(range(4),windows,rotation=35);ax.set_yticks(range(5),[str(a) for a in alphas]);ax.set_title(fold)
        loc=np.unravel_index(np.argmin(grid.to_numpy()),grid.shape);ax.scatter(loc[1],loc[0],marker='x',s=120,color='red')
    fig.suptitle('Validation RMSE theo alpha/window · thang màu riêng từng fold · đỏ = minimum (ties xem CSV)')
    fig.tight_layout();fig.savefig(study/'validation_grid.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);args=parser.parse_args();report(args.study.resolve())
