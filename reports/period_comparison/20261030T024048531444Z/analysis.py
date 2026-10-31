"""Re-score saved next-observation forecasts on the same June–October 2026 targets."""
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p/'src/data/contracts.py').exists())))
from src.data.contracts import BASE_DIR,file_sha256,utc_now
from src.data.source_audit import guarded_hashes,save_json
from src.models.evaluation_2026 import price_metrics,block_bootstrap
from src.models.robustness import score_direction

EVAL=BASE_DIR/'reports/evaluation_2026/20261030T014012158951Z'
RECENCY=BASE_DIR/'reports/recency_bias/20261030T022440928507Z'
before=guarded_hashes()
out=BASE_DIR/'reports/period_comparison'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
out.mkdir(parents=True);shutil.copyfile(__file__,out/'analysis.py')
paths=[EVAL/'predictions.csv.gz',EVAL/'protocol.json',EVAL/'data_snapshot.zip',RECENCY/'predictions.csv.gz',RECENCY/'protocol.json',RECENCY/'manifest.json']
save_json(out/'scope.json',{'created_at_utc':utc_now(),'target_start_inclusive':'2026-06-01','target_end_exclusive':'2026-11-01',
    'actual_data_end':'2026-10-29','grain':'same observed origin AND next-observation target pairs for every strategy',
    'selection':'user-requested retrospective slice of an already inspected period; not a new holdout',
    'refits':0,'source_sha256':{p.relative_to(BASE_DIR).as_posix():file_sha256(p) for p in paths},
    'excluded':'Oct29 shadow model trained after Jun-Sep targets; duplicate rebuilt/frozen research controls',
    'bootstrap':{'seed':20261030,'block_length':10,'resamples':2000},'protected_before':before})
e=pd.read_csv(EVAL/'predictions.csv.gz',parse_dates=['Date','Target_Date_1d'])
r=pd.read_csv(RECENCY/'predictions.csv.gz',parse_dates=['Date','Target_Date_1d'])
e=e[e.cohort.eq('common')].copy()
e=e.rename(columns={'reference_log_return':'actual_log_return','reference_origin_close':'origin_close_adjusted','model':'strategy'})
e['mode']='fixed_pre2026_training_labels'
r=r[r.cohort.eq('previous_common')&~r.strategy.isin(['frozen_2023','rebuilt_2023_same_rows','Zero-Return'])].copy()
r=r.rename(columns={'Reference_Return':'actual_log_return'})
r['mode']=np.where(r.strategy.str.contains('monthly|bias|RollingMean'),'causal_time_order_simulation','fixed_pre2026_training_labels')
cols=['Date','Target_Date_1d','strategy','mode','actual_log_return','predicted_log_return','origin_close_adjusted','actual_close_adjusted','predicted_close_adjusted']
pred=pd.concat([e[cols],r[cols]],ignore_index=True)
pred=pred[pred.Target_Date_1d.ge('2026-06-01')&pred.Target_Date_1d.lt('2026-11-01')].copy()
pred=pred.sort_values(['strategy','Date']).reset_index(drop=True)
anchor=pred[pred.strategy.eq('Zero-Return')].sort_values('Date')
keys=list(zip(anchor.Date,anchor.Target_Date_1d));assert len(keys)==104
acquisition=json.loads((EVAL/'acquisition.json').read_text())
assert file_sha256(EVAL/'data_snapshot.zip')==acquisition['zip_sha256']
with zipfile.ZipFile(EVAL/'data_snapshot.zip') as archive:
    raw=pd.read_csv(io.BytesIO(archive.read('vci.csv')),parse_dates=['Date']).set_index('Date')
for name,g in pred.groupby('strategy'):
    assert not g.duplicated(['Date','Target_Date_1d']).any()
    assert list(zip(g.Date,g.Target_Date_1d))==keys,name
    actual=np.log(raw.loc[g.Target_Date_1d,'Close'].to_numpy()/raw.loc[g.Date,'Close'].to_numpy())
    np.testing.assert_allclose(g.actual_log_return,actual,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(g.origin_close_adjusted,raw.loc[g.Date,'Close'],rtol=1e-12)
    np.testing.assert_allclose(g.actual_close_adjusted,g.origin_close_adjusted*np.exp(actual),rtol=1e-12)
    np.testing.assert_allclose(g.predicted_close_adjusted,g.origin_close_adjusted*np.exp(g.predicted_log_return),rtol=1e-12)

def calculate(g):
    y=g.actual_log_return.to_numpy();q=g.predicted_log_return.to_numpy()
    actual_simple=np.expm1(y)*100;pred_simple=np.expm1(q)*100
    price_error_pct=100*np.abs(np.expm1(q-y))
    return {**score_direction(y,q),**price_metrics(y,q,g.origin_close_adjusted),
        'return_rmse_pp':float(np.sqrt(np.mean((pred_simple-actual_simple)**2))),
        'return_mae_pp':float(np.mean(np.abs(pred_simple-actual_simple))),
        'magnitude_mae_pp':float(np.mean(np.abs(np.abs(pred_simple)-np.abs(actual_simple)))),
        'price_error_le_1pct_n':int(np.sum(price_error_pct<=1)), 'price_error_le_1pct_pct':float(100*np.mean(price_error_pct<=1)),
        'price_error_le_2pct_n':int(np.sum(price_error_pct<=2)), 'price_error_le_2pct_pct':float(100*np.mean(price_error_pct<=2)),
        'actual_return_std_pp':float(actual_simple.std()),'predicted_return_std_pp':float(pred_simple.std()),
        'predicted_to_actual_std_ratio':float(pred_simple.std()/actual_simple.std()),
        'up_recall_pct':float(100*np.mean(q[y>0]>0)) if (y>0).any() and (q!=0).any() else None,
        'down_recall_pct':float(100*np.mean(q[y<0]<0)) if (y<0).any() and (q!=0).any() else None}

summary=[];monthly=[]
for name,g in pred.groupby('strategy'):
    summary.append({'strategy':name,'mode':g['mode'].iloc[0],**calculate(g),
        **block_bootstrap(g.actual_log_return.to_numpy(),g.predicted_log_return.to_numpy(),{'seed':20261030,'block_length':10,'resamples':2000})})
    for month,h in g.groupby(g.Target_Date_1d.dt.to_period('M')):
        monthly.append({'strategy':name,'target_month':str(month),**calculate(h)})
summary=pd.DataFrame(summary);monthly=pd.DataFrame(monthly)
summary.to_csv(out/'summary.csv',index=False);monthly.to_csv(out/'monthly_metrics.csv',index=False)
pred.to_csv(out/'predictions.csv.gz',index=False,compression='gzip',date_format='%Y-%m-%d')
# Reconcile every monthly slice to pooled errors, rather than average monthly RMSE/hit rates.
for name,g in summary.groupby('strategy'):
    slices=monthly[monthly.strategy.eq(name)];row=g.iloc[0]
    assert slices.n.sum()==row.n
    assert np.isclose(np.average(slices.rmse**2,weights=slices.n),row.rmse**2)
    assert slices.price_error_le_1pct_n.sum()==row.price_error_le_1pct_n
    assert slices.price_error_le_2pct_n.sum()==row.price_error_le_2pct_n
    p=pred[pred.strategy.eq(name)];y=p.actual_log_return.to_numpy();q=p.predicted_log_return.to_numpy()
    assert np.isclose(row.mse_skill_vs_zero,1-np.sum((q-y)**2)/np.sum(y*y))
    if (q!=0).any():assert np.isclose(row.hit_rate_pct,100*np.sum((np.sign(q)==np.sign(y))&(y!=0))/np.sum(y!=0))
assert guarded_hashes()==before
save_json(out/'verification.json',{'same_pairs_per_strategy':104,'strategies':len(summary),'monthly_groups':len(monthly),
    'source_close_truth_anchors_verified':True,'monthly_to_pooled_reconciled':True,'refits':0,'canonical_unchanged':True})
short={'LSTM_serving':'LSTM serving','XGBoost_serving':'XGBoost serving','Ridge_serving':'Ridge serving',
    'Ridge_research_new':'Ridge train <=2023','refit_2025_native':'Ridge train <=2025',
    'refit_2025_native_bias60':'Ridge <=2025 + bias','monthly_refit_native':'Ridge monthly raw',
    'monthly_refit_native_bias60':'Ridge monthly + bias','Zero-Return':'Zero Return'}
focus=list(short)
grid=monthly[monthly.strategy.isin(focus)].pivot(index='strategy',columns='target_month',values='mse_skill_vs_zero').reindex(focus)*100
fig,ax=plt.subplots(figsize=(10,5.5));limit=max(abs(grid.min().min()),abs(grid.max().max()))
im=ax.imshow(grid,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto');ax.set_yticks(range(len(grid)),[short[x] for x in grid.index]);ax.set_xticks(range(len(grid.columns)),grid.columns)
for i in range(len(grid)):
    for j in range(len(grid.columns)):
        v=grid.iloc[i,j];ax.text(j,i,f'{v:+.2f}%',ha='center',va='center',color='white' if abs(v)>.6*limit else 'black')
fig.colorbar(im,ax=ax,label='MSE skill vs Zero Return (%)')
ax.set_title('FPT next-observation forecasts · same dates per month\nJun–Oct 2026 (October through 29th); positive skill is better')
fig.tight_layout();fig.savefig(out/'monthly_skill.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,sharey=True)
for ax,name in zip(axes,['LSTM_serving','XGBoost_serving','refit_2025_native']):
    g=pred[pred.strategy.eq(name)]
    ax.plot(g.Target_Date_1d,100*np.expm1(g.actual_log_return),color='#555',lw=.9,label='Actual close-to-close return')
    ax.plot(g.Target_Date_1d,100*np.expm1(g.predicted_log_return),color='#1674a5',lw=1.5,label='Predicted return')
    ax.axhline(0,color='black',lw=.5);ax.set_title(short[name]);ax.set_ylabel('Return (%)');ax.grid(alpha=.2)
axes[0].legend(fontsize=9,ncol=2);axes[-1].set_xlabel('Target date · not intraday High–Low')
fig.suptitle('Forecast amplitude vs actual movement · observed context updated each day');fig.tight_layout();fig.savefig(out/'daily_returns.png',dpi=150);plt.close(fig)
rows=['| Model / strategy | RMSE log-return | MAE giá adj. VND | Hit % | Balanced % | Sai số giá ≤1% | ≤2% | MSE skill vs Zero % |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for name in focus:
    v=summary[summary.strategy.eq(name)].iloc[0]
    rows.append(f'| {short[name]} | {v.rmse:.7f} | {v.price_mae_adjusted_vnd:.2f} | {v.hit_rate_pct:.2f} | {v.balanced_accuracy_pct:.2f} | {v.price_error_le_1pct_pct:.2f}% | {v.price_error_le_2pct_pct:.2f}% | {100*v.mse_skill_vs_zero:+.2f}% |')
counts=anchor.groupby(anchor.Target_Date_1d.dt.to_period('M')).size()
text=f'''# FPT: cùng tập tháng 6–10/2026

Đọc lại forecast đã lưu; **không train hoặc tải nguồn mới**. Tất cả {len(summary)} strategy có đúng104 cặp origin/target chung. Target từ {anchor.Target_Date_1d.min().date()} đến {anchor.Target_Date_1d.max().date()}, October chỉ đến29/10. Số phiên theo tháng: {'; '.join(f'{k}: {v}' for k,v in counts.items())}. Ngày target xác định tháng; origin có thể là phiên cuối tháng trước. Primary common Yahoo/Vietcap: loại hai cặp không khớp trong tháng9 (xem audit nguồn cũ), không cho từng model tự chọn ngày. Chẩn đoán hồi cứu trên khoảng đã xem, không phải holdout mới hoặc forecast đã phát live.

## Cách đọc

- Mỗi origin đã biết OHLCV đến hết phiên t; dự báo close/return quan sát giao dịch kế tiếp. Không có thông tin t+1 trong features. **Không phải đứng đầu tháng6 rồi dự báo liên tục cả5 tháng mà không cập nhật giá thực tế.**
- Fixed weights/scalers chỉ học nhãn có target trước2026 so riêng với luồng monthly/bias/rolling dùng nhãn quá khứ đã chín. Không lấy shadow model train đến29/10 để chấm lại tháng6–9. Current adjusted history chưa point-in-time certified; khác nguồn/features/train cutoff làm so family chưa cô lập riêng thuật toán.
- Thực tế là reference Vietcap adjusted close trên cả origin và target; so biên độ **close-to-close**, không phải High–Low trong ngày. Source chéo KBS được đối chiếu ở study cũ.
- MAE/RMSE giá đo VND adjusted; RMSE return là log-return. `return_rmse_pp`/`return_mae_pp` so **simple return** ở đơn vị điểm phần trăm; `magnitude_mae_pp` so absolute return, phải đọc cùng đúng hướng (đoán+2% và thực tế−2% có cùng độ lớn nhưng sai hướng).
- Hit trên actual khác0; balanced accuracy trung bình recall tăng/giảm. Zero Return là abstention, Hit/Balanced N/A. `price_error_le_1pct_pct` = tỷ lệ abs(predicted_price/actual_price−1)≤0.01, tương tự2%; đây là tỷ lệ đạt ngưỡng đã định nghĩa, không xác suất forecast tiếp theo đúng. Không có một accuracy duy nhất cho hồi quy và MAPE≠1−accuracy.
- Độ lệch chuẩn return dự báo/thực tế cho biết model có co về0 quá mạnh; tỷ lệ thấp chỉ là mô tả biên độ, không tự chứng minh nên tăng độ nhạy. MSE skill cần hơn Zero Return trên cùng ngày. Không chọn model bằng tháng điểm đẹp.

## Kết quả gộp

{chr(10).join(rows)}

Gộp từ mọi forecast, không lấy trung bình không trọng số RMSE/Hit từng tháng. `summary.csv` giữ đầy đủ17 strategy, recalls tăng/giảm, ngưỡng giá, return amplitude và block-bootstrap95% CI mô tả (10 origin,2.000 resamples,seed20261030). CI không điều chỉnh multiple testing/adaptive research. `monthly_metrics.csv` có đủ từng tháng.

![Đối chứng monthly skill](monthly_skill.png)

![Biên độ close-to-close forecast và thực tế](daily_returns.png)

## Nguồn và tái lập

`analysis.py`: phép tính; `scope.json`: hash input/source và phạm vi; `predictions.csv.gz`: các dự báo đã căn cùng ngày; `verification.json`: đối chiếu trực tiếp actual return/giá với raw Vietcap ZIP, đối soát N/pooled errors và8 hash canonical. Source studies: `reports/evaluation_2026/20261030T014012158951Z/` và `reports/recency_bias/20261030T022440928507Z/`. Không thay serving/API/Streamlit và không commit/push. Chạy lại script tạo ID phân tích mới, giữ lịch sử.
'''
(out/'COMPARISON.md').write_text(text,encoding='utf-8')
print('OUTPUT',out)
print(summary[summary.strategy.isin(focus)][['strategy','n','rmse','price_mae_adjusted_vnd','hit_rate_pct','balanced_accuracy_pct','price_error_le_1pct_pct','price_error_le_2pct_pct','mse_skill_vs_zero','predicted_to_actual_std_ratio']].to_string(index=False))
print('UP/DOWN/NEUTRAL',sum(anchor.actual_log_return>0),sum(anchor.actual_log_return<0),sum(anchor.actual_log_return==0))
