"""Executed, read-only notebook for the FPT gap/body feature experiment."""
from pathlib import Path
import argparse
import hashlib
import json
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'notebooks/08_gap_body_feature_comparison_fpt.ipynb'


def build():
    items=json.loads((ROOT/'reports/candle_ablation/studies.json').read_text())
    study=ROOT/'reports/candle_ablation'/[s for s in items if s['status']=='complete'][-1]['study_id']
    files=['protocol.json','verification.json','metrics.csv','predictions.csv.gz','feature_label_snapshot.csv.gz','primary_summary.csv',
           'paired_effects.csv','validation_scores.csv','feature_selection.csv','validation_selected_evaluation.csv','threshold_selection.csv','jun_oct_metrics.csv']
    hashes={f:hashlib.sha256((study/f).read_bytes()).hexdigest() for f in files}
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
    def code(s):cells.append(nbf.v4.new_code_cell(s.strip(),metadata={'jupyter':{'source_hidden':True}}))
    md('''# FPT — Gap mở cửa và thân nến có giúp model không?

## Câu hỏi và cách đọc
**A:** 6 feature hiện tại. **B:** thay `Log_Return` bằng `Gap_Log` và `Body_Log` → 7 feature. Chỉ dữ liệu FPT, cùng ngày/nhãn/fold/cấu hình model. Giữ cả kết quả tốt và xấu; không mặc định B tốt hơn.

Xem phần 1 để hiểu feature, phần 2–3 xem thay đổi metric và khoảng bất định, phần 4 xem giá dự báo/thực tế, phần 5–6 xem hướng/xác suất và selective. Notebook đọc kết quả đã lưu, **không train hoặc tải giá**. Toàn bộ giai đoạn2022–2026 đã được xem, là nghiên cứu hồi cứu thích nghi, chưa chứng minh lợi thế forward.''')
    md('''**Kết quả lượt này: chưa đủ căn cứ thay A.** Mean-fold RMSE 2022–2024 của Ridge_recent tăng 0,0162190 → 0,0162930 (cao hơn 0,46%). Logistic raw balanced tăng rất ít, 51,83% → 51,92% (+0,09 điểm %), CI từng fold vẫn chứa 0. Trên 2026, Ridge_recent B có RMSE cao hơn A 1,44%, Hit nonzero giảm 50,76% → 44,67%.

Giữ cả năm 2025 mà B có điểm RMSE tốt hơn, cùng các năm kém hơn. Không chọn riêng 2025 để tuyên bố cải thiện toàn bộ. Chưa promote feature/model.''')
    md('''## Phương pháp và giới hạn
- Gap tại origin t = log(Open_t / Close_previous_eligible); body = log(Close_t / Open_t). Cộng lại bằng Log_Return_t. **Chờ phiên t đóng cửa**, dùng feature đó đoán quan sát giao dịch tiếp theo. Không dùng Open của phiên cần đoán.
- Previous eligible đúng chuỗi positive-volume/valid-candle của pipeline cũ; không mặc định ngày lịch liền trước, không fill. Giá cùng basis adjusted Vietcap; chưa chứng nhận point-in-time hay lịch sàn.
- Ridge_shared và Logistic học cùng các dòng nonzero đến30/6 năm trước; Q3 sigmoid, Q4 chọn margin; evaluation năm sau. Ridge_recent train cả window3y đến cuối năm trước (gồm return0), cùng cutoff A/B, là đối chứng giá gần hơn.
- Hyperparameters/seed/threshold grid không đổi. A tái dùng artifact cũ, B train mới. A/B trong từng strategy cùng mọi phase; không chọn feature bằng evaluation. Chọn A/B từ validation chỉ cho Ridge_shared và Logistic_raw; không dùng validation đã nằm trong training của Ridge_recent.
- Classifier không có giá dự báo. Actual0 giữ trong evaluation; call vào actual0 tính sai, abstain không được tính đúng đứng giá. Hit nonzero, accuracy mọi calls và coverage khác nhau.
- Alpha/C giữ cố định nhưng đổi cơ sở feature và scaling có thể đổi regularization hiệu dụng; chênh điểm không chứng minh tác động nhân quả của gap/body. Đọc cả các fold và CI, không chọn một tháng đẹp.''')
    md('''## Dữ liệu và tham số
Đổi `VIEW_MONTH` để phóng to một tháng2026, rồi Run all. Bảng/chart gộp vẫn giữ cả giai đoạn. Cần numpy/pandas/matplotlib/IPython và thư mục study; không cần weights. Trên Colab cần tải cả thư mục kết quả và đặt `DATA_DIR` nếu không chạy từ repo. Outputs đã lưu để xem ngay.''')
    code(f'''from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
VIEW_MONTH='2026-10'
DATA_DIR=None
STUDY_ID={study.name!r}
if DATA_DIR is None:
    DATA_DIR=next((p/'reports/candle_ablation'/STUDY_ID for p in [Path.cwd(),*Path.cwd().parents]
                   if (p/'reports/candle_ablation'/STUDY_ID/'metrics.csv').exists()),None)
if DATA_DIR is None:raise FileNotFoundError('Đặt DATA_DIR tới thư mục kết quả candle_ablation của study.')
DATA_DIR=Path(DATA_DIR)
EXPECTED_SHA256={hashes!r}
for f,sha in EXPECTED_SHA256.items():assert hashlib.sha256((DATA_DIR/f).read_bytes()).hexdigest()==sha,f
read=lambda f:pd.read_csv(DATA_DIR/f)
metrics=read('metrics.csv');primary=read('primary_summary.csv');effects=read('paired_effects.csv')
pred=read('predictions.csv.gz');feature=read('feature_label_snapshot.csv.gz')
for frame in [pred,feature]:
    frame['Date']=pd.to_datetime(frame.Date);frame['Target_Date_1d']=pd.to_datetime(frame.Target_Date_1d)
selection=read('feature_selection.csv');selected=read('validation_selected_evaluation.csv');thresholds=read('threshold_selection.csv')
validation=read('validation_scores.csv');june_oct=read('jun_oct_metrics.csv')
verification=json.loads((DATA_DIR/'verification.json').read_text())
assert np.allclose(feature.Gap_Log+feature.Body_Log,feature.Log_Return,atol=1e-12)
ev=pred[pred.phase=='evaluation'].copy()
for (fold,strategy),group in ev.groupby(['fold','strategy']):
    a=group[group.case=='A_returns6'];b=group[group.case=='B_gap_body7']
    assert list(zip(a.Date,a.Target_Date_1d))==list(zip(b.Date,b.Target_Date_1d))
    assert np.allclose(a.actual_log_return,b.actual_log_return)
    for case,g in group.groupby('case'):
        saved=metrics[(metrics.fold==fold)&(metrics.strategy==strategy)&(metrics.case==case)].iloc[0]
        calls=g.direction.ne(0);correct=calls&(g.direction==np.sign(g.actual_log_return))
        assert len(g)==saved.n and calls.sum()==saved.calls_n and correct.sum()==saved.correct_n
        if g.predicted_log_return.notna().all():
            assert np.isclose(np.sqrt(np.mean((g.predicted_log_return-g.actual_log_return)**2)),saved.return_rmse)
assert VIEW_MONTH in ev[ev.year==2026].Target_Date_1d.dt.strftime('%Y-%m').unique()
QA=Path.cwd()/'.cache/candle-notebook-qa';QA.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({{'figure.figsize':(12,4.5),'font.size':11,'axes.grid':True,'grid.alpha':.18}})
COLORS={{'A_returns6':'#1674a5','B_gap_body7':'#277c62'}}
def show(name):
    plt.tight_layout();plt.savefig(QA/(name+'.png'),dpi=140,bbox_inches='tight');plt.show()
def table(frame):display(frame.style.format(precision=3,na_rep='—'))
display(Markdown(f"Study **{{STUDY_ID}}** · {{verification['artifacts_reloaded']}} artifacts reload · {{verification['A_control_prediction_rows_matched']}} forecast rows A khớp control · không train trong notebook."))
''')
    md('''## 1. Gap và thân nến — cùng origin, không phải target tương lai
Các cột dưới tính bằng log-return ×100 (% log-return), không phải simple return. Đường đen là tổng current log-return, không phải return phiên kế tiếp. Bảng giữ Open/Previous Close để có thể kiểm tra công thức.''')
    code('''g=feature[feature.Date.dt.strftime('%Y-%m')==VIEW_MONTH].copy()
fig,ax=plt.subplots(figsize=(13,4.5))
ax.bar(g.Date-pd.Timedelta(hours=5),100*g.Gap_Log,width=.38,color='#c57925',label='Gap qua đêm')
ax.bar(g.Date+pd.Timedelta(hours=5),100*g.Body_Log,width=.38,color='#1674a5',label='Thân nến trong phiên')
ax.plot(g.Date,100*g.Log_Return,'o-',ms=3,color='#222',lw=1,label='Log_Return = gap + body')
ax.axhline(0,color='black',lw=.7);ax.set_xlabel('Ngày origin đã đóng cửa');ax.set_ylabel('Log-return ×100')
ax.set_title(VIEW_MONTH+' · tách chuyển động qua đêm / trong phiên');ax.legend(ncol=3,fontsize=9)
show('01_gap_body')
table(g[['Date','Open','Previous_Eligible_Date','Previous_Eligible_Close','Close','Gap_Log','Body_Log','Log_Return']].head(8))
display(Markdown(f"Sai khác identity lớn nhất trên toàn dữ liệu: **{abs(feature.Gap_Log+feature.Body_Log-feature.Log_Return).max():.2e}**. Dữ liệu canonical không thay đổi."))''')
    md('''## 2. Sai số Ridge: B có tốt hơn A không?
Điểm dương là RMSE của B thấp hơn A: `100 × (1 − RMSE_B/RMSE_A)`. Khoảng95% mô tả dùng paired block-bootstrap10 origin/2.000 resamples, seed cố định; không xử lý hết adaptive research/multiple testing. 2025/2026 là diagnostic, không chọn cấu hình bằng các điểm này.''')
    code('''fig,axes=plt.subplots(1,2,figsize=(14,4.5),sharey=True)
for ax,strategy in zip(axes,['Ridge_shared','Ridge_recent']):
    g=effects[effects.strategy==strategy].sort_values('year');x=g.year.to_numpy()
    ax.vlines(x,g.rmse_gain_pct_ci_low,g.rmse_gain_pct_ci_high,color='#277c62');ax.scatter(x,g.rmse_gain_pct,color='#277c62')
    ax.axhline(0,color='black',ls='--');ax.set_xticks(x);ax.set_title(strategy);ax.set_xlabel('Evaluation year');ax.set_ylabel('Giảm RMSE B so A (%) · 95% CI')
show('02_ridge_rmse_gain')
table(primary[primary.strategy.str.startswith('Ridge')].set_index(['strategy','case'])[['mean_fold_return_rmse','worst_fold_return_rmse','mean_fold_balanced']])''')
    md('''## 3. Đoán hướng và lựa chọn bằng validation
Balanced change là B−A, đơn vị điểm phần trăm; dương tốt hơn. Chỉ so forced methods có cùng mẫu. Selective có thể giữ ngày khác nhau nên không gán cùng một CI cải thiện. Bảng lựa chọn A/B lấy từ validation, không từ evaluation.''')
    code('''fig,axes=plt.subplots(1,2,figsize=(14,4.5),sharey=True)
for ax,strategy in zip(axes,['Ridge_shared','Logistic_raw']):
    g=effects[effects.strategy==strategy].sort_values('year');x=g.year.to_numpy()
    ax.vlines(x,g.balanced_change_pp_ci_low,g.balanced_change_pp_ci_high,color='#1674a5');ax.scatter(x,g.balanced_change_pp,color='#1674a5')
    ax.axhline(0,color='black',ls='--');ax.set_xticks(x);ax.set_title(strategy);ax.set_xlabel('Evaluation year');ax.set_ylabel('Balanced B−A (điểm %) · 95% CI')
show('03_direction_change')
table(primary[primary.strategy.isin(['Ridge_shared','Logistic_raw'])].set_index(['strategy','case'])[['mean_fold_balanced','mean_fold_hit_nonzero','mean_fold_coverage']])
display(selection.pivot(index='fold',columns='model',values='selected_case'))
table(selected[['fold','strategy','selected_case','return_rmse','hit_nonzero_called_pct','balanced_called_nonzero_pct']].set_index(['fold','strategy']))''')
    md('''## 4. Giá thực tế và hai bản Ridge trên cùng trục
Ridge_recent A/B cùng train window đến cuối năm trước, chỉ khác feature. Giá = actual close origin ×exp(predicted return), cùng anchor Vietcap adjusted; đường gần nhau chưa chứng minh tốt hơn Zero Return. Không phải dự báo toàn bộ năm mà không cập nhật giá origin.''')
    code('''r=ev[(ev.fold=='eval_2026')&(ev.strategy=='Ridge_recent')]
a=r[r.case=='A_returns6'];b=r[r.case=='B_gap_body7']
fig,axes=plt.subplots(2,1,figsize=(13,7))
for ax,mask,title in [(axes[0],np.ones(len(a),dtype=bool),'2026 đến29/10'),
                      (axes[1],a.Target_Date_1d.dt.strftime('%Y-%m').eq(VIEW_MONTH).to_numpy(),VIEW_MONTH)]:
    ga=a.loc[mask];gb=b.loc[mask]
    ax.plot(ga.Target_Date_1d,ga.Close*np.exp(ga.actual_log_return)/1000,color='#222',lw=2,label='Giá thực tế')
    ax.plot(ga.Target_Date_1d,ga.Close*np.exp(ga.predicted_log_return)/1000,color=COLORS['A_returns6'],ls='--',label='A · 6 returns')
    ax.plot(gb.Target_Date_1d,gb.Close*np.exp(gb.predicted_log_return)/1000,color=COLORS['B_gap_body7'],ls=':',label='B · gap/body7')
    ax.set_title(title);ax.set_ylabel('Nghìn VND adjusted');ax.set_xlabel('Ngày target')
axes[0].legend(ncol=3,fontsize=9);show('04_actual_predicted_price')
table(metrics[(metrics.year==2026)&metrics.strategy.str.startswith('Ridge')].set_index(['strategy','case'])[['n','return_rmse','return_mae','price_mae_adjusted_vnd','price_mape_pct','mse_skill_vs_zero']])''')
    md('''## 5. Classifier / “chưa đủ tín hiệu”: đọc accuracy cùng coverage
Sigmoid và margin mỗi case chọn bằng các phase cũ như nhau. Raw comparison giữ cùng threshold0.5 để thấy ảnh hưởng feature. Accuracy trên mọi calls gồm actual0 là sai; không so với Hit nonzero mà bỏ qua mẫu số.''')
    code('''v=metrics[(metrics.year==2026)&metrics.strategy.str.startswith('Logistic')].copy()
v['label']=v.case.str[:1]+' · '+v.strategy.str.replace('Logistic_','',regex=False)
fig,axes=plt.subplots(1,2,figsize=(14,4.7),sharey=True)
for ax,col,title in [(axes[0],'called_accuracy_all_pct','Accuracy trên calls'),(axes[1],'coverage_pct','Coverage mọi origin')]:
    values=v[col].to_numpy();ax.barh(range(len(v)),values,color=[COLORS[x] for x in v.case]);ax.set_yticks(range(len(v)),v.label)
    ax.set_xlim(0,100);ax.set_title('2026 · '+title)
    for i,value in enumerate(values):
        if np.isfinite(value):ax.text(max(2,value-1),i,f'{value:.1f}%',ha='right',va='center',fontsize=9,color='white' if value>20 else 'black')
show('05_classifier_coverage')
table(v.set_index(['case','strategy'])[['n','calls_n','abstain_n','hit_nonzero_called_pct','balanced_called_nonzero_pct','up_coverage_pct','down_coverage_pct']])
table(thresholds.set_index(['fold','case'])[['margin','reason']])''')
    md('''## 6. Xác suất từng phiên và cohort tháng6–10
Xem raw và sigmoid riêng. Chấm actual tăng/giảm tại1/0; actual0 đặt dấu x ở0.5 chỉ để hiển thị, không phải nhãn xác suất0.5. Đường xác suất không phải đường giá. Bảng104 phiên giữ đúng cohort Notebook06; không chuyển CI full2026 cho subset.''')
    code('''month=ev[(ev.fold=='eval_2026')&(ev.Target_Date_1d.dt.strftime('%Y-%m')==VIEW_MONTH)]
fig,axes=plt.subplots(2,1,figsize=(13,6.5),sharex=True,sharey=True)
for ax,strategy in zip(axes,['Logistic_raw','Logistic_calibrated']):
    for case in COLORS:
        g=month[(month.case==case)&(month.strategy==strategy)]
        ax.plot(g.Target_Date_1d,g.probability_up,'o-' if case=='A_returns6' else 's--',ms=3,color=COLORS[case],label=case)
    y=np.sign(g.actual_log_return.to_numpy());truth=np.where(y>0,1,np.where(y<0,0,.5));nz=y!=0
    ax.scatter(g.loc[nz,'Target_Date_1d'],truth[nz],color='#333',marker='|',s=70,label='Actual tăng/giảm')
    ax.scatter(g.loc[~nz,'Target_Date_1d'],truth[~nz],color='#aa6a28',marker='x',label='Actual đứng giá')
    ax.axhline(.5,color='black',ls=':',lw=.7);ax.set_ylim(-.08,1.08);ax.set_title(strategy+' · '+VIEW_MONTH);ax.set_ylabel('Ước lượng p(tăng)')
axes[0].legend(fontsize=9,ncol=2);axes[-1].set_xlabel('Ngày target');show('06_daily_probabilities')
table(june_oct.set_index(['strategy','case'])[['n','calls_n','return_rmse','price_mae_adjusted_vnd','hit_nonzero_called_pct','balanced_called_nonzero_pct','coverage_pct']])''')
    md('''## Kết luận và nguồn
Không chốt bằng một tháng hoặc chỉ nhìn đường giá. Đọc A/B trên cùng ngày, validation-selected evaluation, độ lớn thay đổi và CI. Cải thiện direction không tự có nghĩa cải thiện sai số giá; selective phải đọc thêm `same_selected_days.csv`.

Lượt này, B chưa cải thiện ổn định: Logistic raw chỉ tăng mean-fold balanced khoảng 0,09 điểm %, Ridge RMSE trung bình cao hơn. Lựa chọn feature bằng validation cũng không tự bảo đảm evaluation tốt hơn. B selective/fold2023 giữ 136/248 phiên (54,84%) và Hit nonzero 41,46%; coverage tối thiểu60% là điều kiện trên validation, không phải cam kết trên evaluation. Không retune margin dựa vào đoạn đã chấm.

Nguồn cùng study: `REPORT.md`, `DECISION.md`, `protocol.json`, `manifest.json`, `verification.json`, `feature_label_snapshot.csv.gz`, các CSV metrics/predictions/paired_effects. Source/package versions/input hashes lưu trước fit. Case A khớp forecasts đã công bố của study direction trước; model và schema canonical không đổi. Weights nghiên cứu bị gitignore, không cần để xem notebook.

Chỉ nghiên cứu FPT. Không tải nguồn mới, không đổi horizon, không sửa API/Streamlit hoặc promote. Các giai đoạn đã xem, cần các phiên mới để kiểm chứng forward. Không commit/push.''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':'3.13'}})
    nbf.validate(nb);nbf.write(nb,OUTPUT);return nb


def execute(nb):
    import sys
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    try:NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        if km.has_kernel:km.shutdown_kernel(now=True)
    nbf.validate(nb);nbf.write(nb,OUTPUT);print(f'Executed {OUTPUT}; cells={len(nb.cells)}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args();nb=build()
    if args.execute:execute(nb)
