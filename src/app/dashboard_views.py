"""Dashboard views backed by reviewed HTTP data; no model loading or fitting."""
import json
from html import escape
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.app.theme import MINT, BLUE, AMBER, PURPLE, COLORS, chart, hero, note, path_chart

HORIZONS = {'Hôm nay':'today','Ngày mai':'tomorrow','1 tuần':'week','1 tháng':'month'}
MODEL_LABELS = {'Ridge+bias60':'Ridge · hiệu chỉnh bias', 'Ridge':'Ridge · nguyên bản', 'Zero-Return':'Zero Return · baseline'}


def forecast_view(api, history, health, registry):
    hero('FPT / Forecast studio', 'Từ dữ liệu đã có, nhìn về phía trước.',
         'Chọn khoảng dự báo, quan sát đường giá và xem vùng kịch bản. Một mã FPT, cùng một mô hình đã được đóng gói có phiên bản.')
    a,b = st.columns([2,1])
    with a:
        horizon = st.segmented_control('Kỳ dự báo', list(HORIZONS), default='Hôm nay') or 'Hôm nay'
    with b:
        chosen = st.selectbox('Model dự báo', registry['models'], format_func=lambda n:MODEL_LABELS.get(n,n))
    with st.spinner('Đang tính đường dự báo…'):
        result = api('/api/v1/forecast', {'model':chosen, 'horizon':HORIZONS[horizon]})
    final = result['path'][-1]
    day = pd.Timestamp(result['effective_target_date']).strftime('%d/%m/%Y')
    a,b,c,d = st.columns(4)
    a.metric('Giá origin · VND', f"{result['origin_close_adjusted']:,.0f}", help=f"Giá thực tế tại {result['origin_date']}, theo basis điều chỉnh của nguồn.")
    b.metric('Giá cuối khung · VND', f"{final['predicted_close_adjusted']:,.0f}", f"{final['change_from_origin_pct']:+.2f}% so origin", delta_color='off')
    c.metric('Ngày đích dự kiến', day, help='Chỉ bỏ cuối tuần; chưa xác minh lịch nghỉ lễ HOSE.')
    d.metric('Số bước từ origin', str(result['steps']), help='Model học h=1; từ bước thứ 2 dùng giá do chính model dự báo.')
    st.write('')
    if result['weekend_shifted']:
        st.info(f"Ngày chọn {result['requested_end']} rơi vào cuối tuần → chuyển sang ngày làm việc dự kiến {result['effective_target_date']}. Chưa trừ lịch nghỉ lễ.")
    if result['steps'] > 1:
        st.warning('Mô phỏng nhiều bước · model hiện chỉ được train cho h=1. Khung tuần/tháng chưa có đánh giá riêng; sai số có thể tích lũy.')
    else:
        st.caption('Ước tính từ snapshot trước phiên. Bấm dự báo trong hôm nay không đồng nghĩa forecast đã được phát trước giờ mở cửa.')
    chart(path_chart(history,result), height=460)
    st.caption(f"Đường nét liền: dữ liệu thật đến {result['origin_date']}. Nét đứt: dự báo/mô phỏng. Vùng tô: P10–P90 của kịch bản residual bootstrap, không bảo đảm 80% kết quả thật nằm trong vùng.")
    a,b,c = st.columns(3)
    with a:
        note('01 / Học từ lợi suất', 'Ridge dùng 6 feature từ giá FPT. Không tải hoặc tạo giả giá mở cửa, High/Low hay volume của các ngày tương lai.')
    with b:
        note('02 / Mở rộng có giới hạn', f"Khung {horizon.lower()} có {result['steps']} bước từ snapshot. Tối đa 23 bước ngày làm việc, không tự kéo dài nếu dữ liệu quá cũ.")
    with c:
        note('03 / Nhìn cả bất định', 'Biên mô phỏng dùng các block 5 sai số lịch sử. Không bao gồm đầy đủ bất định nguồn, tham số hoặc một regime chưa từng xuất hiện.')
    with st.expander('Bảng dự báo từng bước & tải kết quả'):
        data = pd.DataFrame(result['path']).rename(columns={'estimated_date':'Ngày dự kiến','step':'Bước',
            'predicted_close_adjusted':'Giá dự báo','scenario_lower':'Kịch bản P10','scenario_upper':'Kịch bản P90',
            'change_from_origin_pct':'Thay đổi từ origin (%)'})
        st.dataframe(data[['Bước','Ngày dự kiến','Giá dự báo','Kịch bản P10','Kịch bản P90','Thay đổi từ origin (%)']],
                     hide_index=True, width='stretch',column_config={k:st.column_config.NumberColumn(format='%.2f') for k in ['Giá dự báo','Kịch bản P10','Kịch bản P90','Thay đổi từ origin (%)']})
        a,b=st.columns(2)
        a.download_button('Tải CSV đường dự báo',pd.DataFrame(result['path']).to_csv(index=False),'fpt_forecast_path.csv','text/csv')
        b.download_button('Tải JSON có metadata',json.dumps(result,ensure_ascii=False,indent=2),'fpt_forecast.json','application/json')
    with st.expander('Cách tính, ngày và phiên bản'):
        st.write('Hôm nay/ngày mai: ngày lịch đã chọn, dời cuối tuần sang ngày làm việc kế. Một tuần: 7 ngày lịch tính cả hôm nay. Một tháng: đến trước cùng ngày của tháng sau. Ngày cuối khung tuần/tháng bỏ cuối tuần.')
        st.write('Từng bước: cập nhật 6 return feature từ chuỗi Close, dùng lại Ridge/scaler và bias đã đóng băng. Không fit lại, không dùng target thực tế chưa biết. Baseline Zero Return giữ nguyên giá và không có vùng residual của Ridge.')
        st.json({k:v for k,v in result.items() if k!='path'})


def overview_view(api, history, health):
    hero('FPT / Market overview', 'Một cổ phiếu. Nhiều góc nhìn.',
         'Giá, biên độ và khối lượng thực tế từ snapshot FPT. Khám phá dữ liệu trước khi đọc kết quả mô hình.')
    raw = pd.DataFrame(api('/api/v1/fpt/ohlcv?limit=500')['observations'])
    raw['Date']=pd.to_datetime(raw.Date)
    raw=raw.loc[raw.Volume.gt(0)].sort_values('Date')
    last,previous = raw.iloc[-1],raw.iloc[-2]
    change=(last.Close/previous.Close-1)*100
    a,b,c,d=st.columns(4)
    a.metric('Đóng cửa điều chỉnh',f'{last.Close:,.0f} VND',f'{change:+.2f}% phiên trước',delta_color='off')
    b.metric('Khối lượng phiên cuối',f'{last.Volume/1e6:.2f} triệu')
    c.metric('Biên độ High–Low',f'{(last.High-last.Low)/last.Close*100:.2f}%',help='Biên độ trong phiên cuối; khác lợi suất close-to-close.')
    d.metric('Ngày dữ liệu cuối',last.Date.strftime('%d/%m/%Y'))
    count=st.select_slider('Khoảng quan sát',options=[30,60,120,252],value=120,format_func=lambda n:f'{n} quan sát')
    part=raw.tail(count)
    figure=go.Figure(go.Candlestick(x=part.Date,open=part.Open,high=part.High,low=part.Low,close=part.Close,
        increasing_line_color=MINT,decreasing_line_color='#f18c9a',name='FPT OHLC điều chỉnh'))
    figure.update_layout(xaxis_rangeslider_visible=False,yaxis_title='VND · giá điều chỉnh')
    chart(figure,460)
    fig=px.bar(part,x='Date',y='Volume',labels={'Volume':'Khối lượng (cổ phiếu)','Date':''},color_discrete_sequence=[BLUE])
    chart(fig,220)
    a,b=st.columns(2)
    with a:note('Snapshot, không phải bảng giá realtime',f"Nguồn {health['source']}, đến {health['data_as_of']}. Dữ liệu không tự cập nhật khi chuyển trang hoặc bấm dự báo.")
    with b:note('Không làm phẳng ngày không giao dịch','Giá được điều chỉnh theo nhà cung cấp. Không điền ngày lễ bằng giá phiên trước; chưa xác minh lịch sàn hoặc dữ liệu point-in-time độc lập.')
    with st.expander('OHLCV thật đang hiển thị'):
        st.dataframe(part.sort_values('Date',ascending=False),hide_index=True,width='stretch')
        st.download_button('Tải CSV OHLCV',part.to_csv(index=False),'fpt_actual_ohlcv.csv','text/csv')


def benchmark_view(api):
    hero('FPT / Evidence room', 'Đặt dự báo cạnh kết quả thực tế.',
         'Các model lịch sử được chấm trên cùng ngày đích và cùng nhãn. Chọn tháng để thấy khi nào model tốt hơn, khi nào kém baseline.')
    a,b=st.columns([2,1])
    with a:scope_label=st.selectbox('Tập ngày dùng chung',['2026 đến 29/10 · 202 cặp','Tháng 6–10/2026 · 104 cặp, có LSTM'])
    scope='full_2026' if scope_label.startswith('2026') else 'jun_oct_2026'
    months=range(1,11) if scope=='full_2026' else range(6,11)
    with b:month=st.selectbox('Tháng của nhãn thực tế',['Toàn giai đoạn']+[f'2026-{m:02d}' for m in months])
    result=api(f'/api/v1/backtest?scope={scope}'+(f'&month={month}' if month!='Toàn giai đoạn' else ''))
    frame=pd.DataFrame(result['predictions']);metrics=pd.DataFrame(result['metrics'])
    best=metrics.loc[metrics.rmse.idxmin()]
    a,b,c=st.columns(3)
    a.metric('Số cặp ngày / model',str(int(best.n)))
    b.metric('RMSE thấp nhất trong tập',f'{best.rmse:.6f}',delta=str(best.strategy),delta_color='off')
    c.metric('Actual khác 0 / đứng giá',f"{int(best.nonzero_actual_n)} / {int(best.neutral_actual_n)}")
    selected=st.multiselect('Đường dự báo muốn xem',frame.strategy.drop_duplicates().tolist(),default=frame.strategy.drop_duplicates().tolist())
    actual=frame.drop_duplicates('Target_Date_1d').sort_values('Target_Date_1d')
    fig=go.Figure(go.Scatter(x=actual.Target_Date_1d,y=actual.actual_close_adjusted,name='Giá thực tế',line={'color':'#d9e4ef','width':2.8}))
    for i,name in enumerate(selected):
        part=frame.loc[frame.strategy.eq(name)].sort_values('Target_Date_1d')
        fig.add_trace(go.Scatter(x=part.Target_Date_1d,y=part.predicted_close_adjusted,name=name,
            line={'color':COLORS[i%len(COLORS)],'width':1.7,'dash':'dash' if name=='Zero-Return' else 'solid'}))
    fig.update_layout(yaxis_title='Giá điều chỉnh · VND')
    chart(fig,450)
    st.info('Backtest hồi cứu của model lịch sử, không phải điểm của bundle train đến 29/10/2026. Dự báo một bước có cập nhật context thật mỗi origin; không phải đường mô phỏng nhiều bước ở trang Dự báo.')
    display=metrics.rename(columns={'strategy':'Model','n':'Số cặp','rmse':'RMSE return','price_mae_adjusted_vnd':'MAE giá (VND)',
        'hit_rate_pct':'Hit (%)','balanced_accuracy_pct':'Balanced (%)','direction_coverage_pct':'Coverage (%)'})
    st.dataframe(display[['Model','Số cặp','RMSE return','MAE giá (VND)','Hit (%)','Balanced (%)','Coverage (%)']],hide_index=True,width='stretch',
                 column_config={'RMSE return':st.column_config.NumberColumn(format='%.6f'),**{c:st.column_config.NumberColumn(format='%.2f') for c in ['MAE giá (VND)','Hit (%)','Balanced (%)','Coverage (%)']}})
    st.caption('Hit trên actual khác 0, giữ prediction 0 trong mẫu số khi có tín hiệu. Zero Return không phát hướng nên N/A. Balanced là trung bình recall tăng/giảm; không đồng nghĩa xác suất giá nằm trong khoảng dự báo.')
    a,b=st.columns(2)
    errors=frame.copy();errors['Sai số giá (VND)']=errors.predicted_close_adjusted-errors.actual_close_adjusted
    with a:
        st.subheader('Phân bố sai số giá')
        fig=px.histogram(errors,x='Sai số giá (VND)',color='strategy',nbins=30,barmode='overlay',opacity=.55)
        fig.update_layout(yaxis_title='Số dự báo',legend_font_size=10)
        chart(fig,340)
    with b:
        st.subheader('Sai số theo tháng')
        errors['Tháng']=errors.Target_Date_1d.str[:7]
        errors['squared_error']=(errors.predicted_log_return-errors.actual_log_return)**2
        monthly=errors.groupby(['Tháng','strategy'],as_index=False).squared_error.mean()
        monthly['RMSE return']=np.sqrt(monthly.squared_error)
        fig=px.bar(monthly,x='Tháng',y='RMSE return',color='strategy',barmode='group')
        fig.update_yaxes(tickformat='.3f');chart(fig,340)
    with st.expander('Nguồn, điều kiện so sánh và bảng từng ngày'):
        st.write(result['source_report']);st.caption(result['lstm_context']);st.caption(result['bias_policy'])
        st.dataframe(frame,hide_index=True,width='stretch')
        st.download_button('Tải CSV tập ngày đang xem',frame.to_csv(index=False),'fpt_backtest.csv','text/csv')


def journey_view(api):
    hero('FPT / Research journal', 'Mỗi lượt thử đều để lại bằng chứng.',
         'Từ sửa pipeline, so nguồn đến Ridge và các nhánh không cải thiện. Ngày chạy nghiên cứu khác với thời điểm phát hành forecast.')
    stages=api('/api/v1/training-history')['stages']
    a,b,c=st.columns(3)
    a.metric('Giai đoạn đã ghi nhận',str(len(stages)))
    b.metric('Fit mới ở lượt giao diện','0')
    c.metric('Mã cổ phiếu sử dụng','FPT')
    st.write('')
    left,right=st.columns([1.4,1])
    with left:
        for stage in stages:
            st.markdown(f'''<div class="timeline"><small>{escape(stage['Ngày'])}</small><h4>{escape(stage['Giai đoạn'])}</h4>
            <p>{escape(stage['Công việc'])}</p><p><strong>{escape(stage['Kết luận'])}</strong></p></div>''',unsafe_allow_html=True)
    with right:
        note('Tiến bộ phải có đối chứng','Điểm giữa các nguồn, năm và mẫu số khác nhau không tạo thành một đường accuracy tăng theo ngày train. Chỉ kết luận từ các cặp so sánh phù hợp.')
        for name,caption in [('01_recency_bias.png','202 cặp 2026 · recency và bias'),('02_gap_body.png','A/B cùng ngày từng năm · gap và thân nến')]:
            st.image('docs/assets/training/'+name,caption=caption,width='stretch')
        st.caption('Hai biểu đồ từ báo cáo đã lưu; không có fit mới để vẽ hình.')
    with st.expander('Tra cứu toàn bộ đường dẫn evidence'):
        st.dataframe(pd.DataFrame(stages),hide_index=True,width='stretch')
        st.write('Tài liệu: docs/TRAINING_HISTORY.md · Notebook 05–08 lưu output và biểu đồ đối chứng.')


def methodology_view(api,health):
    hero('FPT / Data & methodology', 'Biết model đã nhìn thấy điều gì.',
         'Đầu vào, nhãn, cách chia thời gian và chất lượng nguồn là một phần của dự báo. Thông tin cần để kiểm tra và tái lập nằm ở đây.')
    result=api('/api/v1/data-quality');quality=result['quality']
    a,b,c,d=st.columns(4)
    a.metric('Dòng raw',f"{quality.get('raw_rows',0):,}")
    b.metric('Feature rows',f"{quality.get('feature_rows',0):,}")
    c.metric('Dòng quarantine',f"{quality.get('excluded_rows',0):,}")
    d.metric('Feature model chính',str(len(result['feature_columns'])))
    a,b=st.columns(2)
    with a:
        note('Supervised regression','X tại phiên t → nhãn log(Close_next / Close_t). Nhãn tự tính từ giá lịch sử vẫn là học có giám sát. Giá dự báo = Close_t × exp(return dự báo).')
        note('Tách theo thời gian','Scaler fit trên Train. Validation chọn cấu hình. Nhãn đi qua ranh giới phase bị loại. Năm đã xem không được gọi lại là holdout chưa chạm.')
        note('Ridge và LSTM có input khác nhau','Ridge nhận vector 6 feature, squared error + L2. LSTM reference dùng 30 quan sát × 16 feature, loss Huber. Không phải mọi model đều cần tensor 60 phiên.')
    with b:
        note('Nguồn và adjustment','Vietcap cho bundle local, Yahoo cho reference cũ, KBS đối chiếu nguồn. Giá theo basis điều chỉnh của provider; chưa chứng nhận lịch sàn hoặc point-in-time.')
        note('Classification và unsupervised','Đoán tăng/giảm là supervised classification. Clustering/PCA/HMM có thể nghiên cứu regime; fit chỉ trên Train. Chưa bật các nhánh này trong model chính.')
        note('Phạm vi hiện tại','Một mã FPT, h=1 đã train. Ngày tuần/tháng là ước tính lịch weekday và mô phỏng recursive. Chưa có kiểm định riêng cho h=5/h=21, không tự retrain hoặc deploy.')
    st.subheader('6 feature thực sự đi vào Ridge')
    st.code(' · '.join(result['feature_columns']),language=None,wrap_lines=True)
    st.caption('Builder chung tính 16 feature và warm-up 49 hàng để giữ tính tương thích; Ridge chỉ lấy allowlist 6 feature. Không dùng Close tuyệt đối làm feature model.')
    with st.expander('Quality report & metadata'):
        st.json(result)
    a,b=st.columns(2)
    a.link_button('Yahoo Finance · FPT.VN','https://finance.yahoo.com/quote/FPT.VN/history/')
    b.link_button('Vietcap · FPT','https://trading.vietcap.com.vn/iq/company?ticker=FPT&tab=overview&isIndex=false')


def roadmap_view(api):
    hero('FPT / Model strategy', 'Một model chính. Một đối thủ xứng đáng.',
         'Đơn giản hóa bản phục vụ, giữ toàn bộ evidence nghiên cứu. Model thứ hai phải bổ sung giá trị đo được, không chỉ để giao diện có thêm tên.')
    a,b=st.columns(2)
    with a:
        note('01 / Model chính — Ridge','Hướng đã chốt: Ridge6, alpha100, window3 năm. Ridge raw và Ridge+bias là hai cấu hình của cùng model; chưa phải ensemble hai model độc lập.')
        st.markdown('**Giữ:** inference nhẹ, giải thích được, artifact/version rõ. **Cần cải thiện:** kiểm định bias và độ ổn định trên dữ liệu mới; train/đánh giá horizon trực tiếp nếu cần tuần/tháng.')
    with b:
        note('02 / Challenger — chưa chốt người thắng','Giữ LSTM reference để so sánh. Tương lai chỉ phục vụ thêm LSTM hoặc XGBoost khi tốt hơn ổn định hoặc bổ sung lỗi khác Ridge, trên cùng nguồn/ngày/nhãn.')
        st.markdown('**Điều kiện:** nhiều fold, baseline phù hợp, ổn định seed và forecast mới phát trước target. Nếu không thêm giá trị, sản phẩm dùng một model là đủ.')
    st.subheader('Lộ trình hoàn thiện')
    rows=[('Đã có','Local studio','API + dashboard, version/hash, đối chứng và nhật ký.'),
          ('Đã có','Khung tuần / tháng','Recursive scenario tối đa23bước, chưa phải model đa horizon đã kiểm định.'),
          ('Tiếp theo','Kiểm định nhiều bước','Backtest recursive và direct h=5/h=21; so cùng origin, baseline và uncertainty. Chốt giao thức trước fit.'),
          ('Tiếp theo','Bằng chứng tương lai','Forecast phát trước target, chờ nhãn, rolling metrics/drift. Không dùng lại 2026 đã xem như holdout mới.'),
          ('Sau kiểm chứng','1–2 model phục vụ','Chỉ giữ challenger/ensemble nếu có lợi ích ngoài mẫu, cost và rollback phù hợp.'),
          ('Tương lai','Deploy có kiểm soát','Chọn host sau đo tài nguyên; HTTPS/auth, source refresh và lịch retrain là công việc riêng, chưa bật.')]
    st.dataframe(pd.DataFrame(rows,columns=['Trạng thái','Hạng mục','Điều kiện / kết quả']),hide_index=True,width='stretch')
    st.info('Zero Return luôn được giữ làm baseline kiểm chứng; không tính nó là model ML thứ hai. Không dùng Hit >50% hoặc một tháng đẹp để chọn production model.')
