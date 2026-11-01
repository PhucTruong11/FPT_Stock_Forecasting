"""Dashboard views backed by reviewed HTTP data; no model loading or fitting."""
import json
from html import escape
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.app.theme import MINT, BLUE, COLORS, chart, hero, note, path_chart
from src.app.analysis_view import render_market_analysis
from src.app.forecast_explainer import render_forecast_explainer
from src.app.forecast_analytics import render_forecast_analytics

HORIZONS = {'Hôm nay':'today','Ngày mai':'tomorrow','1 tuần':'week','1 tháng':'month','Chọn ngày':'custom'}
MODEL_LABELS = {'Ridge+bias60':'Ridge · hiệu chỉnh bias', 'Ridge':'Ridge · nguyên bản', 'Zero-Return':'Zero Return · baseline'}


def forecast_view(api, history, health, registry):
    hero('FPT / Forecast studio', 'Từ dữ liệu đã có, nhìn về phía trước.',
         'Chọn khoảng dự báo, quan sát đường giá và xem vùng kịch bản. Một mã FPT, cùng một mô hình đã được đóng gói có phiên bản.')
    a,b = st.columns([2,1])
    with a:
        horizon = st.segmented_control('Kỳ dự báo', list(HORIZONS), default='Hôm nay') or 'Hôm nay'
    with b:
        chosen = st.selectbox('Model dự báo', registry['models'], format_func=lambda n:MODEL_LABELS.get(n,n))
    payload={'model':chosen,'horizon':HORIZONS[horizon]}
    if horizon=='Chọn ngày':
        limits=health['forecast_date_limits']
        if not limits['custom_available']:
            st.info('Snapshot đã quá cũ để mở khoảng dự báo mới trong giới hạn 23 bước. Cần cập nhật dữ liệu trước khi chọn ngày.');return
        first_date=pd.Timestamp(limits['selectable_min_date']);last_date=pd.Timestamp(limits['selectable_max_date'])
        selected_date=st.date_input('Ngày muốn dự báo',value=min(first_date+pd.Timedelta(days=7),last_date).date(),
            min_value=first_date.date(),max_value=last_date.date(),format='DD/MM/YYYY',
            help='Trong một tháng từ hôm nay. Cuối tuần chuyển sang weekday kế tiếp; chưa xác minh nghỉ lễ HOSE.')
        if selected_date is None:
            st.info('Chọn một ngày đích để xem đường dự báo.');return
        payload['target_date']=selected_date.isoformat()
        st.caption(f"Chọn từ {first_date:%d/%m/%Y} đến {last_date:%d/%m/%Y} · tối đa {limits['max_steps']} bước từ snapshot. Khoảng ngày có thể ngắn hơn một tháng nếu dữ liệu cũ.")
    with st.spinner('Đang tính đường dự báo…'):
        result = api('/api/v1/forecast', payload)
    final = result['path'][-1]
    day = pd.Timestamp(result['effective_target_date']).strftime('%d/%m/%Y')
    a,b,c,d = st.columns(4)
    a.metric('Giá origin · VND', f"{result['origin_close_adjusted']:,.0f}", help=f"Giá thực tế tại {result['origin_date']}, theo basis điều chỉnh của nguồn.")
    b.metric('Giá cuối khung · VND', f"{final['predicted_close_adjusted']:,.0f}", f"{final['change_from_origin_pct']:+.2f}% so origin", delta_color='off')
    c.metric('Ngày đích dự kiến', day, help='Chỉ bỏ cuối tuần; chưa xác minh lịch nghỉ lễ HOSE.')
    d.metric('Số bước từ origin', str(result['steps']), help='Model học h=1; từ bước thứ 2 dùng giá do chính model dự báo.')
    st.write('')
    if result.get('bridge_steps_before_requested_window',0):
        st.info(f"Giá thật gần nhất dừng ở {result['origin_date']}. Đoạn xám nối snapshot tới trước {result['requested_start']} cũng là mô phỏng, không phải giá hôm nay đã quan sát. Đoạn xanh là khoảng dự báo đang chọn.")
    if result['weekend_shifted']:
        st.info(f"Ngày chọn {result['requested_end']} rơi vào cuối tuần → chuyển sang ngày làm việc dự kiến {result['effective_target_date']}. Chưa trừ lịch nghỉ lễ.")
    if result['steps'] > 1:
        st.warning('Mô phỏng nhiều bước · đã có backtest h=5/h=22 trên model lịch sử. Model hiện chỉ train h=1; bundle đang phục vụ và khung lịch hiện tại chưa có kiểm chứng forward tương ứng.')
    else:
        st.caption('Ước tính từ snapshot trước phiên. Bấm dự báo trong hôm nay không đồng nghĩa forecast đã được phát trước giờ mở cửa.')
    chart(path_chart(history,result), height=460)
    band_label='khoảng mô phỏng đã nới từ P10–P90' if any(r.get('band_scale_factor',1.)>1. for r in result['path']) else 'P10–P90 của kịch bản residual bootstrap'
    st.caption(f"Đường nét liền: dữ liệu thật đến {result['origin_date']}. Nét đứt: dự báo/mô phỏng. Vùng tô: {band_label}, không bảo đảm 80% kết quả thật nằm trong vùng.")
    if chosen != 'Zero-Return':
        band=result['uncertainty'].get('band_calibration',{})
        st.caption(f"Hệ số band tại đích: {band.get('final_step_factor',1.):.2f}×. Backtest 5/22 quan sát giao dịch khác khung tuần/tháng theo lịch; hệ số chưa đạt điều kiện sẽ giữ 1×.")
    render_forecast_analytics(result)
    render_forecast_explainer(result)
    with st.expander('Bảng dự báo từng bước & tải kết quả'):
        data = pd.DataFrame(result['path']).rename(columns={'estimated_date':'Ngày dự kiến','step':'Bước',
            'predicted_close_adjusted':'Giá dự báo','scenario_lower':'Biên dưới','scenario_upper':'Biên trên',
            'change_from_origin_pct':'Thay đổi từ origin (%)'})
        data['Phần đường']=np.where(data.in_requested_window,'Trong khoảng chọn','Nối snapshot')
        st.dataframe(data[['Bước','Ngày dự kiến','Phần đường','Giá dự báo','Biên dưới','Biên trên','Thay đổi từ origin (%)']],
                     hide_index=True, width='stretch',column_config={k:st.column_config.NumberColumn(format='%.2f') for k in ['Giá dự báo','Biên dưới','Biên trên','Thay đổi từ origin (%)']})
        a,b=st.columns(2)
        a.download_button('Tải CSV đường dự báo',pd.DataFrame(result['path']).to_csv(index=False),'fpt_forecast_path.csv','text/csv')
        b.download_button('Tải JSON có metadata',json.dumps(result,ensure_ascii=False,indent=2),'fpt_forecast.json','application/json')
    with st.expander('Cách tính, ngày và phiên bản'):
        st.write('Hôm nay/ngày mai: ngày lịch đã chọn, dời cuối tuần sang ngày làm việc kế. Một tuần: 7 ngày lịch tính cả hôm nay. Một tháng: đến trước cùng ngày của tháng sau. Ngày cuối khung tuần/tháng bỏ cuối tuần.')
        st.write('Từng bước: cập nhật 6 return feature từ chuỗi Close, dùng lại Ridge/scaler. Bias đóng băng chỉ áp dụng bước 1; bước sau dùng Ridge thô. Residual được trừ trung bình pool trước bootstrap, không trừ trung bình từng đường mô phỏng. Không fit lại hoặc dùng nhãn tương lai. Zero Return giữ giá và không mượn vùng residual của Ridge.')
        st.write('Chọn ngày: từ hôm nay tới ngày bạn chọn trong một tháng; cuối tuần chuyển sang weekday kế tiếp. Toàn bộ phép tính bắt đầu từ snapshot thật, kể cả bước nối tới hôm nay. Không tạo giá thật cho ngày còn thiếu.')
        st.json({k:v for k,v in result.items() if k!='path'})


def overview_view(api, history, health):
    render_market_analysis(api,health)


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
        st.write('Tài liệu: TRAINING_HISTORY.md · Notebook 05 tổng hợp hành trình; notebook 06 so dự báo với giá thật.')
