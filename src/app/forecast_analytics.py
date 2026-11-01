"""Additional forecast charts derived solely from the current API response."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app.theme import MINT, BLUE, AMBER, chart


def forecast_dashboard_figures(result):
    frame=pd.DataFrame(result['path'])
    dates=frame.estimated_date
    origin=float(result['origin_close_adjusted'])
    # Convert log returns to simple percentage returns; never sum daily percentages.
    step_return=np.expm1(frame.step_log_return.to_numpy(float))*100
    colors=np.where(~frame.in_requested_window,'#8799ad',np.where(step_return>=0,MINT,'#f18c9a'))
    scope=np.where(frame.in_requested_window,'Trong khoảng chọn','Nối snapshot · mô phỏng')
    daily=go.Figure(go.Bar(x=dates,y=step_return,marker_color=colors,name='Return từng bước',
        customdata=scope,hovertemplate='%{x}<br>%{y:+.3f}%<br>%{customdata}<extra></extra>'))
    daily.add_trace(go.Scatter(x=dates,y=step_return,mode='markers',marker={'size':5,'color':colors},
        showlegend=False,hoverinfo='skip'))
    daily.add_hline(y=0,line_color='#72859c',line_width=1)
    daily.update_layout(title='Tăng / giảm dự báo từng bước',showlegend=False,
        yaxis={'title':'So với bước trước · %','tickformat':'.2f'},xaxis_title='Ngày dự kiến')

    cumulative=go.Figure()
    x=[result['origin_date'],*dates]
    has_band=frame.scenario_lower.notna().all() and frame.scenario_upper.notna().all()
    scaled=bool(frame.band_scale_factor.gt(1).any())
    band_name='Band đã nới' if scaled else 'P10–P90 mô phỏng'
    if has_band:
        lower=(frame.scenario_lower.astype(float)/origin-1)*100
        upper=(frame.scenario_upper.astype(float)/origin-1)*100
        cumulative.add_trace(go.Scatter(x=x,y=[0.,*upper],line={'width':0},showlegend=False,hoverinfo='skip'))
        cumulative.add_trace(go.Scatter(x=x,y=[0.,*lower],fill='tonexty',fillcolor='rgba(101,230,196,.14)',
            line={'width':0},name=band_name,hovertemplate='%{x}<br>Biên dưới: %{y:+.2f}%<extra></extra>'))
        cumulative.add_trace(go.Scatter(x=x,y=[0.,*upper],line={'color':'#4d8b84','width':1},
            name='Biên trên',showlegend=False,hovertemplate='%{x}<br>Biên trên: %{y:+.2f}%<extra></extra>'))
    cumulative.add_trace(go.Scatter(x=x,y=[0.,*frame.change_from_origin_pct],mode='lines+markers',
        line={'color':BLUE,'width':2.4},name='Dự báo điểm',hovertemplate='%{x}<br>%{y:+.2f}% so origin<extra></extra>'))
    cumulative.add_hline(y=0,line_color='#72859c',line_dash='dot')
    cumulative.update_layout(title='Thay đổi tích lũy từ giá gốc',
        yaxis={'title':'So với origin · %','tickformat':'.1f'},xaxis_title='Ngày dự kiến')
    figures={'step_returns':daily,'cumulative_returns':cumulative}
    if has_band:
        width=(frame.scenario_upper.astype(float)-frame.scenario_lower.astype(float))/frame.predicted_close_adjusted*100
        spread=go.Figure(go.Scatter(x=dates,y=width,mode='lines+markers',fill='tozeroy',
            fillcolor='rgba(245,199,120,.10)',line={'color':AMBER,'width':2.2},
            name='Độ rộng band / giá điểm',hovertemplate='%{x}<br>Độ rộng: %{y:.2f}% giá dự báo điểm<extra></extra>'))
        spread.update_layout(title='Vùng kịch bản rộng bao nhiêu theo thời gian?',showlegend=False,
            yaxis={'title':'(Biên trên − dưới) / giá điểm · %','tickformat':'.1f','rangemode':'tozero'},xaxis_title='Ngày dự kiến')
        figures['band_width']=spread
    # Daily observations only: do not suggest intraday predictions with auto date ticks.
    for name,figure in figures.items():
        categories=x if name=='cumulative_returns' else dates.to_list()
        ticks=[categories[i] for i in np.linspace(0,len(categories)-1,min(5,len(categories)),dtype=int)]
        figure.update_xaxes(type='category',categoryorder='array',categoryarray=categories,
            tickmode='array',tickvals=ticks,ticktext=[pd.Timestamp(day).strftime('%d/%m') for day in ticks])
    return figures


def render_forecast_analytics(result):
    st.subheader('Phân tích đường dự báo')
    st.caption('Các biểu đồ cùng dùng kết quả phía trên. Ngày dự kiến chỉ bỏ cuối tuần; phần nối từ snapshot vẫn là mô phỏng.')
    if not st.toggle('Hiện biểu đồ phân tích dự báo',value=True):
        return
    figures=forecast_dashboard_figures(result)
    left,right=st.columns(2)
    with left:
        chart(figures['step_returns'],340)
        st.caption('Xanh: tăng/không đổi · hồng: giảm · xám: bước nối trước khoảng chọn. Mỗi cột so với giá của bước liền trước; không phải accuracy hay tín hiệu mua/bán.')
    with right:
        chart(figures['cumulative_returns'],340)
        st.caption(f"Mốc 0% = giá thật cuối cùng {result['origin_close_adjusted']:,.0f} VND tại {result['origin_date']}. Thay đổi tích lũy tính theo tỷ lệ giá, không cộng return từng bước.")
    if 'band_width' not in figures:
        st.info('Zero Return giữ nguyên giá: return từng bước và tích lũy bằng 0. Không có band Ridge để phân tích độ rộng.')
        return
    final=result['path'][-1]
    a,b,c=st.columns(3)
    a.metric('Biên dưới tại ngày đích',f"{final['scenario_lower']:,.0f} VND",
        f"{(final['scenario_lower']/result['origin_close_adjusted']-1)*100:+.2f}% so origin",delta_color='off')
    b.metric('Biên trên tại ngày đích',f"{final['scenario_upper']:,.0f} VND",
        f"{(final['scenario_upper']/result['origin_close_adjusted']-1)*100:+.2f}% so origin",delta_color='off')
    c.metric('Độ rộng band tại đích',f"{(final['scenario_upper']-final['scenario_lower'])/final['predicted_close_adjusted']*100:.2f}%",
        help='(Biên trên − biên dưới) / giá dự báo điểm ×100; không phải xác suất hay mức lỗ tối đa.')
    chart(figures['band_width'],300)
    st.caption('Band phản ánh các đường mô phỏng residual, không bảo đảm độ phủ thực tế 80%. Độ rộng có thể dao động giữa các bước; không bị ép tăng đều. Band đã nới không còn là phân vị P10/P90 nguyên bản.')
