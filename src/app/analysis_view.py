"""Interactive FPT snapshot analysis; descriptive metrics never modify the model."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.app.market_analysis import prepare_market_frame, window_statistics, monthly_returns, snapshot_insights
from src.app.theme import MINT, BLUE, chart, hero, note


def render_market_analysis(api, health):
    hero('FPT / Market lens', 'Đọc xu hướng. Nhìn cả biến động.',
         'Quan sát giá, động lượng, khối lượng và rủi ro của FPT trong cùng một khung thời gian. Chỉ báo mô tả snapshot, tách khỏi kết quả dự báo của model.')
    raw,quality=prepare_market_frame(api('/api/v1/fpt/ohlcv?limit=2500')['observations'])
    last,previous=raw.iloc[-1],raw.iloc[-2]
    change=(last.Close/previous.Close-1)*100
    a,b,c,d=st.columns(4)
    a.metric('Đóng cửa điều chỉnh',f'{last.Close:,.0f} VND',f'{change:+.2f}% phiên trước',delta_color='off')
    b.metric('Khối lượng phiên cuối',f'{last.Volume/1e6:.2f} triệu')
    c.metric('Biên độ High–Low',f'{(last.High-last.Low)/last.Close*100:.2f}%',help='Biên độ phiên cuối, khác lợi suất close-to-close.')
    d.metric('Ngày dữ liệu cuối',last.Date.strftime('%d/%m/%Y'))
    with st.container(border=True):
        a,b=st.columns([2,1])
        with a:period=st.select_slider('Khoảng quan sát',options=['1 tháng','3 tháng','6 tháng','1 năm','3 năm','Toàn bộ'],value='6 tháng')
        with b:kind=st.selectbox('Kiểu đồ thị giá',['Nến OHLC','Đường đóng cửa'])
        overlays=st.multiselect('Lớp chỉ báo trên giá',['SMA20','SMA50','SMA200','Bollinger20'],default=['SMA20','SMA50'])
    counts={'1 tháng':22,'3 tháng':63,'6 tháng':126,'1 năm':252,'3 năm':756,'Toàn bộ':len(raw)}
    part=raw.tail(counts[period]).copy();stats=window_statistics(part)
    st.caption(f"{part.Date.iloc[0]:%d/%m/%Y} → {part.Date.iloc[-1]:%d/%m/%Y} · {len(part):,} quan sát có giao dịch. Mốc tháng/năm là khung xấp xỉ theo số quan sát; chỉ báo tính trên lịch sử đầy đủ trước khi thu khung.")
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.035,row_heights=[.76,.24])
    if 'Bollinger20' in overlays:
        fig.add_trace(go.Scatter(x=part.Date,y=part.BB_upper,line={'width':0},showlegend=False,hoverinfo='skip'),row=1,col=1)
        fig.add_trace(go.Scatter(x=part.Date,y=part.BB_lower,line={'width':0},fill='tonexty',fillcolor='rgba(115,169,255,.11)',name='Bollinger20 ±2σ',hoverinfo='skip'),row=1,col=1)
    if kind=='Nến OHLC':
        fig.add_trace(go.Candlestick(x=part.Date,open=part.Open,high=part.High,low=part.Low,close=part.Close,
            increasing_line_color=MINT,decreasing_line_color='#f18c9a',name='FPT OHLC điều chỉnh'),row=1,col=1)
    else:fig.add_trace(go.Scatter(x=part.Date,y=part.Close,name='Đóng cửa',line={'color':MINT,'width':2.2}),row=1,col=1)
    for n,color in [(20,BLUE),(50,'#f5c778'),(200,'#b39aff')]:
        if f'SMA{n}' in overlays:fig.add_trace(go.Scatter(x=part.Date,y=part[f'SMA_{n}'],name=f'SMA{n}',line={'color':color,'width':1.4}),row=1,col=1)
    colors=np.where(part.Close>=part.Open,'rgba(101,230,196,.5)','rgba(241,140,154,.5)')
    fig.add_trace(go.Bar(x=part.Date,y=part.Volume,marker_color=colors,name='Khối lượng'),row=2,col=1)
    fig.add_trace(go.Scatter(x=part.Date,y=part.Volume_MA20_prior,line={'color':BLUE,'width':1.2},name='Volume TB20 trước'),row=2,col=1)
    fig.update_layout(xaxis_rangeslider_visible=False,yaxis_title='Giá điều chỉnh · VND',yaxis2_title='Cổ phiếu',yaxis2_tickformat='~s')
    chart(fig,620)
    st.caption('SMA là trung bình giá quá khứ. Bollinger20 dùng SMA20 ±2 độ lệch chuẩn giá, không phải vùng P10–P90 hay khoảng dự báo của model. Bấm tên đường để ẩn/hiện.')
    a,b,c,d=st.columns(4)
    a.metric('Thay đổi trong khung',f"{stats['return_pct']:+.2f}%",help='Đóng cửa cuối / đóng cửa đầu khung −1; không phải lợi nhuận một chiến lược.')
    b.metric('Drawdown lớn nhất',f"{stats['max_drawdown_pct']:.2f}%",help='Sụt giảm từ đỉnh đóng cửa đã xuất hiện trong chính khung đang xem.')
    c.metric('Biến động quy đổi năm',f"{stats['annual_volatility_pct']:.2f}%" if stats['annual_volatility_pct'] is not None else 'N/A',help='Độ lệch chuẩn log return trong khung ×√252. Quy ước mô tả, không dự báo volatility.')
    d.metric('Tỷ lệ phiên tăng',f"{stats['up_pct']:.1f}%",help=f"Trên {stats['daily_return_count']} thay đổi close-to-close trong khung; không phải accuracy của model.")
    insights=snapshot_insights(raw)
    a,b,c=st.columns(3)
    with a:note('Xu hướng tại snapshot',insights['trend'])
    with b:note('Động lượng tại snapshot',insights['momentum'])
    with c:note('Khối lượng tại snapshot',insights['volume'])
    if st.toggle('Hiện phân tích chi tiết',value=True):
        st.subheader('Động lượng và vị trí giá')
        a,b=st.columns(2)
        with a:
            rsi=go.Figure(go.Scatter(x=part.Date,y=part.RSI_14,line={'color':'#b39aff','width':2},name='RSI14'))
            rsi.add_hrect(y0=30,y1=70,fillcolor='rgba(115,169,255,.06)',line_width=0)
            for y in [30,70]:rsi.add_hline(y=y,line_dash='dot',line_color='#788ba0')
            rsi.update_layout(title='RSI14 · động lượng',yaxis={'range':[0,100],'tickformat':'.0f'})
            chart(rsi,330)
            st.caption('RSI dùng làm mượt Wilder; 30/70 là ngưỡng tham khảo. Vùng thấp/cao không chứng minh giá sắp đảo chiều.')
        with b:
            macd=go.Figure(go.Bar(x=part.Date,y=part.MACD_hist,name='MACD − Signal',marker_color=np.where(part.MACD_hist>=0,MINT,'#f18c9a')))
            macd.add_trace(go.Scatter(x=part.Date,y=part.MACD,name='MACD12–26',line={'color':BLUE,'width':1.7}))
            macd.add_trace(go.Scatter(x=part.Date,y=part.MACD_signal,name='Signal9',line={'color':'#f5c778','width':1.4}))
            macd.update_layout(title='MACD · chênh lệch EMA',yaxis_title='VND',yaxis_tickformat=',.0f');chart(macd,330)
        a,b,c,d=st.columns(4)
        a.metric('RSI14 hiện tại',f'{last.RSI_14:.1f}' if pd.notna(last.RSI_14) else 'N/A')
        b.metric('Gap mở cửa phiên cuối',f'{last.Gap_pct:+.2f}%',help='Open / Close của quan sát giao dịch trước −1.')
        c.metric('Thân nến phiên cuối',f'{last.Body_pct:+.2f}%',help='Close / Open −1; khác lợi suất close-to-close.')
        d.metric('ATR14 / Close',f'{last.ATR14_pct:.2f}%' if pd.notna(last.ATR14_pct) else 'N/A',help='True Range làm mượt Wilder14 / Close, mô tả biên dao động đã quan sát.')
        st.caption(f"Đáy/đỉnh của 20 quan sát trước phiên cuối: {last.Prior20_low:,.0f} – {last.Prior20_high:,.0f} VND. Mốc đã quan sát, không phải biên chắc chắn của giá tương lai.")
        st.subheader('Rủi ro và phân bố lợi suất')
        a,b=st.columns(2)
        with a:
            dd=(part.Close/part.Close.cummax()-1)*100
            risk=go.Figure(go.Scatter(x=part.Date,y=dd,fill='tozeroy',line={'color':'#f18c9a','width':1.6},name='Drawdown'))
            risk.update_layout(title='Sụt giảm từ đỉnh trong khung',yaxis_title='%',yaxis_tickformat='.1f');chart(risk,300)
        with b:
            returns=part.Close.pct_change(fill_method=None).dropna()*100
            dist=go.Figure(go.Histogram(x=returns,nbinsx=35,marker_color=BLUE,name='Return thực tế'))
            dist.add_vline(x=0,line_dash='dot',line_color='#788ba0')
            dist.update_layout(title='Phân bố lợi suất close-to-close',xaxis_title='Return (%)',yaxis_title='Số quan sát');chart(dist,300)
        st.caption(f"Phiên tăng / giảm / đứng giá: {stats['up_pct']:.1f}% / {stats['down_pct']:.1f}% / {stats['flat_pct']:.1f}%. Tốt nhất {stats['best_day_pct']:+.2f}%, kém nhất {stats['worst_day_pct']:+.2f}% trong một bước close-to-close.")
        month=monthly_returns(raw,part.Date.iloc[0]);matrix=month.pivot(index='Year',columns='Month',values='Return_pct').reindex(columns=range(1,13))
        text=[[f'{v:+.1f}%' if pd.notna(v) else '' for v in row] for row in matrix.to_numpy()]
        heat=go.Figure(go.Heatmap(z=matrix.to_numpy(),x=[f'T{n}' for n in range(1,13)],y=[str(y) for y in matrix.index],
            text=text,texttemplate='%{text}',colorscale='RdYlGn',zmid=0,hoverongaps=False,
            hovertemplate='%{y} · %{x}: %{z:.2f}%<extra></extra>',colorbar={'title':'%'}))
        heat.update_layout(title='Lợi suất tháng · đóng cửa cuối tháng quan sát được',yaxis={'type':'category'},xaxis={'type':'category'})
        chart(heat,max(230,100+len(matrix)*38))
        st.caption('Return tháng tính từ giá cuối tháng quan sát được so tháng trước trên lịch sử đầy đủ. Tháng đầu có thể bắt đầu trước khung đang xem; ô thiếu giữ trống, tháng cuối snapshot có thể chưa hoàn tất. Không cộng các return ngày để thay return tháng.')
    st.caption(f"Nguồn {health['source']} đến {health['data_as_of']} · không realtime, không tự cập nhật weights. Loại {quality['untraded_rows']} hàng không có volume dương và {quality['invalid_traded_rows']} nến giao dịch không hợp lệ khỏi phần phân tích; không forward-fill ngày nghỉ.")
    with st.expander('Bảng chỉ báo và xuất dữ liệu'):
        columns=['Date','Open','High','Low','Close','Volume','Return_pct','Gap_pct','Body_pct','SMA_20','SMA_50','SMA_200','RSI_14','MACD','MACD_signal','Volume_ratio20','Volatility20_annual_pct','ATR14_pct']
        st.dataframe(part[columns].sort_values('Date',ascending=False),hide_index=True,width='stretch')
        st.download_button('Tải CSV giá và chỉ báo',part[columns].to_csv(index=False),'fpt_snapshot_analysis.csv','text/csv')
