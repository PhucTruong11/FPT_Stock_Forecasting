"""Shared presentation for the existing Streamlit dashboard."""
from html import escape
import plotly.graph_objects as go
import streamlit as st

MINT = '#65e6c4'
BLUE = '#73a9ff'
AMBER = '#f5c778'
PURPLE = '#b39aff'
COLORS = [MINT, BLUE, PURPLE, AMBER, '#c4d2e3']


def inject_theme():
    st.markdown('''<style>
    .stApp {background:radial-gradient(ellipse at 90% 0%,#142a38 0%,#0b1220 48%);color:#e9f0f7;font-family:Inter,"Segoe UI",sans-serif}
    [data-testid="stHeader"] {background:transparent}
    [data-testid="stSidebar"] {background:#0d1725;border-right:1px solid #233245}
    [data-testid="stSidebar"]>div:first-child {padding-top:1.5rem}
    .block-container {max-width:1480px;padding:1.7rem 2rem 3rem}
    h1,h2,h3 {letter-spacing:-.035em;color:#f1f6fc!important}
    h1 {font-size:2.5rem!important} h3 {font-size:1.25rem!important}
    p,label,[data-testid="stMarkdownContainer"] {color:#d7e2ed}
    [data-testid="stCaptionContainer"] p {color:#92a5b9;font-size:.83rem;line-height:1.6}
    [data-testid="stMetric"] {background:linear-gradient(135deg,#152334,#111b29);border:1px solid #293b50;border-radius:16px;padding:18px 16px;min-height:122px}
    [data-testid="stMetricValue"] {color:#f6fbff;font-weight:650;font-size:1.5rem!important;letter-spacing:-.04em}
    [data-testid="stMetricLabel"] p {color:#9bb0c5;font-size:.82rem}
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) {flex-wrap:wrap!important}
    [data-testid="stColumn"]:has([data-testid="stMetric"]) {flex:1 1 190px!important;min-width:190px!important}
    [data-testid="stVerticalBlockBorderWrapper"] {border-color:#26374a!important;border-radius:16px}
    [data-testid="stAlert"] {border-radius:12px;border:1px solid #314153}
    [data-testid="stExpander"] {background:#101c2a;border-color:#27384a;border-radius:12px}
    [data-testid="stRadio"] [role="radiogroup"] {gap:8px;flex-wrap:wrap}
    [data-testid="stRadio"] [role="radiogroup"] label {padding:9px 14px;background:#132032;border:1px solid #293b50;border-radius:10px;margin:0;font-size:.88rem}
    [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {background:#1a3c3b;border-color:#65e6c4}
    button[kind="primary"] {background:#65e6c4!important;border-color:#65e6c4!important;color:#0a2224!important;font-weight:700}
    button[kind="secondary"] {border-color:#34485e!important;border-radius:10px}
    [data-testid="stPlotlyChart"] {border:1px solid #26374a;border-radius:16px;overflow:hidden;background:#101c2a}
    .brand {display:flex;align-items:center;gap:12px;margin-bottom:24px}
    .brand-icon {width:42px;height:42px;border-radius:12px;display:grid;place-items:center;background:#65e6c4;color:#0b2827;font-size:23px;font-weight:850}
    .brand-name {font-size:18px;font-weight:750;letter-spacing:-.5px;color:#f1f6fc}
    .eyebrow {font-size:11px;letter-spacing:2px;font-weight:700;text-transform:uppercase;color:#65e6c4;margin-bottom:8px}
    .hero {padding:8px 0 18px}
    .hero h1 {margin:0;padding:0 0 8px;line-height:1.15}
    .hero p {max-width:820px;color:#a1b3c6;margin:4px 0 0;line-height:1.65;font-size:.95rem}
    .status-strip {display:flex;gap:10px;flex-wrap:wrap;margin:4px 0 22px}
    .pill {display:inline-block;padding:6px 10px;border-radius:7px;background:#192a3a;border:1px solid #30475d;color:#bbcede;font-size:11px;letter-spacing:.4px}
    .pill-mint {background:#173832;color:#8af0d4;border-color:#2d6051}
    .note-card {background:#111f2f;border:1px solid #293e52;border-radius:14px;padding:20px;margin:6px 0 16px;min-height:126px}
    .note-card h4 {margin:0 0 8px;color:#edf6ff;font-size:15px}
    .note-card p {margin:0;color:#9fb3c9;font-size:13px;line-height:1.7}
    .forecast-flow {display:flex;gap:24px;margin:18px 0 20px;align-items:stretch}
    .forecast-flow-step {position:relative;flex:1;min-width:0;background:linear-gradient(145deg,#142b36,#111e2c);border:1px solid #345363;border-radius:13px;padding:18px 16px}
    .forecast-flow-step:not(:last-child):after {content:"→";position:absolute;right:-21px;top:45%;color:#65e6c4;font-size:21px}
    .forecast-flow-number {display:inline-block;background:#1c443f;color:#8af0d4;border-radius:7px;padding:3px 8px;font-size:12px;font-weight:700;letter-spacing:1px}
    .forecast-flow-step h4 {margin:12px 0 8px;color:#f0f7ff;font-size:15px;line-height:1.5}
    .forecast-flow-step p {margin:0;color:#afc3d4;font-size:13px;line-height:1.75}
    .forecast-loop {background:#172b39;border:1px dashed #4f7c8d;border-radius:10px;padding:14px 18px;color:#8fe2d0;font-size:13px;line-height:1.8;margin-bottom:14px}
    @media(max-width:1100px) {.forecast-flow{flex-direction:column;gap:24px}.forecast-flow-step:not(:last-child):after{content:"↓";right:calc(50% - 6px);top:auto;bottom:-26px}.forecast-flow-step{padding:14px 16px}.forecast-flow-step h4{display:inline;margin-left:12px}.forecast-flow-step p{margin-top:8px}}
    .timeline {border-left:2px solid #30495c;padding:0 0 22px 20px;margin-left:8px;position:relative}
    .timeline:before {content:"";position:absolute;left:-6px;top:7px;width:10px;height:10px;border-radius:50%;background:#65e6c4}
    .timeline small {color:#7795af;font-size:11px;letter-spacing:.5px}
    .timeline h4 {margin:5px 0;color:#edf6ff;font-size:17px}
    .timeline p {color:#a4b8cd;font-size:13px;margin:3px 0;line-height:1.7}
    .footer {border-top:1px solid #26384c;padding-top:18px;margin-top:35px;color:#6f8aa3;font-size:11px;letter-spacing:.5px}
    @media(max-width:1100px) {[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(4)) > [data-testid="stColumn"]:has([data-testid="stMetric"]){flex:1 1 calc(50% - 1rem)!important}}
    @media(max-width:480px) {[data-testid="stColumn"]:has([data-testid="stMetric"]){flex-basis:100%!important;min-width:0!important}}
    @media(max-width:750px) {.block-container{padding:1.3rem 1rem 2rem} h1{font-size:2rem!important}.hero p{font-size:.86rem}[data-testid="stMetric"]{min-height:100px;padding:14px}.status-strip{gap:6px}.pill{font-size:10px}}
    </style>''', unsafe_allow_html=True)


def hero(kicker, title, subtitle):
    st.markdown(f'<div class="hero"><div class="eyebrow">{escape(kicker)}</div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>', unsafe_allow_html=True)


def note(title, body):
    st.markdown(f'<div class="note-card"><h4>{escape(title)}</h4><p>{escape(body)}</p></div>', unsafe_allow_html=True)


def chart(figure, height=410):
    has_title=bool(figure.layout.title.text)
    figure.update_layout(template='plotly_dark', paper_bgcolor='#101c2a', plot_bgcolor='#101c2a',
                         font={'family':'Segoe UI, sans-serif','color':'#a7bace','size':12},
                         margin={'l':20,'r':24,'t':80 if has_title else 40,'b':35}, height=height, hovermode='x unified',
                         legend={'orientation':'h','yanchor':'bottom','y':1.02,'x':0,'title':None},
                         colorway=COLORS, title_font={'size':15,'color':'#eff6ff'})
    if has_title:figure.update_layout(title={'x':.04,'xanchor':'left','y':.98,'yanchor':'top'})
    figure.update_xaxes(gridcolor='#203147', zeroline=False, showgrid=False)
    figure.update_yaxes(gridcolor='#203147', zeroline=False)
    for axis in figure.select_yaxes():
        if not axis.tickformat:axis.tickformat=',.0f'
    st.plotly_chart(figure, width='stretch', theme=None, config={'displaylogo':False,'modeBarButtonsToRemove':['lasso2d','select2d']})


def path_chart(history, result, count=60):
    history = history.tail(count)
    path = result['path']
    dates = [result['origin_date']] + [row['estimated_date'] for row in path]
    point = [result['origin_close_adjusted']] + [row['predicted_close_adjusted'] for row in path]
    fig = go.Figure()
    if path[0]['scenario_lower'] is not None:
        lower = [result['origin_close_adjusted']] + [r['scenario_lower'] for r in path]
        upper = [result['origin_close_adjusted']] + [r['scenario_upper'] for r in path]
        fig.add_trace(go.Scatter(x=dates,y=upper,line={'width':0},showlegend=False,hoverinfo='skip'))
        band_name='Vùng kịch bản đã hiệu chỉnh' if any(r.get('band_scale_factor',1.)>1. for r in path) else 'Vùng kịch bản P10–P90'
        fig.add_trace(go.Scatter(x=dates,y=lower,fill='tonexty',fillcolor='rgba(101,230,196,.12)',
                                line={'width':0},name=band_name,hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=history.Date,y=history.Close,name='Giá thực tế',line={'color':'#b6c5d7','width':2},
                            hovertemplate='%{y:,.0f} VND<extra>Thực tế</extra>'))
    bridge=[r for r in path if not r['in_requested_window']]
    if bridge:
        fig.add_trace(go.Scatter(x=[result['origin_date']]+[r['estimated_date'] for r in bridge],
            y=[result['origin_close_adjusted']]+[r['predicted_close_adjusted'] for r in bridge],
            name='Mô phỏng nối snapshot',mode='lines+markers',line={'color':'#8799ad','width':2,'dash':'dot'},marker={'size':4},
            hovertemplate='%{y:,.0f} VND<extra>Mô phỏng, không phải giá thật</extra>'))
    start=len(bridge)
    fig.add_trace(go.Scatter(x=dates[start:],y=point[start:],name='Dự báo trong khoảng chọn',mode='lines+markers',
                            line={'color':MINT,'width':2.6,'dash':'dash'},marker={'size':5},
                            hovertemplate='%{y:,.0f} VND<extra>Model</extra>'))
    fig.add_trace(go.Scatter(x=[dates[-1]],y=[point[-1]],mode='markers',name='Ngày đích',
        marker={'color':AMBER,'size':10,'symbol':'diamond'},hovertemplate='%{x}<br>%{y:,.0f} VND<extra>Ngày đích dự kiến</extra>'))
    fig.add_vline(x=result['origin_date'],line_dash='dot',line_color='#556f88')
    fig.add_vline(x=dates[-1],line_dash='dot',line_color=AMBER)
    fig.add_vrect(x0=result['requested_start'],x1=dates[-1],fillcolor='rgba(101,230,196,.035)',line_width=0,layer='below')
    fig.update_layout(yaxis_title='Giá điều chỉnh · VND', xaxis_title='Ngày dự kiến · chưa trừ lịch nghỉ lễ')
    return fig
