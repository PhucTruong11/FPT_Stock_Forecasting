"""FPT Forecast Studio: Streamlit views consuming the local FastAPI backend."""
import os
from html import escape
import pandas as pd
import requests
import streamlit as st
from src.app.theme import inject_theme
from src.app.dashboard_views import (forecast_view, overview_view, benchmark_view,
    journey_view, methodology_view, roadmap_view)

st.set_page_config(page_title='FPT Forecast Studio',page_icon='📈',layout='wide')
inject_theme()
API_URL=os.environ.get('FPT_API_URL','http://127.0.0.1:8000').rstrip('/')


@st.cache_data(ttl=30,show_spinner=False)
def api_get(path):
    response=requests.get(API_URL+path,timeout=30)
    if not response.ok:
        raise ValueError(f'API {response.status_code}: {response.json().get("detail",response.text[:300])}')
    return response.json()


def api(path,payload=None):
    if payload is None:
        return api_get(path)
    response=requests.post(API_URL+path,json=payload,timeout=30)
    if not response.ok:
        detail=response.json().get('detail',response.text[:300])
        raise ValueError(str(detail))
    return response.json()


try:
    health=api('/health');registry=api('/api/v1/models')
except (requests.RequestException,ValueError) as exc:
    st.error('Chưa kết nối API local. Chạy: python -m uvicorn src.api.main:app --port 8000')
    st.caption(str(exc));st.stop()

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">F</div><div><div class="brand-name">FPT Forecast</div><div class="eyebrow" style="margin:3px 0 0;font-size:9px">Research studio</div></div></div>',unsafe_allow_html=True)
    st.markdown('### FPT Corporation')
    st.caption('FPT · HOSE / Việt Nam\n\nGiá điều chỉnh theo nhà cung cấp')
    st.divider()
    st.markdown('**Model đang chạy**')
    st.write('Ridge · 6 return features')
    st.caption(f"Version {health['model_version']}\n\nTrain nhãn đến {health['training_target_end']}")
    st.divider()
    st.markdown('**Nguồn dữ liệu**')
    st.write(health['source'])
    st.caption(f"Snapshot đến {health['data_as_of']}\n\nKhông tự tải giá hoặc cập nhật weights.")
    if st.button('Nạp lại dữ liệu API',width='stretch'):
        api_get.clear();st.rerun()
    st.caption('Chỉ làm mới dữ liệu từ backend đang chạy, không download từ thị trường.')
    st.link_button('Khám phá API',API_URL+'/docs',width='stretch')

st.markdown(f'<div class="status-strip"><span class="pill pill-mint">● LOCAL STUDIO</span><span class="pill">FPT · MỘT MÃ</span><span class="pill">DATA {escape(health["data_as_of"])}</span><span class="pill">HỌC CÓ GIÁM SÁT</span></div>',unsafe_allow_html=True)
page=st.radio('Không gian phân tích',['Tổng quan','Dự báo','Đối chứng','Hành trình model','Dữ liệu & phương pháp','Lộ trình'],index=1,horizontal=True,label_visibility='collapsed')
st.caption(f"Snapshot {health['data_as_of']} · Không phải giá realtime. Khung tuần/tháng là mô phỏng từ model h=1; ngày dự kiến chưa trừ lịch nghỉ lễ.")
try:
    if page in {'Tổng quan','Dự báo'}:
        history=pd.DataFrame(api('/api/v1/fpt/history?limit=1000')['observations'])
    if page=='Dự báo':forecast_view(api,history,health,registry)
    elif page=='Tổng quan':overview_view(api,history,health)
    elif page=='Đối chứng':benchmark_view(api)
    elif page=='Hành trình model':journey_view(api)
    elif page=='Dữ liệu & phương pháp':methodology_view(api,health)
    else:roadmap_view(api)
except (requests.RequestException,ValueError) as exc:
    st.error(f'Không thể hiển thị kết quả: {exc}')
    st.caption('Kiểm tra API, ngày snapshot và version. Không dùng dữ liệu giả hoặc forecast cũ thay cho kết quả không có.')

st.markdown('<div class="footer">FPT FORECAST STUDIO / ĐỒ ÁN MÁY HỌC · LOCAL RESEARCH · CHƯA DEPLOY PUBLIC</div>',unsafe_allow_html=True)
