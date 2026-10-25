"""Streamlit frontend. All predictions and data come from the HTTP backend."""

import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="FPT | ML Research", page_icon="📈", layout="wide")
API_URL = os.environ.get("FPT_API_URL", "http://127.0.0.1:8000").rstrip("/")


def api_get(path):
    response = requests.get(API_URL + path, timeout=30)
    response.raise_for_status()
    return response.json()


st.title("FPT · Phân tích và dự báo")
st.caption("Đồ án Máy học · Giá điều chỉnh · Dự báo một quan sát hợp lệ kế tiếp")
try:
    health = api_get("/health")
    model_info = api_get("/api/v1/models")
    historical = api_get("/api/v1/fpt/history?limit=1000")
    benchmark = api_get("/api/v1/metrics")
except (requests.RequestException, ValueError) as exc:
    st.error("Chưa kết nối được backend. Khởi động: python -m uvicorn src.api.main:app --port 8000")
    st.caption(str(exc))
    st.stop()

with st.sidebar:
    st.header("Mô hình")
    names = model_info["models"]
    default = model_info["default_demo_model"]
    chosen = st.selectbox("Chọn mô hình để so sánh", names, index=names.index(default))
    st.caption(f"Model được chọn theo validation: {model_info['validation_selected_model']}")
    st.caption(f"Phiên bản: {health['model_version']}")
    st.caption(f"Dữ liệu đến: {health['data_as_of']}")

history = pd.DataFrame(historical["observations"]).set_index("Date")
st.subheader("Lịch sử giá đóng cửa điều chỉnh")
st.line_chart(history[["Close"]])
st.info("LSTM là mô hình ML dùng demo. Train Mean thắng theo RMSE validation; giao diện không gọi LSTM là model tốt nhất toàn bộ.")
if st.button("Dự báo quan sát kế tiếp", type="primary"):
    try:
        response = requests.post(API_URL + "/api/v1/predict", json={"model": chosen}, timeout=60)
        response.raise_for_status()
        st.session_state["prediction"] = response.json()
    except requests.RequestException as exc:
        st.error(f"Dự báo thất bại: {exc}")
prediction = st.session_state.get("prediction")
if prediction and prediction["model"] == chosen:
    a, b, c = st.columns(3)
    a.metric("Giá điều chỉnh tại ngày dữ liệu cuối", f"{prediction['latest_close_adjusted']:,.0f} VND")
    b.metric("Giá điều chỉnh dự báo", f"{prediction['predicted_close_adjusted']:,.0f} VND")
    c.metric("Lợi suất dự báo", f"{prediction['predicted_simple_return_pct']:+.3f}%")
    st.caption(f"Dự báo sau {prediction['as_of']}; chưa xác minh lịch HOSE, không gán một ngày đích giả. Đây không phải giá thị trường trực tiếp.")
    with st.expander("Xem JSON từ API"):
        st.json(prediction)

st.subheader("Kết quả ngoài mẫu lịch sử")
table = pd.DataFrame(benchmark["metrics"])
st.dataframe(table[["model", "split", "n", "rmse", "mae", "hit_rate_pct", "direction_coverage_pct"]], hide_index=True)
st.caption("Hit Rate bỏ ngày actual đứng giá, không bỏ dự báo 0 để tăng điểm. Zero-return không phát tín hiệu hướng nên là N/A.")
with st.expander("Giới hạn và cách chọn mô hình"):
    st.write("Năm 2025 đã được xem trong các lần thử. Cải thiện RMSE của LSTM nhỏ, MAE vẫn cao hơn zero-return; chưa chứng minh lợi nhuận giao dịch.")
    st.json(benchmark["selection"])
