"""Compare the frozen forecast with separately retrieved, completed daily prices."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.app.theme import chart


def comparison_chart(result, observed):
    origin = pd.Timestamp(result["origin_date"])
    path = pd.DataFrame(result["path"])
    path["Date"] = pd.to_datetime(path["estimated_date"])
    path = path[["Date", "predicted_close_adjusted"]]
    actual = pd.DataFrame(observed["observations"])
    actual["Date"] = pd.to_datetime(actual.Date)
    actual = actual.sort_values("Date")
    context = actual.loc[actual.Date <= origin].tail(5)
    shown = pd.concat([context, actual.loc[actual.Date > origin]], ignore_index=True)
    fig = go.Figure(go.Scatter(x=shown.Date, y=shown.Close, mode="lines+markers",
        name="Đóng cửa đã quan sát · Vietcap", line={"color": "#72b8ff", "width": 3},
        marker={"size": 6}, hovertemplate="%{x|%d/%m/%Y}<br>%{y:,.0f} VND<extra>Giá đã quan sát</extra>"))
    matched = pd.DataFrame(columns=["Date", "predicted_close_adjusted", "Close"])
    if observed["basis_check"]["allowed"]:
        fig.add_trace(go.Scatter(
            x=[origin] + path.Date.tolist(),
            y=[result["origin_close_adjusted"]] + path.predicted_close_adjusted.tolist(),
            mode="lines+markers", name="Mô phỏng từ snapshot · tính lại lúc xem",
            line={"color": "#f3c879", "width": 2.6, "dash": "dash"}, marker={"size": 4},
            hovertemplate="%{x|%d/%m/%Y}<br>%{y:,.0f} VND<extra>Model mô phỏng</extra>"))
        # Exact-date join only: no filling holidays/missing provider sessions.
        matched = path.merge(actual.loc[actual.Date > origin, ["Date", "Close"]], on="Date",
                             how="inner", validate="one_to_one")
    fig.add_vline(x=origin.isoformat(), line_dash="dot", line_color="#8799ad")
    fig.update_layout(yaxis_title="Giá đóng cửa điều chỉnh · VND", xaxis_title="Ngày quan sát / ngày dự kiến")
    return fig, matched


def render_observed_comparison(api, result, *, clear_api_cache=None):
    st.subheader("Giá thực tế sau snapshot so với dự báo")
    st.caption("Biểu đồ riêng dùng giá đóng cửa lấy thêm từ Vietcap. "
               "Không thay dữ liệu gốc hoặc huấn luyện lại model.")
    refresh = st.button("Cập nhật giá đóng cửa từ Vietcap", key="refresh_observed_prices")
    try:
        if refresh:
            with st.spinner("Đang lấy các phiên đã hoàn tất từ Vietcap…"):
                observed = api("/api/v1/fpt/observed/refresh", {})
            if clear_api_cache is not None:
                clear_api_cache()
        else:
            observed = api("/api/v1/fpt/observed")
    except (requests.RequestException, ValueError) as exc:
        st.warning("Chưa lấy được dữ liệu đối chiếu. Biểu đồ mô phỏng phía trên vẫn dùng snapshot cũ.")
        st.caption(str(exc))
        return
    if observed["status"] != "ready":
        st.info("Chưa tải giá sau snapshot. Bấm cập nhật để lấy dữ liệu Vietcap cho biểu đồ này.")
        return
    source_day = pd.Timestamp(observed["data_as_of"])
    retrieved = pd.Timestamp(observed["retrieved_at_utc"]).tz_convert("Asia/Ho_Chi_Minh")
    st.caption(f"Nguồn: Vietcap FPT · giá đóng cửa cuối {source_day:%d/%m/%Y} · "
               f"lấy lúc {retrieved:%H:%M %d/%m/%Y} (UTC+7). "
               "Chỉ dùng nến D1 trước ngày hiện tại để tránh nến chưa hoàn tất; không tự điền ngày thiếu.")
    if not observed["basis_check"]["allowed"]:
        st.warning("Chưa ghép trực tiếp với dự báo: giá lịch sử giữa nguồn mới và snapshot "
                   "không khớp hoặc chưa đủ phiên chung để kiểm tra điều chỉnh giá. "
                   "Biểu đồ bên dưới chỉ hiện dữ liệu Vietcap, không tính sai số.")
    fig, matched = comparison_chart(result, observed)
    chart(fig, height=400)
    a, b, c = st.columns(3)
    post = [r for r in observed["observations"] if r["Date"] > result["origin_date"]]
    a.metric("Phiên thật sau snapshot", str(len(post)))
    b.metric("Đóng cửa nguồn gần nhất · VND", f"{observed['observations'][-1]['Close']:,.0f}")
    c.metric("MAE giá đối chiếu · VND", f"{np.mean(abs(matched.predicted_close_adjusted - matched.Close)):,.0f}"
             if len(matched) else "Chưa đủ dữ liệu", help="Chỉ tính trên ngày có cả giá quan sát và mô phỏng, sau snapshot.")
    if source_day <= pd.Timestamp(result["origin_date"]):
        st.info("Nguồn hiện chưa trả phiên mới hơn snapshot; không có giá thật để thay phần mô phỏng.")
    else:
        st.caption(f"Đường liền xanh dương: giá đã quan sát. Nét đứt vàng: cùng model và chế độ bias đang chọn. "
                   f"Có {len(matched)} ngày khớp để đối chiếu; điểm sau {source_day:%d/%m/%Y} chưa có giá thật trong nguồn này.")
    st.caption("Đây là đối chiếu hồi cứu: đường mô phỏng được tính lại lúc xem, "
               "không phải bằng chứng dự báo đã được ghi trước khi có giá thật.")
    st.link_button("Xem FPT trên Vietcap", observed["company_url"])
