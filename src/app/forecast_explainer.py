"""Explain the returned forecast without loading, fitting, or calling a model."""
from html import escape

import streamlit as st


def _flow(label, steps):
    cards=''.join(
        f'<div class="forecast-flow-step"><span class="forecast-flow-number">{i:02d}</span>'
        f'<h4>{escape(title)}</h4><p>{escape(body)}</p></div>'
        for i,(title,body) in enumerate(steps,1))
    st.markdown(f'<div class="forecast-flow" role="group" aria-label="{escape(label)}">{cards}</div>',unsafe_allow_html=True)


def render_forecast_explainer(result):
    zero=result['model']=='Zero-Return'
    corrected=result['model']=='Ridge+bias60'
    first=result['path'][0]
    final=result['path'][-1]
    with st.expander('Hiểu cách model dự báo',expanded=True):
        st.caption('Ba sơ đồ đọc từ trái sang phải; trên màn hình hẹp đọc từ trên xuống. Minh họa bám theo model và kết quả bạn đang chọn.')
        pipeline,recursive,uncertainty=st.tabs(['1 · Từ giá đến dự báo','2 · Dự báo nhiều bước','3 · Vùng kịch bản'])
        with pipeline:
            if zero:
                _flow('Luồng baseline Zero Return',[
                    ('Giá thật cuối cùng',f"Close điều chỉnh tại {result['origin_date']}: {result['origin_close_adjusted']:,.0f} VND."),
                    ('Giả định return = 0','Baseline giả định giá không đổi; không dùng đầu ra Ridge hay bias.'),
                    ('Giữ nguyên giá',f"Mọi bước tới {result['effective_target_date']} đều bằng giá origin."),
                ])
            else:
                _flow('Từ giá FPT đến giá dự báo',[
                    ('Lịch sử FPT','21 giá đóng cửa điều chỉnh gần nhất đủ để tính 6 feature lợi suất của Ridge hiện tại.'),
                    ('6 feature lợi suất','Log return hiện tại; log return trễ 1, 2, 5 quan sát; log return tích lũy 5 và 20 quan sát.'),
                    ('Scaler → Ridge',('Dùng scaler và trọng số đã fit. Bước 1 cộng bias60 từ sai số đã có nhãn.' if corrected else 'Dùng scaler và trọng số đã fit; không cộng bias60.')),
                    ('Log return → giá','Lấy giá trước nhân exp(log return dự báo) để được giá đóng cửa điều chỉnh kế tiếp.'),
                ])
                st.caption('Đây là hồi quy có giám sát: khi train, nhãn là log return phiên kế tiếp. Khi dự báo, chỉ dùng lịch sử đã biết và bộ scaler/model đã lưu; không fit lại.')
            st.markdown('**Ví dụ bước đầu của chính kết quả này**')
            st.latex(r'\widehat P_{t+1}=P_t\,e^{\widehat r_{t+1}}')
            st.write(f"{result['origin_close_adjusted']:,.0f} × exp({first['step_log_return']:+.6f}) ≈ {first['predicted_close_adjusted']:,.0f} VND · ngày dự kiến {first['estimated_date']}.")
            st.caption('r là log return, khác % thay đổi giá thông thường. Bước đầu tính từ snapshot, có thể nằm trước khoảng ngày bạn chọn.')
        with recursive:
            if zero:
                _flow('Các bước của Zero Return',[
                    ('Bắt đầu từ snapshot',f"Giá thật tại {result['origin_date']}."),
                    ('Lặp giả định không đổi','Return mỗi bước bằng 0, giữ nguyên mức giá.'),
                    ('Dừng tại ngày đích',f"{result['steps']} bước tới {result['effective_target_date']}; không có vùng residual của Ridge."),
                ])
            else:
                _flow('Vòng lặp dự báo recursive',[
                    ('Bước 1 · lịch sử thật',f"Từ snapshot {result['origin_date']}, tính feature và chạy Ridge"+(' + bias60.' if corrected else ' nguyên bản.')),
                    ('Bước 2 · cập nhật chuỗi','Đưa Close vừa dự báo vào cửa sổ, bỏ giá cũ nhất; tính lại 6 feature rồi dùng Ridge thô.'),
                    ('Lặp đến ngày đích',f"Tiếp tục với chuỗi có giá dự báo, tới bước {result['steps']} ({result['effective_target_date']})."),
                ])
                st.markdown('<div class="forecast-loop">↺ Giá dự báo → cập nhật cửa sổ 21 giá → tính lại feature → Ridge → giá dự báo kế tiếp</div>',unsafe_allow_html=True)
                st.caption('Bias60 chỉ cộng ở bước 1 của Ridge hiệu chỉnh; không cộng dồn vào các bước sau. Trọng số/scaler giữ nguyên trong toàn bộ vòng lặp. Không tạo OHLCV tương lai.')
            bridge=result.get('bridge_steps_before_requested_window',0)
            st.write(f"**Kết quả đang xem:** {result['steps']} bước từ snapshot; {bridge} bước nối trước khoảng chọn. Giá ở đích: **{final['predicted_close_adjusted']:,.0f} VND**.")
            st.caption('Đoạn xám nối snapshot là mô phỏng; đoạn xanh thuộc khoảng bạn chọn. Model học h=1. Backtest lịch sử h=5/h=22 không xác nhận độ chính xác của mọi ngày đích hoặc bundle hiện tại.')
        with uncertainty:
            if zero:
                st.info('Zero Return là đường so sánh giữ nguyên giá. API không gắn vùng sai số của Ridge cho baseline này.')
            else:
                spec=result['uncertainty']
                _flow('Tạo vùng kịch bản residual bootstrap',[
                    ('Sai số đã quan sát',('Dùng tối đa 60 residual hiệu chỉnh đã có nhãn trong bundle.' if corrected else 'Dùng tối đa 60 residual Ridge thô đã có nhãn trong bundle.')),
                    ('Center & lấy block',f"Trừ trung bình cả pool, lấy mẫu block vòng tròn {spec['block_observations']} residual; giữ liên hệ trong block."),
                    ('Mô phỏng nhiều đường',f"Chạy {spec['paths']} đường: Ridge theo lịch sử riêng của từng đường + nhiễu; bias chỉ ở bước đầu nếu bật."),
                    ('Đọc phân vị từng bước','Lấy P10 và P90 của giá mô phỏng ở từng bước; không phải min/max và không phải toàn bộ đường đi cùng được phủ.'),
                ])
                factor=final.get('band_scale_factor',1.)
                st.write(f"**Tại ngày đích:** {final['scenario_lower']:,.0f} – {final['scenario_upper']:,.0f} VND · hệ số độ rộng {factor:.2f}×.")
                st.caption('Nếu hệ số >1, band được nới theo khoảng cách log-price quanh dự báo điểm và không còn là P10/P90 nguyên bản. Khi hệ số =1, giữ phân vị bootstrap gốc.')
                st.caption('Đường giữa là dự báo Ridge riêng, không phải trung vị bắt buộc của 512 đường. P10–P90 mô tả mô phỏng; không bảo đảm độ phủ thực tế 80%, không phải xác suất tăng/giảm và không bao quát mọi rủi ro thị trường.')
