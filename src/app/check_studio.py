"""Explicit UI smoke check against the running localhost API; never part of training."""
import json
import sys
from pathlib import Path
import requests
import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest

# Cho phép chạy trực tiếp `python src/app/check_studio.py` (không chỉ `python -m`).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.app.forecast_analytics import forecast_dashboard_figures


def run():
    root=Path(__file__).resolve().parents[2]
    app=AppTest.from_file(str(root/'src/app/streamlit_app.py'),default_timeout=40).run()
    def clean(label):
        assert not app.exception,(label,[x.value for x in app.exception])
        assert not app.error,(label,[x.value for x in app.error])
    clean('default');checks={}
    explanation_tabs=['1 · Từ giá đến dự báo','2 · Dự báo nhiều bước','3 · Vùng kịch bản']
    assert [tab.label for tab in app.tabs]==explanation_tabs
    assert any('21 giá đóng cửa' in item.value for item in app.markdown)
    assert app.radio[0].options==['Tổng quan','Dự báo','Đối chứng','Hành trình model']
    for period,key in [('Hôm nay','today'),('Ngày mai','tomorrow'),('1 tuần','week'),('1 tháng','month')]:
        response=requests.post('http://127.0.0.1:8000/api/v1/forecast',json={'horizon':key},timeout=20)
        response.raise_for_status();expected=response.json()
        app.get('button_group')[0].set_value(period).run();clean(period)
        assert next(m for m in app.metric if m.label=='Số bước từ origin').value==str(expected['steps'])
        plots=forecast_dashboard_figures(expected)
        prices=np.array([expected['origin_close_adjusted']]+[r['predicted_close_adjusted'] for r in expected['path']])
        np.testing.assert_allclose(plots['step_returns'].data[0].y,(prices[1:]/prices[:-1]-1)*100,atol=1e-10)
        np.testing.assert_allclose(plots['cumulative_returns'].data[-1].y,(prices/prices[0]-1)*100,atol=1e-10)
        bounds=np.array([[r['scenario_lower'],r['scenario_upper']] for r in expected['path']])
        np.testing.assert_allclose(plots['band_width'].data[0].y,(bounds[:,1]-bounds[:,0])/prices[1:]*100)
        assert next(m for m in app.metric if m.label=='Biên dưới tại ngày đích').value==f"{bounds[-1,0]:,.0f} VND"
        checks[period]={'steps':expected['steps'],'ok':True}
    next(x for x in app.toggle if x.label=='Hiện biểu đồ phân tích dự báo').set_value(False).run();clean('forecast analytics off')
    assert not any(m.label=='Biên dưới tại ngày đích' for m in app.metric)
    next(x for x in app.toggle if x.label=='Hiện biểu đồ phân tích dự báo').set_value(True).run();clean('forecast analytics on')
    checks['forecast_analytics']={'plots_match_price_ratios':True,'band_width_matches_api':True,'visibility_toggle':True}
    limits=requests.get('http://127.0.0.1:8000/health',timeout=20).json()['forecast_date_limits']
    app.get('button_group')[0].set_value('Chọn ngày').run();clean('custom date')
    if limits['custom_available']:
        next(x for x in app.date_input if x.label=='Ngày muốn dự báo').set_value(None).run();clean('empty custom date')
        assert any('Chọn một ngày đích' in x.value for x in app.info)
        for day in [limits['selectable_min_date'],limits['selectable_max_date']]:
            next(x for x in app.date_input if x.label=='Ngày muốn dự báo').set_value(pd.Timestamp(day).date()).run();clean(day)
            expected=requests.post('http://127.0.0.1:8000/api/v1/forecast',json={'horizon':'custom','target_date':day},timeout=20).json()
            assert next(m for m in app.metric if m.label=='Ngày đích dự kiến').value==pd.Timestamp(expected['effective_target_date']).strftime('%d/%m/%Y')
            assert next(m for m in app.metric if m.label=='Số bước từ origin').value==str(expected['steps'])
        checks['custom_min_max_dates']={'ok':True,'max_date':limits['selectable_max_date']}
    next(x for x in app.selectbox if x.label=='Model dự báo').select('Zero-Return').run();clean('zero')
    assert next(m for m in app.metric if m.label=='Giá cuối khung · VND').value==next(m for m in app.metric if m.label=='Giá origin · VND').value
    checks['zero_return_keeps_origin_price']=True
    assert any('Zero Return giữ nguyên giá' in item.value for item in app.info)
    assert not any(m.label=='Biên dưới tại ngày đích' for m in app.metric)
    assert any('Zero Return là đường so sánh' in item.value for item in app.info)
    assert not any('Tạo vùng kịch bản residual bootstrap' in item.value for item in app.markdown)
    next(x for x in app.selectbox if x.label=='Model dự báo').select('Ridge').run();clean('raw ridge explanation')
    assert [tab.label for tab in app.tabs]==explanation_tabs
    assert any('không cộng bias60' in item.value for item in app.markdown)
    assert any('Ridge thô đã có nhãn' in item.value for item in app.markdown)
    checks['forecast_explanations']={'three_diagrams':True,'ridge_bias_raw_and_zero_return':True,'uses_returned_forecast_numbers':True}
    for page in ['Tổng quan','Đối chứng','Hành trình model']:
        app.radio[0].set_value(page).run();clean(page);checks[page]={'ok':True}
        if page=='Tổng quan':
            for period in ['1 tháng','Toàn bộ']:
                next(x for x in app.select_slider if x.label=='Khoảng quan sát').set_value(period).run();clean(period)
                shown=app.dataframe[0].value
                assert shown.Date.max().strftime('%Y-%m-%d')==requests.get('http://127.0.0.1:8000/health',timeout=20).json()['data_as_of']
                if period=='1 tháng':
                    assert len(shown)==22 and shown.SMA_200.notna().all()
                else:assert len(shown)>2000
            next(x for x in app.selectbox if x.label=='Kiểu đồ thị giá').select('Đường đóng cửa').run();clean('analysis line')
            next(x for x in app.multiselect if x.label=='Lớp chỉ báo trên giá').set_value(['SMA20','SMA50','SMA200','Bollinger20']).run();clean('all indicators')
            next(x for x in app.multiselect if x.label=='Lớp chỉ báo trên giá').set_value([]).run();clean('no overlays')
            next(x for x in app.toggle if x.label=='Hiện phân tích chi tiết').set_value(False).run();clean('analysis details off')
            next(x for x in app.toggle if x.label=='Hiện phân tích chi tiết').set_value(True).run();clean('analysis details on')
            checks['analysis_controls']={'ok':True,'short_and_full_history':True,'indicators_optional':True}
        if page=='Đối chứng':
            next(x for x in app.selectbox if x.label=='Tập ngày dùng chung').select('Tháng 6–10/2026 · 104 cặp, có LSTM').run();clean('cohort')
            next(x for x in app.selectbox if x.label=='Tháng của nhãn thực tế').select('2026-10').run();clean('month')
            assert set(app.dataframe[0].value['Số cặp'])=={21}
            assert 'LSTM_reference' in app.dataframe[0].value.Model.to_list()
            app.multiselect[0].set_value([]).run();clean('empty model selection')
            checks['same_dates_october']=21
    destination=root/'.cache/studio-ui-checks.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print('4 pages / 4 presets + custom min/max dates / analysis controls / Zero Return / historical filters: OK')


if __name__=='__main__':run()
