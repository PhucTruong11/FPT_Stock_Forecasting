"""Explicit UI smoke check against the running localhost API; never part of training."""
import json
import sys
from pathlib import Path
import requests
from streamlit.testing.v1 import AppTest

# Cho phép chạy trực tiếp `python src/app/check_studio.py` (không chỉ `python -m`).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def run():
    root=Path(__file__).resolve().parents[2]
    app=AppTest.from_file(str(root/'src/app/streamlit_app.py'),default_timeout=40).run()
    def clean(label):
        assert not app.exception,(label,[x.value for x in app.exception])
        assert not app.error,(label,[x.value for x in app.error])
    clean('default');checks={}
    assert app.radio[0].options==['Tổng quan','Dự báo','Đối chứng','Hành trình model']
    for period,key in [('Hôm nay','today'),('Ngày mai','tomorrow'),('1 tuần','week'),('1 tháng','month')]:
        response=requests.post('http://127.0.0.1:8000/api/v1/forecast',json={'horizon':key},timeout=20)
        response.raise_for_status();expected=response.json()
        app.get('button_group')[0].set_value(period).run();clean(period)
        assert next(m for m in app.metric if m.label=='Số bước từ origin').value==str(expected['steps'])
        checks[period]={'steps':expected['steps'],'ok':True}
    next(x for x in app.selectbox if x.label=='Model dự báo').select('Zero-Return').run();clean('zero')
    assert next(m for m in app.metric if m.label=='Giá cuối khung · VND').value==next(m for m in app.metric if m.label=='Giá origin · VND').value
    checks['zero_return_keeps_origin_price']=True
    for page in ['Tổng quan','Đối chứng','Hành trình model']:
        app.radio[0].set_value(page).run();clean(page);checks[page]={'ok':True}
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
    print('4 pages / 4 horizons / Zero Return / historical filters / empty selection: OK')


if __name__=='__main__':run()
