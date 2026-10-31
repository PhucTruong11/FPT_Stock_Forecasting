# Kiểm chứng Ridge/API/Streamlit local — 30/10/2026

Đã dọn cache có thể tái tạo, viết lại README/kiến trúc/nhật ký có biểu đồ và kết nối model Ridge mới vào demo local. **0 fit mới**, không download giá, không đổi reference cũ hoặc ghi thêm forward forecast.

- 108 unit tests đạt; log `.cache/local-demo-tests.txt`, hash trong [checks.json](checks.json).
- AppTest chạy mọi tab không exception, đổi sang Zero Return trả +0.000%; đổi cohort có LSTM và tháng10 trả cùng21cặp/model. [Kết quả UI](dashboard_checks.json).
- HTTP POST thật trên127.0.0.1:8000 khớp inference offline; [JSON](http_prediction.json). Đây là output demo snapshot, không phải forecast đã ghi ledger.
- Đã quan sát browser: chart actual/prediction và bộ lọc tháng cập nhật; tabtraining có bảng evidence. [Ảnh mở dashboard](dashboard.png), [ảnh dự báo Ridge](prediction.png).
- Hash8file canonical/pointer và10artifact reference cũ không đổi. Weights local khớp SHA của shadow đã công bố. Ledger vẫn1eventpending.
- Đã xóa59.378.293bytes cache tái tạo; [inventory từng thư mục](cleanup.json). Giữ reports, input/sourceZIP, weights, notebook, failure/negative results và log nghiên cứu.
- HEAD giữ2cc04f2; không staging, commit hoặc push. Không deploy public.

API: http://127.0.0.1:8000/docs · UI: http://127.0.0.1:8501/. Hai tiến trình được để chạy để người dùng xem. Đây là local snapshot đến29/10/2026; bias/interval đóng băng. Chưa có bằng chứng future edge của bundle mới.

Tài liệu current: [README](../../../README.md), [kiến trúc](../../../docs/ARCHITECTURE.md), [nhật ký training](../../../docs/TRAINING_HISTORY.md).
