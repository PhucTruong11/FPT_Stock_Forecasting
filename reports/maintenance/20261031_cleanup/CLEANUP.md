# Rà soát và dọn workspace — 31/10/2026

Đã xóa **153 file tạm/cache** trong66mục, tổng7.556.999bytes. **70 file hỗ trợ cũ** được gom vào [workspace_support.zip](workspace_support.zip), kích thước1.890.162bytes, và kiểm tra SHA256 từng member trước/sau dọn.

## Đã dọn

- Ảnh PNG xem thử notebook trong `.cache/*-notebook-qa` / `training-journey-qa`; output notebook và ảnh báo cáo chính vẫn giữ nguyên.
- Dữ liệu giải nén và thư mục chuẩn bị cũ trong `.cache/evaluation-2026`, `source-audits`, `source_audit_docs`, `source-comparison-failed-preparation`. Bản rời được lưu trong ZIP hỗ trợ; snapshot/evidence nghiên cứu gốc không sửa.
- Log nghiên cứu/kiểm thử và script chạy một lần ở gốc `.cache`, đã lưu nguyên bytes trong ZIP trước khi bỏ bản rời. Có cả PID record cũ không còn được code sử dụng.
- Cache `__pycache__` dưới `src/` và `tests/`, cùng thư mục test tạm rỗng. Không chạm môi trường `venv`.

## Giữ nguyên

868 file code/config/data/model/notebook/docs/report đã được đối chiếu SHA256 và không đổi. Không xóa model đối chứng hoặc các lượt thử không cải thiện. Giữ hai pointer reference, bundle Ridge local, calibration, raw, dataset snapshots và forward ledger. Giữ cache runtime IPython/Jupyter/Matplotlib/yfinance và bốn log API/Streamlit đang chạy.

Các báo cáo cũ có thể nhắc đường dẫn log `.cache/...`. Xem `archive_files` trong [inventory.json](inventory.json): `archive_member` là đúng đường dẫn cũ bên trong ZIP, SHA256 giữ nguyên để đối chiếu với metadata trước đây. Có thể trích riêng member cần kiểm tra; không cần khôi phục cả workspace.

## Kiểm chứng sau dọn

- [verification.json](verification.json):868file được bảo vệ không đổi;70member ZIP khớp hash; các mục đã chọn đều được xóa.
- Nạp lại `LocalRidgeService` bằng một tiến trình Python mới và dự báo Ridge+bias60 thành công; không dựa vào module/model đang giữ trong RAM của API.
- [http_smoke.json](http_smoke.json): APIready, StreamlitHTTP200, endpointforecast trả kết quả hợp lệ sau dọn. Dữ liệu vẫn đến29/10; không tải mới, ghi ledger hoặc train thêm.
- HEAD giữ `2cc04f2dfc4643d3de22a6b04b52e25a4aecb922`; không staging/commit/push. Không đổi logic nên không chạy lại toàn bộ lượt train hoặc bộ test nghiên cứu.

Không có file của `.git`, `venv`, model, dataset hoặc báo cáo gốc bị xóa. Dashboard tiếp tục chạy tại http://127.0.0.1:8501/.
