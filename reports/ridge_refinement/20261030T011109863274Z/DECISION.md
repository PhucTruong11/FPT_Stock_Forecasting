# Phân tích lỗi và tối ưu Ridge — 30/10/2026

**Có cải thiện nhỏ về RMSE so với Ridge trước, chưa đủ để thay model phục vụ.** Cùng Vietcap, sáu features lợi suất, origin/target và nhãn đánh giá. Kết luận chính dùng evaluation hồi cứu 2022–2024; 2025 tách riêng. Không mở giá 2026.

## 1. Lỗi nằm ở đâu?

Forecast chẩn đoán lấy từ các model/window đã chọn bằng validation trước đó. Ngưỡng Volatility20 fit riêng bằng train mỗi fold; SMA50 ±1% chỉ là proxy vị trí giá, không phải nhãn thị trường được xác nhận.

- **243/729 origin biến động cao** (33,3%) tạo **55,6% squared loss** của Ridge. Tuy vậy, Ridge vẫn có MSE skill **+0,91%** so với Zero trong nhóm này. RMSE cao đi cùng độ lớn biến động nhãn, không tự chứng minh model kém baseline.
- Nhóm biến động trung bình có skill **−1,19%**; nhóm giá gần SMA50 khoảng **−2,8%**. Đây là mô tả điểm yếu tương đối, không chứng minh nhân quả hoặc cho phép xóa các ngày khó khỏi evaluation.
- 2022 có sai số tuyệt đối cao ở hầu hết model lẫn baseline; 2023 thấp hơn. Cần đọc cả baseline/skill khi so các năm.

Chi tiết: [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md), [error_segments.csv](error_segments.csv), [regime_thresholds.json](regime_thresholds.json), [diagnostic_verification.json](diagnostic_verification.json).

## 2. Regularization có giúp không?

Grid khóa trước: alpha **0,1 / 1 / 10 / 100 / 1.000**, window expanding/2y/3y/5y. Chọn bằng validation, tie ưu tiên alpha10 rồi thứ tự config. Scaler chỉ fit train; không thêm feature regime.

| Cùng nhãn/cohort, mean-fold 2022–2024 | RMSE |
|---|---:|
| Ridge expanding, alpha10 | 0,0163539 |
| Ridge đã chọn window, alpha10 — bản trước | 0,0163271 |
| Ridge tối ưu alpha trên window cũ | **0,0162983** |
| Ridge chọn cả alpha/window | **0,0162983** |
| Zero Return | 0,0163497 |
| Train Mean expanding | **0,0162915** |

Ridge mới giảm RMSE khoảng **0,18% so với bản trước**, **0,34% so với Ridge expanding**, **0,31% so với Zero Return**. Nhưng RMSE vẫn cao hơn Train Mean khoảng **0,04%**. Chênh nhỏ, chưa có bằng chứng significance hoặc lợi thế giao dịch.

Joint chọn đúng window như lượt trước: 2y/3y/expanding cho 2022/2023/2024, alpha **1.000** ở cả ba fold. Alpha-only và joint cho cùng kết quả: lợi ích đến từ regularization. Alpha nằm ở biên grid không chứng minh tối ưu toàn cục; không tự mở rộng grid theo evaluation.

## 3. Đánh đổi và độ ổn định

- Worst-fold RMSE tăng **0,0212436 → 0,0212771** (kém khoảng 0,16%) dù mean-fold giảm. Vẫn **2/3 fold** vượt Zero, các fold thắng/thua thay đổi. Chưa tăng độ ổn định ở mọi giai đoạn.
- Hệ số và độ dao động forecast giảm, intercept gần như không đổi: dự báo tiến gần trung bình train hơn. [regularization_effects.csv](regularization_effects.csv) lưu bằng chứng; không gọi đây là học được tín hiệu mới.
- Skill nhóm medium cải thiện **−1,19% → −0,28%**, vẫn kém Zero. Low cải thiện **+0,27% → +1,07%**, high giảm **+0,91% → +0,58%**. Đây là chẩn đoán sau fit, không quay lại thay selection.
- Năm 2025: validation chọn window3y/alpha100. RMSE khoảng **0,019693**, tốt hơn window3y/alpha10 khoảng 0,12%, nhưng còn kém Zero và Train Mean. Không gộp 2025 vào bảng chính.

## 4. Kiểm chứng và lịch sử

Lượt hoàn tất: **80 điều kiện**, **48 fit mới**, 32 điều kiện reuse; **60 model duy nhất** gồm 12 control alpha10 và 48 mới. Đã nạp lại model/scaler, kiểm tra **160 bộ prediction và toàn bộ trường metrics**. Alpha10 khớp control ở mọi fold/window/phase. **68/68 tests qua**.

Lượt `20261030T010848752516Z` dừng ở export vì thiếu import TARGET_DATE_COLUMN, sau một fit đầu tiên. Giữ status/source snapshot/partial model/log riêng; không dùng nó để chọn hoặc gộp metric. Sau sửa import và thêm regression test, chạy lại cùng grid. [studies.json](../studies.json) ghi cả lượt dừng và hoàn tất; 48 fit mới ở trên không bao gồm một partial fit của lượt dừng.

Protocol chẩn đoán lưu trước phân tích; protocol fit tham chiếu hash báo cáo lỗi, lưu trước fit đầu tiên. Đây là nghiên cứu **hồi cứu thích nghi** sau các kết quả cũ, không phải holdout chưa xem.

Nguồn: [REPORT.md](REPORT.md), [protocol.json](protocol.json), [source_snapshot.zip](source_snapshot.zip), [primary_summary.csv](primary_summary.csv), [validation_selection.csv](validation_selection.csv), [strategy_metrics.csv](strategy_metrics.csv), [selected_regime_comparison.csv](selected_regime_comparison.csv), [verification.json](verification.json), [inspect_selected.py](inspect_selected.py).

## Quyết định

Giữ Ridge regularized làm ứng viên nghiên cứu; giữ Train Mean/Zero làm đối chứng. Chưa promote model, chưa sửa API/Streamlit. Trước khi đánh giá holdout mới cần khóa cấu hình/provenance dữ liệu; tiếp tục tune trên evaluation đã xem không tạo thêm bằng chứng ngoài mẫu.

[Notebook 05](../../../notebooks/05_training_journey_fpt.ipynb) đã bổ sung kết quả, lưu 12 biểu đồ. Giữ local, **không commit/push**; tám hash canonical data/model pointers không đổi.
