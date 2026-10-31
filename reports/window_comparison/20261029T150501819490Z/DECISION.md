# Kết luận thử training window — 29/10/2026

**Chưa thay model phục vụ.** Ridge có cải thiện nhỏ khi chọn window theo validation; LSTM và XGBoost không cải thiện trong cùng phép đối chứng. Điều này không chứng minh train gần hiện tại hơn luôn tốt hơn.

Nguồn Vietcap và 6 features lợi suất giữ nguyên; cùng ngày và nhãn validation/evaluation với study nguồn trước. Thử expanding, 2 năm, 3 năm, 5 năm. Chỉ giới hạn origin có nhãn trong train; context/rolling warm-up có thể bắt đầu trước window. Không có thông tin sau origin/cutoff vào scaler. Mọi cấu hình/window/seed được ghi trước khi fit.

## Kết quả chọn bằng validation, mean-fold 2022–2024

| Model | Expanding RMSE | Chọn window bằng validation | Thay đổi |
|---|---:|---:|---|
| Ridge | 0,0163539 | 0,0163271 | Giảm khoảng 0,16% |
| LSTM | 0,0163951 | 0,0164282 | Tăng khoảng 0,20% |
| XGBoost | 0,0164722 | 0,0165185 | Tăng khoảng 0,28% |

Ridge mới thấp hơn Zero Return mean-fold RMSE 0,0163497 một chút, nhưng vẫn cao hơn Train Mean expanding 0,0162915 khoảng 0,22%. Mức chênh rất nhỏ, chưa có bằng chứng significance hoặc khả năng tổng quát ở thị trường tương lai.

Không chốt một window cố định cho mọi thời kỳ: Ridge được chọn 2y cho fold 2022, 3y cho 2023, expanding cho 2024; chẩn đoán 2025 chọn 3y nhưng RMSE lại tăng khoảng 0,17% so với expanding. Các lựa chọn khác nhau này được giữ nguyên trong `validation_selection.csv`; không đổi sang window thắng evaluation sau khi đã nhìn điểm.

## Đếm đúng công việc và bằng chứng

- **56 lượt fit mới**: 24 LSTM, 24 XGBoost, 8 Ridge. Tổng 112 điều kiện; 56 điều kiện tái dùng control có cùng train cohort. Window dài hơn dữ liệu có sẵn không tạo thêm năm dữ liệu hoặc thêm bằng chứng độc lập.
- Đã kiểm tra lại tất cả trường của **320 hàng metrics**, baseline train-only và ngày/nhãn đánh giá chung.
- Đã nạp lại **84 model duy nhất** (28 control cũ + 56 mới), qua 168 đối chiếu prediction; các điều kiện dùng chung weights có forecast giống hệt nhau. Scaler khớp thống kê train-only.
- 62/62 unit tests qua. [Notebook 05](../../../notebooks/05_training_journey_fpt.ipynb) chạy đủ từ đầu đến cuối, lưu 8 biểu đồ và bảng kết quả để team đọc khi chưa chạy kernel.
- Hash tám file canonical data/model giữ nguyên. Không tải/đánh giá giá 2026, không sửa API/Streamlit. **Không commit hoặc push** theo yêu cầu.

Nguồn kiểm chứng: [REPORT.md](REPORT.md), [protocol.json](protocol.json), [source_snapshot.zip](source_snapshot.zip), [cohort_counts.csv](cohort_counts.csv), [summary.csv](summary.csv), [validation_selection.csv](validation_selection.csv), [verification.json](verification.json), [reload_verification.json](reload_verification.json).

## Hướng nghiên cứu sau lượt này

Giữ returns-only, Ridge và các baseline làm đối chứng. Nếu thử tiếp, regularization của Ridge là thay đổi nhỏ, rẻ và dễ diễn giải hơn tăng độ sâu LSTM. Cần protocol mới, chọn bằng validation, cùng cohort/nhãn và giữ đủ kết quả tốt/xấu. Các evaluation 2022–2025 đã được xem nên chỉ tiếp tục dùng cho chẩn đoán; chưa được gọi là forward holdout.
