# Đối chứng training window FPT — 20261029T150501819490Z

## Thiết kế đã khóa trước fit
Nguồn Vietcap, 6 features lợi suất, cấu hình Ridge/XGBoost/LSTM và seed giữ đúng control study 20261029T142732805710Z. Chỉ đổi thời gian của origin train: expanding, 2y, 3y, 5y. Warm-up và context của scaler có thể có ngày trước đầu window; không có context sau origin/cutoff. Evaluation/validation giữ nguyên ngày và nhãn. Chỉ chọn window bằng **mean validation RMSE qua seed**, tie-break theo thứ tự cấu hình, không chọn bằng evaluation.

Có 112 điều kiện window/model/fold/seed, **56 lượt fit mới**, 56 điều kiện tái sử dụng vì cùng train cohort với expanding đã kiểm chứng. Đây không phải 112 lần training mới. Mọi window thiếu đủ số năm từ nguồn vẫn giữ đúng các dòng sẵn có; các window trùng train cohort không được coi là bằng chứng độc lập.

## Kết quả chọn bằng validation, evaluation 2022–2024
Mean-fold không trọng số, mean qua seed trong từng fold. Đây là chẩn đoán hồi cứu đã xem, không phải holdout tương lai.

| Model | Window chọn bằng validation: RMSE | Expanding: RMSE | RMSE giảm |
|---|---:|---:|---:|
| LSTM | 0.0164282 | 0.0163951 | -0.20% |
| Return Persistence | 0.0225811 | 0.0225811 | +0.00% |
| Ridge | 0.0163271 | 0.0163539 | +0.16% |
| Train Mean | 0.0162964 | 0.0162915 | -0.03% |
| XGBoost | 0.0165185 | 0.0164722 | -0.28% |
| Zero-Return | 0.0163497 | 0.0163497 | +0.00% |

Zero Return cùng nhãn/cohort có mean-fold RMSE **0.0163497**. Giảm so với expanding chưa có nghĩa vượt baseline. Xem `validation_selection.csv` để biết window được chọn trong từng fold, gồm cả trường hợp kém control. Không chọn window thắng hậu nghiệm trên evaluation để thay kết quả validation selection.

## Năm 2025 và quyết định
2025 báo cáo riêng trong `summary.csv` và `validation_selection.csv`, không gộp vào bảng chính vì đã được xem ở audit/model development. Không có dữ liệu giá 2026 được tải hoặc đánh giá.

Không tự promote model. Nghiên cứu này nhằm kiểm tra liệu giảm lịch sử train có giúp thích nghi hơn; kết quả không chứng minh một window tốt nhất cho tương lai. Tiếp tục cần forward period chưa dùng để chọn hyperparameter. API/Streamlit và tám hash canonical data/model giữ nguyên. Không commit/push theo yêu cầu người dùng.

## Bằng chứng
`protocol.json` và `source_snapshot.zip` được lưu trước training; input tham chiếu hash gói snapshot ở control study. `cohort_counts.csv` có số mẫu và ngày đầu/cuối train/context; `metrics.csv`, `predictions.csv.gz`, `lstm_history.json`, `manifest.json` lưu cả lượt tốt/xấu và provenance reused/new-fit. `verification.json` tính lại toàn bộ trường metric và baselines; `reload_verification.json` nạp lại unique models, kiểm tra scaler và duplicate forecasts.

```powershell
python -m src.models.window_comparison --config configs/window_comparison_v1.json
python -m src.models.window_comparison_report --study reports/window_comparison/20261029T150501819490Z --reload-artifacts
```
Lần train mới luôn có ID mới. Không chạy lại để ghi đè metric của lượt hiện tại.
