> **Reference lịch sử v2, không phải model local hiện tại.** Xem [quyết định hiện hành và biểu đồ](TRAINING_HISTORY.md), [chạy Ridge local](../README.md).

# Kết quả FPT v2 đã chốt

CPU reference: `20261025T124951351761Z`; seed 42; TensorFlow 2.21.0.

**Quyết định:** dùng pipeline v2 duy nhất; LSTM lookback 30 để demo ML, Train Mean là model thắng validation. Không đổi model theo test.

| Model | Validation RMSE | Test RMSE | Test MAE | Test Hit Rate |
| --- | ---: | ---: | ---: | ---: |
| Zero-Return | 0.014095 | 0.019917 | 0.013146 | N/A |
| Return Persistence | 0.019069 | 0.027995 | 0.021221 | 36.02% |
| Train Mean | 0.013981 | 0.020022 | 0.013429 | 45.02% |
| Always Up | 0.017470 | 0.024133 | 0.019055 | 45.02% |
| Linear Regression | 0.014051 | 0.020080 | 0.013623 | 52.13% |
| Ridge | 0.014035 | 0.020055 | 0.013588 | 52.61% |
| XGBoost | 0.014046 | 0.020017 | 0.013807 | 43.13% |
| LSTM | 0.014033 | 0.019858 | 0.013442 | 54.98% |

Hit Rate test trên 211 actual khác 0; 37 actual đứng giá được báo riêng. Zero-return không phát hướng. LSTM cải thiện RMSE nhỏ, MAE kém zero-return. Always-down hậu nghiệm cũng đạt 54,98% nhưng quyết định khác; không xem 55% là bằng chứng lợi nhuận.

## Đã kiểm chứng

- 36 unit test: data, target boundary, sequence, metric, snapshot, HTTP schema.
- 4 notebook chạy từ đầu; source/scaler/feature order dùng chung.
- Dữ liệu và artifact khớp hash; nạp lại metrics khớp manifest.
- FastAPI model thật khớp inference offline; OHLCV sample đầy đủ cho cùng prediction.
- Streamlit lấy dữ liệu và dự báo qua API; đã thử LSTM và Train Mean.

## Colab

Đã chạy đủ 8 cell trên Colab T4: `20261025T132243824425Z` từ commit `5eddc45`. Hash dữ liệu/code/config/split khớp CPU; cùng Train Mean thắng validation và LSTM lookback 30. LSTM test RMSE gần như trùng CPU. XGBoost test RMSE 0.019939 so với CPU 0.020017 (lệch 0,3935%); trạng thái chính thức là `protocol_match_numeric_difference`, không gọi toàn bộ kết quả trùng tuyệt đối. Tất cả RMSE nằm trong tolerance kỹ thuật 5% định trước; đây không phải kiểm định thống kê.

Model XGBoost native JSON và Keras từ Colab đã nạp được trên Windows; 16 hàng model/split metrics khớp báo cáo Colab. Xem [kiểm chứng portability](../reports/verification/colab_portability.json) và [nhật ký retraining](TRAINING_HISTORY.md).

## Nguồn

- [Manifest](../models/experiments/20261025T124951351761Z/manifest.json)
- [Metrics](../reports/experiments/20261025T124951351761Z/metrics.csv)
- [Predictions](../reports/experiments/20261025T124951351761Z/predictions.csv)
- [API/UI smoke checks](../reports/verification/api_ui_smoke.json)

## Giới hạn

2025 đã được xem trong lần thử trước, là historical test sử dụng lại. Giá điều chỉnh không phải giá khớp lệnh lịch sử; chưa chứng nhận lịch HOSE, corporate actions hoặc point-in-time. Sau lượt reference này đã có [rolling-origin/feature ablation hồi cứu](../reports/robustness/20261029T131205702813Z/REPORT.md); chưa có backtest thực thi.

**Cập nhật nguồn ngày 29/10/2026:** [audit Yahoo–Vietcap–KBS](../reports/data_audits/20261029T135735663765Z/REPORT.md) phát hiện thiếu quan sát giao dịch và bất đồng lợi suất theo ngày trên Yahoo. Các metric bên trên được giữ làm bằng chứng của snapshot v2; chúng chưa chứng minh hiệu quả trên dữ liệu đã hòa giải nguồn. Audit chưa train hoặc promote model mới.

Sau audit đã [train 112 model đối chứng nguồn](../reports/source_comparison/20261029T142732805710Z/DECISION.md), dùng cùng origin/target và nhãn Vietcap cố định. Một số nhánh giảm RMSE nhẹ nhưng chưa vượt baseline nhất quán; chưa thay serving model. Đây là nghiên cứu riêng với protocol/nhãn khác, không ghi đè các metric v2 phía trên.
