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

Colab phải dùng notebook GitHub mới, cùng ZIP snapshot và requirements-colab.txt. Ghi thành run riêng, so với CPU reference bằng src.models.compare_runs. Mức chênh RMSE 5% là tolerance kỹ thuật định trước; không bảo đảm cùng weights hoặc ý nghĩa thống kê. Kết quả Colab sẽ được ghi riêng sau khi thực thi, không giả định trước là khớp.

## Nguồn

- [Manifest](../models/experiments/20261025T124951351761Z/manifest.json)
- [Metrics](../reports/experiments/20261025T124951351761Z/metrics.csv)
- [Predictions](../reports/experiments/20261025T124951351761Z/predictions.csv)
- [API/UI smoke checks](../reports/verification/api_ui_smoke.json)

## Giới hạn

2025 đã được xem trong lần thử trước, là historical test sử dụng lại. Giá điều chỉnh không phải giá khớp lệnh lịch sử; chưa kiểm chứng lịch HOSE, corporate actions hoặc point-in-time độc lập. Chưa có walk-forward, ablation feature-group hoặc backtest thực thi.
