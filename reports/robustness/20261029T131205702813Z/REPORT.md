# Đối chứng độ ổn định và feature ablation FPT

Study `20261029T131205702813Z`; protocol đăng ký trước training lúc `2026-10-29T13:12:05.703194+00:00` (UTC).

## Kết luận của lượt này

- Đã train **63 candidate**: 27 LSTM, 27 XGBoost, 9 Ridge; thêm 5 baseline cho mỗi fold. Có **156 hàng model/phase metrics**, cùng forecast origins trong từng fold/phase.
- LSTM chỉ dùng returns có mean-fold evaluation RMSE `0.0164322`, so với full 16 features `0.0165415`: thay đổi **-0.66%**. Đây là kết quả hồi cứu, không phải mức tăng chất lượng được bảo đảm cho tương lai.
- Returns-only LSTM vẫn cao hơn Zero-Return khoảng **0.26%** theo mean-fold RMSE. Train Mean đạt `0.0163221`; chưa có bằng chứng để thay model đang phục vụ bằng một candidate mới.
- Độ lệch chuẩn RMSE theo seed của full LSTM trung bình `0.00006126`, returns-only `0.00009125`. Ít feature giúp mean error trong mẫu nghiên cứu này nhưng không cải thiện đồng thời mọi chỉ số ổn định.
- **Quyết định: giữ model hiện tại; không tự promote.** Chưa xác minh độc lập lịch HOSE/corporate actions; chưa có acceptance period tương lai chưa dùng tuning.

## Giao thức đối chứng

| Fold | Train đến | Validation đến | Evaluation đến |
| --- | --- | --- | --- |
| eval_2022 | 2020-12-31 | 2021-12-31 | 2022-12-31 |
| eval_2023 | 2021-12-31 | 2022-12-31 | 2023-12-31 |
| eval_2024 | 2022-12-31 | 2023-12-31 | 2024-12-31 |

Seed XGBoost/LSTM: `[42, 123, 2026]`. Ridge xác định chạy một lần/fold/group. Feature groups gồm returns (6), returns+technical (13), full gồm volume (16). LSTM lookback 30, 32 units, Dense 8, Huber, StandardScaler train-only, batch 32, shuffle=False, max 40 epochs, early stopping patience 7. Hyperparameters được giữ cố định; không chọn seed bằng evaluation.

Mọi model dùng cùng origin/target dates và common context 60. Nhãn đi qua ranh giới tập bị loại. Baseline luôn tăng/giảm được khai báo trước study; không thêm sau khi nhìn tỷ lệ nhãn. `splits.csv` lưu ngày và số mẫu chính xác.

## Tổng hợp evaluation: trung bình seed trước, trung bình 3 fold sau

Mean-fold RMSE là trung bình không trọng số các RMSE theo fold, **không phải pooled RMSE**. Cột std là trung bình sample standard deviation giữa các seed trong từng fold. Model chạy một lần không có bằng chứng để ước lượng seed variance. “Fold vượt zero” dùng mean MSE skill so với Zero-Return trong cùng fold, không phải kiểm định ý nghĩa thống kê.

| Feature group | Model | Mean-fold RMSE | Worst-fold RMSE | Mean seed std | Fold có MSE skill > 0 |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | Always Down | 0.0212543 | 0.0240921 | N/A (1 seed/run) | 0/3 |
| baseline | Always Up | 0.0195388 | 0.0240554 | N/A (1 seed/run) | 0/3 |
| baseline | Return Persistence | 0.0225584 | 0.0296110 | N/A (1 seed/run) | 0/3 |
| baseline | Train Mean | 0.0163221 | 0.0211659 | N/A (1 seed/run) | 2/3 |
| baseline | Zero-Return | 0.0163897 | 0.0211437 | N/A (1 seed/run) | 0/3 |
| full | LSTM | 0.0165415 | 0.0214672 | 0.00006126 | 1/3 |
| full | Ridge | 0.0164494 | 0.0213383 | N/A (1 seed/run) | 1/3 |
| full | XGBoost | 0.0167174 | 0.0220803 | 0.00002409 | 1/3 |
| returns | LSTM | 0.0164322 | 0.0211993 | 0.00009125 | 1/3 |
| returns | Ridge | 0.0164030 | 0.0213290 | N/A (1 seed/run) | 1/3 |
| returns | XGBoost | 0.0167062 | 0.0220646 | 0.00003297 | 1/3 |
| returns_technical | LSTM | 0.0164915 | 0.0214868 | 0.00008838 | 1/3 |
| returns_technical | Ridge | 0.0164677 | 0.0213710 | N/A (1 seed/run) | 1/3 |
| returns_technical | XGBoost | 0.0167160 | 0.0220658 | 0.00002139 | 1/3 |

## Chi tiết LSTM: lỗi thay đổi theo cả thời gian và seed

| Fold | Features | Mean seed RMSE | Seed std | Mean balanced accuracy |
| --- | --- | ---: | ---: | ---: |
| eval_2022 | full | 0.0214672 | 0.00012420 | 50.21% |
| eval_2022 | returns | 0.0211993 | 0.00012359 | 49.77% |
| eval_2022 | returns_technical | 0.0214868 | 0.00015185 | 49.72% |
| eval_2023 | full | 0.0125345 | 0.00002484 | 50.68% |
| eval_2023 | returns | 0.0125440 | 0.00008111 | 50.37% |
| eval_2023 | returns_technical | 0.0124234 | 0.00004575 | 49.68% |
| eval_2024 | full | 0.0156229 | 0.00003475 | 51.76% |
| eval_2024 | returns | 0.0155533 | 0.00006905 | 53.57% |
| eval_2024 | returns_technical | 0.0155642 | 0.00006755 | 57.10% |

## Giới hạn và diễn giải

Các giai đoạn 2022–2024 đã nằm trong lịch sử phát triển; kiến trúc/features từng được chọn sau khi xem giai đoạn 2023–2025. Validation/evaluation giữa các fold có chồng lấn lịch sử. Đây là **retrospective diagnostic**, không phải pristine holdout hoặc một phép chọn model không thiên lệch. Không rank theo evaluation rồi công bố winner cho production.

Không kết luận drift hoặc catastrophic forgetting chỉ từ khác biệt các fold. Mức biến động có thể liên quan regime/data/model; study này chưa cô lập nguyên nhân. Pipeline train batch từ đầu, không thực hiện continual fine-tuning. Tối ưu tiếp cần calibration, nhiều bằng chứng ngoài mẫu và shadow forecasts trên dữ liệu tương lai.

Metadata chỉ kiểm chứng nội bộ snapshot: finite features, unique/sorted dates, nhãn được tính độc lập lại từ source closes; không chứng nhận calendar/point-in-time. Mọi pointer serving và reference giữ nguyên trong quá trình chạy.

## Bằng chứng và tái tạo

- [Protocol đã đăng ký](protocol.json), [protocol bytes gốc](protocol.original.json.gz), [source snapshot chính xác](source_snapshot.zip).
- [Data quality](data_quality.json), [split dates/counts](splits.csv), [từng candidate/phase](metrics.csv).
- [Prediction theo origin](predictions.csv.gz), [seed stability](seed_stability.csv), [fold summary](fold_summary.csv).
- [Manifest và hashes](manifest.json), [learning histories](lstm_history.json), [tính độc lập lại metrics](verification.json).

```powershell
python -m src.models.robustness --config configs/robustness_v1.json
python -m src.models.robustness_report
```

Mỗi lần train tạo thư mục mới và lưu đầy đủ kết quả, kể cả kém baseline. Trọng số nghiên cứu nằm ở `models/research/20261029T131205702813Z`; không ghép với scaler/weights của model demo. Chỉ số deployment của ngày 25/10/2026 nằm trong `docs/TRAINING_HISTORY.md`, không so trực tiếp mean-fold RMSE ở đây với một test-year RMSE 2025.

[Nạp lại artifact](reload_verification.json): 63 model, 126 kiểm tra phase; scaler LSTM khớp thống kê train-only.
