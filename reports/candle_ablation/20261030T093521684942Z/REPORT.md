# FPT: A/B gap mở cửa và thân nến — 20261030T093521684942Z

## Giao thức trước fit
A giữ6 returns. B **thay** Log_Return bằng Gap_Log/Body_Log, thành7 feature. Gap=log(Open_t/Close_previous_eligible); Body=log(Close_t/Open_t). Cộng hai phần bằng Log_Return; B thêm một chiều để phân biệt chuyển động qua đêm/trong phiên, không giữ cả3 cột phụ thuộc tuyến tính. Chỉ dùng FPT, không chỉ số/mã khác hoặc target/horizon mới. Cả hai phần đã biết khi origin t đóng cửa, dùng dự đoán quan sát giao dịch kế tiếp.

Prev close lấy đúng chuỗi positive-volume/valid candle của feature builder cũ. Không lấy hôm qua theo lịch, không dùng Open của phiên cần đoán, không fill. Nếu gặp candle invalid thì giữ chính sách cũ (loại candle cho feature, quarantine nhãn chạm candle invalid); chưa được chứng nhận lịch sàn. Snapshot Vietcap khóa đến29/10/2026, không download hoặc sửa schema/canonical16feature.

20 fit mới cho B, 20 artifact A dùng lại sau kiểm tra hash, exact dates, scaler, forecast và versions. Cùng fold/labels/dates/hyperparameters với study direction trước: Ridge_shared và Logistic train nonzero đến30/6; Q3 sigmoid calibration; Q4 threshold validation; evaluation năm sau. Ridge_recent train cả window3y đến31/12 năm trước (gồm return0), giữ thêm làm đối chứng giá. A/B trong từng strategy có đúng cùng dòng train/cal/val/eval. Không đổi C, alpha, grid margin hoặc lớp đứng giá để tìm điểm đẹp.

## Primary2022–2024: các case cố định, không chọn bằng evaluation

| Case | Strategy | Mean-fold RMSE return | Mean-fold Hit nonzero % | Mean-fold balanced % | Mean coverage % |
|---|---|---:|---:|---:|---:|
| A_returns6 | Logistic_calibrated | — | 52.86 | 50.16 | 100.00 |
| A_returns6 | Logistic_raw | — | 55.01 | 51.83 | 100.00 |
| A_returns6 | Logistic_selective | — | 52.86 | 50.16 | 100.00 |
| A_returns6 | Ridge_recent | 0.016219 | 55.60 | 52.30 | 100.00 |
| A_returns6 | Ridge_shared | 0.016244 | 56.76 | 53.36 | 100.00 |
| B_gap_body7 | Logistic_calibrated | — | 52.86 | 50.08 | 100.00 |
| B_gap_body7 | Logistic_raw | — | 55.17 | 51.92 | 100.00 |
| B_gap_body7 | Logistic_selective | — | 50.89 | 49.68 | 84.95 |
| B_gap_body7 | Ridge_recent | 0.016293 | 55.47 | 52.33 | 100.00 |
| B_gap_body7 | Ridge_shared | 0.016295 | 55.75 | 52.47 | 100.00 |

`feature_selection.csv`: Ridge_shared chọn case bằng RMSE validation; Logistic_raw chọn bằng balanced forced validation; tie chọn A. `validation_selected_evaluation.csv` chấm đúng lựa chọn đó. Ridge_recent không được chọn bằng in-sample validation vì nó đã train đến cuối năm; A/B của model này là contrast hồi cứu cố định, chưa phải champion selection. 2025/2026 tách riêng là diagnostic. Mọi giai đoạn đã xem, chưa phải holdout mới.

## Sai số, hướng và selective
Ridge có RMSE/MAE return và price MAE/MAPE cùng anchor Close Vietcap adjusted. Classifier chỉ có xác suất/hướng, không bịa giá dự báo từ p. Hit nonzero và accuracy trên mọi calls giữ riêng; actual0 vẫn trong evaluation, call vào actual0 tính sai. Abstain không được tính đúng nhãn đứng giá. Selective phải đọc coverage từng lớp/calls/neutral counts; không trừ bỏ ngày khó rồi gọi accuracy tăng toàn bộ.

`paired_effects.csv`: B−A Hit/balanced theo điểm%, positive tốt hơn; RMSE gain=100×(1−RMSE_B/RMSE_A), positive tốt hơn. CI95% mô tả từ paired block-bootstrap10 origin/2.000 resamples seed20261030, chỉ các forced strategies. Không gán CI cải thiện cho selective với hai masks khác nhau; `same_selected_days.csv` so control trên đúng ngày B selective giữ lại. CI không sửa hết adaptive research/multiple testing. Fixed alpha/C nhưng đổi cơ sở feature/scaling làm thay regularization hiệu dụng; không quy mọi chênh lệch thành tác động nhân quả của gap.

`jun_oct_metrics.csv` là cùng104 origin/target của Notebook06; không chuyển CI full2026 sang subset. `feature_label_snapshot.csv.gz` giữ raw context, previous date/close, gap/body và nhãn để kiểm tra. `source_snapshot.zip`, `protocol.json`, `manifest.json` giữ nguồn trước fit, config/versions/hashes/dates. `verification.json` reload cả40 artifacts/100 forecast groups, kiểm tra identity/cutoff/scaler, tái lập A và lựa chọn; không fit khi verify.

Không tự promote hoặc sửa serving/API/Streamlit/ledger shadow. Weights nghiên cứu dưới models/research bị gitignore. Phần presentation/decision được viết sau khi xem kết quả, không thay giao thức. Chưa commit/push.
