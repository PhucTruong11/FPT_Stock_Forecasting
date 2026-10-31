# FPT: phân loại tăng/giảm và “chưa đủ tín hiệu” — 20261030T090621177786Z

## Thiết kế đã khóa trước fit
Hai classifier cấu hình cố định: Logistic Regression và XGBoost Classifier, cùng6 return features. Mỗi fold giữ 3 năm train tới30/6 năm trước; July–September hiệu chỉnh sigmoid độc lập; October–December chọn margin; cả năm tiếp theo để chấm (2026 chỉ đến29/10). Không random split, không refit sau chọn margin. Không thêm VNINDEX, không download mới hoặc sửa API/Streamlit.

Classifier và Ridge_shared dùng đúng cùng các dòng train có return khác0. Ridge_recent là control hiện tại, dùng cả return0 và toàn bộ 3 năm tới31/12 năm trước, có nhiều thông tin gần evaluation hơn; không dùng nó để chọn ngưỡng trên validation mà nó đã học. Riêng2026 tái lập đúng202 forecast Ridge refit_2025_native cũ. Các nhóm khác ngày train không cô lập riêng kiến trúc; so raw/calibrated/selective trong một classifier giữ nguyên weights.

Nhãn tăng/giảm là supervised classification; hồi quy giá vẫn giữ riêng. Raw xác suất là ước lượng P(tăng|không đứng giá) từ tập train nonzero, chưa bảo đảm đáng tin hoặc tốt hơn khi sigmoid. Train/calibration loại return đúng0; **mọi origin evaluation vẫn giữ**. Call trên actual0 tính sai; abstain không được tính là đoán đúng đứng giá. Không chuyển probability thành giá hoặc diễn giải return RMSE cho classifier.

Chọn margin trong [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]: p≥0.5+m ⇒ tăng, p<0.5−m ⇒ giảm, còn lại ⇒ chưa đủ tín hiệu. Tối đa balanced accuracy validation trên calls nonzero, cần coverage≥60%, ít nhất30 calls nonzero, ít nhất10 actual mỗi hướng trong nhóm called. Tie ưu tiên coverage cao/margin nhỏ; không đủ ứng viên thì về margin0, không ép model phải bỏ qua. Validation dùng chọn ngưỡng nên điểm ở đó không phải test. Probability calibration/threshold tuning dùng các nhãn trước evaluation, nhưng **mọi giai đoạn2022–2026 đã được xem trước trong workflow**: đây là nghiên cứu hồi cứu thích nghi, không holdout độc lập mới. Adjusted history hiện tại chưa point-in-time certified.

## Primary 2022–2024 — mean từng fold, không trộn2025/2026 vào lựa chọn

| Phương án | Mean-fold accuracy trên calls (%) | Balanced trên calls khác0 (%) | Coverage (%) | Worst-fold balanced (%) |
|---|---:|---:|---:|---:|
| Always_down | 42.42 | 50.00 | 100.00 | 50.00 |
| Always_up | 51.41 | 50.00 | 100.00 | 50.00 |
| Logistic_calibrated | 49.66 | 50.16 | 100.00 | 50.00 |
| Logistic_raw | 51.67 | 51.83 | 100.00 | 49.25 |
| Logistic_selective | 49.66 | 50.16 | 100.00 | 50.00 |
| Majority_train | 51.41 | 50.00 | 100.00 | 50.00 |
| Ridge_recent | 52.21 | 52.30 | 100.00 | 51.03 |
| Ridge_shared | 53.28 | 53.36 | 100.00 | 51.59 |
| XGBoost_calibrated | 49.93 | 50.24 | 100.00 | 49.76 |
| XGBoost_raw | 48.99 | 49.82 | 100.00 | 47.06 |
| XGBoost_selective | 50.06 | 49.92 | 96.91 | 49.76 |

Coverage tính trên mọi origin. Accuracy trên calls gồm actual0 là sai; Hit nonzero trên calls giữ riêng để đối chiếu metric lịch sử. Balanced chỉ giữa actual tăng/giảm **đã được gọi**, vì vậy phải đọc thêm coverage từng lớp, số calls và correct/all-origins. Không so accuracy trên60% ngày với forced trên100% rồi gọi là cải thiện toàn bộ.

## Đối chứng và bằng chứng
`metrics.csv`: từng fold/strategy, N/calls/abstain, neutral calls, Hit, balanced, coverage tăng/giảm, Brier/logloss trên toàn bộ nhãn nonzero (không chỉ các calls được giữ). `matched_subset_controls.csv`: Ridge/majority/always-up/down trên đúng tập ngày selective gọi tín hiệu. Majority chọn từ train, không lấy lớp chiếm ưu thế evaluation để chọn model. Always-up/down là control cố định.

Block-bootstrap10 origin/1.000 resamples, seed20261030, lưu CI95% mô tả của chênh accuracy selective/forced so majority **trên cùng calls**. Giữ mask khi resample toàn bộ origin. Không xử lý hết việc thích nghi/thử nhiều phương án; calls nhỏ hoặc thiếu một lớp không đủ để diễn giải ổn định. `diagnostic_2026_slices.csv` giữ riêng200 cặp common và104 cặp tháng6–10; không áp CI của full cohort cho các subset.

`phase_counts.csv`: N/cutoff từng phase; `threshold_selection.csv`/`validation_thresholds.csv`: toàn bộ ứng viên và lý do chọn, không tune trên evaluation; `predictions.csv.gz`: calibration/validation/evaluation với trạng thái tăng/giảm/chưa đủ tín hiệu. `manifest.json` lưu 20 base fits và 10 sigmoid fits, exact training dates/hash của mọi artifact. `verification.json` reload tất cả, đối soát phase, forecast, metric và lựa chọn; không fit khi verify. Weights dưới `models/research/` bị gitignore; source/protocol/input hash đã lưu. Không tự promote.

`research_bundle.json` ghi các artifact/classifier của fold2026 và cutoff/margin cố định để tham khảo tích hợp sau. Không thay model serving, ledger shadow cũ hoặc endpoint API. Đây không phải model đã được cập nhật tới29/10/2026.

## Tài liệu phương pháp
- [Scikit-learn: tách chọn ngưỡng khỏi fitting](https://scikit-learn.org/stable/modules/classification_threshold.html).
- [Scikit-learn: probability calibration và dữ liệu độc lập](https://scikit-learn.org/stable/modules/calibration.html).
- [XGBoost: binary logistic và regularization](https://xgboost.readthedocs.io/en/stable/parameter.html).

Các giá trị margin cụ thể và stress2025/2026 được đọc trong Notebook07, không chỉ nêu model có điểm đẹp. Chưa commit/push.
