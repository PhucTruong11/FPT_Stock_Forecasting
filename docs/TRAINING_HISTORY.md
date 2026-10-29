# Nhật ký retraining và bằng chứng phát triển model FPT

Ngày ghi nhận: **25/10/2026**, múi giờ **UTC+7 (Bangkok/Việt Nam)**. Dữ liệu thị trường cố định: 2018–2025. Báo cáo này ghi quá trình phát triển trên cùng một snapshot; không phải chuỗi đánh giá hiệu quả đầu tư qua nhiều tháng.

## Kết luận để team trình bày

- Quy trình hiện có bằng chứng tái lập: snapshot và artifact có hash, các model dùng cùng tập mẫu, scaler chỉ fit train, cấu hình được chọn trên validation, API/UI dùng chung inference.
- LSTM lookback **30** được chọn trong nhóm LSTM theo RMSE validation. **Train Mean** thắng khi so tất cả model trên validation; LSTM là model ML phục vụ demo.
- Colab T4 đã chạy đủ **8 cell**, cùng code/data/config/split với CPU. Model JSON/Keras từ Colab đã nạp lại được trên Windows và khớp **16 hàng model/split metrics** của chính lượt Colab.
- Các lượt v2 có RMSE LSTM gần như bằng nhau. Đây là bằng chứng tái lập, **chưa phải bằng chứng độ dự báo tăng dần sau mỗi lần train**. XGBoost còn lệch nhỏ giữa môi trường và được báo công khai.

## 1. Bản trước review: chỉ dùng làm lịch sử tham khảo

Bản backup trước review có LSTM hai tầng 64/32, Dense 16, MinMaxScaler, lookback 60, tối đa 75 epoch, early stopping patience 12. Metadata ghi 1.087 train, 493 validation, 248 test; thực tế chạy 17 epoch.

| Chỉ số được lưu trong metadata cũ | Validation | Test lịch sử 2025 |
| --- | ---: | ---: |
| RMSE log return | 0.014085049 | 0.020040139 |
| MAE log return | 0.010166675 | 0.013511785 |
| Hit rate theo công thức cũ | 52.59% | 47.39% |

Nguồn: [metadata cũ trích nguyên từ backup](../reports/training_history/evidence/legacy_metadata.json), kèm hash ZIP và member. Không khôi phục weights cũ vào pipeline đang dùng.

**Giới hạn so sánh:** metadata cũ thiếu hash snapshot và dự báo theo từng origin. Hàm hit rate cũ loại cả actual 0 lẫn prediction 0 và dùng 50% khi không có tín hiệu; v2 giữ prediction 0 là abstention và trả N/A khi không có tín hiệu hướng. Scaler, kiến trúc, callbacks và quy trình so mẫu cũng thay đổi. Vì vậy không tuyên bố chênh lệch với bảng v2 là mức cải thiện có kiểm soát hay do riêng một thay đổi gây ra. Không dùng số trong ảnh chụp nếu không đối chiếu được artifact tương ứng.

## 2. Các thay đổi đã thực hiện và bằng chứng

| Vấn đề | Xử lý | Bằng chứng kiểm chứng |
| --- | --- | --- |
| Dễ chạy dữ liệu/code khác nhau giữa máy và Colab | ZIP public snapshot 6 file; kiểm tra SHA256; clone code GitHub đã push | [Snapshot manifest](../data/snapshots/manifest.json), `protocol_checks` trong comparison |
| So sánh model khác tập ngày hoặc lookback gây sai khác mẫu | Giữ common context 60; cùng origin/target dates cho mọi model | Split trong experiment manifest; kiểm thử boundary/sequence |
| Dễ chọn theo test hoặc chỉ xem hit rate | Chọn RMSE validation; thêm baseline zero/mean/persistence; báo MAE, coverage và neutral | `selection.json`, `validation_trials.csv`, `metrics.csv` |
| Alias weights/scaler có thể trỏ nhầm lượt | Artifact theo run ID, manifest, feature order và hash; một pointer dùng demo | [CPU manifest](../models/experiments/20261025T124951351761Z/manifest.json) |
| XGBoost pickle từ Colab không nạp được trên Windows | Đổi sang native JSON `save_model`/`load_model` | [Kiểm chứng chuyển model](../reports/verification/colab_portability.json) |
| Backend và UI có thể dùng logic khác notebook | ModelService dùng builder/scaler chung; Streamlit gọi HTTP API | [API/UI smoke check](../reports/verification/api_ui_smoke.json), 36 unit test |

Nguồn kỹ thuật cho XGBoost: [Model IO chính thức](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html). Tài liệu phân biệt native model IO với pickle memory snapshot; không bảo đảm pickle ổn định/portable. Đổi định dạng lưu **không làm model chính xác hơn**.

## 3. Nhật ký các lượt v2 có manifest

Tất cả thời gian bên dưới là **lúc bắt đầu run**, lấy từ run ID UTC rồi đổi sang UTC+7. Không dùng thời gian sửa file để giả làm thời gian training. Giá trị là log return; cột chỉ số là LSTM được chọn trong nhóm trên validation.

| Giờ 25/10/2026 | Run ID | Môi trường | XGBoost format | Lookback LSTM | Val RMSE | Test RMSE | Test MAE | Test hit rate |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 19:14:48 | `20261025T121448987851Z` | CPU | .pkl | 30 | 0.014032665 | 0.019858318 | 0.013441519 | 54.98% |
| 19:41:29 | `20261025T124129642764Z` | GPU | .pkl | 30 | 0.014032665 | 0.019858318 | 0.013441519 | 54.98% |
| 19:49:51 | `20261025T124951351761Z` | CPU | .json | 30 | 0.014032665 | 0.019858318 | 0.013441519 | 54.98% |
| 20:22:43 | `20261025T132243824425Z` | GPU | .json | 30 | 0.014032665 | 0.019858318 | 0.013441519 | 54.98% |

- **19:14:** bộ v2 cùng benchmark, thiết lập CPU reference ban đầu; XGBoost vẫn lưu pickle.
- **19:41:** Colab kiểm tra cùng code. LSTM tái lập; phát hiện pickle XGBoost không nạp được khi mang về Windows. Kết quả đã lưu trước khi giải phóng session cũ.
- **19:49:** train lại CPU sau sửa JSON. Các metric CPU giữ nguyên, chứng minh thay định dạng lưu không làm thay kết quả dự báo. Khóa reference hiện tại `20261025T124951351761Z`.
- **20:22:** train bản JSON trên Colab T4 từ commit `5eddc45`; tải artifact về Windows và nạp lại thành công. Giữ thành lượt kiểm chứng riêng, không thay pointer phục vụ demo bằng kết quả test tốt hơn.

Các manifest trước sửa được giữ trong [evidence](../reports/training_history/evidence), thay vì giữ alias weights cũ. [runs.json](../reports/training_history/runs.json) chứa hash manifest sau chuẩn hóa LF theo Git attributes, config, split, seed, versions và đường dẫn từng nguồn; [metrics_over_runs.csv](../reports/training_history/metrics_over_runs.csv) chứa toàn bộ 64 hàng model/split, không chỉ kết quả LSTM được chọn.

## 4. Giao thức và cấu hình được khóa

| Thành phần | Thiết lập |
| --- | --- |
| Dữ liệu sau quarantine | 1.940 quan sát hợp lệ; 1.891 feature rows; 16 inputs |
| Chia mẫu chung | Train 1.087, validation 493, test lịch sử 248 |
| Train | Origin 2018-06-19 đến 2022-12-29; target cuối 2022-12-30 |
| Validation | Origin 2023-01-03 đến 2024-12-30; target cuối 2024-12-31 |
| Test | Origin 2025-01-02 đến 2025-12-30; target cuối 2025-12-31 |
| Nhãn | Log return của quan sát hợp lệ kế tiếp; không tự suy ra lịch HOSE |
| Scaler | StandardScaler feature và target; fit trên train |
| LSTM | 32 units, dropout 0.2, Dense 8, Huber, Adam lr 0.001, clipnorm 1 |
| Huấn luyện | Batch 32, shuffle=False, tối đa 40 epoch, early stopping patience 7, seed 42 |
| Chọn cấu hình | Lookback 10/30/60 theo RMSE validation; test không dùng chọn |

Dữ liệu không được forward/back-fill để tạo giá giả. Ngữ cảnh quá khứ có thể đi qua ranh giới train/validation để dự báo tại thời điểm hiện tại; nhãn tương lai đi qua ranh giới tập bị loại. Raw/source/outputs được đối chiếu hash trong manifest.

## 5. Bằng chứng chọn lookback bằng validation

CPU reference `20261025T124951351761Z`:

| Candidate | Validation RMSE | Validation MAE | Validation hit rate |
| --- | ---: | ---: | ---: |
| LSTM/10 | 0.014055344 | 0.010011078 | 55.17% |
| LSTM/30 | 0.014032665 | 0.009997314 | 54.74% |
| LSTM/60 | 0.014078786 | 0.010037336 | 53.45% |

Lookback 30 có RMSE validation thấp nhất trong nhóm LSTM. Chênh lệch nhỏ, chỉ có một seed; chưa đủ để kết luận lookback 30 luôn tốt hơn trong thị trường khác. Chỉ tiêu chọn là RMSE nên không chuyển sang candidate có hit rate cao hơn sau khi xem test.

![Training và validation loss của CPU reference](../reports/experiments/20261025T124951351761Z/learning_curves.png)

Nguồn: [lstm_history.json](../reports/experiments/20261025T124951351761Z/lstm_history.json). Loss theo epoch phản ánh tối ưu trong một lượt; không đồng nghĩa chất lượng ngoài mẫu tăng theo thời gian thị trường.

## 6. Đối chiếu CPU và Colab sau sửa JSON

CPU `20261025T124951351761Z`; Colab `20261025T132243824425Z`. Cùng phiên bản numpy/pandas/sklearn/xgboost/joblib và TensorFlow 2.21.0. Python CPU 3.13.15, Colab 3.13.16; môi trường/hardware khác nhau.

| Model | CPU test RMSE | Colab test RMSE | Chênh lệch tương đối tuyệt đối |
| --- | ---: | ---: | ---: |
| Zero-Return | 0.019917213 | 0.019917213 | 0.000000% |
| Return Persistence | 0.027995444 | 0.027995444 | 0.000000% |
| Train Mean | 0.020021735 | 0.020021735 | 0.000000% |
| Always Up | 0.024133456 | 0.024133456 | 0.000000% |
| Linear Regression | 0.020079600 | 0.020079600 | 0.000000% |
| Ridge | 0.020055121 | 0.020055121 | 0.000000% |
| XGBoost | 0.020017319 | 0.019938554 | 0.393487% |
| LSTM | 0.019858318 | 0.019858318 | 0.000000% |

Hash raw, data, feature order, splits, seed, training config và source code đều khớp. Cùng Train Mean thắng validation và cùng LSTM lookback 30.

**Trạng thái chính thức: `protocol_match_numeric_difference`.** Các baseline tuyến tính/hằng và LSTM gần như khớp; XGBoost không qua ngưỡng so sánh rất chặt của nhóm baseline. Tất cả RMSE nằm trong tolerance kỹ thuật **5% định trước**. Không đổi ngưỡng sau khi xem kết quả, không gọi hai model tương đương thống kê. Chưa cô lập nguyên nhân chênh XGBoost giữa môi trường; không quy cho GPU vì XGBoost không được cấu hình train GPU.

- [Comparison sinh trực tiếp trên Colab](../reports/experiments/20261025T132243824425Z/reference_comparison.json)
- [Comparison sau nạp lại trên Windows](../reports/verification/colab_reloaded_comparison.json)
- [Notebook Colab đã thực thi](../reports/experiments/20261025T132243824425Z/executed_colab.ipynb): đủ 8 code cells, không có code error. Lỗi trình duyệt hiển thị iframe output được tách khỏi trạng thái thực thi Python.
- [Kết quả kiểm tra portability](../reports/verification/colab_portability.json): toàn bộ 16 hàng metrics nạp lại khớp báo cáo Colab.

## 7. Hiện đã cải thiện điều gì?

**Đã chứng minh:** khả năng tái lập, tính nhất quán mẫu/metric, truy vết artifact, chọn model trên validation và khả năng chuyển model Colab → Windows.

**Chỉ quan sát được trên test lịch sử:** LSTM RMSE 0.019858 thấp hơn Zero-Return 0.019917 một lượng nhỏ, nhưng MAE LSTM 0.013442 cao hơn Zero-Return 0.013146. LSTM hit rate 54,98% trên 211 actual khác 0; always-down hậu nghiệm cũng đạt 54,98%. Không coi đây là bằng chứng lợi nhuận.

**Chưa chứng minh:** giảm sai số qua nhiều đợt cập nhật dữ liệu, tốt hơn bền vững qua nhiều seed, vượt baseline trên nhiều cửa sổ walk-forward hoặc có lợi nhuận sau chi phí. Năm 2025 đã được xem trong thử nghiệm trước; không gọi nó là pristine holdout. Giá điều chỉnh không phải giá khớp lệnh lịch sử; chưa xác minh lịch HOSE/corporate actions/point-in-time độc lập.

## 8. Cách lưu bằng chứng cho những lần tiếp theo

1. Trước thử nghiệm ghi hypothesis, thay đổi dự kiến, snapshot, khoảng train/validation/evaluation và metric chính. Không dùng kết quả test để chọn thay đổi.
2. Giữ nhóm đối chứng cùng snapshot/ngày đánh giá; mỗi ablation đổi một yếu tố. Nếu thêm dữ liệu, dùng walk-forward và so trên cùng khoảng ngoài mẫu.
3. Gọi pipeline chung; giữ toàn bộ run directory, model/scaler, manifest, prediction và learning history. Ghi cả lượt không cải thiện.
4. Chạy `python -m src.models.training_history` để tái tạo ledger và kiểm tra tính toán; lệnh này không train hoặc chọn model. CSV và JSON cập nhật từ manifest đã lưu, không từ ảnh chụp.
5. Viết kết luận kèm chênh RMSE/MAE, directional denominator, biến động nhiều seed, giới hạn và quyết định giữ/thay. Chỉ promote theo tiêu chí khóa trước, tách khỏi báo cáo test.

```powershell
python -m src.models.training_history
python -m src.models.compare_runs --candidate models/experiments/<run_id> --output reports/colab_comparison.json
```

[calculation_checks.json](../reports/training_history/calculation_checks.json) tính độc lập lại RMSE/MAE/hit rate từ prediction export của reference và kiểm tra khớp manifest. Bản Markdown này là biên bản ngày 25/10/2026; ledger có thể cập nhật về sau mà không sửa ngược kết luận lịch sử này.


## 9. Cập nhật nghiên cứu ngày 29/10/2026

Đã thực hiện [đợt rolling-origin/feature ablation](../reports/robustness/20261029T131205702813Z/REPORT.md): 63 model train, 156 hàng metrics, 3 fold và 3 seed cho từng model stochastic. Mọi forecast origin được đối chiếu; 63 model đã nạp lại và qua 126 kiểm tra validation/evaluation prediction. Bộ kiểm thử hiện tại có 42 test qua.

Các kết quả này thuộc protocol/window khác với test 2025 bên trên, nên không vẽ chúng nối thành một đường tăng accuracy theo ngày train. LSTM returns-only giảm mean-fold RMSE khoảng 0,66% so với full features, nhưng chưa vượt baseline nhất quán và seed variation không cải thiện đồng thời. Quyết định giữ model phục vụ hiện tại. Chỉ mục study và toàn bộ kết quả tốt/xấu nằm trong [studies.json](../reports/robustness/studies.json).

[Vòng lặp monitoring/retraining và giải thích supervised regression/deploy](ML_WORKFLOW_AND_DEPLOYMENT.md) là thiết kế cho bước tiếp theo; chưa có live monitor hoặc auto-promotion.

## 10. Audit dữ liệu trước vòng train tiếp theo — 29/10/2026

[Báo cáo đối chiếu Yahoo cũ/mới, Vietcap và KBS](../reports/data_audits/20261029T135735663765Z/REPORT.md) phát hiện 56 ngày Yahoo volume=0 được cả hai nguồn kia ghi nhận giao dịch, cùng bất đồng lợi suất tập trung năm 2025. Tải lại Yahoo không tự sửa được. Dry-run cùng feature builder tạo 1.947 mẫu Vietcap so với 1.889 mẫu Yahoo, với 39 origin chung đổi ngày target.

Lượt này **không có training mới**, không ghi thêm accuracy giả vào lịch sử model. Giữ hash cả snapshot và pointer v2. Đề xuất Vietcap làm nguồn nghiên cứu v3, KBS đối chiếu; cần khóa bộ nhãn/cohort chung trước thử nghiệm so sánh nguồn. Metric cũ giữ nguyên, được hiểu là kết quả lịch sử trên snapshot Yahoo chưa hòa giải nguồn.
