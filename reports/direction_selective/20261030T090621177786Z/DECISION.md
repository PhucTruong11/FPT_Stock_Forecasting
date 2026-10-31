# Hai nhánh mới: đã triển khai, chưa thay model hiện tại

Study **20261030T090621177786Z**, ngày 30/10/2026. **Không có cải thiện đủ để promote.** Giữ nguyên model serving, API/Streamlit và ledger shadow cũ. Không commit/push.

## Đã làm

- Train Logistic Regression và XGBoost Classifier với cùng 6 return features; đối chứng Ridge cùng dòng train và Ridge recent hiện có. Cấu hình khóa trước fit, chưa mở rộng grid sau khi xem điểm.
- Phân chia theo thời gian: train 3 năm đến 30/6 năm trước → sigmoid calibration Q3 → chọn margin Q4 → evaluation năm sau. Nhãn qua ranh giới phase bị loại. 2022–2024 là primary, 2025/2026 chẩn đoán riêng; toàn bộ đã được xem ở nghiên cứu trước nên không phải holdout mới.
- Chế độ “chưa đủ tín hiệu” với margin chọn bằng validation: coverage tối thiểu 60%, đủ mẫu ở cả hai hướng. Lưu mọi threshold candidate, lý do chọn, calls/abstain, coverage từng lớp và baseline trên cùng ngày.
- Giữ actual đứng giá trong đánh giá. Accuracy trên mọi calls và Hit chỉ trên actual khác 0 là hai cột riêng. Abstain không được tính đúng nhãn đứng giá. Probability và direction không tự chuyển thành giá dự báo.

## Kết quả primary — mean-fold balanced 2022–2024

| Phương án | Balanced trên calls nonzero | Mean coverage |
|---|---:|---:|
| Ridge_shared, cùng dòng train | **53,36%** | 100% |
| Ridge_recent, thêm training đến cuối năm trước | 52,30% | 100% |
| Logistic raw | 51,83% | 100% |
| Logistic sau sigmoid / selective | 50,16% | 100% |
| XGBoost raw | 49,82% | 100% |
| XGBoost sau sigmoid | 50,24% | 100% |
| XGBoost selective | 49,92% | 96,91% |

Chỉ XGBoost/fold 2022 chọn margin 0,025; gọi 225/248 phiên (90,73%). Accuracy trên mọi calls tăng 51,61% → 52,00%, nhưng balanced giảm 50,95% → 50,00%. Không thể báo đây là cải thiện chung. Chín lựa chọn còn lại chọn margin 0; quy tắc ưu tiên coverage khi balanced ngang nhau.

## 2026: vì sao hit cao hơn nhưng chưa tốt hơn?

Sau sigmoid, cả Logistic và XGBoost đoán **giảm 202/202 phiên**. Có 80 tăng, 117 giảm, 5 đứng giá. Vì vậy accuracy trên mọi calls là 117/202 = **57,92%**; Hit trên nhãn nonzero là 117/197 = **59,39%**; balanced **50%**. Đúng bằng control luôn giảm. Đây không phải khả năng phân biệt hướng tốt hơn.

Calibration dùng 62 nhãn nonzero Q3/2025, trong đó 43,55% tăng. Kiểm tra đại số artifact cho thấy p(tăng) sau sigmoid của Logistic nằm trong **[0,4241; 0,4504]**, XGBoost **[0,3781; 0,4969]**, đều dưới ngưỡng 0,5. `probability_diagnostics.csv` lưu hệ số/intercept, min/max và counts; không fit lại. Brier 2026 giảm (Logistic 0,25515 → 0,24144; XGBoost 0,25821 → 0,24063) nhưng không tạo thêm khả năng phân biệt hướng. Kết quả chưa cho phép kết luận một cách hiệu chỉnh khác chắc chắn sẽ tốt hơn.

Validation 2025 chọn margin 0 cho cả hai. Bộ lọc được triển khai, nhưng **chưa được bật cho bundle 2026** theo quy tắc đã khóa. Không tự nâng margin sau khi nhìn evaluation để tạo hit đẹp.

## Đối chiếu đúng 104 phiên tháng 6–10 với Notebook 06

| Phương án | Hit trên 99 actual khác 0 | Balanced | Coverage |
|---|---:|---:|---:|
| Ridge+bias cũ, reference đã công bố | **62,63%** | **57,95%** | 100% |
| Logistic raw | 43,43% | 49,29% | 100% |
| XGBoost raw | 47,47% | 51,28% | 100% |
| Logistic / XGBoost selective | 60,61% | 50,00% | 100% |
| Luôn giảm | 60,61% | 50,00% | 100% |

Reference Ridge+bias được bổ sung sau kết quả để kiểm tra với bảng cũ, **không dùng chọn threshold/model**. Source/hash/script riêng trong `diagnostic_provenance.json` và `inspect_probabilities.py`. Không so 104 với 202 phiên như thể cùng mẫu số.

## Bằng chứng và cách sử dụng

20 base fits + 10 sigmoid fits, 30 artifact được reload; 55 metric groups, 10 threshold choices, 70 validation threshold rows và 60 matched-control groups được đối soát. Ridge_recent tái lập đúng 202 forecast Ridge refit_2025_native cũ. **93 tests đạt**, tám hash canonical/pointer không đổi. Verify không refit, không download. XGBoost chỉ dùng seed 42 trong lượt này; chưa đánh giá độ ổn định seed riêng.

[Notebook 07](../../../notebooks/07_direction_and_insufficient_signal_fpt.ipynb) lưu biểu đồ primary, forced/selective, validation threshold, baseline cùng subset, Brier và tín hiệu từng ngày. [Báo cáo phương pháp](REPORT.md) giải thích định nghĩa metric và giới hạn. `source_snapshot.zip` giữ nguyên source trước fit; phần trình bày notebook được bổ sung chẩn đoán sau khi xem kết quả, không thay model hoặc lựa chọn ngưỡng.

`research_bundle.json` + `src.models.predict_direction.predict_research_signal(bundle_path, feature_frame, model_name)` cho suy luận offline, xuất probability, direction và `tang`/`giam`/`chua_du_tin_hieu`. Cần causal features có Date sau cutoff chọn ngưỡng; helper kiểm tra hash artifact. Đã đối chiếu cả 2 classifier × 202 dòng với forecast lưu. Hàm này không xác nhận bar đã đóng cửa, không ghi forecast live và không kết nối API/Streamlit. Bundle train nhãn chỉ tới 30/6/2025, không giả là đã cập nhật tới 29/10/2026.

Ưu tiên nghiên cứu tiếp: kiểm tra calibration trên nhiều cửa sổ lịch sử và thử thông tin mới bằng ablation; thu forecast thật trước target. Chưa có cơ sở tăng layer/epoch hoặc bật selective để tuyên bố tăng accuracy. Không tự sửa protocol hiện tại để tìm điểm đẹp hơn.
