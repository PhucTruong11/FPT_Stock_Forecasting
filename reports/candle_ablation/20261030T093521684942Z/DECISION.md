# Gap/thân nến FPT: hoàn tất A/B, giữ control 6 feature

Study **20261030T093521684942Z**, ngày 30/10/2026. **Trong cấu hình đã thử, chưa có cải thiện ổn định để thay A.** Không thay model phục vụ, API/Streamlit hoặc ledger shadow. Không commit/push.

## Đối chứng đã làm

- **A_returns6:** 6 feature cũ, giữ nguyên artifact đã công bố.
- **B_gap_body7:** thay Log_Return bằng Gap_Log và Body_Log, không thêm cả ba cột phụ thuộc tuyến tính. Chỉ FPT OHLCV, không thêm mã/chỉ số khác.
- `Gap_Log = log(Open_t / Close_previous_eligible)`, `Body_Log = log(Close_t / Open_t)`. Cả hai đã biết sau khi origin t đóng cửa, không dùng Open tương lai. Tổng bằng Log_Return_t; B cho phép học riêng chuyển động qua đêm/trong phiên.
- Giữ exact dates, nhãn, feature còn lại, phase, window và alpha/C. Ridge_shared/Logistic dùng cùng dòng train nonzero; Ridge_recent dùng đủ window tới cuối năm trước, A/B vẫn cùng ngày. Không chọn feature bằng in-sample validation của Ridge_recent.
- 20 fit B mới, 20 artifact A tái dùng; không download. Source/config/package versions được chốt trước fit. Các giai đoạn đều đã xem, là nghiên cứu hồi cứu thích nghi.

## Primary 2022–2024: mean riêng từng fold

| Metric / strategy | A: 6 returns | B: gap/body7 | Thay đổi |
|---|---:|---:|---:|
| Ridge_shared RMSE return | 0,0162437 | 0,0162947 | Tăng 0,31% |
| Ridge_recent RMSE return | 0,0162190 | 0,0162930 | Tăng 0,46% |
| Ridge_shared balanced | 53,36% | 52,47% | −0,89 điểm % |
| Logistic raw balanced | 51,83% | 51,92% | +0,09 điểm % |

Logistic raw tăng khoảng 0,87/0,83 điểm % balanced ở 2022/2023 nhưng giảm 1,44 điểm % ở 2024. CI95% paired block-bootstrap của từng thay đổi này đều chứa 0; chưa chứng minh cải thiện. Không diễn giải +0,09 điểm % thành tăng accuracy 9%.

Ridge_recent B có point estimate RMSE tốt hơn ở 2024 (rất ít, khoảng0,009%) và 2025 (1,04%), kém hơn ở 2022/2023/2026. CI RMSE gain của các fold đều còn chứa 0. Ridge_shared B tốt hơn ở 2025 khoảng0,65% nhưng kém ở các năm còn lại. Giữ 2025 riêng diagnostic, không lấy năm đẹp thay cho primary.

## Lựa chọn bằng validation

Ridge_shared chọn A ở 2022/2023/2025, B ở 2024/2026; Logistic_raw chọn B ở 2022–2024, A ở 2025/2026. Khi chấm đúng case được chọn, mean-fold RMSE primary của Ridge_shared 0,0162481 so A 0,0162437; balanced 53,23% so A 53,36%. Chưa giúp. Validation chọn B không đồng nghĩa evaluation chắc chắn tốt hơn; không sửa lựa chọn sau khi nhìn kết quả.

## 2026 đầy đủ — cùng 202 cặp, target đến 29/10

| Ridge_recent | A | B |
|---|---:|---:|
| RMSE return | 0,0201191 | 0,0204083 |
| MAE giá adjusted (VND) | 1.080,94 | 1.096,73 |
| Hit / 197 actual khác0 | 50,76% | 44,67% |
| Balanced accuracy | 54,20% | 48,87% |

RMSE B cao hơn 1,44%; CI95% mô tả cho **RMSE gain B so A** [−3,28%; +0,16%] vẫn chứa 0. Vì vậy không gọi là khác biệt chắc chắn, nhưng chưa có lý do thay A. Logistic raw B balanced 50,72% so A52,82%; cả hai sau calibration vẫn nghiêng mạnh về giảm.

Riêng cùng104 phiên tháng6–10 của Notebook06: Ridge_recent RMSE A0,0168589 → B0,0169331; Hit50,51% →44,44%, balanced55,13% →49,68%. Đọc `jun_oct_metrics.csv`; không trộn metric104 với202 phiên hoặc dùng CI full cohort cho subset.

## “Chưa đủ tín hiệu”

Case B chỉ chọn margin0,025 ở fold2023; các fold khác margin0. Trên evaluation2023 nó giữ136/248 phiên (54,84%), Hit nonzero41,46%, balanced49,04%. Ngưỡng coverage60% chỉ là điều kiện chọn trên validation, **không bảo đảm coverage tương lai**. Không retune trên evaluation để ép coverage/accuracy đạt mong muốn. `same_selected_days.csv` so controls trên chính136 ngày đó và các fold còn lại.

## Bằng chứng, giới hạn và artifact

- `verification.json`: reload40 artifacts,100 forecast groups; A khớp5.975 forecast rows cũ; exact dates, scaler, Ridge normal equations, gap/body identity và nhãn được kiểm tra. Verify không fit.
- `paired_effects.csv`: paired bootstrap10 origin/2.000 resamples, seed20261030; CI mô tả, không sửa hết multiple testing/adaptive research. Selective khác mask không được gán CI cải thiện như so forced cùng ngày.
- `source_snapshot.zip` giữ nguyên source trước fit; notebook được thêm nhận xét sau kết quả, không đổi model/giao thức. Weights ở models/research bị gitignore.
- Fixed alpha/C nhưng feature basis/scaling đổi regularization hiệu dụng; không quy chênh lệch thành tác động nhân quả của gap.
- **100 tests đạt**; kiểm tra không nhìn tương lai, skip đúng untraded/invalid và giữ nguyên schema16feature/nhãn canonical. Tám hash canonical/pointer và model phục vụ giữ nguyên.

[Notebook 08](../../../notebooks/08_gap_body_feature_comparison_fpt.ipynb) có biểu đồ decomposition, CI A/B, giá thực tế so hai model, xác suất/hướng và bảng từng phase. [REPORT.md](REPORT.md) giải thích giao thức/định nghĩa; `protocol.json`, `manifest.json`, `feature_label_snapshot.csv.gz`, `predictions.csv.gz` giữ evidence để tái lập.

Quyết định: giữ6feature làm control. Chưa thêm B vào API/Streamlit hoặc default inference. Nếu tiếp tục thử phải tạo protocol mới và kiểm chứng trên dữ liệu phù hợp, không thêm case sau khi thấy điểm xấu rồi giấu lượt này. Cần forecast thực tế trước target để có bằng chứng forward.
