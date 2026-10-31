# FPT: recency và hiệu chỉnh bias — 20261030T022440928507Z

## Câu hỏi và giao thức
So cùng 6 return features, StandardScaler fit riêng train, Ridge alpha=100 và window 3 năm. Giao thức được ghi **trước fit**; không chọn alpha/window bằng điểm 2026. Tất cả nhãn 2026 đã xem ở study trước: đây là **chẩn đoán hồi cứu thích nghi**, không phải holdout độc lập mới hay forecast live. Nguồn Vietcap adjusted snapshot đến 2026-10-29, tái dùng đúng ZIP/hash; không tải lại hoặc nối hai mặt bằng giá.

`frozen_2023` là artifact cũ, train 693 cặp common origin/target đến 2023. `rebuilt_2023_same_rows` giữ đúng 693 ngày nhưng tái tạo X/y từ snapshot mới: kiểm tra ảnh hưởng revision. `refit_2023_native` bổ sung coverage native cùng cutoff. `refit_2025_native` chuyển window đến hết 2025. `monthly_refit_native` mô phỏng refit vào đầu tháng, chỉ dùng target ≤ cuối tháng trước; không phải đã chạy live vào tháng đó. Giá lịch sử adjusted hiện tại chưa được chứng nhận point-in-time.

`*_bias60` cộng mean của tối đa 60 residual **đã có nhãn trước/sát origin sau đóng cửa**, cần ít nhất 20; không trừ mean lỗi cả năm rồi sửa ngược quá khứ. 20 origin đầu giữ raw forecast. `RollingMean60` là baseline mean return của 60 nhãn đã biết, warm-start từ 2025. 2026 prefix dùng để cập nhật theo quy tắc định trước; các ngày sau vẫn chỉ được gọi là mô phỏng theo thời gian.

## Kết quả chính — cùng 202 cặp Vietcap 2026
Origin 2026-01-05–2026-10-28, target đến 2026-10-29. Một quan sát tiếp theo, không dự báo toàn bộ đường giá năm. Cohort phụ 200 cặp khớp study trước ở `metrics.csv`; không chọn cohort có điểm đẹp.

| Strategy | N | RMSE return | MAE giá adj. (VND) | Hit (%) | Balanced (%) | MSE skill vs Zero (%) |
|---|---:|---:|---:|---:|---:|---:|
| refit_2025_native_bias60 | 202 | 0.0201116 | 1064.12 | 60.91 | 56.62 | +0.04 |
| monthly_refit_native | 202 | 0.0201124 | 1074.00 | 52.79 | 54.52 | +0.03 |
| Zero-Return | 202 | 0.0201157 | 1081.70 | nan | nan | +0.00 |
| refit_2025_native | 202 | 0.0201191 | 1080.94 | 50.76 | 54.20 | -0.03 |
| monthly_refit_native_bias60 | 202 | 0.0201704 | 1063.57 | 57.87 | 54.25 | -0.54 |
| RollingMean60 | 202 | 0.0201890 | 1075.32 | 52.79 | 47.01 | -0.73 |
| TrainMean_2025 | 202 | 0.0202382 | 1094.53 | 40.61 | 50.00 | -1.22 |
| frozen_2023_bias60 | 202 | 0.0204764 | 1078.43 | 52.28 | 49.35 | -3.62 |
| refit_2023_native | 202 | 0.0206974 | 1114.20 | 43.15 | 50.36 | -5.87 |
| frozen_2023 | 202 | 0.0207058 | 1114.18 | 43.15 | 49.96 | -5.95 |
| rebuilt_2023_same_rows | 202 | 0.0207058 | 1114.18 | 43.15 | 49.96 | -5.95 |

Hit chỉ tính nhãn thực tế khác 0; Zero Return là abstention nên Hit/Balanced N/A. MAPE không phải xác suất dự báo đúng. Price MAE cùng anchor adjusted Vietcap; không phải giá chưa điều chỉnh hiển thị ngày lịch sử.

## Bias, độ ổn định và bất định
`matched_warmup_comparison.csv` so raw/bias trên cùng tập ngày bias-ready (≥20 residual) và interval-ready (≥60), tránh đổi mẫu tạo cải thiện giả. `monthly_metrics.csv` lưu N/RMSE/skill từng tháng; không chọn model theo tháng thắng nhiều nhất. `bias_diagnostics.csv` ghi mean prediction, actual và error. `uncertainty_descriptive.csv` dùng block bootstrap 10 origin, 2.000 resamples, seed 20261030: CI mô tả phụ thuộc một giai đoạn, không hiệu chỉnh cho việc xem và thử nhiều phương án.

Khoảng dự báo danh nghĩa 80% = quantile 10–90% của tối đa 60 lỗi **calibrated trước đó**; đủ 60 mới phát interval. Đây là residual interval thực nghiệm, **không bảo đảm coverage**, không confidence interval của mean forecast. N/coverage thực tế:
- frozen_2023_bias60: N=142, coverage=80.99%, mean width log-return=0.04518.
- refit_2025_native_bias60: N=142, coverage=82.39%, mean width log-return=0.04602.
- monthly_refit_native_bias60: N=142, coverage=83.10%, mean width log-return=0.04613.

## Artifact và kiểm tra
`protocol.json` + `source_snapshot.zip`: quy tắc, source/package versions và hash trước fit. `manifest.json`: ngày train từng dòng, cutoff, hash model và số fit mới. `verification.json`: reload model, kiểm tra thống kê scaler, normal equations Ridge, nhãn/cutoff, forecast, calibration và mọi metric; không fit lại khi verify. Model production/canonical input/pointer giữ nguyên. Weights nghiên cứu ở `models/research/recency-bias_<study_id>/` bị gitignore: khi tái lập trên máy khác cần chuyển đúng artifact kèm hash hoặc chạy study mới, không giả rằng ZIP source đã chứa weights.

## Shadow phía trước
`shadow_bundle.json` refit riêng đến target đã chín mới nhất, kèm calibration snapshot. Bundle **cố định** đến khi chủ động rebuild; không tự refit mỗi tháng hoặc tự cập nhật bias. Đây là ứng viên để ghi forecast tương lai, không promote serving, không tối ưu trên 2026 rồi tuyên bố lợi thế tương lai.

Sổ `reports/forward_shadow/forecasts.jsonl` chỉ append forecast/outcome với hash-chain và ảnh chụp input. Forecast phải được phát thật trước khi có target, không backdate. Chỉ `issued_local_date < actual_target_date` mới đủ điều kiện báo điểm full-session; phát sáng cùng ngày target chỉ là pre-close shadow và bị loại khỏi nhóm này. Hash-chain local không chứng nhận thời gian bên thứ ba; single writer. Chưa có outcome chín thì không có kết luận cải thiện forward. Hướng dẫn: `docs/FORWARD_SHADOW.md`.
