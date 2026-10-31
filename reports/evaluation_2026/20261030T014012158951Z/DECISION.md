# Model hiện tại dự báo FPT năm 2026 có ổn không? — 30/10/2026

**Phù hợp demo/nghiên cứu, chưa có bằng chứng đủ tốt để dùng một mình cho dự báo thực tế hoặc quyết định giao dịch.** Các model đóng băng đều có RMSE lợi suất cao hơn baseline giữ nguyên giá hiện tại trên cohort chung 2026. Không dùng MAPE nhỏ để gọi là “accuracy 98%”.

## Đã chấm thế nào?

Weights/scaler/model choices được khóa lúc 08:40 UTC+7 ngày 30/10, trước request dữ liệu 2026. Không refit hoặc chọn lại theo 2026. Chấm **200 cặp origin–target chung**: origin 05/01–28/10, target tới **29/10/2026**. Không dùng giá chưa đóng cửa ngày 30/10. Nhãn tham chiếu Vietcap, KBS dùng đối chiếu nguồn.

Model chỉ dự báo **một quan sát hợp lệ kế tiếp** sau khi biết OHLCV của origin; context được cập nhật bằng các giá đã quan sát trong 2026. Không phải dự báo cả đường giá 2026 từ cuối 2025.

Đây là **temporal evaluation hồi cứu**, không phải các dự báo đã được ghi live trước sự kiện. Model được train vào tháng 10/2026 bằng lịch sử cũ; adjusted history hiện tại chưa được chứng nhận point-in-time. Sau lượt này, khoảng 2026 đã mở được ghi trong [holdout_exposures.json](../../holdout_exposures.json), không còn là holdout chưa xem cho những lần tune tiếp.

## Các chỉ số chính — cùng 200 cặp ngày, cùng nhãn

| Model | Return RMSE | Return MAE | Price MAE (VND adj.) | MAPE giá | Hit rate | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Zero Return / giữ giá origin | **0,020114** | **0,015388** | **1.082** | **1,540%** | N/A | N/A |
| LSTM serving | 0,020428 | 0,015523 | 1.092 | 1,555% | 51,28% | 54,27% |
| XGBoost serving | 0,020489 | 0,015725 | 1.106 | 1,576% | 43,59% | 47,65% |
| Ridge serving | 0,020530 | 0,015522 | 1.091 | 1,556% | 46,15% | 50,43% |
| Ridge research mới | 0,020727 | 0,015908 | 1.116 | 1,596% | 42,56% | 49,79% |
| Ridge research cũ | 0,020795 | 0,015960 | 1.120 | 1,601% | 42,05% | 48,72% |

Price MAE ở bảng là **giá đã điều chỉnh, tất cả forecast dùng cùng anchor Close Vietcap tại origin**. Không phải giá thực thi lịch sử và không trộn native price level Yahoo với Vietcap để xếp hạng model. MAPE 1,555% chỉ là sai số phần trăm giá, **không có nghĩa 98,445% forecast đúng**. Naive last-close cũng đạt MAPE 1,540%, nên đường giá nhìn gần thực tế chưa chứng minh giá trị ML.

LSTM đang phục vụ là 16 features/lookback30, train tới 2022, chọn bằng validation2023–2024. Ridge research mới là 6 features, train2021–2023, window3y/alpha100 do validation2024 chọn; chưa train lại tới 2025. Vì khác nguồn/cutoff/features, so LSTM với research Ridge không cô lập riêng thuật toán.

## Ưu điểm có bằng chứng

- Pipeline tái lập được, artifact/scaler có hash; dự báo nhanh với giao thức input rõ ràng, không fit bằng nhãn 2026.
- LSTM có balanced accuracy **54,27%** trên đoạn này, bắt được khoảng **69,23% ngày tăng**. Đây là tín hiệu mô tả cần kiểm chứng tiếp, không đủ để kết luận lợi nhuận.
- Ridge nhẹ, dễ giải thích; chỉnh alpha trước khi mở 2026 giảm RMSE khoảng **0,32% so với Ridge research cũ**. Có thể phân rã forecast đúng thành intercept + đóng góp từng feature; không gán đóng góp đó là nhân quả.

## Nhược điểm và phần cần cải thiện

### 1. Chưa thắng baseline về sai số lợi suất/giá
LSTM MSE skill so Zero **−3,15%**; block-bootstrap 95% CI khoảng **[−6,55%; +0,47%]**, không chứng minh cải thiện. Ridge research mới **−6,19%**, CI khoảng **[−10,59%; −2,05%]**, kém Zero trong phép resampling này. Khoảng CI là mô tả dựa trên block10 origin/2.000 resamples, không bảo đảm tương lai hoặc xử lý mọi bias phát triển.

LSTM chỉ vượt Zero ở **4/10 tháng**, Ridge mới **3/10 tháng**. Mỗi tháng chỉ 15–23 forecast; không lấy một tháng đẹp để chốt chất lượng. Tháng 3 có sai số cao ở model và baseline; tháng 10 Ridge có skill khoảng −27,2% so Zero.

### 2. Thiên lệch tăng trong giai đoạn nhiều ngày giảm
Trong 195 nhãn khác 0: 78 tăng, 117 giảm; 5 origin trung tính. Luôn đoán giảm đạt **60% hit**, balanced accuracy 50%. Không khuyến nghị chọn “always down” dựa vào lớp chiếm ưu thế sau khi đã xem dữ liệu.

LSTM dự báo tăng ở **63%** origin, chỉ bắt được **39,32% ngày giảm**. Ridge mới dự báo tăng **86,5%**, bắt được **13,68% ngày giảm**. Mean log-return thực tế khoảng **−0,256%/quan sát**, LSTM dự báo **+0,107%**, Ridge **+0,219%**. Intercept train dương và các hệ số trên return5/20 đang tạo đóng góp dương khi inputs gần đây âm; kiểm tra đại số trong `ridge_feature_contributions.csv`, không coi đó là giải thích nhân quả thị trường.

### 3. Training cũ, inputs thay phân bố và OHLCV thiếu biến cố mới
LSTM chưa học nhãn sau 2022; research Ridge chưa học nhãn sau 2023. Mean Return20 ở training Ridge khoảng +0,0258, còn inputs2026 khoảng −0,0378; khoảng 13,9% Return20 nằm ngoài khoảng quantile1–99% train. Đây là thay phân bố mô tả, không tự chứng minh concept drift hoặc “model quên”.

OHLCV/lagged returns không trực tiếp biết kết quả kinh doanh, tin tức, thị trường chung hay biến cố sau origin. Chưa có prediction intervals được hiệu chỉnh hoặc backtest thực thi có phí/slippage. RMSE/Hit rate không phải xác suất đúng của từng dự báo.

### 4. Nguồn dữ liệu vẫn phải theo dõi
2026 Vietcap/KBS có cùng 203 ngày volume dương, không nến sai; 202 cặp lợi suất chung không có chênh trên10bps, chênh lớn nhất khoảng2,69bps. Đây là đối chiếu provider, không chứng nhận sàn.

Yahoo có một ngày **09/09/2026** volume0 trong khi Vietcap/KBS ghi nhận giao dịch, và 4 cặp lợi suất khác Vietcap trên10bps. Cohort giao còn 200 cặp; chấm Ridge trên full Vietcap202 cặp vẫn kém Zero, nên kết luận không chỉ do bỏ hai cặp của nguồn Yahoo.

## Ưu tiên cải thiện

1. **Chốt nguồn/provenance và cập nhật training theo thời gian.** So bản cũ với bản refit bằng dữ liệu trước cutoff triển khai, giữ cấu hình đã khóa. Dữ liệu 2026 vừa xem chỉ còn phục vụ chẩn đoán/phát triển; cần forward period sau cutoff chưa dùng chọn model.
2. **Kiểm soát bias và regime trên validation theo thời gian.** So rolling mean/Zero, hiệu chỉnh bias bằng dữ liệu quá khứ đã có nhãn; không trừ mean lỗi toàn bộ 2026 rồi chấm lại chính 2026 để gọi là cải thiện ngoài mẫu. Chưa có bằng chứng cần tăng layer/epoch LSTM.
3. **Bổ sung uncertainty trước tích hợp sử dụng.** Hiệu chỉnh khoảng dự báo và kiểm tra coverage trên dữ liệu sau đó, cùng log timestamp/model/source/input. Features thị trường chung hoặc sự kiện chỉ thử bằng ablation với timestamp khả dụng rõ ràng; vẫn chỉ dự báo FPT.

Không refit hoặc promote trong lượt đánh giá này. Giữ nguyên API/Streamlit và canonical models; không commit/push.

## Bằng chứng
[REPORT.md](REPORT.md), [metrics.csv](metrics.csv), [monthly_metrics.csv](monthly_metrics.csv), [predictions.csv.gz](predictions.csv.gz), [verification.json](verification.json), [data_quality.json](data_quality.json), [input_shift.csv](input_shift.csv), [bias_diagnostics.csv](bias_diagnostics.csv), [inspect_bias.py](inspect_bias.py).

Một lỗi độ chính xác số ở price export được phát hiện trước công bố: exp(float32) khác arithmetic metric float64 tối đa khoảng0,011 VND. Đã chuẩn hóa derived prices/metrics từ **return forecasts đã lưu**, không refit/tải lại/chọn model. `initial_export.zip`, `numeric_export_correction.json`, `normalize_numeric_export.py` giữ bằng chứng; model/data trước mở 2026 được giữ nguyên byte trong các snapshot ZIP.
