# Kết luận đối chứng nguồn và hướng tiếp theo — 29/10/2026

**Chốt hiện tại:** tiếp tục nghiên cứu bằng Vietcap vì audit nguồn cho thấy dữ liệu đầy đủ và nhất quán hơn, nhưng **chưa thay model serving**. Đổi nguồn làm một số kết quả tốt hơn; chưa tạo được lợi thế dự báo ổn định so với baseline. Lượt này không đủ căn cứ công bố “model mới chính xác hơn” trên thị trường tương lai.

## Đã làm gì

Đã train **112 model**: 48 LSTM, 48 XGBoost, 16 Ridge. Hai nguồn Yahoo v2/Vietcap ứng viên v3, hai nhóm features returns/full, bốn fold 2022–2025; XGBoost/LSTM dùng seed 42, 123, 2026. Không đổi cấu hình theo kết quả. Các fold 2022–2024 là so sánh chính; 2025 là phần chẩn đoán đã xem, báo cáo riêng.

Giữ **cùng 1.791 cặp origin–target đủ context** cho hai nguồn. Từng fold dùng cùng số mẫu train/validation/evaluation. Nhãn train native từng nguồn; nhãn validation/early stopping/evaluation đều là Vietcap cố định. Không cho v3 thêm labeled training rows. Có 601 origin mà lịch sử 30 quan sát context của hai nguồn gồm các ngày khác nhau; vì vậy phép so sánh bao gồm ảnh hưởng của độ đầy đủ context, không cô lập riêng sai số báo giá.

58 unit tests đã qua trước lượt chuẩn bị đầu tiên; sau khi sửa lỗi parser trước training và thêm regression test, **59/59 unit tests qua**. Không có lượt model train thất bại trong study. Đã kiểm tra độc lập 264 hàng metrics và nạp lại 112 model, khớp 224 bộ prediction; scaler khớp thống kê train-only. Toàn bộ tám hash canonical data/model pointers giữ nguyên. Features Yahoo dựng lại khớp chính xác features v2 cũ, sai khác lớn nhất bằng 0.

## Kết quả chính 2022–2024

Đây là trung bình không trọng số của RMSE từng fold; trong fold lấy trung bình qua seed. Dấu giảm được tính từ hai nhánh trong cùng nghiên cứu này, không lấy RMSE của run cũ dùng nhãn Yahoo để so.

| Trường hợp | Yahoo | Vietcap | Diễn giải |
| --- | ---: | ---: | --- |
| XGBoost / returns: RMSE | 0,0166373 | 0,0164722 | Giảm khoảng 0,99% |
| LSTM / returns: RMSE | 0,0164172 | 0,0163951 | Giảm khoảng 0,13% |
| LSTM / returns: mean seed SD | 0,0001011 | 0,0000706 | Giảm khoảng 30,1%; chỉ phản ánh biến động seed |
| LSTM / full: RMSE | 0,0164958 | 0,0165224 | Tăng khoảng 0,16%; không phải mọi nhóm đều tốt hơn |

Đối chứng trên cùng nhãn/cohort:

| Model / nguồn | Mean-fold RMSE | Số fold MSE skill > Zero Return |
| --- | ---: | ---: |
| Train Mean / Vietcap | **0,0162915** | 2/3 |
| Zero Return / shared | 0,0163497 | Mốc đối chứng |
| Ridge returns / Vietcap | 0,0163539 | 1/3 |
| LSTM returns / Vietcap | 0,0163951 | 1/3 |
| XGBoost returns / Vietcap | 0,0164722 | 1/3 |

LSTM returns/Vietcap vẫn có RMSE cao hơn Zero Return khoảng **0,28%** và cao hơn Train Mean khoảng **0,64%**. Trong nhánh Vietcap, returns-only giảm RMSE khoảng 0,77% và giảm mean seed SD khoảng 23,7% so với full LSTM; đây là hướng đơn giản hơn để nghiên cứu tiếp, chưa phải lý do promote.

Mức giảm seed SD 30,1% ở đối chiếu nguồn là thay đổi tương đối của một độ lệch chuẩn vốn nhỏ, chỉ dựa trên ba seed mỗi fold; **không phải accuracy tăng 30,1%** hoặc bằng chứng ổn định ở mọi thị trường.

## Riêng năm 2025

Không gộp vào trung bình chính vì năm này đã được xem và audit đã xác định nhiều bất đồng nguồn. Năm 2025 không có cải thiện đồng loạt: trung bình seed của 5/6 nhóm model/features Vietcap có RMSE cao hơn nhánh Yahoo; LSTM/full giảm khoảng 0,07%. **Cả sáu trung bình nhóm Vietcap vẫn kém Zero Return**. Không lấy một seed đẹp để thay cho kết quả nhóm; từng seed tốt/xấu đều được giữ trong bảng đối chứng.

Nhãn tham chiếu Vietcap có **248 mẫu, 111 tăng, 129 giảm, 8 trung tính**. Hit rate dùng 240 mẫu có lợi suất khác 0; luôn đoán giảm đạt **53,75%**, balanced accuracy 50%. Cùng các origin này, nhãn Yahoo cũ có 37 mẫu trung tính và chỉ 211 mẫu khác 0. Vì vậy không so trực tiếp hit rate 54,98% của reference v2 với hit rate study mới rồi kết luận model tăng/giảm chất lượng: nhãn, mẫu số directional metric và giao thức train đều đã khác.

Ví dụ Ridge/full/Vietcap có hit rate 53,75% và balanced accuracy khoảng 55,72%, nhưng RMSE vẫn kém Zero Return. Một metric hướng tốt trong một năm chưa đủ chứng minh chất lượng hồi quy hoặc hiệu quả giao dịch.

## Quyết định

1. Giữ **Vietcap làm nguồn nghiên cứu ứng viên**, KBS là nguồn kiểm tra. Chưa gọi là dữ liệu đã chứng nhận bởi sàn hoặc point-in-time; không vá giá từng ngày bằng suy đoán.
2. Giữ model/reference serving v2 để không tự thay gói đang dùng của team. Những model vừa train là research artifacts riêng, chưa được tích hợp vào API/Streamlit. Các metric v2 cũ tiếp tục là kết quả trên nhãn Yahoo, kèm giới hạn nguồn đã phát hiện.
3. Hướng model ưu tiên cho thí nghiệm tiếp: **returns-only**, giữ Ridge/Train Mean/Zero Return làm đối chứng, LSTM làm candidate học sâu. Chưa có bằng chứng cần tăng số layer hoặc train lâu hơn.
4. Nếu tiếp tục tối ưu, khóa protocol riêng để thử **training window theo thời gian** trước; giữ nguồn, target, cohort đánh giá và cấu hình model cố định. Tác động tăng thêm labeled training rows của Vietcap là một phép so sánh riêng, không gộp vào kết luận lượt này.
5. Chỉ kết luận khả năng tổng quát sau khi có forward period chưa dùng để phát triển/chọn model. Không mở 2026 rồi tiếp tục gọi nó là holdout nếu dùng để chọn window hoặc hyperparameter. Lượt này chưa tải hoặc đánh giá giá 2026.

## File bằng chứng

- [REPORT.md](REPORT.md): giao thức, bảng đầy đủ và biểu đồ.
- [protocol.json](protocol.json), [source_snapshot.zip](source_snapshot.zip), [input_snapshot.zip](input_snapshot.zip): cấu hình/source/data đã khóa trước training.
- [paired_seed_comparison.csv](paired_seed_comparison.csv), [paired_fold_comparison.csv](paired_fold_comparison.csv), [primary_summary.csv](primary_summary.csv): so từng seed/fold và tổng hợp.
- [predictions.csv.gz](predictions.csv.gz), [metrics.csv](metrics.csv), [lstm_history.json](lstm_history.json), [manifest.json](manifest.json): forecast, metrics, learning curves dạng số và artifact hashes.
- [verification.json](verification.json), [reload_verification.json](reload_verification.json), [design_checks.json](design_checks.json): kiểm chứng số liệu, model và thiết kế; [design_checks.py](design_checks.py) tái tạo kiểm tra context/source.
- [metric_details_verification.json](metric_details_verification.json): tính lại toàn bộ metric, gồm balanced accuracy, MSE skill và mẫu số directional metrics.
- [native_label_differences.csv](native_label_differences.csv): chênh lệch nhãn và số mẫu tăng/giảm/trung tính của từng fold/phase; [unit_tests.txt](unit_tests.txt), [training_console.log.gz](training_console.log.gz) lưu log kiểm thử và tiến trình training thực tế.

```powershell
python -m src.models.source_comparison_report --study reports/source_comparison/20261029T142732805710Z
python reports/source_comparison/20261029T142732805710Z/design_checks.py
```

Không cần weights để kiểm tra lại metrics/report từ prediction export. Muốn nạp lại model, thêm `--reload-artifacts` và dùng weights local trong `models/research/source-comparison_20261029T142732805710Z/`. Weights không được track trong Git; chạy protocol mới tạo study ID mới và giữ nguyên lịch sử lượt này.
