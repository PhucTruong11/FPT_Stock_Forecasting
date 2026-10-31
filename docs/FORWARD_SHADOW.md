# Ghi forecast thật và theo dõi model FPT

Mục tiêu là thu bằng chứng trên các target chưa biết lúc phát forecast. Đây là công cụ chạy thủ công local, không tự đặt lịch, refit, thay model serving hoặc kết nối API/Streamlit của team.

## Bundle hiện tại

- Study: `reports/recency_bias/20261030T022440928507Z/`.
- Ridge alpha100, 6 return features, StandardScaler chỉ fit train, window3y, target train đến 29/10/2026.
- Bundle: `shadow_bundle.json`; weights kèm SHA256 ở `models/research/recency-bias_20261030T022440928507Z/`. Thư mục weights bị gitignore: phải chuyển đúng file khi đổi máy.
- Ghi đồng thời raw Ridge, Ridge+bias cố định, Zero Return và Train Mean snapshot. Không chọn winner sau khi xem outcome.
- Bias/interval lấy 60 lỗi đã chín gần nhất của luồng monthly mô phỏng; áp dụng sang bản refit cuối tháng là giả định chuyển calibration, chưa có bảo đảm. Bundle và calibration **cố định** cho đến khi chủ động rebuild thành phiên bản mới. Không đồng nghĩa refit theo tháng đã triển khai tự động.

## Quy trình mỗi lần có dữ liệu đóng cửa mới

1. Chuẩn bị CSV **full history Vietcap cùng basis adjusted**, cột `Date,Open,High,Low,Close,Volume`; giữ dữ liệu lịch sử đủ dài cho feature builder. Không nối Yahoo/giá cũ khác basis. Không ghi đè snapshot canonical. Lưu file mới trong `.cache/forward/`, giữ nguồn/request/hash như pipeline audit hiện có. Công cụ dưới không tự download.
2. Chỉ dùng phiên đã đóng cửa. Cùng ngày origin, CLI chặn trước 16:00 giờ Việt Nam theo quy ước bảo thủ của project; đây không phải xác nhận chính thức của sàn rằng source đã final. Kiểm tra nguồn đã hoàn tất cập nhật. Phát sau đóng cửa origin và **trước ngày target** để đủ điều kiện chấm full-session.
3. Chạy `issue` với bundle cố định. Ledger lưu thời gian thật, forecast, phiên bản model, hash/ảnh chụp input. Không sửa ngày phát để lùi về quá khứ. Cùng bộ input/bundle/origin sẽ bị từ chối ghi lặp.
4. Khi đã có target trong CSV cập nhật, chạy `score`. Target là **quan sát positive-volume kế tiếp** có thật trong dữ liệu, không đoán ngày nghỉ. Scorer thận trọng chỉ chấm target trước ngày local đang chạy; chạy ngày hôm sau hoặc muộn hơn. Chạy lặp không tạo outcome lặp.
5. `verify` kiểm tra hash-chain local. Chain giúp phát hiện chỉnh sửa nếu đối chiếu hash đã giữ, không chứng nhận thời gian độc lập hoặc chống người có quyền viết dựng lại cả ledger. Chỉ dùng **một writer** tại một thời điểm.

Lệnh PowerShell từ root project (thay đường dẫn CSV bằng file thật đã cập nhật):

```powershell
$bundle = 'reports/recency_bias/20261030T022440928507Z/shadow_bundle.json'
$ledger = 'reports/forward_shadow/forecasts.jsonl'
$inputCsv = '.cache/forward/vietcap_full_history.csv'
venv/Scripts/python.exe -m src.models.forward_ledger issue --bundle $bundle --input $inputCsv --ledger $ledger
venv/Scripts/python.exe -m src.models.forward_ledger score --input $inputCsv --ledger $ledger
venv/Scripts/python.exe -m src.models.forward_ledger verify --ledger $ledger
```

`input_snapshots/*.csv.gz` được lưu cạnh ledger. Khi source hồi tố adjusted prices, scorer dùng **cả origin và target trên cùng snapshot mới** để tính actual return, lưu `origin_adjustment_ratio` và re-anchor predicted price; không trộn hai thang giá. Giữ forecast return gốc bất biến. Thay đổi source có thể vẫn đổi nhãn hồi tố: cần audit trước khi diễn giải điểm số.

## Đọc kết quả

- Forecast chưa có target: pending, không tính sai số và không gán nhãn giả.
- Outcome `full_session_eligible=true`: ngày phát local < ngày target. Chỉ nhóm này được tổng hợp như bằng chứng dự báo trước cả phiên. Phát cùng/ngày sau target phải ghi riêng, không trộn.
- So raw/bias/baseline trên **cùng ngày** và cùng nhãn. Từ squared error lấy RMSE, từ absolute error lấy MAE; không chỉ nhìn hit rate. Zero Return là abstention nên không gán hit giả. Với actual khác 0 và dự báo 0, tính không trúng khi so mô hình có hướng; không bỏ dự báo 0 khỏi mẫu actual để làm đẹp hit.
- Khoảng danh nghĩa80% dùng residual quantile lịch sử; đọc coverage và độ rộng cùng nhau. Không bảo đảm tương lai, không phải xác suất model đúng.
- Khi có đủ ngày mới để đọc rolling errors và drift, khóa protocol mới, dùng validation để chọn, lưu tất cả lượt train và chỉ cân nhắc thay model nếu tốt hơn baseline/control trên bằng chứng phù hợp. Hiện **không có auto-promotion** hay ngưỡng drift được xác nhận qua dữ liệu forward.

## Trạng thái khởi tạo — 30/10/2026

Một forecast đã phát lúc **09:25 UTC+7**, origin29/10, model train target đến29/10, trạng thái **pending**, **0 outcome** lúc kiểm tra. Target chưa được tự gán là30/10. Nếu nhãn thực tế kế tiếp là30/10 thì forecast này phát cùng ngày target và **không đủ điều kiện full-session**. Nó kiểm tra đường ghi log; chưa phải bằng chứng cải thiện model tương lai.

Rebuild study mới khi cần (tạo ID mới, không ghi đè kết quả cũ):

```powershell
venv/Scripts/python.exe -m src.models.recency_bias --config configs/recency_bias_v1.json
```

Config hiện tại cố định snapshot đến29/10, nên lệnh này chỉ tái lập nghiên cứu đó, **không tự lấy dữ liệu mới hoặc train tới hôm chạy**. Muốn cập nhật source/cutoff cần tạo config và snapshot nghiên cứu mới có audit; không sửa bundle đang được dùng cho các forecast đã phát.
