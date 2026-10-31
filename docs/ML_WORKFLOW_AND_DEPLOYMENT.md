# Quy trình ML và kế hoạch triển khai

Cập nhật 30/10/2026. Tài liệu thiết kế hiện hành: [ARCHITECTURE.md](ARCHITECTURE.md).

Đã có pipeline offline, bundle Ridge6, FastAPI JSON và Streamlit gọi API **local**. Bundle đóng gói từ artifact sẵn có, không fit trong request. Reference LSTM cũ được giữ cho đối chứng; hai pointer experiment lịch sử không thay.

- [Khái niệm supervised regression, input, nhãn, leakage, walk-forward](ARCHITECTURE.md)
- [Nhật ký training có ảnh và kết quả đối chứng](TRAINING_HISTORY.md)
- [Chạy API/dashboard local](../README.md)
- [Ghi và chấm forecast shadow](FORWARD_SHADOW.md)

## Kế hoạch tương lai — chưa triển khai

1. Thu dự báo phát trước target và chấm khi có nhãn; theo dõi RMSE/MAE, balanced/coverage và baseline cùng ngày.
2. Kiểm tra chất lượng nguồn trước khi quy sai số tăng cho drift. Retrain offline theo giao thức khóa trước, giữ replay và rollback.
3. Đặt acceptance gate và đủ mẫu mới; chạy candidate shadow trước khi thay phiên bản. Không bật auto-promotion từ các năm đã xem.
4. Đo RAM/latency/dependencies của API nhẹ, rồi mới chọn host và đánh giá chi phí/gói miễn phí theo điều kiện lúc triển khai.
5. Khi được yêu cầu deploy mới chuẩn bị Docker/HTTPS/secrets và URL public; hiện chỉ bind127.0.0.1.

Không có deploy public, lịch tự tải data, cron retrain hoặc hóa đơn cloud được tạo trong lượt này. API/UI local không tự ghi forecast vào ledger.
