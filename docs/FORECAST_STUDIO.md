# FPT Forecast Studio — dự báo, mô phỏng và chiến lược model

Cập nhật30/10/2026. Local UI: http://127.0.0.1:8501/ · API: http://127.0.0.1:8000/docs. Snapshot vẫn đến29/10/2026, model/bias giữ nguyên. **0 fit mới**, không download hoặc deploy.

## Giao diện

Sáu trang có mục đích riêng: Tổng quan (nến/volume thật), Dự báo (kỳ ngày/tuần/tháng), Đối chứng (actual và prediction lịch sử trên cùng ngày), Hành trình model (evidence và các lượt không cải thiện), Dữ liệu/phương pháp (schema, nhãn, quarantine), Lộ trình (chốt model và việc tương lai).

Giao diện dark với các panel, biểu đồ dùng cùng màu/đơn vị, điều hướng có nhãn và bố cục responsive. Không dùng ảnh/giá thị trường giả. Frontend gọi HTTP API, không load weights, fit scaler hoặc train model. “Nạp lại dữ liệu API” chỉ xóa cache UI30giây để đọc backend, không gọi nguồn thị trường.

## Kỳ dự báo được định nghĩa ra sao?

| Lựa chọn | Khung tính theo ngày local Việt Nam | Xử lý cuối tuần |
| --- | --- | --- |
| Hôm nay | Ngày hiện tại | Dời sang weekday kế tiếp, báo rõ |
| Ngày mai | Ngày lịch hiện tại+1 | Dời sang weekday kế tiếp, báo rõ |
| 1 tuần | Hôm nay đến hôm nay+6ngày, gồm7ngày lịch | Ngày đích cuối lấy weekday trước/đúng cuối khung |
| 1 tháng | Hôm nay đến trước cùng ngày tháng sau theo DateOffset | Ngày đích cuối lấy weekday trước/đúng cuối khung |

Chuỗi tính toán bắt đầu sau origin thực tế cuối, không nhảy thẳng tới target rồi bỏ các bước cầu nối. Có tối đa23bước weekday; nếu dữ liệu cũ làm vượt giới hạn, API trả422 thay vì rút ngắn âm thầm. Ngày nguồn không được lớn hơn/bằng ngày hiện tại theo hợp đồng inference cũ.

Ví dụ khi gọi30/10/2026 với origin29/10:

- Hôm nay:1bước, ngày dự kiến30/10.
- Ngày mai31/10 là thứBảy:2bước từorigin, ngày dự kiến02/11.
- Tuần:5bước, ngày dự kiến cuối05/11.
- Tháng:21bước, ngày dự kiến cuối27/11.

Đây **không phải lịch giao dịch HOSE được xác minh**, chỉ loại thứBảy/Chủnhật. Ngày nghỉ lễ hoặc sự kiện bất thường có thể làm ngày hiển thị sai. `calendar_verified=false` đi cùng JSON và UI; không khẳng định model đoán đúng giá của một ngày sàn đóng cửa.

## Hôm nay khác bằng chứng forecast tương lai

API phát hành output lúc được gọi. Nếu đã trong/sau phiên thì đó vẫn là ước tính từ snapshot, không được đổi timestamp thành forecast phát trước phiên. Hệ thống không biết outcome mới chỉ vì ngày lịch đã đến, không ghi vào ledger và không tự chấm accuracy. Muốn chứng minh forward cần dự báo phát trước target, lưu input/version/timestamp thật và chờ nhãn.

## Recursive Ridge: thêm tính năng, không thêm model đã train

Model giữ StandardScaler→Ridge(alpha100),6feature,window3năm. Bước đầu khớp `/predict` hiện có. Từ bước2:

1. Dùng21Close cuối, gồm các Close dự báo từ các bước trước.
2. Tính Log_Return, lag1/2/5 và return5/20 đúng thứ tự artifact.
3. Predict return bằng Ridge đã fit; cộng bias cố định nếu chọn `Ridge+bias60`.
4. `Close_next = Close_current × exp(predicted_return)`, rồi tiếp tục.

Không tạo giả Open/High/Low/Volume hoặc forward-fill raw. Không biết future context thật. Vì sai số/context tích lũy, **week/month là kịch bản recursive chưa được kiểm định riêng**. Kết quả h=1 ở backtest không được gán cho horizon21. Nếu cần model tuần/tháng thực thụ, bước kế là protocol so recursive với model directh=5/h=21 trên cùng origin/nhãn và baseline.

### Vùng kịch bản

512đường,seed20261030. Lấy mẫu circular block5 từ60sai số đã biết trong bundle; mỗi đường cập nhật feature theo giá mô phỏng của chính nó. Vẽ phân vị10%và90% từng bước. Ridge raw dùng rawerrors; Ridge+bias dùng correctederrors. Không refit hoặc update calibration. ZeroReturn giữ nguyên giá và không mượn residual band của Ridge.

Vùng này chỉ mô tả biến động khi tái sử dụng sai số lịch sử. Không chứng minh coverage80%, không bao phủ hết bất định tham số/nguồn/regime, không phải khoảng đảm bảo giá thật. Các band từng bước cũng không bảo đảm80% toàn bộ đường thật cùng nằm trong vùng.

## Tương lai nên chốt1hay2model?

**Một champion là đủ nếu challenger chưa thêm giá trị.** Hướng hiện tại: Ridge làm model chính; raw/bias là hai cấu hình của nó, không phải hai model độc lập. ZeroReturn luôn là baseline, không tính là model ML thứhai.

Giữ LSTM reference để đối chứng. Nếu cần model thứhai, LSTM hoặc XGBoost chỉ được chọn sau khi so cùng nguồn/ngày/target, nhiềufold và seed, có kiểm tra baseline và forecast mới phát trước target. Chọn theo độ ổn định/đánhđổi/chi phí; không mặc định mạng sâu hơn là tốt hơn.

Ensemble chỉ xem xét khi hai model có lỗi bổ sung nhau và cải thiện ngoài mẫu; trọng số chọn trênvalidation, không chỉnh trên tập đã chấm để làm đẹp kết quả. Phục vụ tối đa1–2model giúp đơn giản API, giám sát, version và rollback; các artifact nghiên cứu vẫn giữ làm evidence.

## API và kiểm chứng

```json
POST /api/v1/forecast
{"model":"Ridge+bias60","horizon":"month"}
```

Output gồmorigin,khungngàyrequested/effective,steps,path,band,modelversion,trainedhorizon1,multistepvalidatedfalse,calendarverifiedfalse,issuedtimestamp và limitations. Không nhận ngày phát hành do caller tự đặt, không mở horizon tùy ý quá1tháng. Endpoint cũ `/predict` vẫn tương thích.

119unit tests đạt; AppTest kiểm tra6trang/4kỳ/ZeroReturn/lọccohort-tháng/chọnrỗng. [Bằng chứng và ảnh UI](../reports/verification/forecast_studio_20261030/VERIFICATION.md). Đây là kiểm chứng code/contract, chưa phải backtest độ chính xác nhiều bước. Giữ canonical/reference/ledger, không commit/push và chưa deploypublic.
