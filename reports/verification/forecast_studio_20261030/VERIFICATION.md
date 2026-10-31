# FPT Forecast Studio — kiểm chứng local30/10/2026

Đã mở rộng inference và làm lại dashboard6trang. **0 fit mới**; không có kết quả accuracy tuần/tháng mới. Weights/scaler/bias giữ nguyên, không download giá, commit/push hoặc deploypublic.

## Kiểm tra đã chạy

- **119 unit tests đạt**, gồm featureparity với buildercanonical, bước1 khớp predictcũ, recursion bước2 từClose dự báo, ngày cuối tuần/khungtuầntháng, limit23bước, staleinput, trạngtháifrozen/khôngfit, bootstrapseed và HTTPschema. Log`.cache/forecast-studio-tests.txt`; hash trong[checks.json](checks.json).
- AppTest:6trang,4kỳ,ZeroReturn,đổi cohort/tháng10 và bỏhếtđườngmodel khôngexception/error. [ui_checks.json](ui_checks.json). Tái lập với `python -m src.app.check_studio` khi APIlocal đang chạy và snapshot còn phù hợp.
- 12 tổ hợp model/kỳ đã gọi qua HTTPthật. [Forecasts có metadata/version](http_forecasts.json) lưu4kỳ mặc định; hômnay1bước,ngàymai2,tuần5,tháng21 cho ngày30/10 và origin29/10.
- Browser đã quan sát chart/kỳtháng cập nhật, cảnhbáo nhiều bước và UIhẹp. [Ảnh Studio](studio.png), [đường mô phỏng tháng](forecast_month.png), [khung600px](narrow.png), [đo layout](narrow_layout.json). Kích thướcviewport thử được reset, tabQA riêng được đóng; giữ trang người dùng đang xem.
- Hash8file canonical/pointer và10artifact reference khôngđổi. Ledger vẫn1eventpending, không tự ghi forecastUI.

## Giới hạn còn nguyên

Chỉh=1 đã train. Khung tuần/tháng là recursive scenario chưa backtest riêng; vùngP10–P90 là residualblockbootstrap mô tả, không bảo đảmcoverage. Chỉ bỏ cuối tuần, chưa xácminh lịchHOSE. Snapshot đến29/10/2026, không phải realtime. Estimatehômnay không có nghĩa đã phát trước phiên. Endpointpredictcũ vẫn giữ.

Ridge raw và Ridge+bias là hai cấu hình cùng model. Tươnglai giữ Ridge chính và chỉ thêm LSTM/XGBoost nếu có lợi ích kiểm chứng; ZeroReturn làbaseline. Xem[phươngpháp/roadmap](../../../docs/FORECAST_STUDIO.md),[README](../../../README.md).
