# Tài liệu Hướng dẫn Demo, Báo cáo và Trả lời Vấn đáp Đồ án Machine Learning (Dự báo Cổ phiếu FPT)

Tài liệu này được biên soạn dành riêng cho nhóm để chuẩn bị thuyết trình, chạy Demo trực tiếp và trả lời các câu hỏi lý thuyết / thực nghiệm của giảng viên theo đúng các yêu cầu chấm điểm.

---

## Phần 1. Kịch bản chạy Demo trực tiếp trước giảng viên (5 - 7 phút)

### 1.1. Lệnh khởi chạy hệ thống trước giờ báo cáo
Mở 2 cửa sổ Terminal tại thư mục gốc của dự án (`Stock_Prediction_Project`):

```powershell
# Terminal 1: Khởi chạy REST API Backend (FastAPI)
.\venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Khởi chạy Giao diện FPT Forecast Studio (Streamlit)
.\venv\Scripts\python.exe -m streamlit run src/app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

- Giao diện chính: `http://127.0.0.1:8501`
- Tài liệu API (Swagger UI để chứng minh kiến trúc tách biệt Frontend - Backend): `http://127.0.0.1:8000/docs`

### 1.2. Trình tự thuyết minh 4 tab trên giao diện Streamlit
1. **Tab 1 — Tổng quan (Market Overview & Data Quality):**
   - Giới thiệu bộ dữ liệu cổ phiếu FPT từ nguồn Vietcap (2.199 phiên giao dịch hợp lệ đến 29/10/2026).
   - Bật/tắt các lớp chỉ báo mô tả (SMA 20/50/200, Bollinger Bands, RSI, MACD, Drawdown, Heatmap lợi suất tháng).
   - **Điểm nhấn nói với thầy:** Các chỉ báo hiển thị ở đây phục vụ phân tích khám phá dữ liệu (EDA), hoàn toàn chỉ tính từ quá khứ, không dùng `forward-fill` vào ngày nghỉ lễ.

2. **Tab 2 — Dự báo (Forecast Studio):**
   - Chọn mô hình mặc định `Ridge · hiệu chỉnh bias` (`Ridge+bias60`) và chuyển đổi qua lại với `Ridge · nguyên bản` và `Zero Return · baseline`.
   - Chọn các kỳ hạn: **Hôm nay**, **Ngày mai**, **1 tuần**, **1 tháng**, và mở tính năng **Chọn ngày đích tùy chỉnh**.
   - Mở mục **"Hiểu cách model dự báo"** ngay dưới biểu đồ để chỉ cho thầy thấy 3 sơ đồ trực quan:
     1. Luồng tính 6 đặc trưng lợi suất từ 21 giá đóng cửa gần nhất $\to$ `StandardScaler` $\to$ `Ridge(alpha=100)` $\to$ quy đổi từ log return ra giá VNĐ.
     2. Vòng lặp đệ quy đa bước (chỉ cộng `bias60` ở bước 1, từ bước 2 dùng Ridge thô để chống trôi dạt).
     3. Vùng bất định P10–P90 được tạo từ 512 kịch bản `Centered Circular Block Bootstrap` (khối 5 phiên).

3. **Tab 3 — Đối chứng (Historical Backtest Comparison):**
   - Trình bày bảng so sánh trực tiếp trên cùng tập ngày đích năm 2026 (202 cặp cả năm và 104 cặp giai đoạn tháng 6–10/2026 có mặt cả **LSTM**, **Ridge+bias60**, **Ridge thô** và **Zero Return**).
   - Lọc theo từng tháng để chỉ rõ: có những tháng thị trường đi ngang hoặc đảo chiều đột ngột thì mô hình kém hơn baseline, nhưng tính chung cả giai đoạn thì `Ridge+bias60` đạt MAE thấp nhất và độ chính xác hướng cân bằng (Balanced Accuracy) tốt nhất.

4. **Tab 4 — Hành trình model (Training Journey):**
   - Chiếu bảng tiến trình các đợt thực nghiệm của nhóm để chứng minh nhóm đã thử nghiệm từ Deep Learning (LSTM), Tree-based (XGBoost), Classification đến Linear Regularized (Ridge) trước khi đưa ra quyết định cuối cùng.

---

## Phần 2. Trả lời câu hỏi trọng tâm: "Cách chọn mô hình, vì sao chọn mô hình đó, cách làm như thế nào?"

### 2.1. Nhóm chọn mô hình nào làm mô hình cuối cùng?
- **Mô hình chính thức phục vụ trên API và Dashboard:** Tổ hợp **`StandardScaler` $\to$ `Ridge Regression (alpha = 100)` huấn luyện trên cửa sổ trượt 3 năm với 6 đặc trưng lợi suất trễ (`Ridge6`), kết hợp hiệu chỉnh độ chệch nhân quả 60 phiên (`Ridge+bias60`)**.
- **Mô hình giữ làm đối chứng bắt buộc:** `Ridge nguyên bản` (chưa cộng bias), `Zero Return Baseline` (giữ nguyên giá hôm nay), và `LSTM (lookback = 30, 16 features)` trong tập đối chứng lịch sử.

### 2.2. Vì sao nhóm chọn `Ridge6 (alpha=100, window=3y) + bias60` thay vì LSTM hay XGBoost? (5 lý do cốt lõi)
1. **Phù hợp với bản chất nhiễu cao và cỡ mẫu nhỏ của dữ liệu chứng khoán theo ngày:**
   - Dữ liệu giao dịch theo ngày (Daily D1) của 1 mã cổ phiếu trong 3 năm chỉ có khoảng 746–750 mẫu huấn luyện (hoặc ~2.150 mẫu nếu lấy từ 2018).
   - Trên tập dữ liệu kích thước này và tỷ lệ tín hiệu/nhiễu (signal-to-noise ratio) rất thấp, các mô hình nhiều tham số như **LSTM** (hàng nghìn trọng số) hoặc **XGBoost** rất dễ học thuộc nhiễu cục bộ (overfitting). Thực tế trên tập đối chứng 104 phiên (tháng 6–10/2026), LSTM chỉ đạt Hit Rate **47,47%** (Balanced Accuracy **50,83%**), trong khi `Ridge+bias60` đạt Hit Rate **62,63%** (Balanced Accuracy **57,95%**).
2. **Sức mạnh của điều chuẩn L2 mạnh ($\alpha = 100$) và thu gọn đặc trưng (Feature Pruning):**
   - Khi thử bộ 16 đặc trưng kỹ thuật (RSI, MACD, Bollinger, SMA...), các đặc trưng có hiện tượng đa cộng tuyến mạnh. Khi thu gọn về **6 đặc trưng lợi suất thuần túy** và phạt chuẩn $L_2$ với $\alpha = 100$, các trọng số được co nhỏ hợp lý, giúp sai số ngoài mẫu (out-of-sample RMSE) ổn định qua nhiều năm.
3. **Cửa sổ huấn luyện 3 năm thích nghi tốt hơn với cấu trúc thị trường hiện tại:**
   - Qua kiểm định Walk-Forward 4 năm (2022–2025), cửa sổ **3 năm** thắng ở 3/4 năm và có trung bình RMSE thấp nhất (`0,017078` so với `0,017095` của 5 năm và `0,017108` của toàn bộ từ 2018), đồng thời có sai số ở năm xấu nhất thấp nhất. Dữ liệu quá cũ (2018–2020) mang đặc tính thanh khoản và biên độ khác với giai đoạn 2024–2026.
4. **Cơ chế hiệu chỉnh độ chệch nhân quả `bias60` giải quyết sự dịch chuyển xu hướng ngắn hạn:**
   - Mô hình Ridge tĩnh có xu hướng dự báo thận trọng quanh kỳ vọng trung bình của 3 năm trước. Khi cổ phiếu FPT bước vào nhịp điều chỉnh giảm trong nửa cuối năm 2026, việc cộng trung bình sai số nhân quả của 60 phiên gần nhất đã có nhãn (`bias60`) giúp kéo MAE giá xuống mức thấp nhất (**1.064,12 VNĐ**, thấp hơn cả Ridge thô 1.080,94 VNĐ và Zero Return 1.081,70 VNĐ) và nâng Balanced Accuracy toàn năm 2026 từ **54,20%** lên **56,62%**.
5. **Khả năng vận hành đệ quy nhiều bước minh bạch và không cần bịa dữ liệu tương lai:**
   - Vì `Ridge6` chỉ phụ thuộc vào chuỗi giá đóng cửa (`Close`), khi người dùng muốn xem mô phỏng 1 tuần ($h=5$) hoặc 1 tháng ($h=22$), hệ thống có thể tự cập nhật chính xác 6 đặc trưng lợi suất từ các giá đóng cửa dự báo trước đó. Ngược lại, nếu dùng LSTM 16 đặc trưng để dự báo sang ngày thứ 2, thứ 3 thì bắt buộc phải tự bịa ra giá `Open`, `High`, `Low`, `Volume` của các ngày tương lai — điều đó sẽ tạo ra sai số chồng chất không kiểm soát được.

### 2.3. Quy trình thực hiện (Cách làm như thế nào từ đầu đến cuối?)
Nhóm thực hiện theo quy trình **6 bước nghiêm ngặt chống rò rỉ dữ liệu (Zero Leakage Pipeline)**:
1. **Bước 1 — Kiểm toán và làm sạch dữ liệu gốc:** Đối chiếu Yahoo Finance, Vietcap và KBS; chọn Vietcap; loại bỏ các phiên `Volume <= 0` hoặc vi phạm hình học nến (`Low <= Open, Close <= High`); tuyệt đối không `forward-fill` ngày nghỉ.
2. **Bước 2 — Tính đặc trưng nhân quả và gán nhãn tự động:** Tại ngày gốc $t$, chỉ dùng giá đến thời điểm đóng cửa ngày $t$ để tính 6 đặc trưng lợi suất trễ. Nhãn $y_t = \ln(\text{Close}_{\text{next}(t)} / \text{Close}_t)$ được ghép từ phiên hợp lệ kế tiếp và lưu kèm cột `Target_Date_1d`.
3. **Bước 3 — Chia tập theo thời gian có chốt chặn ngày đích (Boundary Purging):** Khi cắt tập Train tại mốc $T_{\text{cutoff}}$, mọi mẫu phải thỏa mãn cả $\text{Date}_t \le T_{\text{cutoff}}$ **và** $\text{Target\_Date\_1d} \le T_{\text{cutoff}}$. Mẫu nằm ở ngày cuối năm có nhãn vắt sang năm sau bị loại khỏi tập Train để chống rò rỉ.
4. **Bước 4 — Chuẩn hóa và Huấn luyện trên từng Fold (Walk-Forward Validation):** Bộ chuẩn hóa `StandardScaler` chỉ được `.fit()` trên tập Train, sau đó `.transform()` trên tập Validation/Evaluation. Dùng kết quả trung bình trên các fold Validation để khóa siêu tham số ($\alpha = 100$, cửa sổ 3 năm).
5. **Bước 5 — Tính toán hiệu chỉnh nhân quả và kiểm định đa bước:** Tại mỗi ngày gốc kiểm định $t$, chỉ lấy các phần dư (residuals) có `Target_Date_1d <= t` (tối đa 60 phiên) để tính `bias60` và chạy 512 đường `Centered Circular Block Bootstrap` cho vùng kịch bản P10–P90.
6. **Bước 6 — Đóng gói bất biến (Immutable Bundle) và Kiểm thử phần mềm:** Đóng gói mô hình đã fit đến 29/10/2026 vào `models/local_demo/ridge6_20261029/` kèm mã băm SHA-256 trong `manifest.json`. API khi khởi động sẽ kiểm tra chữ ký SHA-256 trước khi phục vụ; kiểm chứng toàn bộ bằng 149 unit tests.

---

## Phần 3. Thống kê toàn bộ các đợt Training & Thực nghiệm: "Bao nhiêu thử nghiệm là đủ?"

### 3.1. Tổng số lần huấn luyện và sửa đổi của dự án
Để ngắn gọn, dễ báo cáo và thống nhất khi trình bày trước giảng viên, toàn bộ lịch sử nghiên cứu của dự án được tổng hợp thành **8 đợt huấn luyện mô hình chính thức** (gồm hơn 340 lượt fit cấu hình ứng viên), đi kèm các bước kiểm định hồi cứu, sửa lỗi suy luận đệ quy (`v1` $\to$ `v4`) và đóng gói sản phẩm với **149 unit tests**.

### 3.2. Ý nghĩa của 8 đợt Training chính thức (Mỗi đợt trả lời một câu hỏi khoa học riêng)

| Đợt | Tên đợt Training | Quy mô thực nghiệm | Người phụ trách chính | Ý tưởng cốt lõi & Đóng góp vào quyết định cuối cùng |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Chuẩn hóa Pipeline v2 & Baseline/LSTM (25/10)** | 4 runs đối chứng CPU & Colab GPU A100; 8 mô hình (`LSTM` lb 10, 30, 60) | **Trương Trọng Phúc** | Khắc phục triệt để lỗi rò rỉ dữ liệu bản cũ; chọn `LSTM30` (Test RMSE `0,019858`) làm đại diện tốt nhất của nhánh Deep Learning và chứng minh tái lập 100% CPU $\leftrightarrow$ GPU. |
| **2** | **Thực nghiệm độ bền — Robustness Study (29/10)** | **63 candidates** (nhiều nhóm feature, model, seeds, folds) | **Đỗ Đạt** | Kiểm chứng xem bộ 16 chỉ báo kỹ thuật có ổn định qua nhiều năm và nhiều seed không. Phát hiện chúng gây overfit, dẫn tới quyết định rút gọn đặc trưng. |
| **3** | **Huấn luyện đối chiếu 3 nguồn dữ liệu (29/10)** | **112 candidates** so sánh Yahoo, Vietcap, KBS | **Phạm Khánh Nguyên** | Phát hiện Yahoo có 147 dòng nghỉ lễ `Volume = 0` và 1 nến lỗi; chuyển toàn bộ nghiên cứu chính sang nguồn **Vietcap** sạch. |
| **4** | **Khảo sát cửa sổ & Tinh chỉnh lưới Ridge (29–30/10)** | **104 fits mới** (Window 56 fits + Ridge Grid 48 fits; $\alpha \in [0.1..1000]$) | **Chế Văn Nam** | Thu gọn đầu vào về 6 đặc trưng lợi suất (`Ridge6`) và tìm ra cấu hình trụ cột: **`Ridge6, alpha = 100, cửa sổ 3 năm`**. |
| **5** | **Cập nhật độ trễ & Hiệu chỉnh nhân quả `bias60` (30/10)** | **13 fits mới** (Frozen vs Refit 2025 vs Monthly vs `bias60`) + đối chứng 104 phiên | **Chế Văn Nam** | Refit đến 2025 giảm 2,83% RMSE; cộng thêm `bias60` đạt **MAE thấp nhất (1.064,12 VNĐ)**, **Balanced Accuracy 56,62%** (và 62,63% Hit trên 104 phiên vs LSTM). |
| **6** | **Nhánh Phân loại & Từ chối dự báo — Selective (30/10)** | **20 base models + 10 calibrators** (Logistic, XGBoost + Platt Sigmoid) | **Đỗ Đạt** | **Kết quả âm giá trị:** Sau Platt calibration, mô hình bị lệch sang dự đoán giảm 202/202 ngày (Balanced Acc 50%). Chứng minh giữ bài toán **Hồi quy** là đúng. |
| **7** | **Thực nghiệm tách nến Overnight Gap & Body (30/10)** | **20 fits mới (nhánh B)** đối chứng 20 mô hình nhánh A | **Phạm Khánh Nguyên** | **Kết quả âm giá trị:** Tách Gap & Body làm RMSE năm 2026 tăng (`0,020119` $\to$ `0,020408`) và Hit giảm từ 50,76% xuống 44,67%. Giữ nguyên 6 đặc trưng A. |
| **8** | **Walk-Forward chốt cửa sổ cuối & Đóng gói (01/11)** | **16 fits mới / 18 điều kiện** (3y vs 5y vs all2018 trên 993 mẫu 2022–2025) | **Trương Trọng Phúc** | Khẳng định **3 năm** thắng 3/4 năm, có Mean RMSE và Worst-fold RMSE thấp nhất; trọng số khớp 100% SHA-256 với bundle phục vụ `ridge6_20261029`. |

*(Ghi chú thêm: Các bước kiểm định không huấn luyện lại trọng số mới (`0 fit`) gồm: kiểm định mô hình đóng băng trên 2026, sửa lỗi cộng dồn bias đa bước `v2`, căn tâm Bootstrap `v3` nâng coverage $h=5$ từ $74,59\%$ lên $82,79\%$, và thử nghiệm nới band `v4` với 149 unit tests).*

### 3.3. Cách trả lời khi thầy hỏi: "Bao nhiêu thử nghiệm là đủ để chọn ra mô hình cuối cùng?"
> **Câu trả lời mẫu dành cho nhóm:**
> "Thưa thầy, trong học máy ứng dụng cho dữ liệu chuỗi thời gian tài chính, **không có một con số cố định cho mọi bài toán**, mà quá trình thử nghiệm chỉ dừng lại khi thỏa mãn **4 tiêu chí hội tụ khoa học**:
> 1. **Đã quét đủ các họ mô hình đại diện theo nguyên lý Ockham's Razor (từ đơn giản đến phức tạp):** Nhóm đã đi từ Baselines (`Zero Return`, `Train Mean`) $\to$ Tuyến tính (`Linear`, `Ridge`) $\to$ Phi tuyến dạng cây (`XGBoost`) $\to$ Chuỗi học sâu (`LSTM`) và cả nhánh `Phân loại (Classification)`.
> 2. **Đã kiểm soát và loại trừ các nguồn gây nhiễu ngoài mô hình:** Nhóm đã kiểm toán 3 nguồn dữ liệu (Yahoo, Vietcap, KBS), kiểm tra nhiều nhóm đặc trưng (16 chỉ báo kỹ thuật vs 6 đặc trưng lợi suất vs tách nến Gap/Body) và độ dài lịch sử huấn luyện (2 năm, 3 năm, 5 năm, toàn bộ từ 2018).
> 3. **Mỗi giả thuyết cải tiến đều được đối chứng A/B trên cùng tập ngày đích và giữ lại cả kết quả thất bại:** Ví dụ khi thử nhánh Phân loại hay tách nến Gap/Body không mang lại cải thiện ổn định qua các năm, nhóm dừng việc thêm độ phức tạp không cần thiết.
> 4. **Tránh bẫy 'Multiple Testing / Data Snooping' (thử quá nhiều trên cùng một tập Test đến khi tình cờ trúng một mô hình điểm cao giả tạo):** Khi nhận thấy chênh lệch RMSE giữa các biến thể đã thu hẹp về mức phần nghìn (ví dụ cửa sổ 3 năm chênh với 5 năm chỉ 0,10% và khoảng tin cậy 95% chứa 0), việc tiếp tục thử thêm hàng trăm siêu tham số nữa trên dữ liệu lịch sử đã xem sẽ dẫn đến overfitting vào quá khứ. Lúc đó, dừng tinh chỉnh để đóng băng mô hình và chuyển sang theo dõi dự báo thực tế tương lai (Forward Ledger) là điểm dừng đúng đắn nhất."

---

## Phần 4. Trả lời các câu hỏi lý thuyết & thực nghiệm thường gặp của Giảng viên

### Câu 1: Bài toán của nhóm là Hồi quy (Regression) hay Phân loại (Classification)? Có gán nhãn dữ liệu hay không?
- **Trả lời:**
  - Đây là bài toán **Học có giám sát (Supervised Learning)**.
  - Nhánh chính đang phục vụ là **Hồi quy (Regression)** vì đầu ra mục tiêu của mô hình là một số thực liên tục: tỷ suất sinh lời logarit của phiên kế tiếp $y_t = \ln(\text{Close}_{\text{next}(t)} / \text{Close}_t)$, sau đó quy đổi ra mức giá dự báo $\widehat{\text{Close}}_{\text{next}(t)} = \text{Close}_t \cdot \exp(\hat{y}_t)$.
  - Nhóm **có thực hiện thêm nhánh Phân loại (Classification)** trong Đợt huấn luyện 6 (dùng Logistic Regression và XGBoost Classifier dự đoán nhãn Tăng/Giảm kèm cơ chế từ chối dự báo khi tín hiệu yếu), nhưng thực nghiệm cho thấy nhánh phân loại bị lệch về một phía sau bước hiệu chỉnh xác suất nên nhóm chọn Hồi quy làm nhánh chính.
  - **Về gán nhãn dữ liệu:** Dữ liệu **có nhãn đầy đủ**, được gán nhãn tự động từ lịch sử chuỗi thời gian (Temporal Self-Labeling): nhãn của ngày $t$ được tính tự động từ giá đóng cửa của ngày giao dịch hợp lệ kế tiếp $\text{next}(t)$ sau khi ngày $\text{next}(t)$ đã diễn ra. Không cần con người gán nhãn thủ công, và tuyệt đối không để nhãn tương lai lọt vào bước tính đặc trưng tại ngày $t$.

### Câu 2: Tại sao không đưa trực tiếp giá đóng cửa (`Close`) vào làm nhãn dự báo mà phải dùng `Log Return`?
- **Trả lời:**
  1. **Tính dừng (Stationarity):** Giá cổ phiếu FPT năm 2018 ở vùng 15.000 VNĐ nhưng năm 2026 ở vùng 52.000 – 68.000 VNĐ (có xu hướng tăng dài hạn, trung bình và phương sai thay đổi theo thời gian). Nếu train trên giá `Close` giai đoạn 2021–2023 (vùng 30.000 – 45.000 VNĐ), khi mang sang dự báo năm 2026 (vùng 60.000 VNĐ), miền giá trị đầu vào nằm hoàn toàn ngoài khoảng đã học. Ngược lại, `Log Return` dao động ổn định quanh 0 trong biên độ $\pm 7\%$ ở mọi năm.
  2. **Tính cộng tính theo thời gian (Time-Additivity):** Với lợi suất logarit, lợi suất tích lũy sau $h$ phiên chính bằng tổng các lợi suất từng phiên: $\ln(C_{t+h}/C_t) = \sum_{i=1}^h y_{t+i}$, rất thuận lợi và chính xác khi thực hiện mô phỏng đệ quy nhiều bước và cộng nhiễu Block Bootstrap.
  3. **Đánh giá trung thực:** Dự báo giá trực tiếp rất dễ tạo ra $R^2 \approx 0,99$ ảo vì giá ngày mai luôn rất gần giá hôm nay. Dự báo `Log Return` buộc mô hình phải đo đạc trực tiếp với baseline `Zero Return`.

### Câu 3: Data Leakage (Rò rỉ dữ liệu) trong chuỗi thời gian là gì? Nhóm đã phòng chống ở những khâu nào?
- **Trả lời:** Rò rỉ dữ liệu là việc thông tin của tương lai (sau thời điểm phát dự báo $t$) vô tình lọt vào quá trình tạo đặc trưng, chuẩn hóa hoặc huấn luyện mô hình. Nhóm đã chặn ở **5 chốt**:
  1. **Không chia ngẫu nhiên (No Random/Shuffle Split):** Chia hoàn toàn theo thứ tự thời gian.
  2. **Chốt chặn ngày đích (`Target_Date_1d <= cutoff`):** Mẫu ở ngày cuối cùng của tập Train có nhãn rơi sang ngày đầu tiên của tập Validation bị loại bỏ (purged) khỏi tập Train.
  3. **Scaler chỉ fit trên Train:** `StandardScaler` và `MinMaxScaler` chỉ gọi `.fit()` trên tập Train của từng fold, tuyệt đối không fit trên toàn bộ file CSV trước khi chia tập.
  4. **Hiệu chỉnh `bias60` và `Block Bootstrap` nhân quả:** Tại ngày gốc $t$, chỉ những sai số dự báo của các ngày trước đó mà ngày đích `Target_Date_1d <= t` đã thực sự đóng cửa mới được đưa vào pool 60 phiên.
  5. **Không dùng mô hình đã train đến tháng 10/2026 để chấm điểm ngược lại các tháng đầu năm 2026:** Toàn bộ bảng điểm Backtest năm 2026 trên giao diện đều được chấm từ mô hình lịch sử chỉ học đến hết ngày 31/12/2025.

### Câu 4: Tại sao Hit Rate đạt 60,91% trên tập 2026 nhưng nhóm vẫn nhấn mạnh cần đọc kèm Balanced Accuracy (56,62%)?
- **Trả lời:**
  - Trong 202 phiên của năm 2026 (tính đến 29/10/2026), thị trường trong nhịp điều chỉnh nên có **117 phiên giảm**, **80 phiên tăng** và **5 phiên đứng giá**.
  - Nếu một mô hình ngây thơ **luôn luôn đoán giảm** ở mọi ngày, nó cũng tự động đạt Hit Rate là $117 / 197 = \mathbf{59,39\%}$! Nhưng khi đó Recall lớp Tăng bằng 0% và Recall lớp Giảm bằng 100%, nên **Balanced Accuracy** $=\frac{0\% + 100\%}{2} = \mathbf{50,00\%}$ (tương đương đoán mò).
  - Trong khi đó, `Ridge+bias60` không chỉ đạt Hit Rate **60,91%** mà còn đạt **Balanced Accuracy 56,62%** (và trên tập 104 phiên tháng 6–10/2026 đạt Hit Rate **62,63%**, Balanced Accuracy **57,95%**), chứng tỏ mô hình thực sự phân biệt được cả phiên tăng lẫn phiên giảm chứ không phải chỉ đoán một chiều.

### Câu 5: Tại sao trong dự báo đệ quy tuần/tháng (`recursive-ridge6-v4`), nhóm chỉ cộng `bias60` ở bước 1 mà từ bước 2 trở đi lại dùng Ridge thô?
- **Trả lời:**
  - Thành phần `bias60` ($\approx -0,35\%$/ngày ở cuối tháng 10/2026) chỉ được ước lượng và kiểm chứng cho bài toán dự báo 1 bước ($h=1$).
  - Ở phiên bản `v1`, khi cộng dồn hằng số $-0,35\%$/ngày liên tiếp qua 21 bước đệ quy của 1 tháng, độ chệch này bị nhân lên gấp 21 lần, kéo giá dự báo tháng tụt dốc giả tạo tới **$-8,69\%$** (trong khi bản thân mô hình Ridge chỉ dự báo mức giảm nhẹ $\approx -0,08\%$/ngày).
  - Khi sửa sang `v2` (chỉ áp dụng `bias60` cho bước $h=1$, và từ bước $h \ge 2$ để mô hình Ridge tự suy luận trên chuỗi giá dự báo), đường dự báo 1 tháng trở về mức hợp lý **$-1,97\%$**, phản ánh đúng tính chất hội tụ về kỳ vọng trung bình của chuỗi thời gian tài chính.

### Câu 6: Vùng kịch bản P10–P90 trên biểu đồ được tính như thế nào? Tại sao dùng Circular Block Bootstrap có căn tâm (Centered) mà không dùng công thức phân phối chuẩn $\pm 1,28\sigma$?
- **Trả lời:**
  - Lợi suất chứng khoán thực tế có phần đuôi dày (fat tails / leptokurtic, Excess Kurtosis $+2,81$), lệch (skewness) và có cụm biến động (volatility clustering) trong ngắn hạn, nên giả định phân phối chuẩn thường đánh giá sai rủi ro.
  - Nhóm sử dụng **Circular Block Bootstrap** với kích thước khối **5 phiên liên tiếp** trên 60 sai số lịch sử gần nhất và mô phỏng **512 quỹ đạo giá** để giữ được cấu trúc tự tương quan ngắn hạn trong mỗi tuần giao dịch.
  - Việc **trừ trung bình của pool sai số (Centering ở bản `v3`)** trước khi lấy mẫu giúp tách biệt rõ phần dự báo điểm trung tâm và phần nhiễu ngẫu nhiên xung quanh đường điểm. Nhờ đó, trên tập kiểm định 122 ngày gốc năm 2026, độ phủ thực nghiệm (Coverage) của khoảng P10–P90 ở kỳ hạn $h=5$ tăng từ **74,59% lên 82,79%** (rất sát mức mục tiêu 80%) và giảm đáng kể điểm phạt Interval Score.

---

## Phần 5. Phân công nhiệm vụ đều 25% theo 8 đợt Training & Mảng kỹ thuật (Để thống nhất khi thầy hỏi từng thành viên)

| Thành viên (MSSV) | 2 đợt Training phụ trách (trong 8 đợt) | Mảng kỹ thuật & Báo cáo đảm nhận | Phần vấn đáp trọng tâm | Tỷ lệ |
| :--- | :--- | :--- | :--- | :---: |
| **Trương Trọng Phúc** (`3124410287`) | **Đợt 1** (Pipeline v2 CPU/Colab) & **Đợt 8** (Walk-Forward chốt cửa sổ 3y/5y/2018) | **Setup hệ thống, tiến hành train**, cài đặt suy luận đệ quy (`v2/v3/v4`), Block Bootstrap; viết Chương 1 & 5. | Kiến trúc pipeline chống leakage, tái lập CPU/GPU, dự báo đa bước (`bias60` chỉ bước 1) & Bootstrap P10–P90. | **25%** |
| **Đỗ Đạt** (`3124410054`) | **Đợt 2** (Robustness 63 cấu hình) & **Đợt 6** (Phân loại chọn lọc Logistic/XGBoost) | **Xây dựng REST API (`FastAPI`)**, kiểm tra schema Pydantic, endpoints & kiểm thử API; viết Chương 3. | Luồng `/predict`, `/forecast`, `/backtest`, lý do nhánh Phân loại bị lệch và lý do `LSTM` thua `Ridge` ở 104 phiên. | **25%** |
| **Phạm Khánh Nguyên** (`3124410239`) | **Đợt 3** (Đối chiếu 3 nguồn 112 cấu hình) & **Đợt 7** (Tách nến Gap/Body A/B) | **Xây dựng Giao diện (`Streamlit Studio`)** (4 trang + 3 sơ đồ giải thích mô hình); viết đầu Chương 4. | Demo 4 tab giao diện, giải thích chọn Vietcap thay Yahoo (`147` dòng `Vol=0`) và vì sao bỏ tách nến Gap/Body. | **25%** |
| **Chế Văn Nam** (`3124410217`) | **Đợt 4** (Cửa sổ & Grid $\alpha$ Ridge) & **Đợt 5** (Recency Refit & `bias60`) | **Làm sạch & Feature Engineering** (16 chỉ báo v2 $\to$ 6 return `Ridge6` chống leakage); viết Chương 2 & chọn model Chương 4. | Giải thích 6 feature lợi suất, chọn `Ridge6 (alpha=100, window 3y) + bias60` và Balanced Accuracy vs bẫy Always-Down $59,39\%$. | **25%** |
