# Hướng dẫn nhóm trình bày đồ án dự báo FPT

> **Câu mở đầu:** Nhóm dùng giá lịch sử của FPT để dự báo lợi suất phiên kế tiếp, quy đổi kết quả ra giá đóng cửa và trình bày trên dashboard. Nhóm so sánh từ baseline đến mô hình học sâu rồi chọn Ridge để phục vụ demo.

Đọc mục 1–5 để nắm mạch trình bày. Mỗi bạn xem thêm phần mang tên mình ở mục 7 và tập trả lời bằng ví dụ từ chính phần mình làm.

## Tech stack

| Phần | Công cụ | Dùng trong đồ án |
| --- | --- | --- |
| Thu thập dữ liệu | Python, Requests, yfinance | Gọi API Vietcap lấy OHLCV ngày của FPT. Dùng yfinance lấy bộ Yahoo Finance để đối chiếu và dùng KBS để kiểm tra chéo. |
| Làm sạch và tạo đặc trưng | pandas, NumPy, `ta` | Kiểm tra nến, tính lợi suất và ghép nhãn. Thư viện `ta` tính một số chỉ báo trong bộ đặc trưng mở rộng đã đem thử. |
| Huấn luyện và lưu mô hình | scikit-learn, XGBoost, TensorFlow/Keras, Google Colab, Joblib | scikit-learn huấn luyện Ridge trên CPU. XGBoost và LSTM là các phương án đối chứng, trong đó LSTM có chạy trên Colab GPU. Joblib lưu pipeline Ridge đã fit. |
| REST API | FastAPI, Pydantic, Uvicorn | FastAPI định nghĩa endpoint, Pydantic kiểm tra request, Uvicorn chạy ứng dụng. API nạp gói model đã chọn và trả JSON. |
| Deploy backend | GitHub, Render | GitHub lưu mã và gói demo. Render cài thư viện theo `requirements-local.txt` và chạy FastAPI. |
| Giao diện | Streamlit, Plotly, Requests | Streamlit dựng FPT Forecast Studio, Requests gọi API, Plotly vẽ nến lịch sử, đường dự báo và biểu đồ đối chứng. |

## Quy trình từ lấy dữ liệu đến giao diện cuối

~~~mermaid
flowchart TD
    A["1. Lấy OHLCV ngày của FPT"] --> B["2. Kiểm tra và làm sạch"]
    B --> C["3. Tạo đặc trưng X và nhãn y"]
    C --> D["4. Chia theo thời gian và huấn luyện"]
    D --> E["5. So baseline, chọn và đóng gói Ridge"]
    E --> F["6. Xây REST API bằng FastAPI"]
    F --> G["7. Deploy API lên Render"]
    G -->|HTTP và JSON| H["8. Hiển thị trên Streamlit"]
~~~

1. **Lấy dữ liệu.** Thu thập ngày, giá mở cửa, cao nhất, thấp nhất, đóng cửa và khối lượng của FPT theo phiên ngày. Lưu một bản chụp dữ liệu để lần sau chạy cùng đầu vào. Yahoo được tải qua `download_dataset.py`. Vietcap là nguồn chọn sau bước đối chiếu, KBS hỗ trợ kiểm tra chéo. [Tải Yahoo Finance](../../src/data/download_dataset.py) · [Thu thập và kiểm tra các nguồn](../../src/data/source_audit.py)
2. **Làm sạch.** Chuẩn hóa ngày và cột OHLCV, loại dòng có khối lượng bằng 0, cách ly nến lỗi và giữ thứ tự giao dịch thực tế. Đầu ra gồm các phiên hợp lệ để tính đặc trưng và danh sách dòng bị loại để kiểm tra lại. [Kiểm tra OHLCV](../../src/data/contracts.py) · [Lọc dữ liệu](../../src/features/build_features.py)
3. **Tạo đặc trưng và nhãn.** Từ dữ liệu đến hết phiên `t`, tính đặc trưng `X_t`. Ridge cuối dùng 6 đặc trưng lợi suất từ 21 giá đóng cửa gần nhất. Gắn `Target_Return_1d` là log return của phiên kế tiếp và lưu `Target_Date_1d` để chia tập đúng mốc. Phiên hợp lệ mới nhất vẫn có thể dùng để dự báo khi nhãn chưa xuất hiện. [Tạo mẫu huấn luyện](../../src/features/build_features.py)
4. **Huấn luyện.** Chia Train, Validation và Test theo thời gian. Fit scaler và mô hình trên Train. So baseline, Ridge, XGBoost và LSTM trong các thí nghiệm tương ứng. Đầu ra của mỗi lượt gồm mô hình đã fit và kết quả dự báo để so sánh trên cùng ngày đích. [Huấn luyện đối chứng](../../src/models/experiment.py)
5. **Chọn và đóng gói model.** Dựa trên RMSE ở nhiều giai đoạn, đối chiếu thêm MAE, khả năng đoán hướng và kết quả baseline. Nhóm giữ `StandardScaler → Ridge(alpha=100)` với cửa sổ 3 năm và tùy chọn hiệu chỉnh `bias60`. Gói `ridge6_20261029` chứa model, cấu hình hiệu chỉnh, dữ liệu đóng băng và manifest để API kiểm tra khi nạp. Xem tám đợt ở mục 4 để giải thích từng quyết định. [So sánh cửa sổ](../../src/models/final_window_comparison.py) · [Manifest của gói phục vụ](../../models/local_demo/ridge6_20261029/manifest.json)
6. **Xây API.** Khi khởi động, FastAPI nạp gói Ridge. `POST /api/v1/predict` trả lợi suất và giá đóng cửa dự báo cho một phiên. `POST /api/v1/forecast` trả đường giá qua nhiều phiên và vùng kịch bản P10–P90. Mỗi request chạy suy luận trên model đã nạp. [Mã API](../../src/api/main.py)
7. **Deploy API.** Theo cấu hình trong README, Render lấy mã từ GitHub, cài bằng `pip install -r requirements-local.txt` và khởi động bằng `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`. Trước khi demo, gọi `/health` để xem `status`, `model_version` và `data_as_of`. Mở `/docs`, gửi `{}` vào `POST /api/v1/predict` để thử dự báo một phiên từ snapshot. [Cấu hình Render](../../README.md#kiểm-thử-api-đã-deploy-render-cloud) · [Health](https://fpt-stock-forecasting.onrender.com/health) · [Swagger](https://fpt-stock-forecasting.onrender.com/docs)
8. **Kết nối giao diện cuối.** Streamlit đọc `FPT_API_URL`, gửi lựa chọn của người dùng đến API và vẽ kết quả JSON bằng Plotly. Nhóm có thể mở UI trên máy để gọi backend Render. Bốn trang là **Tổng quan**, **Dự báo**, **Đối chứng** và **Hành trình model**. Nến thể hiện OHLC lịch sử, còn đường dự báo và vùng tô thể hiện giá đóng cửa mô phỏng. [Mã giao diện](../../src/app/streamlit_app.py) · [Cách nối UI với API Render](../../README.md#cách-3-kết-nối-giao-diện-streamlit-fpt-forecast-studio-thẳng-lên-api-render)

## 1. Mô tả bài toán

- **Đầu vào** là các đặc trưng từ giá FPT đã quan sát đến hết phiên `t`, ví dụ lợi suất phiên vừa đóng cửa và lợi suất tích lũy 5 hoặc 20 phiên.
- **Nhãn** là log return của phiên kế tiếp, tính bằng `y_t = ln(Close_kế_tiếp / Close_t)`. Ví dụ giá đóng cửa từ 50.000 lên 51.000 đồng, nhãn là `ln(51.000 / 50.000) ≈ 0,0198`, tương ứng mức tăng giá 2%.
- **Đầu ra** là một số liên tục. Nhóm tính `giá dự báo = giá đóng cửa tại t × exp(lợi suất dự báo)`. Nhánh chính là **hồi quy có giám sát** với nhãn tính tự động từ lịch sử. Phiên mới nhất dùng được để dự báo dù nhãn của nó còn chưa biết.
- **Một phiên** là một ngày giao dịch D1, tổng hợp phiên sáng và chiều. Đợt 6 từng thử nhánh **phân loại** bằng cách đổi nhãn thành tăng/giảm.

## 2. Xác định và chuẩn bị dữ liệu

1. Nhóm kiểm tra Yahoo Finance, Vietcap và KBS, rồi chọn **Vietcap** cho pipeline chính. Trong đợt audit, Yahoo có 147 dòng khối lượng bằng 0 và một nến lỗi. KBS giúp kiểm tra chéo dữ liệu.
2. Bỏ dòng không có giao dịch và cách ly nến sai. Giữ ngày có trong dữ liệu, không điền giá giả cho ngày nghỉ. Nếu nhãn trỏ vào nến lỗi, bỏ mẫu huấn luyện tương ứng.
3. Ridge cuối dùng **6 đặc trưng**: lợi suất hiện tại, lợi suất trễ 1/2/5 phiên và lợi suất tích lũy 5/20 phiên. Tại ngày gốc `t`, tính các đặc trưng từ 21 giá đóng cửa gần nhất đã quan sát.
4. Chia theo thời gian. **Train** để học trọng số, **Validation** để chọn cấu hình, **Test** hoặc một giai đoạn đánh giá sau để xem kết quả lịch sử. Khi cắt Train ở cuối năm, loại mẫu có ngày đích sang năm sau. Fit `StandardScaler` trên Train rồi dùng bộ đã fit cho các tập sau.

## 3. Các mô hình đã thử

| Phương án | Giải thích trong một câu | Vai trò |
| --- | --- | --- |
| Zero Return, Train Mean | Giữ nguyên giá hoặc dùng lợi suất trung bình đã biết | Đường cơ sở để so các model |
| Ridge, dùng scikit-learn | Cộng các đặc trưng theo trọng số, dùng `alpha` để hạn chế trọng số quá lớn | Mô hình phục vụ dashboard |
| XGBoost, dùng `XGBRegressor` | Nhiều cây quyết định nhỏ cùng dự báo. Cây mới sửa một phần sai số trước đó | Đối chứng phi tuyến |
| LSTM, dùng TensorFlow/Keras | Đọc chuỗi phiên theo thứ tự. Nhóm thử nhìn lại 10, 30 và 60 phiên | Đối chứng học sâu, có chạy trên Google Colab GPU |
| Logistic Regression và XGBoost Classifier | Học xác suất tăng từ nhãn hướng | Nhánh thử nghiệm. Sau hiệu chỉnh xác suất, dự báo lệch về phía giảm |

**Khi được hỏi train bằng gì**, Ridge là pipeline Python `StandardScaler → Ridge(alpha=100)` của scikit-learn, gọi `.fit(X_train, y_train)` trên CPU. Nhánh học sâu trong đồ án dùng LSTM.

## 4. Tám đợt thực nghiệm và cách chốt mô hình

Một **đợt thực nghiệm** trả lời một câu hỏi và có thể gồm nhiều lượt fit. Khi trình bày số liệu, nói rõ cùng **tập ngày đích, kỳ hạn và chỉ số** nào. RMSE trong mục này đo log return một phiên, trừ chỗ ghi rõ đơn vị khác.

### Bảng tổng quan 8 đợt

| Đợt | Phụ trách | Thử gì? | Kết quả và quyết định |
| --- | --- | --- | --- |
| 1 | Phúc | Chuẩn hóa pipeline, so baseline và LSTM trên CPU/Colab | LSTM nhìn lại 30 phiên tốt nhất trong nhóm LSTM. Train Mean tốt nhất trên validation, nên giữ cả hai để đối chứng. |
| 2 | Đạt | So model và bộ đặc trưng qua 63 cấu hình | Bộ 6 lợi suất cho RMSE thấp hơn bộ 16 cột với Ridge và LSTM. Tiếp tục nghiên cứu bộ gọn. |
| 3 | Nguyên | Kiểm tra ba nguồn và đối chứng Yahoo/Vietcap | Vietcap ít dòng lỗi hơn Yahoo, Ridge cũng nhỉnh nhẹ trên các cặp chung. Chọn Vietcap. |
| 4 | Nam | So cửa sổ train và điều chỉnh `alpha` của Ridge | Cấu hình Ridge 6 đặc trưng, `alpha=100`, cửa sổ 3 năm được chọn ở lượt validation gần nhất của đợt. |
| 5 | Nam | Fit lại Ridge và thử bias60 trên 202 cặp năm 2026 | Fit lại giảm RMSE 2,83% so với Ridge cũ. Thêm bias60 giảm MAE giá, RMSE vẫn sát Zero Return. |
| 6 | Đạt | Phân loại tăng/giảm và thử bỏ qua tín hiệu yếu | Bản đã hiệu chỉnh đoán giảm cả 202 ngày, Balanced Accuracy chỉ 50%. Giữ hồi quy. |
| 7 | Nguyên | Thử tách lợi suất thành Gap và Body | Trên 2026, RMSE tăng và hit giảm. Giữ 6 đặc trưng ban đầu. |
| 8 | Phúc | So 3 năm, 5 năm và toàn bộ lịch sử từ 2018 | Cửa sổ 3 năm có RMSE trung bình thấp nhất trên 993 cặp validation. Chốt 3 năm, với chênh lệch nhỏ. |

### Đợt 1. Chuẩn hóa pipeline và tìm mốc so sánh — Phúc

- **Thử gì:** Sửa cách tạo nhãn, chia tập và fit scaler. Chạy lại trên CPU và Colab với cùng snapshot/cấu hình, so baseline, Linear Regression, Ridge, XGBoost và LSTM nhìn lại 10/30/60 phiên.
- **Kết quả:** LSTM nhìn lại **30 phiên** được chọn trong họ LSTM, RMSE validation **0,014033** và test 2025 **0,019858**. Kết quả CPU/Colab gần trùng ở mức làm tròn. Trên validation, baseline **Train Mean đạt 0,013981**, thấp hơn LSTM30. Trên test 2025, LSTM30 chỉ nhỉnh nhẹ so với Zero Return (**0,019917**).
- **Quyết định:** Giữ LSTM30 làm đại diện học sâu và dùng baseline xuyên suốt. Kết quả của một tập chưa đủ để chọn LSTM làm model cuối. [Bảng điểm lần chạy tham chiếu](../../reports/experiments/20261025T124951351761Z/metrics.csv)

### Đợt 2. Kiểm tra độ bền khi đổi năm và đặc trưng — Đạt

- **Thử gì:** Chạy **63 cấu hình**, chấm cuốn chiếu qua 2022–2024. Với cùng họ mô hình, thử 6 đặc trưng lợi suất, bộ thêm chỉ báo kỹ thuật và bộ đầy đủ 16 cột. Kiểm tra độ dao động khi đổi seed ở XGBoost và LSTM.
- **Kết quả:** RMSE trung bình của Ridge giảm từ **0,016449** với 16 cột xuống **0,016403** với 6 cột. LSTM cũng giảm từ **0,016542** xuống **0,016432** khi dùng bộ gọn. Hai cấu hình này vẫn cao hơn Zero Return (**0,016390**) và Train Mean (**0,016322**) trong cùng đối chứng.
- **Quyết định:** Thử sâu bộ 6 cột vì bớt đầu vào mà kết quả trong lượt này vẫn tốt hơn bộ đầy đủ. Tiếp tục giữ baseline vì các mô hình chưa thắng đều qua các năm. [RMSE theo nhóm đặc trưng](../../reports/robustness/20261029T131205702813Z/fold_summary.csv)

### Đợt 3. Kiểm tra chất lượng nguồn giá — Nguyên

- **Thử gì:** Kiểm tra Yahoo, Vietcap và KBS đến hết 2025, rồi chạy **112 điều kiện** đối chứng mô hình giữa Yahoo và Vietcap. KBS dùng để kiểm tra chéo dữ liệu.
- **Kết quả:** Yahoo có **147 dòng khối lượng bằng 0, 4 dòng thiếu Close và 1 nến sai**. Vietcap có **2 dòng khối lượng bằng 0**, đủ Close và không có nến sai trong snapshot audit. Ridge dùng 6 lợi suất đạt RMSE **0,016354** với Vietcap và **0,016372** với Yahoo trên phần đối chứng chung. Zero Return đạt **0,016350**.
- **Quyết định:** Chọn Vietcap vì dữ liệu kiểm tra được sạch hơn và RMSE Ridge nhỉnh nhẹ trên các cặp đã ghép. Khoảng cách so với baseline còn nhỏ. [Kiểm toán dữ liệu](../../reports/data_audits/20261029T135735663765Z/summary.json) · [Đối chứng hai nguồn](../../reports/source_comparison/20261029T142732805710Z/primary_summary.csv)

### Đợt 4. Chọn bộ 6 đặc trưng và tinh chỉnh Ridge — Nam

- **Thử gì:** Trên Vietcap, thử cửa sổ mở rộng, 2/3/5 năm rồi tinh chỉnh Ridge với `alpha` từ 0,1 đến 1.000. Hai lượt có **56 + 48 fit mới**. `alpha` càng lớn thì mức phạt lên trọng số càng mạnh.
- **Kết quả:** Trên ba năm chấm chính 2022–2024, chiến lược chỉ chọn cửa sổ đạt RMSE **0,016327**. Chiến lược chọn cả cửa sổ lẫn `alpha` đạt **0,016298**, giảm khoảng **0,18%**. Cùng bảng, Zero Return đạt **0,016350** và Train Mean cửa sổ mở rộng đạt **0,016291**. Ở lượt dùng năm 2025 để kiểm tra thêm, validation chọn cấu hình **3 năm, `alpha=100`**.
- **Quyết định:** Lấy Ridge với 6 lợi suất và `alpha=100` làm ứng viên chính. Cửa sổ 3 năm được đưa vào lượt so sánh khóa cấu hình ở đợt 8. Mức cải thiện ở đợt này chỉ nhỏ và `alpha` được chọn khác nhau giữa các đợt chia theo năm. [Bảng kết quả tinh chỉnh](../../reports/ridge_refinement/20261030T011109863274Z/primary_summary.csv) · [Cấu hình chọn theo năm](../../reports/ridge_refinement/20261030T011109863274Z/validation_selection.csv)

### Đợt 5. Kiểm tra fit lại và hiệu chỉnh bias60 — Nam

- **Thử gì:** **13 fit mới** trong nghiên cứu recency. So Ridge đóng băng từ 2023, Ridge fit lại với nhãn đến hết 2025, Ridge fit lại kèm bias60 và baseline trên cùng **202 cặp năm 2026**. Khi replay, tính bias chỉ từ các sai số đã biết ở thời điểm dự báo.

| Phương án lịch sử | RMSE log return ↓ | MAE giá điều chỉnh ↓ | Balanced Accuracy |
| --- | ---: | ---: | ---: |
| Ridge đóng băng từ 2023 | 0,0207058 | 1.114,18 đồng | 49,96% |
| Ridge fit lại tới hết 2025 | 0,0201191 | 1.080,94 đồng | 54,20% |
| Ridge fit lại + bias60 | 0,0201116 | 1.064,12 đồng | 56,62% |
| Zero Return | 0,0201157 | 1.081,70 đồng | — |

- **Kết quả:** Fit lại giảm RMSE **2,83% so với Ridge cũ**. Thêm bias60 giảm MAE giá và cải thiện chỉ số hướng trong tập này. So với Zero Return, hai RMSE gần bằng nhau.
- **Quyết định:** Giữ cấu hình Ridge có bias60 cho demo và giữ Zero Return làm mốc. Các số trên là của model lịch sử, cần báo đúng phạm vi khi thầy hỏi về độ chính xác. [Kết quả trên 202 cặp](../../TRAINING_HISTORY.md#3-đối-chứng-recency-và-bias-trên-cùng202-cặp2026)

### Đợt 6. Thử phân loại và cho phép bỏ qua tín hiệu yếu — Đạt

- **Thử gì:** Dùng Logistic Regression và XGBoost Classifier cho nhãn hướng, gồm **20 lượt fit mô hình cơ sở và 10 bộ hiệu chỉnh xác suất**. Thử thêm ngưỡng cho phép model bỏ qua ngày thiếu tự tin.
- **Kết quả:** Trên ba năm chấm chính 2022–2024, Balanced Accuracy trung bình của Ridge đối chứng là **53,36%**, Logistic thô **51,83%** và XGBoost phân loại thô **49,82%**. Trên 202 ngày năm 2026, hai bản đã hiệu chỉnh xác suất đều đoán giảm. Hit là **59,39%** do 117/197 phiên có biến động là phiên giảm, nhưng Balanced Accuracy chỉ **50%**.
- **Quyết định:** Giữ hồi quy. Nó cung cấp cả mức tăng/giảm và giá dự báo; nhánh phân loại chưa cho thấy khả năng phân biệt hai hướng tốt hơn đủ ổn định. [Điểm ba năm](../../reports/direction_selective/20261030T090621177786Z/primary_summary.csv) · [Điểm từng tập](../../reports/direction_selective/20261030T090621177786Z/metrics.csv)

### Đợt 7. Thử tách lợi suất ngày thành Gap và Body — Nguyên

- **Thử gì:** Nhánh A dùng 6 đặc trưng ban đầu. Nhánh B thay một cột lợi suất hiện tại bằng hai phần: **Gap** từ giá đóng cửa phiên trước đến mở cửa hôm nay và **Body** từ mở cửa đến đóng cửa hôm nay. Chạy **20 fit mới cho B** và dùng lại **20 model A** để so.
- **Kết quả:** Với Ridge dùng lịch sử gần, RMSE trung bình 2022–2024 tăng từ **0,016219** lên **0,016293**. Trên 2026, RMSE tăng từ **0,020119** lên **0,020408**, hit giảm từ **50,76%** xuống **44,67%**. B có ích trong vài năm riêng lẻ.
- **Quyết định:** Giữ A vì B tăng số đầu vào mà lợi ích không ổn định. Với bộ 6 lợi suất, mô phỏng nhiều bước chỉ cần cập nhật chuỗi Close. [Bảng kết quả Gap/Body](../../reports/candle_ablation/20261030T093521684942Z/metrics.csv)

### Đợt 8. Chốt cửa sổ huấn luyện — Phúc

- **Thử gì:** Giữ nguyên Vietcap, 6 đặc trưng, Ridge `alpha=100` và chỉ thay độ dài lịch sử. So **3 năm, 5 năm và toàn bộ từ 2018** trên cùng **993 cặp** của bốn năm validation 2022–2025. Có **18 điều kiện, 16 fit mới** vì hai điều kiện đủ giống nhau để dùng chung kết quả fit.

| Cửa sổ | RMSE trung bình 4 năm ↓ | RMSE ở năm tệ nhất ↓ |
| --- | ---: | ---: |
| 3 năm | **0,01707812** | **0,02104458** |
| 5 năm | 0,01709506 | 0,02111200 |
| Toàn bộ từ 2018 | 0,01710787 | 0,02111200 |

- **Kết quả:** Cửa sổ 3 năm tốt nhất ở **3/4 năm**. Năm 2023, hai cửa sổ dài hơn có sai số thấp hơn. RMSE trung bình 3 năm giảm **0,099%** so với 5 năm và **0,174%** so với toàn bộ lịch sử. Khoảng tin cậy 95% của chênh lệch MSE đều chứa 0.
- **Quyết định:** Chọn 3 năm theo quy tắc RMSE trung bình đã đặt trước. Bản refit 3 năm tới 29/10/2026 có SHA-256 trùng với model đang chạy, nên dùng lại bundle đã đóng gói. [RMSE từng năm](../../reports/final_window_comparison/20261101T014607720130Z/validation_summary.csv) · [Khoảng tin cậy](../../reports/final_window_comparison/20261101T014607720130Z/paired_validation_ci.csv)

### Tóm lại vì sao giữ Ridge6 + bias60

1. **Dữ liệu và đặc trưng:** Đợt 2 và 3 cho lý do dùng Vietcap và thử bộ gọn. Đợt 7 kiểm tra xem có nên mở rộng đầu vào bằng Gap/Body, rồi giữ 6 lợi suất vì kết quả ổn định hơn trong đối chứng.
2. **Họ mô hình và tham số:** Đợt 1 cho thấy LSTM30 cạnh tranh được ở test 2025 nhưng Train Mean tốt hơn trên validation. Ở đối chứng **104 phiên** tháng 6–10/2026, Ridge fit tới 2025 + bias60 có Balanced Accuracy **57,95%**, so với **50,83%** của LSTM, còn RMSE Ridge vẫn sát Zero Return. Đợt 4 đưa `alpha=100` thành ứng viên và đợt 6 cho thấy nhánh phân loại chưa đủ mạnh. [So sánh trên 104 phiên](../../reports/period_comparison/20261030T024048531444Z/summary.csv)
3. **Cửa sổ và hiệu chỉnh:** Đợt 8 chốt **3 năm** theo RMSE validation. Đợt 5 giải thích vì sao demo giữ tùy chọn **bias60**. Đây là hai quyết định riêng. Các lượt đổi cách cộng bias ở tuần/tháng chỉ thay quy tắc mô phỏng, dùng lại trọng số Ridge. [Nhật ký hai chế độ bias](../../TRAINING_HISTORY.md)

**Bao nhiêu thử nghiệm là đủ?** Không có số cố định. Tám đợt ở đây đủ để giải thích các quyết định của demo vì nhóm đã so với baseline trên cùng ngày đích, thử các giả thuyết khác nhau và giữ cả kết quả không cải thiện. Khi chênh lệch còn nhỏ và đã xem nhiều lần dữ liệu lịch sử, bước có giá trị hơn là ghi dự báo trước ngày đích rồi chấm bằng các phiên mới.

## 5. Kết luận và demo

> **Câu kết:** Nhóm hoàn thành quy trình từ kiểm tra dữ liệu đến huấn luyện, đối chứng và phục vụ dự báo qua API. Ridge6 là cấu hình nhóm chốt cho demo. Sai số lịch sử của Ridge gần baseline giữ nguyên giá. Muốn đánh giá độ ổn định, cần lưu thêm dự báo trước khi biết kết quả rồi mới chấm.

Mở [dashboard preview](http://127.0.0.1:8502/) nếu đã chạy dịch vụ đó. Trên máy khác, chạy theo [README](../../README.md#chạy-demo-trên-máy). Trước giờ trình bày, mở thử `/health` và **Dự báo** để kiểm tra kết nối, phiên bản model và ngày snapshot.

1. **Tổng quan:** Chỉ nến FPT, nguồn Vietcap và ngày dữ liệu cuối **29/10/2026**. Các chỉ báo trên Tổng quan dùng để quan sát dữ liệu. Đầu vào Ridge là bộ 6 lợi suất đã chọn.
2. **Dự báo:** Chọn **Hôm nay** hoặc một **ngày đích** đang khả dụng. Chuyển **Nến Nhật / Đường giá** và so **Mỗi bước · cách cũ / Chỉ bước đầu**. Nến là giá đã quan sát. Phần nối từ snapshot và đường về sau là mô phỏng. Tuần/tháng được tính bằng cách lặp model một phiên. API giới hạn 23 bước ngày làm việc từ snapshot và chưa trừ lịch nghỉ HOSE. Nếu ngày đích đã vượt giới hạn, chuyển sang **Đối chứng** để demo lịch sử.
3. **Đối chứng:** Chọn tập **104 phiên tháng 6–10/2026** để đặt Ridge, LSTM và Zero Return cạnh nhau trên cùng ngày đích. Lọc một tháng để thấy chỉ số thay đổi theo giai đoạn.
4. **Hành trình model và API:** Chiếu các đợt thử nghiệm khi giảng viên hỏi vì sao chọn Ridge. Đạt có thể mở Swagger `/docs`, gọi `POST /api/v1/predict` với `{}` để minh họa JSON một bước từ snapshot, rồi chỉ cách dashboard dùng `POST /api/v1/forecast` để lấy đường dự báo.

**Nói về hai chế độ bias.** `bias60` là hiệu chỉnh từ tối đa 60 sai số đã biết. Dashboard mặc định chọn **Mỗi bước · cách cũ**, cộng cùng mức bias ở mỗi bước. **Chỉ bước đầu** là mặc định của API khi thiếu `bias_mode` và là chế độ dùng trong đối chứng h=5/h=22. Hai cách cho cùng kết quả bước đầu. Với ngày đích 27/11 trong lượt replay đã ghi, hai kịch bản cho khoảng **48.120 đồng** và **51.666 đồng** trên cùng snapshot. Chênh lệch đó đo tác động của quy tắc cộng bias. Xem mục 16 trong [nhật ký thực nghiệm](../../TRAINING_HISTORY.md).

## 6. Câu hỏi lý thuyết cả nhóm nên tập

| Nếu giảng viên hỏi | Câu trả lời ngắn |
| --- | --- |
| Hồi quy hay phân loại, có nhãn không? | Hồi quy có giám sát. Nhãn là log return phiên sau, tính tự động khi đã biết giá đóng cửa thực tế. Đoán tăng/giảm là nhánh phân loại riêng. |
| Rò rỉ dữ liệu là gì? | Dùng thông tin chưa có ở thời điểm dự báo, ví dụ fit scaler trên cả Test hoặc cho mẫu cuối năm có nhãn đầu năm sau vào Train. |
| Walk-forward là gì? | Ở mỗi năm đánh giá, chỉ train bằng lịch sử trước năm đó. Dịch mốc lên rồi lặp lại để quan sát model qua nhiều giai đoạn. |
| Vì sao xem cả RMSE và Balanced Accuracy? | RMSE đo độ lớn sai số và phạt mạnh sai số lớn. Balanced Accuracy lấy trung bình khả năng nhận diện lớp tăng và giảm. Trong tập 2026 đã xét, luôn đoán giảm đạt **59,39% hit**, còn Balanced Accuracy là **50%**. |
| Ridge đạt bao nhiêu? | Trên 202 cặp năm 2026, **bản lịch sử fit tới hết 2025** đạt **60,91% hit** trên 197 phiên có biến động và **56,62% Balanced Accuracy**. Bundle demo fit đến 29/10/2026 cần được đánh giá riêng bằng dữ liệu mới. |
| P10–P90 nghĩa là gì? | Vùng này lấy hai phân vị bao quanh khoảng giữa của 512 giá mô phỏng ở từng bước. Muốn biết vùng dự báo phủ được bao nhiêu giá thực tế, phải kiểm tra trên dữ liệu. |

## 7. Phần mỗi thành viên cần nắm

Phân công và tỷ lệ lấy từ [bảng phân công trong báo cáo](bao_cao_fpt_forecasting.pdf) và [tài liệu demo chi tiết](HUONG_DAN_DEMO_VA_TRA_LOI_VAN_DAP.md). Mỗi bạn chuẩn bị **một file, một bảng kết quả hoặc một thao tác demo** cho phần mình phụ trách.

| Thành viên | Task đã ghi trong báo cáo | Cần tự giải thích hoặc thao tác được | Tỷ lệ ghi trong báo cáo |
| --- | --- | --- | ---: |
| **Trương Trọng Phúc** · `3124410287` | Thiết lập môi trường, chuẩn hóa pipeline, điều phối huấn luyện. Đợt **1 và 8**, dự báo đệ quy và Bootstrap. Viết Chương **1 và 5**. | Mở đầu và kết luận. Giải thích nhãn, đối chứng CPU/Colab, bốn năm đánh giá cuốn chiếu và vì sao chọn 3 năm. Minh họa hai cách áp dụng bias. | **25%** |
| **Đỗ Đạt** · `3124410054` | Xây FastAPI, xác thực input bằng Pydantic, kiểm thử endpoint. Đợt **2 và 6**. Viết Chương **3**. | Mở Swagger và gọi `/health`, `/api/v1/predict`. Nói lý do cần baseline và giải thích vì sao luôn đoán giảm có thể cho hit cao nhưng Balanced Accuracy chỉ 50%. | **25%** |
| **Phạm Khánh Nguyên** · `3124410239` | Xây Streamlit Studio và các sơ đồ giải thích. Đợt **3 và 7**. Viết đầu Chương **4**. | Dẫn demo, chỉ sự khác nhau giữa nến lịch sử và đường dự báo. Nêu bằng chứng chọn Vietcap và kết quả khi thử tách Gap/Body. | **25%** |
| **Chế Văn Nam** · `3124410217` | Làm sạch OHLCV, tạo nhãn và đặc trưng. Đợt **4 và 5**. Viết Chương **2** và phần chọn model ở Chương **4**. | Dùng một mẫu để chỉ ngày gốc, ngày đích và cách tránh rò rỉ. Giải thích bộ 6 lợi suất, vai trò `alpha`, cửa sổ train và tác dụng của bias60. | **25%** |

Thứ tự nói gợi ý: **Phúc mở bài → Nam nói dữ liệu → Đạt nói mô hình → Nguyên nói đối chứng và dẫn demo → Phúc kết luận**. Khi giảng viên hỏi sâu, chuyển đúng người phụ trách. Nếu phần đóng góp thực tế khác bảng trên, cả nhóm xác nhận lại và sửa tỷ lệ trước khi nộp, tổng bằng 100%.

Tài liệu để mở lúc tập: [README chạy demo](../../README.md) · [Kiến trúc](../../ARCHITECTURE.md) · [Nhật ký thực nghiệm](../../TRAINING_HISTORY.md) · [Kịch bản vấn đáp đầy đủ](HUONG_DAN_DEMO_VA_TRA_LOI_VAN_DAP.md).
