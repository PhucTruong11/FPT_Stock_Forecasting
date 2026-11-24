# Hướng dẫn nhóm trình bày đồ án dự báo FPT

> **Câu mở đầu:** Nhóm dùng giá lịch sử của FPT để dự báo lợi suất phiên kế tiếp, quy đổi kết quả ra giá đóng cửa và trình bày trên dashboard. Nhóm so sánh từ baseline đến mô hình học sâu rồi chọn Ridge để phục vụ demo.

Đọc mục 1–5 để nắm mạch trình bày. Mỗi bạn xem thêm phần mang tên mình ở mục 7 và tập trả lời bằng ví dụ từ chính phần mình làm.

Mục 8 có kịch bản demo khoảng 5 phút. Mục 9 ghi các giải thích chuyên sâu đã trao đổi, kèm công thức, ví dụ và đường dẫn tới code hoặc bằng chứng để cả nhóm ôn vấn đáp.

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

## 8. Kịch bản giới thiệu và demo khoảng 5 phút

Chia thành **4 đoạn**, thời lượng đã tính cả thao tác trên màn hình. Các bạn tập nói theo ý, dùng lời gợi ý dưới đây để nối mạch. Trước buổi trình bày, mở sẵn dashboard, Swagger `/docs` và bảng tổng quan 8 đợt ở mục 4. Gọi `/health` để kiểm tra kết nối và ngày dữ liệu cuối.

### Đoạn 1 · Giới thiệu tổng quan · 0:00–0:45 · Phúc

- Tên đề tài, lý do tập trung một mã FPT và sản phẩm nhóm xây dựng.
- Đầu vào là lịch sử giá, đầu ra là lợi suất phiên kế tiếp rồi quy đổi thành giá đóng cửa.
- Xác định bài toán hồi quy có giám sát và giới thiệu nhanh toàn bộ quy trình.

> Chào thầy và các bạn, nhóm em thực hiện đề tài phân tích và dự báo cổ phiếu FPT bằng học máy. Nhóm tập trung vào một mã để kiểm tra dữ liệu và so sánh mô hình sâu hơn. Từ lịch sử giao dịch, hệ thống dự báo lợi suất phiên kế tiếp rồi quy đổi thành giá đóng cửa. Đây là bài toán hồi quy có giám sát, với nhãn được tính từ giá thực tế của phiên sau. Quy trình nhóm thực hiện gồm lấy dữ liệu, làm sạch, tạo đặc trưng, huấn luyện, đối chứng, chọn mô hình và đưa kết quả lên ứng dụng.

**Trên màn hình**, chỉ sơ đồ quy trình ở đầu tài liệu, sau đó chuyển sang phần dữ liệu.

### Đoạn 2 · Dữ liệu và cách chuẩn bị để học · 0:45–1:50 · Nam

- Nguồn dữ liệu OHLCV ngày, lý do chọn Vietcap sau khi kiểm tra Yahoo và KBS.
- Làm sạch, tạo 6 đặc trưng và gắn nhãn phiên kế tiếp.
- Chia Train, Validation, Test theo thời gian và fit scaler trên Train.

> Nhóm thu thập giá mở cửa, cao nhất, thấp nhất, đóng cửa và khối lượng giao dịch FPT từ năm 2018. Sau khi đối chiếu Yahoo Finance, Vietcap và KBS, nhóm chọn Vietcap làm nguồn chính vì snapshot kiểm tra có chất lượng tốt hơn. Khi làm sạch, nhóm loại dòng khối lượng bằng 0, cách ly nến sai và giữ lịch giao dịch thực tế. Từ giá đã quan sát, nhóm tạo các đặc trưng lợi suất. Bộ cuối cùng gồm 6 đặc trưng, tính từ 21 giá đóng cửa gần nhất. Mỗi mẫu được ghép với lợi suất phiên kế tiếp làm nhãn. Dữ liệu được chia theo thời gian, Train để học, Validation để chọn cấu hình và Test để đánh giá. Scaler chỉ fit trên Train để tránh sử dụng thông tin tương lai. Dù có lịch sử từ 2018, mỗi lần fit có thể dùng một cửa sổ gần hơn, tùy cấu hình đem so sánh.

**Trên màn hình**, mở **Tổng quan**, chỉ nến FPT và ngày snapshot **29/10/2026**, rồi chuyển đến bảng tóm tắt thực nghiệm.

### Đoạn 3 · Thử nghiệm và lý do chọn Ridge · 1:50–3:10 · Đạt

- So baseline giữ nguyên giá với Ridge, XGBoost và LSTM.
- Tóm tắt mục đích của 8 đợt, từ chọn dữ liệu đến chọn cửa sổ train.
- Chốt Ridge6, `alpha=100`, cửa sổ 3 năm và hiệu chỉnh bias60.
- Nêu một kết quả lịch sử cùng phạm vi đánh giá của nó.

> Nhóm thử Ridge bằng scikit-learn, XGBoost và LSTM bằng TensorFlow/Keras. Đồng thời nhóm giữ baseline dự báo giá phiên sau bằng giá hiện tại để biết mô hình có tạo thêm giá trị hay không. Tám đợt thực nghiệm lần lượt kiểm tra pipeline, độ ổn định, nguồn dữ liệu, bộ đặc trưng và tham số, hiệu chỉnh sai lệch, phân loại tăng giảm, đặc trưng Gap/Body và độ dài lịch sử. Kết quả dẫn đến cấu hình Ridge dùng 6 lợi suất, alpha bằng 100 và cửa sổ 3 năm. Bộ đặc trưng gọn cho kết quả cạnh tranh trong các đối chứng, còn Gap/Body và nhánh phân loại chưa cải thiện ổn định. Nhóm thêm bias60 để hiệu chỉnh từ các sai số gần nhất đã biết. Trên 202 cặp lịch sử năm 2026, bản Ridge fit tới hết 2025 kèm bias60 đạt Balanced Accuracy 56,62% và RMSE gần baseline. Đây là căn cứ lịch sử cho lựa chọn demo. Bundle fit đến tháng 10/2026 cần được chấm tiếp bằng dự báo ghi trước và kết quả xuất hiện sau.

**Trên màn hình**, chỉ bảng 8 đợt ở mục 4, rồi mở **Đối chứng** với tập **202 cặp** để minh họa đúng phạm vi kết quả vừa nói.

### Đoạn 4 · Demo ứng dụng và kết luận · 3:10–5:00 · Nguyên, Phúc chốt

- Minh họa luồng Streamlit gửi yêu cầu → FastAPI suy luận bằng model đã nạp → trả JSON → vẽ biểu đồ.
- Demo giá lịch sử, một dự báo hợp lệ và đối chứng với giá thực tế.
- Giải thích dự báo nhiều bước, vùng P10–P90 và hướng đánh giá tiếp theo.

> Sau khi chọn mô hình, nhóm đóng gói Ridge cùng scaler, cấu hình hiệu chỉnh và snapshot dữ liệu. FastAPI phục vụ dự báo qua REST API, backend được cấu hình chạy trên Render, còn giao diện Streamlit gọi API để hiển thị kết quả. Ở trang Tổng quan, đây là các nến giá lịch sử. Sang Dự báo, người dùng chọn mô hình và ngày đích khả dụng để xem giá đóng cửa dự báo cùng vùng kịch bản P10–P90. Kỳ hạn tuần hoặc tháng được tạo bằng cách lặp mô hình dự báo một phiên. Khi đi xa hơn, sai số có thể tích lũy và độ phủ của vùng dự báo cần được kiểm tra bằng backtest. Ở trang Đối chứng, nhóm đặt dự báo lịch sử cạnh giá thực tế trên cùng ngày để xem cả sai số và khả năng đoán hướng. Qua đồ án, nhóm xây dựng được quy trình từ dữ liệu đến ứng dụng và lưu bằng chứng cho từng quyết định chọn mô hình. Bước tiếp theo là ghi dự báo trước khi biết giá thật, theo dõi sai số và chỉ thay mô hình khi có kết quả kiểm chứng phù hợp.

**Thứ tự thao tác trong 1 phút 50 giây cuối**

1. **3:10–3:30** — Chỉ nến lịch sử và ngày dữ liệu cuối trên **Tổng quan**. Giới thiệu luồng UI gọi API.
2. **3:30–4:05** — Mở **Dự báo**, chọn ngày còn trong giới hạn API, chỉ giá dự báo và vùng P10–P90. Nếu lịch hiện tại đã vượt giới hạn snapshot, mở Swagger, gọi `POST /api/v1/predict` với `{}` để minh họa dự báo phiên kế tiếp từ snapshot.
3. **4:05–4:40** — Mở **Đối chứng**, chọn **Tháng 6–10/2026 · 104 cặp, có LSTM**. Chỉ giá thực tế, các đường dự báo và bảng chỉ số trên cùng tập ngày. Có thể mở **Hành trình model** nếu thao tác còn thời gian.
4. **4:40–5:00** — Phúc kết luận bằng hai câu cuối trong lời gợi ý, rồi mời thầy đặt câu hỏi.

## 9. Giải thích chuyên sâu từ các buổi trao đổi

Phần này tổng hợp kiến thức đã cùng tìm hiểu đến **24/11/2026**. Số liệu được ghi theo snapshot và từng thí nghiệm. Khi cần trình bày diễn biến 8 đợt, dùng bảng tổng quan và các kết quả chi tiết ở mục 4.

### 9.1. Bài toán, đầu vào, nhãn và đầu ra

Tên đồ án trên bìa là **Phân tích và dự báo giá đóng cửa cổ phiếu FPT bằng học máy**. Mô hình chính học lợi suất phiên kế tiếp, sau đó hệ thống quy đổi kết quả thành giá đóng cửa điều chỉnh. Nhóm tập trung FPT để kiểm tra dữ liệu, thực nghiệm và giải thích lựa chọn sâu hơn trên một mã.

Gọi `C_t` là giá đóng cửa điều chỉnh của phiên gốc `t`. Một mẫu huấn luyện có hai phần.

- **Đầu vào X_t** gồm các feature tính từ giá đã quan sát đến hết phiên `t`.
- **Nhãn y_t** là `ln(C_next / C_t)`, với `C_next` là giá của quan sát giao dịch kế tiếp đủ điều kiện làm nhãn.

Ví dụ, giá tại `t` là 50.000 đồng và phiên sau thực tế là 51.000 đồng. Nhãn của mẫu tại `t` bằng `ln(51.000 / 50.000) ≈ 0,0198`, tương ứng giá tăng 2%. Khi học trên lịch sử, giá phiên sau đã có nên tính được nhãn. Khi đưa ra dự báo mới, nhãn sẽ được bổ sung sau khi quan sát giá thực tế.

Đây là **hồi quy có giám sát**, vì mô hình học từ cặp đầu vào và giá trị mục tiêu. Nhãn được tính tự động từ chuỗi giá. Nếu đổi mục tiêu thành tăng/giảm thì thành bài toán phân loại có giám sát. Các phương pháp như phân cụm hoặc PCA có thể hỗ trợ biểu diễn dữ liệu; nếu thử chúng trong tương lai, các tham số cũng cần được fit trên Train.

**Câu trả lời khi được hỏi về “6 nhãn”** — Ridge đang dùng **6 feature đầu vào và 1 nhãn lợi suất phiên kế tiếp**. Cột `Target_Date_1d` ghi ngày của nhãn để kiểm soát việc chia tập. [Code tạo mẫu](../../src/features/build_features.py)

### 9.2. Kiểm tra và làm sạch OHLCV cụ thể như thế nào?

OHLCV là các cột Open, High, Low, Close và Volume, đi cùng ngày giao dịch. Chuỗi dùng trong nghiên cứu chính được lấy từ Vietcap sau khi đối chiếu với Yahoo Finance và KBS. File `download_dataset.py` là bộ tải Yahoo qua yfinance, còn `source_audit.py` chứa phần lấy và đối chiếu nhiều nguồn.

| Việc kiểm tra | Cách xử lý trong pipeline | Lý do |
| --- | --- | --- |
| Ngày và thứ tự | Chuẩn hóa ngày D1, sắp xếp từ cũ đến mới | Xác định đúng quá khứ và phiên kế tiếp |
| Dòng trùng ngày | Gộp dòng trùng giống nhau, báo lỗi khi số liệu mâu thuẫn | Mỗi ngày có một quan sát thống nhất |
| Giá và khối lượng | Kiểm tra kiểu số, giá dương và Volume nguyên không âm | Phép tính lợi suất cần giá hợp lệ |
| Quan hệ giữa các giá | Kiểm tra Low ≤ Open/Close ≤ High và Low ≤ High | Phát hiện nến sai hình học |
| Volume bằng 0 | Loại khỏi tập quan sát dùng để học và ghi lý do | Những dòng giữ lại giá cũ có thể tạo nhãn lợi suất bằng 0 |
| Giá thiếu hoặc vô hạn | Báo lỗi khi vi phạm điều kiện nguồn đầu vào | Giữ phép tính và nguồn sai sót có thể kiểm tra được |
| Lịch quan sát | Giữ các ngày giao dịch có trong nguồn | Tránh tạo thêm quan sát giả vào ngày nghỉ |

Ví dụ, High bằng 50.000 nhưng Close bằng 52.000 là một nến sai. Nếu quan sát kế tiếp có khối lượng dương nhưng là nến lỗi, pipeline loại nhãn tương ứng của phiên đứng trước. Ngày đích được xác định trước khi loại nến đó để kiểm soát sự thay đổi kỳ hạn.

Trong bundle hiện tại, **2.202 dòng raw → loại 2 dòng Volume bằng 0 → còn 2.200 quan sát hợp lệ**. Giá có basis điều chỉnh của nhà cung cấp. Khi đối chiếu nguồn hoặc ghép dữ liệu mới, cần kiểm tra cùng đơn vị và cách điều chỉnh cổ tức/chia tách.

Ngày gốc và ngày đích được ghi tường minh. `calendar_verified=false` cho biết lịch được suy từ các quan sát nguồn và chưa được xác minh độc lập bằng toàn bộ lịch HOSE. Snapshot đóng băng giúp tái lập đúng đầu vào của một lượt nghiên cứu. [Kiểm tra OHLCV](../../src/data/contracts.py) · [Đối chiếu nguồn](../../src/data/source_audit.py) · [Metadata bundle](../../models/local_demo/ridge6_20261029/manifest.json)

### 9.3. Sáu feature đã chọn và ví dụ cách tính

Viết `C[t]` là giá đóng cửa tại phiên `t` và `r[t] = ln(C[t] / C[t-1])`, trong đó chỉ số thời gian chạy theo các quan sát hợp lệ. Một hàng đầu vào Ridge gồm đúng thứ tự sau.

| Feature | Công thức | Thông tin cung cấp |
| --- | --- | --- |
| `Log_Return` | `r[t]` | Biến động từ phiên trước tới phiên hiện tại |
| `Return_Lag_1` | `r[t-1]` | Lợi suất một phiên tại thời điểm cách hiện tại 1 phiên |
| `Return_Lag_2` | `r[t-2]` | Lợi suất một phiên tại thời điểm cách hiện tại 2 phiên |
| `Return_Lag_5` | `r[t-5]` | Lợi suất một phiên tại thời điểm cách hiện tại 5 phiên |
| `Return_5` | `ln(C[t] / C[t-5])` | Biến động tích lũy trong 5 phiên |
| `Return_20` | `ln(C[t] / C[t-20])` | Biến động tích lũy trong 20 phiên |

Ví dụ, Close hiện tại là 50.000 đồng và Close cách 5 phiên là 48.000 đồng. `Return_5 = ln(50.000 / 48.000) ≈ 0,04082`, tương ứng mức tăng giá tích lũy khoảng 4,167%. Log return được quy đổi sang phần trăm thay đổi giá bằng `(exp(r) − 1) × 100`.

Để tính `Return_20`, cần giá ở cả `t` và `t−20`, nên lịch sử tối thiểu cho bộ 6 feature gồm **21 giá đóng cửa**. Thông tin của nhiều phiên đã được gói vào vector 6 số này.

Pipeline chung tạo **16 feature** để phục vụ các đối chứng, gồm cả biến động, SMA, RSI, ATR và khối lượng. Model cuối chọn 6 cột lợi suất. Các chỉ báo trên dashboard còn có vai trò giúp người xem phân tích dữ liệu.

Builder chung bỏ **49 dòng warm-up** để đủ lịch sử cho SMA50. Vì vậy snapshot còn **2.151 hàng feature**, gồm **2.150 hàng có nhãn** và một hàng mới nhất chờ kết quả phiên sau. API nhận OHLCV tự nhập yêu cầu ít nhất **51 dòng** theo hợp đồng builder chung. Các yêu cầu này xuất phát từ các bước khác nhau trong pipeline. [Công thức và warm-up](../../src/features/build_features.py)

### 9.4. Phân biệt D1, h=1, lookback và cửa sổ train

| Khái niệm | Ý nghĩa trong project |
| --- | --- |
| **D1** | Một nến tổng hợp một ngày giao dịch, gồm cả sáng và chiều |
| **h=1** | Dự báo một quan sát giao dịch kế tiếp sau ngày gốc |
| **H1 / H4** | Khung nến một giờ / bốn giờ, là các khung dữ liệu khác có thể nghiên cứu sau |
| **21 giá đóng cửa** | Lịch sử đủ để tính 6 feature của Ridge |
| **Lookback 30** | LSTM tham chiếu đọc một chuỗi gồm 30 hàng feature |
| **Cửa sổ train 3 năm** | Khoảng lịch sử dùng để học trọng số của một phiên bản Ridge |

Thứ Sáu có thể có ngày đích là thứ Hai nếu đó là quan sát giao dịch hợp lệ kế tiếp. Một bước được tính theo quan sát, còn khoảng cách ngày lịch phụ thuộc cuối tuần, ngày nghỉ và chất lượng nguồn. Muốn bổ sung H1/H4 cần thu thập dữ liệu tương ứng, tạo lại nhãn và đánh giá lại cấu hình.

### 9.5. Chia theo thời gian, chuẩn hóa và tránh rò rỉ

Trong benchmark ban đầu, dữ liệu được chia theo các mốc sau.

~~~text
Train đến hết 2022 → Validation trong 2023–2024 → Test trong 2025
Học trọng số        Chọn cấu hình                Đánh giá cấu hình đã chọn
~~~

Mỗi mẫu có cả ngày gốc và ngày nhãn. Khi chốt Train ở cuối năm, chỉ nhận mẫu có ngày nhãn nằm trong giới hạn đã biết. Nếu mẫu cuối tháng 12 mang nhãn tháng 1 năm sau, mẫu đó nằm ngoài Train tại mốc này.

`StandardScaler` học trung bình `mu_j` và độ lệch chuẩn `sigma_j` của từng cột trên Train. Nó biến đổi giá trị thành `z_j = (x_j − mu_j) / sigma_j`. Validation, Test và các lần dự báo sau dùng lại scaler đã fit.

Feature được phép tính trước khi chia tập nếu từng hàng chỉ sử dụng hiện tại và quá khứ. Chẳng hạn, trung bình trượt kết thúc tại ngày gốc là feature nhân quả. Các bước học tham số từ dữ liệu, như scaler hoặc PCA, cần fit trong phạm vi Train của lượt đánh giá đó.

Các đợt sau có đánh giá cuốn chiếu theo năm. Ví dụ, với cửa sổ 3 năm, để đánh giá năm 2025 thì fit trên 2022–2024. Việc tính feature đầu năm đánh giá có thể sử dụng lịch sử đã có trước đó để đủ ngữ cảnh. Nhãn và các tham số fit vẫn tuân theo mốc thời gian.

**Vai trò của Validation và Test** — Validation phục vụ lựa chọn. Test phục vụ báo cáo sau khi cấu hình được khóa. Khi đã xem một giai đoạn nhiều lần và dựa vào kết quả để điều chỉnh nghiên cứu, cần ghi nhận việc tiếp xúc dữ liệu đó. Giai đoạn 2026 trong project đã được xem trong quá trình nghiên cứu. [Chia tập benchmark](../../src/models/experiment.py) · [Chia cửa sổ theo ngày nhãn](../../src/models/final_window_comparison.py) · [Lịch sử tiếp xúc dữ liệu đánh giá](../../reports/holdout_exposures.json)

### 9.6. Ridge, XGBoost và LSTM học khác nhau thế nào?

**Ridge** là hồi quy tuyến tính có phạt L2 lên trọng số. Sau khi chuẩn hóa đầu vào, model tính `y_hat = b + w1×z1 + ... + w6×z6`. Khi huấn luyện, nó tìm trọng số để giảm tổng sai số bình phương cùng mức phạt `alpha × tổng(w_j²)`. `alpha` lớn hơn tăng mức phạt khi giữ nguyên dữ liệu và quy ước hàm mục tiêu. Trong dự án, scikit-learn thực hiện bước fit trên CPU.

**XGBoost** kết hợp nhiều cây quyết định, mỗi cây chia đầu vào theo các điều kiện. Các cây được thêm theo từng lượt nhằm cải thiện hàm mất mát của tập cây hiện có. Nó có thể học tương tác phi tuyến, chẳng hạn tác động của lợi suất gần nhất phụ thuộc vào xu hướng 20 phiên. `XGBRegressor` phục vụ nhánh hồi quy; `XGBClassifier` từng được thử cho nhánh dự báo hướng.

**LSTM** là kiến trúc mạng nơ-ron xử lý chuỗi theo thời gian. Trạng thái và các cổng của mạng học cách giữ hoặc giảm bớt thông tin khi đọc qua các phiên. Benchmark thử lookback 10, 30 và 60; bản tham chiếu được chọn trong họ LSTM dùng **30 phiên × 16 feature**. Kiến trúc benchmark gồm LSTM 32 unit, Dropout 0,2, Dense 8 unit rồi một đầu ra liên tục. Huấn luyện dùng TensorFlow/Keras, optimizer Adam và loss Huber, có early stopping theo Validation.

| Khía cạnh | Ridge đang dùng | XGBoost đối chứng | LSTM tham chiếu |
| --- | --- | --- | --- |
| Đầu vào | Một vector 6 feature | Một hàng feature theo cấu hình thử | Một chuỗi 30 × 16 ở bản tham chiếu |
| Cách biểu diễn quan hệ | Tổng có trọng số tuyến tính | Tổ hợp cây quyết định | Trạng thái học từ chuỗi có thứ tự |
| Công cụ | scikit-learn | Thư viện XGBoost | TensorFlow/Keras |
| Vai trò | Phục vụ dự báo chính | Kiểm tra mô hình phi tuyến | Kiểm tra nhánh học sâu |

Các feature trễ giúp Ridge và XGBoost tiếp cận thông tin quá khứ. Nhánh học sâu đã triển khai trong đồ án dùng LSTM. Google Colab là môi trường chạy, có GPU phục vụ thí nghiệm LSTM. CPU vẫn phù hợp với Ridge và có thể chạy suy luận LSTM nhỏ khi đủ thư viện. [Code mô hình và huấn luyện](../../src/models/experiment.py)

### 9.7. Gói Ridge hiện tại chứa những gì?

Bundle được chọn là `ridge6_20261029`, nguồn Vietcap, cửa sổ train 3 năm và nhãn đến **29/10/2026**. Lựa chọn mặc định trên API là `Ridge+bias60`. `alpha=100` là siêu tham số được chọn khi đối chứng cấu hình, còn các trọng số được học từ dữ liệu Train.

| Thành phần | Nội dung |
| --- | --- |
| `ridge.pkl` | Pipeline StandardScaler và Ridge đã fit, gồm tham số chuẩn hóa, 6 trọng số, hệ số nền và `alpha=100` |
| `calibration.json` | Cấu hình bias60, sai số lịch sử và thông tin hiệu chỉnh |
| `raw.csv` | Snapshot OHLCV dùng làm đầu vào mặc định |
| `backtest.csv.gz` | Dự báo lịch sử của các model dùng cho đối chứng |
| `manifest.json` | Phiên bản, thứ tự feature, nguồn, ngày dữ liệu, mốc nhãn train và hash artifact |
| `models/local_demo.json` | File trỏ đến bundle đang được phục vụ |

Khi API khởi động, nó kiểm tra manifest, hash và thứ tự feature rồi nạp pipeline. Mỗi request thực hiện suy luận bằng các tham số đã lưu.

Ba lựa chọn trong trang Dự báo có quan hệ như sau.

- **Ridge+bias60** dùng đầu ra Ridge cộng mức hiệu chỉnh.
- **Ridge** dùng trực tiếp đầu ra của cùng pipeline đã fit.
- **Zero Return** giữ giá gốc bằng cách dự báo lợi suất bằng 0.

Kết quả so sánh lịch sử trong `backtest.csv.gz` thuộc các model có mốc train phù hợp với giai đoạn được chấm. Bundle fit đến tháng 10/2026 cần được theo dõi bằng các nhãn xuất hiện sau mốc này. [Bundle và hợp đồng dữ liệu](../../models/local_demo/ridge6_20261029/manifest.json) · [Code nạp và suy luận](../../src/models/local_demo.py)

### 9.8. Dựa vào đâu để nói một lượt training tốt hơn?

Mỗi lượt bắt đầu bằng một câu hỏi cụ thể, chẳng hạn thay nguồn, rút gọn feature, thêm hiệu chỉnh hoặc đổi số năm train. Kết luận được rút từ dự báo trên các ngày đánh giá, so với kết quả thật và baseline tương ứng.

| Chỉ số | Điều nó đo | Cách đọc |
| --- | --- | --- |
| **RMSE log return** | Căn bậc hai của trung bình sai số lợi suất bình phương | Càng thấp càng tốt, nhạy với lần sai lớn |
| **MAE giá** | Trung bình trị tuyệt đối chênh lệch giá dự báo và giá thật | Càng thấp càng tốt, đơn vị VND dễ giải thích |
| **Hit rate** | Tỷ lệ đoán đúng dấu trên các phiên có biến động thực tế | Đọc cùng tỷ lệ tăng/giảm và quy ước dự báo trung tính |
| **Balanced Accuracy** | Trung bình recall của nhóm tăng và nhóm giảm | Cho biết khả năng nhận diện cả hai nhóm |
| **Độ phủ vùng dự báo** | Tỷ lệ giá thật nằm trong band | Đọc cùng độ rộng band và interval score |
| **Độ ổn định** | Mức thay đổi kết quả qua năm, fold hoặc seed | Xem cả trung bình và giai đoạn có kết quả tệ nhất |

Trong một đối chứng, cần cùng ngày gốc/ngày đích, nguồn hoặc basis đã ghép hợp lệ, cùng kỳ hạn và quy tắc chấm. Baseline giúp biết mức cải thiện so với một cách dự báo đơn giản như giữ nguyên giá hoặc dùng lợi suất trung bình Train.

Một ví dụ từ đợt 5, trên cùng **202 cặp năm 2026**, cập nhật Ridge từ bản đóng băng 2023 sang bản fit đến hết 2025 giảm RMSE khoảng **2,83%**. Thêm bias60 đưa MAE giá về khoảng **1.064,12 đồng** và Balanced Accuracy **56,62%**. RMSE của bản này là **0,0201116**, còn Zero Return là **0,0201157**, nên lợi thế theo chỉ số đó rất nhỏ.

Hit **60,91%** của Ridge+bias60 được tính trên **197 phiên có biến động**. Trong cùng tập, luôn đoán giảm cũng đạt **59,39% hit**, vì số phiên giảm chiếm đa số; Balanced Accuracy của cách luôn giảm là **50%**. Vì vậy cần xem nhiều chỉ số và baseline trước khi kết luận khả năng dự báo hướng.

Lỗi Train giảm cho biết quá trình fit đang thích nghi với dữ liệu học. Nếu lỗi Validation tăng cùng lúc, đó là dấu hiệu quá khớp. Unit test kiểm tra logic, hợp đồng API, công thức và khả năng tái lập. Khả năng dự báo cần được chấm bằng các nhãn thực tế trên những giai đoạn phù hợp. [Kết quả recency/bias](../../reports/recency_bias/20261030T022440928507Z/metrics.csv) · [Nhánh phân loại](../../reports/direction_selective/20261030T090621177786Z/metrics.csv)

### 9.9. Vì sao có dữ liệu từ 2018 mà chốt train 3 năm?

Lịch sử dài giúp phân tích nhiều chế độ thị trường, tạo feature và tổ chức các đối chứng. Cửa sổ train quyết định những mẫu nào được dùng để học trọng số tại một mốc. Hai khái niệm này có thể có độ dài khác nhau.

Ở đợt cuối, nhóm giữ nguồn, 6 feature và `alpha=100`, chỉ đổi **3 năm, 5 năm, toàn bộ từ 2018**. Trên cùng 993 cặp validation của bốn năm 2022–2025, RMSE trung bình lần lượt là **0,01707812**, **0,01709506** và **0,01710787**. Cửa sổ 3 năm được chọn theo tiêu chí RMSE trung bình đã đặt trước.

Độ chênh lệch còn nhỏ, khoảng tin cậy mô tả của chênh lệch MSE chứa 0. Kết quả hỗ trợ quyết định dùng 3 năm cho demo. Đánh giá bằng các phiên mới sẽ giúp kiểm tra độ ổn định tiếp theo. Lịch sử dài hơn có thể cung cấp thêm mẫu nhưng cũng chứa các giai đoạn có quan hệ giá khác với hiện tại. [Kết quả chọn cửa sổ](../../reports/final_window_comparison/20261101T014607720130Z/validation_summary.csv) · [Khoảng tin cậy](../../reports/final_window_comparison/20261101T014607720130Z/paired_validation_ci.csv)

### 9.10. Bias60 là gì và tính từ đâu?

Tại một thời điểm đã có nhãn, sai số của Ridge thô được tính bằng `e = lợi suất thực tế − lợi suất dự báo thô`. Bias60 lấy trung bình tối đa **60 sai số gần nhất đã biết**. Code yêu cầu ít nhất **20 sai số đã có nhãn** để kích hoạt hiệu chỉnh.

~~~text
bias60 = trung bình các sai số thô gần nhất đủ điều kiện
lợi suất sau hiệu chỉnh = lợi suất Ridge thô + bias60
~~~

Ví dụ giả định, nếu Ridge gần đây thường dự báo cao hơn thực tế thì trung bình sai số âm. Cộng mức hiệu chỉnh âm sẽ hạ dự báo xuống. Nếu tình huống ngược lại, hiệu chỉnh có thể dương.

Trong backtest, các sai số phải có nhãn ở thời điểm phát dự báo mới được đưa vào tính bias. Trong bundle phục vụ hiện tại, pool sai số và hiệu chỉnh đã được đóng băng theo version. Cập nhật chúng cần một lượt dựng bundle có kiểm soát. [Code hiệu chỉnh nhân quả](../../src/models/recency_bias.py) · [Hiệu chỉnh trong bundle](../../src/models/local_demo.py)

### 9.11. Dự báo một phiên, tuần/tháng và hai chế độ cộng bias

Model chính học mục tiêu **h=1**. Đường tuần/tháng được tạo bằng dự báo đệ quy, bắt đầu từ giá thật cuối cùng rồi dùng giá vừa dự báo để tính lại feature cho bước sau.

~~~text
Lịch sử thật → dự báo bước 1 → cập nhật Close → tính lại 6 feature
                                                   ↓
                         tiếp tục cho bước 2, 3, ... đến ngày đích
~~~

Code hỗ trợ hai chế độ cho Ridge+bias60.

| Chế độ | Quy tắc | Phạm vi bằng chứng |
| --- | --- | --- |
| **Mỗi bước · cách cũ** | Cộng cùng mức bias ở từng bước | Dashboard mặc định chọn chế độ này sau lần đối chiếu tháng 11 |
| **Chỉ bước đầu** | Bước đầu có bias, các bước sau dùng Ridge thô | API mặc định dùng chế độ này khi thiếu `bias_mode`; các backtest h=5/h=22 đã ghi dùng quy tắc này |

Hai chế độ cho cùng kết quả ở bước đầu. Về sau, việc cộng bias lặp lại có thể tạo đường tăng/giảm mạnh hơn. Độ dốc quan sát được thể hiện cả tác động của quy tắc mô phỏng. Việc chọn cách vẽ dự báo cần đọc cùng kết quả kiểm chứng và mốc model.

Dashboard giới hạn **23 bước ngày làm việc kể từ snapshot**. Với snapshot cũ, một phần đường nối đến thời điểm hiện tại cũng được tính bằng mô phỏng. Việc chọn khung một tháng phụ thuộc ngày lịch, còn `h=22` trong backtest là 22 quan sát giao dịch. Ngày dự kiến hiện loại cuối tuần và chưa xác minh đầy đủ lịch nghỉ HOSE. [Code forecast](../../src/models/forecast_path.py) · [Nhật ký hai chế độ và backtest](../../TRAINING_HISTORY.md)

### 9.12. P10–P90, residual bootstrap và độ phủ

**Residual** là sai số giữa kết quả thật và dự báo. Hệ thống dùng phân bố sai số đã biết để mô phỏng thêm các đường giá quanh kết quả của model.

Trong `POST /api/v1/forecast`, quy trình được thực hiện theo các bước sau.

1. Lấy pool sai số phù hợp với model và trừ trung bình của pool để căn tâm.
2. Lấy mẫu lại theo block 5 quan sát, theo cơ chế vòng tròn, để giữ một phần cấu trúc phụ thuộc ngắn hạn có trong sai số lịch sử.
3. Tạo **512 đường mô phỏng**. Mỗi đường cập nhật giá và feature theo ngữ cảnh của chính nó ở từng bước.
4. Tại từng ngày đích, lấy phân vị 10% và 90% của 512 mức giá mô phỏng để tạo band.

| Mốc | Cách hiểu |
| --- | --- |
| **P10** | Khoảng 10% giá mô phỏng tại ngày đó nằm thấp hơn mức này |
| **P90** | Khoảng 90% giá mô phỏng tại ngày đó nằm thấp hơn mức này |
| **Vùng P10–P90** | Khoảng giữa chứa xấp xỉ 80% giá mô phỏng tại từng ngày |

Ví dụ giả định, P10 bằng 48.000 đồng và P90 bằng 53.000 đồng. Vùng tô ở ngày đó trải từ 48.000 đến 53.000 đồng. Đường dự báo điểm được tính bằng Ridge và quy tắc bias đã chọn. Các biên được tính từ các kịch bản sai số mô phỏng.

Con số 80% mô tả phân bố mô phỏng. **Độ phủ thực tế** là tỷ lệ giá thật nằm trong band khi chấm backtest. Cần đọc thêm độ rộng và interval score, vì một vùng rất rộng có thể chứa nhiều kết quả nhưng ít giá trị quan sát. Các phân vị này được tính riêng theo từng ngày; việc toàn bộ một đường giá thật nằm trong band là một sự kiện khác.

Căn tâm pool giúp kiểm soát việc phần trung bình sai số tiếp tục tạo thêm độ trôi trong các đường mô phỏng. Các nhiễu được lấy từ một pool có trung bình bằng 0, còn mức hiệu chỉnh có chủ đích được quy định riêng bằng chính sách bias. Giả định về phân bố sai số này cần được kiểm tra lại khi thị trường thay đổi hoặc có cú sốc mới.

Trong `POST /api/v1/predict`, khoảng một phiên được tính trực tiếp từ phân vị sai số đã lưu. Các đường nhiều bước trên dashboard dùng block bootstrap như trên. Cấu hình hiệu chỉnh độ rộng đã được thử cho h=5/h=22, nhưng hệ số đang kích hoạt của bundle là **1×**. Nếu sau này kích hoạt hệ số nới, cần mô tả đó là band đã hiệu chỉnh. [Hàm mô phỏng](../../src/models/forecast_path.py) · [Khoảng một phiên](../../src/models/local_demo.py) · [Cấu hình hiệu chỉnh band](../../configs/recursive_band_calibration.json)

### 9.13. Data flow từ dữ liệu gốc đến output hiện tại

~~~mermaid
flowchart TD
    subgraph TRAIN["Huấn luyện và lựa chọn"]
        A["OHLCV FPT đã kiểm tra nguồn"] --> B["Làm sạch và tạo feature X, nhãn y"]
        B --> C["Chia theo thời gian, fit trên Train"]
        C --> D["So baseline, Ridge, XGBoost và LSTM"]
        D --> E["Đối chứng, chốt cấu hình và đóng gói"]
    end
    subgraph SERVE["Suy luận trên API và dashboard"]
        U["Người dùng chọn model và kỳ hạn"] --> F["FastAPI nhận yêu cầu"]
        F --> G["Snapshot hoặc OHLCV gửi vào, tính feature"]
        G --> H["Scaler đã fit, Ridge và hiệu chỉnh đã chọn"]
        H --> I["Lợi suất dự báo, quy đổi thành giá"]
        I --> J["JSON, biểu đồ và bảng trên Streamlit"]
    end
    E --> H
~~~

Ví dụ sau lấy từ lần gọi pipeline một phiên trên snapshot Vietcap 29/10/2026, giá gốc **52.700 đồng**. Đây là dự báo cho quan sát kế tiếp sau snapshot, với version `ridge6_20261029`.

~~~json
{
  "model": "Ridge+bias60",
  "as_of": "2026-10-29",
  "horizon_observations": 1,
  "latest_close_adjusted": 52700.0,
  "predicted_log_return": -0.0043491147643509405,
  "predicted_simple_return_pct": -0.433967106027885,
  "predicted_close_adjusted": 52471.2993351233,
  "direction": "giam"
}
~~~

Phép quy đổi là `52.700 × exp(−0,004349...) ≈ 52.471,30 đồng`. Giao diện hiển thị hướng giảm và phần trăm thay đổi từ giá gốc. Nến Nhật dùng OHLC thực đã quan sát, đường về sau dùng giá đóng cửa dự báo. [Suy luận một phiên](../../src/models/local_demo.py) · [Endpoint JSON](../../src/api/main.py) · [Giao diện](../../src/app/streamlit_app.py)

### 9.14. Đóng gói model, API, deploy và UI có vai trò gì?

**Đóng gói model** là lưu đúng pipeline đã fit cùng metadata và các dữ liệu phục vụ cần thiết. File scaler và thứ tự feature phải khớp với model. Hash hỗ trợ phát hiện artifact thay đổi và kiểm tra khả năng tái lập.

**FastAPI** định nghĩa các endpoint. **Pydantic** kiểm tra kiểu dữ liệu và điều kiện request. **Uvicorn** chạy máy chủ Python. Các endpoint chính gồm `/health`, `/api/v1/predict`, `/api/v1/forecast` và `/api/v1/backtest`. Swagger tại `/docs` giúp gọi thử JSON trước khi nối giao diện.

**Render** là nơi cấu hình chạy backend qua mạng theo hướng dẫn trong README. Khi bàn giao, kiểm tra `/health` để biết version và ngày snapshot thực tế của backend đang gọi. Chi phí và điều kiện gói dịch vụ cần kiểm tra tại thời điểm triển khai.

**Streamlit** dựng giao diện, **Requests** gửi HTTP đến địa chỉ `FPT_API_URL`, còn **Plotly** vẽ biểu đồ từ kết quả API. Backend có thể chạy local hoặc ở địa chỉ đã deploy. Một request dùng model đã được nạp để tạo output, còn việc fit lại thuộc quy trình phát triển model.

`requirements-local.txt` phục vụ môi trường API/UI nhẹ. `requirements.txt` có thêm môi trường nghiên cứu, gồm TensorFlow và XGBoost. Việc chạy dự báo mới từ các artifact nghiên cứu cần cả thư viện lẫn đúng model/scaler tương ứng. [Hướng dẫn chạy và cấu hình Render](../../README.md) · [Kiến trúc chi tiết](../../ARCHITECTURE.md)

### 9.15. Model đang dùng và model giữ để nghiên cứu

| Nhóm | Trạng thái đến 24/11/2026 |
| --- | --- |
| **Dự báo trên dashboard** | Ridge+bias60, Ridge thô và Zero Return |
| **Đối chứng lịch sử có sẵn** | Có các dự báo LSTM tham chiếu trong tập chung tháng 6–10/2026, đọc cùng nguồn và mốc train của chúng |
| **Artifact nghiên cứu** | XGBoost, LSTM và các model của những study được lưu theo manifest hoặc protocol để tái lập |
| **Lượt quan sát ba model ngày 23/11** | Đã gọi thử các artifact có sẵn, sau đó gỡ phần mở rộng khỏi dashboard theo lựa chọn của người dùng |

Lượt quan sát ngày 23/11 có số fit mới bằng 0. Các model tham chiếu học trên Yahoo với nhãn đến 30/12/2022 và chọn cấu hình đến cuối 2024. Ridge hiện tại học trên Vietcap đến tháng 10/2026. Khi đưa cùng snapshot Vietcap vào, các đầu ra giúp quan sát phản ứng của từng model trước dữ liệu đó. Việc đánh giá hơn/kém còn cần đối chứng cùng điều kiện và nhãn thật.

`models/local_demo.json` trỏ bundle phục vụ. `models/reference_experiment.json` trỏ bộ tham chiếu đã khóa, còn `models/latest_experiment.json` thuộc pipeline thực nghiệm cũ. Các file này có vai trò riêng trong code. Bằng chứng của lần quan sát được ghi tại mục 17 của nhật ký. [Vai trò các thư mục model](../../README.md#model-đang-dùng-và-model-nghiên-cứu) · [Nhật ký quan sát](../../TRAINING_HISTORY.md#17-quan-sát-xgboost-và-lstm-trên-dashboard-local--23112026)

### 9.16. Bài học lớn nhất và bước cải thiện có kiểm soát

**Chất lượng dữ liệu và cách kiểm chứng quyết định độ đáng tin của kết luận.** Những thử nghiệm đã giúp nhóm rút ra các bài học sau.

1. **Dữ liệu cần kiểm tra bằng quy tắc và đối chiếu nguồn.** Việc phát hiện các dòng Volume bằng 0, Close thiếu và nến sai dẫn tới quyết định dùng snapshot Vietcap.
2. **Thêm độ phức tạp cần có bằng chứng.** Bộ 6 feature cho kết quả cạnh tranh trong các đối chứng. Các nhánh LSTM, phân loại và Gap/Body giúp kiểm tra giả thuyết, với mức lợi ích thay đổi theo giai đoạn.
3. **Mỗi kết quả cần baseline và phạm vi đánh giá.** Hit cao cần xem cùng cân bằng lớp. Lợi thế nhỏ cần xem độ ổn định và khoảng tin cậy. Metric luôn đi cùng nguồn, ngày đích, horizon và mốc train.
4. **Một lần thử có giá trị khi trả lời được câu hỏi.** Giữ cả kết quả chưa cải thiện giúp giải thích vì sao dừng mở rộng feature hoặc chọn lại cửa sổ train.
5. **Bước tiếp theo cần thêm bằng chứng mới.** Ghi dự báo trước khi quan sát giá đích, đợi nhãn, chấm sai số và theo dõi drift. Khi có đủ dữ liệu, đối chứng model được retrain với bản đang dùng, rồi mới quyết định thay version và giữ khả năng quay lại bản cũ.

Drift có thể thể hiện qua thay đổi phân bố feature, phân bố lợi suất hoặc sai số tăng trong các giai đoạn mới. Chuỗi ngắn trên một mã có nhiều nhiễu, nên cần xem cả lượng mẫu và mức thay đổi trước khi quyết định retrain. Quy trình này là kế hoạch quản lý model tiếp theo, còn hiện tại dashboard đọc bundle được đóng gói theo version. [Forward ledger và kế hoạch theo dõi](../../ARCHITECTURE.md#forward-ledger)

**Lời gợi ý khi thầy hỏi nhóm học được gì**

> Qua dự án, nhóm học được cách chuyển một câu hỏi về giá cổ phiếu thành bài toán hồi quy có giám sát, tạo dữ liệu nhân quả và kiểm chứng theo thời gian. Nhóm lựa chọn bằng đối chứng với baseline, xem nhiều chỉ số và lưu cả kết quả chưa cải thiện. Ridge6+bias60 là cấu hình nhóm chốt cho demo dựa trên các nghiên cứu đã thực hiện. Bằng chứng tiếp theo sẽ đến từ các dự báo ghi trước ngày đích và kết quả quan sát ở những phiên mới.
