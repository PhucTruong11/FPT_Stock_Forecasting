# Đánh giá hiện trạng và kế hoạch đề xuất cho đồ án ML

> Tài liệu lịch sử: đánh giá trước khi sửa pipeline v2 và dọn repo. Model, biểu đồ, market proxy và hai kế hoạch cũ đã được lưu vào `data/backups/legacy-v1-20261025.zip`; xem README ở thư mục gốc để biết hiện trạng mới nhất.

Ngày đánh giá: 25/10/2026. Phạm vi: source code, ba notebook, CSV và cấu trúc artifact đang có trong workspace; đối chiếu ảnh yêu cầu môn học do người dùng cung cấp.

Đây là đánh giá và đề xuất, chưa thay đổi pipeline, dữ liệu hay mô hình. Các kế hoạch cũ trong repo được xem là tài liệu tham khảo, không phải bằng chứng chức năng đã hoàn thành hoặc yêu cầu được giảng viên xác nhận.

## 1. Định hướng nên chốt

**Giữ FPT, tập trung dự báo một phiên giao dịch, sửa tính đúng đắn của thực nghiệm trước khi mở rộng.** Repo đã có nền dữ liệu, baseline và LSTM cho FPT; đổi mã lúc này chưa giải quyết các lỗi phương pháp đang có.

Tên đề tài gợi ý: **Phân tích dữ liệu và dự báo lợi suất, giá đóng cửa phiên kế tiếp của cổ phiếu FPT bằng học máy; triển khai FastAPI và ứng dụng Streamlit.**

Câu hỏi nghiên cứu chính: Với thông tin đã biết sau khi kết thúc phiên t, các đặc trưng giá và khối lượng có giúp dự báo lợi suất phiên kế tiếp tốt hơn baseline đơn giản, trên dữ liệu ngoài mẫu, hay không?

Không đặt điều kiện phải thắng thị trường hoặc phải đạt Hit Rate cụ thể để đồ án thành công. Một kết quả âm, được kiểm chứng đúng và giải thích rõ, vẫn có giá trị học thuật.

### Lý do chọn mã cần viết lại

Không có cổ phiếu nào hoàn toàn tách khỏi tác động bên ngoài. Không nên bảo vệ lựa chọn FPT bằng các khẳng định như “ít bị tin đồn/thao túng” hoặc “tăng trưởng ổn định nên dễ dự báo” khi chưa có kiểm chứng. Báo cáo thường niên FPT 2025 nêu tác động của bất định địa chính trị, chính sách tiền tệ và việc khách hàng thắt chặt ngân sách IT đối với hoạt động quốc tế. [Nguồn FPT](https://bctn2025.fpt.com/en/business-performance-analysis/technology-sector/it-services-for-foreign-markets/)

Lý do thực tế để tiếp tục FPT là đã có pipeline và dữ liệu để đào sâu; cần kiểm tra thêm tính đầy đủ, sự kiện điều chỉnh giá và chất lượng nguồn. Đề xuất này không phải xếp hạng FPT là mã ít chịu ngoại lực nhất. Nếu giảng viên có tiêu chí loại trừ ngành cụ thể thì cần đối chiếu tiêu chí đó trước khi chốt mã.

## 2. Đối chiếu yêu cầu trong ảnh

Ảnh mô tả ba phần: backend Python có REST API nhận sample và trả JSON; ứng dụng có UI sử dụng API; notebook xây dựng mô hình. Ảnh chưa đủ để khẳng định đây là ba phần bắt buộc chung hay ba lựa chọn độc lập. Kế hoạch dưới đây chủ động bao phủ cả ba bằng một hệ thống thống nhất.

| Phần | Hiện trạng kiểm tra được | Việc cần hoàn thành |
| --- | --- | --- |
| Notebook xây dựng mô hình | Có notebook huấn luyện LSTM và benchmark, script baseline; chưa có notebook EDA trong workspace | Khôi phục EDA, sửa nhãn/cửa sổ/chia tập, chạy và lưu kết quả có thể tái lập |
| REST API Python | Chưa có `src/api/`; FastAPI và Uvicorn chưa có trong requirements | Backend thực sự load artifact, kiểm tra input, predict và trả JSON; có sample gọi API |
| UI sử dụng API | Có demo ipywidgets trong notebook, chưa có `src/app/` | Streamlit gọi HTTP tới API, hiển thị phân tích và dự báo; xử lý lỗi API |

Điểm tốt: đã tách data/features/models, có Linear Regression và XGBoost để đối chiếu, có seed cho LSTM và scaler chỉ fit trên phần train trong notebook 02. Nên giữ các phần này để phát triển tiếp.

## 3. Phát hiện cần ưu tiên

### P0 — Dữ liệu và nhãn chưa thống nhất theo phiên

Kiểm tra CSV hiện có:

| Nguồn | Số dòng | Khoảng ngày | Ghi chú |
| --- | ---: | --- | --- |
| `data/raw/FPT.VN_raw.csv` | 2.084 | 01/01/2018–31/12/2025 | 143 dòng Volume = 0; 142 trong số này có OHLC bằng nhau |
| `data/interim/FPT.VN_features.csv` | 1.883 | 09/03/2018–31/12/2025 | 27 cột bao gồm Date, tức 26 đầu vào số |
| `data/external/market_index.csv` | 2.083 | 02/01/2018–31/12/2025 | Trùng với hai đặc trưng market đã lưu trong features |

Ba CSV không có ngày trùng hoặc ô thiếu. Điều đó chưa đủ chứng minh dữ liệu đúng lịch giao dịch. Dòng Volume = 0 cần được đối chiếu nguồn/lịch, không tự động kết luận toàn bộ là ngày nghỉ hoặc xóa hết.

`build_features.py` tính Volume_Change bằng pct_change; sau một dòng Volume = 0, tỷ lệ có thể thành vô hạn và bị drop. Có 152 dòng mất thêm sau giai đoạn warm-up 49 dòng đầu, không chỉ mất dữ liệu do SMA_50 như tài liệu đang giải thích.

Trên chuỗi còn lại, có 73 chuyển tiếp mà ngày liền trước trong raw khác ngày liền trước trong features. Việc tạo nhãn bằng `Log_Return.shift(-1)` sau khi đã loại dòng làm 71 giá trị nhãn khác `log(Close_next_retained / Close_current)` với ngưỡng so sánh số thực nhỏ. Ví dụ 25/04/2018 được gắn return của 27/04 tính từ giá 26/04, trong khi dòng 26/04 đã bị loại. Cần xây lịch phiên hợp lệ trước, tạo target và target_date trên lịch đó, rồi mới loại những mẫu không đủ feature; không nối lại nhãn qua các dòng đã mất.

### P0 — LSTM thiếu đầu vào ngày dự báo

Notebook 02 tạo `y[i] = Log_Return[i+1]`, nhưng cửa sổ `X[i-lookback:i]` kết thúc ở i-1. Vì vậy, với giả định dự báo sau đóng cửa ngày i, mô hình thiếu toàn bộ thông tin của ngày i. Baseline lại có các feature ngày i.

Đã chạy hàm tạo tensor hiện tại trên chuỗi đánh số: input cuối ở dòng 2 nhưng nhãn là return dòng 4; bài toán một bước đúng phải dùng return dòng 3. Cần dùng cửa sổ kết thúc tại forecast_origin, ví dụ `X[i-lookback+1:i+1]`, ghép đúng nhãn và lưu ngày tương ứng. Với tập validation/test, cho phép dùng lịch sử trước ranh giới làm context, không fit scaler trên tương lai.

### P0 — Benchmark chưa phải test độc lập chung

Các ranh giới dưới đây được tính từ dữ liệu hiện có và logic code, không suy ra từ mô tả README:

| Thành phần | Khoảng ngày theo feature/forecast-origin row |
| --- | --- |
| Train baseline 80% | 09/03/2018–01/07/2024 |
| Test baseline | 02/07/2024–30/12/2025 |
| Train rows LSTM 70% | 09/03/2018–22/09/2023 |
| Validation rows LSTM 15% | 25/09/2023–12/11/2024 |
| Test rows LSTM 15% | 13/11/2024–30/12/2025 |
| Benchmark notebook 03 sau bỏ 60 dòng context | 27/09/2024–30/12/2025, 316 mẫu |

Nếu chạy theo notebook hiện tại, benchmark chứa 33 forecast-origin rows đã thuộc validation dùng chọn checkpoint LSTM. Ngoài ra các mô hình dùng lịch sử huấn luyện khác nhau. Phải thống nhất lịch train/validation/test và cùng forecast_origin, target_date, horizon để so sánh.

Các ngày trên chưa mô tả ranh giới target_date; nhãn shift có thể vượt sang partition kế tiếp. Khi sửa split phải kiểm tra cả ngày kết thúc nhãn, đặc biệt nếu mở rộng horizon 5/20. Chưa có manifest đủ để xác minh các model đã lưu thực tế được train bằng phiên bản code/split nào; không gán phát hiện của code hiện tại thành kết luận chắc chắn về lịch sử huấn luyện artifact cũ.

### P1 — Thước đo và baseline cần sửa

- Random walk trên giá là `predicted_price = current_price`, tương ứng predicted log return bằng 0. Baseline hiện dùng return hôm nay để dự báo return ngày sau là persistence trên return, nên giữ nhưng đặt tên đúng. [Tham khảo phương pháp naïve](https://otexts.com/fpp3/toolbox.html)
- Báo cáo cả zero-return baseline cho hồi quy và always-up/majority-class baseline cho định hướng. Majority class phải chọn từ train, không lấy majority trên test để lựa chọn chiến lược.
- Hit Rate > 50%, kể cả 52–55%, chưa tự chứng minh lợi thế thống kê hay lợi nhuận. Cần số mẫu, lớp tăng/giảm/đi ngang, độ phủ và khoảng bất định. Không gọi Hit Rate lịch sử là độ tin cậy của từng dự báo.
- Code hiện loại ngày thực tế đứng giá và dự báo bằng 0 khỏi Hit Rate. Chính sách này phải công khai; báo số mẫu bị loại, không mặc nhiên so sánh các mô hình trên các mẫu khác nhau. Zero-return baseline được đánh giá bằng sai số hồi quy, không dùng directional score sau khi mask hết.
- Lookback ablation trong notebook 02 đang báo kết quả trên test cho từng lựa chọn, dễ dẫn tới chọn cấu hình theo test. Chọn lookback trên validation; context khác nhau phải được chấm trên cùng các ngày.

### P1 — Đặc trưng và nguồn market chưa đúng với mô tả

- Features vẫn có OHLC, SMA, Bollinger Bands, ATR và OBV dạng tuyệt đối. Chuyển sang ratio/log-return là cải tiến hợp lý, nhưng không bảo đảm dữ liệu trở thành stationary. Giá test nằm ngoài [0, 1] sau MinMaxScaler không tự nó là leakage.
- Có `.bfill()` trên dữ liệu gốc và market: có thể đưa tương lai vào quá khứ khi có ô thiếu. CSV đang đọc không có ô thiếu nên chưa chứng minh lỗi này đã phát sinh trong dataset hiện tại. Cần bỏ backfill, sort trước xử lý và quy định xử lý thiếu theo thời gian.
- Downloader tạo proxy từ 6 mã, có cả FPT, nhưng đặt tên “VN30”. Comment ở builder lại nói index từ 69 mã; không có thành phần/trọng số/nguồn gốc đủ để kiểm chứng bản CSV hiện tại. Không gọi proxy này là chỉ số VN30 chính thức.
- Nếu giữ proxy: ghi rõ thành phần, trọng số, ngày quan sát và số mã đóng góp từng ngày. Có FPT trong proxy cùng ngày không mặc nhiên là future leakage, nhưng làm phân tích đóng góp market kém độc lập. Nếu dùng dữ liệu index có nguồn phù hợp, đồng bộ lịch và thời điểm dữ liệu sẵn có.

### P1 — Demo và backtest chưa sẵn sàng

- Notebook 04 tham chiếu `TOP_10_TICKERS` chưa được định nghĩa; lỗi NameError khi chạy kernel sạch.
- Hiện code chỉ có dự báo T+1. Phần giới thiệu nói 1/5/20 phiên và rollout nhưng chưa có implementation tương ứng.
- Backtest dùng `(1 + log_return).cumprod()`; với log return phải dùng `exp(cumsum(log_return))`, hoặc chuyển sang simple return bằng expm1 trước khi tính toán.
- `True_Return` đã là lợi suất tương lai từ origin t. Dịch Position thêm một hàng có thể lệch kỳ hưởng lợi suất. Cần định nghĩa rõ thời điểm có tín hiệu, giá khớp lệnh và khoảng nắm giữ, không dịch máy móc.
- Chưa tính phí, slippage, hạn chế khớp lệnh hoặc settlement. Các mức phí/thuế trong tài liệu cũ là giả định, chưa phải backtest đã thực hiện hoặc quy định được kiểm chứng trong lần review này.
- Giá dự báo t+1 đang vẽ ở ngày t. “Giá hiện tại” lấy từ frame có nhãn nên bỏ dòng mới nhất; dữ liệu cuối cùng là 31/12/2025, không phải giá trực tiếp ngày review 25/10/2026.
- OHLC hiện được downloader cấu hình auto_adjust=True. Cần ghi rõ giá điều chỉnh; không đồng nhất với giá khớp lệnh thực tế hoặc tự coi đó là số tiền có thể mua/bán. Muốn mô phỏng giao dịch thật phải quản lý riêng raw prices và corporate actions.

### P2 — Tái lập và tài liệu

Cả ba notebook đang không lưu execution_count/output. Có model và ảnh trong thư mục, nhưng chưa có log/manifest đủ để xác nhận kết quả tái lập. Requirements chỉ dùng cận dưới và thiếu dependencies cho API/UI. Có hai bộ tên model/scaler tương đương, dễ load nhầm. README mô tả API, GRU, direct multi-horizon, backtest có phí và features chuẩn hóa như đã hoàn tất trong khi code chưa có.

## 4. Phạm vi triển khai đề xuất

### Bản hoàn chỉnh cần có

1. Một mã FPT, snapshot dữ liệu có nguồn và thời điểm tải rõ ràng.
2. EDA về missing sessions, zero volume, sự kiện điều chỉnh, phân phối return và các giai đoạn biến động. EDA phục vụ chọn model dùng train; phân tích test sau khi khóa cấu hình được ghi là phân tích hậu nghiệm.
3. Nhãn chính `y_t = log(P[t+1] / P[t])`, tính theo lịch phiên đã xác minh. Dự báo sau đóng cửa t; quy đổi `P_hat[t+1] = P[t] * exp(y_hat_t)`. Không gọi giá quy đổi này là kỳ vọng số học nếu chưa xử lý bias do phép biến đổi.
4. Zero-return, persistence-return, Ridge hoặc Linear Regression, XGBoost; thêm một LSTM nhỏ làm đối chứng học sâu. Không mặc định LSTM là model đưa lên API.
5. Đánh giá ngoài mẫu chung: MAE/RMSE trên return, MAE trên giá quy đổi cùng chuẩn điều chỉnh, directional accuracy với chính sách neutral cố định; phân tích lỗi và so sánh baseline.
6. FastAPI nhận sample, trả JSON; Streamlit dùng API và hiển thị ngày dữ liệu, model version, dự báo và kết quả ngoài mẫu.

### Phần thêm khi còn thời gian

Ưu tiên một ablation có ý nghĩa: nhóm return/volume cơ bản so với thêm technical indicators, rồi mới thêm market context. Dùng cùng split và cùng ngân sách tuning.

Sau khi bản một phiên hoàn chỉnh mới thêm dự báo trực tiếp 5 phiên. Dự báo 20 phiên, GRU, ensemble, Optuna lớn, news/sentiment và backtest giao dịch thực tế để sau. Đây là lựa chọn cắt phạm vi, không phải thiếu sót bắt buộc theo ảnh yêu cầu.

## 5. Thiết kế thực nghiệm

- Split theo ngày cố định: phát triển trên 2018–2024, dành 2025 làm final holdout theo kế hoạch mới. Trong development có thể dùng expanding-window với các validation block 2022, 2023, 2024. Tất cả lựa chọn feature/model/threshold đều thực hiện tại đây.
- Năm 2025 đã xuất hiện trong code benchmark cũ, nên không gọi đó là “chưa từng xem” nếu nhóm đã dùng kết quả để tuning. Ghi rõ đây là holdout được khóa cho pipeline sửa lại; nếu cần xác nhận hoàn toàn mới, dùng một giai đoạn sau đó chưa dùng để quyết định mô hình, sau khi kiểm tra khả năng lấy dữ liệu.
- Partition bằng forecast_origin và target_date, loại train sample nếu nhãn đi vào validation/test. Với horizon h, dùng gap phù hợp h và lịch phiên thực tế; không lấy lookback làm quy tắc gap duy nhất. Context lịch sử qua ranh giới là hợp lệ khi đã biết tại thời điểm dự báo. [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- Fit scaler/imputer/feature selection riêng trong từng fold. Không thực hiện selection trên toàn dataset rồi mới chia tập.
- Khởi đầu khoảng 10–15 đặc trưng: return và lag 1/2/5, rolling volatility 5/20, SMA distance 20/50, normalized ATR, RSI, volume ratio/z-score. Số lượng cuối cùng do validation quyết định; không đếm Date là feature.
- LSTM khởi đầu một lớp 32 units, lookback 10/30/60 chọn theo validation; số epoch giới hạn bằng EarlyStopping. Huber là một lựa chọn cần đánh giá, không tự bảo đảm cải thiện.
- Chọn model bằng sai số validation so baseline và độ ổn định giữa các fold; test cuối chỉ báo cáo sau khi khóa cấu hình. Có thể report skill = 1 - MSE_model/MSE_zero_return; skill âm là kém baseline.
- Nếu có khoảng bất định, hiệu chỉnh bằng dữ liệu trước test và báo coverage ngoài mẫu. Nếu chưa kiểm chứng calibration thì không hiển thị nhãn “95% confidence”. Kiểm định nâng cao/block bootstrap là phần thêm; không coi binomial p-value mặc định là đủ khi sai số có phụ thuộc thời gian hoặc đã thử nhiều cấu hình.

## 6. Kiến trúc và sản phẩm bàn giao

Luồng chính: **snapshot dữ liệu → kiểm tra lịch/nhãn → features dùng chung → train/evaluate → artifact đã chọn → FastAPI → Streamlit**.

Các module nên có: `src/data`, `src/features`, `src/models/train.py`, `src/models/predict.py`, `src/evaluation`, `src/api/main.py`, `src/app/streamlit_app.py`; một config chung cho ticker/horizon/split/features. Notebook gọi các module này, tránh chép lại ba pipeline khác nhau.

Notebook đề xuất:

- `01_eda_fpt.ipynb`: nguồn dữ liệu, lịch phiên, chất lượng và EDA.
- `02_train_fpt.ipynb`: nhãn, split, baselines và các mô hình học.
- `03_evaluate_fpt.ipynb`: bảng ngoài mẫu, error analysis, ablation và giới hạn.

Artifact mỗi phiên bản: model, scaler nếu có, danh sách/thứ tự features, lookback, horizon, target definition, training cutoff, data hash, phiên bản thư viện, seed và validation metrics. Sau khi sửa nhãn/cửa sổ/features phải train lại; không tiếp tục dùng weight cũ để tuyên bố kết quả của pipeline mới.

API tối thiểu:

- `GET /health`: sẵn sàng, model version và ngày dữ liệu.
- `POST /predict`: sample OHLCV có ngày, đủ lịch sử theo feature/lookback đã chọn, nhất quán chuẩn giá; kiểm tra sort/duplicate/thiếu dữ liệu. Trả ticker, as_of, horizon_sessions=1, predicted_log_return, predicted_close_adjusted, price_basis và model_version. Có file sample request đi kèm để thầy thử.
- `GET /metrics`: chỉ số test đã lưu, gắn khoảng đánh giá và chính sách neutral; tách khỏi confidence của từng prediction.

MVP có thể chạy hoàn toàn từ snapshot; không phụ thuộc tải Yahoo trực tiếp mỗi lần bấm. Nếu cho phép replay as_of quá khứ phải dùng prediction ngoài mẫu đã lưu hoặc model chỉ train tới thời điểm đó, tránh dùng model học từ tương lai để demo như dự báo thật.

UI gồm phân tích lịch sử, dự báo một phiên, bảng đối chiếu ngoài mẫu và thông tin ngày/model. Một nút dự báo phải tạo HTTP request thật tới backend. Không cần biến đồ án thành ứng dụng khuyến nghị mua bán. Nếu giữ backtest minh họa, ghi đúng giả định và giới hạn thực thi.

## 7. Lộ trình tham khảo 10–14 ngày làm việc

Đây là ước lượng tổ chức công việc, không phải deadline hay số thành viên đã được người dùng xác nhận.

| Chặng | Công việc | Tiêu chí xong |
| --- | --- | --- |
| Ngày 1–3 | Xác minh lịch/zero volume; sửa target, sequence, split; thêm EDA | Kiểm tra được origin/target_date; không backfill; ghi lý do giữ/loại từng loại dòng |
| Ngày 4–6 | Chạy baseline, Ridge/LR, XGBoost; thử LSTM nhỏ | Chung các ngày đánh giá; tuning chỉ trên development; lưu config và metrics |
| Ngày 7–9 | Đóng gói model tốt nhất; viết API và nối Streamlit | Sample → API → JSON → UI chạy trên kernel/process sạch |
| Ngày 10–12 | Khóa model, final evaluation, error analysis, ablation | Có bảng kết quả từ code và giải thích khi kém baseline |
| Ngày 13–14 | README, dependency lock, báo cáo/slides và tổng duyệt | Máy/môi trường sạch chạy theo hướng dẫn; demo không cần notebook state cũ |

Nếu nhóm đúng là bốn người như Team_Workflow cũ: người 1 data/EDA/features; người 2 train/tuning/artifact; người 3 evaluation/API; người 4 UI/tích hợp/báo cáo. Thống nhất schema dữ liệu và JSON ngay từ đầu để các phần làm song song không lệch nhau.

Kiểm tra đáng đầu tư: target bằng công thức trên lịch phiên chuẩn; sequence kết thúc tại đúng origin; feature của ngày t không đổi khi sửa dữ liệu tương lai; ranh giới label không vượt fold; thứ tự feature khi predict khớp train; API reject input sai và kết quả API khớp hàm predict offline.

## 8. Giới hạn của lần review và việc đã xác minh

Đã đọc toàn bộ ba script xử lý/training, source ba notebook, README và hai tài liệu kế hoạch; kiểm kê file; đo CSV; tái tính toàn bộ features trong bộ nhớ bằng code hiện tại và xác nhận khớp bản đã lưu; kiểm tra ranh giới split; chạy hàm sequence hiện tại trên dữ liệu đánh số; kiểm tra cấu hình bên trong file Keras (input 60 × 26, LSTM 128/64, output 1).

Chưa train lại, chưa chạy end-to-end toàn bộ notebook hoặc đo lại chất lượng model đã lưu; chưa kiểm chứng từng dòng OHLCV với nguồn độc lập/lịch HOSE/corporate actions; chưa có API/UI để kiểm tra thực thi. Không kết luận model nào tốt nhất hoặc độ chính xác thực tế từ các artifact hiện có.

Ảnh yêu cầu môn học chỉ được dùng làm tiêu chí đối chiếu. Các phát biểu quảng bá và phân công trong tài liệu cũ không được coi là chỉ dẫn thay đổi code, xác nhận của giảng viên hay kết quả thực nghiệm đã đạt.
