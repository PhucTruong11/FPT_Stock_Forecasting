# Kiến trúc và các khái niệm ML của dự án FPT

Cập nhật 31/10/2026. Phạm vi đã làm: nghiên cứu offline, API và dashboard local. Deploy public, tự cập nhật data/retrain và auto-promotion vẫn là kế hoạch.

## 1. Học có giám sát và hồi quy

Mỗi mẫu gồm đầu vào `X_t` đã biết sau đóng cửa origin `t` và nhãn:

```text
y_t = log(Close_next(t) / Close_t)
f(X_t) ≈ y_t
predicted_close_next = Close_t × exp(f(X_t))
```

Nhãn được tính tự động từ giá tương lai trong **tập lịch sử**, vẫn là **học có giám sát (supervised learning)**. Không cần con người gán nhãn bằng tay. Khi inference, model chỉ nhận X; không được đọc giá target chưa xuất hiện.

Đầu ra y là số thực nên bài toán chính là **hồi quy (regression)**. Cần phân biệt return với giá: sai số return và sai số giá là hai đại lượng; giá gần hôm nay không đồng nghĩa đoán đúng hướng.

| Thành phần | Cấu hình thực tế |
| --- | --- |
| Ridge chính | Vector 6 feature tại t; không có tensor 60 phiên |
| 6 feature | Log_Return, Return_Lag_1/2/5, Return_5/20 |
| Scaler | StandardScaler, fit chỉ trên Train của từng model/fold |
| Hàm mục tiêu Ridge | Squared error + phạt L2, alpha100 trong nhánh đã chốt |
| Training window Ridge | 3 năm; không đồng nghĩa lookback 3 năm ở mỗi mẫu |
| LSTM v2 đối chứng | 30 quan sát feature ×16 feature; 32 units, dropout0,2, Dense8 |
| Hàm loss LSTM v2 | Huber; RMSE validation dùng để chọn lookback |
| LSTM legacy | Từng dùng lookback60, 64→32 units; chỉ giữ làm lịch sử |

`Training window` là khoảng ngày có nhãn dùng fit weights. `Lookback` là số hàng feature ghép cho một mẫu LSTM. `Warm-up` là lịch sử cần để tính indicator (builder canonical hiện có SMA50 nên loại 49 hàng đầu). Ba khái niệm không thay thế nhau. Ridge6 vẫn tái dùng builder16 hiện tại để giữ đúng ngày/feature đã kiểm chứng; chưa tối ưu warm-up xuống chỉ vì bỏ technical features.

## 2. Phân loại và học không giám sát

Nếu đổi nhãn thành dấu của return, đây là **phân loại có giám sát**. Hai classifier Logistic/XGBoost đã được thử, kèm calibration và margin để trả `tang`, `giam`, `chua_du_tin_hieu`. Nhánh này chưa cải thiện ổn định nên chưa được bật trong API chính. Dấu của dự báo Ridge không phải xác suất tăng/giảm đã được calibration.

Clustering/PCA/HMM là các cách có thể mô hình hóa cấu trúc hoặc trạng thái ẩn mà không dùng nhãn return như hồi quy. Chúng không tự cho ra dự báo return chỉ vì đã fit; cần thêm giả định/mô hình liên hệ trạng thái với return. Có thể thử regime biến động cao/thấp hoặc xu hướng/đi ngang làm feature phụ, nhưng **chưa triển khai**. Fit cluster/PCA/HMM trên Train; áp dụng tham số đã khóa cho Val/Test. Với HMM, không dùng smoothing dựa vào quan sát tương lai để tạo feature tại t.

## 3. Dữ liệu và chỉ một mã FPT

Yahoo/yfinance là nguồn của pipeline v2 cũ (`FPT.VN`, `auto_adjust=True`). Sau audit, nghiên cứu mới dùng snapshot Vietcap FPT; KBS kiểm tra chéo. Demo local đọc snapshot Vietcap đóng gói, không gọi nhà cung cấp trong từng request.

Khoảng vài nghìn quan sát ngày không có nghĩa vài nghìn mẫu độc lập. Rolling returns và chuỗi LSTM chồng lấp, số mẫu hữu hiệu thấp hơn. Mạng lớn dễ overfit; không mặc định tăng layer/epoch sẽ tốt hơn Ridge. V2 Yahoo cụ thể có 1.087 Train,493 Val,248 Test; số đó **không áp dụng** cho mọi fold Vietcap hoặc tập2026. Xem `phase_counts.csv`/manifest mỗi study để lấy số chính xác.

Chỉ FPT được dùng làm feature. Không có gộp mã, embedding ID, ablation đa mã hoặc IC xếp hạng chéo mã. Nếu bổ sung IC, dùng tương quan Spearman giữa prediction và actual **theo thời gian trên cùng các origin**, ghi rõ n và sự phụ thuộc thời gian; hiện chưa phải KPI chính của demo.

Không tạo lịch bằng forward-fill ngày lễ. Zero-volume bị tách để review, không tự chứng nhận đó là ngày nghỉ. Nến bất hợp lệ và nhãn chạm nến bất hợp lệ được quarantine. `next(t)` hiện là quan sát positive-volume hợp lệ kế tiếp theo hợp đồng nguồn; `calendar_verified=false`.

Provider-adjusted history có thể thay đổi khi nhà cung cấp back-adjust. So giá phải cùng basis của origin/target; không lấy giá origin từ snapshot cũ nhân return rồi so giá target thuộc snapshot mới mà bỏ qua điều chỉnh. Hash xác nhận bytes, không chứng nhận dữ liệu point-in-time.

## 4. Chia thời gian, nhãn và walk-forward

```mermaid
flowchart LR
    R[Raw có version] --> C[Validate và quarantine]
    C --> X[Feature chỉ dùng quá khứ/current]
    C --> Y[Nhãn từ next close lịch sử]
    X --> S[Chia theo origin và target date]
    Y --> S
    S --> TR[Train: fit scaler và model]
    TR --> V[Validation: chọn cấu hình/ngưỡng]
    V --> E[Evaluation sau cutoff: chỉ chấm]
    E --> D[Metric + dự báo từng ngày + quyết định]
```

Không random split chuỗi thời gian. Nhãn đi qua ranh giới phase phải bị loại: origin thuộc Train nhưng target thuộc Val không được dùng để fit Train. Context lịch sử có thể nằm trước ranh giới để dự báo tại một origin mới, vì dữ liệu ấy đã biết tại origin.

Walk-forward lặp nhiều cutoff train → validation → evaluation để thấy chất lượng khác nhau theo giai đoạn. Model mỗi fold có weights/scaler riêng. Evaluation đã xem nhiều lần không còn là holdout mới; các study2022–2026 hiện là nghiên cứu hồi cứu có điều chỉnh theo kết quả trước. Không chọn tháng/năm đẹp rồi bỏ phần còn lại.

Bản chính chỉ có **h=1**. `h=5` là đề xuất thử riêng, cần định nghĩa return5 quan sát, purge nhãn qua ranh giới và xử lý target chồng lấp. Horizon khoảng một tháng với rất ít nhãn không chồng lấp khó đủ bằng chứng; không khẳng định sẵn một số lượng mẫu nếu chưa tính theo cohort.

## 5. Bundle local và model lịch sử

```mermaid
flowchart TB
    SH[Shadow Ridge đã fit đến 29/10/2026] --> PK[Đóng gói model/scaler + calibration + raw + hash]
    PK --> LP[models/local_demo.json]
    LP --> SV[LocalRidgeService]
    SV --> API[FastAPI trên 127.0.0.1:8000]
    API --> UI[Streamlit trên 127.0.0.1:8501]
    OLD[Forecast lịch sử Ridge/LSTM đã lưu] --> BT[Backtest cùng ngày đích]
    BT --> API
```

- `models/latest_experiment.json` và `reference_experiment.json` giữ reference20261025 cũ để tái lập, không bị đổi khi chạy demo mới.
- `models/local_demo.json` chọn bundle Ridge riêng. Manifest kiểm tra hash trước khi load; chỉ load artifact do project tạo, không nhận pickle upload từ API.
- `ridge.pkl` đã chứa Pipeline StandardScaler→Ridge; không fit scaler trong request.
- `Ridge+bias60` cộng mean của tối đa60 sai số đã có nhãn trong bundle. Calibration cố định, không tự học mỗi lần bấm.
- Khoảng residual dùng phân vị10%–90% của60 sai số corrected đã biết. Đây là khoảng thực nghiệm có nominal80%, không phải độ tin cậy chính xác được bảo đảm.
- Input mặc định là snapshot; custom input chỉ nhận ngày đã qua, origin không trước cutoff training/calibration, nến cuối không được âm thầm bỏ rồi dự báo ngày cũ.
- API không tự train, download, ghi ledger hoặc promote. Dashboard chỉ nhận JSON từ API.

Backtest năm2026 dùng model train trước các target tương ứng. Bản Ridge train đến29/10/2026 **không** được dùng chấm từ tháng1–10 như dự báo ngoài mẫu. Tên `Ridge_2025+bias60` trên dashboard chỉ model lịch sử; khác bundle `ridge6_20261029` đang inference. LSTM lịch sử có context Yahoo, được re-anchor giá và chấm cùng nhãn Vietcap trong cohort104 cặp; không coi đây là thí nghiệm chỉ khác architecture.

## 6. Đọc metric đúng

| Metric | Ý nghĩa và giới hạn |
| --- | --- |
| RMSE/MAE log return | Sai số numeric; thấp hơn tốt hơn; không phải xác suất dự báo đúng |
| MAE giá adjusted | Sai lệch VND trên cùng basis giá; phụ thuộc mức giá |
| Hit rate | Đúng dấu trên actual khác0; vẫn giữ prediction0 trong mẫu số khi có tín hiệu |
| Balanced accuracy | Trung bình recall tăng/giảm; giảm ảnh hưởng mất cân bằng lớp |
| Coverage | Tỷ lệ có phát tín hiệu hướng; cần đọc cùng hit khi có abstain |
| MSE skill vs zero | 1−MSE_model/MSE_zero trên cùng ngày; số âm là kém zero |
| CI block-bootstrap | Mô tả bất định có xét block; không xóa hết adaptive research/multiple testing |

Zero Return không phát hướng nên Hit/Balanced là N/A; không dùng50% giả. “Luôn giảm” có thể đạt hit cao vì tập có nhiều ngày giảm. Trên202 cặp2026 có80 tăng,117 giảm,5 đứng giá: luôn giảm đạt117/197=59,39% Hit, balanced50%.

## 7. Giám sát và cải thiện trong tương lai

```mermaid
flowchart TD
    A[Forecast phát trước target, ghi version/input hash] --> B[Chờ nhãn thực tế]
    B --> C[Rolling error + baseline cùng ngày]
    C --> D{Có suy giảm đủ mẫu?}
    D -->|Có| E[Kiểm tra data lỗi, drift, thay đổi nguồn]
    E --> F[Retrain candidate offline, khóa giao thức]
    F --> G[Shadow cùng incumbent trên dữ liệu mới]
    G --> H{Đạt acceptance gate đã định trước?}
    H -->|Có| I[Đổi version có kiểm soát, giữ rollback]
    H -->|Không| A
    D -->|Không| A
    I --> A
```

Input drift là thay đổi phân phối feature, chưa tự chứng minh model hỏng. Performance/concept drift cần nhãn để kiểm tra quan hệ X→y và sai số. Catastrophic forgetting thường liên quan cập nhật/online learning; pipeline batch hiện tại chưa có bằng chứng đó. Không gọi mọi điểm kém là “quên”.

Ledger shadow hiện có một forecast pending; không sửa thời điểm hoặc viết lại quá khứ. Bấm dự báo UI không tự tạo evidence forward. Gate tự thay model, lịch retrain, API public/HTTPS/auth, Docker và cloud hosting đều **chưa triển khai**. Lượt hiện tại chỉ chạy local; lựa chọn nhà cung cấp và chi phí sẽ đánh giá sau.

## 8. Tái lập và bàn giao

Mỗi lượt mới cần protocol/config, nguồn có hash, ngày origin/target, scaler/model version, predictions, metrics, decision và cả failure log. Đóng gói phục vụ là một sự kiện khác training: lượt local này có **0 fit mới**. Không dùng thời gian tạo file thay thời gian model được train.

Xem [TRAINING_HISTORY.md](TRAINING_HISTORY.md), [README](reports/training_history/research_notes.json), [forward ledger](ARCHITECTURE.md#forward-ledger). Các kết quả và source snapshot cũ giữ nguyên giá trị lịch sử; tài liệu current này quyết định cách hiểu bản local hiện tại.

## 9. Bổ sung Forecast Studio — vẫn chỉ model h=1 đã train

`src/models/forecast_path.py` thêm recursive scenario cho hôm nay/ngày mai/tuần/tháng từ Close, không sinh futureOHLCV. Tối đa23bướcweekday, không chứng nhận lịchHOSE, không fit hoặc ghiledger. VùngP10–P90 từ512circularblockbootstrap paths chỉ là vùng kịch bản, không guaranteecoverage. `/api/v1/forecast` tách khỏi `/predict` để giữ hợp đồng một bước cũ.

Ridge raw/bias là hai cấu hình cùng model. Tương lai chỉ giữ thêm LSTM/XGBoost làm challenger nếu có lợi ích kiểm chứng ngoài mẫu; chưa chọn model thứhai hoặc ensemble. [Định nghĩa đầy đủ và cách kiểm chứng](ARCHITECTURE.md#forecast-studio).

## Forecast Studio

D1 là dữ liệu một ngày giao dịch (gộp sáng và chiều). h=1 là một quan sát kế tiếp; trên D1 tương ứng một phiên ngày, không phải nến H1 một giờ.

| Lựa chọn trên UI | Ngày yêu cầu |
| --- | --- |
| Hôm nay | Ngày local hiện tại; cuối tuần dời sang weekday kế tiếp |
| Ngày mai | Ngày local +1; cuối tuần dời sang weekday kế tiếp |
| Tuần | 7 ngày lịch tính cả hôm nay; ngày cuối lấy weekday trong khung |
| Tháng | Đến trước cùng ngày tháng sau; ngày cuối lấy weekday trong khung |
| Chọn ngày | Ngày đích từ hôm nay tới một tháng sau; cuối tuần dời sang weekday kế tiếp, vẫn tối đa23bước từ snapshot |

Tính toàn bộ bước từ origin thực tế đến target, tối đa 23 weekday; quá giới hạn thì từ chối để cập nhật nguồn. Lịch chưa trừ ngày lễ HOSE. Model chỉ học h=1: bước sau tính lại Log_Return, lag1/2/5, return5/20 từ 21 Close cuối, gồm các Close dự báo. Không sinh OHLCV giả, không dùng nhãn tương lai, không fit lại.

`ForecastRequest` chỉ nhận `target_date` khi `horizon=custom` và bắt buộc có ngày ở chế độ này. Backend kiểm tra ngày không có time/timezone, nằm từ local hôm nay tới `today+DateOffset(months=1)` và không vượt23bước. `/health.forecast_date_limits` cung cấp min/max cho lịch UI; frontend không tự suy khác múi giờ. `in_requested_window` tách đoạn nối snapshot khỏi khoảng người dùng yêu cầu, kèm `bridge_steps_before_requested_window`; cả hai vẫn là forecast kể từ dữ liệu thật cuối cùng. API từ chối thay vì tạo giá hiện tại còn thiếu.

`market_analysis.py` chỉ chuẩn bị OHLCV hợp lệ và tính mô tả bằng pandas/numpy; `analysis_view.py` render trong Tổng quan, không tạo tab phương pháp/lộ trình mới. Lọc volume không dương/nến lỗi có đếm số hàng, không forward-fill. SMA/Bollinger, RSI kiểu Wilder, MACD/Signal, volume so20quan sát trước, ATR và volatility252quan sát/năm đều chỉ dùng quá khứ. Tính indicator trước khi cắt khung hiển thị; hiệu suất/drawdown trong khung dùng Close đầu/cuối và đỉnh lũy kế của chính khung. Heatmap lấy Close tháng cuối quan sát được, không điền tháng thiếu và nêu tháng cuối có thể chưa đủ. Các chỉ báo này không đi vào Pipeline Ridge, không biểu diễn xác suất tăng/giảm, không thay kết quả backtest.

`recursive-ridge6-v4`: bias chỉ cộng bước 1; các bước sau dùng Ridge thô trên context dự báo. Đường điểm giữ nguyên từ v2, không gọi là giá kỳ vọng trung bình sau biến đổi log. Vùng P10–P90 dùng 512 đường residual circular block-bootstrap, block5, seed20261030. Raw Ridge dùng raw errors, Ridge+bias dùng corrected errors đã biết; lấy 60 residual cuối và **trừ trung bình pool trước khi lấy mẫu**. Không center từng block/đường vì sẽ làm giảm giả tạo biến động tích lũy. Calibration/model/bias vẫn đóng băng; chỉ đổi cách sử dụng residual, không ghi lại bundle.

Centering giả định trung bình residual không phải drift tương lai bền vững; không bảo đảm coverage80% hoặc coverage đồng thời cho cả đường. Bộ đối chứng5/21 trước và5/22 sau dùng model lịch sử train đến cuối2025, chỉ dùng nhãn trưởng thành tại từng origin. Lượt5/22 dựng một `LocalRidgeService` lịch sử có raw/features bị cắt tới origin, dùng lại chính `predict` và `forecast`. `observation_dates` là tham số nghiên cứu chứa ngày, không chứa giá tương lai và không xuất hiện trong schema request HTTP. Đây là hồi cứu2026 đã xem, không kiểm chứng bundle cuối10/2026 hoặc model direct nhiều bước. Giao diện tuần/tháng dùng khung lịch nên không đồng nhất với đúng5/22quan sát. Kết quả và độ nhạy không chồng lấp ở [TRAINING_HISTORY.md](TRAINING_HISTORY.md). Zero Return giữ giá gốc, không mượn band của Ridge. Ước tính cùng ngày không được xem là forecast đã phát trước phiên.

V4 đọc [recursive_band_calibration.json](configs/recursive_band_calibration.json). Chỉ h5/h22 có factor riêng; h1 và mọi h chưa liệt kê giữ1×, không nội suy. `widen_band` nhân khoảng cách từ point tới lower/upper trên log-price scale, không nhân mức giá. Nếu factor>1 thì bounds đã hiệu chỉnh không còn là literal percentiles P10/P90; metadata tách base quantiles và factor. Policy kiểm tra cutoff target≤origin và version model, baseline không có band. **Hiện mọi factor active=1**; các candidate vẫn lưu để xem, không được dùng để làm đẹp coverage quá khứ.

Calibration cố định dùng target≤30/06 rồi chấm62origin từ01/07; hệ số chọn được đều1. Rolling calibration dùng tối đa60 forecast nhiều bước đã có target, tối thiểu20. H5 tăng coverage nhưng làm interval score xấu hơn; h22 chưa có forecast đã chấm thực sự nới band. Review từ chối hệ số chưa có ít nhất20 forecast mở rộng được chấm hoặc nằm ngoài vùng đã thử. Candidate1,47× chỉ được biết sau khi nhãn cuối kỳ trưởng thành, chưa được xem là hệ số đã kiểm chứng. Policy hiện đóng băng, không tự cập nhật khi gọi API; chuyển hệ số của model2025 sang model đang phục vụ cũng chưa có bằng chứng forward. Runner mặc định chỉ xuất evidence mới; cài policy cần cờ riêng `--install-policy`, không có auto-promote model.

Chỉ một Ridge chính là đủ nếu model thứ hai chưa thêm giá trị. Raw/bias là hai cấu hình của cùng model. LSTM/XGBoost chỉ làm challenger nếu tốt hơn ổn định hoặc bổ sung lỗi trên dữ liệu phù hợp; ensemble chọn trọng số bằng validation, không sửa theo test.

## Forward ledger

Công cụ chạy thủ công, không nối tự động với nút dự báo UI. Giữ bundle cố định, dùng CSV Vietcap full-history đã đóng cửa và cùng basis adjusted. Lưu forecast trước khi biết target; không backdate hoặc ghi đè.

```powershell
$bundle = 'reports/recency_bias/20261030T022440928507Z/shadow_bundle.json'
$ledger = 'reports/forward_shadow/forecasts.jsonl'
$inputCsv = '.cache/forward/vietcap_full_history.csv'
.\venv\Scripts\python.exe -m src.models.forward_ledger issue --bundle $bundle --input $inputCsv --ledger $ledger
.\venv\Scripts\python.exe -m src.models.forward_ledger score --input $inputCsv --ledger $ledger
.\venv\Scripts\python.exe -m src.models.forward_ledger verify --ledger $ledger
```

CSV trong ví dụ phải được chuẩn bị riêng; CLI không download. `issue` cùng ngày origin chặn trước16:00 giờ Việt Nam theo quy ước bảo thủ của project, không phải chứng nhận source final. `score` chỉ xử lý target có thật trước ngày đang chạy; chỉ `issued_local_date < actual_target_date` mới đủ điều kiện full-session. Dùng cả origin/target cùng snapshot mới khi nguồn back-adjust giá, giữ prediction return đã phát và adjustment ratio. Hash-chain không phải chứng nhận thời gian bên thứ ba; chỉ dùng một writer.

Bundle shadow và calibration cố định đến khi chủ động rebuild. `python -m src.models.recency_bias` với config cũ chỉ tái lập nghiên cứu, không tự tải data tới hôm chạy. Ledger hiện có một forecast pending ngày30/10; nếu target cũng30/10 thì không đủ điều kiện full-session. Chưa có dữ liệu forward đủ để kết luận model tốt hơn.

## Quy ước repo sau dọn

Chỉ ba Markdown gốc là tài liệu current. `docs/assets/` chỉ giữ ảnh được dùng. Evidence nghiên cứu có protocol/hash, CSV/JSON, source/input ZIP và notebook outputs. Diễn giải chi tiết gốc ở `reports/training_history/research_notes.json`; `src.models.research_notes` đọc/ghi và kiểm tra hash, tương thích diagnostic hash của study cũ. Không tạo thêm REPORT/DECISION Markdown khi chạy report generator.

Các artifact nguồn/model có manifest byte-hash vẫn giữ nguyên, không đổi kiến trúc/scaler/feature hoặc fit model trong lượt cleanup. Các report script và notebook nghiên cứu vẫn được giữ để tái lập; chúng không được import khi chạy API nhẹ. Deploy public, nguồn cập nhật, direct h=5/h=21 và auto-retrain là việc tương lai.

## Phạm vi giao diện và notebook hiện hành

Streamlit chỉ render bốn mục phân tích/dự báo: Tổng quan, Dự báo, Đối chứng, Hành trình model. Phương pháp và lộ trình chỉ nằm trong tài liệu gốc, không lặp thành trang UI. API data-quality vẫn giữ làm hợp đồng kiểm tra dữ liệu, không cần một tab riêng.

Notebook01 cho EDA,02 cho training baseline/LSTM lịch sử,05 cho toàn bộ hành trình và các nhánh không được chọn,06 cho giá thực tế so forecast. Phần classifier/gap ở05 chỉ đọc kết quả đã lưu. Generator tập trung `training_journey_notebook.py` thay hai generator notebook riêng; train script mới đưa generator chung vào source snapshot. Snapshot cũ giữ nguyên nội dung/hashes.

Trang Dự báo dùng `src/app/forecast_explainer.py` để vẽ ba sơ đồ từ response API: pipeline 21 Close → 6 return feature → scaler/Ridge → giá; vòng lặp recursive (bias chỉ bước 1); residual block bootstrap và phân vị từng bước. Module chỉ trình bày response, không nạp weights, gọi model hoặc fit; nhánh Zero Return mô tả giả định giá không đổi và không có band Ridge.

`src/app/forecast_analytics.py` chuyển response forecast thành ba biểu đồ, không gọi inference thêm: simple return từng bước = `expm1(step_log_return) × 100`; tích lũy và band so với origin; độ rộng band = `(upper − lower) / point × 100`. Trục ngang dùng các ngày dự kiến rời rạc, không tạo điểm intraday. Zero Return không hiển thị band. Kiểm tra UI đối chiếu các giá trị với tỷ lệ giá API, cả 4 preset, ngày chọn và chế độ ẩn/hiện.
