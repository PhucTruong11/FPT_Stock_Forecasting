# Đối chiếu nguồn dữ liệu FPT — 29/10/2026

**Kết luận:** cần xử lý nguồn dữ liệu trước đợt thử model tiếp theo. Tải lại Yahoo không giải quyết các vấn đề đã thấy. Vietcap là ứng viên cho snapshot nghiên cứu v3, với KBS làm nguồn kiểm tra. Chưa thay snapshot v2 hoặc model đang phục vụ, chưa train thêm, chưa đánh giá dữ liệu 2026.

## 1. Phạm vi và bằng chứng

Audit ID: `20261029T135735663765Z`. Cùng mã FPT, cùng cửa sổ **2018-01-01 ≤ ngày < 2026-01-01**, cùng đơn vị giá VND. Yahoo cũ được lấy nguyên byte từ snapshot v2; Yahoo mới tải vào thư mục riêng với cùng cờ `auto_adjust=True, repair=False, keepna=True`. Tải thêm Yahoo `auto_adjust=False, actions=True` để kiểm tra quan hệ Close/Adj Close.

Nguồn độc lập với Yahoo là chart API Vietcap. Sau khi phát hiện bất đồng, bổ sung KBS để xác minh; việc bổ sung được ghi riêng trong `kbs_acquisition.json`, không sửa gói bằng chứng đầu tiên. Đây là kiểm tra hồi cứu, không phải thí nghiệm model đăng ký trước hoặc bằng chứng ba nguồn có hệ thống thu thập hoàn toàn độc lập với nhau.

- [summary.json](summary.json): kết quả máy đọc được.
- [acquisition.json](acquisition.json), [kbs_acquisition.json](kbs_acquisition.json): request, thời điểm, phiên bản thư viện, hash và quy ước.
- [evidence.zip](evidence.zip): Yahoo cũ/mới, metadata, Yahoo chưa auto-adjust kèm actions, response Vietcap nguyên byte và bảng chuẩn hóa.
- [kbs_evidence.zip](kbs_evidence.zip): response KBS nguyên byte và bảng chuẩn hóa.
- [comparisons.zip](comparisons.zip): đối chiếu OHLCV từng ngày, từng trường; đối chiếu lợi suất và bảng adjustment bridge. `*_left/right` theo thứ tự trong tên cặp; `refreshed` là Yahoo mới.
- [flagged_observations.csv](flagged_observations.csv), [yearly_quality.csv](yearly_quality.csv), [changed_target_dates.csv](changed_target_dates.csv): các điểm cần xem nhanh.

Tám file dữ liệu/pointer được kiểm tra hash trước/sau; [protected_before.json](protected_before.json) và [protected_after.json](protected_after.json) giống nhau. ZIP giữ byte gốc nên việc Git chuẩn hóa xuống dòng của JSON/CSV ngoài ZIP không làm mất bằng chứng nguồn.

## 2. Quy ước để so sánh đúng

Chart Vietcap/KBS trả giá VND trực tiếp. Adapter vnstock có bước đổi sang nghìn VND; audit gọi HTTP trực tiếp nên **không nhân/chia 1.000**. Các chart được mô tả là giá điều chỉnh theo sự kiện quyền. Timestamp epoch Vietcap được chuyển UTC → Asia/Ho_Chi_Minh rồi lấy ngày; nhãn ngày KBS `07:00` giữ nguyên ngày lịch.

Nguồn tham chiếu kỹ thuật: [tài liệu thị trường vnstock](https://www.vnstocks.com/docs/vnstock/thi-truong-market), [Vietcap request/parser](https://github.com/thinh-vu/vnstock/blob/main/vnstock/explorer/vci/quote.py), [Vietcap endpoint/field map](https://github.com/thinh-vu/vnstock/blob/main/vnstock/explorer/vci/const.py), [KBS parser và đổi đơn vị](https://github.com/thinh-vu/vnstock/blob/main/vnstock/explorer/kbs/quote.py). Không cài vnstock hoặc đổi dependency của pipeline để lấy các bảng này.

Đây vẫn là **adjusted history nhìn từ thời điểm tải hiện tại**. Chưa có sổ corporate actions được kiểm chứng từng sự kiện hoặc lịch HOSE chính thức. Hai chart đồng thuận là bằng chứng đối chiếu, không tự trở thành chứng nhận của sàn. Định nghĩa volume giữa các nguồn chưa được hòa giải, nên không gọi mọi chênh lệch volume là dữ liệu sai.

Ngưỡng chẩn đoán trong protocol trước khi tải: giá gần bằng nếu `rtol=1e-6, atol=0.02 VND`; chênh mức giá trên 0,1% được gắn cờ. Chênh **log-return trên 10 bps** được xem xét sâu; 10 bps = 0,001 đơn vị log-return. Ngưỡng này phục vụ tìm bất đồng, không phải chuẩn chất lượng của sàn hoặc tiêu chí model thắng.

## 3. Kết quả kiểm tra dòng và ngày

| Nguồn | Tổng dòng trong cửa sổ | Volume dương | Volume = 0 | Close thiếu | Nến sai OHLC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Yahoo v2 đã khóa | 2.088 | 1.941 | 147 | 4 | 1 |
| Yahoo tải lại | 2.088 | 1.941 | 147 | 4 | 1 |
| Vietcap | 1.999 | 1.997 | 2 | 0 | 0 |
| KBS | 1.997 | 1.997 | 0 | 0 | 1 |

Yahoo mới không có dòng OHLCV nào thay đổi vượt ngưỡng so với v2. Sai khác Close lớn nhất chỉ khoảng **0,000041%**; không đồng nghĩa file giống byte. Các lỗi/điểm thiếu vẫn lặp lại.

**56 ngày Yahoo có volume = 0 nhưng cả Vietcap và KBS đều có volume > 0**, bao gồm cả bốn ngày Yahoo thiếu Close: 03–05/12/2019 và 04/03/2020. Không thể xem toàn bộ 147 dòng zero-volume của Yahoo là ngày nghỉ. Trong 56 ngày này, phân bố theo năm là 4, 10, 9, 25, 3, 2, 3, 0 cho 2018–2025.

Hai dòng zero-volume của Vietcap là 23–24/01/2018; KBS không có hai dòng này. Audit không gán chúng thành phiên giao dịch. Có 89 ngày chỉ Yahoo xuất hiện và không có ở hai chart kia; chưa gắn nhãn ngày nghỉ chính thức cho các ngày đó.

Yahoo có nến sai ngày **02/11/2021**, Close 43.938,08 cao hơn High 43.530,41. Vietcap/KBS đều có nến hợp lệ ở ngày này, Close lần lượt 43.741,15 và 43.752. KBS cũng có lỗi riêng **07/08/2019**: Open 15.337 thấp hơn Low 15.575; không dùng KBS làm nguồn hoàn hảo để thay toàn bộ dữ liệu.

## 4. Khác mức giá và khác nhãn lợi suất

Không suy ra lỗi từ mức adjusted price khác nhau đơn thuần: hệ số điều chỉnh/độ làm tròn có thể khác. Vì mục tiêu ML là lợi suất, kiểm tra thêm log-return trên hai ngày liền nhau được cả hai nguồn ghi nhận có volume dương. Nếu thiếu một ngày ở một nguồn, loại cặp chạm ngày thiếu ở cả hai phía, không nối tắt để tạo lợi suất nhiều phiên.

| Cặp nguồn | Cặp ngày tính được lợi suất | Cặp lệch > 10 bps | Trung vị lệch tuyệt đối | Lệch lớn nhất |
| --- | ---: | ---: | ---: | ---: |
| Yahoo mới – Vietcap | 1.898 | 137 | 0,0012 bps | 670,33 bps |
| Vietcap – KBS | 1.996 | 2 | 0,0693 bps | 21,52 bps |

**131/137 bất đồng Yahoo–Vietcap tập trung năm 2025**; năm đó so được 249 cặp ngày. Vì vậy kiểm tra `missing/duplicate/invalid candle` nội bộ là cần nhưng chưa đủ: 2025 không thiếu ngày volume dương so với hai nguồn còn lại, mà giá theo ngày vẫn bất đồng đáng kể.

Ví dụ giá Close điều chỉnh (VND):

| Ngày | Yahoo mới | Vietcap | KBS |
| --- | ---: | ---: | ---: |
| 20/08/2025 | 89.016,64 | 86.708,74 | 86.733 |
| 21/08/2025 | 86.709,13 | 90.258,74 | 90.284 |
| 22/08/2025 | 90.259,16 | 87.862,49 | 87.887 |

Ở ví dụ này, một số giá Yahoo gần giá Vietcap của ngày trước, còn Vietcap và KBS đồng thuận hướng biến động theo ngày. Đây là **dấu hiệu cần điều tra việc gắn ngày/giá cũ**; chưa xác minh nguyên nhân phía Yahoo. Không tự shift toàn bộ chuỗi: bất đồng không đồng nhất mọi ngày và sửa bằng quy tắc suy đoán có thể tạo lỗi mới.

Hai cặp lợi suất Vietcap–KBS lệch >10 bps là **14–15/03/2019**, cùng chạm điểm giá 14/03/2019; giữ nguyên nguồn, lưu cờ để kiểm tra độ nhạy về sau. Các tính toán audit bao gồm cả nến lỗi để thể hiện bất đồng; feature builder vẫn loại nến lỗi theo hợp đồng hiện tại.

17 dòng action do Yahoo báo được lưu trong [yahoo_reported_actions.csv](yahoo_reported_actions.csv). Close auto-adjust khớp Adj Close ở cả 2.084 ngày có Close; điều đó chỉ xác nhận cách biến đổi nội bộ của Yahoo, không xác nhận action ledger hoặc giá/nhãn ngày của Yahoo là đúng. Cửa sổ action không bao gồm 2026 nên chưa giải thích đầy đủ các điều chỉnh lịch sử bởi sự kiện sau 2025.

## 5. Ảnh hưởng tới pipeline hiện có

Đã chạy thử **hàm build feature thuần**, không ghi đè các file train và không fit model:

| Nguồn đầu vào | Dòng hợp lệ sau loại | Dòng features | Mẫu supervised | Dòng bị loại |
| --- | ---: | ---: | ---: | ---: |
| Yahoo v2 | 1.940 | 1.891 | 1.889 | 148 |
| Vietcap | 1.997 | 1.948 | 1.947 | 2 |
| KBS | 1.996 | 1.947 | 1.945 | 1 |

Vietcap tạo thêm **58 mẫu có nhãn**. Quan trọng hơn, trong 1.889 origin có nhãn chung Yahoo–Vietcap, **39 origin có ngày đích khác nhau**. Như vậy thay nguồn không chỉ tăng số dòng: nó thay ngữ nghĩa `next observation`, giá, volume, rolling features và nhãn. Không so trực tiếp hai RMSE từ hai tập origin/nhãn khác nhau rồi tuyên bố model cải thiện.

## 6. Quyết định và thí nghiệm tiếp theo

1. Giữ v2 và các kết quả cũ làm reference lịch sử, đánh dấu rằng audit nguồn chưa đạt mức cần để khẳng định chất lượng dự báo thị trường. Không viết lại metric cũ bằng nhãn mới.
2. Chọn **Vietcap làm ứng viên nguồn chính cho v3**, KBS làm đối chứng kiểm tra ngày/giá; giữ cờ 14/03/2019 và không trộn/điền giá từng ngày từ hai nhà cung cấp một cách tùy ý. Hai dòng volume=0 vẫn được giữ ở raw và loại theo hợp đồng feature.
3. Trước training v3: khóa snapshot/hash/schema mới; metadata ghi rõ provider, giá điều chỉnh, calendar chưa chứng nhận và các cờ còn mở. Không dùng metadata Yahoo để mô tả dữ liệu Vietcap. Đối chiếu action ledger với thông báo doanh nghiệp/sàn trước khi gọi dữ liệu là point-in-time hoặc dùng mô phỏng giao dịch.
4. Chạy đối chứng có kiểm soát: dùng cùng origin + target date và **một nhãn đánh giá v3 cố định** cho mọi candidate; baseline Zero Return/Train Mean, cùng folds/seed. Mỗi fold fit hai candidate với nguồn train v2/v3 nhưng cùng cấu hình và cutoff; scaler chỉ fit train tương ứng. Ở evaluation, mỗi nhánh dùng features của nguồn tương ứng trên cohort ngày chung, so cùng nhãn v3. Sau đó mới thử returns-only/full trên nguồn đã chốt. Không dùng model serving đã train đến hết 2022 để chấm fold evaluation 2022. Mỗi cấu hình và cohort phải được ghi trước khi fit; không trộn tác động đổi nhãn, đổi nguồn, đổi feature và đổi model thành một chỉ số.
5. Giai đoạn 2022–2025 tiếp tục là nghiên cứu hồi cứu đã được xem. Đóng băng kế hoạch holdout tương lai trước khi mở dữ liệu mới; chưa tải/đánh giá 2026 trong lượt này. Chỉ thay model nếu đạt gate đã khóa, không dựa vào một hit rate đẹp.

**Chưa có bằng chứng accuracy tăng sau audit.** Kết quả hiện tại là chẩn đoán được vấn đề nguồn, có dữ liệu ứng viên và quy trình đối chứng để đánh giá thay đổi sau này. API/Streamlit giữ nguyên để bạn phụ trách tích hợp tiếp tục công việc.

## 7. Tái tạo

```powershell
# Không dùng mạng: kiểm tra hash ZIP, tái tạo bảng/summary và dry-run feature.
python -m src.data.source_audit --analyze reports/data_audits/20261029T135735663765Z
python reports/data_audits/20261029T135735663765Z/verify.py

# Một lần tải mới luôn có ID riêng, không ghi đè snapshot v2.
python -m src.data.source_audit
# Chỉ bổ sung KBS khi cần xác minh bất đồng của run mới:
python -m src.data.source_audit --confirm-kbs reports/data_audits/<new_run_id>

python -m unittest discover -s tests -v
```

Bảng lớn nằm trong `comparisons.zip`. Các ZIP so sánh có thể khác byte do timestamp của archive khi tái tạo; nội dung CSV, hash nguồn và kết quả số mới là điều cần kiểm chứng. **51/51 unit tests qua**, gồm 9 regression tests riêng cho parser/đơn vị/ngày, uniqueness, missingness, phiên thiếu và tính bất biến lợi suất với hệ số giá cố định. [verification.json](verification.json) ghi kết quả kiểm tra độc lập và việc tái tạo số liệu từ ZIP; [verify.py](verify.py) giữ phép tính đối chiếu độc lập để chạy lại.
