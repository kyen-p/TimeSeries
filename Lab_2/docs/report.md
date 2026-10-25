# BÁO CÁO TỔNG HỢP: PHÂN TÍCH VÀ DỰ BÁO CHUỖI THỜI GIAN (ARIMA / SARIMA)

Báo cáo tổng hợp hai phần:

- **Phần A – Thực hành:** Bài tập 1–5 trên chuỗi `dataset1` (204 quan sát hằng tháng, 07/1991–06/2008).
- **Phần B – Nâng cao:** Đồ án SARIMA vs ARIMA trên chuỗi Air Passengers (144 quan sát hằng tháng, 1949–1960).

Mọi số liệu dưới đây lấy từ kết quả chạy của hai notebook.

---

## Tóm tắt điều hành

| Nội dung | Phần A (dataset1) | Phần B (Air Passengers) |
|---|---|---|
| Xu hướng | Tăng mạnh (xu hướng gấp ~6,6 lần) | Tăng +285,7% từ đầu đến cuối chuỗi |
| Mùa vụ | Chu kỳ 12 tháng (tháng 1 cao nhất, hệ số 1,331) | Chu kỳ 12 tháng (tháng 7 cao nhất, TB 351,3) |
| ADF chuỗi gốc | p = 1,0 → không dừng | p = 0,9919 → không dừng |
| Cấu hình sai phân | d = 1, D = 1, s = 12 | d = 1, D = 1, s = 12 |
| ADF sau sai phân kết hợp | p = 5,85e-05 → dừng | p = 1,86e-28 → dừng |
| Mô hình ARIMA tốt nhất | ARIMA(1,1,1) theo AIC/BIC | ARIMA(1,1,1) |
| Phần dư ARIMA | Chưa là nhiễu trắng (ACF lag 12 ≈ 0,84) | Dự báo gần như nằm ngang |
| Mô hình có mùa vụ | SARIMA(1,1,1)(1,1,1,12): MAPE 9,1% | SARIMA(0,1,1)(0,1,1,12): RMSE 20,93 |
| Cải thiện nhờ SARIMA | MAPE ~18% → 9,1% (gần một nửa) | RMSE giảm 77,1% so với ARIMA |

**Kết luận chung:** Ở cả hai chuỗi, ARIMA thuần không xử lý được mùa vụ nên dự báo kém. Mức cải thiện lớn đến từ việc thêm **thành phần mùa vụ (SARIMA)**, không phải từ việc tăng bậc sai phân d hay thêm tham số p, q.

---

# PHẦN A – THỰC HÀNH (dataset1)

## Bài 1. Khám phá dữ liệu (EDA)

- 204 quan sát theo tháng, 07/1991–06/2008, không có giá trị thiếu; `date` được đặt làm index thời gian.
- **Xu hướng: có.** Chuỗi tăng từ khoảng 3 (1991) lên gần 30 (đỉnh 29,67 vào 01/2008). Xu hướng (trung bình trượt 12 tháng căn giữa) tăng từ 3,5 (01/1992) lên 23,2 (12/2007).
- **Mùa vụ: có.** Mẫu hình lặp lại mỗi 12 tháng. Hệ số mùa vụ cao nhất ở tháng 1 (1,331), kế đến tháng 12 (1,230); thấp nhất ở tháng 2 (0,780).
- **Tính dừng: nhiều khả năng không dừng.** Trung bình tăng theo thời gian và biên độ dao động to dần (chênh lệch trong năm 1992 khoảng 3 đơn vị, năm 2007 khoảng 11,6 đơn vị).

Hệ số mùa vụ theo tháng:

| Tháng | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hệ số | 1,331 | 0,780 | 0,876 | 0,858 | 0,933 | 0,917 | 0,979 | 0,990 | 0,986 | 1,046 | 1,076 | 1,230 |

## Bài 2. Kiểm định ADF trên chuỗi gốc

- **H₀:** chuỗi có nghiệm đơn vị (không dừng). **H₁:** chuỗi dừng.
- **p-value** là mức bằng chứng chống lại H₀; p ≤ 0,05 thì bác bỏ H₀.

| Chỉ số | Giá trị |
|---|---|
| ADF statistic | 3,1452 |
| p-value | 1,0000 |
| Số lag dùng (AIC) | 15 |
| Giá trị tới hạn 1% / 5% / 10% | −3,4656 / −2,8770 / −2,5750 |

**Kết luận: chuỗi không dừng.** Phù hợp với quan sát trực quan:

- Trung bình trượt 12 tháng tăng gấp khoảng 6,6 lần (3,5 → 23,2).
- Độ lệch chuẩn trượt tăng gấp khoảng 4,7 lần (0,63 → 2,96).
- Nửa đầu và nửa sau chuỗi khác nhau rõ rệt:

| Giai đoạn | Trung bình | Độ lệch chuẩn |
|---|---|---|
| 07/1991–12/1999 | 6,01 | 1,96 |
| 01/2000–06/2008 | 15,38 | 4,80 |

## Bài 3. Sai phân và kiểm định lại

| Chuỗi | ADF stat | p-value | Dừng? | Độ lệch chuẩn | ACF lag 1 | ACF lag 12 |
|---|---|---|---|---|---|---|
| Gốc | 3,1452 | 1,0 | Chưa | 5,957 | 0,921 | – |
| Sai phân bậc 1 (d=1) | −2,4952 | 0,1167 | Chưa (sát ngưỡng) | 2,242 | −0,365 | 0,757 |
| Sai phân bậc 2 (d=2) | −10,2923 | 3,5e-18 | Có | 3,705 | −0,635 | 0,746 |
| Chỉ sai phân mùa vụ (D=1) | −2,0927 | 0,2475 | Chưa | 1,219 | 0,373 | – |
| d=1 + D=1 (s=12) | −4,7834 | 5,85e-05 | Có | 1,356 | −0,574 | −0,172 |

Các phát hiện chính:

1. **Sai phân bậc 1** loại bỏ xu hướng nhưng ADF (p = 0,1167) chưa vượt ngưỡng. ACF có đỉnh cao tại lag 12, 24, 36 (0,757; 0,615; 0,539), cho thấy phần còn lại là **mùa vụ chu kỳ 12 tháng**.
2. **Sai phân bậc 2** làm ADF đạt ngưỡng nhưng gây **sai phân quá mức**: độ lệch chuẩn tăng 2,242 → 3,705, ACF lag 1 xuống −0,635, và ACF lag 12 gần như không đổi (0,757 → 0,746). Tăng d không loại được mùa vụ.
3. **Kết hợp d = 1 và D = 1 (s = 12)** cho chuỗi dừng rõ ràng (p = 5,85e-05), ACF lag 12 chỉ còn −0,172.
4. **Độ vững của ADF bậc 1:** với số lag tối đa 0–14, p đều < 0,05; chỉ cấu hình 15 lag (AIC chọn) cho p = 0,1167. Kết luận "bậc 1 chưa dừng" vì vậy mong manh và không nên dùng làm khẳng định chắc chắn.
5. **Kiểm chứng bằng dự báo** (train 80% / test 20%):

| Mô hình | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| ARIMA(0,1,1) | 3,46 | 4,15 | 18,4 |
| ARIMA(1,1,1) | 3,67 | 4,62 | 18,0 |
| ARIMA(2,1,1) | 3,67 | 4,63 | 18,1 |
| ARIMA(0,2,1) | 4,18 | 4,88 | 24,9 |
| ARIMA(1,2,1) | 3,89 | 4,56 | 23,1 |
| ARIMA(2,2,1) | 3,64 | 4,30 | 21,5 |
| SARIMA(1,1,1)(1,1,1,12) | 1,56 | 2,00 | 9,1 |

Theo MAPE, mọi mô hình d = 1 (18,0–18,4%) đều tốt hơn mọi mô hình d = 2 (21,5–24,9%). Cải thiện lớn nhất đến từ SARIMA (MAPE 9,1%).

**Kết luận Bài 3: d = 1** (kết hợp D = 1, s = 12 để xử lý mùa vụ). Giới hạn: kết quả dự báo chỉ dựa trên một lần chia train/test.

## Bài 4. Phân tích ACF và PACF (sau sai phân bậc 1)

Ngưỡng tin cậy 95%: ±0,138 (n = 203).

| | Lag vượt ngưỡng | Giá trị chính |
|---|---|---|
| ACF | 1, 11, 12, 13, 23, 24, 25, 35, 36 | lag 1 = −0,365; lag 12 = 0,757; lag 24 = 0,615; lag 36 = 0,539 |
| PACF | 1, 2, 3, 4, 9, 10, 11, 12, 13, 17 | lag 1 = −0,365; lag 2 = −0,147; lag 3 = −0,182; lag 4 = −0,156; lag 11 = −0,634; lag 12 = 0,433 |

- **ACF cắt cụt sau lag 1** (các lag 2–10 nằm trong ngưỡng) → gợi ý **q = 1**. Các đỉnh tại lag 11–13, 23–25, 35–36 là dấu hiệu mùa vụ, không dùng để chọn q thường.
- **PACF không cắt cụt gọn** (lag 1–4 cùng âm và đều vượt ngưỡng) → chọn **p = 0** làm mô hình chính.
- **Đề xuất:** ARIMA(0,1,1) là mô hình chính; ARIMA(1,1,1) và ARIMA(2,1,1) để so sánh.
- ARIMA(p,1,q) không xử lý được mùa vụ còn lại, nên cần SARIMA.

## Bài 5. Thiết lập ARIMA, dự báo và đánh giá

**Chia dữ liệu theo thứ tự thời gian (không shuffle):**

- Train: 163 quan sát (07/1991–01/2005)
- Test: 41 quan sát (02/2005–06/2008)

**Thông tin mô hình (fit trên train, cả ba đều hội tụ):**

| Mô hình | AIC | BIC |
|---|---|---|
| ARIMA(0,1,1) | 587,92 | 594,09 |
| ARIMA(1,1,1) | **582,47** | **591,73** |
| ARIMA(2,1,1) | 584,46 | 596,81 |

**Chẩn đoán phần dư:**

| Mô hình | Mean | Std | ACF lag 1 | ACF lag 12 | p LB lag 10 | p LB lag 12 | p LB lag 24 |
|---|---|---|---|---|---|---|---|
| ARIMA(0,1,1) | 0,207 | 1,455 | 0,071 | 0,854 | 0,0284 | 0,0 | 0,0 |
| ARIMA(1,1,1) | 0,298 | 1,404 | −0,048 | 0,840 | 0,4456 | 0,0 | 0,0 |
| ARIMA(2,1,1) | 0,298 | 1,405 | −0,045 | 0,840 | 0,4433 | 0,0 | 0,0 |

- Mean phần dư xấp xỉ 0 so với độ lệch chuẩn (~1,4).
- Tự tương quan ngắn hạn đã được xử lý tốt ở ARIMA(1,1,1) và (2,1,1) (Ljung–Box lag 10, p ≈ 0,44); ARIMA(0,1,1) còn sót (p = 0,028).
- **Phần dư chưa phải nhiễu trắng:** ACF lag 12 ≈ 0,84 và Ljung–Box lag 12, 24 đều có p ≈ 0 → còn tự tương quan mùa vụ rất mạnh.

**Đánh giá dự báo trên tập test:**

| Mô hình | AIC | BIC | MAE | RMSE | MAPE (%) |
|---|---|---|---|---|---|
| ARIMA(0,1,1) | 587,92 | 594,09 | 3,4598 | 4,1468 | 18,39 |
| ARIMA(1,1,1) | 582,47 | 591,73 | 3,6667 | 4,6243 | 18,04 |
| ARIMA(2,1,1) | 584,46 | 596,81 | 3,6702 | 4,6285 | 18,06 |

- Đường dự báo gần như **nằm ngang** (~17,5), chỉ bám mức trung bình, không bám xu hướng tăng và chu kỳ 12 tháng.
- Sai lệch lớn nhất ở các đỉnh mùa vụ (tháng 11, 12, 1): 01/2008 lệch 12,13; 01/2007 lệch 10,50; 12/2007 lệch 8,27. Các tháng 6–9 sai lệch nhỏ (~2–2,5). Dự báo phần lớn thấp hơn thực tế (lệch trung bình ~1,9).
- Ba mô hình dự báo gần như giống nhau (MAPE ~18%), chênh lệch giữa chúng nhỏ nên không nên kết luận mô hình nào "thắng" rõ ràng khi cùng thiếu mùa vụ.

**Trả lời các câu hỏi chính:**

- *Vì sao không shuffle?* Tránh rò rỉ thông tin từ tương lai và giữ cấu trúc tự tương quan; test là dữ liệu chưa thấy dùng để đo khả năng dự báo thật.
- *Ý nghĩa p, d, q:* p = bậc tự hồi quy (AR); d = số lần sai phân (ở đây 1); q = bậc trung bình trượt (MA).
- *Residuals có phải nhiễu trắng không?* Chưa, còn tự tương quan mùa vụ.
- *Điều chỉnh:* thêm sai phân mùa vụ (D = 1, s = 12) hoặc dùng SARIMA(p,1,q)(P,D,Q)₁₂. Thêm p, q không giải quyết được lag 12.
- *Chỉ số nhạy với outlier:* RMSE (4,62 > MAE 3,67 do các đỉnh tháng 1). RMSE ≈ 24% trung bình tập test (19,5), là sai số khá lớn, khớp với việc mô hình chưa bắt được mùa vụ.

**Kết luận Phần A:** ARIMA(1,1,1) tốt nhất theo AIC/BIC, nhưng cả ba mô hình ARIMA đều thiếu thành phần mùa vụ nên cần chuyển sang SARIMA (đã cho MAPE ~9% ở Bài 3).

---

# PHẦN B – NÂNG CAO: ĐỒ ÁN DỰ BÁO LƯỢNG HÀNH KHÁCH HÀNG KHÔNG (SARIMA vs ARIMA)

**Dữ liệu:** Air Passengers, lượng hành khách quốc tế theo tháng (nghìn người), 144 quan sát, 01/1949–12/1960.
**Thiết kế đánh giá:** test = 12 tháng cuối (1960); mọi mô hình chỉ fit trên train (132 tháng, 01/1949–12/1959). Chọn bậc SARIMA bằng AIC trên train, không dùng test để chọn mô hình.

## B1. Nhận diện xu hướng và mùa vụ

| Thống kê | Giá trị |
|---|---|
| Trung bình | 280,30 |
| Độ lệch chuẩn | 119,97 |
| Min / Max | 104 / 622 |
| Trung vị | 265,5 |
| Số giá trị thiếu | 0 |

- Trung bình tháng cao nhất: **Tháng 7 (351,3)**; thấp nhất: **Tháng 11 (232,8)**.
- Trung bình năm tăng từ 126,7 (1949) lên 476,2 (1960); toàn chuỗi tăng 285,7%.
- Biên độ dao động tăng dần cùng mức trung bình, các đỉnh rơi vào mùa hè, mẫu lặp lại hằng năm → xu hướng tăng và mùa vụ 12 tháng (đã phân rã bằng STL).

## B2. ADF trên chuỗi gốc

ADF statistic = 0,8154; p-value = 0,9919 (13 lag, 130 quan sát) → **chưa bác bỏ H₀, chuỗi không dừng.**

## B3. Sai phân thường và sai phân mùa vụ

| Chuỗi | ADF stat | p-value | Dừng ở 5%? |
|---|---|---|---|
| Chuỗi gốc | 0,8154 | 0,9919 | Không |
| Sai phân thường (d=1) | −2,8293 | 0,0542 | Không (sát ngưỡng) |
| Sai phân mùa vụ (D=1, s=12) | −3,3830 | 0,0116 | Có |
| Kết hợp (d=1, D=1, s=12) | −15,5956 | ≈ 0 (1,86e-28) | Có |

Cấu hình **d = 1, D = 1, s = 12** cho chuỗi dừng và được dùng cho mô hình.

## B4. ACF/PACF → xác định (p, q, P, Q)

ACF/PACF tính trên **train** đã sai phân (không dùng test); ngưỡng tham khảo 95% ≈ ±0,180.

| Lag | ACF | PACF |
|---|---|---|
| 1 | −0,243 | −0,243 |
| 2 | 0,123 | 0,068 |
| 12 | −0,118 | −0,047 |
| 13 | 0,097 | 0,061 |
| 24 | 0,050 | 0,106 |
| 36 | 0,024 | −0,014 |

Chỉ lag 1 vượt ngưỡng → gợi ý **p = 0, q = 1, P = 0**, và thử **Q = 0 hoặc Q = 1**. Đây là gợi ý tham khảo, bậc cuối cùng chọn bằng AIC trên train.

## B5. Fit SARIMA

| Mô hình | AIC | BIC |
|---|---|---|
| **SARIMA(0,1,1)×(0,1,1,12)** | **803,41** | **811,37** |
| SARIMA(0,1,1)×(0,1,0,12) | 887,29 | 892,82 |

Chọn **SARIMA(0,1,1)×(0,1,1,12)** (AIC thấp nhất, thấp hơn khoảng 84 đơn vị).

**Chẩn đoán phần dư:** Ljung–Box lag 12: thống kê = 16,1454, p = 0,0955 > 0,05 → chưa thấy tự tương quan còn lại có ý nghĩa ở mức 5%.

## B6. Dự báo và đánh giá SARIMA (12 tháng test)

**MAE = 16,09; RMSE = 20,93** (nghìn hành khách).

| Tháng | Thực tế | SARIMA | ARIMA(1,1,1) |
|---|---|---|---|
| 01/1960 | 417 | 422,8 | 433,5 |
| 02/1960 | 391 | 404,5 | 418,0 |
| 03/1960 | 419 | 466,8 | 426,4 |
| 04/1960 | 461 | 456,5 | 421,8 |
| 05/1960 | 472 | 479,7 | 424,3 |
| 06/1960 | 535 | 533,4 | 423,0 |
| 07/1960 | 622 | 607,5 | 423,7 |
| 08/1960 | 606 | 618,7 | 423,3 |
| 09/1960 | 508 | 522,6 | 423,5 |
| 10/1960 | 461 | 467,5 | 423,4 |
| 11/1960 | 390 | 422,2 | 423,5 |
| 12/1960 | 432 | 463,8 | 423,4 |

SARIMA bám sát đường cong mùa vụ, kể cả đỉnh tháng 7–8; sai lệch lớn nhất ở 03/1960 (+47,8) và 12/1960 (+31,8), 11/1960 (+32,2).

## B7. So sánh với ARIMA(1,1,1)

| Mô hình | MAE | RMSE |
|---|---|---|
| **SARIMA(0,1,1)×(0,1,1,12)** | **16,09** | **20,93** |
| ARIMA(1,1,1) (AIC = 1257,04; BIC = 1265,66) | 66,24 | 91,23 |

- ARIMA dự báo gần như một đường phẳng (~423), không tái hiện mùa vụ.
- **RMSE của SARIMA thấp hơn 77,1%** so với ARIMA.
- Lưu ý: AIC của hai mô hình (803 so với 1257) không so sánh trực tiếp được vì khác bậc sai phân; kết luận dựa trên sai số dự báo trên cùng tập test.

---

# TỔNG HỢP VÀ BÀI HỌC CHUNG

## Quy trình xây dựng mô hình (áp dụng nhất quán ở cả hai chuỗi)

1. **EDA:** vẽ chuỗi, nhận diện xu hướng và mùa vụ.
2. **ADF trên chuỗi gốc:** xác nhận không dừng.
3. **Sai phân:** d = 1 để bỏ xu hướng, D = 1 (s = 12) để bỏ mùa vụ; kiểm tra lại bằng ADF và ACF.
4. **ACF/PACF** trên chuỗi đã sai phân → gợi ý bậc (p, q) và (P, Q).
5. **Fit** các ứng viên trên train, so sánh AIC/BIC.
6. **Chẩn đoán phần dư** (mean, ACF, Ljung–Box).
7. **Dự báo trên test** và đánh giá MAE/RMSE/MAPE; so sánh với ARIMA.

## So sánh hai phần

| Tiêu chí | Phần A (dataset1) | Phần B (Air Passengers) |
|---|---|---|
| Số quan sát / Train–Test | 204 / 80–20 (163–41) | 144 / 132–12 |
| Mô hình ARIMA | (0,1,1), (1,1,1), (2,1,1) | (1,1,1) |
| Mô hình có mùa vụ | SARIMA(1,1,1)(1,1,1,12) (so sánh nhanh ở Bài 3) | SARIMA(0,1,1)(0,1,1,12) (fit và chẩn đoán đầy đủ) |
| Kết quả ARIMA | MAPE ~18%, RMSE 4,1–4,6 | MAE 66,24, RMSE 91,23 |
| Kết quả SARIMA | MAPE 9,1%, RMSE 2,00 | MAE 16,09, RMSE 20,93 |
| Phần dư ARIMA | Còn tự tương quan lag 12 | Dự báo phẳng, không bắt mùa vụ |
| Phần dư SARIMA | (chưa chẩn đoán riêng) | Ljung–Box lag 12: p = 0,0955 (ổn) |

## Kết luận chính

1. Cả hai chuỗi đều có **xu hướng tăng** và **mùa vụ chu kỳ 12 tháng**; chuỗi gốc không dừng (ADF p ≈ 1).
2. **d = 1 kết hợp D = 1 (s = 12)** là cấu hình đúng. Tăng d lên 2 gây sai phân quá mức mà không loại được mùa vụ.
3. ACF/PACF sau sai phân gợi ý cấu trúc **MA** (q = 1; thêm Q = 1 cho mùa vụ), phù hợp dạng "airline model" (0,1,1)(0,1,1)₁₂.
4. ARIMA không có thành phần mùa vụ cho dự báo gần như phẳng, phần dư còn tự tương quan lag 12. Các tinh chỉnh p, q chỉ thay đổi rất nhỏ kết quả.
5. **SARIMA cải thiện rất lớn:** MAPE giảm từ ~18% xuống 9,1% (dataset1) và RMSE giảm 77,1% (Air Passengers).

## Hạn chế và hướng phát triển

- Đánh giá chỉ dựa trên **một lần chia train/test** (Phần B chỉ 12 tháng test); nên dùng kiểm định cuốn chiếu (rolling-origin) trước khi triển khai thực tế.
- Kết quả ADF bậc 1 nhạy với số lag (Bài 3), nên dùng thêm ACF và dự báo để xác nhận d.
- Ở Phần A, SARIMA mới chỉ được so sánh nhanh ở Bài 3; nên bổ sung chẩn đoán phần dư và lựa chọn bậc bằng AIC như Phần B.
- Ở Phần B, ARIMA(1,1,1) không có sai phân mùa vụ, nên so sánh là "có/không có mùa vụ" hơn là so sánh tối ưu hóa từng mô hình; có thể thử thêm biến đổi log để ổn định phương sai (biên độ tăng theo mức).
- Có thể bổ sung khoảng dự báo (prediction interval) và thử các ứng viên SARIMA rộng hơn.
