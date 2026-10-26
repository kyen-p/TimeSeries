# Lab 2: Kiểm định ADF và mô hình ARIMA/SARIMA

---

## 1. Giới thiệu

Bài Lab thực hiện quy trình xây dựng mô hình dự báo chuỗi thời gian: **kiểm định tính dừng (ADF) → sai phân → ACF/PACF để chọn bậc → fit mô hình → chẩn đoán phần dư → dự báo và đánh giá**.

- **Phần cơ bản (dataset1):** doanh số theo tháng (`a10.csv`), xây dựng mô hình ARIMA, đánh giá bằng MAE, RMSE, MAPE.
- **Phần nâng cao (dataset2):** AirPassengers, xây dựng mô hình SARIMA với s = 12 và so sánh với ARIMA.

Dữ liệu chia 80% train, 20% test, giữ nguyên thứ tự thời gian (không shuffle). Với dataset2, tập test là 12 tháng cuối.

**Kết quả chính (dataset1 – a10)**

| Nội dung            | Kết quả                                                                                      |
| ------------------- | -------------------------------------------------------------------------------------------- |
| Chất lượng dữ liệu  | 204 quan sát (07/1991 – 06/2008), không thiếu; min 2.81, max 29.67, trung bình 10.69         |
| Xu hướng và mùa vụ  | Trung bình trượt 12 tháng tăng 3.5 → 23.2 (gấp 6.6 lần); mùa vụ rõ: đỉnh tháng 1 (hệ số 1.33), đáy tháng 2 (0.78); độ lệch chuẩn trượt tăng 0.63 → 2.96 |
| ADF chuỗi gốc       | ADF statistic = 3.145, p = 1.0000 → **không dừng**                                            |
| ADF sau sai phân 1  | ADF statistic = -2.495, p = 0.117 → **chưa dừng** (p > 0.05)                                 |
| ADF sau sai phân 2  | ADF statistic = -10.292, p = 3.544e-18 ( ≈ 0) → dừng                                                        |
| Mô hình chọn        | ARIMA(1,1,1):  AIC = 582.47, BIC = 591.73;                                                     |
| Chẩn đoán phần dư   | Ljung-Box (lag 12) p ≈ 0 → phần dư **còn tự tương quan** (còn mùa vụ chưa được xử lý)        |
| Đánh giá (41 tháng) | MAE = 3.667, RMSE = 4.624, MAPE = 18.04%                                                     |


**Kết quả chính (dataset2 – AirPassengers)**

| Nội dung            | Kết quả                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------- |
| Chất lượng dữ liệu  | 144 quan sát, không thiếu; min 104, max 622, trung bình 280.30 (nghìn hành khách)        |
| Xu hướng và mùa vụ  | Xu hướng tăng rõ; mùa vụ chu kỳ 12 tháng, dạng nhân; đỉnh tháng 7–8, đáy tháng 11        |
| ADF chuỗi gốc       | [ADF statistic = 0.815369, p-value=0.991880] → không dừng                                                    |
| Sai phân            | d = 1, D = 1, s = 12 → chuỗi dao động quanh 0, biên độ ổn định hơn                       |
| ACF/PACF            | Cả hai biểu đồ chỉ có lag 1 nổi bật (khoảng -0.24)                                       |
| Mô hình chọn        | SARIMA(0,1,1)(0,1,1,12); AIC [803.41 ], BIC [811.37]                                              |
| Chẩn đoán phần dư   | Dao động quanh 0, gần như không tự tương quan; lag 12 vượt nhẹ ngưỡng                    |
| Đánh giá (12 tháng) | SARIMA: MAE [16.09], RMSE [20.93]; ARIMA(1,1,1): MAE [66.24], RMSE [91.23]                           |
| Kết luận            | SARIMA vượt trội ARIMA: ARIMA dự báo gần như đường phẳng, SARIMA bám sát đỉnh và đáy |



---

## 2. Cấu trúc thư mục

```
Lab2_TimeSeries
├── data 
│ ├── processed
│ │ ├── a10_diff.csv
│ │ └── airline_diff.csv 
│ └── raw
│ ├── a10.csv 
│ └── airline-passengers.csv 
├── docs 
│ ├── bai_tap_thuc_hanh
│ │ └── lab2_description.ipynb 
│ ├── bai_tap_nang_cao
│ │ └── lab2_description.ipynb 
│ └── report.md 
├── figure 
│ ├── bai_tap_thuc_hanh
│ │ ├── 01_.....png
│ │ └── ...
│ └── bai_tap_nang_cao
│ ├── 01_.....png
│ └── ...
├── scripts 
│ ├── bai_tap_thuc_hanh
│ │ ├── processed.py 
│ │ └── visual.py 
│ └── bai_tap_nang_cao
│ ├── processed.py 
│ └── visual.py 
└── README.md
```

| Thư mục / file    | Nhiệm vụ                                               |
| ----------------- | ------------------------------------------------------ |
| `data/raw/`       | Lưu dữ liệu gốc, **không sửa**                         |
| `data/processed/` | Lưu kết quả do code sinh ra; xóa đi thì chạy lại là có |
| `scripts/`        | Mã nguồn: tiền xử lý, xây dựng mô hình, vẽ biểu đồ     |
| `docs/`           | Báo cáo và mô tả bài Lab                               |
| `figures/`        | Hình ảnh biểu đồ do code sinh ra                       |
| `README.md`       | Giới thiệu và hướng dẫn chạy                           |

**Luồng dữ liệu**

```
data/raw ──► [tien_xu_ly].py ──► data/processed ──► [mo_hinh].py ──► figures/ + kết quả MAE/RMSE ──► docs/report.md
```

---

## 3. Dữ liệu

**Dataset 1: a10 (phần cơ bản)**

| Thuộc tính  | Giá trị                                                                         |
| ----------- | ------------------------------------------------------------------------------- |
| Ý nghĩa     | Doanh số theo tháng [xác nhận lại ý nghĩa cột dữ liệu]                          |
| Số quan sát | 204                                                                           |
| Cột         | `date`, `value`                                             |
| Nguồn       | `https://raw.githubusercontent.com/selva86/datasets/master/a10.csv`             |

**Dataset 2: AirPassengers (phần nâng cao)**

| Thuộc tính  | Giá trị                                                                                    |
| ----------- | ------------------------------------------------------------------------------------------ |
| Ý nghĩa     | Số hành khách hàng không quốc tế mỗi tháng (nghìn người), 1949–1960                        |
| Số quan sát | 144 (12 năm × 12 tháng)                                                                    |
| Cột         | `Month` (tháng), `Passengers` (số hành khách)                                              |
| Chu kỳ      | Hàng tháng, mùa vụ s = 12                                                                  |
| Nguồn       | `https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv`       |

---

## 4. Cài đặt

Yêu cầu: **Python 3.9 trở lên**.

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn
```

---

## 5. Cách chạy

Chạy từ thư mục gốc `Lab2_ARIMA_SARIMA/`, theo **đúng thứ tự**:

```bash
python scripts/[tien_xu_ly].py   # 1. tiền xử lý  → tạo data/processed/*.csv
python scripts/[mo_hinh].py      # 2. fit mô hình, dự báo, đánh giá → tạo figures/*.png
```

- Cần có Internet nếu code đọc dữ liệu trực tiếp từ link GitHub.
- Script mô hình đọc dữ liệu từ `data/processed/`, nên **phải chạy bước tiền xử lý trước**.

**Kiểm tra tái lập:** xóa `data/processed/` và `figures/`, chạy lại các lệnh trên, các file sẽ được sinh lại đầy đủ.

---

## 6. Ghi chú

- Nếu làm bằng notebook: mở bằng Jupyter hoặc tải lên Google Colab rồi chọn _Run all_ để chạy lại toàn bộ.
- Dữ liệu gốc trong `data/raw/` **không bị chỉnh sửa**; mọi thay đổi đều do code thực hiện và lưu sang `data/processed/`.
- Chia train/test theo thứ tự thời gian, **không shuffle**. Tập test chỉ dùng để đánh giá, không dùng để chọn bậc mô hình.
- Bậc mô hình chọn dựa trên ACF/PACF, sau đó so sánh bằng AIC/BIC. Nhận xét về tính dừng và phần dư dựa trên biểu đồ và kiểm định ADF.
- Chuỗi AirPassengers có mùa vụ dạng nhân (biên độ tăng theo mức chuỗi). Đề không yêu cầu, nhưng có thể lấy log trước khi fit để ổn định phương sai.
