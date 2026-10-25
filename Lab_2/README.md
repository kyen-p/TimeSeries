# Lab 2: Kiểm định ADF và mô hình ARIMA/SARIMA

---

## 1. Giới thiệu

Bài Lab thực hiện quy trình xây dựng mô hình dự báo chuỗi thời gian: **kiểm định tính dừng (ADF) → sai phân → ACF/PACF để chọn bậc → fit mô hình → chẩn đoán phần dư → dự báo và đánh giá**.

- **Phần cơ bản (dataset1):** doanh số theo tháng (`a10.csv`), xây dựng mô hình ARIMA, đánh giá bằng MAE, RMSE, MAPE.
- **Phần nâng cao (dataset2):** AirPassengers, xây dựng mô hình SARIMA với s = 12 và so sánh với ARIMA.

Dữ liệu chia 80% train, 20% test, giữ nguyên thứ tự thời gian (không shuffle). Với dataset2, tập test là 12 tháng cuối.

**Kết quả chính (dataset2 – AirPassengers)**

| Nội dung            | Kết quả                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------- |
| Chất lượng dữ liệu  | 144 quan sát, không thiếu; min 104, max 622, trung bình 280.30 (nghìn hành khách)        |
| Xu hướng và mùa vụ  | Xu hướng tăng rõ; mùa vụ chu kỳ 12 tháng, dạng nhân; đỉnh tháng 7–8, đáy tháng 11        |
| ADF chuỗi gốc       | [ADF statistic, p-value] → không dừng                                                    |
| Sai phân            | d = 1, D = 1, s = 12 → chuỗi dao động quanh 0, biên độ ổn định hơn                       |
| ACF/PACF            | Cả hai biểu đồ chỉ có lag 1 nổi bật (khoảng -0.24)                                       |
| Mô hình chọn        | SARIMA(0,1,1)(0,1,1,12); AIC [..], BIC [..]                                              |
| Chẩn đoán phần dư   | Dao động quanh 0, gần như không tự tương quan; lag 12 vượt nhẹ ngưỡng                    |
| Đánh giá (12 tháng) | SARIMA: MAE [..], RMSE [..]; ARIMA(1,1,1): MAE [..], RMSE [..]                           |
| Kết luận            | **SARIMA vượt trội ARIMA**: ARIMA dự báo gần như đường phẳng, SARIMA bám sát đỉnh và đáy |

**Kết quả chính (dataset1 – a10)**

| Nội dung            | Kết quả                                 |
| ------------------- | --------------------------------------- |
| ADF chuỗi gốc       | [ADF statistic, p-value] → [kết luận]   |
| ADF sau sai phân    | [ADF statistic, p-value] → d = [..]     |
| Mô hình chọn        | ARIMA([p],[d],[q]); AIC [..], BIC [..]  |
| Đánh giá (tập test) | MAE [..], RMSE [..], MAPE [..]          |

---

## 2. Cấu trúc thư mục

```
Lab2_ARIMA_SARIMA/
├── data/
│   ├── raw/
│   │   ├── a10.csv
│   │   └── airline-passengers.csv
│   └── processed/
│       └── [file-da-xu-ly].csv
├── scripts/
│   ├── [tien_xu_ly].py
│   ├── [mo_hinh].py
│   └── [truc_quan].py
├── docs/
│   ├── lab2_description.md
│   └── report.md
├── figures/
│   ├── 01_line_plot.png
│   ├── 02_stl_decomposition.png
│   ├── 03_seasonal_plot_heatmap.png
│   ├── 04_differencing.png
│   ├── 05_acf_pacf.png
│   ├── 06_residual_diagnostics.png
│   ├── 07_sarima_forecast.png
│   └── 08_sarima_vs_arima.png
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
| Số quan sát | [..]                                                                            |
| Cột         | `date`, `value` [xác nhận tên cột]                                              |
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
