## Chuẩn bị thư viện và dữ liệu 
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
SEASONAL_PERIOD = 12
TEST_MONTHS = 12
ALPHA = 0.05

data = pd.read_csv(DATA_URL, parse_dates=["Month"])
data = data.set_index("Month").sort_index()
data.index = pd.DatetimeIndex(data.index, freq="MS")
passengers = data["Passengers"].astype(float)

if passengers.index.has_duplicates:
    raise ValueError("Dữ liệu có tháng bị lặp.")
if passengers.isna().any():
    raise ValueError("Dữ liệu có giá trị thiếu; hãy kiểm tra lại nguồn dữ liệu.")
if not passengers.index.is_monotonic_increasing:
    raise ValueError("Thời gian phải được sắp xếp tăng dần.")

print(f"Số quan sát: {len(passengers)}")
print(f"Khoảng thời gian: {passengers.index.min():%Y-%m} đến {passengers.index.max():%Y-%m}")
print(f"Tần suất: {passengers.index.freqstr}")
print("\nNăm dòng đầu:")
print(data.head().to_string())
# ---------------------------------------------------------------- 


## 1. Vẽ chuỗi thời gian → nhận diện mùa vụ
##Thống kê mô tả, biểu đồ chuỗi thời gian, seasonal plot và heatmap giúp quan sát xu hướng tăng cũng như chu kỳ theo tháng.
print("Thống kê mô tả:")
print(data["Passengers"].describe().round(2).to_string())
print(f"\nSố giá trị thiếu: {int(data['Passengers'].isna().sum())}")
print(f"Tổng lượng hành khách trong dữ liệu (nghìn): {passengers.sum():,.0f}")

# Tính Min, Max, Mean
min_val = passengers.min()
max_val = passengers.max()
mean_val = passengers.mean()

# Xác định thời điểm đạt Min và Max
min_date = passengers.idxmin()
max_date = passengers.idxmax()

fig, ax = plt.subplots(figsize=(12, 4.5))

# Vẽ chuỗi gốc
ax.plot(passengers.index, passengers,
        color='steelblue', linewidth=1.8, label='Passengers')

# Vẽ 3 đường ngang
ax.axhline(min_val, color='green', linestyle='--',
           linewidth=1.5, label=f'Min = {min_val:.2f}')

ax.axhline(max_val, color='red', linestyle='--',
           linewidth=1.5, label=f'Max = {max_val:.2f}')

ax.axhline(mean_val, color='orange', linestyle='--',
           linewidth=1.5, label=f'Mean = {mean_val:.2f}')

# Đánh dấu Min và Max ngay trên đường Line Plot
ax.scatter(min_date, min_val, color='green',
           s=90, zorder=5, edgecolor='black')

ax.scatter(max_date, max_val, color='red',
           s=90, zorder=5, edgecolor='black')

# Tìm vị trí đường Line Plot cắt mức Mean
# Sử dụng nội suy tuyến tính giữa hai quan sát liên tiếp
values = passengers.to_numpy()
crossings = [
    i for i in range(len(values) - 1)
    if (values[i] - mean_val) * (values[i + 1] - mean_val) <= 0
    and values[i] != values[i + 1]
]

if crossings:
    i = crossings[0]
    ratio = (mean_val - values[i]) / (values[i + 1] - values[i])
    mean_date = passengers.index[i] + (
        passengers.index[i + 1] - passengers.index[i]
    ) * ratio

    ax.scatter(mean_date, mean_val, color='orange',
               s=90, zorder=5, edgecolor='black')

ax.set(
    title="Lượng hành khách theo tháng (1949–1960)",
    xlabel="Thời gian",
    ylabel="Hành khách (nghìn)"
)

ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

print(f"Min  = {min_val:.2f}")
print(f"Max  = {max_val:.2f}")
print(f"Mean = {mean_val:.2f}")

## Seasonal Plot
seasonal_data = data.assign(
    Year=data.index.year,
    Month_num=data.index.month,
)
seasonal_table = seasonal_data.pivot(index="Month_num", columns="Year", values="Passengers")
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Biểu đồ seasonal plot giữ nguyên
for year in seasonal_table.columns:
    axes[0].plot(range(1, 13), seasonal_table[year], marker="o", linewidth=1, label=str(year))
axes[0].set_xticks(range(1, 13), month_labels)
axes[0].set(title="Seasonal plot theo tháng", xlabel="Tháng", ylabel="Hành khách (nghìn)")
axes[0].grid(alpha=0.25)
axes[0].legend(title="Năm", bbox_to_anchor=(1.02, 1), loc="upper left", ncol=2, fontsize=8)

# Heatmap đã chỉnh sửa
image = axes[1].imshow(seasonal_table.values, aspect="auto", origin="upper", cmap="YlOrRd")

axes[1].set(title="Heatmap tháng × năm", xlabel="Năm", ylabel="Tháng")
axes[1].set_xticks(range(len(seasonal_table.columns)), seasonal_table.columns, rotation=45)
axes[1].set_yticks(range(12), month_labels)

# Tạo đường kẻ ô rõ ràng
axes[1].set_xticks(np.arange(-0.5, len(seasonal_table.columns), 1), minor=True)
axes[1].set_yticks(np.arange(-0.5, len(seasonal_table.index), 1), minor=True)
axes[1].grid(which="minor", color="black", linestyle="-", linewidth=0.8)
axes[1].tick_params(which="minor", bottom=False, left=False)

# Ghi giá trị vào từng ô
for i in range(seasonal_table.shape[0]):
    for j in range(seasonal_table.shape[1]):
        value = seasonal_table.iloc[i, j]
        axes[1].text(
            j, i, f"{value:.0f}",
            ha="center", va="center",
            color="black", fontsize=8
        )

fig.colorbar(image, ax=axes[1], label="Hành khách (nghìn)")
plt.tight_layout()
plt.show()

monthly_mean = seasonal_data.groupby("Month_num")["Passengers"].mean()
annual_mean = seasonal_data.groupby("Year")["Passengers"].mean()
print(f"\nTrung bình tháng cao nhất: {month_labels[monthly_mean.idxmax() - 1]} ({monthly_mean.max():.1f})")
print(f"Trung bình tháng thấp nhất: {month_labels[monthly_mean.idxmin() - 1]} ({monthly_mean.min():.1f})")
print(f"Trung bình năm đầu/cuối: {annual_mean.iloc[0]:.1f} ({annual_mean.index[0]}) / {annual_mean.iloc[-1]:.1f} ({annual_mean.index[-1]})")
print(f"Thay đổi từ đầu đến cuối chuỗi: {(passengers.iloc[-1] / passengers.iloc[0] - 1) * 100:.1f}%")

##  Phân rã STL: xu hướng, mùa vụ và phần dư
stl_result = STL(passengers, period=SEASONAL_PERIOD, robust=True).fit()
fig = stl_result.plot()
fig.set_size_inches(12, 8)
fig.suptitle("Phân rã STL: xu hướng, mùa vụ và phần dư", y=1.01)
plt.tight_layout()
plt.show()
# ---------------------------------------------------------------- 

## 2. ADF chuỗi gốc → kiểm tra tính không dừng
def run_adf(series):
    result = adfuller(series.dropna(), autolag="AIC", result_object=False)
    return {"ADF statistic": result[0], "p-value": result[1], "lags": result[2], "observations": result[3]}


adf_original = run_adf(passengers)
print(f"ADF statistic chuỗi gốc: {adf_original['ADF statistic']:.6f}")
print(f"p-value chuỗi gốc: {adf_original['p-value']:.6f}")
print(f"Số độ trễ: {adf_original['lags']}; số quan sát dùng trong kiểm định: {adf_original['observations']}")
print("Kết luận:", "bác bỏ H₀ ở mức 5%." if adf_original["p-value"] < ALPHA else "chưa đủ bằng chứng bác bỏ H₀; chuỗi có dấu hiệu không dừng.")

# ---------------------------------------------------------------- 

## 3. Sai phân thường + mùa vụ
difference_series = {
    "Chuỗi gốc": passengers,
    "Sai phân thường (d=1)": passengers.diff(1),
    "Sai phân mùa vụ (D=1, s=12)": passengers.diff(SEASONAL_PERIOD),
    "Sai phân kết hợp (d=1, D=1, s=12)": passengers.diff(1).diff(SEASONAL_PERIOD),
}

adf_rows = []
for label, series in difference_series.items():
    result = run_adf(series)
    adf_rows.append({"Chuỗi": label, **result, "Dừng ở mức 5%": result["p-value"] < ALPHA})
adf_table = pd.DataFrame(adf_rows).set_index("Chuỗi")
print(adf_table.round(6).to_string())

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(passengers)
axes[0].set_title("Chuỗi gốc")
axes[1].plot(difference_series["Sai phân thường (d=1)"], color="tab:orange")
axes[1].set_title("Sai phân thường: d = 1")
axes[2].plot(difference_series["Sai phân kết hợp (d=1, D=1, s=12)"], color="tab:green")
axes[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
axes[2].set_title("Sai phân thường và mùa vụ: d = 1, D = 1, s = 12")
axes[2].set_xlabel("Thời gian")
for ax in axes:
    ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

print("\nDiễn giải: p-value ADF nhỏ hơn 0,05 là bằng chứng chuỗi dừng theo kiểm định; đây không phải bảo đảm cho mọi cấu hình kiểm định.")

# ---------------------------------------------------------------- 

## 4. ACF/PACF → xác định (p, q, P, Q)
train = passengers.iloc[:-TEST_MONTHS]
test = passengers.iloc[-TEST_MONTHS:]
train_diff = train.diff(1).diff(SEASONAL_PERIOD).dropna()

print(f"Train: {train.index[0]:%Y-%m} đến {train.index[-1]:%Y-%m} ({len(train)} tháng)")
print(f"Test:  {test.index[0]:%Y-%m} đến {test.index[-1]:%Y-%m} ({len(test)} tháng)")

fig, axes = plt.subplots(2, 1, figsize=(11, 7))
plot_acf(train_diff, lags=36, ax=axes[0])
axes[0].set_title("ACF: train sau sai phân d=1, D=1, s=12")
plot_pacf(train_diff, lags=36, method="ywm", ax=axes[1])
axes[1].set_title("PACF: train sau sai phân d=1, D=1, s=12")
plt.tight_layout()
plt.show()

acf_values = acf(train_diff, nlags=36)
pacf_values = pacf(train_diff, nlags=36, method="ywm")
key_lags = [1, 2, 12, 13, 24, 36]
correlation_table = pd.DataFrame(
    {"ACF": [acf_values[lag] for lag in key_lags], "PACF": [pacf_values[lag] for lag in key_lags]},
    index=key_lags,
)
correlation_table.index.name = "Lag"
print("\nACF/PACF tại một số lag:")
print(correlation_table.round(3).to_string())
print(f"Ngưỡng tham khảo gần đúng 95%: ±{1.96 / np.sqrt(len(train_diff)):.3f}")

# ---------------------------------------------------------------- 

## 5. Fit SARIMA
sarima_candidates = [
    ((0, 1, 1), (0, 1, 0, SEASONAL_PERIOD)),
    ((0, 1, 1), (0, 1, 1, SEASONAL_PERIOD)),
]

sarima_fits = {}
for order, seasonal_order in sarima_candidates:
    label = f"SARIMA{order}x{seasonal_order}"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        fit = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    sarima_fits[label] = {"order": order, "seasonal_order": seasonal_order, "fit": fit}

aic_table = pd.DataFrame(
    {label: {"AIC": item["fit"].aic, "BIC": item["fit"].bic} for label, item in sarima_fits.items()}
).T.sort_values("AIC")
print("SARIMA ứng viên (fit trên train):")
print(aic_table.round(2).to_string())

best_sarima_label = aic_table.index[0]
best_sarima = sarima_fits[best_sarima_label]
sarima_fit = best_sarima["fit"]
print(f"\nChọn {best_sarima_label} theo AIC thấp nhất trên train.")


### Chẩn đoán phần dư SARIMA (bổ sung)
sarima_residuals = pd.Series(sarima_fit.resid).dropna()
ljung_box = acorr_ljungbox(sarima_residuals, lags=[12], model_df=2, return_df=True)
print("Kiểm định Ljung–Box trên phần dư SARIMA (lag 12):")
print(ljung_box.round(4).to_string())

fig, axes = plt.subplots(2, 1, figsize=(11, 6))
axes[0].plot(sarima_residuals)
axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
axes[0].set(title="Phần dư SARIMA theo thời gian", xlabel="Thời gian", ylabel="Phần dư")
plot_acf(sarima_residuals, lags=36, ax=axes[1])
axes[1].set_title("ACF phần dư SARIMA")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------- 
## 6. Forecast và đánh giá SARIMA (MAE, RMSE)

## Dự báo + tính MAE, RMSE
sarima_forecast = sarima_fit.get_forecast(steps=len(test)).predicted_mean
sarima_forecast.index = test.index

def calculate_metrics(actual, predicted):
    errors = np.asarray(actual) - np.asarray(predicted)
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors**2)))


sarima_mae, sarima_rmse = calculate_metrics(test, sarima_forecast)
sarima_metrics = pd.DataFrame(
    {"MAE": [sarima_mae], "RMSE": [sarima_rmse]},
    index=[best_sarima_label],
)
print("Đánh giá SARIMA trên tập test (đơn vị: nghìn hành khách):")
print(sarima_metrics.round(2).to_string())

sarima_forecast_table = pd.DataFrame({"Thực tế": test, "Dự báo SARIMA": sarima_forecast})
print("\nDự báo SARIMA chi tiết:")
print(sarima_forecast_table.round(1).to_string())

## Vẽ biểu đồ
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(train.index, train, label="Train", color="tab:blue")
ax.plot(test.index, test, label="Thực tế (test)", color="black", marker="o")
ax.plot(test.index, sarima_forecast, label=f"Dự báo {best_sarima_label}", marker="o")
ax.set(title="Forecast SARIMA trên 12 tháng cuối", xlabel="Thời gian", ylabel="Hành khách (nghìn)")
ax.legend()
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------- 


## 7. So sánh với ARIMA
ARIMA_ORDER = (1, 1, 1)
arima_fit = ARIMA(train, order=ARIMA_ORDER, trend="n").fit()
arima_forecast = arima_fit.get_forecast(steps=len(test)).predicted_mean
arima_forecast.index = test.index

arima_mae, arima_rmse = calculate_metrics(test, arima_forecast)
results = pd.DataFrame(
    {
        "MAE": [sarima_mae, arima_mae],
        "RMSE": [sarima_rmse, arima_rmse],
    },
    index=[best_sarima_label, f"ARIMA{ARIMA_ORDER}"],
).sort_values("RMSE")
print(f"ARIMA{ARIMA_ORDER}: AIC = {arima_fit.aic:.2f}; BIC = {arima_fit.bic:.2f}")
print("\nSo sánh trên cùng tập test (đơn vị: nghìn hành khách):")
print(results.round(2).to_string())

forecast_table = pd.DataFrame(
    {"Thực tế": test, "Dự báo SARIMA": sarima_forecast, "Dự báo ARIMA": arima_forecast}
)
print("\nDự báo chi tiết:")
print(forecast_table.round(1).to_string())

best_test_model = results.index[0]
print(f"\nMô hình có RMSE thấp nhất trên tập test: {best_test_model}")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(train.index, train, label="Train", color="tab:blue")
ax.plot(test.index, test, label="Thực tế (test)", color="black", marker="o")
ax.plot(test.index, sarima_forecast, label=f"Dự báo {best_sarima_label}", marker="o")
ax.plot(test.index, arima_forecast, label=f"Dự báo ARIMA{ARIMA_ORDER}", marker="o")
ax.set(title="So sánh SARIMA với ARIMA trên 12 tháng cuối", xlabel="Thời gian", ylabel="Hành khách (nghìn)")
ax.legend()
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------- 

## Kết luận
best_row = results.iloc[0]
worst_row = results.iloc[-1]
relative_rmse_improvement = (worst_row["RMSE"] - best_row["RMSE"]) / worst_row["RMSE"] * 100
combined_adf_pvalue = adf_table.loc["Sai phân kết hợp (d=1, D=1, s=12)", "p-value"]
lb_pvalue = ljung_box["lb_pvalue"].iloc[0]

print("1. Chuỗi có xu hướng tăng và mẫu mùa vụ lặp theo chu kỳ 12 tháng.")
print(f"2. ADF sau sai phân d=1, D=1, s=12: p-value = {combined_adf_pvalue:.6g}; "
      f"{'bác bỏ H₀ ở mức 5%' if combined_adf_pvalue < ALPHA else 'chưa bác bỏ H₀ ở mức 5%'}.")
print(f"3. SARIMA được chọn theo AIC train: {best_sarima_label} (AIC = {sarima_fit.aic:.2f}).")
print(f"4. Mô hình có RMSE test thấp nhất: {results.index[0]} "
      f"(MAE = {best_row['MAE']:.2f}; RMSE = {best_row['RMSE']:.2f}).")
print(f"5. RMSE giảm {relative_rmse_improvement:.1f}% so với mô hình còn lại trên giai đoạn test.")
print(f"6. Ljung–Box phần dư SARIMA tại lag 12: p-value = {lb_pvalue:.4f}; "
      f"{'chưa thấy tự tương quan còn lại có ý nghĩa ở mức 5%' if lb_pvalue >= ALPHA else 'còn dấu hiệu tự tương quan; nên xem xét cải thiện mô hình'}.")

print("\nLưu ý: đánh giá chỉ dùng một giai đoạn test 12 tháng; nên cân nhắc kiểm định cuốn chiếu (rolling-origin) trước khi triển khai dự báo thực tế.")



