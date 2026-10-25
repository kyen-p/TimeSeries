# Chuẩn bị thư viện và dữ liệu 
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



