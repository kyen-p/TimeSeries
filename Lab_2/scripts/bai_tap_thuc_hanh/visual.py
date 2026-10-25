"""Vẽ đồ thị Bài tập thực hành (Dataset 1 - a10) và lưu vào figure/bai_tap_thuc_hanh/.

Đầu vào : data/processed/a10_diff.csv
Đầu ra  : figure/bai_tap_thuc_hanh/01_*.png, 02_*.png, ...
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # chỉ lưu file, không mở cửa sổ
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# scripts/bai_tap_thuc_hanh/visual.py -> parents[2] là thư mục Lab gốc
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "dataset1_diff.csv"
FIG = ROOT / "figure" / "bai_tap_thuc_hanh"
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["date"], index_col="date").sort_index()
y = df["value"].astype(float)


def luu(ten_file):
    """Lưu hình hiện tại vào thư mục figure rồi đóng."""
    plt.tight_layout()
    plt.savefig(FIG / ten_file, dpi=150)
    plt.close()
    print("Đã lưu:", ten_file)


# ---------------------------------------------------------------- Bài 1
def bai1():
    # 01. Chuỗi gốc
    plt.figure(figsize=(13, 4))
    plt.plot(y.index, y, color="#246BFD")
    plt.title("Chuỗi gốc: giá trị hằng tháng (07/1991 - 06/2008)")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị")
    plt.grid(True, alpha=0.3)
    luu("01_chuoi_goc.png")

    # 02. Xu hướng (trung bình trượt 12 tháng, căn giữa)
    ma12 = y.rolling(window=12).mean()
    trend = (ma12.shift(-5) + ma12.shift(-6)) / 2
    plt.figure(figsize=(13, 4))
    plt.plot(y.index, y, color="lightgray", label="Chuỗi gốc")
    plt.plot(trend.index, trend, color="red", linewidth=2, label="Xu hướng")
    plt.title("Xu hướng của chuỗi")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị")
    plt.legend()
    plt.grid(True, alpha=0.3)
    luu("02_xu_huong.png")

    # 03. Hệ số mùa vụ theo tháng
    ratio = y / trend
    seasonal_index = ratio.groupby(ratio.index.month).mean()
    seasonal_index = seasonal_index / seasonal_index.mean()
    plt.figure(figsize=(10, 4))
    plt.bar(seasonal_index.index, seasonal_index.values, color="#00A844")
    plt.axhline(1, color="gray", linewidth=1)
    plt.xticks(range(1, 13))
    plt.title("Hệ số mùa vụ theo tháng")
    plt.xlabel("Tháng")
    plt.ylabel("Hệ số mùa vụ")
    plt.grid(True, alpha=0.3)
    luu("03_he_so_mua_vu.png")


# ---------------------------------------------------------------- Bài 2
def bai2():
    # 04. Trung bình trượt và độ lệch chuẩn trượt
    roll_mean = y.rolling(window=12).mean()
    roll_std = y.rolling(window=12).std()
    plt.figure(figsize=(13, 4))
    plt.plot(y.index, y, color="lightgray", label="Chuỗi gốc")
    plt.plot(roll_mean.index, roll_mean, color="orange", linewidth=2,
             label="Trung bình trượt 12 tháng")
    plt.plot(roll_std.index, roll_std, color="green", linewidth=2,
             label="Độ lệch chuẩn trượt 12 tháng")
    plt.title("Trung bình trượt và độ lệch chuẩn trượt (cửa sổ 12 tháng)")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị")
    plt.legend()
    plt.grid(True, alpha=0.3)
    luu("04_trung_binh_do_lech_chuan_truot.png")

    # 05. ADF statistic so với các giá trị tới hạn
    stat, pval, lags, nobs, crit, _ = adfuller(y, regression="c", autolag="AIC")
    plt.figure(figsize=(10, 3.2))
    for muc, mau in zip(["1%", "5%", "10%"], ["#d62728", "#ff7f0e", "#bcbd22"]):
        plt.axvline(crit[muc], color=mau, linestyle="--",
                    label=f"Tới hạn {muc} = {crit[muc]:.3f}")
    plt.axvline(stat, color="blue", linewidth=3, label=f"ADF statistic = {stat:.3f}")
    plt.axvspan(-5, crit["5%"], color="green", alpha=0.12)
    plt.text(-4.9, 0.5, "Vùng dừng\n(statistic < tới hạn 5%)", va="center", fontsize=9)
    plt.xlim(-5, 4)
    plt.yticks([])
    plt.title("ADF statistic so với các giá trị tới hạn")
    plt.xlabel("Giá trị")
    plt.legend(loc="upper left", fontsize=8)
    luu("05_adf_so_voi_toi_han.png")


# ---------------------------------------------------------------- Bài 3
y_diff1 = y.diff().dropna()
y_diff2 = y_diff1.diff().dropna()


def bai3():
    # 06. Sai phân bậc 1
    plt.figure(figsize=(13, 4))
    plt.plot(y_diff1.index, y_diff1, color="#246BFD")
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Chuỗi sau sai phân bậc 1")
    plt.xlabel("Thời gian")
    plt.ylabel("Thay đổi so với tháng trước")
    plt.grid(True, alpha=0.3)
    luu("06_sai_phan_bac1.png")

    # 07. Sai phân bậc 2
    plt.figure(figsize=(13, 4))
    plt.plot(y_diff2.index, y_diff2, color="#00A844")
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Chuỗi sau sai phân bậc 2")
    plt.xlabel("Thời gian")
    plt.ylabel("Thay đổi của sai phân")
    plt.grid(True, alpha=0.3)
    luu("07_sai_phan_bac2.png")

    # 08. ACF sau sai phân bậc 1 và bậc 2 (đỉnh mùa vụ lag 12, 24, 36)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    plot_acf(y_diff1, lags=36, ax=axes[0], title="ACF sau sai phân bậc 1")
    plot_acf(y_diff2, lags=36, ax=axes[1], title="ACF sau sai phân bậc 2")
    for ax in axes:
        ax.set_xlabel("Lag")
    luu("08_acf_sau_sai_phan.png")


# ---------------------------------------------------------------- Bài 4
def bai4():
    # 09. ACF và PACF của chuỗi sau sai phân bậc 1
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(y_diff1, lags=36, ax=ax[0], title="ACF - chuỗi sau sai phân bậc 1")
    plot_pacf(y_diff1, lags=36, ax=ax[1], method="ywm",
              title="PACF - chuỗi sau sai phân bậc 1")
    for a in ax:
        a.set_xlabel("Lag")
    luu("09_acf_pacf.png")


# ---------------------------------------------------------------- Bài 5
yt = y.asfreq("MS")
n_train = int(len(yt) * 0.8)
train, test = yt.iloc[:n_train], yt.iloc[n_train:]
BEST = (1, 1, 1)                      # mô hình tốt nhất theo AIC/BIC


def bai5():
    # 10. Chia train / test
    plt.figure(figsize=(13, 4))
    plt.plot(train, label="Train", color="steelblue")
    plt.plot(test, label="Test", color="darkorange")
    plt.title("Chia dữ liệu train / test")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị")
    plt.legend()
    plt.grid(True, alpha=0.3)
    luu("10_train_test.png")

    model = ARIMA(train, order=BEST).fit()

    # 11. Phần dư và ACF của phần dư
    res = model.resid.iloc[1:]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].plot(res, color="steelblue")
    ax[0].axhline(0, color="red", linestyle="--")
    ax[0].set_title(f"Phần dư của ARIMA{BEST} theo thời gian")
    ax[0].set_xlabel("Thời gian")
    ax[0].set_ylabel("Residual")
    ax[0].grid(True, alpha=0.3)
    plot_acf(res, lags=36, ax=ax[1])
    ax[1].set_title("ACF của phần dư")
    luu("11_residuals.png")

    # 12. Train / Test / Forecast
    du_bao = model.forecast(steps=len(test))
    plt.figure(figsize=(13, 5))
    plt.plot(train, label="Train", color="steelblue")
    plt.plot(test, label="Test (thực tế)", color="darkorange")
    plt.plot(du_bao, label=f"Forecast ARIMA{BEST}", color="green", linestyle="--")
    plt.title("Dự báo ARIMA trên tập test")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị")
    plt.legend()
    plt.grid(True, alpha=0.3)
    luu("12_forecast.png")


if __name__ == "__main__":
    bai1()
    bai2()
    bai3()
    bai4()
    bai5()