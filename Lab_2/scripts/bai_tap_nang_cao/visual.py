"""Vẽ đồ thị Bài tập nâng cao (Dataset 2 - AirPassengers, SARIMA) và lưu vào figure/bai_tap_nang_cao/.

Đầu vào : data/processed/dataset2_diff.csv
Đầu ra  : figure/bai_tap_nang_cao/01_*.png, 02_*.png, ...
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "dataset2_diff.csv"
FIG = ROOT / "figure" / "bai_tap_nang_cao"
FIG.mkdir(parents=True, exist_ok=True)

S = 12                 # chu kỳ mùa vụ
TEST_MONTHS = 12       # 12 tháng cuối làm tập test

df = pd.read_csv(DATA, parse_dates=["date"], index_col="date").sort_index()
df.index = pd.DatetimeIndex(df.index, freq="MS")
y = df["value"].astype(float)
years = sorted(y.index.year.unique())
YEAR_COLORS = dict(zip(years, plt.cm.turbo(np.linspace(0.05, 0.95, len(years)))))
THANG = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

train, test = y.iloc[:-TEST_MONTHS], y.iloc[-TEST_MONTHS:]


# hàm dùng chung
def so(x, nd=1):
    """Định dạng số kiểu Việt Nam: dùng dấu phẩy thập phân."""
    return f"{x:.{nd}f}".replace(".", ",")


def luu(ten_file):
    """Lưu hình hiện tại vào thư mục figure rồi đóng."""
    plt.tight_layout()
    plt.savefig(FIG / ten_file, dpi=150)
    plt.close()
    print("Đã lưu:", ten_file)


def chia_nam(ax, s):
    """Vạch chấm dọc ngăn từng năm (1/1) và trục x ghi số năm."""
    for yr in years:
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="0.55", lw=0.7, ls=":")
    ax.set_xlim(s.index[0] - pd.Timedelta(days=15), s.index[-1] + pd.Timedelta(days=45))
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def duong_thong_ke(ax, s, zero_ref=False):
    """Đường Mean / Max / Min, đánh dấu điểm Max và Min trên đường."""
    s = s.dropna()
    ax.axhline(s.mean(), color="#e08a00", ls="--", lw=1.4, label=f"Mean = {so(s.mean())}")
    ax.axhline(s.max(), color="#d62728", ls="--", lw=1.2, label=f"Max = {so(s.max())} ({s.idxmax():%m/%Y})")
    ax.axhline(s.min(), color="#2ca02c", ls="--", lw=1.2, label=f"Min = {so(s.min())} ({s.idxmin():%m/%Y})")
    ax.scatter([s.idxmax()], [s.max()], color="#d62728", s=70, zorder=5, edgecolor="k")
    ax.scatter([s.idxmin()], [s.min()], color="#2ca02c", s=70, zorder=5, edgecolor="k")
    if zero_ref:
        ax.axhline(0, color="k", lw=0.8)


def sai_so(actual, pred):
    e = np.asarray(actual) - np.asarray(pred)
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))


# Mục 1: nhận diện xu hướng và mùa vụ
def muc1():
    # 01. Chuỗi gốc
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y.index, y, color="steelblue", lw=1.8, label="Passengers")
    duong_thong_ke(ax, y)
    ax.set(title="Chuỗi gốc: lượng hành khách hằng tháng (1949–1960)", xlabel="Năm", ylabel="Hành khách (nghìn)")
    chia_nam(ax, y)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    luu("01_chuoi_goc.png")

    bang = pd.pivot_table(df.assign(Y=df.index.year, M=df.index.month), index="M", columns="Y", values="value")

    # 02. Seasonal plot
    fig, ax = plt.subplots(figsize=(12, 5))
    for yr in bang.columns:
        ax.plot(range(1, 13), bang[yr], marker="o", lw=1, color=YEAR_COLORS[yr], label=str(yr))
    mm = bang.mean(axis=1)
    ax.plot(range(1, 13), mm, color="k", lw=2.4, ls="--",
            label=f"Mean theo tháng (đỉnh {THANG[mm.idxmax() - 1]} {so(mm.max(), 0)}, đáy {THANG[mm.idxmin() - 1]} {so(mm.min(), 0)})")
    ax.scatter([mm.idxmax(), mm.idxmin()], [mm.max(), mm.min()], color="k", s=80, zorder=6)
    ax.set_xticks(range(1, 13), THANG)
    ax.set(title="Seasonal plot theo tháng", xlabel="Tháng", ylabel="Hành khách (nghìn)")
    ax.grid(alpha=0.25)
    ax.legend(title="Năm", bbox_to_anchor=(1.02, 1), loc="upper left", ncol=2, fontsize=8)
    luu("02_seasonal_plot.png")

    # 03. Heatmap tháng × năm
    fig, ax = plt.subplots(figsize=(13, 6.8))
    v = bang.values
    im = ax.imshow(v, aspect="auto", origin="upper", cmap="YlOrRd")
    ax.set_xticks(range(len(years)), years)
    ax.set_yticks(range(12), THANG)
    ax.set_xticks(np.arange(-0.5, len(years), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 12, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for i in range(12):
        for j in range(len(years)):
            ax.text(j, i, f"{v[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v[i, j] > 450 else "black")
    for (i, j), c in ((np.unravel_index(v.argmax(), v.shape), "blue"), (np.unravel_index(v.argmin(), v.shape), "green")):
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=c, lw=3))
    for j, yr in enumerate(years):
        c = bang[yr]
        ax.text(j, 12.15, f"Min {c.min():.0f}\nMean {c.mean():.0f}\nMax {c.max():.0f}",
                ha="center", va="top", fontsize=7.5, clip_on=False)
    for i in range(12):
        ax.text(len(years) - 0.35, i, f"Mean {bang.iloc[i].mean():.0f}", ha="left", va="center", fontsize=8, clip_on=False)
    ax.set(title=f"Heatmap tháng × năm (viền xanh dương = Max {v.max():.0f}; viền xanh lá = Min {v.min():.0f})")
    fig.colorbar(im, ax=ax, label="Hành khách (nghìn)", pad=0.12)
    fig.subplots_adjust(bottom=0.2, right=0.9)
    plt.savefig(FIG / "03_heatmap.png", dpi=150)
    plt.close()
    print("Đã lưu: 03_heatmap.png")

    # 04. Phân rã STL
    stl = STL(y, period=S, robust=True).fit()
    phan = [("Observed (chuỗi gốc)", stl.observed, "#1f77b4"), ("Trend (xu hướng)", stl.trend, "orange"),
            ("Seasonal (mùa vụ)", stl.seasonal, "green"), ("Residual (phần dư)", stl.resid, "gray")]
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    for k, (ax, (ten, s, c)) in enumerate(zip(axes, phan)):
        ax.plot(s, color=c)
        duong_thong_ke(ax, s, zero_ref=(k >= 2))
        chia_nam(ax, s)
        ax.set_ylabel(ten)
        ax.legend(loc="upper left", fontsize=8, ncol=3)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Năm")
    fig.suptitle("Phân rã STL: xu hướng, mùa vụ và phần dư")
    luu("04_stl.png")


# Mục 3: sai phân
def muc3():
    chuoi = [("Chuỗi gốc", y, "#1f77b4"), ("Sai phân thường (d=1)", df["diff1"], "tab:orange"),
             ("Sai phân mùa vụ (D=1, s=12)", df["diff12"], "tab:red"),
             ("Sai phân kết hợp (d=1, D=1, s=12)", df["diff_both"], "tab:green")]
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for k, (ax, (ten, s, c)) in enumerate(zip(axes, chuoi)):
        ax.plot(s, color=c, lw=1.5)
        duong_thong_ke(ax, s, zero_ref=(k >= 1))
        chia_nam(ax, y)
        ax.set_title(ten, fontsize=10)
        ax.legend(loc="upper left", fontsize=8, ncol=3, title=f"Std = {so(s.dropna().std())}", title_fontsize=8)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Năm")
    luu("05_sai_phan.png")


# Mục 4: ACF / PACF trên train đã sai phân
def muc4():
    td = train.diff(1).diff(S).dropna()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    plot_acf(td, lags=36, ax=axes[0])
    axes[0].set_title("ACF: train sau sai phân d=1, D=1, s=12")
    plot_pacf(td, lags=36, method="ywm", ax=axes[1])
    axes[1].set_title("PACF: train sau sai phân d=1, D=1, s=12")
    luu("06_acf_pacf.png")


# Mục 5-7: SARIMA, phần dư, dự báo, so sánh với ARIMA
def muc5_7():
    sarima = SARIMAX(train, order=(0, 1, 1), seasonal_order=(0, 1, 1, S), trend="n",
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    nhan = "SARIMA(0,1,1)x(0,1,1,12)"

    # 07. Phần dư SARIMA
    r = pd.Series(sarima.resid).dropna()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    axes[0].plot(r)
    axes[0].axhline(0, color="black", ls="--", lw=0.8)
    axes[0].set(title="Phần dư SARIMA theo thời gian", xlabel="Thời gian", ylabel="Phần dư")
    plot_acf(r, lags=36, ax=axes[1])
    axes[1].set_title("ACF phần dư SARIMA")
    luu("07_phan_du_sarima.png")

    f_sarima = sarima.get_forecast(steps=len(test)).predicted_mean
    f_sarima.index = test.index
    mae_s, rmse_s = sai_so(test, f_sarima)

    # 08. Dự báo SARIMA
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train.index, train, color="tab:blue", label="Train")
    ax.plot(test.index, test, color="black", marker="o", label="Thực tế (test)")
    ax.plot(test.index, f_sarima, marker="o", label=f"Dự báo {nhan}")
    ax.set(title=f"Dự báo SARIMA trên 12 tháng cuối (MAE = {so(mae_s, 2)}; RMSE = {so(rmse_s, 2)})",
           xlabel="Thời gian", ylabel="Hành khách (nghìn)")
    ax.legend()
    ax.grid(alpha=0.25)
    luu("08_du_bao_sarima.png")

    # 09. So sánh với ARIMA(1,1,1)
    f_arima = ARIMA(train, order=(1, 1, 1), trend="n").fit().get_forecast(steps=len(test)).predicted_mean
    f_arima.index = test.index
    mae_a, rmse_a = sai_so(test, f_arima)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train.index, train, color="tab:blue", label="Train")
    ax.plot(test.index, test, color="black", marker="o", label="Thực tế (test)")
    ax.plot(test.index, f_sarima, marker="o", label=f"{nhan} (RMSE = {so(rmse_s, 2)})")
    ax.plot(test.index, f_arima, marker="o", label=f"ARIMA(1,1,1) (RMSE = {so(rmse_a, 2)})")
    ax.set(title="So sánh SARIMA với ARIMA trên 12 tháng cuối", xlabel="Thời gian", ylabel="Hành khách (nghìn)")
    ax.legend()
    ax.grid(alpha=0.25)
    luu("09_sarima_vs_arima.png")


if __name__ == "__main__":
    muc1()
    muc3()
    muc4()
    muc5_7()