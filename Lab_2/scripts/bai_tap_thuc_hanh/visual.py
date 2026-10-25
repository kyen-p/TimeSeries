"""Vẽ đồ thị Bài tập thực hành (Dataset 1 - a10) và lưu vào figure/bai_tap_thuc_hanh/.

Đầu vào : data/processed/dataset1_diff.csv
Đầu ra  : figure/bai_tap_thuc_hanh/01_*.png, 02_*.png, ...
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "dataset1_diff.csv"
FIG = ROOT / "figure" / "bai_tap_thuc_hanh"
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["date"], index_col="date").sort_index()
y = df["value"].astype(float)


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


def dinh_dang_truc_thoi_gian(ax):
    """Trục thời gian dạng MM/YYYY, vạch chính mỗi 2 năm, vạch phụ mỗi năm."""
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))


def ghi(ax, ts, nd, mau, dy_dau=14):
    """Đánh dấu và ghi giá trị ở đầu và cuối đường."""
    ax.plot(ts.index[[0, -1]], ts.iloc[[0, -1]], "o", color=mau)
    ax.annotate(f"{so(ts.iloc[0], nd)}\n({ts.index[0]:%m/%Y})", (ts.index[0], ts.iloc[0]),
                textcoords="offset points", xytext=(0, dy_dau), ha="center", color=mau, fontweight="bold")
    ax.annotate(f"{so(ts.iloc[-1], nd)}\n({ts.index[-1]:%m/%Y})", (ts.index[-1], ts.iloc[-1]),
                textcoords="offset points", xytext=(10, -8), ha="left", color=mau, fontweight="bold")


def ve_sai_phan(x, mau, tieu_de, nhan_y, ten_file):
    """Vẽ chuỗi sai phân: mức 0, dải ±1 độ lệch chuẩn, điểm min và max."""
    tb, sd = x.mean(), x.std()
    t_min, t_max = x.idxmin(), x.idxmax()
    plt.figure(figsize=(13, 4.8))
    plt.plot(x.index, x, color=mau, linewidth=1.3)
    plt.axhline(0, color="red", linestyle="--", linewidth=1, label=f"Mức 0 (trung bình = {so(tb, 3)})")
    plt.axhspan(-sd, sd, color="gray", alpha=0.15, label=f"±1 độ lệch chuẩn (±{so(sd, 3)})")
    plt.plot([t_min, t_max], [x.min(), x.max()], "o", color="black")
    plt.annotate(f"Min = {so(x.min(), 2)} ({t_min:%m/%Y})", (t_min, x.min()), textcoords="offset points",
                 xytext=(8, -8), ha="right", va="top", fontweight="bold")
    plt.annotate(f"Max = {so(x.max(), 2)} ({t_max:%m/%Y})", (t_max, x.max()), textcoords="offset points",
                 xytext=(8, 8), ha="right", va="bottom", fontweight="bold")
    mien = x.max() - x.min()
    plt.ylim(x.min() - 0.14 * mien, x.max() + 0.14 * mien)
    plt.title(tieu_de)
    plt.xlabel("Thời gian")
    plt.ylabel(nhan_y)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    dinh_dang_truc_thoi_gian(plt.gca())
    luu(ten_file)


# Bài 1
ma12 = y.rolling(window=12).mean()
trend = (ma12.shift(-5) + ma12.shift(-6)) / 2   # căn giữa cửa sổ 12 tháng


def bai1():
    # 01. Chuỗi gốc: min, max, trung bình
    tb, vmin, vmax = y.mean(), y.min(), y.max()
    t_min, t_max = y.idxmin(), y.idxmax()
    plt.figure(figsize=(13, 4.5))
    plt.plot(y.index, y, color="#246BFD", label="Chuỗi gốc")
    plt.axhline(tb, color="gray", linestyle="--", linewidth=1.2, label=f"Trung bình = {so(tb, 2)}")
    plt.plot([t_min, t_max], [vmin, vmax], "o", color="red")
    plt.annotate(f"Min = {so(vmin, 2)}\n({t_min:%m/%Y})", (t_min, vmin), textcoords="offset points",
                 xytext=(14, -6), ha="left", va="top", color="red", fontweight="bold")
    plt.annotate(f"Max = {so(vmax, 2)}\n({t_max:%m/%Y})", (t_max, vmax), textcoords="offset points",
                 xytext=(-14, 1), ha="right", va="center", color="red", fontweight="bold")
    plt.title("Chuỗi gốc: giá trị hằng tháng (07/1991 - 06/2008)")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị")
    plt.ylim(-3, 32)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    dinh_dang_truc_thoi_gian(plt.gca())
    luu("01_chuoi_goc.png")

    # 02. Xu hướng (trung bình trượt 12 tháng, căn giữa)
    tr = trend.dropna()
    plt.figure(figsize=(13, 4.8))
    plt.plot(y.index, y, color="lightgray", label="Chuỗi gốc")
    plt.plot(tr.index, tr, color="red", linewidth=2, label="Xu hướng")
    plt.plot(tr.index[[0, -1]], tr.iloc[[0, -1]], "o", color="darkred")
    plt.annotate(f"{so(tr.iloc[0])}\n({tr.index[0]:%m/%Y})", (tr.index[0], tr.iloc[0]),
                 textcoords="offset points", xytext=(0, 14), ha="center", color="darkred", fontweight="bold")
    plt.annotate(f"{so(tr.iloc[-1])}\n({tr.index[-1]:%m/%Y})", (tr.index[-1], tr.iloc[-1]),
                 textcoords="offset points", xytext=(10, -8), ha="left", color="darkred", fontweight="bold")
    plt.title(f"Xu hướng của chuỗi: tăng gấp khoảng {so(tr.iloc[-1] / tr.iloc[0])} lần")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.set_xlim(y.index[0] - pd.DateOffset(months=6), y.index[-1] + pd.DateOffset(months=24))
    dinh_dang_truc_thoi_gian(ax)
    luu("02_xu_huong.png")

    # 03. Hệ số mùa vụ theo tháng
    ratio = y / trend
    seasonal_index = ratio.groupby(ratio.index.month).mean()
    seasonal_index = seasonal_index / seasonal_index.mean()   # trung bình 12 hệ số = 1
    thang_cao, thang_thap = seasonal_index.idxmax(), seasonal_index.idxmin()
    mau = ["#d62728" if m == thang_cao else "#1f77b4" if m == thang_thap else "#00A844"
           for m in seasonal_index.index]
    plt.figure(figsize=(11, 4.8))
    cot = plt.bar(seasonal_index.index, seasonal_index.values, color=mau)
    for c, v in zip(cot, seasonal_index.values):
        plt.text(c.get_x() + c.get_width() / 2, v + 0.02, so(v, 3), ha="center", va="bottom", fontsize=9)
    plt.axhline(1, color="gray", linestyle="--", linewidth=1, label="Mức trung bình = 1")
    plt.xticks(range(1, 13), [f"Tháng {m}" for m in range(1, 13)], fontsize=8)
    plt.ylim(0, seasonal_index.max() * 1.15)
    plt.title(f"Hệ số mùa vụ theo tháng (cao nhất: tháng {thang_cao}, thấp nhất: tháng {thang_thap})")
    plt.xlabel("Tháng")
    plt.ylabel("Hệ số mùa vụ")
    plt.legend(loc="upper center")
    plt.grid(True, axis="y", alpha=0.3)
    luu("03_he_so_mua_vu.png")


# Bài 2
def bai2():
    # 04. Trung bình trượt và độ lệch chuẩn trượt 
    roll_mean = y.rolling(window=12).mean().dropna()
    roll_std = y.rolling(window=12).std().dropna()
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    ax[0].plot(y.index, y, color="lightgray", label="Chuỗi gốc")
    ax[0].plot(roll_mean.index, roll_mean, color="orange", linewidth=2, label="Trung bình trượt 12 tháng")
    ghi(ax[0], roll_mean, 1, "darkorange")
    ax[0].set_title(f"Trung bình trượt (cửa sổ 12 tháng): tăng gấp khoảng {so(roll_mean.iloc[-1] / roll_mean.iloc[0])} lần")
    ax[0].set_ylabel("Giá trị")
    ax[0].legend(loc="upper left")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(roll_std.index, roll_std, color="green", linewidth=2, label="Độ lệch chuẩn trượt 12 tháng")
    ghi(ax[1], roll_std, 2, "darkgreen")
    ax[1].set_title(f"Độ lệch chuẩn trượt (cửa sổ 12 tháng): tăng gấp khoảng {so(roll_std.iloc[-1] / roll_std.iloc[0])} lần")
    ax[1].set_xlabel("Thời gian")
    ax[1].set_ylabel("Độ lệch chuẩn")
    ax[1].set_ylim(0, roll_std.max() * 1.25)
    ax[1].legend(loc="upper left")
    ax[1].grid(True, alpha=0.3)
    ax[1].set_xlim(y.index[0] - pd.DateOffset(months=6), y.index[-1] + pd.DateOffset(months=24))
    dinh_dang_truc_thoi_gian(ax[1])
    luu("04_trung_binh_do_lech_chuan_truot.png")

    # 05. ADF statistic so với các giá trị tới hạn
    stat, pval, lags, nobs, crit, _ = adfuller(y, regression="c", autolag="AIC")
    plt.figure(figsize=(10, 3.2))
    for muc, mau in zip(["1%", "5%", "10%"], ["#d62728", "#ff7f0e", "#bcbd22"]):
        plt.axvline(crit[muc], color=mau, linestyle="--", label=f"Tới hạn {muc} = {crit[muc]:.3f}")
    plt.axvline(stat, color="blue", linewidth=3, label=f"ADF statistic = {stat:.3f}")
    plt.axvspan(-5, crit["5%"], color="green", alpha=0.12)
    plt.text(-4.9, 0.5, "Vùng dừng\n(statistic < tới hạn 5%)", va="center", fontsize=9)
    plt.xlim(-5, 4)
    plt.yticks([])
    plt.title("ADF statistic so với các giá trị tới hạn")
    plt.xlabel("Giá trị")
    plt.legend(loc="upper left", fontsize=8)
    luu("05_adf_so_voi_toi_han.png")


# Bài 3
y_diff1 = y.diff().dropna()
y_diff2 = y_diff1.diff().dropna()


def bai3():
    # 06, 07. Sai phân bậc 1 và bậc 2
    ve_sai_phan(y_diff1, "#246BFD", "Chuỗi sau sai phân bậc 1",
                "Thay đổi so với tháng trước", "06_sai_phan_bac1.png")
    ve_sai_phan(y_diff2, "#00A844", "Chuỗi sau sai phân bậc 2",
                "Thay đổi của sai phân", "07_sai_phan_bac2.png")

    # 08. ACF sau sai phân bậc 1 và bậc 2 (đỉnh mùa vụ lag 12, 24, 36)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    plot_acf(y_diff1, lags=36, ax=axes[0], title="ACF sau sai phân bậc 1")
    plot_acf(y_diff2, lags=36, ax=axes[1], title="ACF sau sai phân bậc 2")
    for ax in axes:
        ax.set_xlabel("Lag")
    luu("08_acf_sau_sai_phan.png")


# Bài 4
def bai4():
    # 09. ACF và PACF của chuỗi sau sai phân bậc 1
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(y_diff1, lags=36, ax=ax[0], title="ACF - chuỗi sau sai phân bậc 1")
    plot_pacf(y_diff1, lags=36, ax=ax[1], method="ywm", title="PACF - chuỗi sau sai phân bậc 1")
    for a in ax:
        a.set_xlabel("Lag")
    luu("09_acf_pacf.png")


# Bài 5
yt = y.asfreq("MS")
n_train = int(len(yt) * 0.8)
train, test = yt.iloc[:n_train], yt.iloc[n_train:]
BEST = (1, 1, 1)                     


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

    # 12. Train / Test / Forecast, đánh dấu 2 tháng lệch nhiều nhất
    du_bao = model.forecast(steps=len(test))
    sai_so_tt = (test - du_bao).abs()
    top2 = sorted(sai_so_tt.sort_values(ascending=False).head(2).index)
    plt.figure(figsize=(13, 5))
    plt.plot(train, label="Train", color="steelblue")
    plt.plot(test, label="Test (thực tế)", color="darkorange")
    plt.plot(du_bao, label=f"Forecast ARIMA{BEST}", color="green", linestyle="--")
    for i, t in enumerate(top2):      # i = 0: đỉnh sớm (nhãn bên trái); i = 1: đỉnh muộn (nhãn bên phải)
        plt.vlines(t, du_bao[t], test[t], color="red", linewidth=2)
        plt.plot(t, test[t], "o", color="red")
        plt.annotate(f"Lệch {so(sai_so_tt[t], 2)} ({t:%m/%Y})", (t, test[t]), textcoords="offset points",
                     xytext=(-10, 6) if i == 0 else (12, 0), ha="right" if i == 0 else "left",
                     va="bottom" if i == 0 else "center", color="red", fontweight="bold")
    plt.title("Dự báo ARIMA trên tập test (đánh dấu 2 tháng lệch nhiều nhất)")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.set_xlim(yt.index[0] - pd.DateOffset(months=6), yt.index[-1] + pd.DateOffset(months=30))
    ax.set_ylim(top=33)
    dinh_dang_truc_thoi_gian(ax)
    luu("12_forecast.png")


if __name__ == "__main__":
    bai1()
    bai2()
    bai3()
    bai4()
    bai5()