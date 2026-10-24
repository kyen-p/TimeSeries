"""Xử lý dữ liệu airline-passengers: sai phân thường, sai phân mùa vụ, kiểm định ADF.

Đầu vào : data/raw/airline-passengers.csv
Đầu ra  : data/processed/airline_diff.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path


def find_root():
    # scripts/bai_tap_nang_cao/processed.py -> parents[2] là thư mục gốc Lab2_TimeSeries
    try:
        return Path(__file__).resolve().parents[2]
    except (NameError, IndexError):
        pass
    # Chạy trong notebook: đi ngược lên tìm thư mục có "data"
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "data").exists():
            return p
    return cwd


ROOT = find_root()
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

input_file = ROOT / "data" / "raw" / "airline-passengers.csv"
output_dir = ROOT / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(input_file if input_file.exists() else URL)
    df["Month"] = pd.to_datetime(df["Month"])
    df["Passengers"] = pd.to_numeric(df["Passengers"])
    return df.set_index("Month").sort_index()


def add_differences(df):
    # Không dùng log
    df["diff1"] = df["Passengers"].diff()         # sai phân thường, d=1
    df["diff12"] = df["Passengers"].diff(12)      # sai phân mùa vụ, D=12
    df["airline_diff"] = df["diff1"].diff(12)     # kết hợp d=1, D=12

    # Có biến đổi log (mô hình airline kinh điển)
    df["log_passengers"] = np.log(df["Passengers"])
    df["airline_diff_log"] = df["log_passengers"].diff().diff(12)
    return df


def describe(name, series):
    values = series.dropna().to_numpy()
    print(f"[{name}]")
    print("  Số giá trị     :", np.size(values))
    print("  Trung bình     :", round(np.mean(values), 4))
    print("  Độ lệch chuẩn  :", round(np.std(values, ddof=1), 4))


def run_adf(df):
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        print("\nChưa cài statsmodels, bỏ qua ADF. Cài bằng: pip install statsmodels")
        return

    print("\n--- Kiểm định ADF (p-value < 0.05 => chuỗi dừng) ---")
    for col in ["Passengers", "diff1", "diff12", "airline_diff", "airline_diff_log"]:
        stat, pvalue, *_ = adfuller(df[col].dropna())
        verdict = "dừng" if pvalue < 0.05 else "KHÔNG dừng"
        print(f"{col:18s} ADF = {stat:8.4f} | p-value = {pvalue:.4f} | {verdict}")


def main():
    df = add_differences(load_data())

    describe("Sai phân d=1, D=12", df["airline_diff"])
    describe("Sai phân d=1, D=12 (log)", df["airline_diff_log"])
    run_adf(df)

    output_file = output_dir / "airline_diff.csv"
    df.to_csv(output_file, index=True)
    print("\nĐã xử lý dữ liệu thành công!")
    print("CSV:", output_file)


if __name__ == "__main__":
    main()
