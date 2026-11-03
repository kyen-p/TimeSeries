"""Xử lý Dataset 2 (AirPassengers): làm sạch, sai phân thường, sai phân mùa vụ và sai phân kết hợp.

Đầu vào : data/raw/dataset2_raw.csv
Đầu ra  : data/processed/dataset2_diff.csv  (cột: date, value, diff1, diff12, diff_both)
"""
import sys
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Đọc dữ liệu và chuẩn hóa tên cột
df = pd.read_csv(RAW / "dataset2_raw.csv")
df.columns = df.columns.str.strip()
df = df.rename(columns={"Month": "date", "Passengers": "value"})
print("Kích thước dữ liệu:", df.shape)

# 2. Chuyển kiểu dữ liệu
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# 3. Kiểm tra dữ liệu thiếu, trùng
print("Dữ liệu thiếu:")
print(df.isnull().sum())
print("Số dòng trùng:", df.duplicated().sum())
df = df.drop_duplicates().dropna(subset=["date", "value"])

# 4. Sắp xếp theo thời gian, kiểm tra thời điểm trùng và tháng thiếu
df = df.sort_values("date").reset_index(drop=True)
print("Số thời điểm trùng:", df["date"].duplicated().sum())
day_du = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
print("Các tháng bị thiếu:", day_du.difference(df["date"]).tolist())

# 5. Sai phân thường (d=1), mùa vụ (D=1, s=12) và kết hợp
df["diff1"] = df["value"].diff()
df["diff12"] = df["value"].diff(12)
df["diff_both"] = df["diff1"].diff(12)

# 6. Lưu kết quả
df.to_csv(OUT / "dataset2_diff.csv", index=False)
print("Đã lưu:", OUT / "dataset2_diff.csv")