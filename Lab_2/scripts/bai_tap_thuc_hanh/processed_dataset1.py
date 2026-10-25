"""Xử lý Dataset 1 (a10): làm sạch, sai phân bậc 1 và bậc 2.

Đầu vào : data/raw/dataset1_raw.csv
Đầu ra  : data/processed/dataset1_diff.csv  (cột: date, value, diff1, diff2)
"""
import sys
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Đọc dữ liệu
df = pd.read_csv(RAW / "dataset1_raw.csv")
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

# 5. Sai phân bậc 1 và bậc 2
df["diff1"] = df["value"].diff()
df["diff2"] = df["diff1"].diff()

# 6. Lưu kết quả
df.to_csv(OUT / "dataset1_diff.csv", index=False)
print("Đã lưu:", OUT / "dataset1_diff.csv")