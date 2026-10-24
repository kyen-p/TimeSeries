
import pandas as pd
from pathlib import Path

# 1. Xác định thư mục gốc của project
ROOT = Path(__file__).resolve().parents[2]

# 2. Khai báo đường dẫn dữ liệu
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# 3. Tạo thư mục processed nếu chưa có
OUT.mkdir(parents=True, exist_ok=True)

# 4. Đọc dữ liệu
df = pd.read_csv(RAW / "dataset1_raw.csv")

# 5. Kiểm tra dữ liệu
print("Kich thuoc du lieu:", df.shape)
print(df.head())
df.info()

# 6. Chuyển kiểu dữ liệu
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# 7. Kiểm tra dữ liệu thiếu
print("Du lieu thieu:")
print(df.isnull().sum())

# 8. Kiểm tra và xóa dòng trùng hoàn toàn
print("So dong trung:", df.duplicated().sum())
df = df.drop_duplicates()

# 9. Sắp xếp theo thời gian
df = df.sort_values("date").reset_index(drop=True)

# 11. Kiểm tra thời điểm trùng
print("So thoi diem trung:",
      df["date"].duplicated().sum())

# 12. Kiểm tra thống kê cơ bản
print(df["value"].describe())

# 13. Kiểm tra tháng bị thiếu
thang_day_du = pd.date_range(
    start=df["date"].min(),
    end=df["date"].max(),
    freq="MS"
)

thang_thieu = thang_day_du.difference(df["date"])
print("Cac thang bi thieu:", thang_thieu.tolist())

# 14. Lưu dữ liệu sau xử lí
df.to_csv(OUT / "dataset1_clean.csv",
          index=False, encoding="utf-8-sig")

print("Da luu file dataset1_clean.csv")
