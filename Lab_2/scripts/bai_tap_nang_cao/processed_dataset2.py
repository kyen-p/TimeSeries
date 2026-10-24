
import pandas as pd
from pathlib import Path

# 1. Khai báo đường dẫn
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# 2. Tạo thư mục đầu ra nếu chưa có
OUT.mkdir(parents=True, exist_ok=True)

# 3. Đọc dữ liệu
file_raw = RAW / "dataset2_raw.csv"

if not file_raw.exists():
    raise FileNotFoundError(
        "Khong tim thay dataset2_raw.csv. "
        "Hay kiem tra ten file trong danh sach ben tren."
    )

df = pd.read_csv(file_raw)

# 4. Kiểm tra dữ liệu ban đầu
print("\nKich thuoc du lieu:", df.shape)
print(df.head())
df.info()

# 5. Chuẩn hóa tên cột thành date và value
df.columns = df.columns.str.strip()
df = df.rename(columns={
    "Month": "date",
    "Passengers": "value"
})

# 6. Chuyển kiểu dữ liệu
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# 7. Kiểm tra dữ liệu thiếu
print("\nDu lieu thieu:")
print(df.isnull().sum())

# 8. Xóa dòng trùng hoàn toàn
print("\nSo dong trung:", df.duplicated().sum())
df = df.drop_duplicates()

# 9. Loại bỏ dòng thiếu ngày hoặc giá trị
df = df.dropna(subset=["date", "value"])

# 10. Sắp xếp theo thời gian
df = df.sort_values("date").reset_index(drop=True)

# 11. Kiểm tra thời điểm trùng
print("\nSo thoi diem trung:",
      df["date"].duplicated().sum())

# 12. Thống kê cơ bản
print("\nThong ke du lieu:")
print(df["value"].describe())

# 13. Kiểm tra tháng bị thiếu
thang_day_du = pd.date_range(
    start=df["date"].min(),
    end=df["date"].max(),
    freq="MS"
)

thang_thieu = thang_day_du.difference(df["date"])
print("\nCac thang bi thieu:", thang_thieu.tolist())

# 14. Lưu dữ liệu sau xử lý
file_out = OUT / "dataset2_clean.csv"

df.to_csv(
    file_out,
    index=False,
    encoding="utf-8-sig"
)

print("\nDa luu file: dataset2_clean.csv")