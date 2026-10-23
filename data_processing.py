#5. Tiền xử lý dữ liệu
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Khai báo đường dẫn
PROJECT_DIR = Path(
    r"G:\.shortcut-targets-by-id\1X9Wo6KizsPTddHOHc8jojsn0xmnVENC1\[Nhóm 10] ML & AI trong tài chính\ML_ChongChiu&PhucHoiCoPhieu"
)

input_path = PROJECT_DIR / "data" / "processed"  / "preparation_preprocessing" / "df_model_train.csv"
output_path = PROJECT_DIR / "data" / "processed"  / "preparation_preprocessing" / "df_scaled.csv"

# Đọc dữ liệu trước
df_model_train = pd.read_csv(input_path)

# Các nhóm biến cần scaling
standard_cols = [
    "volatility_60d",
    "mean_return_60d",
    "beta",
    "resilience",
    "relative_resilience",
    "recovery_ratio_60d"
]

minmax_cols = [
    "avg_volume_60d",
    "momentum_60d"
]

# Sao chép dữ liệu để giữ nguyên dữ liệu gốc
df_scaled = df_model_train.copy()

# StandardScaler
scaler_standard = StandardScaler()
df_scaled[standard_cols] = scaler_standard.fit_transform(
    df_model_train[standard_cols]
)

# Min-Max Scaling
scaler_minmax = MinMaxScaler()
df_scaled[minmax_cols] = scaler_minmax.fit_transform(
    df_model_train[minmax_cols]
)

# Kiểm tra kết quả
print(df_scaled[standard_cols + minmax_cols].head())

output_path = r"G:\.shortcut-targets-by-id\1X9Wo6KizsPTddHOHc8jojsn0xmnVENC1\[Nhóm 10] ML & AI trong tài chính\ML_ChongChiu&PhucHoiCoPhieu\data\processed\preparation_preprocessing\df_scaled.csv"
df_scaled.to_csv(output_path, index=False, encoding="utf-8-sig")
print("Đã lưu file tại:", output_path)