import os
from typing import List, Tuple, Dict, Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Định nghĩa các hằng số cho tên cột mục tiêu và các cột phân loại
TARGET_COLUMN = "price"
CATEGORICAL_COLUMNS = [ # Cột phân loại trong dữ liệu
    "airline",
    "source_city",
    "destination_city",
    "departure_time",
    "arrival_time",
    "stops",
    "class",
]
NUMERICAL_COLUMNS = [ # Cột số trong dữ liệu
    "duration",
    "days_left",
]
DROP_COLUMNS = [ # Cột không cần thiết sẽ bị loại bỏ
    "Unnamed: 0",
    "flight",
]

# Hàm tạo các thư mục outputs cần thiết để lưu kết quả
def ensure_output_dirs() -> Dict[str, str]:
    base = os.path.join("outputs")
    paths = { # Tạo các thư mục con trong thư mục outputs
        "base": base,
        "eda": os.path.join(base, "eda"),
        "models": os.path.join(base, "models"),
        "plots": os.path.join(base, "plots"),
        "metrics": os.path.join(base, "metrics"),
        "shap": os.path.join(base, "shap"),
    }
    for p in paths.values(): # Tạo thư mục nếu chưa tồn tại
        os.makedirs(p, exist_ok=True)
    return paths

# Hàm đọc và làm sạch dữ liệu từ file CSV
def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in DROP_COLUMNS: # Loại bỏ các cột không cần thiết
        if col in df.columns: # Kiểm tra sự tồn tại của cột trước khi loại bỏ
            df = df.drop(columns=[col])
    return df

# Hàm xác định các cột phân loại và cột số thực sự có trong dữ liệu
def validate_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cats = [c for c in CATEGORICAL_COLUMNS if c in df.columns] # Lọc các cột phân loại có trong DataFrame
    nums = [c for c in NUMERICAL_COLUMNS if c in df.columns] # Lọc các cột số có trong DataFrame
    return cats, nums

# Hàm xây dựng pipeline tiền xử lý dữ liệu (chuẩn hóa, one-hot)
def build_preprocessor(categorical_cols: List[str], numerical_cols: List[str]) -> ColumnTransformer:
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False) # Mã hóa one-hot cho cột phân loại
    numerical_transformer = StandardScaler() # Chuẩn hóa cho cột số

    preprocessor = ColumnTransformer( # Kết hợp các bước tiền xử lý cho cột phân loại và cột số
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numerical_transformer, numerical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor

# Hàm tách đặc trưng và mục tiêu từ DataFrame
def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset") # Kiểm tra sự tồn tại của cột mục tiêu
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y
