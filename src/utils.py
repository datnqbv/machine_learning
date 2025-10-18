import os
from typing import List, Tuple, Dict, Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Default column configuration. Adjust if your dataset differs.
TARGET_COLUMN = "price"
CATEGORICAL_COLUMNS = [
    "airline",
    "source_city",
    "destination_city",
    "departure_time",
    "arrival_time",
    "stops",
    "class",
]
NUMERICAL_COLUMNS = [
    "duration",
    "days_left",
]
DROP_COLUMNS = [
    "Unnamed: 0",
    "flight",
]


def ensure_output_dirs() -> Dict[str, str]:
    base = os.path.join("outputs")
    paths = {
        "base": base,
        "eda": os.path.join(base, "eda"),
        "models": os.path.join(base, "models"),
        "plots": os.path.join(base, "plots"),
        "metrics": os.path.join(base, "metrics"),
        "shap": os.path.join(base, "shap"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def validate_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cats = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    nums = [c for c in NUMERICAL_COLUMNS if c in df.columns]
    return cats, nums


def build_preprocessor(categorical_cols: List[str], numerical_cols: List[str]) -> ColumnTransformer:
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numerical_transformer, numerical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y
