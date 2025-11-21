import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .utils import load_dataset, ensure_output_dirs, TARGET_COLUMN, CATEGORICAL_COLUMNS

# Hàm vẽ và lưu biểu đồ phân phối giá
def plot_price_histogram(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET_COLUMN], bins=50, kde=True)
    plt.title("Price Distribution")
    plt.xlabel("Price") 
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "price_histogram.png"))
    plt.close()

# Hàm vẽ và lưu biểu đồ hộp giá theo từng danh mục
def plot_box_by_category(df: pd.DataFrame, col: str, out_dir: str) -> None:
    if col not in df.columns:
        return
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x=col, y=TARGET_COLUMN)
    plt.title(f"Price by {col}")
    plt.xlabel(col)
    plt.ylabel("Price")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"price_by_{col}.png"))
    plt.close()

# Hàm vẽ và lưu biểu đồ thể hiện mối quan hệ giữa days_left và giá
def plot_days_left_relationship(df: pd.DataFrame, out_dir: str) -> None:
    if "days_left" not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="days_left", y=TARGET_COLUMN, alpha=0.3)
    sns.regplot(data=df, x="days_left", y=TARGET_COLUMN, scatter=False, color="red")
    plt.title("Price vs Days Left")
    plt.xlabel("Days Left")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "price_vs_days_left.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser() # Định nghĩa đối số dòng lệnh
    parser.add_argument("--csv", type=str, required=True, help="Path to Clean_Dataset.csv")
    args = parser.parse_args() # Lấy đối số dòng lệnh

    paths = ensure_output_dirs() # Tạo các thư mục outputs cần thiết để lưu kết quả
    eda_dir = paths["eda"] # Thư mục để lưu kết quả EDA

    df = load_dataset(args.csv) # Đọc và làm sạch dữ liệu từ file CSV

    # Histogram of price
    plot_price_histogram(df, eda_dir)

    # Price by airline, stops, class
    for cat in ["airline", "stops", "class"]: # Cột phân loại để vẽ biểu đồ hộp
        plot_box_by_category(df, cat, eda_dir)

    # Relationship with days_left
    plot_days_left_relationship(df, eda_dir)


if __name__ == "__main__":
    main()
