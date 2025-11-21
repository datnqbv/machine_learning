import argparse
import pandas as pd
from joblib import load


def main():
    parser = argparse.ArgumentParser() # Định nghĩa đối số dòng lệnh
    parser.add_argument("--model", type=str, required=True, help="Path to saved pipeline .joblib") # Đường dẫn tới mô hình đã lưu
    parser.add_argument("--csv", type=str, required=True, help="CSV with samples to predict") # Đường dẫn tới file CSV chứa dữ liệu để dự đoán
    args = parser.parse_args() # Lấy đối số dòng lệnh

    pipe = load(args.model) # Tải pipeline đã lưu từ file .joblib
    df = pd.read_csv(args.csv) # Đọc dữ liệu từ file CSV
    preds = pipe.predict(df) # Dự đoán sử dụng pipeline đã tải
    for p in preds:
        print(p)


if __name__ == "__main__":
    main()
