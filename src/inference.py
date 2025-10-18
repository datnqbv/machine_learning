import argparse
import pandas as pd
from joblib import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved pipeline .joblib")
    parser.add_argument("--csv", type=str, required=True, help="CSV with samples to predict")
    args = parser.parse_args()

    pipe = load(args.model)
    df = pd.read_csv(args.csv)
    preds = pipe.predict(df)
    for p in preds:
        print(p)


if __name__ == "__main__":
    main()
