import argparse
import json
import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

import shap

from .utils import (
    load_dataset,         # Đọc và làm sạch dữ liệu từ file CSV
    ensure_output_dirs,   # Tạo các thư mục outputs cần thiết để lưu kết quả
    build_preprocessor,   # Xây dựng pipeline tiền xử lý dữ liệu (chuẩn hóa, one-hot)
    split_features_target,# Tách dữ liệu thành features (X) và target (y)
    validate_columns,     # Xác định các cột phân loại và cột số thực sự có trong dữ liệu
)


RANDOM_STATE = 42
 #Hàm  khởi tạo và trả về ba mô hình dự báo giá
def get_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "linear": LinearRegression(),
        "rf": RandomForestRegressor(
            n_estimators=400,         #Số lượng cây quyết định trong rừng (400 cây).
            max_features="sqrt",      #Số lượng đặc trưng được xem xét khi chia mỗi node là căn bậc hai số đặc trưng.
            max_depth=12,             #Độ sâu tối đa của mỗi cây là 12.
            min_samples_leaf=2,       #Mỗi lá phải có ít nhất 2 mẫu.
            n_jobs=-1,                # Sử dụng tất cả các CPU để huấn luyện song song.
            random_state=RANDOM_STATE, # Đảm bảo kết quả có thể lặp lại (tái tạo được).
        ),
        "xgb": XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            objective='reg:squarederror',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    return models


# Hàm trả về MAE (sai số tuyệt đối trung bình), RMSE (căn bậc hai sai số bình phương trung bình), và R2 (hệ số xác định).
def evaluate_and_log(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}

# Hàm vẽ và lưu biểu đồ so sánh giữa giá thực tế và giá dự đoán
def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: str) -> None:
    plt.figure(figsize=(6,6))
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

 # Hàm lấy danh sách tên các đặc trưng (feature names) sau khi đã tiền xử lý (one-hot, chuẩn hóa...), để sử dụng cho việc trực quan hóa hoặc giải thích mô hình
def get_feature_names(preprocessor, categorical_cols, numerical_cols):
    # Works with sklearn >= 1.0 when verbose_feature_names_out=False
    try:
        return list(preprocessor.get_feature_names_out(categorical_cols + numerical_cols))
    except Exception:
        # Fallback: manually assemble names
        cat_ohe = preprocessor.named_transformers_["cat"]
        if hasattr(cat_ohe, "get_feature_names_out"):
            cat_names = list(cat_ohe.get_feature_names_out(categorical_cols))
        else:
            cat_names = []
        return cat_names + numerical_cols

# Hàm vẽ và lưu biểu đồ cột thể hiện tầm quan trọng của các đặc trưng (feature importance) trong mô hình
def plot_feature_importance(importances: np.ndarray, feature_names: list, title: str, out_path: str, top_k: int = 25):
    idx = np.argsort(importances)[::-1][:top_k]
    plt.figure(figsize=(8, max(4, int(0.3 * len(idx)))))
    sns.barplot(x=importances[idx], y=[feature_names[i] for i in idx])
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

 # Hàm tính toán giá trị SHAP để giải thích mô hình cây, vẽ và lưu biểu đồ SHAP summary plot
def compute_and_save_shap(tree_model, X_transformed: np.ndarray, feature_names: list, out_path_prefix: str):
    explainer = shap.TreeExplainer(tree_model)
    # Sample to speed up
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(X_transformed.shape[0], size=min(2000, X_transformed.shape[0]), replace=False)
    X_sample = X_transformed[sample_idx]
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_path_prefix + "_shap_summary.png") 
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True) # Đường dẫn tới file CSV chứa dữ liệu huấn luyện
    args = parser.parse_args() # Lấy đối số dòng lệnh

    paths = ensure_output_dirs() # Tạo các thư mục outputs cần thiết để lưu kết quả
    
    df = load_dataset(args.csv) # Đọc và làm sạch dữ liệu từ file CSV
    X, y = split_features_target(df)
    categorical_cols, numerical_cols = validate_columns(df) #Xác định các cột phân loại và cột số thực sự có trong dữ liệu
    preprocessor = build_preprocessor(categorical_cols, numerical_cols) #Xây dựng pipeline tiền xử lý dữ liệu (chuẩn hóa, one-hot)
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    models = get_models()

    all_metrics: Dict[str, Dict[str, float]] = {} # Lưu trữ các chỉ số đánh giá cho từng mô hình
    
    # Huấn luyện, đánh giá, lưu mô hình và tạo các biểu đồ cho từng mô hình
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("pre", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train) # Huấn luyện mô hình
        y_pred = pipe.predict(X_test) # Dự đoán trên tập kiểm tra

        metrics = evaluate_and_log(y_test, y_pred) # Tính toán các chỉ số đánh giá
        all_metrics[name] = metrics # Lưu chỉ số đánh giá cho mô hình hiện tại

        # Save model
        model_path = os.path.join(paths["models"], f"{name}_pipeline.joblib")
        dump(pipe, model_path)

        # Vẽ và lưu biểu đồ so sánh giữa giá thực tế và giá dự đoán
        plot_pred_vs_actual( 
            y_test.values if hasattr(y_test, "values") else y_test,
            y_pred,
            title=f"Predicted vs Actual - {name}",
            out_path=os.path.join(paths["plots"], f"pred_vs_actual_{name}.png")
        )

        # Feature importance and SHAP
        tree_estimator = pipe.named_steps["model"]
        try:
            # transform (do not refit) to get training matrix used for shap
            X_train_transformed = pipe.named_steps["pre"].transform(X_train) # Chuyển đổi tập huấn luyện mà không cần huấn luyện lại bộ tiền xử lý
        except Exception:
            # fallback
            X_train_transformed = pipe.named_steps["pre"].fit_transform(X_train)

        feature_names = get_feature_names(pipe.named_steps["pre"], categorical_cols, numerical_cols)
        # Feature importance plot
        if hasattr(tree_estimator, "feature_importances_"):
            try:
                importances = tree_estimator.feature_importances_ # Lấy tầm quan trọng của các đặc trưng từ mô hình cây
                plot_feature_importance( 
                    importances=np.array(importances),
                    feature_names=feature_names,
                    title=f"Feature Importance - {name}",
                    out_path=os.path.join(paths["plots"], f"feature_importance_{name}.png")
                )
            except Exception:
                pass

        # SHAP (only for tree-based models where TreeExplainer applies)
        try:
            compute_and_save_shap(
                tree_model=tree_estimator,
                X_transformed=X_train_transformed,
                feature_names=feature_names,
                out_path_prefix=os.path.join(paths["shap"], f"{name}")
            )
        except Exception:
            pass

    # Save metrics
    with open(os.path.join(paths["metrics"], "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()
