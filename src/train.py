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
    load_dataset,
    ensure_output_dirs,
    build_preprocessor,
    split_features_target,
    validate_columns,
)


RANDOM_STATE = 42

def get_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "linear": LinearRegression(),
        "rf": RandomForestRegressor(
            n_estimators=400,
            max_features="sqrt",
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
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


def evaluate_and_log(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


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
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    paths = ensure_output_dirs()

    df = load_dataset(args.csv)
    X, y = split_features_target(df)
    categorical_cols, numerical_cols = validate_columns(df)
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    models = get_models()

    all_metrics: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("pre", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = evaluate_and_log(y_test, y_pred)
        all_metrics[name] = metrics

        # Save model
        model_path = os.path.join(paths["models"], f"{name}_pipeline.joblib")
        dump(pipe, model_path)

        # Pred vs Actual plot
        plot_pred_vs_actual(
            y_test.values if hasattr(y_test, "values") else y_test,
            y_pred,
            title=f"Predicted vs Actual - {name}",
            out_path=os.path.join(paths["plots"], f"pred_vs_actual_{name}.png")
        )

        # Feature importance and SHAP for tree-based models (RandomForest, XGBoost, etc.)
        tree_estimator = pipe.named_steps["model"]
        try:
            # transform (do not refit) to get training matrix used for shap
            X_train_transformed = pipe.named_steps["pre"].transform(X_train)
        except Exception:
            # fallback
            X_train_transformed = pipe.named_steps["pre"].fit_transform(X_train)

        feature_names = get_feature_names(pipe.named_steps["pre"], categorical_cols, numerical_cols)

        if hasattr(tree_estimator, "feature_importances_"):
            try:
                importances = tree_estimator.feature_importances_
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
