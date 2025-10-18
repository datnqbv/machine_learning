import os
import json
import pandas as pd
import streamlit as st
from joblib import load

st.set_page_config(page_title="Flight Price Predictor", layout="centered")

st.title("Flight Price Prediction")

CSV_PATH = "Clean_Dataset.csv"
MODELS_DIR = os.path.join("outputs", "models")

@st.cache_data
def load_df(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

@st.cache_data
def list_models(models_dir: str):
    if not os.path.isdir(models_dir):
        return []
    return [f for f in os.listdir(models_dir) if f.endswith("_pipeline.joblib")]

@st.cache_data
def load_metrics():
    metrics_path = os.path.join("outputs", "metrics", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    if not os.path.exists(CSV_PATH):
        st.warning("Upload your dataset 'Clean_Dataset.csv' to project root to populate choices.")
        df = pd.DataFrame()
    else:
        df = load_df(CSV_PATH)

    model_files = list_models(MODELS_DIR)
    selected_model_file = st.selectbox("Select trained model", options=model_files)
    
    # Load and display metrics
    metrics = load_metrics()
    if selected_model_file and metrics:
        model_name = selected_model_file.replace("_pipeline.joblib", "")
        if model_name in metrics:
            st.subheader("Model Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{metrics[model_name]['R2']:.3f}")
            with col2:
                st.metric("MAE", f"{metrics[model_name]['MAE']:,.0f}")
            with col3:
                st.metric("RMSE", f"{metrics[model_name]['RMSE']:,.0f}")

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        if not df.empty:
            airline = col1.selectbox("airline", sorted(df["airline"].dropna().unique())) if "airline" in df.columns else col1.text_input("airline")
            source_city = col2.selectbox("source_city", sorted(df["source_city"].dropna().unique())) if "source_city" in df.columns else col2.text_input("source_city")
            destination_city = col1.selectbox("destination_city", sorted(df["destination_city"].dropna().unique())) if "destination_city" in df.columns else col1.text_input("destination_city")
            departure_time = col2.selectbox("departure_time", sorted(df["departure_time"].dropna().unique())) if "departure_time" in df.columns else col2.text_input("departure_time")
            arrival_time = col1.selectbox("arrival_time", sorted(df["arrival_time"].dropna().unique())) if "arrival_time" in df.columns else col1.text_input("arrival_time")
            stops = col2.selectbox("stops", sorted(df["stops"].dropna().unique())) if "stops" in df.columns else col2.text_input("stops")
            seat_class = col1.selectbox("class", sorted(df["class"].dropna().unique())) if "class" in df.columns else col1.text_input("class")
        else:
            airline = col1.text_input("airline")
            source_city = col2.text_input("source_city")
            destination_city = col1.text_input("destination_city")
            departure_time = col2.text_input("departure_time")
            arrival_time = col1.text_input("arrival_time")
            stops = col2.text_input("stops")
            seat_class = col1.text_input("class")

        duration = col2.number_input("duration (minutes)", min_value=0.0, value=120.0)
        days_left = col1.number_input("days_left", min_value=0, value=30)

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        if not selected_model_file:
            st.error("Please train models first (run training script) and select one.")
            return
        model_path = os.path.join(MODELS_DIR, selected_model_file)
        pipe = load(model_path)
        sample = pd.DataFrame([
            {
                "airline": airline,
                "source_city": source_city,
                "destination_city": destination_city,
                "departure_time": departure_time,
                "arrival_time": arrival_time,
                "stops": stops,
                "class": seat_class,
                "duration": duration,
                "days_left": days_left,
            }
        ])
        pred = pipe.predict(sample)[0]
        st.success(f"Predicted price: {pred:,.0f}")

        st.caption("Tip: Ensure input categories exist in training data to avoid distribution shift.")


if __name__ == "__main__":
    main()
