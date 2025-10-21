import os
import json
import pandas as pd
import streamlit as st
from joblib import load

st.set_page_config(page_title="Flight Price Predictor", layout="wide", page_icon="✈️")

# Constants
CSV_PATH = "Clean_Dataset.csv"
MODELS_DIR = os.path.join("outputs", "models")
METRICS_PATH = os.path.join("outputs", "metrics", "metrics.json")
PLOTS_DIR = os.path.join("outputs", "plots")
EDA_DIR = os.path.join("outputs", "eda")
SHAP_DIR = os.path.join("outputs", "shap")

# Small CSS tweaks for softer modern palette and better contrast
st.markdown(
    """
    <style>
    /* app background: soft indigo -> teal gradient */
    .stApp { background: linear-gradient(135deg, #eef6ff 0%, #f2fbfb 100%); color: #0f1724; }

    /* header */
    h1 { color: #0f1724; font-weight: 800; }
    .stMarkdown p { color: #334155; }

    /* card-like panels */
    .card { padding: 14px; border-radius: 12px; background: rgba(255,255,255,0.95); box-shadow: 0 6px 30px rgba(15,23,36,0.06); }
    .big-metric { font-size: 28px; font-weight:700; color:#075985; }

    /* tabs and accent color */
    /* Tabs: make them rounded pills and remove default white boxes */
    .stTabs [role="tab"] {
        color: #0f1724;
        background: transparent !important;
        border-radius: 999px;
        padding: 8px 18px;
        margin-right: 6px;
        border: 1px solid transparent;
        transition: all 150ms ease-in-out;
        box-shadow: none;
    }
    /* Container for tabs: subtle background to integrate tabs */
    .stTabs>div:first-child { background: rgba(255,255,255,0.6); padding: 8px; border-radius: 12px; }
    /* Active tab style */
    .stTabs [role="tab"][aria-selected="true"] {
        background: linear-gradient(90deg,#06b6d4 0%, #0ea5a4 100%) !important;
        color: white !important;
        border-color: rgba(10, 20, 30, 0.08) !important;
        box-shadow: 0 6px 18px rgba(2,6,23,0.08);
    }
    /* Hover */
    .stTabs [role="tab"]:hover { transform: translateY(-2px); }

    /* small responsive tweaks */
    .stButton>button { background-color: #0ea5a4; border: none; }
    .stCaption { color: #475569; }

    /* improve readability for images and captions */
    .stImage img { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Optional model descriptions (used to show info in the app)
MODEL_INFO = {
    "linear": {
        "name": "Linear Regression",
        "desc": "Linear Regression: simple, interpretable baseline."
    },
    "rf": {
        "name": "Random Forest",
        "desc": "Random Forest: ensemble of decision trees, good for non-linear patterns."
    },
    "xgb": {
        "name": "XGBoost",
        "desc": "XGBoost: gradient boosting tree model, strong performance on tabular data."
    }
}

@st.cache_data
def load_df(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

@st.cache_data
def list_models(models_dir: str):
    if not os.path.isdir(models_dir):
        return []
    return [f for f in os.listdir(models_dir) if f.endswith("_pipeline.joblib")]

@st.cache_data
def load_metrics(path: str = METRICS_PATH):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    # Load dataset
    if not os.path.exists(CSV_PATH):
        st.sidebar.warning("Upload 'Clean_Dataset.csv' to project root to enable helpful inputs.")
        df = pd.DataFrame()
    else:
        df = load_df(CSV_PATH)

    # Sidebar: model selection
    st.sidebar.header("Configuration")
    model_files = list_models(MODELS_DIR)
    selected_model_file = st.sidebar.selectbox("Trained model", options=[""] + model_files)

    # Load metrics
    metrics = load_metrics()

    # Header area
    st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'>", unsafe_allow_html=True)
    st.markdown("<div><h1>✈️ Flight Price Predictor</h1><p>Interactive demo — choose a model and enter flight details to predict price.</p></div>", unsafe_allow_html=True)
    # show quick metrics for selected model
    if selected_model_file:
        model_name = selected_model_file.replace("_pipeline.joblib", "")
        if model_name in metrics:
            m = metrics[model_name]
            st.markdown(f"<div class='card'><div class='big-metric'>R²: {m['R2']:.3f}</div><div>MAE: {m['MAE']:,.0f}</div><div>RMSE: {m['RMSE']:,.0f}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Main tabs
    tab_predict, tab_eda, tab_models = st.tabs(["Predict", "EDA", "Models"])

    # In tab content we will show details below


    # Predict tab
    with tab_predict:
        st.markdown("### Flight Price Prediction")
        st.info("Enter flight details to predict the price. If no model is available, please train one first using the sidebar or the Models tab.")
        st.markdown("""
        **Instructions:**
        - Select airline, source city, destination city, departure/arrival time, number of stops, class, duration, and days left.
        - If no data is available, you can try entering the sample values below.
        """)
        st.markdown("**Example:** Airline: VietJet, Source: Hanoi, Destination: Ho Chi Minh City, Departure: Morning, Arrival: Afternoon, Stops: 0, Class: Economy, Duration: 2.0, Days left: 30")
        with st.form("input_form"):
            row1, row2 = st.columns(2)
            if not df.empty:
                airline = row1.selectbox("Airline", sorted(df["airline"].dropna().unique())) if "airline" in df.columns else row1.text_input("Airline")
                source_city = row2.selectbox("Source", sorted(df["source_city"].dropna().unique())) if "source_city" in df.columns else row2.text_input("Source")
                destination_city = row1.selectbox("Destination", sorted(df["destination_city"].dropna().unique())) if "destination_city" in df.columns else row1.text_input("Destination")
                departure_time = row2.selectbox("Departure time", sorted(df["departure_time"].dropna().unique())) if "departure_time" in df.columns else row2.text_input("Departure time")
                arrival_time = row1.selectbox("Arrival time", sorted(df["arrival_time"].dropna().unique())) if "arrival_time" in df.columns else row1.text_input("Arrival time")
                stops = row2.selectbox("Stops", sorted(df["stops"].dropna().unique())) if "stops" in df.columns else row2.text_input("Stops")
                seat_class = row1.selectbox("Class", sorted(df["class"].dropna().unique())) if "class" in df.columns else row1.text_input("Class")
            else:
                airline = row1.text_input("Airline")
                source_city = row2.text_input("Source")
                destination_city = row1.text_input("Destination")
                departure_time = row2.text_input("Departure time")
                arrival_time = row1.text_input("Arrival time")
                stops = row2.text_input("Stops")
                seat_class = row1.text_input("Class")

            duration = row2.number_input("Duration (hours)", min_value=0.0, value=2.0, step=0.1)
            days_left = row1.number_input("Days left", min_value=0, value=30)

            submitted = st.form_submit_button("Predict")

        if submitted:
            if not selected_model_file:
                st.error("You need to train and select a model before making predictions.")
            else:
                model_path = os.path.join(MODELS_DIR, selected_model_file)
                pipe = load(model_path)
                sample = pd.DataFrame([{"airline": airline,
                                        "source_city": source_city,
                                        "destination_city": destination_city,
                                        "departure_time": departure_time,
                                        "arrival_time": arrival_time,
                                        "stops": stops,
                                        "class": seat_class,
                                        "duration": duration,
                                        "days_left": days_left}])
                pred = pipe.predict(sample)[0]
                st.success(f"Predicted price: {pred:,.0f} RUB")
                st.caption("Note: For best results, use values similar to those in the training data.")

    # EDA tab
    with tab_eda:
        st.markdown("### Exploratory Data Analysis (EDA)")
        st.info("Explore the dataset, view price distributions, and relationships between features.")
        st.markdown("""
        **Chart explanations:**
        - Price distribution helps detect outliers and overall trends.
        - Boxplots by airline, stops, and class allow group comparisons.
        - Scatter plot of price vs days left shows the effect of booking time.
        """)
        if df.empty:
            st.warning("No data available. You can view sample images below or upload your dataset.")
            st.image("https://i.imgur.com/1Q9Z1ZB.png", caption="Sample price distribution")
            st.image("https://i.imgur.com/2Q9Z1ZB.png", caption="Sample boxplot by airline")
        else:
            c1, c2 = st.columns(2)
            hist_p = os.path.join(EDA_DIR, "price_histogram.png")
            days_p = os.path.join(EDA_DIR, "price_vs_days_left.png")
            if os.path.exists(hist_p):
                c1.image(hist_p, caption="Price distribution")
            if os.path.exists(days_p):
                c2.image(days_p, caption="Price vs days left")

            st.markdown("#### Boxplots by group")
            col1, col2, col3 = st.columns(3)
            a_p = os.path.join(EDA_DIR, "price_by_airline.png")
            s_p = os.path.join(EDA_DIR, "price_by_stops.png")
            cl_p = os.path.join(EDA_DIR, "price_by_class.png")
            if os.path.exists(a_p):
                col1.image(a_p, caption="By airline")
            if os.path.exists(s_p):
                col2.image(s_p, caption="By number of stops")
            if os.path.exists(cl_p):
                col3.image(cl_p, caption="By class")

    # Models tab
    with tab_models:
        st.markdown("### Models & Explanations")
        st.info("View details of trained models, evaluation metrics, and SHAP explanations. If no model is available, please train one using the Predict tab or sidebar.")
        st.markdown("""
        **Models used:**
        - Linear Regression: simple, interpretable.
        - Random Forest: ensemble of decision trees, good for non-linear data.
        - XGBoost: powerful boosting, high performance for tabular data.

        **Evaluation metrics:**
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - R²: Model fit (coefficient of determination)
        """)
        if not selected_model_file:
            st.warning("No model selected. You can view sample images below or train a model.")
            st.image("https://i.imgur.com/3Q9Z1ZB.png", caption="Sample feature importance")
            st.image("https://i.imgur.com/4Q9Z1ZB.png", caption="Sample SHAP summary")
        else:
            model_name = selected_model_file.replace("_pipeline.joblib", "")
            st.markdown(f"#### {model_name.upper()}")
            if model_name in MODEL_INFO:
                st.markdown(MODEL_INFO[model_name]['desc'])

            # show plots if exist
            fi = os.path.join(PLOTS_DIR, f"feature_importance_{model_name}.png")
            sp = os.path.join(SHAP_DIR, f"{model_name}_shap_summary.png")
            if os.path.exists(fi):
                st.markdown("**Feature importance**")
                st.image(fi)
            if os.path.exists(sp):
                st.markdown("**SHAP summary**")
                st.image(sp)


if __name__ == "__main__":
    main()
