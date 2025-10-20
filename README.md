# Flight Price Prediction Project

This project performs EDA, preprocessing, model training (Linear Regression, Random Forest, XGboost), evaluation, SHAP-based explanations, and a Streamlit demo app using your dataset `Clean_Dataset.csv`.

## Project Structure
- `data/` (created at runtime)
- `outputs/` (EDA figures, metrics, models)
- `src/`
  - `eda.py`
  - `train.py`
  - `inference.py`
  - `utils.py`
- `app/`
  - `streamlit_app.py`
- `requirements.txt`

## Quickstart
1. Create and activate a virtual environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run EDA (figures saved to `outputs/eda/`):
```bash
python -m src.eda --csv Clean_Dataset.csv
```
4. Train models and evaluate (artifacts in `outputs/`):
```bash
python -m src.train --csv Clean_Dataset.csv
```
5. Launch Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

## Notes
- The scripts expect the following columns (typical for flight price data):
  - Target: `price`
  - Categorical: `airline, source_city, destination_city, departure_time, arrival_time, stops, class`
  - Numerical: `duration, days_left`
  - Will drop: `Unnamed: 0, flight` if present
- Adjust column names in `src/utils.py` if your dataset differs.
