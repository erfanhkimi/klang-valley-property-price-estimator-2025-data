# Klang Valley Property Price Estimator

A small machine-learning (supervised learning) web app that estimates a property's price per sq ft from its township, tenure and property type, then multiplies by the size you enter.

**Live demo:** https://klang-valley-property-price-estimator.onrender.com/

## How it works

1. `src/clean_data.py` cleans the Kaggle township-level data (median price per sq ft by township, tenure and property type), drops townships with too few transactions and trims outliers.
2. `src/train.py` trains a linear regression baseline and a random forest, compares them on a held-out 20% test set, and saves the better one.
3. `app/main.py` serves the model through a FastAPI endpoint (`POST /predict`). Interactive docs are at `/docs`.
4. `app/static/index.html` is a simple form that calls the API.

## Results

| Model | MAE (RM per sq ft) | R² |
|---|---|---|
| Linear regression (baseline) | _fill in_ | _fill in_ |
| Random forest | _fill in_ | _fill in_ |

Top features: _fill in from the training output_.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# put the Kaggle CSV in data/raw/, then:
python src/clean_data.py data/raw/<your_file>.csv --klang-valley   # drop the flag to keep all of Malaysia
python src/train.py
uvicorn app.main:app --reload
```

Open: (http://127.0.0.1:8000/)

## Deploy on Render

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Commit `models/model.joblib` and `models/meta.json` so the server can load them.
- Pin your package versions (`pip freeze`) so the server uses the same scikit-learn version that trained the model.

## Limitations

- The data is aggregated: median prices per township, not individual properties. Size, floor, condition and rooms are not modelled.
- The data is a 2025 snapshot and gets out of date.
- Rare locations and unusual properties will be less accurate.
- This is an estimate for learning purposes, not a valuation.

## Data

House Prices in Malaysia (2025) by lyhatt on Kaggle: https://www.kaggle.com/datasets/lyhatt/house-prices-in-malaysia-2025
