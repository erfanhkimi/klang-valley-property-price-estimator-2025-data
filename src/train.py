"""Predict price per sq ft (PSF) from township, state, tenure and property type.
Trains a baseline and a random forest, keeps the better one.

Usage:  python src/train.py
Output: models/model.joblib and models/meta.json
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["township", "location", "state", "tenure", "property_type"]
TARGET = "median_psf"


def build(regressor):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)])
    # Predict log(PSF): prices are right-skewed, so errors behave better
    model = TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("pre", pre), ("model", model)])


def main():
    df = pd.read_csv(ROOT / "data" / "clean.csv")
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "Linear regression (baseline)": build(LinearRegression()),
        "Random forest": build(RandomForestRegressor(
            n_estimators=100, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=42)),
    }

    results, best_name, best_model, best_mae = {}, None, None, float("inf")
    for name, pipe in candidates.items():
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        mae, r2 = mean_absolute_error(y_te, pred), r2_score(y_te, pred)
        results[name] = {"mae_psf": round(mae, 1), "r2": round(r2, 3)}
        print(f"{name:32s} MAE: RM {mae:>7,.1f} per sq ft   R2: {r2:.3f}")
        if mae < best_mae:
            best_name, best_model, best_mae = name, pipe, mae

    # Form options and a township -> location lookup for the API
    townships_by_state = {
        s: sorted(g["township"].unique().tolist()) for s, g in df.groupby("state")
    }
    location_lookup = {
        f"{r.state}|{r.township}": r.location
        for r in df.drop_duplicates(["state", "township"]).itertuples()
    }
    meta = {
        "best_model": best_name,
        "metrics": results,
        "states": sorted(townships_by_state),
        "townships_by_state": townships_by_state,
        "location_lookup": location_lookup,
        "property_types": sorted(df["property_type"].unique().tolist()),
        "tenures": sorted(df["tenure"].unique().tolist()),
        "typical_error_psf": round(best_mae, 1),
    }
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(best_model, ROOT / "models" / "model.joblib", compress=3)
    (ROOT / "models" / "meta.json").write_text(json.dumps(meta))
    print(f"\nSaved best model: {best_name}")

    if best_name == "Random forest":
        names = best_model.named_steps["pre"].get_feature_names_out()
        imp = best_model.named_steps["model"].regressor_.feature_importances_
        print("Top features:")
        for n, v in sorted(zip(names, imp), key=lambda t: -t[1])[:10]:
            print(f"  {n:50s} {v:.3f}")


if __name__ == "__main__":
    main()
