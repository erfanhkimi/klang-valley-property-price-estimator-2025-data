import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Malaysia Property Price Estimator",
    description="Estimates price per sq ft from township, tenure and property type, then multiplies by size. "
                "Estimates only, not a valuation.",
)

model = joblib.load(ROOT / "models" / "model.joblib")
meta = json.loads((ROOT / "models" / "meta.json").read_text())


class Property(BaseModel):
    state: str
    township: str
    property_type: str
    tenure: str
    size: float = Field(gt=0, description="Built-up size in sq ft")


@app.get("/meta")
def get_meta():
    """Dropdown options for the form (without the internal lookup table)."""
    return {k: v for k, v in meta.items() if k != "location_lookup"}


@app.post("/predict")
def predict(p: Property):
    location = meta["location_lookup"].get(f"{p.state}|{p.township}", p.state)
    row = pd.DataFrame([{
        "township": p.township, "location": location, "state": p.state,
        "tenure": p.tenure, "property_type": p.property_type,
    }])
    psf = float(model.predict(row)[0])
    err = meta["typical_error_psf"]
    return {
        "psf": round(psf, 1),
        "estimated_price": round(psf * p.size, -3),
        "low": round(max(psf - err, 0) * p.size, -3),
        "high": round((psf + err) * p.size, -3),
        "typical_error_psf": err,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
