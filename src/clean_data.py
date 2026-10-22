"""Clean the raw Kaggle CSV (aggregated by township) and write data/clean.csv.

Usage:  python src/clean_data.py data/raw/<your_file>.csv [--klang-valley]

--klang-valley keeps only Kuala Lumpur and Selangor rows.
Column names are matched case-insensitively. Edit COLUMN_MAP if yours differ.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# our name -> possible raw names (lowercase, spaces as underscores)
COLUMN_MAP = {
    "township": ["township", "neighbourhood"],
    "location": ["location", "district", "area"],
    "state": ["state", "region"],
    "tenure": ["tenure"],
    "property_type": ["property_type", "type"],
    "median_psf": ["median_psf", "psf", "price_psf"],
    "median_price": ["median_price", "price"],
    "transactions": ["transactions", "transaction_count"],
}
REQUIRED = ["township", "state", "median_psf"]
MIN_TRANSACTIONS = 3  # medians built from very few sales are noisy


def norm(c: str) -> str:
    return c.strip().lower().replace(" ", "_")


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {norm(c): c for c in df.columns}
    rename = {}
    for ours, candidates in COLUMN_MAP.items():
        for cand in candidates:
            if cand in lookup:
                rename[lookup[cand]] = ours
                break
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns {missing}. Your columns: {list(df.columns)}. Edit COLUMN_MAP.")
    return df


def to_number(v):
    nums = re.findall(r"\d[\d,]*\.?\d*", str(v))
    return float(nums[0].replace(",", "")) if nums else np.nan


def main(path: str, klang_valley: bool = False) -> None:
    df = rename_columns(pd.read_csv(path))
    print(f"Loaded {len(df):,} rows")
    print("Columns found:", list(df.columns))

    if "location" not in df.columns:
        df["location"] = df["state"]
    for col in ["tenure", "property_type"]:
        df[col] = df[col].astype(str).str.strip() if col in df.columns else "Unknown"
    for col in ["township", "location", "state"]:
        df[col] = df[col].astype(str).str.strip()

    if klang_valley:
        text = df["state"] + " " + df["location"]
        df = df[text.str.contains("kuala lumpur|selangor", case=False, regex=True)]
        print(f"Kept {len(df):,} Kuala Lumpur / Selangor rows")

    for col in ["median_psf", "median_price", "transactions"]:
        if col in df.columns:
            df[col] = df[col].map(to_number)

    df = df.dropna(subset=["median_psf"])
    df = df[df["median_psf"] > 0]
    if "transactions" in df.columns:
        df = df[df["transactions"].fillna(0) >= MIN_TRANSACTIONS]
        print(f"After requiring >= {MIN_TRANSACTIONS} transactions: {len(df):,} rows")
    df = df.drop_duplicates()

    lo, hi = df["median_psf"].quantile([0.01, 0.99])
    df = df[df["median_psf"].between(lo, hi)]

    if len(df) < 300:
        print("WARNING: very few rows left. Try lowering MIN_TRANSACTIONS or skip --klang-valley.")

    keep = [c for c in ["township", "location", "state", "tenure", "property_type",
                        "median_psf", "median_price", "transactions"] if c in df.columns]
    out = Path(__file__).resolve().parents[1] / "data" / "clean.csv"
    df[keep].to_csv(out, index=False)
    print(f"Saved {len(df):,} clean rows -> {out}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    if len(args) != 1 or any(f != "--klang-valley" for f in flags):
        sys.exit("Usage: python src/clean_data.py data/raw/<file>.csv [--klang-valley]")
    main(args[0], klang_valley="--klang-valley" in flags)
