"""Import the real Kaggle water potability CSV into water_quality_dataset.

Usage:
    python -m backend.ml.load_dataset

- Inspects the real CSV columns first (never assumes; maps case-insensitively
  and reports renames/mismatches).
- Generates sample_id 1..N (CSV has no id column).
- Replaces the table for idempotent re-runs (no duplicate rows).
- Prints a data-quality report and saves it to outputs/data_quality_report.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from backend.ml.config import (
    DATASET_CSV,
    DB_URL_DEFAULT,
    FEATURE_COLS,
    LABEL_COL,
    OUTPUTS_DIR,
    PHYSICAL_RANGES,
)


def load_raw_csv(path: str = DATASET_CSV) -> pd.DataFrame:
    """Read the CSV and verify its columns against expectations."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the Kaggle 'Water Potability' "
            "CSV there as water_potability.csv"
        )
    df = pd.read_csv(path)
    print(f"[load] read {len(df)} rows from {path}")
    print(f"[load] columns found: {list(df.columns)}")

    # Case-insensitive mapping so minor naming differences are handled + reported.
    lookup = {c.strip().lower(): c for c in df.columns}
    rename, missing = {}, []
    for expected in FEATURE_COLS + [LABEL_COL]:
        actual = lookup.get(expected.lower())
        if actual is None:
            missing.append(expected)
        elif actual != expected:
            rename[actual] = expected
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {missing}. Found: {list(df.columns)}"
        )
    if rename:
        print(f"[load] renaming columns to standard names: {rename}")
        df = df.rename(columns=rename)

    # Never assume the label is clean 0/1; show what we saw.
    labels = pd.unique(df[LABEL_COL])
    print(f"[load] distinct label values: {sorted(labels, key=str)}")
    return df[FEATURE_COLS + [LABEL_COL]].copy()


def data_quality_report(df: pd.DataFrame) -> dict:
    """Missing values, impossible pH, outliers (1.5*IQR), class balance."""
    report = {"n_rows": int(len(df)), "missing": {}, "outliers": {}}
    for col in FEATURE_COLS:
        report["missing"][col] = int(df[col].isna().sum())
        vals = df[col].dropna()
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        report["outliers"][col] = int(((vals < lo) | (vals > hi)).sum())

    ph_lo, ph_hi = PHYSICAL_RANGES["ph"]
    ph = df["ph"].dropna()
    report["ph_impossible"] = int(((ph < ph_lo) | (ph > ph_hi)).sum())
    report["ph_extremes_0_or_14"] = int(((ph == 0) | (ph == 14)).sum())
    report["class_balance"] = {
        "potable": int((df[LABEL_COL] == 1).sum()),
        "not_potable": int((df[LABEL_COL] == 0).sum()),
    }
    report["duplicates"] = int(df.duplicated(subset=FEATURE_COLS + [LABEL_COL]).sum())
    return report


def import_to_db(df: pd.DataFrame, db_url: str | None = None) -> int:
    """Write df to water_quality_dataset with sample_id 1..N. Returns row count."""
    load_dotenv()
    url = db_url or os.getenv("DATABASE_URL", DB_URL_DEFAULT)
    out = df.copy()
    out.insert(0, "sample_id", np.arange(1, len(out) + 1))
    engine = create_engine(url)
    out.to_sql("water_quality_dataset", engine, if_exists="replace", index=False)
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM water_quality_dataset")).scalar()
    print(f"[db] wrote {n} rows to water_quality_dataset at {url}")
    return int(n)


def main() -> pd.DataFrame:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df = load_raw_csv()
    report = data_quality_report(df)

    report_path = os.path.join(OUTPUTS_DIR, "data_quality_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== DATA QUALITY REPORT ===")
    print(f"rows: {report['n_rows']}")
    print(f"full-row duplicates: {report['duplicates']}")
    print("missing per column:", report["missing"])
    print("outliers per column (1.5*IQR):", report["outliers"])
    print(f"pH outside 0-14: {report['ph_impossible']} (of which exactly 0 or 14: {report['ph_extremes_0_or_14']})")
    print(
        f"class balance: potable={report['class_balance']['potable']} "
        f"({100 * report['class_balance']['potable'] / len(df):.1f}%), "
        f"not_potable={report['class_balance']['not_potable']}"
    )
    print(f"[saved] {report_path}")

    n = import_to_db(df)
    assert n == len(df), "row count mismatch after import"
    return df


if __name__ == "__main__":
    sys.exit(main() or 0)
