"""Import the REAL Telangana irrigation groundwater dataset (xlsx) into the
`irrigation_dataset` table, with cleaning, DQ report and honest label handling.

Usage:
    python -m backend.ml.load_irrigation_dataset

Cleaning (documented, never silent):
- column renames to clean names (E.C -> EC, T.H -> TH, 'RSC  meq  / L' -> RSC,
  Classification -> ussl_class, Classification.1 -> rsc_class)
- USSL junk labels ('O.G', 'OG', 'BELOW THE GRAPH') -> NULL (excluded from ML,
  kept in the dataset for transparency)
- RSC label spelling: 'M.R' merged into 'MR' (same meaning, 3 rows)
- one fully-missing chemistry row dropped
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from backend.ml.config import DB_URL_DEFAULT, IRRIGATION_XLSX, OUTPUTS_DIR

JUNK_USSL = {"O.G", "OG", "BELOW THE GRAPH"}
RENAME = {
    "pH": "ph", "E.C": "EC", "T.H": "TH",
    "RSC  meq  / L": "RSC",
    "Classification": "ussl_class", "Classification.1": "rsc_class",
}


def load_raw(xlsx_path: str = IRRIGATION_XLSX) -> pd.DataFrame:
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(
            f"Irrigation dataset not found at {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    print(f"[load] read {len(df)} rows, columns: {list(df.columns)}")
    df = df.rename(columns=RENAME)

    # Numeric coercion for all chemistry columns.
    chemistry = ["ph", "EC", "TDS", "CO3", "HCO3", "Cl", "F", "NO3",
                 "SO4", "Na", "K", "Ca", "Mg", "TH", "SAR", "RSC"]
    for col in chemistry:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where ALL chemistry is missing (single row in this file).
    all_na = df[chemistry].isna().all(axis=1)
    if all_na.sum():
        print(f"[clean] dropping {int(all_na.sum())} row(s) with no chemistry values")
        df = df[~all_na]

    # Clean labels: junk -> None, M.R -> MR.
    df["ussl_class"] = df["ussl_class"].astype("object").where(
        ~df["ussl_class"].isin(JUNK_USSL), None)
    n_junk = int(df["ussl_class"].isna().sum() - pd.isna(
        pd.read_excel(xlsx_path)["Classification"]).sum())
    df["rsc_class"] = df["rsc_class"].replace({"M.R": "MR"})
    print(f"[clean] USSL junk/missing labels set to None: {n_junk + int(df['ussl_class'].isna().sum())}")
    print("[clean] USSL class distribution:\n", df["ussl_class"].value_counts().to_string())
    print("[clean] RSC class distribution:\n", df["rsc_class"].value_counts().to_string())
    return df


def import_to_db(df: pd.DataFrame, db_url: str | None = None) -> int:
    load_dotenv()
    url = db_url or os.getenv("DATABASE_URL", DB_URL_DEFAULT)
    out = df.copy()
    out.insert(0, "sample_id", np.arange(1, len(out) + 1))
    engine = create_engine(url)
    out.to_sql("irrigation_dataset", engine, if_exists="replace", index=False)
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM irrigation_dataset")).scalar()
    print(f"[db] wrote {n} rows to irrigation_dataset at {url}")
    return int(n)


def main() -> pd.DataFrame:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df = load_raw()
    report = {
        "n_rows": int(len(df)),
        "missing": {c: int(df[c].isna().sum()) for c in
                    ["ph", "EC", "TDS", "CO3", "HCO3", "Cl", "F", "NO3",
                     "SO4", "Na", "K", "Ca", "Mg", "TH", "SAR", "RSC"]},
        "ussl_distribution": df["ussl_class"].value_counts(dropna=False).to_dict(),
        "rsc_distribution": df["rsc_class"].value_counts().to_dict(),
        "districts": int(df["district"].nunique()),
        "note": "Real Telangana (India) groundwater monitoring data, pre- and "
                "post-monsoon seasons. USSL = Richards (1954) salinity x sodium "
                "classification; RSC = residual sodium carbonate classes.",
    }
    report_path = os.path.join(OUTPUTS_DIR, "irrigation_dq_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[saved] {report_path}")

    n = import_to_db(df)
    assert n == len(df), "row count mismatch after import"
    return df


if __name__ == "__main__":
    sys.exit(main() or 0)
