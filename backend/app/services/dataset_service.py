"""Dataset Explorer data: saved explorer JSONs + summary statistics table."""
import pandas as pd
from sqlalchemy import create_engine

from backend.app.services.plots_service import DATASET_DIR, _read
from backend.ml.config import DB_URL_DEFAULT, FEATURE_COLS, LABEL_COL, UNITS
from backend.ml.make_dataset_plots import load_dataset_from_db


def dataset_summary() -> dict:
    """Everything the Dataset Explorer page needs in one payload."""
    data = {
        "class_balance": _read(f"{DATASET_DIR}/class_balance.json"),
        "missing_values": _read(f"{DATASET_DIR}/missing_values.json"),
        "feature_distributions": _read(f"{DATASET_DIR}/feature_distributions.json"),
        "correlation_matrix": _read(f"{DATASET_DIR}/correlation_matrix.json"),
        "outlier_counts": _read(f"{DATASET_DIR}/outlier_counts.json"),
        "imputation_comparison": _read(f"{DATASET_DIR}/imputation_comparison.json"),
        "summary_stats": summary_stats_table(),
    }
    data["available"] = all(
        data[k] is not None
        for k in ("class_balance", "missing_values", "correlation_matrix")
    )
    data["hint"] = (
        None if data["class_balance"] is not None
        else "Dataset explorer artifacts not found. Run: python -m backend.ml.train_all"
    )
    return data


def summary_stats_table() -> list:
    """count/mean/std/min/quartiles/max per feature + missing count, from the DB."""
    try:
        df = load_dataset_from_db()
    except RuntimeError:
        return []
    if df.empty:
        return []  # table exists but import not run yet -> show hint in UI
    rows = []
    desc = df[FEATURE_COLS].describe()
    for col in FEATURE_COLS:
        rows.append({
            "feature": col,
            "unit": UNITS[col],
            "count": int(desc.loc["count", col]),
            "missing": int(df[col].isna().sum()),
            "mean": round(float(desc.loc["mean", col]), 3),
            "std": round(float(desc.loc["std", col]), 3),
            "min": round(float(desc.loc["min", col]), 3),
            "25%": round(float(desc.loc["25%", col]), 3),
            "50%": round(float(desc.loc["50%", col]), 3),
            "75%": round(float(desc.loc["75%", col]), 3),
            "max": round(float(desc.loc["max", col]), 3),
        })
    return rows
