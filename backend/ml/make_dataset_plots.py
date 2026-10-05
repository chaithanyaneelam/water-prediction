"""Dataset Explorer graphs 1-6, all computed from the REAL imported dataset.

Saves each graph's data as JSON (for Chart.js) and, where Chart.js cannot
draw it, a matplotlib PNG. The API serves both (see /api/dataset/summary
and /api/plots/<name>).

Graphs:
1. class_balance.json            - doughnut
2. missing_values.json           - bar
3. feature_distributions.json    - overlaid histograms per feature, split by Potability
4. correlation_matrix.json       - colored HTML grid (frontend renders values)
5. boxplots.png (+ outlier_counts.json) - matplotlib boxplots per feature
6. imputation_comparison.json    - before vs after median-imputation histograms
"""
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: save files, never open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from backend.ml.config import (
    DB_URL_DEFAULT,
    FEATURE_COLS,
    LABEL_COL,
    OUTPUTS_DIR,
    UNITS,
)

PLOT_DIR = os.path.join(OUTPUTS_DIR, "dataset")


def _save_json(name: str, data) -> str:
    path = os.path.join(PLOT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"[graph] saved {name}")
    return path


def load_dataset_from_db(db_url: str | None = None) -> pd.DataFrame:
    """Read the imported training dataset back from the database."""
    load_dotenv()
    url = db_url or os.getenv("DATABASE_URL", DB_URL_DEFAULT)
    engine = create_engine(url)
    try:
        df = pd.read_sql("SELECT * FROM water_quality_dataset", engine)
    except Exception as exc:  # give a helpful, actionable error
        raise RuntimeError(
            "Could not read water_quality_dataset. Run the import first: "
            "python -m backend.ml.load_dataset"
        ) from exc
    print(f"[data] loaded {len(df)} rows from database")
    return df


def graph1_class_balance(df: pd.DataFrame):
    counts = df[LABEL_COL].value_counts().to_dict()
    data = {
        "labels": ["Not potable (0)", "Potable (1)"],
        "values": [int(counts.get(0, 0)), int(counts.get(1, 0))],
    }
    return _save_json("class_balance.json", data)


def graph2_missing_values(df: pd.DataFrame):
    data = {
        "labels": FEATURE_COLS,
        "values": [int(df[c].isna().sum()) for c in FEATURE_COLS],
    }
    return _save_json("missing_values.json", data)


def graph3_feature_distributions(df: pd.DataFrame):
    """Per feature: shared bins, overlaid counts for each Potability class."""
    out = {}
    for col in FEATURE_COLS:
        vals = df[col].dropna()
        bins = np.histogram_bin_edges(vals, bins=30)
        pot = df.loc[df[LABEL_COL] == 1, col].dropna()
        not_pot = df.loc[df[LABEL_COL] == 0, col].dropna()
        h_pot, _ = np.histogram(pot, bins=bins)
        h_not, _ = np.histogram(not_pot, bins=bins)
        out[col] = {
            "unit": UNITS[col],
            "bin_edges": [round(float(b), 3) for b in bins],
            "bin_centers": [round(float(b), 3) for b in (bins[:-1] + bins[1:]) / 2],
            "potable": [int(v) for v in h_pot],
            "not_potable": [int(v) for v in h_not],
        }
    return _save_json("feature_distributions.json", out)


def graph4_correlation(df: pd.DataFrame):
    cols = FEATURE_COLS + [LABEL_COL]
    corr = df[cols].corr()
    return _save_json(
        "correlation_matrix.json",
        {
            "features": cols,
            "matrix": [[round(float(v), 3) for v in row] for row in corr.values],
            "note": "Pearson correlation. Potability is shown last; weak correlations "
            "with it are the honest reason accuracy stays around 65-70%.",
        },
    )


def graph5_boxplots(df: pd.DataFrame):
    """Boxplots per feature (matplotlib; also outlier counts as JSON)."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    outlier_counts = {}
    for ax, col in zip(axes.flat, FEATURE_COLS):
        vals = df[col].dropna()
        ax.boxplot(vals, vert=False, flierprops={"markersize": 2, "alpha": 0.4})
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_counts[col] = int(((vals < lo) | (vals > hi)).sum())
        ax.set_title(f"{col} ({UNITS[col]}) - {outlier_counts[col]} outliers")
        ax.set_yticks([])
    fig.suptitle("Feature boxplots with 1.5*IQR outliers (real dataset)")
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "boxplots.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[graph] saved boxplots.png")
    _save_json("outlier_counts.json", {"labels": FEATURE_COLS, "values": [outlier_counts[c] for c in FEATURE_COLS]})
    return path


def graph6_imputation_comparison(df: pd.DataFrame):
    """Before (dropna) vs after (median-imputed) histograms for columns with NaNs."""
    out = {}
    for col in FEATURE_COLS:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        before = df[col].dropna()
        median = float(before.median())
        after = df[col].fillna(median)
        bins = np.histogram_bin_edges(after, bins=40)
        h_before, _ = np.histogram(before, bins=bins)
        h_after, _ = np.histogram(after, bins=bins)
        out[col] = {
            "unit": UNITS[col],
            "median_used": round(median, 3),
            "n_imputed": n_missing,
            "bin_centers": [round(float(b), 3) for b in (bins[:-1] + bins[1:]) / 2],
            "before": [int(v) for v in h_before],
            "after": [int(v) for v in h_after],
        }
    return _save_json("imputation_comparison.json", out)


def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)
    df = load_dataset_from_db()
    graph1_class_balance(df)
    graph2_missing_values(df)
    graph3_feature_distributions(df)
    graph4_correlation(df)
    graph5_boxplots(df)
    graph6_imputation_comparison(df)
    print(f"[done] Dataset Explorer artifacts written to {PLOT_DIR}")


if __name__ == "__main__":
    main()
