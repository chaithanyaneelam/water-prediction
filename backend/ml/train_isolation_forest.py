"""Isolation Forest anomaly detection + rule checks (graphs 21-23).

Honesty rules:
- Fitted on TRAINING features only (no labels exist for anomalies, and none are
  invented). No accuracy is reported for it anywhere - there is no ground truth.
- Contamination: 'auto' - with zero labelled anomalies, choosing a percentage
  would be arbitrary; 'auto' lets the model use its offset-based threshold and
  we SHOW the score distribution so the threshold is transparent.
- At inference the IF score is combined with simple physical rule checks
  (negative values, pH outside 0-14, ...) and the "reason" lists what fired.

Usage:
    python -m backend.ml.train_isolation_forest [--smoke]
"""
import argparse
import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from backend.ml.config import FEATURE_COLS, OUTPUTS_DIR, RANDOM_STATE
from backend.ml.preprocess import (
    IQRClipper, MedianImputer, StandardScalerSafe, make_feature_frame,
)
from sklearn.pipeline import Pipeline
from backend.ml.make_dataset_plots import load_dataset_from_db
from backend.ml.train_classifier import load_xy

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
OUT_DIR = os.path.join(OUTPUTS_DIR, "models")


def rule_reasons(row) -> list:
    """Physical/chemical sanity rules. Returns list of human-readable reasons."""
    reasons = []
    if row["ph"] is not None and not (0.0 <= row["ph"] <= 14.0):
        reasons.append(f"pH {row['ph']:.2f} outside the physical range 0-14")
    for col in FEATURE_COLS:
        v = row[col]
        if v is not None and v < 0:
            reasons.append(f"negative value for {col} ({v:.2f})")
    # Turbidity far above drinking-water norms is a practical red flag.
    if row["Turbidity"] is not None and row["Turbidity"] > 50:
        reasons.append(f"Turbidity {row['Turbidity']:.1f} NTU is extremely high (> 50)")
    return reasons


def build_if_pipeline(contamination="auto") -> Pipeline:
    return Pipeline([
        ("impute", MedianImputer()),
        ("clip", IQRClipper()),
        ("scale", StandardScalerSafe()),
        ("model", IsolationForest(
            n_estimators=200, contamination=contamination,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])


def _save(name: str, data: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[saved] {path}")


def main(smoke: bool = False) -> dict:
    print(f"=== ISOLATION FOREST (smoke={smoke}) ===")
    X, _y = load_xy(smoke)  # labels intentionally unused (unsupervised)
    X_train, X_eval = X.iloc[: int(len(X) * 0.8)], X.iloc[int(len(X) * 0.8):]
    print(f"[split] fit on {len(X_train)} training rows only; "
          f"{len(X_eval)} held-out rows shown in plots")

    pipe = build_if_pipeline(contamination="auto")
    pipe.fit(X_train)

    scores_eval = -pipe.decision_function(X_eval)  # higher = more anomalous
    flags_eval = pipe.predict(X_eval) == -1
    print(f"[if] flagged {int(flags_eval.sum())}/{len(X_eval)} held-out rows as anomalous")

    # Score histogram + threshold (graph 21). Threshold = 0 on the negated
    # decision_function; decision_function uses train-fitted offset.
    hist, edges = np.histogram(scores_eval, bins=25)
    _save("anomaly_score_hist.json", {
        "bin_centers": [round(float(b), 4) for b in (edges[:-1] + edges[1:]) / 2],
        "counts": [int(v) for v in hist],
        "threshold": 0.0,
        "note": "Negated decision_function: right of the 0 line = Isolation Forest "
                "anomaly region. Threshold comes from the training offset (contamination='auto').",
    })

    # 2D PCA scatter with anomalies highlighted (graph 22).
    X_trans = pipe[:-1].transform(X_eval)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X_trans)
    coords = pca.transform(X_trans)
    _save("anomaly_pca.json", {
        "normal": {"x": [round(float(v), 3) for v, f in zip(coords[:, 0], flags_eval) if not f],
                   "y": [round(float(v), 3) for v, f in zip(coords[:, 1], flags_eval) if not f]},
        "anomaly": {"x": [round(float(v), 3) for v, f in zip(coords[:, 0], flags_eval) if f],
                    "y": [round(float(v), 3) for v, f in zip(coords[:, 1], flags_eval) if f]},
        "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
    })

    # Rule checks over ALL data -> anomalies-by-reason bar (graph 23).
    X_all = X.copy()
    reason_counts: dict = {}
    if_flags_all = pipe.predict(X_all) == -1
    reason_counts["Isolation Forest flag"] = int(if_flags_all.sum())
    for _, row in X_all.iterrows():
        r = {c: (None if pd_isna(row[c]) else float(row[c])) for c in FEATURE_COLS}
        for reason in rule_reasons(r):
            key = reason.split(" (")[0].split(" is ")[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1
    _save("anomaly_reasons.json", {
        "labels": list(reason_counts.keys()),
        "values": list(reason_counts.values()),
        "note": "Counts over the full dataset. IF flag counts rows the model flags; "
                "rule counts are physical-check violations. A reading's 'reason' lists "
                "every rule that fired plus the IF flag.",
    })

    import joblib
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipe, os.path.join(MODELS_DIR, "isolation_forest.joblib"))
    print(f"[saved] model -> {MODELS_DIR}/isolation_forest.joblib")
    return {"n_flagged": int(flags_eval.sum()), "n_eval": int(len(X_eval))}


def pd_isna(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(smoke=args.smoke)
