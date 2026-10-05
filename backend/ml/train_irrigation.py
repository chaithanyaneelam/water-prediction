"""Irrigation water quality: RandomForest predicts the USSL class from chemistry.

Honesty rules (same discipline as the potability models):
- Split FIRST: untouched 20% stratified test set (random_state=42).
- Preprocessing (median impute + scale) inside the pipeline, fitted on train folds.
- Label columns (SAR, ussl_class, rsc_class, RSC) are NEVER features.
- Rare USSL classes (< 10 samples) are excluded from training - documented.
- The deterministic rule engine (irrigation_rules.py) is the primary decision path;
  this ML model is a cross-check that reports its true agreement/confusion.

Usage:
    python -m backend.ml.train_irrigation

Artifacts -> outputs/irrigation/*.json and saved_models/rf_irrigation.joblib
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from backend.ml import config as ml_config
from backend.ml.config import IRRIGATION_FEATURES, RANDOM_STATE
from backend.ml.irrigation_rules import compute_sar, ussl_class
from sklearn.pipeline import Pipeline
from backend.ml.preprocess import MedianImputer, StandardScalerSafe

OUT_DIR_NAME = "irrigation"


def _save(name: str, data) -> None:
    out_dir = os.path.join(ml_config.OUTPUTS_DIR, OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[saved] {path}")


def load_irrigation_from_db():
    from sqlalchemy import create_engine
    from dotenv import load_dotenv
    from backend.ml.config import DB_URL_DEFAULT
    load_dotenv()
    url = os.getenv("DATABASE_URL", DB_URL_DEFAULT)
    df = pd.read_sql("SELECT * FROM irrigation_dataset", create_engine(url))
    if df.empty:
        raise RuntimeError("irrigation_dataset is empty - run "
                           "python -m backend.ml.load_irrigation_dataset first")
    return df


def verify_rule_engine(df: pd.DataFrame) -> dict:
    """Check the deterministic formulas against the dataset's own columns."""
    has = df.dropna(subset=["SAR"]).copy()
    sar_calc = has.apply(
        lambda r: compute_sar(r["Na"], r["Ca"], r["Mg"]), axis=1)
    sar_agree = float((np.abs(sar_calc - has["SAR"]) <
                       0.05 * has["SAR"].abs().clip(lower=0.1)).mean())
    ok = has.dropna(subset=["ussl_class"]).copy()
    rule_class = ok.apply(lambda r: ussl_class(r["EC"], r["SAR"])["class"], axis=1)
    ussl_agree = float((rule_class == ok["ussl_class"]).mean())
    return {
        "n_checked_sar": int(len(has)),
        "sar_agreement_5pct": round(sar_agree, 4),
        "n_checked_ussl": int(len(ok)),
        "ussl_rule_agreement": round(ussl_agree, 4),
        "note": "Rule engine = standard formulas (SAR from Na/Ca/Mg; USSL from "
                "EC and SAR). Near-perfect agreement with the dataset's own "
                "columns validates the implementation.",
    }


def main() -> dict:
    print("=== IRRIGATION MODEL TRAINING ===")
    df = load_irrigation_from_db()

    # Rule-engine verification on the full dataset.
    rule_report = verify_rule_engine(df)
    print(f"[rules] SAR agreement: {rule_report['sar_agreement_5pct']*100:.1f}% | "
          f"USSL agreement: {rule_report['ussl_rule_agreement']*100:.1f}%")
    _save("rule_verification.json", rule_report)

    # Keep classes with >= 10 samples (documented honesty: rare classes dropped).
    counts = df["ussl_class"].value_counts()
    keep = counts[counts >= ml_config.IRRIGATION_USSL_MIN_CLASS_SIZE].index.tolist()
    data = df[df["ussl_class"].isin(keep)].copy()
    dropped = int(len(df) - len(data))
    print(f"[data] {len(data)} rows across {len(keep)} USSL classes "
          f"({dropped} rows in rare/missing classes excluded from ML)")
    _save("class_distribution.json", {
        "labels": counts[keep].index.tolist(),
        "values": [int(c) for c in counts[keep].values],
        "excluded_rows": dropped,
        "excluded_classes": {str(k): int(v) for k, v in
                             counts[~counts.index.isin(keep)].items()},
    })

    X = data[IRRIGATION_FEATURES].copy()
    if "ussl_class" in X.columns:
        raise AssertionError("Leakage guard: ussl_class must never be a feature")
    y = data["ussl_class"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"[split] train={len(X_train)} test={len(X_test)} (untouched)")
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    def pipe(est):
        return Pipeline([("impute", MedianImputer()),
                         ("scale", StandardScalerSafe()),
                         ("model", est)])

    grid = GridSearchCV(
        pipe(RandomForestClassifier(random_state=RANDOM_STATE,
                                    class_weight="balanced_subsample", n_jobs=-1)),
        ({"model__n_estimators": [10], "model__max_depth": [5, None]}
         if len(X_train) < 100 else
         {"model__n_estimators": [200, 400],
          "model__max_depth": [None, 10, 20],
          "model__min_samples_leaf": [1, 2, 5]}),
        cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f"[tuned] {grid.best_params_} (CV macro-F1={grid.best_score_:.3f})")

    cvres = cross_validate(best, X_train, y_train, cv=cv,
                           scoring=("accuracy", "f1_macro"), n_jobs=-1)
    cv_summary = {
        "cv_accuracy_mean": round(float(cvres["test_accuracy"].mean()), 4),
        "cv_accuracy_std": round(float(cvres["test_accuracy"].std()), 4),
        "cv_f1_macro_mean": round(float(cvres["test_f1_macro"].mean()), 4),
        "cv_f1_macro_std": round(float(cvres["test_f1_macro"].std()), 4),
    }

    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)
    classes = best.named_steps["model"].classes_.tolist()
    test_metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1_macro": round(float(f1_score(y_test, pred, average="macro")), 4),
        "majority_baseline_accuracy": round(
            float((y_test == y_train.value_counts().idxmax()).mean()), 4),
    }
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, labels=classes, zero_division=0)
    per_class = {c: {"precision": round(float(p), 3), "recall": round(float(r), 3),
                     "f1": round(float(fl), 3)}
                 for c, p, r, fl in zip(classes, prec, rec, f1)}
    cm = confusion_matrix(y_test, pred, labels=classes).tolist()
    print(f"[test] accuracy={test_metrics['accuracy']} "
          f"macroF1={test_metrics['f1_macro']} "
          f"(majority baseline {test_metrics['majority_baseline_accuracy']})")

    _save("irrigation_metrics.json", {
        "classes": classes,
        "best_params": {k: str(v) for k, v in grid.best_params_.items()},
        "cv": cv_summary,
        "test": test_metrics,
        "per_class": per_class,
        "confusion_matrix": cm,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "excluded_rare_rows": dropped,
        "honesty_note": "The USSL class is nearly a deterministic function of EC "
                        "(salinity C1-C4) and SAR (sodium S1-S4), so high accuracy is "
                        "expected and NOT inflated: the rule engine computes it exactly. "
                        "The ML model is a cross-check for noisy/boundary labels and for "
                        "inputs with missing ion values (imputed).",
    })
    _save("confusion_matrix.json", {"labels": classes, "matrix": cm})

    imp = best.named_steps["model"].feature_importances_
    order = np.argsort(imp)[::-1]
    _save("feature_importance.json", {
        "features": [IRRIGATION_FEATURES[i] for i in order],
        "values": [round(float(imp[i]), 4) for i in order],
    })

    # Classic USSL diagram data: EC (log-x) vs SAR, one series per class.
    scatter = {}
    for cls in keep:
        sub = data[data["ussl_class"] == cls]
        scatter[cls] = {"ec": [round(float(v), 1) for v in sub["EC"]],
                        "sar": [round(float(v), 2) for v in sub["SAR"]]}
    _save("ussl_scatter.json", {
        "series": scatter,
        "zones": {"ec_class_boundaries": [250, 750, 2250],
                  "sar_class_boundaries": [10, 18, 26]},
        "note": "USSL (Richards 1954) diagram: salinity hazard (EC, x-axis) vs "
                "sodium hazard (SAR, y-axis). C1-C4 = increasing salinity, "
                "S1-S4 = increasing sodium hazard.",
    })

    import joblib
    saved_dir = ml_config.SAVED_MODELS_DIR
    os.makedirs(saved_dir, exist_ok=True)
    joblib.dump(best, os.path.join(saved_dir, "rf_irrigation.joblib"))
    print(f"[saved] model -> {saved_dir}/rf_irrigation.joblib")
    return test_metrics


if __name__ == "__main__":
    main()
