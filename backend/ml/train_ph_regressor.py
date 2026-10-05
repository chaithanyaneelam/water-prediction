"""pH regressor: predict missing pH from the other 8 parameters (never Potability).

Rules:
- Trains ONLY on rows where 0 < ph < 14 (boundary/extreme pH rows are dropped).
- Pipeline: median impute -> IQR clip -> scale -> RandomForestRegressor, all
  fitted on training folds only.
- Metrics: MAE, RMSE, R2 (5-fold KFold CV mean+-std AND one-shot test), shown
  next to a "predict the training mean" baseline. R2 may be near 0 - reported
  honestly. The app uses this model to fill a missing pH only if it clearly
  beats the baseline (see SAVED decision in the metrics JSON).

Usage:
    python -m backend.ml.train_ph_regressor [--smoke]
"""
import argparse
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV, cross_validate, train_test_split

from backend.ml.config import (
    FEATURE_COLS, LABEL_COL, OUTPUTS_DIR, PH_PREDICTOR_COLS, RANDOM_STATE,
)
from backend.ml.preprocess import build_regressor_pipeline, make_feature_frame
from backend.ml.make_dataset_plots import load_dataset_from_db

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
OUT_DIR = os.path.join(OUTPUTS_DIR, "models")


def load_ph_frame(smoke: bool):
    if smoke:
        from backend.ml.config import BASE_DIR
        from backend.ml.load_dataset import load_raw_csv
        fixture = os.path.join(BASE_DIR, "tests", "fixtures", "mini_water_potability.csv")
        df = load_raw_csv(fixture)
        print(f"[smoke] using TEST-ONLY fixture ({len(df)} rows)")
    else:
        df = load_dataset_from_db()

    # Keep only physically sensible pH rows for training/evaluation.
    mask = (df["ph"] > 0) & (df["ph"] < 14)
    dropped = int((~mask).sum())
    if dropped:
        print(f"[filter] dropping {dropped} rows with pH outside (0,14) exclusive")
    df = df[mask]
    X = make_feature_frame(df, PH_PREDICTOR_COLS)  # 8 inputs, no ph, no label
    y = df["ph"].astype(float)
    return X, y


def _metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(rmse, 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def _save(name: str, data: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[saved] {path}")


def main(smoke: bool = False) -> dict:
    print(f"=== pH REGRESSOR TRAINING (smoke={smoke}) ===")
    X, y = load_ph_frame(smoke)
    print(f"[data] {len(X)} usable rows (pH in 0..14 exclusive)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE  # regression: no stratify
    )
    print(f"[split] train={len(X_train)} test={len(X_test)}")
    cv = KFold(n_splits=3 if smoke else 5, shuffle=True, random_state=RANDOM_STATE)

    # --- baseline: always predict the training mean (fitted on train only) ---
    baseline = build_regressor_pipeline(DummyRegressor(strategy="mean"))
    baseline.fit(X_train, y_train)
    base_pred_test = baseline.predict(X_test)
    base_metrics_test = _metrics(y_test, base_pred_test)
    b_cv = cross_validate(baseline, X_train, y_train, cv=cv,
                          scoring=("neg_mean_absolute_error", "r2"), n_jobs=-1)
    base_cv = {
        "mae": round(float(-b_cv["test_neg_mean_absolute_error"].mean()), 4),
        "mae_std": round(float(b_cv["test_neg_mean_absolute_error"].std()), 4),
        "r2": round(float(b_cv["test_r2"].mean()), 4),
        "r2_std": round(float(b_cv["test_r2"].std()), 4),
    }
    print(f"[baseline] test: {base_metrics_test}  (predict-the-mean)")

    # --- tuned RandomForestRegressor on the full pipeline ---
    grid_params = ({"model__n_estimators": [10], "model__max_depth": [3, 5]}
                   if smoke else
                   {"model__n_estimators": [200, 400],
                    "model__max_depth": [None, 10, 20],
                    "model__min_samples_leaf": [1, 2, 5]})
    rf_pipe = build_regressor_pipeline(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    )
    grid = GridSearchCV(rf_pipe, grid_params, cv=cv, scoring="neg_mean_absolute_error",
                        n_jobs=-1, refit=True)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f"[tuned] best params: {grid.best_params_} (CV MAE={-grid.best_score_:.3f})")

    r_cv = cross_validate(best, X_train, y_train, cv=cv,
                          scoring=("neg_mean_absolute_error", "r2"), n_jobs=-1)
    rf_cv = {
        "mae": round(float(-r_cv["test_neg_mean_absolute_error"].mean()), 4),
        "mae_std": round(float(r_cv["test_neg_mean_absolute_error"].std()), 4),
        "r2": round(float(r_cv["test_r2"].mean()), 4),
        "r2_std": round(float(r_cv["test_r2"].std()), 4),
    }
    rf_pred_test = best.predict(X_test)
    rf_metrics_test = _metrics(y_test, rf_pred_test)
    print(f"[rf] test: {rf_metrics_test}")

    # Decision: does the regressor clearly beat the mean baseline?
    beats = bool(rf_cv["mae"] < base_cv["mae"] * 0.98 and rf_metrics_test["mae"] < base_metrics_test["mae"])
    decision = (
        "USE MODEL for missing pH in the app (beats predict-the-mean baseline on CV and test MAE)"
        if beats else
        "USE MEDIAN IMPUTATION for missing pH in the app (model does NOT clearly beat the baseline)"
    )
    print(f"[decision] {decision}")

    # --- graphs 18-20 data ---
    _save("ph_regressor.json", {
        "baseline": {"cv": base_cv, "test": base_metrics_test,
                     "note": "DummyRegressor(strategy='mean') fitted on training split only."},
        "model": {"best_params": {k: str(v) for k, v in grid.best_params_.items()},
                  "cv": rf_cv, "test": rf_metrics_test},
        "app_decision": decision,
        "beats_baseline": beats,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "honesty_note": "R2 near 0 means pH is essentially independent of the other "
                        "parameters in this dataset - an honest, expected outcome.",
    })
    _save("ph_scatter.json", {  # graph 18: predicted vs actual + y=x
        "actual": [round(float(v), 3) for v in y_test],
        "predicted": [round(float(v), 3) for v in rf_pred_test],
        "y_x_line": {"x": [0, 14], "y": [0, 14]},
    })
    residuals = (np.asarray(y_test) - rf_pred_test)
    hist, edges = np.histogram(residuals, bins=15)
    _save("ph_residuals.json", {  # graph 19: residual scatter + histogram
        "actual": [round(float(v), 3) for v in y_test],
        "residuals": [round(float(v), 3) for v in residuals],
        "hist_centers": [round(float(b), 3) for b in (edges[:-1] + edges[1:]) / 2],
        "hist_counts": [int(v) for v in hist],
    })
    _save("ph_metrics_vs_baseline.json", {  # graph 20
        "labels": ["MAE", "RMSE", "R2"],
        "baseline": [base_metrics_test["mae"], base_metrics_test["rmse"], base_metrics_test["r2"]],
        "model": [rf_metrics_test["mae"], rf_metrics_test["rmse"], rf_metrics_test["r2"]],
    })

    import joblib
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best, os.path.join(MODELS_DIR, "rf_ph_regressor.joblib"))
    joblib.dump(baseline, os.path.join(MODELS_DIR, "mean_ph_baseline.joblib"))
    print(f"[saved] models -> {MODELS_DIR}")
    return {"rf_cv": rf_cv, "rf_test": rf_metrics_test,
            "base_cv": base_cv, "base_test": base_metrics_test}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(smoke=args.smoke)
