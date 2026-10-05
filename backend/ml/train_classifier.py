"""Train and compare potability classifiers: Random Forest vs Decision Tree vs XGBoost.

Honesty rules enforced here:
- The 20% test set is split FIRST and never used for tuning.
- All preprocessing (median imputation, IQR clipping, scaling, SMOTE) lives inside
  the pipeline, so it is fitted on training folds only.
- SMOTE (imbalanced-learn) is applied ONLY inside CV folds via imblearn's Pipeline.
- Potability is never a feature (see make_feature_frame guard).
- No accuracy inflation: expect ~65-70%; weak feature-label correlation is the cause.

Usage:
    python -m backend.ml.train_classifier            # real run (DB dataset)
    python -m backend.ml.train_classifier --smoke    # tiny fixture, quick grids

Artifacts -> backend/ml/outputs/models/*.json and backend/ml/saved_models/*.joblib
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from backend.ml import config as ml_config
from backend.ml.config import FEATURE_COLS, LABEL_COL, RANDOM_STATE
from backend.ml.preprocess import (
    build_classifier_pipeline,
    make_feature_frame,
)
from backend.ml.make_dataset_plots import load_dataset_from_db


def _smote_kwargs(smoke: bool) -> dict:
    # In smoke mode folds are tiny; SMOTE needs k_neighbors < minority count.
    return {"k_neighbors": 2} if smoke else {}


def _grids(smoke: bool) -> dict:
    if smoke:
        return {
            "RandomForest": {"model__n_estimators": [10], "model__max_depth": [3, None]},
            "DecisionTree": {"model__max_depth": [2, None]},
            "XGBoost": {"model__n_estimators": [10], "model__max_depth": [2]},
        }
    return {
        "RandomForest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2, 5],
        },
        "DecisionTree": {
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 5, 10],
        },
        "XGBoost": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1],
        },
    }


def _imblearn_estimator(name: str, smoke: bool):
    """Estimator + whether it uses the SMOTE pipeline (inside CV folds only)."""
    if name == "RandomForest":
        est = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        return build_classifier_pipeline(est, with_smote=True, smote_kwargs=_smote_kwargs(smoke))
    if name == "DecisionTree":
        est = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")
        return build_classifier_pipeline(est, with_smote=False)
    if name == "XGBoost":
        est = XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=4, verbosity=0
        )
        return build_classifier_pipeline(est, with_smote=True, smote_kwargs=_smote_kwargs(smoke))
    raise ValueError(name)


def load_xy(smoke: bool):
    if smoke:
        from backend.ml.config import BASE_DIR
        fixture = os.path.join(
            BASE_DIR, "tests", "fixtures", "mini_water_potability.csv"
        )
        from backend.ml.load_dataset import load_raw_csv
        df = load_raw_csv(fixture)
        print(f"[smoke] using TEST-ONLY fixture ({len(df)} rows) - never real training")
    else:
        df = load_dataset_from_db()
    X = make_feature_frame(df, FEATURE_COLS)  # guard: label never a feature
    y = df[LABEL_COL].astype(int)
    return X, y


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_potable": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_potable": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_potable": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_not_potable": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_not_potable": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_not_potable": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(set(y_true)) > 1 else None,
    }


def _save(name: str, data: dict) -> None:
    out_dir = os.path.join(ml_config.OUTPUTS_DIR, "models")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[saved] {path}")


def _baseline_models(X_train, X_test, y_train, y_test) -> list:
    """Plain RF/DT on raw data with a simple split.

    sklearn trees cannot consume NaN, so the ONLY preprocessing is a median
    imputer fitted on the training split (never on test) - documented honestly
    in the note field.
    """
    rows = []
    for name, est in [
        ("RandomForest", RandomForestClassifier(random_state=RANDOM_STATE)),
        ("DecisionTree", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ]:
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", est),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        m = classification_metrics(y_test, pred, proba)
        rows.append({
            "model": name,
            "variant": "baseline (raw data, simple split, median imputer fitted on train only)",
            **m,
            "note": "No tuning, no SMOTE, no clipping/scaling. This is the floor to beat.",
        })
        print(f"[baseline] {name}: accuracy={m['accuracy']:.3f}")
    return rows


def main(smoke: bool = False) -> dict:
    print(f"=== CLASSIFIER TRAINING (smoke={smoke}) ===")
    X, y = load_xy(smoke)
    print(f"[data] {len(X)} rows, class balance: {y.value_counts().to_dict()}")

    # 1) Split FIRST: untouched stratified test set, fixed random_state.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[split] train={len(X_train)} test={len(X_test)} (test is untouched from here on)")
    cv = StratifiedKFold(n_splits=3 if smoke else 5, shuffle=True, random_state=RANDOM_STATE)

    baselines = _baseline_models(X_train, X_test, y_train, y_test)

    # 2) Tune with GridSearchCV using INNER CV on the training split only.
    models_summary, roc_curves, pr_curves, fitted = {}, {}, {}, {}
    for name in ["RandomForest", "DecisionTree", "XGBoost"]:
        print(f"\n--- tuning {name} (inner CV on train only) ---")
        est = _imblearn_estimator(name, smoke)
        grid = GridSearchCV(
            est, _grids(smoke)[name], cv=cv, scoring="f1", n_jobs=-1, refit=True
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        print(f"[tuned] best params: {grid.best_params_} (CV f1={grid.best_score_:.3f})")

        # CV mean+-std on the TRAINING split (test still untouched).
        cvres = cross_validate(
            best, X_train, y_train, cv=cv,
            scoring=("accuracy", "f1", "roc_auc"), n_jobs=-1,
        )
        cv_summary = {
            "cv_accuracy_mean": float(cvres["test_accuracy"].mean()),
            "cv_accuracy_std": float(cvres["test_accuracy"].std()),
            "cv_f1_mean": float(cvres["test_f1"].mean()),
            "cv_f1_std": float(cvres["test_f1"].std()),
            "cv_roc_auc_mean": float(cvres["test_roc_auc"].mean()),
            "cv_roc_auc_std": float(cvres["test_roc_auc"].std()),
        }
        print(f"[cv] accuracy {cv_summary['cv_accuracy_mean']:.3f} +- {cv_summary['cv_accuracy_std']:.3f}")

        # 3) FINAL one-shot evaluation on the untouched test set.
        pred = best.predict(X_test)
        proba = best.predict_proba(X_test)[:, 1]
        test_metrics = classification_metrics(y_test, pred, proba)
        print(f"[test] accuracy={test_metrics['accuracy']:.3f} roc_auc={test_metrics['roc_auc']:.3f}")

        cm = confusion_matrix(y_test, pred).tolist()
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_curves[name] = {"fpr": [round(float(v), 4) for v in fpr],
                            "tpr": [round(float(v), 4) for v in tpr],
                            "auc": test_metrics["roc_auc"]}
        p, r, _ = precision_recall_curve(y_test, proba)
        pr_curves[name] = {"precision": [round(float(v), 4) for v in p],
                           "recall": [round(float(v), 4) for v in r]}

        models_summary[name] = {
            "best_params": {k: str(v) for k, v in grid.best_params_.items()},
            "grid_cv_f1_best": float(grid.best_score_),
            **cv_summary,
            "test": test_metrics,
            "confusion_matrix": cm,
            "labels": ["not potable (0)", "potable (1)"],
        }
        fitted[name] = best

    _save("classifier_metrics.json", {
        "models": models_summary,
        "baselines": baselines,
        "note": "CV numbers are 5-fold mean+-std on the training split; test numbers are "
                "one-shot on the untouched 20% split. Modest scores (~65-70%) are expected: "
                "features correlate only weakly with the potability label in this dataset.",
    })
    _save("roc_curves.json", {
        "models": roc_curves,
        "diagonal": {"fpr": [0, 1], "tpr": [0, 1]},
    })
    _save("pr_curves.json", pr_curves)

    # Calibration (reliability) from tuned RandomForest on the test set.
    # Skipped in smoke mode: quantile bins need more test samples than the fixture has.
    rf = fitted["RandomForest"]
    if smoke:
        _save("calibration.json", {"skipped": "smoke mode"})
    else:
        proba_test = rf.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(
            y_test, proba_test, n_bins=8, strategy="quantile"
        )
        _save("calibration.json", {
            "prob_pred": [round(float(v), 4) for v in mean_pred],
            "prob_true": [round(float(v), 4) for v in frac_pos],
            "perfect": {"x": [0, 1], "y": [0, 1]},
            "note": "How close predicted probabilities are to true frequencies (RF, test set).",
        })

    # Threshold curve from OUT-OF-FOLD predictions on TRAINING data (no test peeking).
    oof = cross_val_predict(rf, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    thr_data = {"thresholds": thresholds.tolist(), "precision": [], "recall": [], "f1": []}
    for t in thresholds:
        pred_t = (oof >= t).astype(int)
        thr_data["precision"].append(round(float(precision_score(y_train, pred_t, zero_division=0)), 4))
        thr_data["recall"].append(round(float(recall_score(y_train, pred_t)), 4))
        thr_data["f1"].append(round(float(f1_score(y_train, pred_t)), 4))
    _save("threshold_curve.json", {
        **thr_data,
        "note": "Computed from out-of-fold predictions on TRAINING data only.",
    })

    # Feature importances (RF + XGB) and RFE ranking - all fitted on train only.
    importances = {}
    for name in ["RandomForest", "XGBoost"]:
        imp = fitted[name].named_steps["model"].feature_importances_
        importances[name] = {
            "features": FEATURE_COLS,
            "values": [round(float(v), 4) for v in imp],
        }
    rfe_pipe = Pipeline([
        ("impute", fitted["RandomForest"].named_steps["impute"]),
        ("clip", fitted["RandomForest"].named_steps["clip"]),
        ("rfe", RFE(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                    n_features_to_select=4)),
    ])
    rfe_pipe.fit(X_train, y_train)
    ranking = rfe_pipe.named_steps["rfe"].ranking_.tolist()
    importances["RFE_ranking"] = {"features": FEATURE_COLS, "values": ranking,
                                  "note": "Rank 1 = selected (top 4); higher = eliminated earlier."}
    _save("feature_importance.json", importances)

    # Learning curve + validation curves (train vs CV score; overfitting check).
    if not smoke:
        sizes, tr, cvv = learning_curve(
            rf, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 8),
        )
        lc = {"train_sizes": sizes.tolist(),
              "train_mean": tr.mean(axis=1).round(4).tolist(),
              "train_std": tr.std(axis=1).round(4).tolist(),
              "cv_mean": cvv.mean(axis=1).round(4).tolist(),
              "cv_std": cvv.std(axis=1).round(4).tolist()}
        _save("learning_curve.json", lc)
        _save_learning_curve_png(lc)

        vc = {}
        for pname, values in [("model__n_estimators", [10, 50, 100, 200, 400, 600]),
                              ("model__max_depth", [2, 4, 6, 8, 10, 15, 20, None])]:
            tr_s, cv_s = validation_curve(
                build_classifier_pipeline(
                    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                    with_smote=True, smote_kwargs=_smote_kwargs(smoke),
                ),
                X_train, y_train, param_name=pname, param_range=values,
                cv=cv, scoring="f1", n_jobs=-1,
            )
            key = pname.split("__")[1]
            vc[key] = {"values": [str(v) for v in values],
                       "train_mean": tr_s.mean(axis=1).round(4).tolist(),
                       "cv_mean": cv_s.mean(axis=1).round(4).tolist()}
        _save("validation_curves.json", vc)
    else:
        _save("learning_curve.json", {"skipped": "smoke mode"})
        _save("validation_curves.json", {"skipped": "smoke mode"})

    # Baseline vs improved summary table.
    improved_rows = []
    for name, s in models_summary.items():
        improved_rows.append({
            "model": name,
            "variant": "improved (pipeline: impute->clip->scale->SMOTE/class_weight, tuned inner CV)",
            "accuracy": s["test"]["accuracy"],
            "f1_potable": s["test"]["f1_potable"],
            "roc_auc": s["test"]["roc_auc"],
            "note": f"tuned: {s['best_params']}",
        })
    _save("baseline_vs_improved.json", {"rows": baselines + improved_rows})

    # Persist models with joblib.
    import joblib
    saved_dir = ml_config.SAVED_MODELS_DIR
    os.makedirs(saved_dir, exist_ok=True)
    for name, model in fitted.items():
        joblib.dump(model, os.path.join(saved_dir, f"{name.lower()}_potability.joblib"))
    print(f"[saved] models -> {saved_dir}")

    print("\n=== HONEST SUMMARY (test set, one-shot) ===")
    for name, s in models_summary.items():
        t = s["test"]
        print(f"{name:13s} acc={t['accuracy']:.3f} f1_pot={t['f1_potable']:.3f} "
              f"f1_not={t['f1_not_potable']:.3f} auc={t['roc_auc']:.3f}")
    return models_summary


def _save_learning_curve_png(lc: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    tr = np.array(lc["train_mean"]); trs = np.array(lc["train_std"])
    cvm = np.array(lc["cv_mean"]); cvs = np.array(lc["cv_std"])
    ax.fill_between(lc["train_sizes"], tr - trs, tr + trs, alpha=0.15, color="tab:blue")
    ax.fill_between(lc["train_sizes"], cvm - cvs, cvm + cvs, alpha=0.15, color="tab:orange")
    ax.plot(lc["train_sizes"], tr, "o-", color="tab:blue", label="Training F1")
    ax.plot(lc["train_sizes"], cvm, "o-", color="tab:orange", label="CV F1")
    ax.set_xlabel("Training samples"); ax.set_ylabel("F1 (potable)")
    ax.set_title("Learning curve - RandomForest (fitted on training split only)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(ml_config.OUTPUTS_DIR, "models", "learning_curve.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[saved] {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="run on the tiny test-only fixture with mini grids")
    args = parser.parse_args()
    main(smoke=args.smoke)
