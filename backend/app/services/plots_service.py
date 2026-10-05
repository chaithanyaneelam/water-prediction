"""Serves saved graph JSON artifacts (model comparison + dataset explorer)."""
import json
import os

from backend.ml.config import OUTPUTS_DIR

MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
DATASET_DIR = os.path.join(OUTPUTS_DIR, "dataset")

MODEL_FILES = [
    "classifier_metrics", "roc_curves", "pr_curves", "calibration",
    "threshold_curve", "feature_importance", "learning_curve",
    "validation_curves", "baseline_vs_improved", "ph_regressor",
    "ph_scatter", "ph_residuals", "ph_metrics_vs_baseline",
    "anomaly_score_hist", "anomaly_pca", "anomaly_reasons",
]
DATASET_FILES = [
    "class_balance", "missing_values", "feature_distributions",
    "correlation_matrix", "outlier_counts", "imputation_comparison",
]


def get_model_json(name: str):
    if name not in MODEL_FILES:
        return None
    return _read(os.path.join(MODELS_DIR, f"{name}.json"))


def get_dataset_json(name: str):
    if name not in DATASET_FILES:
        return None
    return _read(os.path.join(DATASET_DIR, f"{name}.json"))


def png_path(name: str):
    """Resolve a PNG name to a path, or None. Names are whitelisted."""
    candidates = [
        os.path.join(MODELS_DIR, f"{name}.png"),
        os.path.join(OUTPUTS_DIR, "dataset", f"{name}.png"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _read(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
