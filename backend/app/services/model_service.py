"""Model inference service. Models are loaded ONCE at app startup.

Artifacts produced by `python -m backend.ml.train_all`:
- saved_models/xgboost_potability.joblib      (primary serving model)
- saved_models/rf_ph_regressor.joblib
- saved_models/mean_ph_baseline.joblib
- saved_models/isolation_forest.joblib
- outputs/models/ph_regressor.json            (beats_baseline decision)
"""
import json
import os

import joblib
import pandas as pd

from backend.ml.config import (
    FEATURE_COLS, GUIDELINE_LIMITS, OUTPUTS_DIR, PH_PREDICTOR_COLS,
)

SAVED = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "saved_models")
PH_METRICS_PATH = os.path.join(OUTPUTS_DIR, "models", "ph_regressor.json")


class ModelService:
    def __init__(self):
        self.models = {}
        self.ph_beats_baseline = False
        self.ph_median = 7.0
        self.loaded = False

    def load(self) -> dict:
        """Load all saved models once. Returns a status summary."""
        self.models = {
            "classifier": joblib.load(os.path.join(SAVED, "xgboost_potability.joblib")),
            "ph_model": joblib.load(os.path.join(SAVED, "rf_ph_regressor.joblib")),
            "ph_baseline": joblib.load(os.path.join(SAVED, "mean_ph_baseline.joblib")),
            "anomaly": joblib.load(os.path.join(SAVED, "isolation_forest.joblib")),
        }
        # The pH app decision was computed and saved at training time.
        if os.path.exists(PH_METRICS_PATH):
            with open(PH_METRICS_PATH, encoding="utf-8") as f:
                ph_info = json.load(f)
            self.ph_beats_baseline = bool(ph_info.get("beats_baseline", False))
        # Median pH of the training data (used for imputation fallback).
        imputer = getattr(self.models["ph_model"], "named_steps", {}).get("impute")
        if imputer is not None and hasattr(imputer, "medians_"):
            self.ph_median = float(imputer.medians_.get("ph", 7.0))
        self.loaded = True
        return {
            "classifier": "xgboost_potability.joblib",
            "ph_model": "rf_ph_regressor.joblib",
            "ph_baseline": "mean_ph_baseline.joblib",
            "anomaly": "isolation_forest.joblib",
            "ph_app_decision": "model" if self.ph_beats_baseline else "median",
        }

    def ensure_loaded(self) -> None:
        """Raise a clear error when deployment artifacts were not generated."""
        if not self.loaded:
            raise RuntimeError(
                "Prediction models are unavailable. Run the model training "
                "commands during deployment before serving predictions."
            )

    # ------------------------------------------------------------------ helpers
    def _frame(self, parsed: dict) -> pd.DataFrame:
        row = {c: parsed.get(c) for c in FEATURE_COLS}
        return pd.DataFrame([row], columns=FEATURE_COLS)

    def predict_ph(self, parsed: dict):
        """Predict pH when missing. Returns (value, filled_by in {model,median,none})."""
        self.ensure_loaded()
        if parsed.get("ph") is not None:
            return parsed["ph"], "none"
        if self.ph_beats_baseline:
            X = self._frame(parsed)[PH_PREDICTOR_COLS]
            v = float(self.models["ph_model"].predict(X)[0])
            return round(min(max(v, 0.0), 14.0), 2), "model"
        return round(float(self.ph_median), 2), "median"

    def predict_potability(self, parsed: dict):
        self.ensure_loaded()
        X = self._frame(parsed)
        proba = float(self.models["classifier"].predict_proba(X)[0][1])
        return int(proba >= 0.5), proba

    def predict_anomaly(self, parsed: dict):
        """IF score/flag + physical rule checks -> (flag, score, reasons list)."""
        self.ensure_loaded()
        X = self._frame(parsed)
        score = float(-self.models["anomaly"].decision_function(X)[0])
        flag = bool(self.models["anomaly"].predict(X)[0] == -1)
        reasons = []
        if flag:
            reasons.append("Isolation Forest outlier score above threshold")
        if parsed.get("ph") is not None and not (0.0 <= parsed["ph"] <= 14.0):
            reasons.append(f"pH {parsed['ph']:.2f} outside the physical range 0-14")
        for col in FEATURE_COLS:
            v = parsed.get(col)
            if v is not None and v < 0:
                reasons.append(f"negative value for {col} ({v:.2f})")
        if parsed.get("Turbidity") is not None and parsed["Turbidity"] > 50:
            reasons.append(f"Turbidity {parsed['Turbidity']:.1f} NTU is extremely high (> 50)")
        return flag, score, reasons

    def guideline_hits(self, parsed: dict) -> list:
        """Parameters outside guideline limits (display only - never labels/features)."""
        hits = []
        for col, lim in GUIDELINE_LIMITS.items():
            v = parsed.get(col)
            if v is None:
                continue
            lo, hi = lim.get("min"), lim.get("max")
            if lo is not None and v < lo:
                hits.append({"parameter": col, "value": v, "limit": f">= {lo}"})
            if hi is not None and v > hi:
                hits.append({"parameter": col, "value": v, "limit": f"<= {hi}"})
        return hits


# Module-level singleton, created once at app startup.
model_service = ModelService()
