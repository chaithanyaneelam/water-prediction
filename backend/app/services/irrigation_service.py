"""Irrigation service: validate -> rule engine (exact) -> ML cross-check -> persist."""
import json
import os

import joblib
import pandas as pd

from backend.app.extensions import db
from backend.app.models import IrrigationReading
from backend.ml import config as ml_config
from backend.ml.config import IRRIGATION_FEATURES
from backend.ml.irrigation_rules import compute_rsc, compute_sar, verdict as rule_verdict

SAVED = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "saved_models")
IRRIGATION_FEATURES_REQUIRED = ["EC", "HCO3", "Na", "Ca", "Mg"]
IRRIGATION_FIELDS = ["ph", "EC", "TDS", "CO3", "HCO3", "Cl", "F", "NO3",
                     "SO4", "Na", "K", "Ca", "Mg", "TH"]

IRRIGATION_RANGES = {
    "ph": (0.0, 14.0), "EC": (0.0, None), "TDS": (0.0, None), "CO3": (0.0, None),
    "HCO3": (0.0, None), "Cl": (0.0, None), "F": (0.0, None), "NO3": (0.0, None),
    "SO4": (0.0, None), "Na": (0.0, None), "K": (0.0, None), "Ca": (0.0, None),
    "Mg": (0.0, None), "TH": (0.0, None),
}


class ValidationError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__("; ".join(errors))


def _to_float(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("must be a number")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{value}' is not a number")
    if v != v:  # NaN -> missing
        return None
    if v in (float("inf"), float("-inf")):
        raise ValueError("infinity is not a valid measurement")
    return v


def validate_irrigation(data: dict) -> dict:
    errors, parsed = [], {}
    for field in IRRIGATION_FIELDS:
        try:
            v = _to_float(data.get(field))
        except ValueError as exc:
            errors.append(f"{field}: {exc}")
            continue
        if field in IRRIGATION_FEATURES_REQUIRED and v is None:
            errors.append(f"{field}: this field is required")
            continue
        if v is not None:
            lo, hi = IRRIGATION_RANGES[field]
            if v < lo:
                errors.append(f"{field}: {v} is below the physical minimum ({lo})")
            if hi is not None and v > hi:
                errors.append(f"{field}: {v} is above the physical maximum ({hi})")
            parsed[field] = v
    if errors:
        raise ValidationError(errors)
    return parsed


_model = None


def _get_model():
    """Load the irrigation ML model lazily, once."""
    global _model
    if _model is None:
        path = os.path.join(SAVED, "rf_irrigation.joblib")
        if os.path.exists(path):
            _model = joblib.load(path)
    return _model


def predict_irrigation(parsed: dict, source: str = "manual") -> dict:
    """Rule-engine verdict (exact) + optional ML class cross-check."""
    ec = parsed["EC"]
    na, ca, mg = parsed["Na"], parsed["Ca"], parsed["Mg"]
    co3 = parsed.get("CO3") or 0.0
    hco3 = parsed["HCO3"]
    sar = compute_sar(na, ca, mg)
    rsc = compute_rsc(co3, hco3, ca, mg)
    v = rule_verdict(ec, sar, rsc)

    # ML cross-check: predict USSL class from whatever chemistry is available
    # (the pipeline's median imputer fills gaps; label columns are never inputs).
    ml_class, ml_proba = None, None
    model = _get_model()
    if model is not None:
        row = {c: parsed.get(c) for c in IRRIGATION_FEATURES}
        X = pd.DataFrame([row], columns=IRRIGATION_FEATURES)
        proba = model.predict_proba(X)[0]
        idx = int(proba.argmax())
        ml_class = str(model.named_steps["model"].classes_[idx])
        ml_proba = round(float(proba[idx]), 4)

    reading = IrrigationReading(
        source=source,
        **{f: parsed.get(f) for f in IRRIGATION_FIELDS},
        SAR=v["sar"], RSC=v["rsc"],
        ussl_class_rule=v["ussl_class"], rsc_class_rule=v["rsc_class"],
        suitable=v["suitable"],
        verdict_notes=" | ".join(v["notes"]),
        ml_ussl_class=ml_class, ml_probability=ml_proba,
    )
    db.session.add(reading)
    db.session.commit()

    return {
        "reading_id": reading.id,
        "sar": v["sar"],
        "rsc": v["rsc"],
        "ussl_class": v["ussl_class"],
        "salinity": v["salinity"],
        "sodium": v["sodium"],
        "rsc_class": v["rsc_class"],
        "rsc_meaning": v["rsc_meaning"],
        "suitable": v["suitable"],
        "verdict": v["verdict"],
        "notes": v["notes"],
        "ml_cross_check": (
            {"class": ml_class, "probability": ml_proba}
            if ml_class is not None else
            {"class": None, "probability": None,
             "note": "ML model not trained yet - run: python -m backend.ml.train_irrigation"}),
    }


def irrigation_summary() -> dict:
    """Graph data + metrics for the Irrigation page."""
    def _read(name):
        path = os.path.join(ml_config.OUTPUTS_DIR, "irrigation", name)
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    total = db.session.query(IrrigationReading).count()
    suitable_n = db.session.query(IrrigationReading).filter(
        IrrigationReading.suitable.is_(True)).count()
    return {
        "available": _read("irrigation_metrics.json") is not None,
        "hint": None if _read("irrigation_metrics.json") is not None else
                "Irrigation model artifacts not found. Run: "
                "python -m backend.ml.load_irrigation_dataset && "
                "python -m backend.ml.train_irrigation",
        "class_distribution": _read("class_distribution.json"),
        "ussl_scatter": _read("ussl_scatter.json"),
        "confusion_matrix": _read("confusion_matrix.json"),
        "feature_importance": _read("feature_importance.json"),
        "metrics": _read("irrigation_metrics.json"),
        "rule_verification": _read("rule_verification.json"),
        "stored": {"total": total, "suitable": suitable_n},
    }
