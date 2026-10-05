"""Prediction service: validate -> infer -> persist -> enrich (treatment text).

Used by both the single /api/predict and the bulk /api/predict/bulk endpoints.
"""
import io

import pandas as pd

from backend.app.extensions import db
from backend.app.models import Prediction, Reading
from backend.app.services.model_service import model_service
from backend.app.validation import ValidationError, validate_reading
from backend.ml.config import FEATURE_COLS, GUIDELINE_LIMITS

# Simple rule-based treatment suggestions (display only).
TREATMENT_RULES = [
    ("Turbidity", "high", "Elevated turbidity: filtration or coagulation recommended."),
    ("ph", "low", "Low pH: neutralisation (e.g. lime dosing) recommended."),
    ("ph", "high", "High pH: acid dosing or blending with lower-pH water recommended."),
    ("Solids", "high", "High dissolved solids: reverse osmosis or blending recommended."),
    ("Hardness", "high", "Hard water: softening (ion exchange) recommended."),
    ("Chloramines", "high", "High disinfectant residual: activated carbon filtration."),
    ("Sulfate", "high", "High sulfate: reverse osmosis or distillation recommended."),
    ("Trihalomethanes", "high", "High THMs: activated carbon adsorption recommended."),
    ("Organic_carbon", "high", "High organic carbon: coagulation + activated carbon."),
    ("Conductivity", "high", "High conductivity: check for dissolved salts; RO if persistent."),
]


def treatment_suggestions(parsed: dict) -> list:
    tips = []
    for param, direction, text in TREATMENT_RULES:
        lim = GUIDELINE_LIMITS.get(param)
        if not lim:
            continue
        v = parsed.get(param)
        if v is None:
            continue
        if direction == "high" and lim.get("max") is not None and v > lim["max"]:
            tips.append(text)
        elif direction == "low" and lim.get("min") is not None and v < lim["min"]:
            tips.append(text)
    return tips


def predict_and_persist(parsed: dict, source: str = "manual", user_id: int | None = None) -> dict:
    """Run all three models, store reading+prediction, return the API payload."""
    potability, proba = model_service.predict_potability(parsed)
    ph_value, ph_filled_by = model_service.predict_ph(parsed)
    anomaly_flag, anomaly_score, reasons = model_service.predict_anomaly(parsed)

    reading = Reading(source=source, user_id=user_id,
                      **{c: parsed.get(c) for c in FEATURE_COLS})
    db.session.add(reading)
    db.session.flush()  # get reading.id

    prediction = Prediction(
        reading_id=reading.id,
        potability=potability,
        probability=round(proba, 4),
        ph_predicted=None if ph_filled_by == "none" else ph_value,
        ph_filled_by=ph_filled_by,
        is_anomaly=anomaly_flag,
        anomaly_score=round(anomaly_score, 4),
        anomaly_reason="; ".join(reasons) if reasons else None,
    )
    db.session.add(prediction)
    db.session.commit()

    return {
        "reading_id": reading.id,
        "potability": potability,
        "potability_label": "potable" if potability == 1 else "not potable",
        "probability": round(proba, 4),
        "ph": ph_value,
        "ph_filled_by": ph_filled_by,
        "anomaly": {
            "flag": anomaly_flag,
            "score": round(anomaly_score, 4),
            "reason": "; ".join(reasons) if reasons else "none",
        },
        "guideline_hits": model_service.guideline_hits(parsed),
        "treatment": treatment_suggestions(parsed),
    }


def rebuild_result(reading_id: int, user_id: int) -> dict | None:
    """Rebuild the API payload for a stored reading (for re-sending reports).

    Returns None when the reading does not exist, has no prediction yet, or
    belongs to a different user - callers answer 404 in that case.
    """
    reading = db.session.get(Reading, reading_id)
    if reading is None or reading.user_id != user_id or reading.prediction is None:
        return None
    p = reading.prediction
    parsed = {c: getattr(reading, c) for c in FEATURE_COLS}
    return {
        "reading_id": reading.id,
        "potability": p.potability,
        "potability_label": "potable" if p.potability == 1 else "not potable",
        "probability": p.probability,
        "ph": p.ph_predicted,
        "ph_filled_by": p.ph_filled_by,
        "anomaly": {
            "flag": p.is_anomaly,
            "score": p.anomaly_score,
            "reason": p.anomaly_reason or "none",
        },
        "guideline_hits": model_service.guideline_hits(parsed),
        "treatment": treatment_suggestions(parsed),
    }


def parse_bulk_csv(content: str, source: str = "bulk") -> dict:
    """Parse + validate + predict a CSV. Returns results, row errors, and summary."""
    try:
        # utf-8-sig transparently strips the BOM that Excel exports add.
        df = pd.read_csv(io.StringIO(content), encoding="utf-8-sig")
    except Exception as exc:
        raise ValidationError([f"Could not parse CSV: {exc}"])

    def _norm(col: str) -> str:
        return str(col).strip().lstrip("\ufeff").strip().lower()

    lookup = {_norm(c): c for c in df.columns}
    missing = [f for f in FEATURE_COLS if f.lower() not in lookup]
    if missing:
        raise ValidationError([
            f"CSV is missing required column(s): {missing}. "
            f"Found these columns instead: {list(df.columns)}. "
            "The header row must be: " + ",".join(FEATURE_COLS) +
            " (see sample_bulk_upload.csv in the project folder)."
        ])
    df = df.rename(columns={lookup[f.lower()]: f for f in FEATURE_COLS if f.lower() in lookup})

    results, errors = [], []
    for i, row in df.iterrows():
        raw = {c: row.get(c) for c in FEATURE_COLS}
        try:
            parsed = validate_reading(raw, require_all=False)
            if any(parsed.get(c) is None for c in FEATURE_COLS if c != "ph"):
                errors.append({"row": i + 2, "errors": ["missing required value(s)"]})
                continue
            res = predict_and_persist(parsed, source=source)
            res["row"] = i + 2
            results.append(res)
        except ValidationError as exc:
            errors.append({"row": i + 2, "errors": exc.errors})
        except Exception as exc:  # keep bulk runs going on unexpected rows
            errors.append({"row": i + 2, "errors": [str(exc)]})

    n_potable = sum(r["potability"] == 1 for r in results)
    return {
        "total_rows": int(len(df)),
        "predicted": len(results),
        "failed_rows": len(errors),
        "potable": n_potable,
        "not_potable": len(results) - n_potable,
        "anomalies": sum(1 for r in results if r["anomaly"]["flag"]),
        "results": results,
        "errors": errors,
    }
