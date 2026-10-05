"""Prediction endpoints."""
from flask import Blueprint, jsonify, request

from backend.app.services.prediction_service import parse_bulk_csv, predict_and_persist
from backend.app.validation import ValidationError, validate_reading

bp = Blueprint("predict", __name__, url_prefix="/api/predict")


@bp.post("")
def predict_single():
    data = request.get_json(silent=True) or {}
    # 'source' is reserved for the demo simulator (source=simulated); the web
    # form never sends it, so normal use is always stored as 'manual'.
    source = data.get("source") if data.get("source") in ("manual", "simulated") else "manual"
    try:
        parsed = validate_reading(data)
    except ValidationError as exc:
        return jsonify({"ok": False, "errors": exc.errors}), 400
    result = predict_and_persist(parsed, source=source)
    result["ok"] = True
    return jsonify(result)


@bp.post("/bulk")
def predict_bulk():
    if "file" in request.files:
        content = request.files["file"].read().decode("utf-8", errors="replace")
    else:
        content = request.get_data(as_text=True)
    if not content.strip():
        return jsonify({"ok": False, "errors": ["empty request: attach a CSV file "
                        "or send CSV text"]}), 400
    try:
        result = parse_bulk_csv(content, source="bulk")
    except ValidationError as exc:
        return jsonify({"ok": False, "errors": exc.errors}), 400
    result["ok"] = True
    return jsonify(result)
