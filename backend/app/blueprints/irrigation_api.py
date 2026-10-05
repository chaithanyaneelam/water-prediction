"""Irrigation endpoints."""
from flask import Blueprint, jsonify, request

from backend.app.services.irrigation_service import (
    ValidationError, irrigation_summary, predict_irrigation, validate_irrigation,
)

bp = Blueprint("irrigation", __name__, url_prefix="/api/irrigation")


@bp.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        parsed = validate_irrigation(data)
    except ValidationError as exc:
        return jsonify({"ok": False, "errors": exc.errors}), 400
    result = predict_irrigation(parsed)
    result["ok"] = True
    return jsonify(result)


@bp.get("/summary")
def summary():
    return jsonify({"ok": True, **irrigation_summary()})
