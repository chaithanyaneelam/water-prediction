"""Irrigation endpoints (login required - results go to the user's channel)."""
from flask import Blueprint, jsonify, request

from backend.app.blueprints.notify_hook import notify_result, require_user
from backend.app.services.irrigation_service import (
    ValidationError, irrigation_summary, predict_irrigation, validate_irrigation,
)

bp = Blueprint("irrigation", __name__, url_prefix="/api/irrigation")


@bp.post("/predict")
def predict():
    user, err = require_user()
    if err is not None:
        return err
    data = request.get_json(silent=True) or {}
    try:
        parsed = validate_irrigation(data)
    except ValidationError as exc:
        return jsonify({"ok": False, "errors": exc.errors}), 400
    result = predict_irrigation(parsed, user_id=user.id)
    result["ok"] = True
    notify_result(user, result, kind="irrigation")
    return jsonify(result)


@bp.get("/summary")
def summary():
    return jsonify({"ok": True, **irrigation_summary()})
