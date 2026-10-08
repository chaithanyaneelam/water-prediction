"""Prediction endpoints (login required - results go to the user's channel)."""
from flask import Blueprint, jsonify, request

from backend.app.blueprints.notify_hook import notify_result, require_user
from backend.app.services.prediction_service import (
    parse_bulk_csv,
    predict_and_persist,
    rebuild_result,
)
from backend.app.validation import ValidationError, validate_reading

bp = Blueprint("predict", __name__, url_prefix="/api/predict")


@bp.post("")
def predict_single():
    user, err = require_user()
    if err is not None:
        return err
    data = request.get_json(silent=True) or {}
    # 'source' is reserved for the demo simulator (source=simulated); the web
    # form never sends it, so normal use is always stored as 'manual'.
    source = data.get("source") if data.get("source") in ("manual", "simulated") else "manual"
    try:
        parsed = validate_reading(data)
    except ValidationError as exc:
        return jsonify({"ok": False, "errors": exc.errors}), 400
    try:
        result = predict_and_persist(parsed, source=source, user_id=user.id)
    except RuntimeError as exc:
        return jsonify({"ok": False, "errors": [str(exc)]}), 503
    result["ok"] = True
    notify_result(user, result, kind="potability")
    return jsonify(result)


@bp.post("/<int:reading_id>/send")
def send_report(reading_id: int):
    """(Re-)send the stored report for one reading to the user's channel:
    email for users registered with an email, SMS for phone users."""
    user, err = require_user()
    if err is not None:
        return err
    result = rebuild_result(reading_id, user.id)
    if result is None:
        return jsonify({"ok": False,
                        "errors": ["report not found (or it belongs to another user)"]}), 404
    notify_result(user, result, kind="potability")
    return jsonify({"ok": True, "reading_id": reading_id,
                    "notification": result.get("notification")})


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
