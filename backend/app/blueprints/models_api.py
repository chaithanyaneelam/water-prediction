"""Model comparison metrics + PNG plot serving."""
import os

from flask import Blueprint, jsonify, send_file

from backend.app.services import plots_service

bp = Blueprint("models", __name__, url_prefix="/api")


@bp.get("/models/metrics")
def model_metrics():
    """Everything the Model Comparison page needs, with availability flags."""
    payload = {
        "ok": True,
        "available": {},
        "hint": "Model artifacts not found. Run: python -m backend.ml.train_all",
    }
    for name in plots_service.MODEL_FILES:
        data = plots_service.get_model_json(name)
        payload[name] = data
        payload["available"][name] = data is not None
    return jsonify(payload)


@bp.get("/plots/<name>")
def plot_png(name: str):
    path = plots_service.png_path(name)
    if path is None:
        return jsonify({"ok": False, "error": f"plot '{name}' not found. "
                        "Run: python -m backend.ml.train_all"}), 404
    return send_file(os.path.abspath(path), mimetype="image/png", conditional=True)
