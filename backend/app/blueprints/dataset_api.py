"""Dataset explorer endpoint."""
from flask import Blueprint, jsonify

from backend.app.services.dataset_service import dataset_summary

bp = Blueprint("dataset", __name__, url_prefix="/api/dataset")


@bp.get("/summary")
def summary():
    return jsonify({"ok": True, **dataset_summary()})
