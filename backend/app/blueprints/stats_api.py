"""Dashboard stats endpoint."""
from flask import Blueprint, jsonify

from backend.app.services.readings_service import dashboard_stats

bp = Blueprint("stats", __name__, url_prefix="/api")


@bp.get("/stats")
def stats():
    return jsonify({"ok": True, **dashboard_stats()})
