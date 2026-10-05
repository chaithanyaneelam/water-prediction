"""History + export endpoints."""
import csv
import io

from flask import Blueprint, Response, jsonify, request

from backend.app.services.readings_service import export_rows, query_readings
from backend.ml.config import FEATURE_COLS

bp = Blueprint("readings", __name__, url_prefix="/api")


@bp.get("/readings")
def list_readings():
    page = request.args.get("page", 1)
    per_page = min(int(request.args.get("per_page", 20)), 100)
    result = query_readings(
        page=page, per_page=per_page,
        potable=request.args.get("potable"),
        anomaly=request.args.get("anomaly"),
        source=request.args.get("source"),
        date_from=request.args.get("date_from"),
        date_to=request.args.get("date_to"),
        search=request.args.get("search"),
    )
    return jsonify({"ok": True, **result})


@bp.get("/export/readings.csv")
def export_csv():
    items = export_rows()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "created_at", "source"] + FEATURE_COLS +
                    ["potability", "probability", "ph_predicted", "ph_filled_by",
                     "is_anomaly", "anomaly_score", "anomaly_reason"])
    for it in items:
        writer.writerow(
            [it["id"], it["created_at"], it["source"]] +
            [it[c] for c in FEATURE_COLS] +
            [it["potability"], it["probability"], it["ph_predicted"],
             it["ph_filled_by"], it["is_anomaly"], it["anomaly_score"],
             it["anomaly_reason"]]
        )
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=readings_export.csv"},
    )
