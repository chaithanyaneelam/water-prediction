"""Readings and dashboard statistics services (data comes from the database)."""
from datetime import datetime, timedelta

from sqlalchemy import func, text

from backend.app.extensions import db
from backend.app.models import Prediction, Reading
from backend.ml.config import FEATURE_COLS, GUIDELINE_LIMITS


def query_readings(page=1, per_page=20, potable=None, anomaly=None,
                   source=None, date_from=None, date_to=None, search=None):
    """Paginated, filterable history joined with predictions."""
    q = db.session.query(Reading, Prediction).join(
        Prediction, Reading.id == Prediction.reading_id
    )
    if potable in ("0", "1", 0, 1):
        q = q.filter(Prediction.potability == int(potable))
    if anomaly in ("0", "1", 0, 1, "true", "false"):
        q = q.filter(Prediction.is_anomaly == (str(anomaly).lower() in ("1", "true")))
    if source:
        q = q.filter(Reading.source == source)
    if date_from:
        q = q.filter(Reading.created_at >= datetime.fromisoformat(date_from))
    if date_to:
        q = q.filter(Reading.created_at <= datetime.fromisoformat(date_to))
    if search:
        like = f"%{search}%"
        q = q.filter(db.or_(
            *[func.cast(getattr(Reading, c), db.String).like(like) for c in FEATURE_COLS]
        ))
    total = q.count()
    rows = q.order_by(Reading.created_at.desc(), Reading.id.desc()).offset(
        (max(int(page), 1) - 1) * int(per_page)
    ).limit(int(per_page)).all()
    items = []
    for r, p in rows:
        items.append({
            "id": r.id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "source": r.source,
            **{c: getattr(r, c) for c in FEATURE_COLS},
            "potability": p.potability,
            "probability": p.probability,
            "ph_predicted": p.ph_predicted,
            "ph_filled_by": p.ph_filled_by,
            "is_anomaly": p.is_anomaly,
            "anomaly_score": p.anomaly_score,
            "anomaly_reason": p.anomaly_reason,
        })
    return {"total": total, "page": int(page), "per_page": int(per_page), "items": items}


def export_rows():
    """All reading+prediction rows (for the CSV export endpoint)."""
    result = query_readings(page=1, per_page=10_000_000)
    return result["items"]


def dashboard_stats() -> dict:
    """Dashboard numbers + chart data for graphs 24-27."""
    total = db.session.query(func.count(Reading.id)).scalar() or 0
    potable = db.session.query(func.count(Prediction.id)).filter(
        Prediction.potability == 1).scalar() or 0
    anomalies = db.session.query(func.count(Prediction.id)).filter(
        Prediction.is_anomaly.is_(True)).scalar() or 0

    # Graph 24: potable vs not potable doughnut
    not_potable = (db.session.query(func.count(Prediction.id)).scalar() or 0) - potable

    # Graph 25: readings + anomalies over time (daily, last 30 days)
    since = datetime.utcnow() - timedelta(days=30)
    rows = db.session.query(
        func.date(Reading.created_at).label("day"),
        func.count(Reading.id).label("n"),
        func.sum(db.case((Prediction.is_anomaly.is_(True), 1), else_=0)).label("anom"),
    ).join(Prediction, Prediction.reading_id == Reading.id).filter(
        Reading.created_at >= since
    ).group_by(func.date(Reading.created_at)).order_by(func.date(Reading.created_at)).all()
    over_time = {
        "days": [str(r.day) for r in rows],
        "readings": [int(r.n) for r in rows],
        "anomalies": [int(r.anom or 0) for r in rows],
    }

    # Graph 26: average parameter vs guideline limit (from real stored readings)
    averages = {}
    for c in FEATURE_COLS:
        avg = db.session.query(func.avg(getattr(Reading, c))).scalar()
        averages[c] = round(float(avg), 3) if avg is not None else None
    param_comparison = {
        "labels": FEATURE_COLS,
        "averages": [averages[c] for c in FEATURE_COLS],
        "limits": [
            GUIDELINE_LIMITS[c].get("max") or
            (GUIDELINE_LIMITS[c]["min"] + GUIDELINE_LIMITS[c]["max"]) / 2
            if c == "ph" and "max" in GUIDELINE_LIMITS["ph"] else None
            for c in FEATURE_COLS
        ],
        "footnote": "Guideline limits are WHO/BIS/EPA reference values for display only; "
                    "many potable-labelled samples in the dataset exceed them.",
    }

    # Graph 27: pH distribution with the 6.5-8.5 band highlighted
    ph_values = [float(v[0]) for v in db.session.query(Reading.ph).filter(
        Reading.ph.isnot(None)).all()]
    hist, edges = [], []
    if ph_values:
        import numpy as np
        h, e = np.histogram(ph_values, bins=20)
        hist, edges = [int(v) for v in h], [round(float(x), 3) for x in e]

    recent = query_readings(page=1, per_page=10)["items"]
    return {
        "totals": {
            "readings": total,
            "potable": potable,
            "not_potable": not_potable,
            "anomalies": anomalies,
            "potable_pct": round(100 * potable / total, 1) if total else 0,
            "anomaly_pct": round(100 * anomalies / total, 1) if total else 0,
        },
        "class_balance": {
            "labels": ["Potable", "Not potable"],
            "values": [potable, not_potable],
        },
        "over_time": over_time,
        "param_comparison": param_comparison,
        "ph_distribution": {
            "bin_edges": edges,
            "bin_centers": [round((edges[i] + edges[i + 1]) / 2, 3) for i in range(len(edges) - 1)],
            "counts": hist,
            "band": [6.5, 8.5],
        },
        "recent": recent,
    }
