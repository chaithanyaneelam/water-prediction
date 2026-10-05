"""Auth + notification endpoints."""
from flask import Blueprint, jsonify, request

from backend.app.extensions import db
from backend.app.models import Notification
from backend.app.services.auth_service import (
    AuthError, authenticate, current_user, login_session, logout_session, register,
)

bp = Blueprint("auth", __name__, url_prefix="/api/auth")


def _user_json(u):
    return {
        "id": u.id, "email": u.email,
        "display_name": u.display_name, "notify_channel": u.notify_channel,
        "destination": u.destination,
    }


@bp.post("/register")
def register_route():
    data = request.get_json(silent=True) or {}
    try:
        user = register(
            email=data.get("email"),
            password=data.get("password"), display_name=data.get("display_name"),
            preferred_channel=data.get("preferred_channel"),
        )
    except AuthError as exc:
        return jsonify({"ok": False, "errors": exc.errors}), 400
    login_session(user)
    return jsonify({"ok": True, "user": _user_json(user)})


@bp.post("/login")
def login_route():
    data = request.get_json(silent=True) or {}
    try:
        user = authenticate(data.get("identifier"), data.get("password"))
    except AuthError as exc:
        return jsonify({"ok": False, "errors": exc.errors}), 401
    login_session(user)
    return jsonify({"ok": True, "user": _user_json(user)})


@bp.post("/logout")
def logout_route():
    logout_session()
    return jsonify({"ok": True})


@bp.get("/me")
def me():
    u = current_user()
    if u is None:
        return jsonify({"ok": True, "user": None})
    return jsonify({"ok": True, "user": _user_json(u)})


@bp.get("/notifications")
def notifications():
    u = current_user()
    if u is None:
        return jsonify({"ok": False, "errors": ["login required"]}), 401
    rows = (Notification.query.filter_by(user_id=u.id)
            .order_by(Notification.created_at.desc()).limit(50).all())
    return jsonify({"ok": True, "items": [{
        "id": n.id, "channel": n.channel, "destination": n.destination,
        "subject": n.subject, "body": n.body, "status": n.status,
        "error": n.error,
        "created_at": n.created_at.isoformat() if n.created_at else None,
    } for n in rows]})


@bp.post("/notifications/<int:nid>/retry")
def retry(nid: int):
    """Re-attempt delivery of a failed/outbox message (after configuring SMTP/SMS)."""
    from backend.app.services.notify_service import deliver
    u = current_user()
    if u is None:
        return jsonify({"ok": False, "errors": ["login required"]}), 401
    n = db.session.get(Notification, nid)
    if n is None or n.user_id != u.id:
        return jsonify({"ok": False, "errors": ["notification not found"]}), 404
    fresh = deliver(u, n.subject, n.body)
    return jsonify({"ok": True, "status": fresh.status, "error": fresh.error})
