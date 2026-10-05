"""Wiring between predictions and the auth/notification system."""
from flask import jsonify

from backend.app.services.auth_service import current_user
from backend.app.services.notify_service import deliver


def require_user():
    """Return (user, error_response). error_response is None when logged in."""
    user = current_user()
    if user is None:
        return None, (jsonify({
            "ok": False,
            "errors": ["login required to run predictions - register with an "
                       "email (results by email) or a phone number (results by SMS)"]
        }), 401)
    return user, None


def notify_result(user, result: dict, kind: str) -> None:
    """Send/store the prediction message via the user's channel. Never raises.

    The channel follows the registration identifier: a user who registered
    with an email gets an email (Brevo), a user who registered with a phone
    number gets an SMS (TextBelt). A summary of the attempt is attached to
    ``result["notification"]`` so the pages can confirm delivery right after
    the Predict / Check Irrigation Suitability button is clicked.
    """
    try:
        from backend.app.services import notify_service
        subject, body = (notify_service.irrigation_message(result) if kind == "irrigation"
                         else notify_service.potability_message(result))
        n = deliver(user, subject, body)
        result["notification"] = {
            "channel": n.channel,
            "destination": n.destination,
            "status": n.status,
            "error": n.error,
        }
    except Exception as exc:  # notifications must never break predictions
        print(f"[notify] failed to deliver: {exc}")
