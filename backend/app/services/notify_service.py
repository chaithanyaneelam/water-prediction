"""Prediction notification service.

Channel per user (chosen at registration):
- email -> Brevo (Sendinblue) transactional email API   (BREVO_API_KEY in .env)
- sms   -> TextBelt SMS API                             (SMS_API_KEY in .env)
- outbox-> stored only (default when nothing is configured) - fully offline

Every message is ALWAYS stored in the notifications table, so the flow is
demonstrable without any external service and delivery problems are recorded
in the error column instead of breaking the prediction request.
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from backend.app.extensions import db
from backend.app.models import Notification


def _env(key: str) -> str | None:
    v = os.getenv(key)
    return v.strip() if v and v.strip() else None


def email_configured() -> bool:
    return bool(_env("BREVO_API_KEY") and _env("BREVO_SENDER_EMAIL"))


def sms_configured() -> bool:
    return bool(_env("SMS_API_KEY"))


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode(errors="replace")[:300]
    except Exception:
        return str(exc)


def _send_email(destination: str, subject: str, body: str) -> tuple[str, str | None]:
    """Send via the Brevo transactional email API (developers.brevo.com).

    BREVO_SENDER_EMAIL must be a sender verified in the Brevo account. If the
    account uses IP whitelisting, requests from other IPs get a 401 whose body
    explains how to authorize the address (Security > Authorised IPs).
    """
    payload = json.dumps({
        "sender": {
            "name": _env("BREVO_SENDER_NAME") or "WATERNET",
            "email": _env("BREVO_SENDER_EMAIL"),
        },
        "to": [{"email": destination}],
        "subject": subject,
        "textContent": body,
    }).encode()
    req = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email", data=payload, method="POST")
    req.add_header("api-key", _env("BREVO_API_KEY") or "")
    req.add_header("content-type", "application/json")
    req.add_header("accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if 200 <= resp.status < 300:
                return "sent", None
            return "failed", f"Brevo HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return "failed", f"Brevo: {_http_error_detail(exc)}"
    except Exception as exc:
        return "failed", f"Brevo: {exc}"


def _send_sms(destination: str, body: str) -> tuple[str, str | None]:
    """Send via the TextBelt SMS API (textbelt.com).

    TextBelt replies 200 with {"success": false, "error": "..."} for quota or
    key problems, so the error text (e.g. 'Out of quota') is surfaced directly.
    """
    data = urllib.parse.urlencode({
        "phone": destination,
        "message": body[:480],  # keep well under TextBelt's hard message cap
        "key": _env("SMS_API_KEY") or "",
    }).encode()
    req = urllib.request.Request("https://textbelt.com/text", data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            out = json.loads(resp.read().decode(errors="replace") or "{}")
    except urllib.error.HTTPError as exc:
        return "failed", f"TextBelt: {_http_error_detail(exc)}"
    except Exception as exc:
        return "failed", f"TextBelt: {exc}"
    if out.get("success"):
        return "sent", None
    return "failed", f"TextBelt: {out.get('error', 'unknown error')}"


def deliver(user, subject: str, body: str) -> Notification:
    """Store + (best effort) send one notification. Never raises."""
    channel = user.notify_channel
    destination = user.destination
    status, error = "outbox", None

    if channel == "email" and destination:
        if email_configured():
            status, error = _send_email(destination, subject, body)
        else:
            status, error = "outbox", "Brevo not configured (see .env.example)"
    elif channel == "sms" and destination:
        if sms_configured():
            status, error = _send_sms(destination, body)
        else:
            status, error = "outbox", "TextBelt not configured (see .env.example)"

    n = Notification(
        user_id=user.id, channel=channel, destination=destination,
        subject=subject, body=body, status=status, error=error,
        sent_at=datetime.now(timezone.utc) if status == "sent" else None,
    )
    db.session.add(n)
    db.session.commit()
    return n


def potability_message(result: dict) -> tuple[str, str]:
    """(subject, body) for a drinking-water prediction result dict."""
    label = result["potability_label"].upper()
    prob = f"{100 * result['probability']:.1f}%"
    ph = result.get("ph")
    ph_txt = f"{ph}" if ph is not None else "imputed"
    anom = "YES - " + result["anomaly"]["reason"] if result["anomaly"]["flag"] else "no"
    subject = f"WATERNET: your water sample #{result['reading_id']} is {label}"
    body = (f"WATERNET prediction for sample #{result['reading_id']}:\n"
            f"- Potability: {label} (probability potable: {prob})\n"
            f"- pH used: {ph_txt}\n"
            f"- Anomaly: {anom}\n"
            + (f"- Treatment: {' '.join(result['treatment'])}" if result["treatment"] else ""))
    return subject, body


def irrigation_message(result: dict) -> tuple[str, str]:
    subject = (f"WATERNET: sample #{result['reading_id']} irrigation class "
               f"{result['ussl_class']} - {'SUITABLE' if result['suitable'] else 'NOT SUITABLE'}")
    body = (f"WATERNET irrigation check for sample #{result['reading_id']}:\n"
            f"- USSL class: {result['ussl_class']} (SAR {result['sar']}, RSC {result['rsc']} {result['rsc_class']})\n"
            f"- Verdict: {result['verdict']}\n"
            + "\n".join("- " + n for n in result["notes"]))
    return subject, body
