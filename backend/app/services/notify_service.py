"""Prediction notification service.

Channel per user (chosen at registration):
- email -> SMTP email (configured via .env; see .env.example)
- sms   -> Twilio-compatible REST API (configured via .env)
- outbox-> stored only (default when nothing is configured) - fully offline

Every message is ALWAYS stored in the notifications table, so the flow is
demonstrable without any external service. Send failures are recorded, never
raised into the request.
"""
import os
import smtplib
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from email.mime.text import MIMEText

from backend.app.extensions import db
from backend.app.models import Notification


def _env(key: str) -> str | None:
    v = os.getenv(key)
    return v.strip() if v and v.strip() else None


def smtp_configured() -> bool:
    return bool(_env("SMTP_HOST") and _env("SMTP_FROM") and
                (_env("SMTP_USER") is not None))


def sms_configured() -> bool:
    return bool(_env("TWILIO_ACCOUNT_SID") and _env("TWILIO_AUTH_TOKEN")
                and _env("TWILIO_FROM"))


def _send_email(destination: str, subject: str, body: str) -> tuple[str, str | None]:
    host = _env("SMTP_HOST")
    port = int(_env("SMTP_PORT") or 587)
    user, password = _env("SMTP_USER"), _env("SMTP_PASSWORD")
    sender = _env("SMTP_FROM")
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = destination
    with smtplib.SMTP(host, port, timeout=15) as server:
        server.starttls()
        if user:  # some relays need no auth
            server.login(user, password or "")
        server.sendmail(sender, [destination], msg.as_string())
    return "sent", None


def _send_sms(destination: str, body: str) -> tuple[str, str | None]:
    sid, token, sender = _env("TWILIO_ACCOUNT_SID"), _env("TWILIO_AUTH_TOKEN"), _env("TWILIO_FROM")
    data = urllib.parse.urlencode({
        "To": destination, "From": sender, "Body": body,
    }).encode()
    req = urllib.request.Request(
        f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
        data=data, method="POST")
    import base64
    req.add_header("Authorization", "Basic " +
                   base64.b64encode(f"{sid}:{token}".encode()).decode())
    with urllib.request.urlopen(req, timeout=15) as resp:
        if 200 <= resp.status < 300:
            return "sent", None
    return "failed", f"unexpected status from SMS gateway"


def deliver(user, subject: str, body: str) -> Notification:
    """Store + (best effort) send one notification. Never raises."""
    channel = user.notify_channel
    destination = user.destination
    status, error = "outbox", None

    if channel == "email" and destination:
        if smtp_configured():
            try:
                status, error = _send_email(destination, subject, body)
            except Exception as exc:
                status, error = "failed", f"SMTP: {exc}"
        else:
            status, error = "outbox", "SMTP not configured (see .env.example)"
    elif channel == "sms" and destination:
        if sms_configured():
            try:
                status, error = _send_sms(destination, body)
            except Exception as exc:
                status, error = "failed", f"SMS gateway: {exc}"
        else:
            status, error = "outbox", "SMS gateway not configured (see .env.example)"

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
