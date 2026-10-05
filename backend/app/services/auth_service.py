"""Authentication service: register with email OR phone, hashed passwords.

No external dependencies: hashing via werkzeug, sessions via Flask's session.
The notification channel follows the identifier the user registered with:
email -> email notifications, phone -> SMS notifications.
"""
import re

from flask import session

from backend.app.extensions import db
from backend.app.models import User

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^\+?[0-9]{10,15}$")


class AuthError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__("; ".join(errors))


def _norm_email(value):
    if value in (None, ""):
        return None
    v = str(value).strip().lower()
    if not EMAIL_RE.match(v):
        raise AuthError([f"email: '{value}' does not look like a valid email address"])
    return v


def _norm_phone(value):
    if value in (None, ""):
        return None
    v = re.sub(r"[\s\-()]", "", str(value))
    if not PHONE_RE.match(v):
        raise AuthError(["phone: enter 10-15 digits, optionally starting with +"])
    return v


def register(email, phone, password, display_name=None, preferred_channel=None) -> User:
    errors = []
    phone_display = None
    try:
        email = _norm_email(email)
    except AuthError as exc:
        errors += exc.errors
    try:
        phone_display = _norm_phone(phone)
    except AuthError as exc:
        errors += exc.errors
    if email is None and phone_display is None:
        errors.append("identifier: provide an email address or a phone number")
    if not password or len(str(password)) < 8:
        errors.append("password: must be at least 8 characters")
    if errors:
        raise AuthError(errors)

    # Exactly one identifier: if both given, email wins and phone is stored too.
    channel = (preferred_channel or ("email" if email else "sms")).lower()
    if channel not in ("email", "sms", "outbox"):
        channel = "email" if email else "sms"
    if channel == "email" and not email:
        channel = "sms" if phone_display else "outbox"
    if channel == "sms" and not phone_display:
        channel = "email" if email else "outbox"

    if email and User.query.filter_by(email=email).first():
        raise AuthError([f"email: '{email}' is already registered"])
    if phone_display and User.query.filter_by(phone=phone_display).first():
        raise AuthError([f"phone: '{phone_display}' is already registered"])

    user = User(email=email, phone=phone_display, display_name=display_name,
                notify_channel=channel)
    user.set_password(str(password))
    db.session.add(user)
    db.session.commit()
    return user


def authenticate(identifier: str, password: str) -> User:
    """Login with either email or phone number."""
    if not identifier or not password:
        raise AuthError(["identifier and password are both required"])
    ident = str(identifier).strip().lower()
    user = User.query.filter(
        (User.email == ident) | (User.phone == re.sub(r"[\s\-()]", "", str(identifier)))
    ).first()
    if user is None or not user.check_password(str(password)):
        raise AuthError(["wrong identifier or password"])
    return user


def login_session(user: User) -> None:
    session.clear()
    session["user_id"] = user.id


def logout_session() -> None:
    session.clear()


def current_user() -> User | None:
    uid = session.get("user_id")
    return db.session.get(User, uid) if uid else None
