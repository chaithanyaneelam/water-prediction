"""Authentication service: register with an email, hashed passwords.

No external dependencies: hashing via werkzeug, sessions via Flask's session.
Accounts are email-only (phone registration was removed); predictions are
notified by email (Brevo) or kept in the outbox when no provider is set.
"""
import re

from flask import session

from backend.app.extensions import db
from backend.app.models import User

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__("; ".join(errors))


def _norm_email(value):
    if value in (None, ""):
        raise AuthError(["email: an email address is required"])
    v = str(value).strip().lower()
    if not EMAIL_RE.match(v):
        raise AuthError([f"email: '{value}' does not look like a valid email address"])
    return v


def register(email, password, display_name=None, preferred_channel=None) -> User:
    """Create an account. Channels: 'email' (default) or 'outbox' (stored only)."""
    errors = []
    try:
        email = _norm_email(email)
    except AuthError as exc:
        errors += exc.errors
        email = None
    if not password or len(str(password)) < 8:
        errors.append("password: must be at least 8 characters")
    if errors:
        raise AuthError(errors)

    channel = (preferred_channel or "email").lower()
    if channel not in ("email", "outbox"):
        channel = "email"

    if User.query.filter_by(email=email).first():
        raise AuthError([f"email: '{email}' is already registered"])

    user = User(email=email, phone=None, display_name=display_name,
                notify_channel=channel)
    user.set_password(str(password))
    db.session.add(user)
    db.session.commit()
    return user


def authenticate(identifier: str, password: str) -> User:
    """Login with the registered email address."""
    if not identifier or not password:
        raise AuthError(["email and password are both required"])
    ident = str(identifier).strip().lower()
    user = User.query.filter_by(email=ident).first()
    if user is None or not user.check_password(str(password)):
        raise AuthError(["wrong email or password"])
    return user


def login_session(user: User) -> None:
    session.clear()
    session["user_id"] = user.id


def logout_session() -> None:
    session.clear()


def current_user() -> User | None:
    uid = session.get("user_id")
    return db.session.get(User, uid) if uid else None
