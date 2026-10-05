"""Auth + notification tests: register (email), login, protected predictions,
outbox delivery, message content."""
import pytest

from backend.app.tests_utils import smoke_models  # noqa: F401

VALID = {
    "ph": 7.1, "Hardness": 190, "Solids": 20000, "Chloramines": 7,
    "Sulfate": 300, "Conductivity": 410, "Organic_carbon": 10,
    "Trihalomethanes": 60, "Turbidity": 4,
}


def _register(client, email, **kw):
    payload = dict(email=email, password="test-password-123", **kw)
    return client.post("/api/auth/register", json=payload)


def test_register_with_email_and_login(client):
    r = _register(client, "a@x.com")
    assert r.status_code == 200
    assert r.get_json()["user"]["notify_channel"] == "email"
    client.post("/api/auth/logout")
    r = client.post("/api/auth/login",
                    json={"identifier": "a@x.com", "password": "test-password-123"})
    assert r.status_code == 200


def test_register_requires_email_phone_not_accepted(client):
    # Phone-only registration is no longer supported (email-only auth).
    r = client.post("/api/auth/register",
                    json={"phone": "+919876543210", "password": "test-password-123"})
    assert r.status_code == 400
    # An email is mandatory.
    r = client.post("/api/auth/register", json={"password": "test-password-123"})
    assert r.status_code == 400


def test_register_rejects_bad_identifier_and_short_password(client):
    r = client.post("/api/auth/register", json={"email": "not-an-email",
                                                "password": "longenough1"})
    assert r.status_code == 400
    r = client.post("/api/auth/register", json={"email": "b@x.com", "password": "short"})
    assert r.status_code == 400
    r = client.post("/api/auth/register", json={"password": "longenough1"})
    assert r.status_code == 400  # no email given


def test_register_duplicate_email_rejected(client):
    assert _register(client, "dup@x.com").status_code == 200
    assert _register(client, "dup@x.com").status_code == 400


def test_login_rejects_phone_identifier(client):
    # Accounts are email-only: a phone number can no longer log in.
    assert _register(client, "phoneless@x.com").status_code == 200
    client.post("/api/auth/logout")
    r = client.post("/api/auth/login",
                    json={"identifier": "+919876543210", "password": "test-password-123"})
    assert r.status_code == 401


def test_predict_requires_login(client, smoke_models):
    res = client.post("/api/predict", json=VALID)
    assert res.status_code == 401
    assert "login required" in res.get_json()["errors"][0]


def test_predict_stores_notification_for_email_user(auth_client, smoke_models):
    res = auth_client.post("/api/predict", json=VALID)
    assert res.status_code == 200
    body = res.get_json()
    notifs = auth_client.get("/api/auth/notifications").get_json()["items"]
    assert len(notifs) == 1
    n = notifs[0]
    assert n["channel"] == "email"
    assert n["destination"].endswith("@test.local")
    # Without SMTP configured it must be in the outbox, not silently 'sent'.
    assert n["status"] == "outbox"
    assert "POTABLE" in n["subject"].upper()
    assert f"{100 * body['probability']:.1f}%" in n["body"]


def test_irrigation_predict_sends_email_notification(client, smoke_models):
    r = client.post("/api/auth/register",
                    json={"email": "irr-user@x.com", "password": "test-password-123"})
    assert r.status_code == 200
    res = client.post("/api/irrigation/predict", json={
        "EC": 1407, "Na": 95, "Ca": 48, "Mg": 111.826, "HCO3": 240, "CO3": 0})
    assert res.status_code == 200
    notifs = client.get("/api/auth/notifications").get_json()["items"]
    assert any(n["channel"] == "email" and "C3S1" in n["subject"] for n in notifs)


def test_notifications_require_login(client):
    assert client.get("/api/auth/notifications").status_code == 401


def test_send_report_resends_prediction(auth_client, smoke_models):
    res = auth_client.post("/api/predict", json=VALID)
    rid = res.get_json()["reading_id"]
    r = auth_client.post(f"/api/predict/{rid}/send")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["notification"]["channel"] == "email"
    assert body["notification"]["status"] == "outbox"  # no provider keys in tests
    notifs = auth_client.get("/api/auth/notifications").get_json()["items"]
    assert len(notifs) == 2  # auto-send + manual resend


def test_send_report_missing_or_foreign_reading(auth_client, smoke_models):
    # unknown reading id
    assert auth_client.post("/api/predict/999999/send").status_code == 404
    # another user's reading is invisible too (registering logs the new user in)
    res = auth_client.post("/api/predict", json=VALID)
    rid = res.get_json()["reading_id"]
    r = auth_client.post("/api/auth/register",
                         json={"email": "second-user@x.com", "password": "test-password-123"})
    assert r.status_code == 200
    assert auth_client.post(f"/api/predict/{rid}/send").status_code == 404


def test_login_page_renders(client):
    assert client.get("/login").status_code == 200
