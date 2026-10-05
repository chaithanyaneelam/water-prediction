"""Auth + notification tests: register (email/phone), login, protected predictions,
outbox delivery, message content."""
import pytest

from backend.app.tests_utils import smoke_models  # noqa: F401

VALID = {
    "ph": 7.1, "Hardness": 190, "Solids": 20000, "Chloramines": 7,
    "Sulfate": 300, "Conductivity": 410, "Organic_carbon": 10,
    "Trihalomethanes": 60, "Turbidity": 4,
}


def _register(client, suffix, **kw):
    payload = dict(password="test-password-123", **kw)
    if suffix.endswith("@x.com"):
        payload["email"] = suffix
    else:
        payload["phone"] = suffix
    return client.post("/api/auth/register", json=payload)


def test_register_with_email_and_login(client):
    r = _register(client, "a@x.com")
    assert r.status_code == 200
    assert r.get_json()["user"]["notify_channel"] == "email"
    client.post("/api/auth/logout")
    r = client.post("/api/auth/login",
                    json={"identifier": "a@x.com", "password": "test-password-123"})
    assert r.status_code == 200


def test_register_with_phone_and_login(client):
    r = _register(client, "+919876543210")
    assert r.status_code == 200
    assert r.get_json()["user"]["notify_channel"] == "sms"
    client.post("/api/auth/logout")
    r = client.post("/api/auth/login",
                    json={"identifier": "+919876543210", "password": "test-password-123"})
    assert r.status_code == 200


def test_register_rejects_bad_identifier_and_short_password(client):
    r = client.post("/api/auth/register", json={"email": "not-an-email",
                                                "password": "longenough1"})
    assert r.status_code == 400
    r = client.post("/api/auth/register", json={"email": "b@x.com", "password": "short"})
    assert r.status_code == 400
    r = client.post("/api/auth/register", json={"password": "longenough1"})
    assert r.status_code == 400  # neither email nor phone


def test_register_duplicate_email_rejected(client):
    assert _register(client, "dup@x.com").status_code == 200
    assert _register(client, "dup@x.com").status_code == 400


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


def test_irrigation_predict_sends_sms_channel_notification(client, smoke_models):
    r = client.post("/api/auth/register",
                    json={"phone": "+919800000001", "password": "test-password-123"})
    assert r.status_code == 200
    res = client.post("/api/irrigation/predict", json={
        "EC": 1407, "Na": 95, "Ca": 48, "Mg": 111.826, "HCO3": 240, "CO3": 0})
    assert res.status_code == 200
    notifs = client.get("/api/auth/notifications").get_json()["items"]
    assert any(n["channel"] == "sms" and "C3S1" in n["subject"] for n in notifs)


def test_notifications_require_login(client):
    assert client.get("/api/auth/notifications").status_code == 401


def test_login_pages_render(client):
    for path in ("/login", "/notifications"):
        assert client.get(path).status_code == 200
