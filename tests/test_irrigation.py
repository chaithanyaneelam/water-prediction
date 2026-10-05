"""Irrigation module tests: rule formulas (hand-computed), API, leakage guard."""
import pytest

from backend.app.tests_utils import client, smoke_models  # noqa: F401
from backend.ml.config import IRRIGATION_FEATURES
from backend.ml.irrigation_rules import (
    classify_rsc, compute_rsc, compute_sar, ussl_class, verdict,
)

VALID = {"EC": 1407, "Na": 95, "Ca": 48, "Mg": 111.826, "HCO3": 240, "CO3": 0}


def test_sar_formula_hand_computed():
    # Na 95 mg/L = 4.131 meq; Ca 48 = 2.395 meq; Mg 111.826 = 9.205 meq
    # SAR = 4.131 / sqrt((2.395 + 9.205) / 2) = 1.716 (matches dataset row 1: 1.7153)
    assert round(compute_sar(95, 48, 111.826), 3) == 1.716


def test_rsc_formula_hand_computed():
    # (CO3/30 + HCO3/61.02) - (Ca/20.04 + Mg/12.15) = -7.666 meq/L
    assert round(compute_rsc(0, 240, 48, 111.826), 3) == -7.666


def test_ussl_class_boundaries():
    assert ussl_class(200, 5)["class"] == "C1S1"
    assert ussl_class(500, 15)["class"] == "C2S2"
    assert ussl_class(1000, 20)["class"] == "C3S3"
    assert ussl_class(3000, 30)["class"] == "C4S4"
    assert ussl_class(750, 10)["class"] == "C3S2"  # boundary: 750 -> C3, 10 -> S2
    assert ussl_class(250, 10)["class"] == "C2S2"  # boundary: 250 -> C2


def test_rsc_class_boundaries():
    assert classify_rsc(-2.0)["code"] == "P.S."
    assert classify_rsc(1.0)["code"] == "P.S."
    assert classify_rsc(1.3)["code"] == "MR"
    assert classify_rsc(2.4)["code"] == "MR"
    assert classify_rsc(3.0)["code"] == "U.S."


def test_verdict_flags_hazards():
    # High salinity + high sodium + unsuitable RSC
    v = verdict(3000, 30, 3.0)
    assert v["suitable"] is False
    assert any("salinity" in n.lower() for n in v["notes"])
    assert any("sodium" in n.lower() for n in v["notes"])

    # Clean water: C1S1, safe RSC
    v2 = verdict(150, 2, -1.0)
    assert v2["suitable"] is True
    assert v2["verdict"].startswith("SUITABLE")


def test_label_columns_never_features():
    for label in ("SAR", "ussl_class", "rsc_class", "RSC"):
        assert label not in IRRIGATION_FEATURES


def test_api_predict_roundtrip(auth_client, smoke_models):
    res = auth_client.post("/api/irrigation/predict", json=VALID)
    assert res.status_code == 200
    body = res.get_json()
    assert body["ok"] is True
    assert body["ussl_class"] == "C3S1"
    assert abs(body["sar"] - 1.716) < 0.001
    assert isinstance(body["suitable"], bool)

    # Persisted and visible in summary
    summary = auth_client.get("/api/irrigation/summary").get_json()
    assert summary["stored"]["total"] >= 1


def test_api_validation_missing_required(auth_client):
    res = auth_client.post("/api/irrigation/predict",
                      json={"EC": 1407, "Ca": 48, "Mg": 111, "HCO3": 240})
    assert res.status_code == 400
    assert any("Na" in e for e in res.get_json()["errors"])


def test_api_validation_negative_ec(auth_client):
    res = auth_client.post("/api/irrigation/predict",
                      json={"EC": -5, "Na": 95, "Ca": 48, "Mg": 111, "HCO3": 240})
    assert res.status_code == 400
    assert any("EC" in e for e in res.get_json()["errors"])
