"""API tests: health, predict (valid/invalid), bulk, readings, stats, export, dataset, plots."""
import io

VALID = {
    "ph": 7.1, "Hardness": 190, "Solids": 20000, "Chloramines": 7,
    "Sulfate": 300, "Conductivity": 410, "Organic_carbon": 10,
    "Trihalomethanes": 60, "Turbidity": 4,
}


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.get_json()["ok"] is True


def test_predict_valid_and_persisted(client, smoke_models):
    res = client.post("/api/predict", json=VALID)
    assert res.status_code == 200
    body = res.get_json()
    assert body["ok"] is True
    assert body["potability"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["ph"] == 7.1
    assert body["ph_filled_by"] == "none"
    assert "reason" in body["anomaly"]
    assert isinstance(body["guideline_hits"], list)
    assert isinstance(body["treatment"], list)

    # It must be persisted and visible in history.
    hist = client.get("/api/readings").get_json()
    assert hist["total"] >= 1
    assert any(it["id"] == body["reading_id"] for it in hist["items"])


def test_predict_missing_ph_is_imputed(client, smoke_models):
    payload = {k: v for k, v in VALID.items() if k != "ph"}
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    body = res.get_json()
    assert body["ph"] is not None          # imputed
    assert body["ph_filled_by"] in ("model", "median")


def test_predict_validation_errors(client):
    res = client.post("/api/predict", json={"ph": 22, "Hardness": -5})
    assert res.status_code == 400
    errors = res.get_json()["errors"]
    assert any("ph" in e for e in errors)
    assert any("Hardness" in e for e in errors)
    # And nothing was stored.
    hist = client.get("/api/readings").get_json()
    assert hist["total"] == 0


def test_predict_missing_required_field(client):
    payload = {k: v for k, v in VALID.items() if k != "Turbidity"}
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 400
    assert any("Turbidity" in e for e in res.get_json()["errors"])


def test_bulk_csv(client, smoke_models):
    csv = (
        "ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity\n"
        "7.0,200,20000,7,300,400,10,60,4\n"
        ",180,18000,6,280,380,9,55,3.5\n"
        "6.0,210,22000,8,340,440,12,70,4.5\n"
    )
    res = client.post("/api/predict/bulk", data=csv, content_type="text/csv")
    assert res.status_code == 200
    body = res.get_json()
    assert body["total_rows"] == 3
    assert body["predicted"] == 3
    assert body["failed_rows"] == 0
    assert len(body["results"]) == 3


def test_bulk_csv_with_bad_rows(client, smoke_models):
    csv = (
        "ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity\n"
        "7.0,200,20000,7,300,400,10,60,4\n"
        "7.0,abc,20000,7,300,400,10,60,4\n"
    )
    res = client.post("/api/predict/bulk", data=csv, content_type="text/csv")
    assert res.status_code == 200
    body = res.get_json()
    assert body["predicted"] == 1
    assert body["failed_rows"] == 1


def test_bulk_missing_column_rejected(client):
    csv = "ph,Hardness\n7.0,200\n"
    res = client.post("/api/predict/bulk", data=csv, content_type="text/csv")
    assert res.status_code == 400


def test_readings_filters(client, smoke_models):
    client.post("/api/predict", json=VALID)
    res = client.get("/api/readings?potable=1")
    assert res.status_code == 200
    for it in res.get_json()["items"]:
        assert it["potability"] == 1


def test_stats_shape(client, smoke_models):
    client.post("/api/predict", json=VALID)
    res = client.get("/api/stats")
    assert res.status_code == 200
    d = res.get_json()
    assert d["totals"]["readings"] >= 1
    for key in ("class_balance", "over_time", "param_comparison",
                "ph_distribution", "recent"):
        assert key in d


def test_export_csv(client, smoke_models):
    client.post("/api/predict", json=VALID)
    res = client.get("/api/export/readings.csv")
    assert res.status_code == 200
    text = res.get_data(as_text=True)
    assert text.splitlines()[0].startswith("id,created_at,source")
    assert len(text.splitlines()) >= 2


def test_models_metrics_and_plots(client, smoke_models):
    res = client.get("/api/models/metrics")
    assert res.status_code == 200
    d = res.get_json()
    assert d["available"]["classifier_metrics"] is True
    assert d["available"]["ph_regressor"] is True

    res = client.get("/api/plots/nonexistent-plot")
    assert res.status_code == 404


def test_pages_render(client):
    for path in ("/", "/predict", "/bulk", "/history", "/comparison", "/explorer"):
        res = client.get(path)
        assert res.status_code == 200
        assert b"WATERNET" in res.data
