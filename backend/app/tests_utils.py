"""Shared helpers for API tests (temp DB + smoke models)."""
import os
import sys

import pytest
from pytest import MonkeyPatch

from backend.app import create_app
from backend.app.models import db


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch (pytest's built-in is function-scoped)."""
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture()
def app(tmp_path, monkeypatch):
    """App factory against a temp SQLite file DB."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test_api.db'}")
    app = create_app()
    app.config.update(TESTING=True)
    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def smoke_models(tmp_path_factory, monkeypatch_session):
    """Train tiny models once for the whole test session (test fixture data only).

    IMPORTANT: artifact directories are redirected into a temp folder so the
    smoke run can NEVER overwrite the real trained models in saved_models/
    or outputs/ (that exact bug destroyed real artifacts once).
    """
    tmp = tmp_path_factory.mktemp("smoke_artifacts")
    from backend.ml import config as ml_config
    monkeypatch_session.setattr(ml_config, "OUTPUTS_DIR", str(tmp / "outputs"))
    monkeypatch_session.setattr(ml_config, "SAVED_MODELS_DIR", str(tmp / "saved_models"))

    from backend.ml.train_classifier import main as tc
    from backend.ml.train_isolation_forest import main as tif
    from backend.ml.train_ph_regressor import main as tp

    tc(smoke=True)
    tp(smoke=True)
    tif(smoke=True)
    return True
