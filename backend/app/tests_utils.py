"""Shared helpers for API tests (temp DB + smoke models)."""
import os
import sys

import pytest

from backend.app import create_app
from backend.app.models import db


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
def smoke_models(tmp_path_factory):
    """Train tiny models once for the whole test session (test fixture data only)."""
    from backend.ml.train_classifier import main as tc
    from backend.ml.train_isolation_forest import main as tif
    from backend.ml.train_ph_regressor import main as tp

    tc(smoke=True)
    tp(smoke=True)
    tif(smoke=True)
    return True
