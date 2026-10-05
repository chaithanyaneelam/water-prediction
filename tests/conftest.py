"""Pytest configuration: expose shared fixtures for API tests."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.tests_utils import (  # noqa: F401,E402
    app, auth_client, client, monkeypatch_session, smoke_models,
)
