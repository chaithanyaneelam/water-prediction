"""Pytest configuration: expose shared fixtures for API tests."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.tests_utils import app, client, smoke_models  # noqa: F401,E402
