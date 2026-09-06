"""Tests for the llamactl API."""

from fastapi.testclient import TestClient

from llamactl import __version__
from llamactl.api import app

client = TestClient(app)


def test_healthz_returns_ok() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
