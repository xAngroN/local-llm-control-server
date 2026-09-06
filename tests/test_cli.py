"""Tests for the llamactl command line interface.

``main([...])`` is called directly and the httpx client is monkeypatched,
so no API server and no podman are involved (except for the ``logs``
fallback test, which uses the shared fake podman fixture).
"""

import json

import pytest

from llamactl.cli import main


class FakeResponse:
    """Minimal stand-in for an httpx response."""

    def __init__(self, payload: dict | list, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeClient:
    """Records calls; returns the configured response (or raises)."""

    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, str, dict | None, dict | None]] = []

    def request(self, method: str, path: str, json=None) -> FakeResponse:
        self.calls.append((method, path, json, None))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def get(self, path: str, params=None) -> FakeResponse:
        self.calls.append(("GET", path, None, params))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


@pytest.fixture()
def client(monkeypatch):
    """Monkeypatch llamactl.cli._client with a recording fake."""

    def make(response: FakeResponse | Exception) -> FakeClient:
        fake = FakeClient(response)
        monkeypatch.setattr("llamactl.cli._client", lambda t=10.0: fake)
        return fake

    return make


def test_profiles_calls_get_profiles(client, capsys) -> None:
    fake = client(FakeResponse({"safe": {"ctx_size": 32768, "parallel": 1}}))
    rc = main(["profiles"])
    assert rc == 0
    assert fake.calls == [("GET", "/profiles", None, None)]
    out = capsys.readouterr().out
    assert "safe" in out


def test_profiles_json(client, capsys) -> None:
    payload = {"safe": {"ctx_size": 32768}}
    fake = client(FakeResponse(payload))
    rc = main(["profiles", "--json"])
    assert rc == 0
    assert fake.calls == [("GET", "/profiles", None, None)]
    assert json.loads(capsys.readouterr().out) == payload


def test_start_without_arg_uses_safe(client) -> None:
    fake = client(FakeResponse({"state": "starting"}))
    rc = main(["start"])
    assert rc == 0
    assert fake.calls == [("POST", "/start", {"profile": "safe"}, None)]


def test_start_with_profile(client) -> None:
    fake = client(FakeResponse({"state": "starting"}))
    rc = main(["start", "large"])
    assert rc == 0
    assert fake.calls == [("POST", "/start", {"profile": "large"}, None)]


def test_stop_calls_post_stop(client) -> None:
    fake = client(FakeResponse({"state": "stopped"}))
    rc = main(["stop"])
    assert rc == 0
    assert fake.calls == [("POST", "/stop", None, None)]


def test_restart_calls_post_reload(client) -> None:
    fake = client(FakeResponse({"state": "starting"}))
    rc = main(["restart", "fast"])
    assert rc == 0
    assert fake.calls == [("POST", "/reload", {"profile": "fast"}, None)]


def test_restart_without_profile_is_usage_error(capsys) -> None:
    rc = main(["restart"])
    assert rc == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_status_calls_get_status(client) -> None:
    fake = client(FakeResponse({"state": "ready", "profile": "safe"}))
    rc = main(["status"])
    assert rc == 0
    assert fake.calls == [("GET", "/status", None, None)]


def test_metrics_calls_get_metrics(client, capsys) -> None:
    fake = client(FakeResponse({"vram_used_gib": 2.0}))
    rc = main(["metrics"])
    assert rc == 0
    assert fake.calls == [("GET", "/metrics", None, None)]
    assert "vram_used_gib" in capsys.readouterr().out


def test_suspend_calls_post_suspend(client) -> None:
    fake = client(FakeResponse({"state": "ready", "suspend_requested": True}))
    rc = main(["suspend"])
    assert rc == 0
    assert fake.calls == [("POST", "/suspend", None, None)]


def test_logs_calls_api_with_tail_param(client, capsys) -> None:
    fake = client(FakeResponse({"lines": ["a", "b"]}))
    rc = main(["logs", "--tail", "2"])
    assert rc == 0
    assert fake.calls == [("GET", "/logs", None, {"tail": 2})]
    out = capsys.readouterr().out
    assert "a" in out and "b" in out


def test_logs_json(client, capsys) -> None:
    fake = client(FakeResponse({"lines": ["a"]}))
    rc = main(["logs", "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {"lines": ["a"]}


def test_logs_falls_back_to_podman_when_unreachable(
    fake_podman, client, monkeypatch, capsys
) -> None:
    import httpx

    import llamactl.cli as cli_module

    def raising_client(timeout=10.0):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(cli_module, "_client", raising_client)

    from llamactl.config import load_config
    from llamactl.podman import Podman

    common, _ = load_config()
    Podman().run(["run", "--name", common.container_name, "busybox", "true"])
    rc = main(["logs", "--tail", "3"])
    assert rc == 0
    # The fallback printed the fake podman log lines on stdout.
    assert "fake-podman" in capsys.readouterr().out


def test_api_error_returns_1(client, capsys) -> None:
    client(FakeResponse({"detail": "unknown profile 'nope'"}, 404))
    rc = main(["start", "nope"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "404" in err and "unknown profile" in err


def test_unreachable_server_returns_1(monkeypatch, capsys) -> None:
    import httpx

    import llamactl.cli as cli_module

    def raising_client(timeout=10.0):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(cli_module, "_client", raising_client)
    rc = main(["status"])
    assert rc == 1
    assert "systemctl --user status llamactl" in capsys.readouterr().err


def test_unknown_subcommand_returns_2(capsys) -> None:
    rc = main(["bogus"])
    assert rc == 2


def test_no_args_returns_2(capsys) -> None:
    rc = main([])
    assert rc == 2
