"""Tests for the llamactl metrics collector.

The test environment has neither sysfs nor a model server: VRAM is
tested against sysfs files recreated in a temporary directory, and all
model-server traffic is either served by a local HTTP server or stubbed
by monkeypatching ``httpx.get``.
"""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient
from pathlib import Path
from wsgiref.simple_server import make_server

from llamactl.api import create_app
from llamactl.config import load_config
from llamactl.lifecycle import LifecycleManager
from llamactl.metrics import MetricsCollector, _parse_prometheus
from llamactl.podman import Podman
from llamactl.state import InstanceState, InstanceTracker

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES_TOML = REPO_ROOT / "config" / "profiles.toml"

PROMETHEUS_SAMPLE = """\
# HELP llamacpp:prompt_tokens_seconds Average prompt throughput in tokens/s
# TYPE llamacpp:prompt_tokens_seconds gauge
llamacpp:prompt_tokens_seconds 51.0
# HELP llamacpp:predicted_tokens_seconds Average generation throughput in tokens/s
# TYPE llamacpp:predicted_tokens_seconds gauge
llamacpp:predicted_tokens_seconds 30.0
some_other_metric 42.5
bad_line_without_value
"""


def _write_sysfs(tmp_path: Path, used: str | None, total: str | None) -> str:
    """Recreate the sysfs layout under tmp_path/card0/device/."""
    device = tmp_path / "card0" / "device"
    device.mkdir(parents=True, exist_ok=True)
    if used is not None:
        (device / "mem_info_vram_used").write_text(used + "\n", encoding="ascii")
    if total is not None:
        (device / "mem_info_vram_total").write_text(total + "\n", encoding="ascii")
    return str(tmp_path) + "/card*/device/mem_info_vram_used"


def test_read_vram_parses_bytes_and_gib(tmp_path) -> None:
    glob = _write_sysfs(tmp_path, "1073741824", "8589934592")
    vram = MetricsCollector(vram_glob=glob).read_vram()
    assert vram["used_bytes"] == 1073741824
    assert vram["total_bytes"] == 8589934592
    assert vram["used_gib"] == 1.0
    assert vram["total_gib"] == 8.0


def test_read_vram_missing_files_yield_none(tmp_path) -> None:
    # Empty directory: the glob matches nothing -> all None, no exception.
    empty = tmp_path / "empty"
    empty.mkdir()
    vram = MetricsCollector(vram_glob=str(empty) + "/card*/device/mem_info_vram_used").read_vram()
    assert vram == {
        "used_bytes": None,
        "total_bytes": None,
        "used_gib": None,
        "total_gib": None,
    }

    # Used file present, total missing -> total stays None, no exception.
    glob = _write_sysfs(tmp_path, "123", None)
    vram = MetricsCollector(vram_glob=glob).read_vram()
    assert vram["used_bytes"] == 123
    assert vram["total_bytes"] is None
    assert vram["used_gib"] == 123 / (1024.0**3)
    assert vram["total_gib"] is None


def test_parse_prometheus_sample_numbers() -> None:
    samples = _parse_prometheus(PROMETHEUS_SAMPLE)
    assert samples["llamacpp:prompt_tokens_seconds"] == 51.0
    assert samples["llamacpp:predicted_tokens_seconds"] == 30.0
    assert samples["some_other_metric"] == 42.5
    # Comment lines and malformed lines are skipped.
    assert "bad_line_without_value" not in samples
    assert "# HELP" not in samples


def _start_model_server(handler) -> tuple[int, object]:
    """Start a wsgiref HTTP server on an ephemeral port in a daemon thread."""
    server = make_server("127.0.0.1", 0, handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server.server_address[1], server


def test_read_model_metrics_from_local_server(tmp_path) -> None:
    import httpx

    def handler(environ, start_response):
        path = environ["PATH_INFO"]
        if path == "/props":
            body = b'{"model_path": "/models/qwen2.5-7b-instruct.gguf"}'
            start_response("200 OK", [("Content-Type", "application/json")])
            return [body]
        if path == "/metrics":
            body = PROMETHEUS_SAMPLE.encode("ascii")
            start_response("200 OK", [("Content-Type", "text/plain")])
            return [body]
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"not found"]

    port, server = _start_model_server(handler)
    try:
        metrics = MetricsCollector(base_url=f"http://127.0.0.1:{port}").read_model_metrics()
    finally:
        server.shutdown()
    assert metrics["model"] == "/models/qwen2.5-7b-instruct.gguf"
    # Values are the tokens/second counters llama.cpp already publishes;
    # the collector reads them as-is without computing a rate.
    assert metrics["prompt_tps"] == 51.0
    assert metrics["generation_tps"] == 30.0


def test_read_model_metrics_unexpected_prometheus_names_yield_none(monkeypatch) -> None:
    """A realistic llama.cpp /metrics payload without our expected names
    must not raise -- throughput stays None instead of breaking /metrics."""
    import httpx

    import llamactl.metrics as metrics_module

    real_llamacpp_output = """\
    # HELP llama_load_time_seconds Time taken to load the model
    # TYPE llama_load_time_seconds gauge
    llama_load_time_seconds 1.234
    # HELP llama_prompt_tokens Total tokens processed by the prompt
    # TYPE llama_prompt_tokens counter
    llama_prompt_tokens 0
    # HELP llama_tokens_predicted Total tokens predicted
    # TYPE llama_tokens_predicted counter
    llama_tokens_predicted 0
    # HELP llama_eval_time_seconds Total time spent evaluating the prompt
    # TYPE llama_eval_time_seconds counter
    llama_eval_time_seconds 0.0
    # HELP llama_sample_time_seconds Total time spent sampling tokens
    # TYPE llama_sample_time_seconds counter
    llama_sample_time_seconds 0.0
    """

    def fake_get(url, **kwargs):
        if str(url).endswith("/props"):
            return _FakeResponse(200, b'{"model_path": "/models/m.gguf"}', "application/json")
        if str(url).endswith("/metrics"):
            return _FakeResponse(200, real_llamacpp_output.encode("ascii"), "text/plain")
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(metrics_module.httpx, "get", fake_get)
    result = MetricsCollector().read_model_metrics()
    assert result == {"model": "/models/m.gguf", "prompt_tps": None, "generation_tps": None}


def test_read_model_metrics_unreachable_yields_none(monkeypatch) -> None:
    import httpx

    import llamactl.metrics as metrics_module

    def boom(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(metrics_module.httpx, "get", boom)
    collector = MetricsCollector(base_url="http://127.0.0.1:1")
    result = collector.read_model_metrics()
    assert result == {"model": None, "prompt_tps": None, "generation_tps": None}


def test_collect_stopped_does_not_call_model_server(tmp_path, monkeypatch) -> None:
    import llamactl.metrics as metrics_module

    glob = _write_sysfs(tmp_path, "1073741824", "8589934592")
    collector = MetricsCollector(vram_glob=glob)

    def no_http(*args, **kwargs):
        raise AssertionError("model server must not be contacted while stopped")

    monkeypatch.setattr(metrics_module.httpx, "get", no_http)
    result = collector.collect(InstanceState.STOPPED, None)
    assert result["model"] is None
    assert result["prompt_tps"] is None
    assert result["generation_tps"] is None
    assert result["state"] == "stopped"
    assert result["profile"] is None
    # Only the VRAM baseline is present.
    assert result["vram"]["used_bytes"] == 1073741824
    assert result["vram"]["total_bytes"] == 8589934592
    assert result["vram"]["used_gib"] == 1.0


def test_collect_ready_reads_throughput_and_vram(tmp_path, monkeypatch) -> None:
    import llamactl.metrics as metrics_module

    glob = _write_sysfs(tmp_path, "2147483648", "8589934592")
    collector = MetricsCollector(vram_glob=glob)
    monkeypatch.setattr(
        collector,
        "read_model_metrics",
        lambda: {
            "model": "/models/m.gguf",
            "prompt_tps": 51.0,
            "generation_tps": 30.0,
        },
    )
    result = collector.collect(InstanceState.READY, "fast")
    assert result["model"] == "/models/m.gguf"
    assert result["prompt_tps"] == 51.0
    assert result["generation_tps"] == 30.0
    assert result["state"] == "ready"
    assert result["profile"] == "fast"
    # VRAM consumption relative to total.
    assert result["vram"]["used_bytes"] == 2147483648
    assert result["vram"]["total_bytes"] == 8589934592
    assert result["vram"]["used_gib"] == 2.0
    assert result["vram"]["total_gib"] == 8.0


def test_get_metrics_endpoint_stopped_returns_200(fake_podman, monkeypatch) -> None:
    """GET /metrics answers 200 while stopped without touching the model server."""
    import llamactl.metrics as metrics_module

    common, profiles = load_config(PROFILES_TOML)
    tracker = InstanceTracker()
    manager = LifecycleManager(common, profiles, Podman(), tracker)
    tracker.transition(InstanceState.STOPPED)

    def no_http(*args, **kwargs):
        raise AssertionError("model server must not be contacted while stopped")

    monkeypatch.setattr(metrics_module.httpx, "get", no_http)
    with TestClient(create_app(manager)) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "stopped"
    assert body["model"] is None
    assert body["vram"] is not None


def test_get_metrics_endpoint_ready_returns_200(fake_podman, monkeypatch) -> None:
    """GET /metrics answers 200 while ready with throughput and model."""
    import llamactl.metrics as metrics_module

    common, profiles = load_config(PROFILES_TOML)
    tracker = InstanceTracker()
    manager = LifecycleManager(common, profiles, Podman(), tracker)
    tracker.transition(InstanceState.READY, profile="fast")

    def fake_get(url, **kwargs):
        if str(url).endswith("/props"):
            return _FakeResponse(
                200, b'{"model_path": "/models/m.gguf"}', "application/json"
            )
        if str(url).endswith("/metrics"):
            return _FakeResponse(
                200, PROMETHEUS_SAMPLE.encode("ascii"), "text/plain"
            )
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(metrics_module.httpx, "get", fake_get)
    # create_app() builds its own MetricsCollector() (imported into
    # llamactl.api's namespace) with the real sysfs glob; without isolating
    # it here, this assertion depends on whether the host running pytest
    # happens to have amdgpu sysfs VRAM files (it does on the real
    # deployment target), unlike every other VRAM check in this file which
    # injects a fake path.
    import llamactl.api as api_module

    monkeypatch.setattr(
        api_module,
        "MetricsCollector",
        lambda **kwargs: MetricsCollector(**{**kwargs, "vram_glob": "/nonexistent/*"}),
    )
    client = TestClient(create_app(manager))
    # The context manager would trigger the startup reconcile, which resets
    # the tracker to stopped; enter it before setting the state instead.
    with client:
        tracker.transition(InstanceState.READY, profile="fast")
        response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "ready"
    assert body["profile"] == "fast"
    assert body["model"] == "/models/m.gguf"
    assert body["prompt_tps"] == 51.0
    assert body["generation_tps"] == 30.0
    # VRAM baseline is still present (sysfs glob patched to not match -> None).
    assert body["vram"]["used_bytes"] is None


class _FakeResponse:
    """Minimal httpx.Response stand-in for the two model-server endpoints."""

    def __init__(self, status_code: int, body: bytes, content_type: str) -> None:
        self.status_code = status_code
        self.content = body
        self.headers = {"Content-Type": content_type}

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> dict:
        import json

        return json.loads(self.content.decode("utf-8"))
