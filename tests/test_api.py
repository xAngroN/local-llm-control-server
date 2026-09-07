"""Tests for the llamactl API."""

from pathlib import Path

from fastapi.testclient import TestClient

from llamactl import __version__
from llamactl.api import create_app
from llamactl.config import load_config
from llamactl.lifecycle import LifecycleManager
from llamactl.podman import Podman, PodmanError
from llamactl.state import InstanceState, InstanceTracker

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES_TOML = REPO_ROOT / "config" / "profiles.toml"

ALL_PROFILES = {"fast", "large", "safe", "shared"}


def make_client(fake_podman) -> TestClient:
    """Build a TestClient whose app uses a manager wired to the fake podman."""
    common, profiles = load_config(PROFILES_TOML)
    manager = LifecycleManager(common, profiles, Podman(), InstanceTracker())
    return TestClient(create_app(manager))


def test_healthz_returns_ok(fake_podman) -> None:
    client = make_client(fake_podman)
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__


def test_profiles_lists_all_four_with_ctx_and_slot_ctx(fake_podman) -> None:
    client = make_client(fake_podman)
    response = client.get("/profiles")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == ALL_PROFILES
    # shared: 32768 total ctx over 4 slots -> 8192 per slot.
    assert body["shared"]["ctx_size"] == 32768
    assert body["shared"]["parallel"] == 4
    assert body["shared"]["slot_ctx_size"] == 8192
    # fast: single slot, total ctx == slot ctx.
    assert body["fast"]["ctx_size"] == 16384
    assert body["fast"]["parallel"] == 1
    assert body["fast"]["slot_ctx_size"] == 16384
    for name in ALL_PROFILES:
        entry = body[name]
        assert entry["ctx_size"] == entry["parallel"] * entry["slot_ctx_size"]


def test_profiles_expose_effective_tuning(fake_podman) -> None:
    client = make_client(fake_podman)
    body = client.get("/profiles").json()
    entry = body["fast"]
    # per-profile value
    assert entry["batch_size"] == 2048
    assert entry["ubatch_size"] == 512
    # inherited from [common]
    assert entry["n_gpu_layers"] == 999
    assert entry["flash_attn"] == "on"
    assert entry["cont_batching"] is True
    # speculative decoding off by default in the bundled config
    assert entry["spec_type"] is None
    assert entry["spec_draft_n_max"] is None


def test_profile_detail_endpoint_returns_effective_config(fake_podman) -> None:
    client = make_client(fake_podman)
    response = client.get("/profiles/shared")
    assert response.status_code == 200
    body = response.json()
    assert body["ctx_size"] == 32768
    assert body["slot_ctx_size"] == 8192
    assert body["n_gpu_layers"] == 999
    assert body["flash_attn"] == "on"


def test_profile_detail_unknown_returns_404(fake_podman) -> None:
    client = make_client(fake_podman)
    response = client.get("/profiles/nope")
    assert response.status_code == 404
    assert "nope" in response.json()["detail"]


def test_start_valid_profile_returns_state_and_runs_container(fake_podman) -> None:
    client = make_client(fake_podman)
    response = client.post("/start", json={"profile": "fast"})
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "loading"
    assert body["profile"] == "fast"
    assert body["container_id"] is not None
    common, _ = load_config(PROFILES_TOML)
    entry = fake_podman.containers()[common.container_name]
    assert entry["status"] == "running"
    assert body["container_id"] == entry["id"]


def test_start_unknown_profile_returns_404_with_name(fake_podman) -> None:
    client = make_client(fake_podman)
    response = client.post("/start", json={"profile": "nope"})
    assert response.status_code == 404
    assert "nope" in response.json()["detail"]
    assert fake_podman.containers() == {}


def test_second_start_returns_409(fake_podman) -> None:
    client = make_client(fake_podman)
    assert client.post("/start", json={"profile": "fast"}).status_code == 200
    response = client.post("/start", json={"profile": "safe"})
    assert response.status_code == 409


def test_stop_returns_200_and_stopped_state(fake_podman) -> None:
    client = make_client(fake_podman)
    client.post("/start", json={"profile": "fast"})
    response = client.post("/stop")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "stopped"
    assert body["profile"] is None
    assert body["container_id"] is None
    common, _ = load_config(PROFILES_TOML)
    assert fake_podman.containers()[common.container_name]["status"] == "exited"


def test_stop_with_nothing_running_is_not_an_error(fake_podman) -> None:
    client = make_client(fake_podman)
    response = client.post("/stop")
    assert response.status_code == 200
    assert response.json()["state"] == "stopped"


def test_reload_switches_profile(fake_podman) -> None:
    client = make_client(fake_podman)
    client.post("/start", json={"profile": "fast"})
    response = client.post("/reload", json={"profile": "large"})
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "loading"
    assert body["profile"] == "large"
    common, _ = load_config(PROFILES_TOML)
    entry = fake_podman.containers()[common.container_name]
    assert entry["status"] == "running"
    assert body["container_id"] == entry["id"]


def test_reload_unknown_profile_returns_404_and_keeps_instance(fake_podman) -> None:
    client = make_client(fake_podman)
    started = client.post("/start", json={"profile": "fast"}).json()
    response = client.post("/reload", json={"profile": "nope"})
    assert response.status_code == 404
    assert "nope" in response.json()["detail"]
    # The running instance is untouched.
    common, _ = load_config(PROFILES_TOML)
    entry = fake_podman.containers()[common.container_name]
    assert entry["status"] == "running"
    assert entry["id"] == started["container_id"]


def test_status_idle_reports_stopped_with_expected_fields(fake_podman) -> None:
    client = make_client(fake_podman)
    response = client.get("/status")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "stopped"
    assert body["profile"] is None
    assert body["since"]


def test_status_after_start_reports_active_state_and_profile(fake_podman) -> None:
    client = make_client(fake_podman)
    assert client.post("/start", json={"profile": "fast"}).status_code == 200
    response = client.get("/status")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "loading"
    assert body["profile"] == "fast"
    assert body["container_id"] is not None
    assert body["since"]


def test_status_reports_every_state_as_lowercase_string(fake_podman) -> None:
    # Build with a known tracker so we can set states directly.
    common, profiles = load_config(PROFILES_TOML)
    tracker = InstanceTracker()
    manager = LifecycleManager(common, profiles, Podman(), tracker)
    with TestClient(create_app(manager)) as c:
        for state in InstanceState:
            tracker.transition(state, profile="fast")
            response = c.get("/status")
            assert response.status_code == 200
            assert response.json()["state"] == state.value


def test_podman_error_maps_to_500_with_message(fake_podman) -> None:
    common, profiles = load_config(PROFILES_TOML)
    tracker = InstanceTracker()
    manager = LifecycleManager(common, profiles, Podman(), tracker)

    def boom(args: list[str]) -> str:
        raise PodmanError(125, "no such device", args)

    manager._podman.start_container = boom  # type: ignore[method-assign]
    client = TestClient(create_app(manager))

    response = client.post("/start", json={"profile": "fast"})
    assert response.status_code == 500
    assert "no such device" in response.json()["detail"]
    # State endpoint contract: tracker state is stopped after the failure.
    assert tracker.to_dict()["state"] == "stopped"


def test_create_app_without_manager_calls_reconcile_once(fake_podman) -> None:
    common, profiles = load_config(PROFILES_TOML)
    manager = LifecycleManager(common, profiles, Podman(), InstanceTracker())
    calls = {"n": 0}
    original = manager.reconcile

    def counting_reconcile() -> object:
        calls["n"] += 1
        return original()

    manager.reconcile = counting_reconcile  # type: ignore[method-assign]
    with TestClient(create_app(manager)) as client:
        # reconcile must have run exactly once, before any request is served.
        assert calls["n"] == 1
        assert client.get("/healthz").status_code == 200
    assert calls["n"] == 1


def test_create_app_without_manager_builds_from_config(fake_podman) -> None:
    """create_app() with no manager must load the bundled config and serve."""
    with TestClient(create_app()) as client:
        assert client.get("/healthz").status_code == 200
        body = client.get("/profiles").json()
        assert set(body) == ALL_PROFILES


def _suspend_client(fake_podman, monkeypatch) -> TestClient:
    """Client whose POST /suspend uses a stubbed power.suspend()."""
    import llamactl.api as api_module

    monkeypatch.setattr(api_module, "suspend", lambda **kw: None)
    return make_client(fake_podman)


def test_suspend_with_running_instance_does_not_stop_container(
    fake_podman, monkeypatch
) -> None:
    """POST /suspend freezes, never stops: state and container untouched."""
    client = _suspend_client(fake_podman, monkeypatch)
    started = client.post("/start", json={"profile": "fast"}).json()

    stop_calls = {"n": 0}
    real_manager = client.app.state.manager

    def spy_stop():
        stop_calls["n"] += 1
        return real_manager.stop()

    real_manager.stop = spy_stop  # type: ignore[method-assign]

    response = client.post("/suspend")
    assert response.status_code == 200
    body = response.json()
    assert body["suspend_requested"] is True
    # Instance state is unchanged: still loading, same container, same profile.
    assert body["state"] == "loading"
    assert body["profile"] == "fast"
    assert body["container_id"] == started["container_id"]
    # The container keeps running: nothing stopped it.
    common, _ = load_config(PROFILES_TOML)
    entry = fake_podman.containers()[common.container_name]
    assert entry["status"] == "running"
    assert stop_calls["n"] == 0
    # And the tracker still reports the same state.
    assert client.get("/status").json()["state"] == "loading"


def test_suspend_idle_returns_200(fake_podman, monkeypatch) -> None:
    """POST /suspend with no running instance also returns 200."""
    import llamactl.api as api_module

    client = make_client(fake_podman)
    original = api_module.suspend
    api_module.suspend = lambda **kw: None  # type: ignore[assignment]
    try:
        response = client.post("/suspend")
    finally:
        api_module.suspend = original
    assert response.status_code == 200
    body = response.json()
    assert body["suspend_requested"] is True
    assert body["state"] == "stopped"
