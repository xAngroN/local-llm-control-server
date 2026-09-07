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


def make_writable_client(fake_podman, tmp_path) -> tuple[TestClient, Path]:
    """TestClient whose manager persists profile CRUD to a temp config file.

    Copies the bundled profiles.toml into ``tmp_path`` so profile
    create/update/delete never touch the repo fixture. Returns
    ``(client, config_path)``.
    """
    cfg = tmp_path / "profiles.toml"
    cfg.write_text(PROFILES_TOML.read_text(encoding="utf-8"), encoding="utf-8")
    common, profiles = load_config(cfg)
    manager = LifecycleManager(
        common, profiles, Podman(), InstanceTracker(), config_path=cfg
    )
    return TestClient(create_app(manager)), cfg


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


# --- Profile CRUD ---------------------------------------------------------

def _new_profile_body(**over) -> dict:
    body = {
        "name": "mtp",
        "model": "m.gguf",
        "ctx_size": 4096,
        "kv_cache_type_k": "q8_0",
        "kv_cache_type_v": "q8_0",
        "parallel": 1,
        "batch_size": 512,
        "spec_type": "draft-mtp",
        "spec_draft_n_max": 4,
    }
    body.update(over)
    return body


def test_create_profile_persists_and_is_listed(fake_podman, tmp_path) -> None:
    client, cfg = make_writable_client(fake_podman, tmp_path)
    response = client.post("/profiles", json=_new_profile_body())
    assert response.status_code == 201
    body = response.json()
    assert body["model"] == "m.gguf"
    assert body["spec_type"] == "draft-mtp"
    assert body["spec_draft_n_max"] == 4
    # visible via the API...
    assert "mtp" in client.get("/profiles").json()
    assert client.get("/profiles/mtp").status_code == 200
    # ...and persisted to the file (survives a fresh load).
    _, profiles = load_config(cfg)
    assert "mtp" in profiles
    # existing comments were preserved.
    assert "shared" in cfg.read_text()


def test_create_duplicate_returns_409(fake_podman, tmp_path) -> None:
    client, _ = make_writable_client(fake_podman, tmp_path)
    response = client.post("/profiles", json=_new_profile_body(name="safe"))
    assert response.status_code == 409


def test_create_missing_field_returns_422(fake_podman, tmp_path) -> None:
    client, cfg = make_writable_client(fake_podman, tmp_path)
    body = _new_profile_body()
    del body["ctx_size"]
    response = client.post("/profiles", json=body)
    assert response.status_code == 422
    # nothing was written.
    assert "mtp" not in load_config(cfg)[1]


def test_create_without_name_returns_422(fake_podman, tmp_path) -> None:
    client, _ = make_writable_client(fake_podman, tmp_path)
    body = _new_profile_body()
    del body["name"]
    assert client.post("/profiles", json=body).status_code == 422


def test_update_profile_changes_fields(fake_podman, tmp_path) -> None:
    client, cfg = make_writable_client(fake_podman, tmp_path)
    response = client.put(
        "/profiles/fast",
        json={
            "model": "fast.gguf",
            "ctx_size": 8192,
            "kv_cache_type_k": "f16",
            "kv_cache_type_v": "f16",
            "parallel": 1,
            "batch_size": 256,
        },
    )
    assert response.status_code == 200
    assert response.json()["ctx_size"] == 8192
    _, profiles = load_config(cfg)
    assert profiles["fast"].ctx_size == 8192
    assert profiles["fast"].batch_size == 256


def test_update_unknown_returns_404(fake_podman, tmp_path) -> None:
    client, _ = make_writable_client(fake_podman, tmp_path)
    response = client.put("/profiles/nope", json=_new_profile_body())
    assert response.status_code == 404


def test_delete_profile_removes_it(fake_podman, tmp_path) -> None:
    client, cfg = make_writable_client(fake_podman, tmp_path)
    response = client.delete("/profiles/shared")
    assert response.status_code == 200
    assert response.json()["deleted"] == "shared"
    assert "shared" not in client.get("/profiles").json()
    assert "shared" not in load_config(cfg)[1]


def test_delete_unknown_returns_404(fake_podman, tmp_path) -> None:
    client, _ = make_writable_client(fake_podman, tmp_path)
    assert client.delete("/profiles/nope").status_code == 404


def test_cannot_delete_or_update_running_profile(fake_podman, tmp_path) -> None:
    client, _ = make_writable_client(fake_podman, tmp_path)
    assert client.post("/start", json={"profile": "fast"}).status_code == 200
    # fast is now the active (loading) profile.
    assert client.delete("/profiles/fast").status_code == 409
    assert client.put(
        "/profiles/fast",
        json={
            "model": "fast.gguf",
            "ctx_size": 8192,
            "kv_cache_type_k": "f16",
            "kv_cache_type_v": "f16",
            "parallel": 1,
            "batch_size": 256,
        },
    ).status_code == 409


# --- geometry / VRAM measurement / labels / config / logs ------------------

def test_profile_detail_exposes_kv_types_labels_and_vram(fake_podman) -> None:
    client = make_client(fake_podman)
    body = client.get("/profiles/safe").json()
    # kv cache types are now in the response (consumer 4.2).
    assert body["kv_cache_type_k"] == "q8_0"
    assert body["kv_cache_type_v"] == "q8_0"
    # labels default to an empty object (consumer 4.3).
    assert body["labels"] == {}
    # never-run profile reports an unknown VRAM measurement (consumer 4.1).
    assert body["vram"] == {
        "peak_bytes": None,
        "measured_at": None,
        "source": "unknown",
    }
    # geometry/estimate keys present (None here: fixture model file absent).
    assert "geometry" in body
    assert "estimated_vram" in body


def test_measured_peak_recorded_then_invalidated_by_update(
    fake_podman, tmp_path
) -> None:
    client, _ = make_writable_client(fake_podman, tmp_path)
    manager = client.app.state.manager
    manager.record_vram_peak("fast", 17_000_000_000)
    body = client.get("/profiles/fast").json()
    assert body["vram"]["source"] == "measured"
    assert body["vram"]["peak_bytes"] == 17_000_000_000
    assert body["vram"]["measured_at"]
    # An update that rewrites the profile drops the measurement (bound to
    # the parameters that produced it) -> back to unknown.
    client.put(
        "/profiles/fast",
        json={
            "model": "fast.gguf", "ctx_size": 8192, "kv_cache_type_k": "f16",
            "kv_cache_type_v": "f16", "parallel": 1, "batch_size": 256,
        },
    )
    body = client.get("/profiles/fast").json()
    assert body["vram"]["source"] == "unknown"
    assert body["vram"]["peak_bytes"] is None


def test_create_profile_with_labels_roundtrips(fake_podman, tmp_path) -> None:
    client, cfg = make_writable_client(fake_podman, tmp_path)
    client.post("/profiles", json=_new_profile_body(labels={"tier": "large"}))
    body = client.get("/profiles/mtp").json()
    assert body["labels"] == {"tier": "large"}
    _, profiles = load_config(cfg)
    assert profiles["mtp"].labels == {"tier": "large"}


def test_config_endpoint_reports_inference_address(fake_podman) -> None:
    client = make_client(fake_podman)
    body = client.get("/config").json()
    assert body["inference_host_port"] == 8080
    assert body["inference_url"].endswith(":8080")
    assert "models_dir" in body


def test_preflight_returns_fit_structure(fake_podman) -> None:
    client = make_client(fake_podman)
    body = client.get("/profiles/safe/preflight").json()
    # Structure is present even when the model file is absent (estimate None).
    assert body["profile"] == "safe"
    assert body["basis"] in ("measured", "estimate")
    assert "fits" in body
    assert "total_vram_bytes" in body


def test_preflight_unknown_404(fake_podman) -> None:
    client = make_client(fake_podman)
    assert client.get("/profiles/nope/preflight").status_code == 404


def test_logs_endpoint_returns_lines(fake_podman) -> None:
    client = make_client(fake_podman)
    client.post("/start", json={"profile": "fast"})
    response = client.get("/logs")
    assert response.status_code == 200
    assert isinstance(response.json()["lines"], list)


def test_trial_unknown_404(fake_podman) -> None:
    client = make_client(fake_podman)
    assert client.post("/profiles/nope/trial", json={}).status_code == 404


def test_trial_conflicts_when_instance_active(fake_podman) -> None:
    client = make_client(fake_podman)
    client.post("/start", json={"profile": "fast"})
    response = client.post("/profiles/safe/trial", json={"timeout": 0})
    assert response.status_code == 409


def test_trial_timeout_zero_starts_and_stops(fake_podman) -> None:
    client = make_client(fake_podman)
    response = client.post("/profiles/fast/trial", json={"timeout": 0})
    assert response.status_code == 200
    body = response.json()
    assert body["started"] is True
    assert body["result"] == "timeout"
    assert body["ready"] is False
    # Trial cleaned up: nothing is left active.
    assert client.get("/status").json()["state"] in ("stopped", "crashed")


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
    # No profile running -> no model.
    assert body["model"] is None
    assert body["since"]


def test_status_after_start_reports_active_state_and_profile(fake_podman) -> None:
    client = make_client(fake_podman)
    assert client.post("/start", json={"profile": "fast"}).status_code == 200
    response = client.get("/status")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "loading"
    assert body["profile"] == "fast"
    # The active profile's model file is surfaced next to the profile name.
    assert body["model"] == "fast.gguf"
    assert body["container_id"] is not None
    assert body["since"]


def test_start_and_reload_responses_include_model(fake_podman) -> None:
    client = make_client(fake_podman)
    started = client.post("/start", json={"profile": "fast"}).json()
    assert started["profile"] == "fast"
    assert started["model"] == "fast.gguf"
    reloaded = client.post("/reload", json={"profile": "large"}).json()
    assert reloaded["profile"] == "large"
    assert reloaded["model"] == "large.gguf"


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
