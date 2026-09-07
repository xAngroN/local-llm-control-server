"""FastAPI application for the llamactl control server.

This module only translates HTTP calls into :class:`LifecycleManager`
method calls and exceptions into HTTP status codes.  All lifecycle logic
stays in :mod:`llamactl.lifecycle`; there are no podman calls and no
state logic in this module.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException

from llamactl import __version__
from llamactl.config import load_config, resolve_profiles_path, resolve_tuning
from llamactl.gguf import estimate_vram, read_gguf_info
from llamactl.events import ContainerEventWatcher
from llamactl.health import HealthPoller
from llamactl.lifecycle import (
    AlreadyRunningError,
    LifecycleManager,
    ProfileExistsError,
    ProfileInUseError,
    UnknownProfileError,
)
from llamactl.metrics import MetricsCollector
from llamactl.power import PowerError, suspend
from llamactl.podman import Podman, PodmanError
from llamactl.state import InstanceState, InstanceTracker
from llamactl.suspend_watch import PowerStateWatcher

logger = logging.getLogger(__name__)


def _vram_measurement(profile) -> dict:
    """The nested ``vram`` block for a profile detail.

    ``source`` is ``"measured"`` once a run has recorded a peak (bound to
    the current parameters -- any profile edit drops it, so a resized
    profile reports ``"unknown"`` again) and ``"unknown"`` otherwise.
    """
    peak = profile.measured_vram_peak_bytes
    return {
        "peak_bytes": peak,
        "measured_at": profile.measured_vram_peak_at,
        "source": "measured" if peak is not None else "unknown",
    }


def _model_path(manager: LifecycleManager, profile) -> str:
    """Absolute host path of a profile's GGUF model file."""
    return os.path.join(manager._common.models_dir, profile.model)


def _model_file_size(manager: LifecycleManager, profile) -> int | None:
    """Size of the profile's model file in bytes, ``None`` when absent."""
    try:
        return os.path.getsize(_model_path(manager, profile))
    except OSError:
        return None


def _profile_detail(
    manager: LifecycleManager, name: str, with_geometry: bool = False
) -> dict:
    """Build the read-back body for a single profile.

    Reports the sizing fields plus the *effective* tuning configuration
    (:func:`resolve_tuning`) -- i.e. the per-profile value merged over the
    ``[common]`` default -- so callers can read exactly which llama.cpp
    knobs (``-b``/``-ub``, flash attention, ngl, continuous batching and
    the MTP/speculative-decoding settings) the profile will run with.

    Always includes the model file size and the last measured peak VRAM.
    When ``with_geometry`` is set, the GGUF header geometry and a VRAM
    estimate are read from the model file too (a small mmap parse); the
    list endpoint keeps that off to stay cheap.
    """
    profile = manager._profiles[name]
    tuning = resolve_tuning(profile, manager._common)
    body = {
        "model": profile.model,
        "ctx_size": profile.ctx_size,
        "parallel": profile.parallel,
        "slot_ctx_size": profile.slot_ctx_size,
        "batch_size": profile.batch_size,
        # kv cache types drive the KV footprint, so the consumer needs them
        # in the response (they were only in the file before).
        "kv_cache_type_k": profile.kv_cache_type_k,
        "kv_cache_type_v": profile.kv_cache_type_v,
        "ubatch_size": tuning["ubatch_size"],
        "n_gpu_layers": tuning["n_gpu_layers"],
        "flash_attn": tuning["flash_attn"],
        "cont_batching": tuning["cont_batching"],
        "cache_reuse": tuning["cache_reuse"],
        "spec_type": tuning["spec_type"],
        "spec_draft_n_max": tuning["spec_draft_n_max"],
        "spec_draft_n_min": tuning["spec_draft_n_min"],
        "labels": profile.labels or {},
        "model_file_size_bytes": _model_file_size(manager, profile),
        "vram": _vram_measurement(profile),
    }
    if with_geometry:
        info = read_gguf_info(_model_path(manager, profile))
        body["geometry"] = info.as_dict() if info is not None else None
        body["estimated_vram"] = (
            estimate_vram(
                info,
                profile.ctx_size,
                profile.kv_cache_type_k,
                profile.kv_cache_type_v,
            )
            if info is not None else None
        )
    return body


def _profiles_response(manager: LifecycleManager) -> dict:
    """Build the /profiles body: name -> effective profile detail (no GGUF)."""
    return {
        name: _profile_detail(manager, name)
        for name in manager.list_profiles()
    }


def _status_body(manager: LifecycleManager) -> dict:
    """Tracker status enriched with the active profile's model file.

    The :class:`InstanceTracker` is deliberately config-agnostic: it only
    knows the profile *name*. Every runtime-state response (``/status``,
    ``/start``, ``/stop``, ``/reload``, ``/suspend``) should also report
    *which model* the active profile runs, so the model file is resolved
    here from the manager's profiles and inserted right after ``profile``.
    It is ``None`` when nothing is running (profile ``None``) or the active
    profile is no longer in the config.
    """
    body = manager._tracker.to_dict()
    profile_name = body.get("profile")
    profile = manager._profiles.get(profile_name) if profile_name else None
    model = profile.model if profile is not None else None
    result: dict = {}
    for key, value in body.items():
        result[key] = value
        if key == "profile":
            result["model"] = model
    return result


def _gib(value: int | None) -> float | None:
    return round(value / (1024.0**3), 3) if value is not None else None


def _preflight(manager: LifecycleManager, collector, name: str) -> dict:
    """Estimate whether ``name`` fits in VRAM without loading it.

    The required VRAM is the measured peak when one exists (preferred:
    exact, parameter-bound) and otherwise the GGUF-derived estimate
    (an upper bound). ``fits`` compares that against total VRAM; it is
    ``None`` when neither a requirement nor the total can be determined.
    """
    detail = _profile_detail(manager, name, with_geometry=True)
    profile = manager._profiles[name]
    total_bytes = collector.read_vram().get("total_bytes")
    measured = profile.measured_vram_peak_bytes
    estimate = detail.get("estimated_vram") or {}
    est_total = estimate.get("total_bytes")
    if measured is not None:
        required, basis = measured, "measured"
    else:
        required, basis = est_total, "estimate"
    fits = None
    headroom = None
    if required is not None and total_bytes is not None:
        headroom = total_bytes - required
        fits = headroom >= 0
    return {
        "profile": name,
        "model": profile.model,
        "geometry": detail.get("geometry"),
        "model_file_size_bytes": detail.get("model_file_size_bytes"),
        "estimated_vram": detail.get("estimated_vram"),
        "vram": detail.get("vram"),
        "basis": basis,
        "required_bytes": required,
        "required_gib": _gib(required),
        "total_vram_bytes": total_bytes,
        "total_vram_gib": _gib(total_bytes),
        "headroom_bytes": headroom,
        "headroom_gib": _gib(headroom),
        "fits": fits,
    }


def _probe_health(base_url: str) -> bool:
    """One HTTP health probe of the model server; ``True`` only on 200."""
    import httpx

    try:
        response = httpx.get(base_url.rstrip("/") + "/health", timeout=3.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def _run_trial(
    manager: LifecycleManager,
    collector,
    name: str,
    timeout: int,
    keep: bool,
    poll_interval: float = 1.0,
) -> dict:
    """Load a profile once, observe the outcome, then stop it again.

    Starts the container (raising :class:`AlreadyRunningError` when an
    instance is already active), then polls until the model server answers
    ``/health`` (``ready``), the container dies (``crashed``, typically
    OOM) or ``timeout`` seconds pass (``timeout``). The peak VRAM seen is
    recorded via :meth:`LifecycleManager.record_vram_peak`. Unless ``keep``
    is set and it became ready, the container is stopped again so the trial
    leaves the system as it found it.
    """
    base_url = f"http://127.0.0.1:{manager._common.host_port}"
    manager.start(name)  # may raise AlreadyRunningError / PodmanError
    deadline = time.monotonic() + max(0, timeout)
    peak = 0
    result = "timeout"
    while time.monotonic() < deadline:
        used = collector.read_vram().get("used_bytes")
        if used:
            peak = max(peak, used)
        if not manager._podman.is_running(manager._common.container_name):
            result = "crashed"
            break
        if _probe_health(base_url):
            result = "ready"
            break
        time.sleep(poll_interval)
    try:
        log_tail = manager._podman.logs(manager._common.container_name, tail=50)
    except PodmanError:
        log_tail = []
    recorded = (
        manager.record_vram_peak(name, peak) if result == "ready" and peak else None
    )
    kept = keep and result == "ready"
    if not kept:
        try:
            manager.stop()
        except PodmanError:
            pass
    return {
        "profile": name,
        "model": manager._profiles[name].model,
        "started": True,
        "result": result,
        "ready": result == "ready",
        "peak_bytes": peak or None,
        "peak_gib": _gib(peak or None),
        "recorded_peak_bytes": recorded,
        "log_tail": log_tail,
        "kept": kept,
    }


def create_app(manager: LifecycleManager | None = None) -> FastAPI:
    """Build the FastAPI application for a given lifecycle manager.

    When ``manager`` is ``None`` the application constructs its own
    manager from :func:`load_config`, :class:`Podman` and a fresh
    :class:`InstanceTracker`.  In either case the startup hook (FastAPI
    lifespan) runs :meth:`LifecycleManager.reconcile` once before any
    request is served; a :class:`PodmanError` during reconcile is logged
    but does not prevent the server from starting.
    """
    if manager is None:
        path = resolve_profiles_path()
        common, profiles = load_config(path)
        manager = LifecycleManager(
            common, profiles, Podman(), InstanceTracker(), config_path=path
        )
    tracker = manager._tracker
    # The model server listens on common.host_port, not a fixed port -- both
    # the health poller and the metrics collector must target that port
    # instead of their class defaults, otherwise a non-default host_port
    # (e.g. because 8080 is already taken by something else on the host)
    # makes every probe miss the real server and demotes it to "degraded"
    # even though it is healthy.
    model_base_url = f"http://127.0.0.1:{manager._common.host_port}"
    collector = MetricsCollector(base_url=model_base_url)
    # Event stream and health polling run in their own daemon threads and
    # are started with the app, so the API server sees crashes and
    # readiness without any HTTP request.  Both are torn down again on
    # shutdown below.
    watcher = ContainerEventWatcher(
        manager._podman,
        manager._common.container_name,
        manager.handle_container_event,
    )
    poller = HealthPoller(tracker, base_url=model_base_url)
    # The host can suspend on its own (known bug with spontaneous
    # suspends), so the API also watches the logind PrepareForSleep
    # signal: on resume the real container state is re-collected via
    # reconcile before anything is reported again.
    power_watcher = PowerStateWatcher(tracker, manager.reconcile)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            manager.reconcile()
        except PodmanError as err:
            logger.warning("startup reconcile failed: %s", err)
        watcher.start()
        poller.start()
        power_watcher.start()
        try:
            yield
        finally:
            power_watcher.stop()
            watcher.stop()
            poller.stop()

    app = FastAPI(title="llamactl", lifespan=lifespan)
    app.state.manager = manager

    @app.get("/healthz")
    def healthz() -> dict:
        """Liveness probe."""
        return {"status": "ok", "version": __version__}

    @app.get("/profiles")
    def profiles() -> dict:
        """List all profiles with their effective sizing + tuning config."""
        return _profiles_response(manager)

    @app.get("/profiles/{name}")
    def profile_detail(name: str) -> dict:
        """Return one profile's config plus GGUF geometry + VRAM estimate."""
        if name not in manager.list_profiles():
            raise HTTPException(404, detail=f"unknown profile {name!r}")
        return _profile_detail(manager, name, with_geometry=True)

    @app.get("/profiles/{name}/preflight")
    def preflight(name: str) -> dict:
        """Estimate whether a profile fits in VRAM *without* loading it.

        Combines the profile's GGUF geometry + file size into a VRAM
        estimate and compares it against total GPU VRAM. When a measured
        peak from a previous run exists it is preferred over the estimate
        (``basis``). ``fits`` is ``None`` when total VRAM is unknown.
        """
        if name not in manager.list_profiles():
            raise HTTPException(404, detail=f"unknown profile {name!r}")
        return _preflight(manager, collector, name)

    @app.post("/profiles/{name}/trial")
    def trial(name: str, payload: dict = Body(default={})) -> dict:
        """Actually load a profile once and report the outcome.

        Starts the container, waits up to ``timeout`` seconds for the
        model server to become healthy or the container to die, records
        the peak VRAM and a log tail, then stops it again unless
        ``keep`` is true and it became ready. 404 for unknown profiles,
        409 when an instance is already active (stop it first).
        """
        if name not in manager.list_profiles():
            raise HTTPException(404, detail=f"unknown profile {name!r}")
        timeout = int(payload.get("timeout", 60))
        keep = bool(payload.get("keep", False))
        try:
            return _run_trial(manager, collector, name, timeout, keep)
        except AlreadyRunningError as err:
            raise HTTPException(409, detail=str(err)) from err
        except PodmanError as err:
            raise HTTPException(500, detail=err.stderr) from err

    @app.get("/logs")
    def logs(tail: int = 50) -> dict:
        """Return the last ``tail`` lines of the model container's log."""
        try:
            lines = manager._podman.logs(manager._common.container_name, tail=tail)
        except PodmanError as err:
            raise HTTPException(500, detail=err.stderr) from err
        return {"lines": lines}

    @app.post("/profiles", status_code=201)
    def create_profile(payload: dict = Body(...)) -> dict:
        """Create and persist a new profile.

        Body: the profile fields plus a ``name`` (e.g.
        ``{"name": "x", "model": "m.gguf", "ctx_size": 4096, ...}``).
        Returns the created profile's effective detail. 409 if the name
        exists, 422 on invalid/missing fields.
        """
        table = dict(payload)
        name = table.pop("name", None)
        if not isinstance(name, str) or not name:
            raise HTTPException(422, detail="missing 'name' field")
        try:
            manager.create_profile(name, table)
        except ProfileExistsError as err:
            raise HTTPException(409, detail=str(err)) from err
        except ValueError as err:
            raise HTTPException(422, detail=str(err)) from err
        except OSError as err:
            raise HTTPException(500, detail=f"could not persist profile: {err}") from err
        return _profile_detail(manager, name)

    @app.put("/profiles/{name}")
    def update_profile(name: str, payload: dict = Body(...)) -> dict:
        """Replace an existing profile's fields (in memory and on disk).

        404 if unknown, 409 if it is the running instance's profile, 422
        on invalid fields.
        """
        table = dict(payload)
        table.pop("name", None)  # name comes from the path; ignore any in body
        try:
            manager.update_profile(name, table)
        except UnknownProfileError as err:
            raise HTTPException(404, detail=str(err)) from err
        except ProfileInUseError as err:
            raise HTTPException(409, detail=str(err)) from err
        except ValueError as err:
            raise HTTPException(422, detail=str(err)) from err
        except OSError as err:
            raise HTTPException(500, detail=f"could not persist profile: {err}") from err
        return _profile_detail(manager, name)

    @app.delete("/profiles/{name}")
    def delete_profile(name: str) -> dict:
        """Delete a profile. 404 if unknown, 409 if it is in use."""
        try:
            manager.delete_profile(name)
        except UnknownProfileError as err:
            raise HTTPException(404, detail=str(err)) from err
        except ProfileInUseError as err:
            raise HTTPException(409, detail=str(err)) from err
        except OSError as err:
            raise HTTPException(500, detail=f"could not persist profile: {err}") from err
        return {"deleted": name}

    @app.post("/start")
    def start(payload: dict = Body(...)) -> dict:
        """Start the instance for the requested profile."""
        profile_name = payload.get("profile")
        if not isinstance(profile_name, str):
            raise HTTPException(422, detail="missing 'profile' field")
        try:
            manager.start(profile_name)
        except UnknownProfileError as err:
            raise HTTPException(404, detail=str(err)) from err
        except AlreadyRunningError as err:
            raise HTTPException(409, detail=str(err)) from err
        except PodmanError as err:
            raise HTTPException(500, detail=err.stderr) from err
        return _status_body(manager)

    @app.post("/stop")
    def stop() -> dict:
        """Stop the running instance, if any."""
        try:
            manager.stop()
        except PodmanError as err:
            raise HTTPException(500, detail=err.stderr) from err
        return _status_body(manager)

    @app.get("/status")
    def status() -> dict:
        """Return the current instance state (read-only, no podman calls).

        Includes the active profile's ``model`` file alongside the profile
        name (see :func:`_status_body`).
        """
        return _status_body(manager)

    @app.get("/metrics")
    def metrics() -> dict:
        """Collect VRAM and model-server metrics for the current state.

        In state ``ready`` the model server is queried for throughput and
        the loaded model; in every other state only the VRAM baseline is
        reported and the model server is not contacted at all.
        """
        status = tracker.snapshot()
        state = (
            status.state if isinstance(status.state, InstanceState)
            else InstanceState(status.state)
        )
        report = collector.collect(state, status.profile)
        # While ready, feed the observed VRAM into the profile's peak so a
        # profile that has run reports a measured requirement afterwards.
        if state is InstanceState.READY and status.profile:
            used = (report.get("vram") or {}).get("used_bytes")
            manager.record_vram_peak(status.profile, used)
        return report

    @app.get("/config")
    def config() -> dict:
        """Static server addressing so the client keeps one source of truth.

        Notably the inference address: llamactl (this control API) is on a
        different port than the model server that actually serves ``/v1``,
        and the client should not have to configure that separately.
        """
        host_port = manager._common.host_port
        return {
            "version": __version__,
            "control_container": manager._common.container_name,
            "models_dir": manager._common.models_dir,
            "inference_host_port": host_port,
            "inference_url": f"http://127.0.0.1:{host_port}",
        }

    @app.post("/suspend")
    def suspend_host() -> dict:
        """Suspend the host (Suspend-to-RAM) without stopping the instance.

        The container is frozen with the machine, not stopped: no stop
        or reload logic is invoked and the tracker state is left as-is.
        """
        try:
            suspend()
        except PowerError as err:
            raise HTTPException(500, detail=str(err)) from err
        result = _status_body(manager)
        result["suspend_requested"] = True
        return result

    @app.post("/reload")
    def reload(payload: dict = Body(...)) -> dict:
        """Switch the running instance to another profile."""
        profile_name = payload.get("profile")
        if not isinstance(profile_name, str):
            raise HTTPException(422, detail="missing 'profile' field")
        try:
            manager.reload(profile_name)
        except UnknownProfileError as err:
            raise HTTPException(404, detail=str(err)) from err
        except AlreadyRunningError as err:
            raise HTTPException(409, detail=str(err)) from err
        except PodmanError as err:
            raise HTTPException(500, detail=err.stderr) from err
        return _status_body(manager)

    return app
