"""FastAPI application for the llamactl control server.

This module only translates HTTP calls into :class:`LifecycleManager`
method calls and exceptions into HTTP status codes.  All lifecycle logic
stays in :mod:`llamactl.lifecycle`; there are no podman calls and no
state logic in this module.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException

from llamactl import __version__
from llamactl.config import load_config, resolve_profiles_path, resolve_tuning
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


def _profile_detail(manager: LifecycleManager, name: str) -> dict:
    """Build the read-back body for a single profile.

    Reports the sizing fields plus the *effective* tuning configuration
    (:func:`resolve_tuning`) -- i.e. the per-profile value merged over the
    ``[common]`` default -- so callers can read exactly which llama.cpp
    knobs (``-b``/``-ub``, flash attention, ngl, continuous batching and
    the MTP/speculative-decoding settings) the profile will run with.
    """
    profile = manager._profiles[name]
    tuning = resolve_tuning(profile, manager._common)
    return {
        "model": profile.model,
        "ctx_size": profile.ctx_size,
        "parallel": profile.parallel,
        "slot_ctx_size": profile.slot_ctx_size,
        "batch_size": profile.batch_size,
        "ubatch_size": tuning["ubatch_size"],
        "n_gpu_layers": tuning["n_gpu_layers"],
        "flash_attn": tuning["flash_attn"],
        "cont_batching": tuning["cont_batching"],
        "cache_reuse": tuning["cache_reuse"],
        "spec_type": tuning["spec_type"],
        "spec_draft_n_max": tuning["spec_draft_n_max"],
        "spec_draft_n_min": tuning["spec_draft_n_min"],
    }


def _profiles_response(manager: LifecycleManager) -> dict:
    """Build the /profiles body: name -> full effective profile detail."""
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
        """Return one profile's effective sizing + tuning configuration."""
        if name not in manager.list_profiles():
            raise HTTPException(404, detail=f"unknown profile {name!r}")
        return _profile_detail(manager, name)

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
        return collector.collect(
            status.state if isinstance(status.state, InstanceState)
            else InstanceState(status.state),
            status.profile,
        )

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
