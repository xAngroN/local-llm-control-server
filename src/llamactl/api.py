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
from llamactl.config import load_config
from llamactl.lifecycle import (
    AlreadyRunningError,
    LifecycleManager,
    UnknownProfileError,
)
from llamactl.podman import Podman, PodmanError
from llamactl.state import InstanceTracker

logger = logging.getLogger(__name__)


def _profiles_response(manager: LifecycleManager) -> dict:
    """Build the /profiles body: name -> ctx_size / parallel / slot ctx."""
    result: dict = {}
    for name in manager.list_profiles():
        profile = manager._profiles[name]
        result[name] = {
            "ctx_size": profile.ctx_size,
            "parallel": profile.parallel,
            "slot_ctx_size": profile.slot_ctx_size,
        }
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
        common, profiles = load_config()
        manager = LifecycleManager(
            common, profiles, Podman(), InstanceTracker()
        )
    tracker = manager._tracker

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            manager.reconcile()
        except PodmanError as err:
            logger.warning("startup reconcile failed: %s", err)
        yield

    app = FastAPI(title="llamactl", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict:
        """Liveness probe."""
        return {"status": "ok", "version": __version__}

    @app.get("/profiles")
    def profiles() -> dict:
        """List all configured profiles with ctx_size, parallel, slot ctx."""
        return _profiles_response(manager)

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
        return tracker.to_dict()

    @app.post("/stop")
    def stop() -> dict:
        """Stop the running instance, if any."""
        try:
            manager.stop()
        except PodmanError as err:
            raise HTTPException(500, detail=err.stderr) from err
        return tracker.to_dict()

    @app.get("/status")
    def status() -> dict:
        """Return the current instance state (read-only, no podman calls)."""
        return tracker.to_dict()

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
        return tracker.to_dict()

    return app
