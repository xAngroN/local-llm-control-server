"""Lifecycle management for the single model instance.

The host has one GPU with limited VRAM, so at most one model instance
may run at any time. :meth:`LifecycleManager.start` therefore checks
both the tracker state and the real podman container state before
starting, and the whole flow is guarded by an instance lock so two
concurrent starts cannot both proceed.

Only ``start`` lives here; stop and reload are separate concerns and
``ready`` is only ever set by the later health check, not by this
module.
"""

from __future__ import annotations

import threading

from .config import CommonConfig, Profile, render_podman_args
from .podman import Podman, PodmanError
from .state import InstanceState, InstanceStatus, InstanceTracker

#: States that mean "an instance is currently active" (not stopped,
#: not crashed). Anything else may be started over.
_INACTIVE_STATES = (InstanceState.STOPPED, InstanceState.CRASHED)


class UnknownProfileError(ValueError):
    """Raised when a start is requested for a profile that is not configured."""


class AlreadyRunningError(RuntimeError):
    """Raised when an instance is already active or its container is running."""


class LifecycleManager:
    """Owns the start flow for the single model instance."""

    def __init__(
        self,
        common: CommonConfig,
        profiles: dict[str, Profile],
        podman: Podman,
        tracker: InstanceTracker,
    ) -> None:
        self._common = common
        self._profiles = profiles
        self._podman = podman
        self._tracker = tracker
        self._lock = threading.Lock()

    def list_profiles(self) -> list[str]:
        """Return the names of all configured profiles."""
        return list(self._profiles)

    def start(self, profile_name: str) -> InstanceStatus:
        """Start the model instance for ``profile_name``.

        Raises :class:`UnknownProfileError` for unconfigured names and
        :class:`AlreadyRunningError` when an instance is already active
        (tracker) or the container is already running (podman). On
        success the tracker ends in ``loading`` with the profile name
        and container id set; on :class:`PodmanError` it ends in
        ``stopped`` with the error message and the exception is
        re-raised.
        """
        with self._lock:
            if profile_name not in self._profiles:
                raise UnknownProfileError(f"unknown profile: {profile_name!r}")

            status = self._tracker.snapshot()
            if status.state not in _INACTIVE_STATES:
                raise AlreadyRunningError(
                    f"instance already active (state={status.state.value})"
                )
            if self._podman.is_running(self._common.container_name):
                raise AlreadyRunningError(
                    f"container {self._common.container_name!r} is already running"
                )

            self._tracker.transition(InstanceState.STARTING)
            profile = self._profiles[profile_name]
            args = render_podman_args(profile, self._common)
            try:
                container_id = self._podman.start_container(args)
            except PodmanError as err:
                self._tracker.transition(InstanceState.STOPPED)
                # InstanceTracker.transition(STOPPED) resets instance fields
                # and ignores extra kwargs, so the failure message is stored
                # directly (message is not part of the reset set).
                with self._tracker._lock:
                    self._tracker._status.message = str(err)
                raise
            self._tracker.transition(
                InstanceState.LOADING,
                profile=profile_name,
                container_id=container_id,
            )
            return self._tracker.snapshot()
