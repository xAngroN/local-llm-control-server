"""Lifecycle management for the single model instance.

The host has one GPU with limited VRAM, so at most one model instance
may run at any time. :meth:`LifecycleManager.start` therefore checks
both the tracker state and the real podman container state before
starting, and the whole flow is guarded by an instance lock so two
concurrent starts cannot both proceed.

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


PROFILE_LABEL = "llamactl.profile"


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
        # Marks the most recent shutdown as self-initiated.  Crash
        # detection later reads this flag so a deliberate stop is not
        # reported as a crash; the next :meth:`start` clears it.
        self._expected_stop = False

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
            return self._start_locked(profile_name)

    def _start_locked(self, profile_name: str) -> InstanceStatus:
        """Start flow; the caller must already hold :attr:`_lock`."""
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

        # A new start supersedes any previous deliberate stop; clear the
        # expected-stop flag so this instance's shutdown is judged afresh.
        self._expected_stop = False
        self._tracker.transition(InstanceState.STARTING)
        profile = self._profiles[profile_name]
        args = render_podman_args(profile, self._common)
        # Tag the container with its profile so a restarted manager can
        # recover the active profile name via ``reconcile``.  The flag
        # goes into the podman-run flag section, before the image and
        # server arguments, exactly where real ``podman run`` expects it.
        image_index = args.index(profile.image)
        args[image_index:image_index] = [
            "--label",
            f"{PROFILE_LABEL}={profile_name}",
        ]
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

    def stop(self) -> InstanceStatus:
        """Stop the current model instance, if any.

        A stop with nothing active (tracker ``stopped`` and no running
        container) is not an error and simply returns the ``stopped``
        state.  Otherwise the container is stopped (``podman stop`` also
        removes it because it was started with ``--rm``) and the tracker
        ends in ``stopped`` with ``profile`` reset to ``None``.

        Before touching the container, ``self._expected_stop`` is set so
        crash detection later recognises this shutdown as intentional.
        The flag is intentionally left in place until the next
        :meth:`start` resets it.
        """
        with self._lock:
            return self._stop_locked()

    def _stop_locked(self) -> InstanceStatus:
        """Stop flow; the caller must already hold :attr:`_lock`."""
        status = self._tracker.snapshot()
        if status.state is InstanceState.STOPPED and not self._podman.is_running(
            self._common.container_name
        ):
            # Idle stop: nothing to do, no error.
            self._tracker.transition(InstanceState.STOPPED)
            return self._tracker.snapshot()

        self._expected_stop = True
        self._podman.stop_container(self._common.container_name)
        self._tracker.transition(InstanceState.STOPPED)
        return self._tracker.snapshot()

    def reconcile(self) -> InstanceStatus:
        """Adopt the real podman state at API-server startup.

        Compares the actual container state against the manager's own
        (always empty) startup state and reports it, without starting or
        stopping anything:

        * no container -> ``stopped``;
        * exited container -> ``stopped`` with the exit code taken over
          into :attr:`InstanceStatus.exit_code`;
        * running container -> ``loading`` with the container ID and the
          profile name read from the ``llamactl.profile`` label set by
          :meth:`_start_locked`. A running container whose profile cannot
          be derived is still adopted, with ``profile = None`` and an
          explanatory :attr:`InstanceStatus.message`.
        """
        with self._lock:
            return self._reconcile_locked()

    def _reconcile_locked(self) -> InstanceStatus:
        """Reconcile flow; the caller must already hold :attr:`_lock`."""
        info = self._podman.inspect(self._common.container_name)
        if info is None:
            self._tracker.transition(InstanceState.STOPPED)
            return self._tracker.snapshot()

        state = info.get("State") or {}
        container_id = info.get("Id") or info.get("ID")
        if state.get("Running"):
            profile, message = self._profile_from_container(info)
            self._tracker.transition(
                InstanceState.LOADING,
                profile=profile,
                container_id=container_id,
                message=message,
            )
        else:
            exit_code = state.get("ExitCode")
            self._tracker.transition(InstanceState.STOPPED)
            # InstanceTracker.transition(STOPPED) resets instance fields
            # and ignores extra kwargs, so the exit code is stored directly.
            with self._tracker._lock:
                self._tracker._status.exit_code = exit_code
        return self._tracker.snapshot()

    def _profile_from_container(self, info: dict) -> tuple[str | None, str | None]:
        """Recover the profile name of a running container.

        Checks the ``llamactl.profile`` label first, then the stored
        container arguments (as the fake podman records them). Returns
        ``(profile, message)``; when the profile cannot be derived the
        message explains that an orphaned container was adopted.
        """
        labels = info.get("Labels")
        if isinstance(labels, dict):
            candidate = labels.get(PROFILE_LABEL)
            if candidate in self._profiles:
                return candidate, None
        for arg in info.get("Args") or []:
            if isinstance(arg, str):
                if arg.startswith(f"{PROFILE_LABEL}="):
                    candidate = arg.partition("=")[2]
                elif arg.startswith("--label") and f"{PROFILE_LABEL}=" in arg:
                    candidate = arg.partition(f"{PROFILE_LABEL}=")[2]
                else:
                    continue
                if candidate in self._profiles:
                    return candidate, None
        return (
            None,
            f"adopted orphaned running container "
            f"{self._common.container_name!r}: profile "
            f"({PROFILE_LABEL}) could not be derived from container",
        )

    def reload(self, profile_name: str) -> InstanceStatus:
        """Switch the running instance to ``profile_name`` atomically.

        The whole stop-then-start flow runs under the instance lock so no
        other call can slip in between the two halves. An unknown
        ``profile_name`` raises :class:`UnknownProfileError` before anything
        is stopped, so a typo can never kill a running instance. If the
        start of the new profile fails the tracker ends in ``stopped``
        with the failure message (the old profile is never reported as
        active) and the exception is re-raised.
        """
        with self._lock:
            if profile_name not in self._profiles:
                raise UnknownProfileError(f"unknown profile: {profile_name!r}")
            self._stop_locked()
            return self._start_locked(profile_name)
