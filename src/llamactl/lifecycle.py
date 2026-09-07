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
from pathlib import Path

from .config import (
    CommonConfig,
    Profile,
    append_profile,
    build_profile,
    overwrite_profile,
    remove_profile,
    render_podman_args,
    resolve_profiles_path,
)
from .podman import Podman, PodmanError
from .state import InstanceState, InstanceStatus, InstanceTracker

#: States that mean "an instance is currently active" (not stopped,
#: not crashed). Anything else may be started over.
_INACTIVE_STATES = (InstanceState.STOPPED, InstanceState.CRASHED)

#: Event actions that mean "the container ended" (deliberate or not).
_ENDING_EVENT_ACTIONS = ("die", "stop")

#: States in which the container is considered gone, so an ending event
#: must not be re-reported as a crash.
_GONE_STATES = (InstanceState.STOPPED, InstanceState.SUSPENDED)


PROFILE_LABEL = "llamactl.profile"


class UnknownProfileError(ValueError):
    """Raised when a start is requested for a profile that is not configured."""


class AlreadyRunningError(RuntimeError):
    """Raised when an instance is already active or its container is running."""


class ProfileExistsError(ValueError):
    """Raised when creating a profile whose name is already configured."""


class ProfileInUseError(RuntimeError):
    """Raised when editing/deleting the profile of the running instance."""


class LifecycleManager:
    """Owns the start flow for the single model instance."""

    def __init__(
        self,
        common: CommonConfig,
        profiles: dict[str, Profile],
        podman: Podman,
        tracker: InstanceTracker,
        config_path: Path | None = None,
    ) -> None:
        self._common = common
        self._profiles = profiles
        self._podman = podman
        self._tracker = tracker
        # File that profile CRUD persists to. ``None`` -> resolved lazily
        # via the same LLAMACTL_PROFILES/default logic as load_config, so a
        # manager built from the default config writes back to that file.
        self._config_path = config_path
        self._lock = threading.Lock()
        # Marks the most recent shutdown as self-initiated.  Crash
        # detection later reads this flag so a deliberate stop is not
        # reported as a crash; the next :meth:`start` clears it.
        self._expected_stop = False

    def list_profiles(self) -> list[str]:
        """Return the names of all configured profiles."""
        return list(self._profiles)

    # ------------------------------------------------------------------
    # Profile CRUD (validated in-memory update + persisted to the file)
    # ------------------------------------------------------------------

    def _profiles_path(self) -> Path:
        """Resolve the file profile CRUD reads/writes."""
        return resolve_profiles_path(self._config_path)

    def _active_profile(self) -> str | None:
        """Name of the profile of a non-stopped instance, else ``None``."""
        status = self._tracker.snapshot()
        if status.state in _INACTIVE_STATES:
            return None
        return status.profile

    def create_profile(self, name: str, table: dict) -> Profile:
        """Validate, register and persist a new profile.

        ``table`` is the raw key/value mapping (without ``name``). Raises
        :class:`ProfileExistsError` if the name is taken and
        :class:`ValueError` on invalid fields; nothing is persisted in
        either case. On success the profile is added to the in-memory set
        and appended to the config file (existing comments preserved).
        """
        with self._lock:
            if name in self._profiles:
                raise ProfileExistsError(f"profile {name!r} already exists")
            profile = build_profile(name, table, self._common.image)
            append_profile(self._profiles_path(), name, table)
            self._profiles[name] = profile
            return profile

    def update_profile(self, name: str, table: dict) -> Profile:
        """Replace an existing profile's fields, in memory and on disk.

        Raises :class:`UnknownProfileError` when the profile does not
        exist, :class:`ProfileInUseError` when it is the profile of the
        currently running instance (stop it first), and :class:`ValueError`
        on invalid fields. Persistence rewrites only this profile's table.
        """
        with self._lock:
            if name not in self._profiles:
                raise UnknownProfileError(f"unknown profile: {name!r}")
            if self._active_profile() == name:
                raise ProfileInUseError(
                    f"profile {name!r} is in use by the running instance"
                )
            profile = build_profile(name, table, self._common.image)
            overwrite_profile(self._profiles_path(), name, table)
            self._profiles[name] = profile
            return profile

    def delete_profile(self, name: str) -> None:
        """Remove a profile from memory and the config file.

        Raises :class:`UnknownProfileError` when absent and
        :class:`ProfileInUseError` when it belongs to the running instance.
        """
        with self._lock:
            if name not in self._profiles:
                raise UnknownProfileError(f"unknown profile: {name!r}")
            if self._active_profile() == name:
                raise ProfileInUseError(
                    f"profile {name!r} is in use by the running instance"
                )
            remove_profile(self._profiles_path(), name)
            del self._profiles[name]

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

        Reads the ``llamactl.profile`` label from the inspect output, using
        the real podman convention (``Config.Labels``) first, then the
        command args for a label flag that was not materialised into a
        label. Returns ``(profile, message)``; when the profile cannot be
        derived the message explains that an orphaned container was adopted.
        """
        config = info.get("Config") or {}
        if not isinstance(config, dict):
            config = {}
        # Real ``podman inspect`` surfaces labels under ``Config.Labels``.
        for source in (config.get("Labels"), info.get("Labels")):
            if isinstance(source, dict):
                candidate = source.get(PROFILE_LABEL)
                if isinstance(candidate, str) and candidate in self._profiles:
                    return candidate, None
        # Fall back to a ``--label`` flag recorded in the command args.
        cmd = config.get("Cmd") or info.get("Args") or []
        for arg in cmd:
            if isinstance(arg, str) and arg.startswith(f"{PROFILE_LABEL}="):
                candidate = arg.partition("=")[2]
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


    def handle_container_event(self, event: dict) -> None:
        """Handle one podman container event from the event watcher.

        This is the callback :class:`~llamactl.events.ContainerEventWatcher`
        invokes on its own thread, so all state changes go through the
        thread-safe :class:`~llamactl.state.InstanceTracker`.

        For a ``die``/``stop`` event on the managed container the
        ``self._expected_stop`` flag (set by :meth:`stop` and the stop
        half of :meth:`reload`) distinguishes a self-initiated shutdown
        from an unexpected death:

        * flag set  -> the tracker moves to ``stopped`` and the flag is
          cleared; a deliberate shutdown is never a crash;
        * flag unset and the tracker not already ``stopped`` or
          ``suspended`` -> the tracker moves to ``crashed`` with the exit
          code taken from the event (``exitCode`` field, falling back to
          :meth:`Podman.exit_code`) and the last 50 log lines from
          :meth:`Podman.logs` stored in ``log_tail``.

        Any other event (e.g. ``start``) is ignored.  No automatic
        restart is attempted here -- crashes are only detected and
        reported.
        """
        action = event.get("Action")
        if action not in _ENDING_EVENT_ACTIONS:
            return
        name = event.get("name")
        if name != self._common.container_name:
            return
        if self._expected_stop:
            # Self-initiated shutdown (stop or the stop half of a
            # reload): report stopped and clear the flag so the next
            # unannounced death is judged on its own.
            self._expected_stop = False
            self._tracker.transition(InstanceState.STOPPED)
            return
        current = self._tracker.snapshot().state
        if current in _GONE_STATES:
            # Already reported gone; a late or duplicate ending event
            # must not turn a stopped/suspended instance into crashed.
            return
        exit_code = event.get("exitCode")
        if exit_code is None:
            try:
                exit_code = self._podman.exit_code(name)
            except PodmanError:
                exit_code = None
        if exit_code is not None:
            exit_code = int(exit_code)
        log_tail = self._podman.logs(name, tail=50)
        self._tracker.transition(
            InstanceState.CRASHED,
            exit_code=exit_code,
            log_tail=log_tail,
        )
