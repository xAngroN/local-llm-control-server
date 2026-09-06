"""Thread-safe state tracking for a single model instance.

The :class:`InstanceTracker` owns one :class:`InstanceStatus` guarded by a
``threading.Lock``. It is designed to be accessed from multiple threads
(podman event stream, health polling, HTTP requests). This module only
provides the data structure; no logic that derives states lives here.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class InstanceState(str, Enum):
    """Lifecycle states of a single model instance."""

    STOPPED = "stopped"
    STARTING = "starting"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"
    CRASHED = "crashed"
    SUSPENDED = "suspended"


@dataclass
class InstanceStatus:
    """Snapshot of one instance's state at some point in time."""

    state: InstanceState = InstanceState.STOPPED
    profile: str | None = None
    container_id: str | None = None
    since: datetime = field(default_factory=datetime.now)
    exit_code: int | None = None
    log_tail: list[str] = field(default_factory=list)
    message: str | None = None

    def copy(self) -> "InstanceStatus":
        """Return an independent copy (log_tail is copied as a list)."""
        return InstanceStatus(
            state=self.state,
            profile=self.profile,
            container_id=self.container_id,
            since=self.since,
            exit_code=self.exit_code,
            log_tail=list(self.log_tail),
            message=self.message,
        )


class InstanceTracker:
    """Thread-safe holder of a single :class:`InstanceStatus`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = InstanceStatus()

    def snapshot(self) -> InstanceStatus:
        """Return a copy of the current status (safe to keep and mutate)."""
        with self._lock:
            return self._status.copy()

    def transition(self, state: InstanceState, **fields: object) -> None:
        """Move the instance to ``state``, refreshing ``since``.

        Any additional keyword fields are applied to the status. When
        transitioning to :attr:`InstanceState.STOPPED`, ``profile``,
        ``container_id``, ``exit_code`` and ``log_tail`` are reset to their
        empty values.
        """
        with self._lock:
            self._status.state = state
            self._status.since = datetime.now()
            if state is InstanceState.STOPPED:
                self._status.profile = None
                self._status.container_id = None
                self._status.exit_code = None
                self._status.log_tail = []
            else:
                for name, value in fields.items():
                    setattr(self._status, name, value)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of the current status."""
        with self._lock:
            status = self._status
            result: dict = {
                "state": status.state.value,
                "profile": status.profile,
                "container_id": status.container_id,
                "since": status.since.isoformat(),
                "exit_code": status.exit_code,
                "log_tail": list(status.log_tail),
                "message": status.message,
            }
        return result
