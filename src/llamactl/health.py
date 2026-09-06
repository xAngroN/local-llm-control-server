"""Health polling for the model server.

The model server (llama.cpp) exposes a ``/health`` endpoint. This module
polls it on a fixed interval and derives a *server liveness* state for the
tracker:

* a ``200`` answer while the instance is ``loading``/``degraded``/``ready``
  promotes it to ``ready``;
* ``failure_threshold`` consecutive failed probes while the instance is
  ``ready``/``loading`` demote it to ``degraded``.

The distinction between ``degraded`` and ``crashed`` is the point of this
module: ``degraded`` means the process is alive but not answering, while
``crashed`` means the process is unexpectedly gone and is detected exclusively
via the podman event stream (see :mod:`llamactl.events`), never here. A live,
unresponsive process must therefore never be reported as ``crashed``.

When the tracker is ``stopped``, ``crashed`` or ``suspended`` no probe is
issued at all and nothing is set.
"""

from __future__ import annotations

import threading

import httpx

from .state import InstanceState, InstanceTracker

#: States that mean "the server should be answering": a healthy probe moves
#: them to ``ready``, a run of failures demotes them to ``degraded``.
_ACTIVE_STATES = (
    InstanceState.LOADING,
    InstanceState.READY,
    InstanceState.DEGRADED,
)

#: States from which a probe is issued at all. ``stopped``/``crashed``/
#: ``suspended`` mean there is no live process to poll, so we skip and
#: leave the state untouched.
_PROBE_STATES = _ACTIVE_STATES

#: States from which a run of failures demotes to ``degraded``. ``degraded``
#: is already demoted, so failures there only keep it degraded (the streak
#: counter is reset to avoid redundant transitions).
_DEGRADE_STATES = (
    InstanceState.LOADING,
    InstanceState.READY,
)


class HealthPoller:
    """Poll the model server's ``/health`` endpoint and update the tracker."""

    def __init__(
        self,
        tracker: InstanceTracker,
        base_url: str = "http://127.0.0.1:8080",
        interval: float = 5.0,
        failure_threshold: int = 3,
    ) -> None:
        self._tracker = tracker
        self._base_url = base_url
        self._interval = interval
        self._failure_threshold = failure_threshold
        self._consecutive_failures = 0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def running(self) -> bool:
        """Return whether the polling thread is currently active."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the background polling thread (no-op if already running)."""
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="llamactl-health-poller", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background polling thread and wait for it to exit."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._interval + 5.0)
        self._thread = None

    def _run(self) -> None:
        """Loop body: poll once every ``interval`` seconds until stopped."""
        while not self._stop_event.wait(self._interval):
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001 - never let the thread die
                # A probe must not take down the polling loop; the next
                # iteration simply retries.
                pass

    def poll_once(self) -> bool:
        """Issue a single health probe and update the tracker.

        Returns ``True`` when the server answered with HTTP 200, ``False``
        otherwise. When the tracker is ``stopped``, ``crashed``,
        ``suspended`` or ``starting`` no HTTP request is issued and nothing
        is set.
        """
        status = self._tracker.snapshot()
        state = status.state
        if state not in _PROBE_STATES:
            return False

        ok = self._probe()
        if ok:
            self._consecutive_failures = 0
            # Every state we probe is one a healthy 200 promotes to ready.
            self._tracker.transition(InstanceState.READY)
            return True

        self._consecutive_failures += 1
        if (
            state in _DEGRADE_STATES
            and self._consecutive_failures >= self._failure_threshold
        ):
            self._tracker.transition(InstanceState.DEGRADED)
            self._consecutive_failures = 0
        return False

    def _probe(self) -> bool:
        """Perform the HTTP health check.

        Isolated in its own method so tests can monkeypatch it without
        touching a real server. Returns ``True`` only on HTTP 200.
        """
        url = self._base_url.rstrip("/") + "/health"
        try:
            response = httpx.get(url, timeout=3.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200
