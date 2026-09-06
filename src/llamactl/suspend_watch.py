"""Host power-state watcher: observes suspend/resume transitions.

A suspend can be triggered from outside this API (the machine has a known,
separately tracked bug with spontaneous suspends), so instead of only
watching our own ``/suspend`` action we listen for the host's logind
``PrepareForSleep`` signal on the session/user D-Bus.

The concrete command (``busctl --user monitor`` / ``dbus-monitor``) and the
interpretation of one output line are isolated in the overridable methods
:meth:`PowerStateWatcher._monitor_command` and
:meth:`PowerStateWatcher.parse_line` so both can be exercised without a
running D-Bus.

The watch loop runs in a daemon thread following the pattern of
:meth:`ContainerEventWatcher.start`/``stop`` in :mod:`llamactl.events`.
On *suspend start* the current state is remembered and the tracker is set
to ``suspended`` so the frozen container is not reported as crashed.  On
*resume* the ``on_resume`` callback is invoked exactly once; in the API
server that callback is wired to :meth:`LifecycleManager.reconcile`, so
the real container state is re-collected before anything is reported
again.

If the monitor stream breaks (missing bus, process death) the thread
restarts it after a short delay until :meth:`PowerStateWatcher.stop` is
called.  Exceptions in ``parse_line`` or the state transition are logged
and never kill the watcher thread.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from typing import Any

from .state import InstanceState, InstanceTracker

logger = logging.getLogger(__name__)

#: The logind interface whose ``PrepareForSleep`` signal marks a host
#: suspend/resume transition (``boolean`` true = about to suspend,
#: false = just resumed).
LOGIND_INTERFACE = "org.freedesktop.login1.Manager"

#: Delay before restarting a broken monitor stream.
RESTART_DELAY_SECONDS = 2.0
#: How long ``stop()`` waits for the thread to exit after the monitor
#: subprocess is terminated.
STOP_JOIN_TIMEOUT_SECONDS = 10.0
#: How long ``stop()`` waits for the subprocess to exit before killing it.
PROCESS_EXIT_TIMEOUT_SECONDS = 5.0


def _match_fields(line: str) -> tuple[str, str, str, str]:
    """Split one logind monitor line into (sender, interface, member, arg).

    The field position depends on the tool:

    * ``busctl --user monitor`` prints
      ``<date> <time> <sender> <interface> <member> <arg...>`` (some
      builds merge date and time into one token); the interface is
      recognised by its dotted shape, so the leading timestamp tokens
      are skipped automatically;
    * ``dbus-monitor`` prints
      ``signal <path> <interface> <member> <arg...>  (sender=<sender>)``.

    Returns empty strings for the pieces that are absent.
    """
    tokens = line.split()
    if tokens[:1] == ["signal"]:
        interface = tokens[2] if len(tokens) > 2 else ""
        member = tokens[3] if len(tokens) > 3 else ""
        arg = tokens[4] if len(tokens) > 4 else ""
        sender = line.rsplit("sender=", 1)[-1].lstrip(")(").strip()
        return (sender, interface, member, arg)
    # busctl style: find the interface token (dotted, not a timestamp,
    # not a path) and take the member and first argument after it.
    for index, token in enumerate(tokens):
        if (
            token.count(".") >= 2
            and not token[:1].isdigit()
            and not token.startswith("/")
        ):
            interface = token
            member = tokens[index + 1] if index + 1 < len(tokens) else ""
            arg = tokens[index + 2] if index + 2 < len(tokens) else ""
            return ("", interface, member, arg)
    return ("", "", "", "")


class PowerStateWatcher:
    """Watch the host for suspend/resume transitions and update the tracker.

    Parameters
    ----------
    tracker:
        The shared :class:`InstanceTracker` to move to ``suspended`` on a
        suspend and to leave alone on a resume (the resume callback
        re-collects the real state via reconcile).
    on_resume:
        Callback invoked exactly once per resume; in the API server it is
        wired to :meth:`LifecycleManager.reconcile`.
    """

    def __init__(
        self,
        tracker: InstanceTracker,
        on_resume: Callable[[], None],
    ) -> None:
        self._tracker = tracker
        self._on_resume = on_resume
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # The monitor subprocess currently owned by the thread, so
        # stop() can terminate it and wake the blocked readline.
        self._proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()
        # State remembered when a suspend began; kept so the watcher can
        # be inspected and so a resume is only acted on once per cycle.
        self._pre_suspend_state: InstanceState | None = None
        self._state_lock = threading.Lock()
        # Counters used by tests to assert "exactly once" semantics.
        self._resume_calls = 0

    # ------------------------------------------------------------------
    # Overridable seams (testable without a running D-Bus)
    # ------------------------------------------------------------------

    def _monitor_command(self) -> list[str]:
        """Build the command for the logind signal monitor.

        ``busctl --user monitor`` is preferred (it emits one line per
        message with sender, interface and member); ``dbus-monitor`` is
        the fallback for systems without ``busctl``.  The returned list
        is never run through a shell.
        """
        if shutil.which("busctl") is not None:
            return ["busctl", "--user", "monitor"]
        return [
            "dbus-monitor",
            "--session",
            f"type='signal',interface='{LOGIND_INTERFACE}',"
            "member='PrepareForSleep'",
        ]

    def parse_line(self, line: str) -> bool | None:
        """Interpret one monitor output line as a power-state transition.

        Returns ``True`` when the host is about to suspend, ``False`` when
        it just resumed, and ``None`` for any line that is irrelevant.
        """
        _sender, interface, member, arg = _match_fields(line)
        if interface != LOGIND_INTERFACE or member != "PrepareForSleep":
            return None
        if arg == "true":
            return True
        if arg == "false":
            return False
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the watcher daemon thread; returns immediately."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        thread = threading.Thread(
            target=self._run, name="llamactl-power-state-watcher", daemon=True
        )
        self._thread = thread
        thread.start()

    def stop(self) -> None:
        """Stop the watcher: end the monitor subprocess and the thread."""
        self._stop_event.set()
        with self._proc_lock:
            proc, self._proc = self._proc, None
        if proc is not None:
            self._terminate(proc)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=STOP_JOIN_TIMEOUT_SECONDS)
        self._thread = None

    @property
    def running(self) -> bool:
        """Return whether the watcher thread is currently active."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_stream(self) -> subprocess.Popen:
        """Start the monitor subprocess and register it for termination."""
        cmd = self._monitor_command()
        # stdout is unbuffered in the child so signal lines reach the
        # reader immediately even under python-style monitors.
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )
        with self._proc_lock:
            self._proc = proc
        return proc

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        try:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=PROCESS_EXIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass

    def _handle_suspend_start(self) -> None:
        """Remember the current state and move the tracker to suspended."""
        with self._state_lock:
            self._pre_suspend_state = self._tracker.snapshot().state
        self._tracker.transition(InstanceState.SUSPENDED)

    def _handle_resume(self) -> None:
        """Invoke the resume callback exactly once per cycle."""
        with self._state_lock:
            self._resume_calls += 1
            self._pre_suspend_state = None
        try:
            self._on_resume()
        except Exception:  # noqa: BLE001 - never kill the watcher thread
            logger.exception("on_resume callback failed")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            proc: subprocess.Popen | None = None
            try:
                proc = self._open_stream()
                assert proc.stdout is not None
                for line in proc.stdout:
                    if self._stop_event.is_set():
                        break
                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    try:
                        transition = self.parse_line(line)
                    except Exception:  # noqa: BLE001
                        logger.exception("failed to parse monitor line: %r", line)
                        continue
                    if transition is True:
                        try:
                            self._handle_suspend_start()
                        except Exception:  # noqa: BLE001
                            logger.exception("failed to mark suspend")
                    elif transition is False:
                        self._handle_resume()
            except Exception:  # noqa: BLE001
                logger.exception("power-state monitor stream broke")
            finally:
                if proc is not None:
                    self._terminate(proc)
                    with self._proc_lock:
                        if self._proc is proc:
                            self._proc = None
            if self._stop_event.is_set():
                break
            # Wait out the restart delay in one slice; stop() wakes it.
            self._stop_event.wait(RESTART_DELAY_SECONDS)


__all__ = [
    "LOGIND_INTERFACE",
    "PowerStateWatcher",
    "RESTART_DELAY_SECONDS",
    "STOP_JOIN_TIMEOUT_SECONDS",
]
