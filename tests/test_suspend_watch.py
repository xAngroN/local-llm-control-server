"""Tests for the host power-state watcher (suspend/resume handling).

Covers the two testable seams of :class:`PowerStateWatcher` (``parse_line``
and ``_monitor_command`` -- both runnable without a D-Bus) plus the
behavioural contract: a suspend moves a ready instance to ``suspended``
(not ``crashed``), a ``die`` event while ``suspended`` never produces
``crashed``, and the resume callback fires exactly once.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest

from llamactl.config import load_config
from llamactl.lifecycle import LifecycleManager
from llamactl.podman import Podman
from llamactl.state import InstanceState, InstanceTracker
from llamactl.suspend_watch import LOGIND_INTERFACE, PowerStateWatcher

from pathlib import Path

PROFILES_TOML = Path(__file__).resolve().parent.parent / "config" / "profiles.toml"

BUSCTL_SUSPEND = (
    "2026-09-06 12:00:00 org.freedesktop.login1 "
    f"{LOGIND_INTERFACE} PrepareForSleep true"
)
BUSCTL_RESUME = (
    "2026-09-06 12:00:05 org.freedesktop.login1 "
    f"{LOGIND_INTERFACE} PrepareForSleep false"
)
DBUSMON_SUSPEND = (
    f"signal /org/freedesktop/login1 {LOGIND_INTERFACE} "
    "PrepareForSleep true  (sender=org.freedesktop.login1)"
)
DBUSMON_RESUME = (
    f"signal /org/freedesktop/login1 {LOGIND_INTERFACE} "
    "PrepareForSleep false  (sender=org.freedesktop.login1)"
)
OTHER_SIGNAL = (
    "2026-09-06 12:00:00 org.freedesktop.systemd1 "
    "org.freedesktop.systemd1.Manager JobRemoved 42"
)


@pytest.fixture()
def manager(fake_podman) -> LifecycleManager:
    common, profiles = load_config(PROFILES_TOML)
    return LifecycleManager(common, profiles, Podman(), InstanceTracker())


def test_parse_line_recognises_suspend_and_resume_lines() -> None:
    watcher = PowerStateWatcher(InstanceTracker(), lambda: None)
    assert watcher.parse_line(BUSCTL_SUSPEND) is True
    assert watcher.parse_line(BUSCTL_RESUME) is False
    assert watcher.parse_line(DBUSMON_SUSPEND) is True
    assert watcher.parse_line(DBUSMON_RESUME) is False
    assert watcher.parse_line(OTHER_SIGNAL) is None
    assert watcher.parse_line("") is None
    assert watcher.parse_line("no-dots-here") is None


def test_monitor_command_returns_runnable_command_list() -> None:
    watcher = PowerStateWatcher(InstanceTracker(), lambda: None)
    cmd = watcher._monitor_command()
    assert isinstance(cmd, list) and cmd
    assert all(isinstance(part, str) for part in cmd)
    assert cmd[0] in ("busctl", "dbus-monitor")
    if cmd[0] == "busctl":
        assert "monitor" in cmd
    else:
        assert "--session" in cmd
        assert any("PrepareForSleep" in part for part in cmd)


class _ScriptedWatcher(PowerStateWatcher):
    """Watcher whose monitor stream is a script that emits prepared lines."""

    def __init__(self, lines: list[str], on_resume, tmp_path) -> None:
        super().__init__(InstanceTracker(), on_resume)
        self._lines = lines
        self._tmp_path = tmp_path
        self._done = threading.Event()
        self._tracker.transition(InstanceState.READY)

    def _monitor_command(self) -> list[str]:
        script = self._tmp_path / "fake_monitor.py"
        script.write_text(
            "import sys\n"
            f"for line in {self._lines!r}:\n"
            "    sys.stdout.write(line + '\\n')\n"
            "    sys.stdout.flush()\n"
        )
        return [sys.executable, str(script)]

    def _run(self) -> None:
        # Run the loop in-process so the test controls completion and the
        # thread does not fight stop() over subprocess lifecycle.
        for line in self._lines:
            if self._stop_event.is_set():
                break
            transition = self.parse_line(line)
            if transition is True:
                self._handle_suspend_start()
            elif transition is False:
                self._handle_resume()
        self._done.set()

    def wait_done(self, timeout: float = 5.0) -> None:
        assert self._done.wait(timeout), "watcher did not finish the scripted lines"


def test_suspend_from_ready_yields_suspended_not_crashed(
    tmp_path,
) -> None:
    calls: list[str] = []
    watcher = _ScriptedWatcher(
        [BUSCTL_SUSPEND], lambda: calls.append("resume"), tmp_path
    )
    watcher._tracker.transition(InstanceState.READY)
    watcher.start()
    try:
        watcher.wait_done()
    finally:
        watcher.stop()
    assert watcher._tracker.snapshot().state is InstanceState.SUSPENDED
    assert calls == []
    assert watcher._pre_suspend_state is InstanceState.READY


def test_resume_calls_callback_exactly_once(tmp_path) -> None:
    calls: list[str] = []

    def on_resume() -> None:
        calls.append("resume")

    watcher = _ScriptedWatcher(
        [BUSCTL_SUSPEND, BUSCTL_RESUME, DBUSMON_RESUME, OTHER_SIGNAL],
        on_resume,
        tmp_path,
    )
    watcher._tracker.transition(InstanceState.READY)
    watcher.start()
    try:
        watcher.wait_done()
    finally:
        watcher.stop()
    assert calls == ["resume"]
    assert watcher._resume_calls == 1
    assert watcher._pre_suspend_state is None


def test_resume_forgets_prior_state(tmp_path) -> None:
    watcher = _ScriptedWatcher(
        [BUSCTL_SUSPEND, DBUSMON_RESUME], lambda: None, tmp_path
    )
    watcher._tracker.transition(InstanceState.READY)
    watcher.start()
    try:
        watcher.wait_done()
    finally:
        watcher.stop()
    assert watcher._pre_suspend_state is None


def test_suspend_remembers_prior_state(tmp_path) -> None:
    watcher = _ScriptedWatcher([BUSCTL_SUSPEND], lambda: None, tmp_path)
    watcher._tracker.transition(InstanceState.READY)
    watcher.start()
    try:
        watcher.wait_done()
    finally:
        watcher.stop()
    assert watcher._pre_suspend_state is InstanceState.READY


def test_die_while_suspended_does_not_crash(manager) -> None:
    tracker = manager._tracker
    tracker.transition(InstanceState.READY)
    tracker.transition(InstanceState.SUSPENDED)
    manager.handle_container_event(
        {"Action": "die", "name": manager._common.container_name, "exitCode": 137}
    )
    assert tracker.snapshot().state is InstanceState.SUSPENDED


def test_stop_while_suspended_does_not_crash(manager) -> None:
    tracker = manager._tracker
    tracker.transition(InstanceState.SUSPENDED)
    manager.handle_container_event(
        {"Action": "stop", "name": manager._common.container_name}
    )
    assert tracker.snapshot().state is InstanceState.SUSPENDED


def test_die_while_still_ready_does_crash(manager) -> None:
    """Sanity check: without a suspend the same event IS a crash."""
    tracker = manager._tracker
    tracker.transition(InstanceState.READY)
    manager.handle_container_event(
        {"Action": "die", "name": manager._common.container_name, "exitCode": 137}
    )
    assert tracker.snapshot().state is InstanceState.CRASHED


def test_full_thread_loop_with_subprocess(tmp_path, monkeypatch) -> None:
    """The real start/stop thread loop with a real subprocess monitor."""
    import llamactl.suspend_watch as sw

    monkeypatch.setattr(sw, "RESTART_DELAY_SECONDS", 0.1)
    calls: list[str] = []
    watcher = PowerStateWatcher(InstanceTracker(), lambda: calls.append("resume"))
    watcher._tracker.transition(InstanceState.READY)

    def _cmd() -> list[str]:
        script = tmp_path / "fake_monitor.py"
        lines = [BUSCTL_SUSPEND, BUSCTL_RESUME]
        script.write_text(
            "import sys\n"
            f"for line in {lines!r}:\n"
            "    sys.stdout.write(line + '\\n')\n"
            "    sys.stdout.flush()\n"
        )
        return [sys.executable, str(script)]

    monkeypatch.setattr(watcher, "_monitor_command", _cmd)
    watcher.start()
    assert watcher.running
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if calls == ["resume"] and watcher._tracker.snapshot().state is InstanceState.SUSPENDED:
            break
        time.sleep(0.05)
    watcher.stop()
    assert watcher.running is False
    assert watcher._tracker.snapshot().state is InstanceState.SUSPENDED
    assert calls == ["resume"]
    assert watcher._resume_calls == 1


def test_start_is_idempotent_and_stop_safe() -> None:
    watcher = PowerStateWatcher(InstanceTracker(), lambda: None)
    watcher.start()
    watcher.start()  # second start must not double-launch
    watcher.stop()
    watcher.stop()  # stopping twice must not raise
    assert watcher.running is False


def test_on_resume_exception_never_breaks_state(tmp_path) -> None:
    def boom() -> None:
        raise RuntimeError("reconcile blew up")

    watcher = _ScriptedWatcher(
        [BUSCTL_SUSPEND, BUSCTL_RESUME], boom, tmp_path
    )
    watcher._tracker.transition(InstanceState.READY)
    watcher.start()
    try:
        watcher.wait_done()
    finally:
        watcher.stop()
    assert watcher._resume_calls == 1
    assert watcher._pre_suspend_state is None
