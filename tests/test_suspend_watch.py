"""Tests for the host power-state watcher (suspend/resume).

Covers the D-Bus-free seams (``parse_line`` and ``_monitor_command``) and
the lifecycle behaviour: a suspend from ``ready`` must land in
``suspended`` (never ``crashed``), a ``die`` event arriving while
``suspended`` must not turn the instance ``crashed``, and the resume
callback is invoked exactly once per cycle.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from llamactl.config import load_config
from llamactl.lifecycle import LifecycleManager
from llamactl.podman import Podman
from llamactl.state import InstanceState, InstanceTracker
from llamactl.suspend_watch import LOGIND_INTERFACE, PowerStateWatcher

PROFILES_TOML = Path(__file__).resolve().parent.parent / "config" / "profiles.toml"

BUSCTL_SUSPEND = (
    "2026-09-06 12:00:00.000000 org.freedesktop.login1 "
    "org.freedesktop.login1.Manager PrepareForSleep true"
)
BUSCTL_RESUME = (
    "2026-09-06 12:00:05.000000 org.freedesktop.login1 "
    "org.freedesktop.login1.Manager PrepareForSleep false"
)
DBUSMON_SUSPEND = (
    "signal /org/freedesktop/login1 "
    "org.freedesktop.login1.Manager PrepareForSleep true  "
    "(sender=org.freedesktop.login1, serial=42)"
)
DBUSMON_RESUME = (
    "signal /org/freedesktop/login1 "
    "org.freedesktop.login1.Manager PrepareForSleep false  "
    "(sender=org.freedesktop.login1, serial=43)"
)


# ---------------------------------------------------------------------------
# parse_line: both monitor formats, both transitions, irrelevant lines
# ---------------------------------------------------------------------------


class TestParseLine:
    def setup_method(self) -> None:
        self.watcher = PowerStateWatcher(InstanceTracker(), lambda: None)

    def test_busctl_suspend_line_is_true(self) -> None:
        assert self.watcher.parse_line(BUSCTL_SUSPEND) is True

    def test_busctl_resume_line_is_false(self) -> None:
        assert self.watcher.parse_line(BUSCTL_RESUME) is False

    def test_dbus_monitor_suspend_line_is_true(self) -> None:
        assert self.watcher.parse_line(DBUSMON_SUSPEND) is True

    def test_dbus_monitor_resume_line_is_false(self) -> None:
        assert self.watcher.parse_line(DBUSMON_RESUME) is False

    @pytest.mark.parametrize(
        "line",
        [
            "",
            "   ",
            "some other log line",
            "2026-09-06 12:00:00 org.freedesktop.login1 "
            "org.freedesktop.login1.Manager PrepareForShutdown true",
            "2026-09-06 12:00:00 org.freedesktop.systemd1 "
            "org.freedesktop.systemd1.Manager PrepareForSleep true",
            "2026-09-06 12:00:00 org.freedesktop.login1 "
            "org.freedesktop.login1.Manager PrepareForSleep 1",
        ],
    )
    def test_irrelevant_lines_are_none(self, line: str) -> None:
        assert self.watcher.parse_line(line) is None


# ---------------------------------------------------------------------------
# _monitor_command: no running D-Bus needed
# ---------------------------------------------------------------------------


class TestMonitorCommand:
    def test_returns_a_list_of_strings(self) -> None:
        cmd = PowerStateWatcher(InstanceTracker(), lambda: None)._monitor_command()
        assert isinstance(cmd, list)
        assert cmd and all(isinstance(part, str) for part in cmd)

    def test_prefers_busctl_when_available(self, monkeypatch) -> None:
        import llamactl.suspend_watch as sw

        monkeypatch.setattr(sw.shutil, "which", lambda name: "/usr/bin/" + name)
        assert (
            PowerStateWatcher(InstanceTracker(), lambda: None)._monitor_command()
            == ["busctl", "--user", "monitor"]
        )

    def test_falls_back_to_dbus_monitor_without_busctl(self, monkeypatch) -> None:
        import llamactl.suspend_watch as sw

        monkeypatch.setattr(sw.shutil, "which", lambda name: None)
        cmd = PowerStateWatcher(InstanceTracker(), lambda: None)._monitor_command()
        assert cmd[0] == "dbus-monitor"
        assert any(
            LOGIND_INTERFACE in part and "PrepareForSleep" in part for part in cmd
        )


# ---------------------------------------------------------------------------
# Suspend / resume transitions against the tracker
# ---------------------------------------------------------------------------


class TestSuspendTransition:
    def test_suspend_from_ready_lands_in_suspended_not_crashed(self) -> None:
        tracker = InstanceTracker()
        tracker.transition(InstanceState.READY, profile="fast")
        resumed: list[int] = []
        watcher = PowerStateWatcher(tracker, resumed.append)

        watcher._handle_suspend_start()

        assert tracker.snapshot().state is InstanceState.SUSPENDED
        assert tracker.snapshot().state is not InstanceState.CRASHED
        assert watcher._pre_suspend_state is InstanceState.READY
        assert resumed == []

    def test_suspend_remembers_previous_state(self) -> None:
        tracker = InstanceTracker()
        tracker.transition(InstanceState.LOADING, profile="large")
        watcher = PowerStateWatcher(tracker, lambda: None)

        watcher._handle_suspend_start()

        assert watcher._pre_suspend_state is InstanceState.LOADING
        assert tracker.snapshot().state is InstanceState.SUSPENDED


class TestResumeCallback:
    def test_resume_invokes_callback_per_cycle(self) -> None:
        calls = []
        watcher = PowerStateWatcher(InstanceTracker(), lambda: calls.append(1))

        watcher._handle_resume()
        watcher._handle_resume()

        assert len(calls) == 2
        assert watcher._resume_calls == 2

    def test_resume_callback_failure_does_not_break_watcher(self) -> None:
        def boom() -> None:
            raise RuntimeError("no bus")

        watcher = PowerStateWatcher(InstanceTracker(), boom)
        watcher._handle_resume()  # must not raise
        assert watcher._resume_calls == 1

    def test_resume_forwards_to_manager_reconcile(self, fake_podman) -> None:
        """The API wiring maps on_resume to manager.reconcile."""
        common, profiles = load_config(PROFILES_TOML)
        manager = LifecycleManager(common, profiles, Podman(), InstanceTracker())
        calls: list[int] = []
        original = manager.reconcile

        def counting() -> object:
            calls.append(1)
            return original()

        manager.reconcile = counting  # type: ignore[method-assign]
        watcher = PowerStateWatcher(manager._tracker, manager.reconcile)
        watcher._handle_resume()
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# The suspended rule in lifecycle: die while suspended stays suspended
# ---------------------------------------------------------------------------


class TestSuspendedBlocksCrashed:
    def test_die_while_suspended_does_not_crash(self, fake_podman) -> None:
        common, profiles = load_config(PROFILES_TOML)
        manager = LifecycleManager(common, profiles, Podman(), InstanceTracker())
        # Simulate a ready instance that the host just suspended.
        manager._tracker.transition(InstanceState.READY, profile="fast")
        manager._tracker.transition(InstanceState.SUSPENDED)

        manager.handle_container_event(
            {"Action": "die", "name": common.container_name, "exitCode": 137}
        )

        snapshot = manager._tracker.snapshot()
        assert snapshot.state is InstanceState.SUSPENDED
        assert snapshot.state is not InstanceState.CRASHED

    def test_die_while_ready_still_crashes(self, fake_podman) -> None:
        common, profiles = load_config(PROFILES_TOML)
        manager = LifecycleManager(common, profiles, Podman(), InstanceTracker())
        manager._tracker.transition(InstanceState.READY, profile="fast")
        manager.handle_container_event(
            {"Action": "die", "name": common.container_name, "exitCode": 137}
        )
        assert manager._tracker.snapshot().state is InstanceState.CRASHED

    def test_die_for_other_container_is_ignored(self, fake_podman) -> None:
        common, profiles = load_config(PROFILES_TOML)
        manager = LifecycleManager(common, profiles, Podman(), InstanceTracker())
        manager._tracker.transition(InstanceState.READY, profile="fast")
        manager.handle_container_event(
            {"Action": "die", "name": "some-other-container", "exitCode": 1}
        )
        assert manager._tracker.snapshot().state is InstanceState.READY


# ---------------------------------------------------------------------------
# Full stream behaviour with a fake monitor command (no D-Bus involved)
# ---------------------------------------------------------------------------


class TestWatchLoopWithFakeStream:
    @staticmethod
    def _echo_script(tmp_path: Path) -> Path:
        script = tmp_path / "monitor.py"
        script.write_text(
            "import sys, time\n"
            "for line in sys.argv[1:]:\n"
            "    print(line, flush=True)\n"
            "    time.sleep(0.05)\n"
            "time.sleep(60)\n",
            encoding="utf-8",
        )
        return script

    def test_suspend_then_resume_updates_tracker_and_calls_back(self, tmp_path):
        """Feed suspend+resume lines through the real thread loop."""
        script = self._echo_script(tmp_path)
        tracker = InstanceTracker()
        tracker.transition(InstanceState.READY, profile="fast")
        calls: list[int] = []

        class FakeCommand(PowerStateWatcher):
            def _monitor_command(self) -> list[str]:
                return [
                    sys.executable,
                    str(script),
                    BUSCTL_SUSPEND,
                    BUSCTL_RESUME,
                    BUSCTL_SUSPEND,
                ]

        watcher = FakeCommand(tracker, lambda: calls.append(1))
        watcher.start()
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and watcher._resume_calls < 1:
                time.sleep(0.05)
            assert watcher._resume_calls >= 1
        finally:
            watcher.stop()

        assert len(calls) == 1  # one callback for the single resume line
        assert watcher._pre_suspend_state is InstanceState.READY
        assert tracker.snapshot().state is InstanceState.SUSPENDED

    def test_stop_is_prompt_and_idempotent(self, tmp_path) -> None:
        script = tmp_path / "monitor.py"
        script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")

        class FakeCommand(PowerStateWatcher):
            def _monitor_command(self) -> list[str]:
                return [sys.executable, str(script)]

        watcher = FakeCommand(InstanceTracker(), lambda: None)
        watcher.start()
        assert watcher.running
        started = time.monotonic()
        watcher.stop()
        watcher.stop()
        assert not watcher.running
        assert time.monotonic() - started < 5.0
