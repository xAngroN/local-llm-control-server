"""Tests for llamactl.events using the fake podman binary.

The fake podman's ``events`` stream emits a ``die`` event for every
container recorded in the state file whose status is not ``running``
when the stream starts, so each test creates a container, stops it,
and then starts the watcher.  The stream then delivers the die event
immediately -- there is no polling interval to wait for.
"""

from __future__ import annotations

import os
import threading
import time

from llamactl.events import ContainerEventWatcher
from llamactl.podman import Podman

DEADLINE_SECONDS = 2.0
NAME = "watched"


def _make_stopped_container(fake_podman) -> None:
    """Create a container via the fake podman, then stop it."""
    podman = Podman()
    podman.run(["run", "--name", NAME, "busybox", "true"])
    podman.stop_container(NAME, timeout=1)
    entry = fake_podman.containers()[NAME]
    assert entry["status"] == "exited"


def _live_child_pids() -> list[int]:
    """PIDs of this process's still-running children."""
    children = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat", "r") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
            if int(fields[1]) == os.getpid():
                children.append(int(pid))
        except (OSError, ValueError, IndexError):
            continue
    return children


def _any_child_alive() -> bool:
    for pid in _live_child_pids():
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            continue
    return False


def test_die_event_reaches_callback_quickly(fake_podman) -> None:
    """A die event is delivered without waiting for a polling interval."""
    _make_stopped_container(fake_podman)

    received: list[dict] = []
    got_die = threading.Event()

    def on_event(event: dict) -> None:
        received.append(event)
        if event.get("Action") == "die":
            got_die.set()

    watcher = ContainerEventWatcher(Podman(), NAME, on_event)
    t0 = time.monotonic()
    watcher.start()
    try:
        assert got_die.wait(DEADLINE_SECONDS), (
            "no die event within the deadline; got so far: "
            f"{[e.get('Action') for e in received]}"
        )
        elapsed = time.monotonic() - t0
        assert elapsed < DEADLINE_SECONDS, f"took {elapsed:.2f}s"
    finally:
        watcher.stop()


def test_stop_ends_thread_and_subprocess(fake_podman) -> None:
    """stop() ends the thread and the podman events subprocess; no
    callbacks arrive afterwards."""
    _make_stopped_container(fake_podman)

    received: list[dict] = []
    lock = threading.Lock()

    def on_event(event: dict) -> None:
        with lock:
            received.append(event)

    watcher = ContainerEventWatcher(Podman(), NAME, on_event)
    watcher.start()
    thread = watcher._thread
    assert thread is not None
    assert thread.daemon
    assert thread.is_alive()

    # Let the fake stream finish; the watcher is now in its restart wait.
    time.sleep(3.0)
    watcher.stop()

    assert not thread.is_alive(), "watcher thread still alive after stop()"
    # Give zombies a moment to be reaped; the subprocess must not survive.
    for _ in range(20):
        if not _any_child_alive():
            break
        time.sleep(0.1)
    assert not _any_child_alive(), "podman subprocess still running after stop()"

    with lock:
        count_before = len(received)
    # Give any stray callback a chance to arrive; it must not.
    time.sleep(1.0)
    with lock:
        assert len(received) == count_before, "callback arrived after stop()"


def test_callback_exception_does_not_kill_watcher(fake_podman, monkeypatch) -> None:
    """An exception in the callback is logged, not fatal to the watcher."""
    import llamactl.events as events_mod

    _make_stopped_container(fake_podman)

    boom = threading.Event()
    calls: list[dict] = []

    def on_event(event: dict) -> None:
        calls.append(event)
        if event.get("Action") == "die":
            boom.set()
        raise RuntimeError("callback exploded")

    # Keep the log output quiet; the watcher must log, not crash.
    monkeypatch.setattr(events_mod.logger, "exception", lambda *a, **k: None)

    watcher = ContainerEventWatcher(Podman(), NAME, on_event)
    watcher.start()
    try:
        assert boom.wait(DEADLINE_SECONDS), "no die event reached callback"
        # The watcher must survive the raising callback.
        thread = watcher._thread
        assert thread is not None
        assert thread.is_alive()
    finally:
        watcher.stop()
    # The die event was delivered despite the callback raising.
    assert any(e.get("Action") == "die" for e in calls)
