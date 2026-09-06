"""Tests for llamactl.events using the fake podman binary.

The fake podman's ``events`` stream emits a ``die`` event for every
container recorded in the state file whose status is not ``running``
when the stream starts, so each test creates a container, stops it,
and then starts the watcher.  The stream then delivers the die event
immediately -- there is no polling interval to wait for.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time

import pytest

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


def test_stop_ends_thread_and_no_more_callbacks(fake_podman) -> None:
    """stop() ends the thread; no callbacks arrive afterwards."""
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

    # The fake stream exits on its own; the watcher is now in its
    # restart wait.  stop() must end the thread there, too.
    time.sleep(3.0)
    watcher.stop()

    assert not thread.is_alive(), "watcher thread still alive after stop()"
    with lock:
        count_before = len(received)
    # Give any stray callback a chance to arrive; it must not.
    time.sleep(1.0)
    with lock:
        assert len(received) == count_before, "callback arrived after stop()"


def test_stop_preempts_thread_blocked_on_live_stream(fake_podman) -> None:
    """stop() ends a thread blocked reading a live stream, fast.

    Models the production case of a running container: the watcher
    thread is parked on the next event while the (fake) podman events
    subprocess stays alive.  ``stop()`` must terminate that subprocess
    to force an EOF on the blocked read -- it cannot wait for an event
    to arrive or for a join timeout.  Uses ``threading.Event`` with a
    timeout, no fixed sleeps on the critical path.
    """
    _make_stopped_container(fake_podman)  # exercises fixture wiring

    got = threading.Event()
    procs: list[subprocess.Popen] = []

    class _LiveStream(Podman):
        """Podman whose events stream stays alive and never emits.

        Uses the new optional ``handle`` parameter: it records the
        subprocess so the test can assert ``stop()`` terminated it.
        """

        def events(self, filters, handle=None):
            cmd = [self.binary, "events", "--format", "json", *filters]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            if handle is not None:
                handle(proc)
            procs.append(proc)
            got.set()
            try:
                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
            finally:
                pass  # process cleanup is the watcher's job via the handle

    podman = _LiveStream()
    received: list[dict] = []
    watcher = ContainerEventWatcher(podman, NAME, received.append)  # type: ignore[arg-type]
    watcher.start()
    thread = watcher._thread
    assert thread is not None
    assert thread.daemon
    assert got.wait(DEADLINE_SECONDS), "events stream subprocess never started"
    # The thread is now parked on the blocked read.

    t0 = time.monotonic()
    watcher.stop()
    elapsed = time.monotonic() - t0

    assert not thread.is_alive(), "watcher thread still alive after stop()"
    assert elapsed < DEADLINE_SECONDS, (
        f"stop() took {elapsed:.2f}s to end a thread blocked on read"
    )
    assert received == [], "no callback should arrive while reading is blocked"
    assert procs, "events() never reported its subprocess via the handle"
    deadline = time.monotonic() + DEADLINE_SECONDS
    while time.monotonic() < deadline:
        if all(p.poll() is not None for p in procs):
            break
        time.sleep(0.05)
    else:
        pytest.fail("podman events subprocess survived stop()")


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
