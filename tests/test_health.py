"""Tests for the llamactl health poller.

The model server is not reachable in the test environment, so the HTTP
call is isolated in ``HealthPoller._probe`` and monkeypatched. These tests
exercise :meth:`HealthPoller.poll_once` directly, without a running
thread.
"""

from __future__ import annotations

from llamactl.health import HealthPoller
from llamactl.state import InstanceState, InstanceTracker


def _make_tracker(state: InstanceState) -> InstanceTracker:
    tracker = InstanceTracker()
    tracker.transition(state)
    return tracker


def test_success_sets_ready_from_loading(monkeypatch) -> None:
    tracker = _make_tracker(InstanceState.LOADING)
    poller = HealthPoller(tracker)
    monkeypatch.setattr(poller, "_probe", lambda: True)

    assert poller.poll_once() is True
    assert tracker.snapshot().state is InstanceState.READY


def test_success_sets_ready_from_degraded(monkeypatch) -> None:
    tracker = _make_tracker(InstanceState.DEGRADED)
    poller = HealthPoller(tracker)
    monkeypatch.setattr(poller, "_probe", lambda: True)

    assert poller.poll_once() is True
    assert tracker.snapshot().state is InstanceState.READY


def test_success_from_ready_stays_ready(monkeypatch) -> None:
    tracker = _make_tracker(InstanceState.READY)
    poller = HealthPoller(tracker)
    monkeypatch.setattr(poller, "_probe", lambda: True)

    assert poller.poll_once() is True
    assert tracker.snapshot().state is InstanceState.READY


def test_three_consecutive_failures_from_ready_become_degraded_not_crashed(
    monkeypatch,
) -> None:
    tracker = _make_tracker(InstanceState.READY)
    poller = HealthPoller(tracker, failure_threshold=3)
    monkeypatch.setattr(poller, "_probe", lambda: False)

    assert poller.poll_once() is False
    assert poller.poll_once() is False
    assert poller.poll_once() is False

    state = tracker.snapshot().state
    assert state is InstanceState.DEGRADED
    assert state is not InstanceState.CRASHED


def test_single_failure_does_not_change_state(monkeypatch) -> None:
    tracker = _make_tracker(InstanceState.READY)
    poller = HealthPoller(tracker, failure_threshold=3)
    monkeypatch.setattr(poller, "_probe", lambda: False)

    assert poller.poll_once() is False
    assert tracker.snapshot().state is InstanceState.READY


def test_two_failures_then_success_keeps_ready_and_resets_streak(monkeypatch) -> None:
    tracker = _make_tracker(InstanceState.READY)
    poller = HealthPoller(tracker, failure_threshold=3)

    def probe() -> bool:
        nonlocal result
        return result

    result = False
    monkeypatch.setattr(poller, "_probe", probe)

    assert poller.poll_once() is False  # streak 1
    assert poller.poll_once() is False  # streak 2
    assert tracker.snapshot().state is InstanceState.READY

    result = True
    assert poller.poll_once() is True  # resets the streak
    assert tracker.snapshot().state is InstanceState.READY

    # Two more failures after the reset still should not be enough.
    result = False
    assert poller.poll_once() is False
    assert poller.poll_once() is False
    assert tracker.snapshot().state is InstanceState.READY


def test_stopped_state_is_not_polled(monkeypatch) -> None:
    tracker = _make_tracker(InstanceState.STOPPED)
    poller = HealthPoller(tracker)
    called = {"n": 0}

    def probe() -> bool:
        called["n"] += 1
        return True

    monkeypatch.setattr(poller, "_probe", probe)

    assert poller.poll_once() is False
    assert called["n"] == 0
    assert tracker.snapshot().state is InstanceState.STOPPED


def test_crashed_state_is_not_polled(monkeypatch) -> None:
    tracker = _make_tracker(InstanceState.CRASHED)
    poller = HealthPoller(tracker)
    called = {"n": 0}

    def probe() -> bool:
        called["n"] += 1
        return True

    monkeypatch.setattr(poller, "_probe", probe)

    assert poller.poll_once() is False
    assert called["n"] == 0
    assert tracker.snapshot().state is InstanceState.CRASHED


def test_suspended_state_is_not_polled(monkeypatch) -> None:
    tracker = _make_tracker(InstanceState.SUSPENDED)
    poller = HealthPoller(tracker)
    called = {"n": 0}

    def probe() -> bool:
        called["n"] += 1
        return True

    monkeypatch.setattr(poller, "_probe", probe)

    assert poller.poll_once() is False
    assert called["n"] == 0
    assert tracker.snapshot().state is InstanceState.SUSPENDED
