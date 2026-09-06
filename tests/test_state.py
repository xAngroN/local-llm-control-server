"""Tests for the llamactl instance state tracker."""

import json
from datetime import datetime

import pytest

from llamactl.state import InstanceState, InstanceStatus, InstanceTracker

ALL_STATES = [
    InstanceState.STOPPED,
    InstanceState.STARTING,
    InstanceState.LOADING,
    InstanceState.READY,
    InstanceState.DEGRADED,
    InstanceState.CRASHED,
    InstanceState.SUSPENDED,
]


def test_enum_has_exactly_seven_states() -> None:
    assert {s.value for s in InstanceState} == {
        "stopped",
        "starting",
        "loading",
        "ready",
        "degraded",
        "crashed",
        "suspended",
    }
    assert len(InstanceState) == 7


@pytest.mark.parametrize("state", ALL_STATES, ids=lambda s: s.value)
def test_each_state_round_trips_through_to_dict(state: InstanceState) -> None:
    tracker = InstanceTracker()
    tracker.transition(state, profile="p1", message="m")
    data = tracker.to_dict()
    assert data["state"] == state.value
    # since is an ISO-8601 string that parses back to a datetime
    parsed = datetime.fromisoformat(data["since"])
    assert isinstance(parsed, datetime)
    # dict must be JSON-serializable
    json.dumps(data)


def test_default_state_is_stopped() -> None:
    tracker = InstanceTracker()
    assert tracker.snapshot().state is InstanceState.STOPPED


def test_snapshot_returns_a_copy() -> None:
    tracker = InstanceTracker()
    tracker.transition(
        InstanceState.READY,
        profile="p1",
        container_id="abc123",
        log_tail=["line1"],
        exit_code=None,
    )
    snap = tracker.snapshot()
    assert snap is not tracker.snapshot()

    # Mutating the snapshot must not affect the internal state
    snap.profile = "changed"
    snap.state = InstanceState.CRASHED
    snap.log_tail.append("mutated")
    snap.message = "mutated"
    fresh = tracker.snapshot()
    assert fresh.state is InstanceState.READY
    assert fresh.profile == "p1"
    assert fresh.log_tail == ["line1"]
    assert fresh.message is None

    # Transitioning the tracker must not affect the earlier snapshot
    tracker.transition(InstanceState.STOPPED)
    assert snap.state is InstanceState.CRASHED
    assert snap.profile == "changed"


def test_transition_updates_since_and_fields() -> None:
    tracker = InstanceTracker()
    before = tracker.snapshot().since
    tracker.transition(
        InstanceState.LOADING,
        profile="p2",
        container_id="cid-1",
        exit_code=None,
        message="loading model",
    )
    snap = tracker.snapshot()
    assert snap.state is InstanceState.LOADING
    assert snap.profile == "p2"
    assert snap.container_id == "cid-1"
    assert snap.message == "loading model"
    assert snap.since >= before


def test_stopped_resets_profile_container_exitcode_logtail() -> None:
    tracker = InstanceTracker()
    tracker.transition(
        InstanceState.READY,
        profile="p1",
        container_id="abc",
        exit_code=0,
        log_tail=["a", "b"],
        message="ready",
    )
    tracker.transition(InstanceState.STOPPED)
    snap = tracker.snapshot()
    assert snap.state is InstanceState.STOPPED
    assert snap.profile is None
    assert snap.container_id is None
    assert snap.exit_code is None
    assert snap.log_tail == []


def test_stopped_wins_over_explicit_fields() -> None:
    tracker = InstanceTracker()
    tracker.transition(
        InstanceState.STOPPED,
        profile="should-be-cleared",
        container_id="nope",
        exit_code=1,
        log_tail=["x"],
    )
    snap = tracker.snapshot()
    assert snap.profile is None
    assert snap.container_id is None
    assert snap.exit_code is None
    assert snap.log_tail == []


def test_to_dict_is_json_serializable_with_iso_since() -> None:
    tracker = InstanceTracker()
    tracker.transition(
        InstanceState.CRASHED,
        profile="p1",
        container_id="cid",
        exit_code=137,
        log_tail=["segfault"],
        message="oom killed",
    )
    data = tracker.to_dict()
    assert data == {
        "state": "crashed",
        "profile": "p1",
        "container_id": "cid",
        "since": datetime.fromisoformat(data["since"]).isoformat(),
        "exit_code": 137,
        "log_tail": ["segfault"],
        "message": "oom killed",
    }
    assert isinstance(data["since"], str)
    round_tripped = json.loads(json.dumps(data))
    assert round_tripped["state"] == "crashed"
    assert round_tripped["log_tail"] == ["segfault"]


def test_status_dataclass_defaults() -> None:
    status = InstanceStatus()
    assert status.state is InstanceState.STOPPED
    assert status.profile is None
    assert status.container_id is None
    assert status.exit_code is None
    assert status.log_tail == []
    assert status.message is None
    assert isinstance(status.since, datetime)
