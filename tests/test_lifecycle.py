"""Tests for llamactl.lifecycle using the fake podman binary."""

from __future__ import annotations

from pathlib import Path

import pytest

from llamactl.config import load_config
from llamactl.lifecycle import (
    AlreadyRunningError,
    LifecycleManager,
    UnknownProfileError,
)
from llamactl.podman import Podman, PodmanError
from llamactl.state import InstanceState, InstanceTracker

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES_TOML = REPO_ROOT / "config" / "profiles.toml"

EXPECTED_CTX = {
    "fast": "16384",
    "large": "65536",
    "safe": "32768",
    "shared": "32768",
}
EXPECTED_PARALLEL = {
    "fast": "1",
    "large": "1",
    "safe": "1",
    "shared": "4",
}


@pytest.fixture()
def manager(fake_podman) -> LifecycleManager:
    common, profiles = load_config(PROFILES_TOML)
    return LifecycleManager(common, profiles, Podman(), InstanceTracker())


def _arg_after(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_list_profiles_returns_all_four() -> None:
    common, profiles = load_config(PROFILES_TOML)
    mgr = LifecycleManager(common, profiles, Podman(), InstanceTracker())
    assert sorted(mgr.list_profiles()) == ["fast", "large", "safe", "shared"]


@pytest.mark.parametrize("name", ["fast", "large", "safe", "shared"])
def test_start_creates_running_container_with_profile_args(
    fake_podman, name: str
) -> None:
    common, profiles = load_config(PROFILES_TOML)
    mgr = LifecycleManager(common, profiles, Podman(), InstanceTracker())

    status = mgr.start(name)

    entry = fake_podman.containers()[common.container_name]
    assert entry["status"] == "running"

    # Tracker ends in loading with profile name and container id set.
    assert status.state is InstanceState.LOADING
    assert status.profile == name
    assert status.container_id == entry["id"]
    snap = mgr._tracker.snapshot()
    assert snap.state is InstanceState.LOADING
    assert snap.profile == name
    assert snap.container_id == entry["id"]
    assert snap.message is None

    # The args actually passed to podman carry the profile's ctx/parallel.
    args = fake_podman.last_run_args(common.container_name)
    assert args is not None
    assert _arg_after(args, "--ctx-size") == EXPECTED_CTX[name]
    assert _arg_after(args, "--parallel") == EXPECTED_PARALLEL[name]


def test_unknown_profile_raises_without_podman_or_tracker_touch(
    fake_podman,
) -> None:
    common, profiles = load_config(PROFILES_TOML)
    mgr = LifecycleManager(common, profiles, Podman(), InstanceTracker())
    before = mgr._tracker.to_dict()

    with pytest.raises(UnknownProfileError, match="nope"):
        mgr.start("nope")

    # No container was started at all.
    assert fake_podman.containers() == {}
    # Tracker untouched.
    assert mgr._tracker.to_dict()["state"] == "stopped"
    assert mgr._tracker.snapshot().profile is None
    assert mgr._tracker.snapshot().container_id is None
    assert mgr._tracker.snapshot().message is None
    assert before == mgr._tracker.to_dict()


def test_second_start_raises_and_first_container_keeps_running(
    fake_podman,
) -> None:
    common, profiles = load_config(PROFILES_TOML)
    tracker = InstanceTracker()
    mgr = LifecycleManager(common, profiles, Podman(), tracker)

    first = mgr.start("fast")
    status_before = tracker.snapshot()

    with pytest.raises(AlreadyRunningError):
        mgr.start("fast")
    with pytest.raises(AlreadyRunningError):
        mgr.start("safe")

    # The original container is still running and untouched.
    entry = fake_podman.containers()[common.container_name]
    assert entry["status"] == "running"
    assert entry["id"] == first.container_id
    assert fake_podman.last_run_args(common.container_name) == fake_podman.last_run_args()

    # State unchanged: still loading with the first profile/id.
    after = tracker.snapshot()
    assert after.state is InstanceState.LOADING
    assert after.profile == "fast"
    assert after.container_id == first.container_id
    assert after.since == status_before.since
    assert after.message is None


def test_podman_failure_ends_stopped_with_message(fake_podman) -> None:
    common, profiles = load_config(PROFILES_TOML)
    tracker = InstanceTracker()
    mgr = LifecycleManager(common, profiles, Podman(), tracker)

    real_start = mgr._podman.start_container

    def boom(args: list[str]) -> str:
        raise PodmanError(125, "boom: no such device", args)

    mgr._podman.start_container = boom  # type: ignore[method-assign]
    try:
        with pytest.raises(PodmanError):
            mgr.start("safe")
    finally:
        mgr._podman.start_container = real_start  # type: ignore[method-assign]

    snap = tracker.snapshot()
    assert snap.state is InstanceState.STOPPED
    assert snap.message is not None
    assert "boom: no such device" in snap.message
    assert snap.profile is None
    assert snap.container_id is None


def test_already_running_container_detected_via_podman(
    fake_podman,
) -> None:
    """Even when the tracker says stopped, a running podman container blocks start."""
    common, profiles = load_config(PROFILES_TOML)
    tracker = InstanceTracker()
    mgr = LifecycleManager(common, profiles, Podman(), tracker)

    # Simulate a container left over from a previous run while the
    # tracker has been reset to stopped.
    mgr.start("fast")
    tracker.transition(InstanceState.STOPPED)

    with pytest.raises(AlreadyRunningError):
        mgr.start("large")

    # First container still running, tracker still stopped.
    entry = fake_podman.containers()[common.container_name]
    assert entry["status"] == "running"
    assert tracker.snapshot().state is InstanceState.STOPPED
