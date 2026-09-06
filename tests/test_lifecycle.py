"""Tests for llamactl.lifecycle using the fake podman binary."""

from __future__ import annotations

from pathlib import Path

import pytest

from llamactl.config import load_config
from llamactl.lifecycle import (
    PROFILE_LABEL,
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
    assert _arg_after(args, "--label") == f"{PROFILE_LABEL}={name}"
    # The --label flag must precede the image and every server argument
    # (it is invalid to a real ``podman run`` after the image/command
    # arguments), i.e. it belongs in the podman-run flag section.
    idx_label = args.index("--label")
    idx_model = args.index("--model")
    idx_volume_val = args.index(args[args.index("--volume") + 1])
    idx_image = args.index(profiles[name].image)
    assert idx_volume_val < idx_label < idx_image < idx_model
    # The image immediately follows the label value.
    assert args[idx_label + 2] == profiles[name].image


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


def test_stop_running_instance_stops_container(fake_podman) -> None:
    """Stopping a running instance ends the container and resets the tracker."""
    common, profiles = load_config(PROFILES_TOML)
    tracker = InstanceTracker()
    mgr = LifecycleManager(common, profiles, Podman(), tracker)

    mgr.start("fast")
    entry = fake_podman.containers()[common.container_name]
    assert entry["status"] == "running"

    status = mgr.stop()

    # The fake podman marks the container exited with code 0.
    entry = fake_podman.containers()[common.container_name]
    assert entry["status"] == "exited"
    assert entry["exit_code"] == 0

    # Tracker ends in stopped with profile reset.
    assert status.state is InstanceState.STOPPED
    assert status.profile is None
    assert status.container_id is None
    snap = tracker.snapshot()
    assert snap.state is InstanceState.STOPPED
    assert snap.profile is None

    # The shutdown is flagged as self-initiated for crash detection.
    assert mgr._expected_stop is True


def test_reload_switches_profile_and_runs_again(fake_podman) -> None:
    """Reload fast->large->fast each ends with one running container under the new profile."""
    common, profiles = load_config(PROFILES_TOML)
    tracker = InstanceTracker()
    mgr = LifecycleManager(common, profiles, Podman(), tracker)

    mgr.start("fast")

    status = mgr.reload("large")

    assert status.state is InstanceState.LOADING
    assert status.profile == "large"
    entry = fake_podman.containers()[common.container_name]
    assert entry["status"] == "running"
    assert status.container_id == entry["id"]
    assert status.message is None
    # The new container ran with the large profile's args.
    args = fake_podman.last_run_args(common.container_name)
    assert _arg_after(args, "--ctx-size") == EXPECTED_CTX["large"]

    status = mgr.reload("fast")

    assert status.state is InstanceState.LOADING
    assert status.profile == "fast"
    entry = fake_podman.containers()[common.container_name]
    assert entry["status"] == "running"
    assert status.container_id == entry["id"]
    assert status.message is None
    args = fake_podman.last_run_args(common.container_name)
    assert _arg_after(args, "--ctx-size") == EXPECTED_CTX["fast"]
    # Only one container has ever been running at a time.
    assert len(fake_podman.containers()) == 1


def test_reload_unknown_profile_leaves_running_instance_untouched(
    fake_podman,
) -> None:
    """A typo in reload must not stop the running instance."""
    common, profiles = load_config(PROFILES_TOML)
    tracker = InstanceTracker()
    mgr = LifecycleManager(common, profiles, Podman(), tracker)

    first = mgr.start("fast")
    before = tracker.to_dict()

    with pytest.raises(UnknownProfileError, match="nope"):
        mgr.reload("nope")

    # Container still running, same id, tracker state unchanged.
    entry = fake_podman.containers()[common.container_name]
    assert entry["status"] == "running"
    assert entry["id"] == first.container_id
    snap = tracker.snapshot()
    assert snap.state is InstanceState.LOADING
    assert snap.profile == "fast"
    assert snap.container_id == first.container_id
    after = tracker.to_dict()
    assert after["profile"] == before["profile"]
    assert after["state"] == before["state"]


def test_reload_failed_start_reports_stopped_with_message(fake_podman) -> None:
    """If the new profile's start fails, the old profile must not be reported."""
    common, profiles = load_config(PROFILES_TOML)
    tracker = InstanceTracker()
    mgr = LifecycleManager(common, profiles, Podman(), tracker)

    mgr.start("fast")
    real_start = mgr._podman.start_container

    def boom(args: list[str]) -> str:
        raise PodmanError(125, "reload start failed", args)

    mgr._podman.start_container = boom  # type: ignore[method-assign]
    try:
        with pytest.raises(PodmanError):
            mgr.reload("large")
    finally:
        mgr._podman.start_container = real_start  # type: ignore[method-assign]

    # Stopped with the failure message; the old profile is gone.
    snap = tracker.snapshot()
    assert snap.state is InstanceState.STOPPED
    assert snap.profile is None
    assert snap.container_id is None
    assert snap.message is not None
    assert "reload start failed" in snap.message
    # No container left running.
    entry = fake_podman.containers()[common.container_name]
    assert entry["status"] == "exited"


def test_stop_with_nothing_running_is_not_an_error(fake_podman) -> None:
    """Stopping with nothing active returns stopped without raising."""
    common, profiles = load_config(PROFILES_TOML)
    tracker = InstanceTracker()
    mgr = LifecycleManager(common, profiles, Podman(), tracker)

    # No start ever happened; tracker is stopped and no container exists.
    status = mgr.stop()

    assert status.state is InstanceState.STOPPED
    assert status.profile is None
    assert fake_podman.containers() == {}
    assert tracker.snapshot().state is InstanceState.STOPPED


def test_reconcile_empty_podman_state_reports_stopped(fake_podman) -> None:
    """Reconciling with no container at all reports stopped."""
    common, profiles = load_config(PROFILES_TOML)
    tracker = InstanceTracker()
    mgr = LifecycleManager(common, profiles, Podman(), tracker)

    status = mgr.reconcile()

    assert status.state is InstanceState.STOPPED
    assert status.profile is None
    assert status.container_id is None
    assert status.exit_code is None
    assert status.message is None
    assert fake_podman.containers() == {}


def test_reconcile_adopts_preexisting_running_container_with_label(
    fake_podman,
) -> None:
    """A running container written into the fake state is adopted with id and profile."""
    common, profiles = load_config(PROFILES_TOML)
    name = common.container_name
    container_id = "abc123def456"
    state = {
        name: {
            "id": container_id,
            "status": "running",
            "exit_code": None,
            "run_args": [
                "run",
                "--name",
                name,
                "--label",
                f"{PROFILE_LABEL}=safe",
                "image",
            ],
            "logs": [],
        }
    }
    fake_podman._write(state)
    tracker = InstanceTracker()
    mgr = LifecycleManager(common, profiles, Podman(), tracker)

    status = mgr.reconcile()

    assert status.state is InstanceState.LOADING
    assert status.profile == "safe"
    assert status.container_id == container_id
    assert status.message is None
    snap = mgr._tracker.snapshot()
    assert snap.state is InstanceState.LOADING
    assert snap.profile == "safe"
    assert snap.container_id == container_id
    # Reconcile only observes: nothing was started or stopped.
    assert fake_podman.containers()[name]["status"] == "running"


def test_reconcile_running_container_without_profile_label(
    fake_podman,
) -> None:
    """A running container with no derivable profile is adopted with message, not an exception."""
    common, profiles = load_config(PROFILES_TOML)
    name = common.container_name
    container_id = "f00feed00f12"
    state = {
        name: {
            "id": container_id,
            "status": "running",
            "exit_code": None,
            "run_args": ["run", "--name", name, "image"],
            "logs": [],
        }
    }
    fake_podman._write(state)
    tracker = InstanceTracker()
    mgr = LifecycleManager(common, profiles, Podman(), tracker)

    status = mgr.reconcile()

    assert status.state is InstanceState.LOADING
    assert status.profile is None
    assert status.container_id == container_id
    assert status.message is not None
    assert "llamactl.profile" in status.message
    snap = mgr._tracker.snapshot()
    assert snap.state is InstanceState.LOADING
    assert snap.profile is None
    assert snap.container_id == container_id


def test_reconcile_after_manager_restart_finds_running_instance(fake_podman) -> None:
    """After a simulated manager restart (fresh tracker + manager) reconcile re-adopts the instance."""
    common, profiles = load_config(PROFILES_TOML)
    name = common.container_name
    first_tracker = InstanceTracker()
    first = LifecycleManager(common, profiles, Podman(), first_tracker)

    started = first.start("large")
    entry = fake_podman.containers()[name]
    assert entry["status"] == "running"
    assert started.state is InstanceState.LOADING
    assert started.profile == "large"

    # Simulate an API-server restart: brand-new tracker and manager, but
    # the same fake podman state (the container keeps running).
    restarted = LifecycleManager(common, profiles, Podman(), InstanceTracker())

    status = restarted.reconcile()

    assert status.state is InstanceState.LOADING
    assert status.profile == "large"
    assert status.container_id == entry["id"]
    assert status.container_id == started.container_id
    assert status.message is None
    snap = restarted._tracker.snapshot()
    assert snap.state is InstanceState.LOADING
    assert snap.profile == "large"
    assert snap.container_id == entry["id"]
    # Still exactly one container, untouched by reconcile.
    assert len(fake_podman.containers()) == 1
    assert fake_podman.containers()[name]["status"] == "running"


def test_reconcile_exited_container_reports_stopped_with_exit_code(fake_podman) -> None:
    """An exited container is reported as stopped with the exit code taken over."""
    common, profiles = load_config(PROFILES_TOML)
    name = common.container_name
    container_id = "deadbeef0042"
    state = {
        name: {
            "id": container_id,
            "status": "exited",
            "exit_code": 137,
            "run_args": ["run", "--name", name, "image"],
            "logs": [],
        }
    }
    fake_podman._write(state)
    tracker = InstanceTracker()
    mgr = LifecycleManager(common, profiles, Podman(), tracker)

    status = mgr.reconcile()

    assert status.state is InstanceState.STOPPED
    assert status.exit_code == 137
    assert status.profile is None
    assert status.container_id is None
