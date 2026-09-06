"""Tests for the podman CLI wrapper using a fake binary via LLAMACTL_PODMAN."""

import stat
import subprocess

import pytest

from llamactl.podman import Podman, PodmanError



def test_default_binary_without_env(monkeypatch) -> None:
    monkeypatch.delenv("LLAMACTL_PODMAN", raising=False)
    p = Podman()
    assert p.binary == "podman"


def test_env_var_resolved_at_construction_time(tmp_path, monkeypatch) -> None:
    """LLAMACTL_PODMAN set after import must still be honored at construction."""
    path = tmp_path / "late-podman"
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    # Set the env var only AFTER the module was imported (test module
    # imported llamactl.podman at the top), proving it is read per-construction.
    monkeypatch.setenv("LLAMACTL_PODMAN", str(path))
    assert Podman().binary == str(path)


def test_run_success(fake_podman) -> None:
    p = Podman(binary=fake_podman.binary)
    out = p.run(["run"])
    assert out  # a fake container id is emitted on stdout


def test_run_failure_raises_podman_error_with_attributes(fake_podman) -> None:
    p = Podman(binary=fake_podman.binary)
    with pytest.raises(PodmanError) as excinfo:
        p.run(["not-a-subcommand"])
    err = excinfo.value
    assert err.returncode == 125
    assert "unknown subcommand" in err.stderr
    # stderr is carried, not swallowed
    assert isinstance(err, RuntimeError)


def test_start_container_returns_id(fake_podman) -> None:
    p = Podman(binary=fake_podman.binary)
    cid = p.start_container(["run", "-d", "--name", "c1", "image:latest"])
    assert len(cid) == 12
    assert fake_podman.containers()["c1"]["id"] == cid


def test_stop_container_missing_is_not_an_error(fake_podman) -> None:
    p = Podman(binary=fake_podman.binary)
    # must return normally, not raise
    assert p.stop_container("missing") is None


def test_stop_container_existing_ok(fake_podman) -> None:
    p = Podman(binary=fake_podman.binary)
    p.start_container(["run", "-d", "--name", "up", "image:latest"])
    assert p.stop_container("up") is None
    assert p.exit_code("up") == 0


def test_inspect_missing_returns_none(fake_podman) -> None:
    p = Podman(binary=fake_podman.binary)
    assert p.inspect("missing") is None


def test_inspect_existing_returns_dict(fake_podman) -> None:
    p = Podman(binary=fake_podman.binary)
    p.start_container(["run", "-d", "--name", "up", "image:latest"])
    info = p.inspect("up")
    assert isinstance(info, dict)
    assert info["State"]["Running"] is True


def test_inspect_unwraps_array_output(fake_podman) -> None:
    # podman emits a JSON array for a single name; inspect() must return the dict.
    p = Podman(binary=fake_podman.binary)
    p.start_container(["run", "-d", "--name", "crashed", "image:latest"])
    fake_podman.kill("crashed", exit_code=3)
    info = p.inspect("crashed")
    assert isinstance(info, dict)
    assert info["State"]["ExitCode"] == 3
    assert info["State"]["Running"] is False


def test_is_running(fake_podman) -> None:
    p = Podman(binary=fake_podman.binary)
    p.start_container(["run", "-d", "--name", "up", "image:latest"])
    p.start_container(["run", "-d", "--name", "crashed", "image:latest"])
    fake_podman.kill("crashed", exit_code=3)
    assert p.is_running("up") is True
    assert p.is_running("crashed") is False
    assert p.is_running("missing") is False


def test_exit_code(fake_podman) -> None:
    p = Podman(binary=fake_podman.binary)
    p.start_container(["run", "-d", "--name", "crashed", "image:latest"])
    p.start_container(["run", "-d", "--name", "up", "image:latest"])
    fake_podman.kill("crashed", exit_code=3)
    assert p.exit_code("crashed") == 3
    assert p.exit_code("up") is None
    assert p.exit_code("missing") is None


def test_logs(fake_podman) -> None:
    p = Podman(binary=fake_podman.binary)
    p.start_container(["run", "-d", "--name", "up", "image:latest"])
    lines = p.logs("up", tail=10)
    assert lines == ["fake-podman started container up"]
    assert p.logs("broken") == []


def test_events_yields_parsed_json(fake_podman) -> None:
    p = Podman(binary=fake_podman.binary)
    cid = p.start_container(["run", "-d", "--name", "abc", "image:latest"])
    fake_podman.kill("abc", exit_code=1)
    events = list(p.events(["--filter", "type=container"]))
    actions = {e["Action"] for e in events}
    assert actions == {"start", "die"}
    assert all(e["Type"] == "container" and e["id"] == cid for e in events)


def test_events_does_not_block_on_import_or_ctor(fake_podman) -> None:
    # Constructing and calling events() returns an iterator without running
    # the process until iteration begins.
    p = Podman(binary=fake_podman.binary)
    p.start_container(["run", "-d", "--name", "lazy", "image:latest"])
    it = p.events([])
    assert not isinstance(it, (list, tuple))
    # first item triggers the subprocess
    first = next(it)
    assert first["Action"] == "start"


def test_run_records_state_and_flags(fake_podman) -> None:
    """run --name stores a running entry with id, flags, and emits the id."""
    p = Podman(binary=fake_podman.binary)
    cid = p.start_container(
        ["run", "-d", "--name", "inst-1", "image:latest", "--ctx-size", "4096"]
    )
    assert len(cid) == 12 and cid == cid.lower()
    containers = fake_podman.containers()
    assert "inst-1" in containers
    entry = containers["inst-1"]
    assert entry["status"] == "running"
    assert entry["id"] == cid
    # last_run_args returns the full argument list that reached the fake
    args = fake_podman.last_run_args("inst-1")
    assert args == ["run", "-d", "--name", "inst-1", "image:latest", "--ctx-size", "4096"]
    # default binary resolution picks up LLAMACTL_PODMAN too
    assert Podman().binary == fake_podman.binary
    assert p.is_running("inst-1") is True
    assert p.exit_code("inst-1") is None


def test_stop_sets_exited_with_zero(fake_podman) -> None:
    p = Podman(binary=fake_podman.binary)
    p.start_container(["run", "-d", "--name", "inst-2", "image:latest"])
    p.stop_container("inst-2")
    entry = fake_podman.containers()["inst-2"]
    assert entry["status"] == "exited"
    assert entry["exit_code"] == 0
    assert p.is_running("inst-2") is False
    assert p.exit_code("inst-2") == 0


def test_inspect_unknown_exits_125_with_no_such_container(fake_podman) -> None:
    """inspect on an unknown container: rc=125, 'no such container' on stderr."""
    proc = subprocess.run(
        [fake_podman.binary, "inspect", "--format", "json", "ghost"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 125
    assert "no such container" in proc.stderr
    # and the wrapper translates that into None
    assert Podman(binary=fake_podman.binary).inspect("ghost") is None


def test_kill_simulates_unexpected_death(fake_podman) -> None:
    """kill() via the state file marks the container exited without stop."""
    p = Podman(binary=fake_podman.binary)
    p.start_container(["run", "-d", "--name", "inst-3", "image:latest"])
    fake_podman.kill("inst-3", exit_code=137)
    entry = fake_podman.containers()["inst-3"]
    assert entry["status"] == "exited"
    assert entry["exit_code"] == 137
    assert p.is_running("inst-3") is False
    assert p.exit_code("inst-3") == 137
    with pytest.raises(KeyError):
        fake_podman.kill("never-started")


def test_logs_tail_returns_stored_lines(fake_podman) -> None:
    p = Podman(binary=fake_podman.binary)
    p.start_container(["run", "-d", "--name", "inst-4", "image:latest"])
    lines = p.logs("inst-4", tail=10)
    assert lines == ["fake-podman started container inst-4"]
    # after stop a second line is stored; tail=1 keeps only the newest
    p.stop_container("inst-4")
    assert p.logs("inst-4", tail=1) == ["fake-podman stopped container inst-4"]
    assert p.logs("ghost") == []


def test_ps_lists_running_containers(fake_podman) -> None:
    p = Podman(binary=fake_podman.binary)
    p.start_container(["run", "-d", "--name", "a", "image:latest"])
    p.start_container(["run", "-d", "--name", "b", "image:latest"])
    p.stop_container("a")
    out = p.run(["ps"])
    assert "a" not in out
    assert "b" in out


def test_events_yields_start_and_die_for_killed_container(fake_podman) -> None:
    p = Podman(binary=fake_podman.binary)
    cid = p.start_container(["run", "-d", "--name", "inst-5", "image:latest"])
    fake_podman.kill("inst-5", exit_code=1)
    events = list(p.events(["--filter", "type=container"]))
    actions = {(e["Action"], e["id"]) for e in events}
    assert ("start", cid) in actions
    assert ("die", cid) in actions


def test_run_uses_no_shell(fake_podman, monkeypatch) -> None:
    """Verify subprocess.run is invoked with a list and no shell=True."""
    import llamactl.podman as mod

    calls = []

    real_run = subprocess.run

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs.get("shell") is not True
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.subprocess, "run", spy)
    p = Podman(binary=fake_podman.binary)
    p.run(["run"])
    assert calls, "subprocess.run was not called"
    args, kwargs = calls[0]
    # first positional arg is the command list
    cmd = args[0] if args else kwargs.get("args")
    assert isinstance(cmd, list)
    assert cmd[0] == fake_podman.binary
