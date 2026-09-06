"""Tests for the podman CLI wrapper using a fake binary via LLAMACTL_PODMAN."""

import json
import os
import stat
import subprocess

import pytest

from llamactl.podman import Podman, PodmanError

FAKE = r"""
import json
import sys

args = sys.argv[1:]

if args[:1] == ["run"]:
    # emit a container id
    print("abc123def456")
    sys.exit(0)

if args[:1] == ["stop"]:
    if args[-1] == "missing":
        sys.stderr.write('Error: no such container: missing\n')
        sys.exit(125)
    if args[-1] == "boom":
        sys.stderr.write("Error: timeout reached while stopping container\n")
        sys.exit(125)
    sys.exit(0)

if args[:1] == ["inspect"]:
    name = args[-1]
    if name == "missing":
        sys.stderr.write('Error: no such container: missing\n')
        sys.exit(125)
    state = {"Running": name == "up", "ExitCode": 3 if name == "crashed" else 0}
    # real ``podman inspect --format json <name>`` returns a JSON *array*
    print(json.dumps([{"State": state}]))
    sys.exit(0)

if args[:1] == ["logs"]:
    name = args[-1]
    if name == "broken":
        sys.stderr.write('Error: no such container: broken\n')
        sys.exit(125)
    print("line1\nline2\nline3")
    sys.exit(0)

if args[:1] == ["events"]:
    print(json.dumps({"Type": "container", "Action": "start", "id": "abc"}))
    print(json.dumps({"Type": "container", "Action": "die", "id": "abc"}))
    sys.exit(0)

if args[:1] == ["boom"]:
    sys.stderr.write("kaboom: something went wrong\n")
    sys.exit(7)

sys.exit(0)
"""


@pytest.fixture()
def fake_podman(tmp_path, monkeypatch) -> str:
    path = tmp_path / "fake-podman"
    path.write_text("#!/usr/bin/env python3\n" + FAKE)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("LLAMACTL_PODMAN", str(path))
    return str(path)


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
    p = Podman(binary=fake_podman)
    out = p.run(["run"])
    assert out == "abc123def456"


def test_run_failure_raises_podman_error_with_attributes(fake_podman) -> None:
    p = Podman(binary=fake_podman)
    with pytest.raises(PodmanError) as excinfo:
        p.run(["boom"])
    err = excinfo.value
    assert err.returncode == 7
    assert "kaboom" in err.stderr
    # stderr is carried, not swallowed
    assert isinstance(err, RuntimeError)


def test_start_container_returns_id(fake_podman) -> None:
    p = Podman(binary=fake_podman)
    cid = p.start_container(["run", "-d", "image:latest"])
    assert cid == "abc123def456"


def test_stop_container_missing_is_not_an_error(fake_podman) -> None:
    p = Podman(binary=fake_podman)
    # must return normally, not raise
    assert p.stop_container("missing") is None


def test_stop_container_existing_ok(fake_podman) -> None:
    p = Podman(binary=fake_podman)
    assert p.stop_container("up") is None


def test_stop_container_real_error_still_raises(fake_podman) -> None:
    p = Podman(binary=fake_podman)
    with pytest.raises(PodmanError):
        p.stop_container("boom")


def test_inspect_missing_returns_none(fake_podman) -> None:
    p = Podman(binary=fake_podman)
    assert p.inspect("missing") is None


def test_inspect_existing_returns_dict(fake_podman) -> None:
    p = Podman(binary=fake_podman)
    info = p.inspect("up")
    assert isinstance(info, dict)
    assert info["State"]["Running"] is True


def test_inspect_unwraps_array_output(fake_podman) -> None:
    # podman emits a JSON array for a single name; inspect() must return the dict.
    p = Podman(binary=fake_podman)
    info = p.inspect("crashed")
    assert isinstance(info, dict)
    assert info["State"]["ExitCode"] == 3


def test_is_running(fake_podman) -> None:
    p = Podman(binary=fake_podman)
    assert p.is_running("up") is True
    assert p.is_running("crashed") is False
    assert p.is_running("missing") is False


def test_exit_code(fake_podman) -> None:
    p = Podman(binary=fake_podman)
    assert p.exit_code("crashed") == 3
    assert p.exit_code("up") == 0
    assert p.exit_code("missing") is None


def test_logs(fake_podman) -> None:
    p = Podman(binary=fake_podman)
    lines = p.logs("up", tail=10)
    assert lines == ["line1", "line2", "line3"]
    assert p.logs("broken") == []


def test_events_yields_parsed_json(fake_podman) -> None:
    p = Podman(binary=fake_podman)
    events = list(p.events(["--filter", "type=container"]))
    assert events == [
        {"Type": "container", "Action": "start", "id": "abc"},
        {"Type": "container", "Action": "die", "id": "abc"},
    ]


def test_events_does_not_block_on_import_or_ctor(fake_podman) -> None:
    # Constructing and calling events() returns an iterator without running
    # the process until iteration begins.
    p = Podman(binary=fake_podman)
    it = p.events([])
    assert not isinstance(it, (list, tuple))
    # first item triggers the subprocess
    first = next(it)
    assert first["Action"] == "start"


def test_run_uses_no_shell(fake_podman, monkeypatch) -> None:
    """Verify subprocess.run is invoked with a list and no shell=True."""
    import llamactl.podman as mod

    calls = []

    real_run = subprocess.run

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs.get("shell") is not True
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    p = Podman(binary=fake_podman)
    p.run(["run"])
    assert calls, "subprocess.run was not called"
    args, kwargs = calls[0]
    # first positional arg is the command list
    cmd = args[0] if args else kwargs.get("args")
    assert isinstance(cmd, list)
    assert cmd[0] == fake_podman
