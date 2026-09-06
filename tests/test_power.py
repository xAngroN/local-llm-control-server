"""Tests for the host suspend helper with an injected runner."""

from __future__ import annotations

import subprocess

import pytest

from llamactl.power import PowerError, suspend


def _completed(returncode: int, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


def test_success_calls_systemctl_suspend_exactly_once() -> None:
    calls: list[tuple[list, dict]] = []

    def runner(cmd: list, **kwargs: object) -> subprocess.CompletedProcess:
        calls.append((cmd, kwargs))
        return _completed(0)

    suspend(runner)

    assert len(calls) == 1
    (cmd, kwargs), = calls
    assert cmd == ["systemctl", "suspend"]
    assert "shell" not in kwargs
    assert kwargs.get("shell", False) is not True


def test_first_failure_falls_back_to_loginctl_suspend() -> None:
    calls: list[list] = []

    def runner(cmd: list, **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(cmd)
        if cmd[0] == "systemctl":
            return _completed(1, "polkit: authorization denied (systemctl)")
        return _completed(0)

    suspend(runner)

    assert calls == [
        ["systemctl", "suspend"],
        ["loginctl", "suspend"],
    ]


def test_both_failures_raise_power_error_with_both_stderr() -> None:
    calls: list[list] = []

    def runner(cmd: list, **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return _completed(1, f"denied by {cmd[0]}")

    with pytest.raises(PowerError) as excinfo:
        suspend(runner)

    assert calls == [
        ["systemctl", "suspend"],
        ["loginctl", "suspend"],
    ]
    message = str(excinfo.value)
    assert "systemctl" in message
    assert "loginctl" in message
    assert "denied by systemctl" in message
    assert "denied by loginctl" in message


def test_no_command_line_contains_sudo() -> None:
    calls: list[list] = []

    def runner(cmd: list, **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return _completed(1)

    with pytest.raises(PowerError):
        suspend(runner)

    for cmd in calls:
        assert "sudo" not in cmd
        assert "pkexec" not in cmd
