"""Shared pytest fixtures: a fake podman binary backed by a state file."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

FAKE_SCRIPT = Path(__file__).resolve().parent / "fakes" / "fake_podman.py"


class FakePodman:
    """Helper object exposing the fake podman state to tests."""

    def __init__(self, state_file: Path, binary: str) -> None:
        self.state_file = state_file
        self.binary = binary

    def _read(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _write(self, state: dict) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)

    def containers(self) -> dict:
        """Return the full state as a dict mapping name -> entry."""
        return self._read()

    def last_run_args(self, name: str | None = None) -> list[str] | None:
        """Return the full argument list passed to ``run`` for a container.

        With ``name=None`` the most recent entry (by insertion order) is
        used.  Returns ``None`` when no run is recorded.
        """
        state = self._read()
        if not state:
            return None
        if name is None:
            name = next(reversed(state))
        entry = state.get(name)
        return list(entry["run_args"]) if entry else None

    def kill(self, name: str, exit_code: int = 1) -> None:
        """Simulate an unexpected container death (no ``stop`` involved).

        Sets status to ``exited`` with the given non-zero exit code.
        Raises ``KeyError`` when the container is unknown.
        """
        state = self._read()
        if name not in state:
            raise KeyError(f"unknown container: {name}")
        state[name]["status"] = "exited"
        state[name]["exit_code"] = exit_code
        state[name]["logs"].append(f"fake-podman: container {name} died")
        self._write(state)


def _make_executable(script: Path) -> Path:
    """Materialize an executable fake podman inside a temp dir is handled
    by the fixture; here we only ensure the source script is executable
    so it can be invoked directly as well."""
    mode = script.stat().st_mode
    if not mode & stat.S_IEXEC:
        script.chmod(mode | stat.S_IEXEC)
    return script


def _fake_podman_fixture(tmp_path, monkeypatch) -> FakePodman:
    state_dir = tmp_path / "fake-podman"
    state_dir.mkdir()
    state_file = state_dir / "state.json"
    # Copy the fake script into the temp dir so the fake only ever
    # writes inside the fixture-provided temporary directory.
    fake_script = state_dir / "fake_podman.py"
    fake_script.write_text(FAKE_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    fake_script.chmod(fake_script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("LLAMACTL_PODMAN", str(fake_script))
    monkeypatch.setenv("FAKE_PODMAN_STATE", str(state_file))
    return FakePodman(state_file, str(fake_script))


@pytest.fixture()
def fake_podman(tmp_path, monkeypatch) -> FakePodman:
    """Fixture providing the fake podman binary and a state helper object.

    Sets ``LLAMACTL_PODMAN`` to the fake script and ``FAKE_PODMAN_STATE``
    to a temporary state file, and cleans both up after the test.
    """
    return _fake_podman_fixture(tmp_path, monkeypatch)
