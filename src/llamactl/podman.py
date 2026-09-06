"""Podman CLI wrapper: the single place where podman processes are started.

This module knows nothing about profiles or instance states -- it only
encapsulates process invocations. All podman invocations run as argument
lists through ``subprocess`` without ``shell=True`` (podman runs rootless
as a user process; no podman socket or API client is used).

The binary path can be overridden via the ``LLAMACTL_PODMAN`` environment
variable so the later lifecycle tests can run without real podman.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Iterator


class PodmanError(RuntimeError):
    """Raised when a podman invocation exits with a non-zero return code."""

    def __init__(self, returncode: int, stderr: str, args: list[str] | None = None) -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.args = args
        detail = f"podman {' '.join(args) if args else ''} failed (rc={returncode}): {stderr.strip()}"
        super().__init__(detail)


class Podman:
    """Thin wrapper around the podman CLI."""

    def __init__(self, binary: str | None = None) -> None:
        # Resolve the binary at construction time, not import time, so the
        # LLAMACTL_PODMAN override stays effective for later lifecycle tests.
        if binary is None:
            binary = os.environ.get("LLAMACTL_PODMAN", "podman")
        self.binary = binary

    def run(self, args: list[str], timeout: float | None = None) -> str:
        """Run ``[binary, *args]`` and return stripped stdout.

        Raises :class:`PodmanError` when the process exits non-zero.
        """
        cmd = [self.binary, *args]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise PodmanError(proc.returncode, proc.stderr, cmd)
        return proc.stdout.strip()

    def start_container(self, podman_args: list[str]) -> str:
        """Start a container from rendered podman args, return its ID."""
        return self.run(podman_args)

    def stop_container(self, name: str, timeout: int = 30) -> None:
        """Stop a container; a missing container is not an error."""
        try:
            self.run(["stop", "--time", str(timeout), name])
        except PodmanError as err:
            if "no such container" not in err.stderr.lower():
                raise

    def inspect(self, name: str) -> dict | None:
        """Inspect a container; returns ``None`` when it does not exist."""
        try:
            out = self.run(["inspect", "--format", "json", name])
        except PodmanError as err:
            if "no such container" in err.stderr.lower():
                return None
            raise
        data = json.loads(out)
        # Real ``podman inspect`` emits a JSON array (one element per name).
        if isinstance(data, list):
            return data[0] if data else None
        return data

    def is_running(self, name: str) -> bool:
        """Return whether the container currently has a running state."""
        info = self.inspect(name)
        if not info:
            return False
        state = info.get("State") or {}
        return bool(state.get("Running"))

    def exit_code(self, name: str) -> int | None:
        """Return the container's exit code, or ``None`` if unavailable."""
        info = self.inspect(name)
        if not info:
            return None
        state = info.get("State") or {}
        code = state.get("ExitCode")
        return int(code) if code is not None else None

    def logs(self, name: str, tail: int = 50) -> list[str]:
        """Return the last ``tail`` log lines; empty list on failure."""
        try:
            out = self.run(["logs", "--tail", str(tail), name])
        except PodmanError:
            return []
        return out.splitlines() if out else []

    def events(
        self,
        filters: list[str],
        handle: Callable[[subprocess.Popen], None] | None = None,
    ) -> Iterator[dict]:
        """Stream podman events as parsed JSON objects (line by line).

        The subprocess is started lazily on first iteration, so importing
        this module (or constructing :class:`Podman`) never blocks.

        ``handle`` is an optional callback invoked with the
        :class:`subprocess.Popen` as soon as the subprocess is started,
        letting a caller (e.g. the event watcher) terminate the process
        from another thread to unblock a reader parked on the stream.
        Existing callers may keep using ``p.events([...])`` unchanged.
        """
        cmd = [self.binary, "events", "--format", "json", *filters]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        assert proc.stdout is not None
        try:
            if handle is not None:
                handle(proc)
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
