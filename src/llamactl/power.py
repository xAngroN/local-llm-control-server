"""Host suspend-to-RAM helper (no privilege escalation).

``systemctl suspend`` and ``loginctl suspend`` are invoked as an
unprivileged user -- without ``sudo`` and without ``pkexec``.  Whether
that is permitted depends on the logind/polkit policy of the active
session; a restrictive policy is a one-off polkit rule away (documented
operational workaround), not a reason to escalate privileges in code.

Suspending the host does not stop the model instance: the container is
frozen with the machine and the suspend watcher (previous task) takes
over state tracking.  No stop or reload logic is invoked here.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

#: Candidate commands in priority order; the first success wins.
#: Deliberately plain commands: no ``sudo``, no ``pkexec``.
_SUSPEND_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("systemctl", "suspend"),
    ("loginctl", "suspend"),
)


class PowerError(RuntimeError):
    """Raised when every suspend attempt fails."""


def _attempt(cmd: tuple[str, ...], runner: Callable[..., subprocess.CompletedProcess]) -> subprocess.CompletedProcess:
    """Run one candidate command as an argument list (no shell)."""
    return runner(list(cmd), capture_output=True, text=True)


def suspend(
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    """Suspend the host (Suspend-to-RAM) as an unprivileged user.

    Tries the candidate commands in order; the first exit code zero
    ends the attempt.  When all attempts fail, a :class:`PowerError`
    is raised whose text carries the stderr of every attempt.

    ``runner`` is injectable for tests; it must be called with the
    command as a list (never ``shell=True``).
    """
    failures: list[str] = []
    for cmd in _SUSPEND_COMMANDS:
        proc = _attempt(cmd, runner)
        if proc.returncode == 0:
            return
        failures.append(f"{' '.join(cmd)} failed (rc={proc.returncode}): {proc.stderr.strip()}")
    raise PowerError("; ".join(failures))
