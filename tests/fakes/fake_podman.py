#!/usr/bin/env python3
"""Fake podman binary for lifecycle tests (no real podman, no GPU).

State is persisted in a JSON file whose path comes from the
``FAKE_PODMAN_STATE`` environment variable.  The file maps container
names to entries shaped like::

    {
        "id": "f00...",
        "status": "running" | "exited",
        "exit_code": 0 | null,
        "run_args": ["run", "--name", "x", ...],
        "logs": ["line1", ...]
    }

Supported subcommands:

    run ... --name <n> ...
        Writes a ``running`` entry with a fake container id and the full
        argument list, prints the id on stdout.

    stop <n>
        Sets status to ``exited`` with ``exit_code = 0``.

    inspect <n>
        Prints the entry as a JSON list, or exits with code 125 and
        ``no such container`` on stderr.

    logs --tail N <n>
        Prints the stored log lines (last N when ``--tail`` given).

    events
        Emits a few JSON events (start/die per running container) then
        exits so the stream never blocks.

    ps
        Lists running containers (one name per line).

An unexpected death can be simulated by editing the state file (the
fixture provides ``kill(name, exit_code)``): status ``exited`` with
``exit_code != 0`` without ``stop`` ever being called.
"""

from __future__ import annotations

import json
import os
import sys


def _state_path() -> str:
    path = os.environ.get("FAKE_PODMAN_STATE")
    if not path:
        sys.stderr.write("Error: FAKE_PODMAN_STATE is not set\n")
        sys.exit(125)
    return path


def _load(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save(path: str, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def _fake_id(name: str) -> str:
    """Deterministic, podman-ish 12-char container id derived from name."""
    digest = 0
    for ch in name:
        digest = (digest * 131 + ord(ch)) % (16 ** 12)
    return format(digest, "012x")


def _missing(name: str) -> None:
    sys.stderr.write(f'Error: no such container: {name}\n')
    sys.exit(125)


def _name_after(args: list[str]) -> str:
    """Last non-flag argument (container name at the end, podman-style)."""
    name = None
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg in ("--tail", "-t"):
            skip = True
            continue
        if arg.startswith("-"):
            continue
        name = arg
    if name is None:
        sys.stderr.write("Error: no container name given\n")
        sys.exit(125)
    return name


def cmd_run(args: list[str], path: str, state: dict) -> None:
    # ``args`` includes the leading ``run`` verb (kept by the dispatcher),
    # so the stored run_args is the complete received argument list.
    name = None
    i = 0
    while i < len(args):
        if args[i] == "--name" and i + 1 < len(args):
            name = args[i + 1]
            i += 2
            continue
        i += 1
    if name is None:
        name = "auto-%s" % _fake_id(json.dumps(args))
    entry = {
        "id": _fake_id(name),
        "status": "running",
        "exit_code": None,
        "run_args": args,
        "logs": [f"fake-podman started container {name}"],
    }
    state[name] = entry
    _save(path, state)
    sys.stdout.write(entry["id"] + "\n")


def cmd_stop(args: list[str], path: str, state: dict) -> None:
    name = _name_after(args)
    entry = state.get(name)
    if entry is None:
        _missing(name)
    entry["status"] = "exited"
    entry["exit_code"] = 0
    entry["logs"].append(f"fake-podman stopped container {name}")
    _save(path, state)


def cmd_inspect(args: list[str], path: str, state: dict) -> None:
    name = _name_after(args)
    entry = state.get(name)
    if entry is None:
        _missing(name)
    state: dict = {"Running": entry["status"] == "running"}
    if entry["exit_code"] is not None:
        state["ExitCode"] = entry["exit_code"]
    record = {"Id": entry["id"], "Name": name, "State": state}
    sys.stdout.write(json.dumps([record]) + "\n")


def cmd_logs(args: list[str], path: str, state: dict) -> None:
    name = _name_after(args)
    entry = state.get(name)
    if entry is None:
        _missing(name)
    tail = None
    i = 0
    while i < len(args):
        if args[i] in ("--tail", "-t") and i + 1 < len(args):
            try:
                tail = int(args[i + 1])
            except ValueError:
                tail = None
            i += 2
            continue
        i += 1
    lines = entry["logs"]
    if tail is not None and tail >= 0:
        lines = lines[-tail:]
    sys.stdout.write("\n".join(lines) + ("\n" if lines else ""))


def cmd_events(args: list[str], path: str, state: dict) -> None:
    # Emit a short, deterministic, non-blocking event stream: one
    # start event per container known to the state file, and a die
    # event for containers that are not running.
    for name, entry in sorted(state.items()):
        event = {
            "Type": "container",
            "Action": "start",
            "id": entry["id"],
            "name": name,
        }
        sys.stdout.write(json.dumps(event) + "\n")
        if entry["status"] != "running":
            die = {
                "Type": "container",
                "Action": "die",
                "id": entry["id"],
                "name": name,
                "exitCode": entry["exit_code"],
            }
            sys.stdout.write(json.dumps(die) + "\n")
    sys.stdout.flush()


def cmd_ps(args: list[str], path: str, state: dict) -> None:
    for name in sorted(state):
        if state[name]["status"] == "running":
            sys.stdout.write(f"{state[name]['id']}\t{name}\n")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        sys.stderr.write("Error: no subcommand given\n")
        return 125
    path = _state_path()
    state = _load(path)
    cmd = args[0]
    rest = args[1:]
    dispatch = {
        "run": cmd_run,
        "stop": cmd_stop,
        "inspect": cmd_inspect,
        "logs": cmd_logs,
        "events": cmd_events,
        "ps": cmd_ps,
    }
    handler = dispatch.get(cmd)
    if handler is None:
        sys.stderr.write(f'Error: unknown subcommand: {cmd}\n')
        return 125
    handler(args, path, state)  # store the full subcommand incl. verb
    return 0


if __name__ == "__main__":
    sys.exit(main())
