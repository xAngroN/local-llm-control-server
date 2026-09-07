"""Command line interface for llamactl.

``llamactl serve`` starts the API server.  Every other subcommand is a thin
HTTP client of that server (``http://127.0.0.1:$LLAMACTL_PORT``, default
8081) and never touches the lifecycle logic itself: the server is the only
supervisor of the container.  The single exception is ``logs``, which falls
back to reading the container log directly through the podman wrapper when
the API server is not reachable.

Exit codes: 0 on success, 1 on API error or unreachable server,
2 on incorrect usage.
"""

import argparse
import json
import os
import sys

DEFAULT_BIND = "0.0.0.0"
DEFAULT_PORT = "8081"
DEFAULT_PROFILE = "safe"
DEFAULT_LOG_TAIL = 50
TIMEOUT_SECONDS = 10.0

UNREACHABLE_HINT = (
    "llamactl: API-Server nicht erreichbar unter {url} -- "
    "siehe `systemctl --user status llamactl`"
)


def _server_url() -> str:
    """Base URL of the local API server from $LLAMACTL_PORT."""
    port = os.environ.get("LLAMACTL_PORT", DEFAULT_PORT)
    return f"http://127.0.0.1:{port}"


def _client(timeout: float = TIMEOUT_SECONDS):
    """Return an httpx client pointed at the local API server."""
    import httpx

    return httpx.Client(base_url=_server_url(), timeout=timeout)


def _print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def _print_key_value(payload: dict, indent: int = 0) -> None:
    """Render a mapping as ``key value`` lines (one value per line).

    Nested mappings (as returned by ``/profiles`` per profile and by
    ``/metrics`` for ``vram``) are printed as a ``key:`` header followed
    by their entries indented two spaces, instead of dumping a raw Python
    ``dict`` repr on one line.
    """
    pad = "  " * indent
    for key, value in payload.items():
        if isinstance(value, dict):
            print(f"{pad}{key}:")
            _print_key_value(value, indent + 1)
        else:
            print(f"{pad}{key} {value}")


def _print_lines(lines: list[str]) -> None:
    for line in lines:
        print(line)


def _call(command, method: str, path: str, payload: dict | None = None) -> int:
    """Run one API call, print the result and return the exit code.

    ``command`` must expose ``json`` (raw output mode) or ``text``
    (human-readable mode).
    """
    try:
        with _client() as client:
            response = client.request(method, path, json=payload)
    except Exception as err:  # connection refused, timeout, ...
        print(UNREACHABLE_HINT.format(url=_server_url()), file=sys.stderr)
        print(f"llamactl: {err}", file=sys.stderr)
        return 1
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text
        print(f"llamactl: API-Fehler {response.status_code}: {detail}",
              file=sys.stderr)
        return 1
    if command.json:
        _print_json(response.json())
    else:
        _print_key_value(response.json())
    return 0


def _cmd_profiles(args: argparse.Namespace) -> int:
    name = getattr(args, "name", None)
    if name:
        return _call(args, "GET", f"/profiles/{name}")
    return _call(args, "GET", "/profiles")


def _profile_payload(args: argparse.Namespace) -> dict | None:
    """Parse the profile fields for create/update from --json or stdin.

    Returns the parsed mapping, or ``None`` (after printing an error) when
    the input is missing or not a JSON object.
    """
    raw = args.fields
    if raw is None and not sys.stdin.isatty():
        raw = sys.stdin.read()
    if not raw:
        print("llamactl: Profilfelder fehlen (--fields '<obj>' oder via stdin)",
              file=sys.stderr)
        return None
    try:
        data = json.loads(raw)
    except ValueError as err:
        print(f"llamactl: ungültiges JSON: {err}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print("llamactl: Profilfelder müssen ein JSON-Objekt sein",
              file=sys.stderr)
        return None
    return data


def _cmd_profile_create(args: argparse.Namespace) -> int:
    payload = _profile_payload(args)
    if payload is None:
        return 2
    payload["name"] = args.name  # path/name wins over any name in the body
    return _call(args, "POST", "/profiles", payload)


def _cmd_profile_update(args: argparse.Namespace) -> int:
    payload = _profile_payload(args)
    if payload is None:
        return 2
    payload.pop("name", None)
    return _call(args, "PUT", f"/profiles/{args.name}", payload)


def _cmd_profile_delete(args: argparse.Namespace) -> int:
    return _call(args, "DELETE", f"/profiles/{args.name}")


def _cmd_start(args: argparse.Namespace) -> int:
    profile = args.profile or DEFAULT_PROFILE
    return _call(args, "POST", "/start", {"profile": profile})


def _cmd_stop(args: argparse.Namespace) -> int:
    return _call(args, "POST", "/stop")


def _cmd_restart(args: argparse.Namespace) -> int:
    return _call(args, "POST", "/reload", {"profile": args.profile})


def _cmd_status(args: argparse.Namespace) -> int:
    return _call(args, "GET", "/status")


def _cmd_metrics(args: argparse.Namespace) -> int:
    return _call(args, "GET", "/metrics")


def _cmd_suspend(args: argparse.Namespace) -> int:
    return _call(args, "POST", "/suspend")


def _cmd_logs(args: argparse.Namespace) -> int:
    """Show the last container log lines.

    Reads through the API server first; when the server is not reachable,
    fall back to the podman wrapper directly (the only allowed exception
    to the thin-client rule).
    """
    try:
        with _client() as client:
            response = client.get("/logs", params={"tail": args.tail})
    except Exception as err:
        try:
            from llamactl.config import load_config
            from llamactl.podman import Podman

            common, _profiles = load_config()
            lines = Podman().logs(common.container_name, tail=args.tail)
        except Exception as fallback_err:
            print(UNREACHABLE_HINT.format(url=_server_url()), file=sys.stderr)
            print(f"llamactl: {err}", file=sys.stderr)
            print(f"llamactl: Podman-Fallback fehlgeschlagen: {fallback_err}",
                  file=sys.stderr)
            return 1
        if args.json:
            _print_json(lines)
        else:
            _print_lines(lines)
        return 0
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text
        print(f"llamactl: API-Fehler {response.status_code}: {detail}",
              file=sys.stderr)
        return 1
    if args.json:
        _print_json(response.json())
    else:
        body = response.json()
        lines = body.get("lines", body) if isinstance(body, dict) else body
        _print_lines(lines)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from llamactl.api import create_app

    bind = os.environ.get("LLAMACTL_BIND", DEFAULT_BIND)
    port = int(os.environ.get("LLAMACTL_PORT", DEFAULT_PORT))
    # Pass the application object (not a dotted path) so uvicorn reuses
    # the lifespan-built manager and does not re-import the module.
    uvicorn.run(create_app(), host=bind, port=port)
    return 0


class _Parser(argparse.ArgumentParser):
    """ArgumentParser that returns exit code 2 instead of raising SystemExit."""

    def error(self, message: str):  # type: ignore[override]
        self.print_usage(sys.stderr)
        raise SystemExit(f"{self.prog}: error: {message}\n")


def build_parser() -> _Parser:
    """Build the argument parser with all subcommands."""
    parser = _Parser(
        prog="llamactl",
        description=(
            "Steuerung der lokalen LLM-Infrastruktur. Alle Subkommandos "
            "außer 'serve' sprechen den API-Server unter "
            "http://127.0.0.1:$LLAMACTL_PORT (Default 8081) an."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_json_flag(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--json",
            action="store_true",
            dest="json",
            help="Rohantwort (JSON) statt menschenlesbarer Ausgabe",
        )

    serve = sub.add_parser("serve", help="Starte den API-Server")
    serve.set_defaults(func=_cmd_serve)

    profiles = sub.add_parser(
        "profiles",
        help="Profile anzeigen (Sizing + effektives Tuning: -b/-ub, -ngl, "
        "-fa, cont-batching, cache-reuse, MTP/spec). Mit NAME nur ein Profil.",
    )
    profiles.add_argument(
        "name",
        nargs="?",
        help="Optionaler Profilname; zeigt nur dieses Profil (GET /profiles/<name>)",
    )
    profiles.set_defaults(func=_cmd_profiles)
    add_json_flag(profiles)

    def add_fields_flag(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--fields",
            default=None,
            help="Profilfelder als JSON-Objekt (sonst von stdin gelesen), "
            'z. B. \'{"model":"m.gguf","ctx_size":4096,'
            '"kv_cache_type_k":"q8_0","kv_cache_type_v":"q8_0",'
            '"parallel":1,"batch_size":512}\'',
        )

    profile_create = sub.add_parser(
        "profile-create", help="Neues Profil anlegen (POST /profiles)"
    )
    profile_create.add_argument("name", help="Name des neuen Profils")
    add_fields_flag(profile_create)
    profile_create.set_defaults(func=_cmd_profile_create)
    add_json_flag(profile_create)

    profile_update = sub.add_parser(
        "profile-update", help="Profil ändern (PUT /profiles/<name>)"
    )
    profile_update.add_argument("name", help="Name des Profils")
    add_fields_flag(profile_update)
    profile_update.set_defaults(func=_cmd_profile_update)
    add_json_flag(profile_update)

    profile_delete = sub.add_parser(
        "profile-delete", help="Profil löschen (DELETE /profiles/<name>)"
    )
    profile_delete.add_argument("name", help="Name des Profils")
    profile_delete.set_defaults(func=_cmd_profile_delete)
    add_json_flag(profile_delete)

    start = sub.add_parser(
        "start", help="Starte die Instanz (Default-Profil: 'safe')"
    )
    start.add_argument(
        "profile", nargs="?", default=None,
        help="Profilname (Default: %s)" % DEFAULT_PROFILE,
    )
    start.set_defaults(func=_cmd_start)
    add_json_flag(start)

    stop = sub.add_parser("stop", help="Stoppe die laufende Instanz")
    stop.set_defaults(func=_cmd_stop)
    add_json_flag(stop)

    restart = sub.add_parser(
        "restart", help="Starte die Instanz mit anderem Profil neu (Reload)"
    )
    restart.add_argument("profile", help="Profilname")
    restart.set_defaults(func=_cmd_restart)
    add_json_flag(restart)

    status = sub.add_parser("status", help="Zeige den aktuellen Status")
    status.set_defaults(func=_cmd_status)
    add_json_flag(status)

    metrics = sub.add_parser(
        "metrics", help="Zeige VRAM- und Modell-Metriken"
    )
    metrics.set_defaults(func=_cmd_metrics)
    add_json_flag(metrics)

    logs = sub.add_parser(
        "logs", help="Zeige die letzten Logzeilen des Containers"
    )
    logs.add_argument(
        "--tail", type=int, default=DEFAULT_LOG_TAIL, metavar="N",
        help=f"Anzahl der letzten Zeilen (Default: {DEFAULT_LOG_TAIL})",
    )
    logs.set_defaults(func=_cmd_logs)
    add_json_flag(logs)

    suspend = sub.add_parser(
        "suspend", help="Suspendiere den Host (Suspend-to-RAM)"
    )
    suspend.set_defaults(func=_cmd_suspend)
    add_json_flag(suspend)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``llamactl`` console script."""
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as err:
        # argparse errors (unknown subcommand, missing args) exit with 2.
        return err.code if isinstance(err.code, int) else 2
    return args.func(args)
