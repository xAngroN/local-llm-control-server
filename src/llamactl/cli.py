"""Command line interface for llamactl."""

import os
import sys

DEFAULT_BIND = "0.0.0.0"
DEFAULT_PORT = "8081"


def _parse_args(argv: list[str] | None) -> list[str]:
    if argv is None:
        argv = sys.argv[1:]
    return argv


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``llamactl`` console script."""
    args = _parse_args(argv)

    if not args or args[0] != "serve":
        print(
            "usage: llamactl serve\n"
            "  binds to $LLAMACTL_BIND (default 0.0.0.0) on "
            "$LLAMACTL_PORT (default 8081)",
            file=sys.stderr,
        )
        return 2

    import uvicorn

    from llamactl.api import create_app

    bind = os.environ.get("LLAMACTL_BIND", DEFAULT_BIND)
    port = int(os.environ.get("LLAMACTL_PORT", DEFAULT_PORT))
    # Pass the application object (not a dotted path) so uvicorn reuses
    # the lifespan-built manager and does not re-import the module.
    uvicorn.run(create_app(), host=bind, port=port)
    return 0
