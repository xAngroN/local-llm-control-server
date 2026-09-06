"""FastAPI application for the llamactl control server."""

from fastapi import FastAPI

from llamactl import __version__

app = FastAPI(title="llamactl")


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe."""
    return {"status": "ok", "version": __version__}
