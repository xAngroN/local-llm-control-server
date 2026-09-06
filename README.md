# local-llm-control-server

Kleiner API-Server (FastAPI), der die lokale LLM-Infrastruktur (llama.cpp in
einem Podman-Container auf Port 8080) steuert. Dieser Server selbst lauscht auf
`0.0.0.0:8081` (Default, via `LLAMACTL_BIND` / `LLAMACTL_PORT` überschreibbar)
und benötigt weder Root-Rechte noch Podman-Socket.

## Installation

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
```

## Start

```sh
llamactl serve
# oder: python -m llamactl serve
# oder: uvicorn llamactl.api:app --host 0.0.0.0 --port 8081
```

## Endpunkte

- `GET /healthz` -> `{"status": "ok", "version": "0.1.0"}`

## Tests

```sh
pytest
```
