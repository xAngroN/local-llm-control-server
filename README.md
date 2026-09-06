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

## Befehle

Die CLI ist ein dünner Client des API-Servers: alle Subkommandos außer
`serve` sprechen `http://127.0.0.1:$LLAMACTL_PORT` (Default 8081) an.
Rückgabewerte: `0` Erfolg, `1` API-Fehler/Server nicht erreichbar (Meldung auf
stderr), `2` falsche Benutzung. `--json` gibt die Rohantwort aus, standardmäßig
wird eine Zeile pro Wert ausgegeben.

| Befehl | Beschreibung |
| --- | --- |
| `llamactl serve` | Startet den API-Server (Bind/Port via `LLAMACTL_BIND` / `LLAMACTL_PORT`). |
| `llamactl profiles` | Listet Profilnamen mit `ctx_size`, `parallel` und Slot-Kontext. |
| `llamactl start [profil]` | Startet die Instanz; ohne Argument wird das Profil `safe` verwendet. |
| `llamactl stop` | Stoppt die laufende Instanz. |
| `llamactl restart <profil>` | Alias für Reload: startet die Instanz mit anderem Profil neu. |
| `llamactl status` | Zeigt den aktuellen Status der Instanz. |
| `llamactl metrics` | Zeigt VRAM- und Modell-Metriken. |
| `llamactl logs [--tail N]` | Letzte Logzeilen des Containers; fällt auf den Podman-Wrapper zurück, wenn der Server nicht erreichbar ist. |
| `llamactl suspend` | Suspendiert den Host (Suspend-to-RAM). |

Ist der API-Server nicht erreichbar, meldet die CLI das und verweist auf
`systemctl --user status llamactl`.

## Endpunkte

- `GET /healthz` -> `{"status": "ok", "version": "0.1.0"}`

## Tests

```sh
pytest
```
