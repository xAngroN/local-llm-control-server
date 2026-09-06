# Validierung: Parallelbetrieb mit dem Altbestand

Runbook für den Parallelbetrieb des neuen API-Servers neben der bestehenden
Quadlet-Unit `llamacpp-qwen38.service` (Altskript `qwen`) während der
Validierungsphase. Ziel: nachweisen, dass sich die drei portierten Profile
(`fast`, `large`, `safe`) genauso verhalten wie unter dem alten `qwen`-Skript —
bevor irgendetwas abgeschaltet wird. Der `shared`-Profil ist **nicht** Teil
dieser Gleichwertigkeitsprüfung, weil es im Altbestand keine Entsprechung hat.

## Regeln des Parallelbetriebs

1. **Der neue API-Server läuft auf Port 8081** (Default, überschreibbar via
   `LLAMACTL_PORT`). Die bestehende Unit `llamacpp-qwen38.service` und das alte
   `qwen`-Skript bleiben davon unberührt — sie dürfen in dieser Phase weder
   gestartet noch gestoppt werden (außer man führt die Prüfung manuell durch,
   siehe unten).
2. **Der vom neuen Server gestartete Modellcontainer muss einen anderen Hostport
   als der Altbestand verwenden.** Der einzige Hebel dafür ist die
   Konfigurationsdatei `config/profiles.toml` — es wird dafür **kein
   Sonderfall in den Code** eingebaut. Im `[common]`-Block:
   - `host_port = 8082` (Altbestand läuft auf 8080),
   - `container_name = "llamactl-model-parity"` (eigener Name, damit sich die
     beiden Container nicht gegenseitig verdrängen — derselbe Name würde beim
     Start den jeweils anderen Container löschen).
3. **VRAM-Limit: nie beide Modellinstanzen gleichzeitig.** Die GPU hat rund
   19,98 GiB VRAM, die Profile belegen 18,17–18,99 GiB. Deshalb darf **immer
   nur eine der beiden Instanzen** (Altbestand `llamacpp-qwen38` **oder** neue
   Parity-Instanz) gleichzeitig laufen. Vor jeder Prüfung:
   `systemctl status llamacpp-qwen38.service` → darf **nicht** `active (running)`
   sein.

> Wichtig: Der Parallelbetrieb betrifft nur die *Steuerung* (API-Server auf
> 8081 neben der Unit). Die *Modellinstanzen* selbst laufen nie parallel.

## Prüfzeilen (neue CLI ↔ altes `qwen`-Skript)

Die CLI ist ein dünner Client des API-Servers auf `http://127.0.0.1:8081`;
`--json` gibt die Rohantwort aus, ohne `--json` eine Zeile pro Wert.
Rückgabewerte: `0` Erfolg, `1` API-Fehler/Server nicht erreichbar, `2`
falsche Benutzung.

| Befehl | Neue Prüfzeile | Erwartetes Ergebnis | Altes `qwen`-Entsprechung |
| --- | --- | --- | --- |
| start (fast) | `llamactl start fast` | Exit 0; Status wechselt von `stopped` über `starting`/`loading` in `ready`; Modell-Container `llamactl-model-parity` auf Hostport 8082 lauscht | `qwen start fast` |
| start (large) | `llamactl start large` | Exit 0; wie oben, `ready` nach Ladezeit des q8_0-Modells (65536 Ctx) | `qwen start large` |
| start (safe) | `llamactl start safe` | Exit 0; `ready` im niedrigsten VRAM-Modus (Default-Profil) | `qwen start safe` |
| stop | `llamactl stop` | Exit 0; Status wird `stopped`, `exit_code 0`; Container wird entfernt (idempotent, auch ohne laufende Instanz kein Fehler) | `qwen stop` |
| reload | `llamactl restart safe` (alias für Reload) | Exit 0; Instanz wird mit dem neuen Profil neu gestartet; Status endet in `loading` und danach in `ready` | `qwen reload safe` (bzw. `qwen stop && qwen start safe`) |
| status | `llamactl status` | Exit 0; Zeilen `state ready`, `profile safe`, `container_id …`, `since …` (ISO-Zeitstempel) | `qwen status` (bzw. `podman ps` gegen den Altbestand) |
| metrics | `llamactl metrics` | Exit 0; `state ready`, `profile …`, `model …`, `prompt_tps`/`generation_tps` vorhanden, `vram.used_gib` ≤ 18,99 (im Budget von 19,98 GiB) | `qwen metrics` (bzw. `curl -s localhost:8080/metrics` im Altbestand) |

Hinweis zur reload-Prüfung: `llamactl restart <profil>` ist der Reload-Befehl
(es gibt kein Subkommando namens `reload`); die CLI trifft damit `POST /reload`.

## Automatisierte Gleichwertigkeitsprüfung

Das Skript `scripts/validate-parity.sh` führt für jedes der drei portierten
Profile (`fast`, `large`, `safe`) automatisch die Abfolge

```
start → auf ready warten → status prüfen → metrics prüfen → stop → status prüfen
```

durch, protokolliert jede einzelne Prüfung als `OK` oder `FEHLER` und gibt am
Ende eine Zusammenfassung aus. Rückgabewert `0` **nur** bei vollständigem
Erfolg; bei einem oder mehreren Fehlern `1`.

Voraussetzungen für den Lauf:

- API-Server läuft: `systemctl --user status llamactl` → `active (running)`,
  `curl -s localhost:8081/healthz` → `{"status":"ok",...}`
- Altbestand ist gestoppt: `systemctl status llamacpp-qwen38.service` →
  **nicht** `active (running)` (VRAM-Regel, siehe oben).
- `config/profiles.toml` ist auf `host_port = 8082` und
  `container_name = "llamactl-model-parity"` gesetzt.

Aufruf:

```sh
scripts/validate-parity.sh
```

Das Skript ist rein prüfend: Es berührt die alte Unit nicht (kein
Start/Stop/Reload von `llamacpp-qwen38.service`) und verändert keine Dateien.
