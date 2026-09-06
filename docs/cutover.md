# Cutover: Umstellung vom Altbestand auf den API-Server

Runbook für die dauerhafte Umstellung der llama.cpp-Instanz von der alten
Quadlet-Unit `llamacpp-qwen38.service` (gesteuert über das `qwen`-Skript) auf
den `llamactl`-API-Server. Dieses ist der **einzige** Schritt mit dauerhafter
Wirkung auf den Altbestand — er wird deshalb nur manuell nachvollzogen
ausgeführt, nicht automatisiert. Kein Skript dieses Projekts darf hier
selbsttätig etwas abschalten; alle Befehle werden von der ausführenden Person
gesetzt und anschließend geprüft.

Voraussetzung: Die Gleichwertigkeitsprüfung des Parallelbetriebs
(siehe [`validation.md`](validation.md)) ist bestanden.

## 1. Voraussetzungen (vor dem ersten Abschaltschritt)

Alle drei Punkte müssen erfüllt sein, bevor irgendetwas gestoppt wird:

1. **Gleichwertigkeitsprüfung bestanden**

   ```sh
   scripts/validate-parity.sh
   ```

   Prüfung: Exit-Code `0` und in der Ausgabe keine Zeile mit `FEHLER`
   (Zusammenfassung „Alle Prüfungen erfolgreich.“). Das Skript ist rein
   prüfend; es verändert nichts.

2. **Boot-Prüfung bestanden**

   ```sh
   scripts/verify-boot-readiness.sh
   ```

   Prüfung: drei `OK`-Zeilen (Linger, `llamactl`-Dienst, Health-Endpoint)
   und Exit-Code `0`. Damit ist sichergestellt, dass nach dem Cutover der
   API-Server ohne vorherige Anmeldung bootet (Linger, `deploy.md` §8).

3. **`shared`-Profil vermessen**

   Laut [`profiles.md`](profiles.md) ist der VRAM-Spitzenwert des Profils
   `shared` **noch nicht gemessen** und darf nicht als sicher behandelt
   werden.

   ```sh
   DURATION=120 ./scripts/benchmark-vram.sh shared
   ```

   Prüfung: Der gemessene Spitzenwert liegt innerhalb des VRAM-Budgets von
   19,98 GiB (bekannte Werte: 18,17–18,99 GiB) und wird in
   `docs/profiles.md` eingetragen.

## 2. Modellport zurücksetzen

Während des Parallelbetriebs war der Modellport in
`~/.config/llamactl/profiles.toml` auf `host_port = 8082` verschoben (und
`container_name = "llamactl-model-parity"` gesetzt), weil der Altbestand auf
Port 8080 lief. Nach dem Cutover ist der Altbestand weg — der neue
Modellserver läuft wieder auf dem regulären Port.

```sh
# in ~/.config/llamactl/profiles.toml, [common]:
#   host_port = 8080
#   container_name = "llamactl-model"
```

Prüfung:

```sh
grep -n 'host_port\|container_name' ~/.config/llamactl/profiles.toml
```

muss `host_port = 8080` und `container_name = "llamactl-model"` zeigen.
Anschließend den API-Server neu starten, damit die Konfiguration greift:

```sh
systemctl --user restart llamactl
curl -s localhost:8081/healthz
```

Prüfung: `active (running)` bzw. `{"status":"ok",...}`.

> Nur nötig, wenn der Port für den Parallelbetrieb tatsächlich verschoben
> war; ist `host_port` bereits 8080, diesen Schritt überspringen.

## 3. Alte Unit stoppen, deaktivieren und entfernen

Vorher sicherstellen, dass keine Modellinstanz läuft (VRAM-Regel):

```sh
systemctl status llamacpp-qwen38.service   # darf nicht active (running) sein
```

Stopp:

```sh
systemctl --user stop llamacpp-qwen38.service
```

Prüfung:

```sh
systemctl status llamacpp-qwen38.service
```

muss `inactive (dead)` (oder `inactive (exited)`) melden.

Deaktivierung:

```sh
systemctl --user disable llamacpp-qwen38.service
```

Prüfung: `systemctl status llamacpp-qwen38.service` zeigt unter *Loaded*
`enabled` **nicht** mehr (die Symlinks werden entfernt).

Quadlet-Datei entfernen. Die zugehörige Datei ist die `.container`-Datei, aus
der die Unit generiert wurde:

```sh
# Ort der Quadlet-Datei ermitteln:
systemctl --user show llamacpp-qwen38.container --property=FragmentPath

# Datei entfernen (typischer Pfad):
rm ~/.config/containers/systemd/llamacpp-qwen38.container
```

Prüfung: Die Datei existiert nicht mehr und die Unit ist aus dem View
verschwunden:

```sh
systemctl --user list-unit-files | grep llamacpp-qwen38   # keine Treffer
```

Daemon neu laden:

```sh
systemctl --user daemon-reload
```

Prüfung:

```sh
systemctl --user status llamactl
```

ist weiterhin `active (running)` — der API-Server ist von dem Reload nicht
betroffen.

## 4. `qwen`-Skript ausmustern

Das `qwen`-Skript wird **verschoben, nicht gelöscht**, damit die Rücknahme
(§6) möglich bleibt. Es liegt nicht in diesem Repository, sondern im
Home-Verzeichnis des Host-Benutzers (z. B. `~/.local/bin/qwen`).

```sh
# Ablageort anlegen und Skript verschieben
mkdir -p ~/.retired
mv ~/.local/bin/qwen ~/.retired/qwen.disabled
```

Prüfung:

```sh
command -v qwen     # liefert nichts
ls -l ~/.retired/qwen.disabled
```

Das `qwen`-Skript wird durch die `llamactl`-CLI ersetzt. Gegenüberstellung
aller alten Kommandos:

| Altes `qwen`-Kommando | Neues `llamactl`-Äquivalent |
| --- | --- |
| `qwen start large` | `llamactl start large` |
| `qwen start fast` | `llamactl start fast` |
| `qwen start safe` (Default) | `llamactl start safe` (bzw. `llamactl start` ohne Argument) |
| `qwen stop` | `llamactl stop` |
| `qwen restart large` (bzw. `qwen stop && qwen start large`) | `llamactl restart large` |
| `qwen status` | `llamactl status` |
| `qwen profiles` | `llamactl profiles` |
| `qwen logs` | `llamactl logs` (ggf. `llamactl logs --tail 200`) |
| `qwen metrics` (bzw. `curl -s localhost:8080/metrics`) | `llamactl metrics` |

Zusätzlich, nur vom API-Server verfügbar: `llamactl serve` (Server starten)
und `llamactl suspend` (Host in Suspend-to-RAM). Die CLI ist ein dünner
Client des API-Servers auf `http://127.0.0.1:8081`; Rückgabewerte `0`/`1`/`2`
wie in der `README.md` beschrieben.

## 5. Abschlussprüfung

Nach den Schritten 2–4 gilt der Cutover als abgeschlossen, wenn alle drei
Prüfungen stimmen:

1. **Nur der API-Server verwaltet den Container.**

   ```sh
   podman ps
   ```

   zeigt **genau einen** vom API-Server verwalteten llama.cpp-Container
   (Name laut `container_name`, Default `llamactl-model`) **oder keinen**
   (wenn gerade keine Instanz läuft). In keinem Fall darf ein Container
   gehören, den die alte Unit gestartet hätte.

2. **`llamactl status` stimmt damit überein.**

   ```sh
   llamactl status
   ```

   meldet `state ready` mit `container_id …`, falls `podman ps` einen
   Container zeigt, sonst `state stopped`. Beide Meldungen müssen
   konsistent sein.

3. **Die alte Unit ist verschwunden.**

   ```sh
   systemctl --user list-units | grep llamacpp-qwen38
   ```

   liefert **keine** Treffer.

## 6. Rücknahme

Im Fehlerfall (API-Server verhält sich dauerhaft falsch, Profil-Problem,
das nicht neu lösbar ist) wird die alte Unit wieder aktiviert:

1. Die Quadlet-Datei wiederherstellen. Sie ist die generierende Quelle von
   `llamacpp-qwen38.service`; liegt sie nicht mehr vor, aus der Sicherung
   bzw. aus der ursprünglichen Installation zurückkopieren:

   ```sh
   mkdir -p ~/.config/containers/systemd
   cp <sicherung>/llamacpp-qwen38.container ~/.config/containers/systemd/llamacpp-qwen38.container
   ```

2. Daemon neu laden und Unit aktivieren und starten:

   ```sh
   systemctl --user daemon-reload
   systemctl --user enable --now llamacpp-qwen38.service
   ```

3. Prüfung:

   ```sh
   systemctl status llamacpp-qwen38.service   # active (running)
   curl -s localhost:8080/health   # Modellserver antwortet
   ```

4. Das `qwen`-Skript zurückverschieben:

   ```sh
   mv ~/.retired/qwen.disabled ~/.local/bin/qwen
   chmod +x ~/.local/bin/qwen
   ```

5. Falls in Schritt 2 der Modellport auf 8080 gesetzt wurde: Für den
   Parallelbetrieb mit dem Altbestand erneut `host_port = 8082` und
   `container_name = "llamactl-model-parity"` in
   `~/.config/llamactl/profiles.toml` setzen und `systemctl --user restart
   llamactl` ausführen (VRAM-Regel: nie beide Modellinstanzen gleichzeitig).

## Hinweis zum `ki`-Skript

Das `ki`-Skript wird in dieser Änderung **nicht** angefasst. Es startet heute
`llamacpp-qwen38` über `systemctl --user start` und zeigt damit weiterhin auf
den alten Mechanismus. Nach dem Cutover **läuft dieser Aufruf ins Leere**,
weil die Unit entfernt wurde — `ki` bleibt bis auf diese Auswirkung
unverändert. Die Umstellung von `ki` auf die neue API (`llamactl`) ist eine
**eigene Folgeänderung** und gehört nicht in dieses Runbook.

