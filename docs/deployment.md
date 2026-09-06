# Deployment auf dem Bazzite-Host

Diese Anleitung installiert den `llamactl`-API-Server als `systemd --user`-
Dienst auf einem Bazzite-Host.

## Voraussetzungen

- Das `llamactl`-Repository ist lokal geklont.
- `python3` mit `python3-venv` ist installiert (standardmäßig vorhanden).

Auf Bazzite ist `/usr` unveränderlich (rpm-ostree), während `/var/home`
beliebig beschreibbar ist. Deshalb liegen venv, Binary und Konfiguration
alle im Home-Verzeichnis des Benutzers. Es wird **keine systemweite Unit**
angelegt, und es werden **keine Root-Rechte (`sudo`) benötigt** — nichts
muss in `/usr` oder `/etc` geschrieben werden, und es ist **keine
`rpm-ostree`-Layerung** erforderlich, weil alles unterhalb von
`/var/home` liegt. Der API-Server ist bewusst kein Container und wird auch
nicht als Quadlet ausgeliefert.

> Hinweis: Die Unit in dieser Anleitung ist die des **API-Servers**. Die
> bestehende Modell-Unit `llamacpp-qwen38.service` wird hier nicht
> berührt — sie wird erst in der separaten Cutover-Aufgabe geändert.

## 1. venv anlegen

```sh
mkdir -p ~/.local/share/llamactl
python3 -m venv ~/.local/share/llamactl/venv
```

## 2. Paket installieren

```sh
~/.local/share/llamactl/venv/bin/pip install -e <pfad-zum-repository>
```

Dabei ist `<pfad-zum-repository>` der lokale Pfad des geklonten
`llamactl`-Repositories (das Editable-Install verlinkt nur nach
`~/.local/share/llamactl/venv`, schreibt also nichts nach `/usr`).

Prüfen, dass der Einstiegspunkt vorhanden ist:

```sh
~/.local/share/llamactl/venv/bin/llamactl --help
```

## 3. Konfiguration kopieren

```sh
mkdir -p ~/.config/llamactl
cp <pfad-zum-repository>/config/profiles.toml ~/.config/llamactl/profiles.toml
```

## 4. Unit installieren

```sh
mkdir -p ~/.config/systemd/user
cp <pfad-zum-repository>/deploy/llamactl.service ~/.config/systemd/user/llamactl.service
```

## 5. Dienst aktivieren und starten

```sh
systemctl --user daemon-reload
systemctl --user enable --now llamactl
```

## 6. Prüfschritte

Status prüfen:

```sh
systemctl --user status llamactl
```

Health-Endpoint prüfen — erwartete Antwort
`{"status":"ok","version":"..."}`:

```sh
curl -s localhost:8081/healthz
```

## 7. Neustart und erneute Prüfung

```sh
systemctl --user restart llamactl
systemctl --user status llamactl
curl -s localhost:8081/healthz
```

Beide Befehle müssen erneut erfolgreich sein: `active (running)` und
`"status":"ok"`.
