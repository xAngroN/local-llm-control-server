#!/usr/bin/env bash
# verify-boot-readiness.sh — prüft, ob der Host nach dem Boot ohne
# vorherige Anmeldung betriebsbereit ist.
#
# Führt genau drei Prüfungen nacheinander aus:
#   1. Linger ist für den aktuellen Benutzer aktiviert (Linger=yes)
#   2. Der Benutzerdienst llamactl ist aktiv
#   3. Der Health-Endpoint auf localhost:8081 antwortet
#
# Jede Prüfung gibt eine Zeile mit OK oder FEHLER aus. Bei der ersten
# fehlgeschlagenen Prüfung endet das Skript mit Rückgabewert 1.
# Es installiert, aktiviert und startet nichts — es prüft nur.
set -euo pipefail

# Fehlende Werkzeuge sind ein Prüfschaden, kein Systemfehler.
for tool in loginctl systemctl curl; do
    if ! command -v "$tool" > /dev/null 2>&1; then
        echo "FEHLER: Werkzeug nicht installiert: $tool"
        exit 1
    fi
done

BENUTZER="${USER:-$(id -un)}"

# 1. Linger-Status
linger=$(loginctl show-user "$BENUTZER" --property=Linger 2>/dev/null || true)
if [ "$linger" = "Linger=yes" ]; then
    echo "OK: Linger aktiv für $BENUTZER"
else
    echo "FEHLER: Linger nicht aktiv für $BENUTZER ($linger)"
    exit 1
fi

# 2. Unit-Status
state=$(systemctl --user show -p ActiveState --value llamactl 2>/dev/null || true)
if [ "$state" = "active" ]; then
    echo "OK: llamactl-Dienst aktiv"
else
    echo "FEHLER: llamactl-Dienst nicht aktiv ($state)"
    exit 1
fi

# 3. HTTP-Erreichbarkeit
if curl -s --fail localhost:8081/healthz > /dev/null; then
    echo "OK: Health-Endpoint antwortet"
else
    echo "FEHLER: Health-Endpoint nicht erreichbar"
    exit 1
fi
