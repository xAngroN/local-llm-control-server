#!/usr/bin/env bash
#
# benchmark-vram.sh — misst den VRAM-Spitzenverbrauch eines Profils
#
# Startet das angegebene Profil (Default: shared), wartet auf den Zustand
# "ready" (Zeitlimit 300 s), misst den VRAM-Verbrauch in einem Intervall von
# 2 Sekunden über eine konfigurierbare Dauer (Umgebungsvariable DURATION,
# Default 120 s), gibt den Spitzenwert in GiB aus und stellt ihn gegen das
# Budget von 19,98 GiB. Am Ende wird die Instanz wieder gestoppt.
#
# Rückgabewert: 0, wenn der Spitzenwert im Budget liegt; 1, wenn der
# Spitzenwert das Budget überschreitet oder "ready" nicht erreicht wird.

set -euo pipefail

PROFILE="${1:-shared}"
DURATION="${DURATION:-120}"
INTERVAL=2
READY_TIMEOUT=300
BUDGET="19.98"

case "$DURATION" in
    ''|*[!0-9]*)
        echo "FEHLER: DURATION muss eine ganzzahlige Dauer in Sekunden sein (ist: '$DURATION')" >&2
        exit 1
        ;;
esac

stop_instance() {
    # stop ist idempotent; Fehler ignorieren, damit der Exit-Code erhalten bleibt
    llamactl stop --json >/dev/null 2>&1 || true
}
trap stop_instance EXIT

echo "Profil: $PROFILE | Messdauer: ${DURATION}s | Intervall: ${INTERVAL}s | Budget: ${BUDGET} GiB"

if ! llamactl start "$PROFILE" --json >/dev/null; then
    echo "FEHLER: Instanz konnte nicht gestartet werden (Profil: $PROFILE)" >&2
    exit 1
fi
echo "OK: Start angefordert, warte auf ready (Zeitlimit ${READY_TIMEOUT}s) ..."

ready=0
wait_start=$(date +%s)
while true; do
    if [ $(( $(date +%s) - wait_start )) -ge "$READY_TIMEOUT" ]; then
        break
    fi
    state_json=$(llamactl status --json 2>/dev/null || true)
    if printf '%s' "$state_json" | grep -q '"state"[[:space:]]*:[[:space:]]*"ready"'; then
        ready=1
        break
    fi
    sleep "$INTERVAL"
done

if [ "$ready" -ne 1 ]; then
    echo "FEHLER: Zustand 'ready' wurde innerhalb von ${READY_TIMEOUT}s nicht erreicht" >&2
    exit 1
fi
echo "OK: Instanz ist bereit, messe VRAM für ${DURATION}s ..."

peak=""
end_ts=$(( $(date +%s) + DURATION ))
while [ "$(date +%s)" -lt "$end_ts" ]; do
    metrics_json=$(llamactl metrics --json 2>/dev/null || true)
    val=$(printf '%s' "$metrics_json" \
        | grep -o '"used_gib":[[:space:]]*[-+0-9.eE]*' \
        | head -n 1 \
        | sed 's/^[^0-9.+-]*//' \
        || true)
    if [ -n "$val" ]; then
        if [ -z "$peak" ]; then
            peak="$val"
        else
            peak=$(awk -v p="$peak" -v v="$val" 'BEGIN { print (v + 0 > p + 0) ? v : p }')
        fi
    fi
    sleep "$INTERVAL"
done

if [ -z "$peak" ]; then
    echo "FEHLER: kein VRAM-Wert konnte ausgelesen werden" >&2
    exit 1
fi

peak_fmt=$(awk -v p="$peak" 'BEGIN { printf "%.2f", p + 0 }')
reserve=$(awk -v b="$BUDGET" -v p="$peak" 'BEGIN { printf "%.2f", b - p }')
over=$(awk -v b="$BUDGET" -v p="$peak" 'BEGIN { print (p + 0 > b + 0) ? 1 : 0 }')

echo "Spitzenwert: ${peak_fmt} GiB"
echo "Budget: 19,98 GiB"
echo "Verbleibende Reserve: ${reserve} GiB"

if [ "$over" -eq 1 ]; then
    echo "FEHLER: Spitzenwert ${peak_fmt} GiB überschreitet das Budget von 19,98 GiB" >&2
    exit 1
fi

echo "OK: Spitzenwert liegt innerhalb des Budgets"
exit 0
