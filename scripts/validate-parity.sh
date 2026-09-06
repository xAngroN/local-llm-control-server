#!/usr/bin/env bash
#
# validate-parity.sh — Gleichwertigkeitsprüfung der portierten Profile
#
# Durchläuft für jedes der drei portierten Profile (fast, large, safe) die
# Abfolge:
#
#   start → auf ready warten → status prüfen → metrics prüfen → stop → status prüfen
#
# Jede einzelne Prüfung wird als "OK: ..." oder "FEHLER: ..." protokolliert.
# Am Ende wird eine Zusammenfassung ausgegeben. Der Rückgabewert ist 0 nur bei
# vollständigem Erfolg, sonst 1.
#
# Das Skript ist rein prüfend: Es berührt die bestehende Unit
# llamacpp-qwen38.service weder (kein Start/Stop/Reload des Altbestands) noch
# verändert es Dateien. Der API-Server muss bereits auf Port 8081 laufen
# (systemctl --user status llamactl), und die Konfiguration muss für den
# Parallelbetrieb hergerichtet sein (host_port = 8082,
# container_name = "llamactl-model-parity" in config/profiles.toml).
#
# Voraussetzungen vor dem Aufruf:
#   - API-Server läuft:  systemctl --user status llamactl   -> active (running)
#   - Altbestand frei:   systemctl status llamacpp-qwen38.service
#                        darf NICHT "active (running)" sein (VRAM-Regel: nie
#                        beide Modellinstanzen gleichzeitig).

set -euo pipefail

PROFILES=("fast" "large" "safe")
READY_TIMEOUT=300
POLL_INTERVAL=2

ok_count=0
err_count=0

# --- Protokollierung ---------------------------------------------------------

ok() {
    # $1: Prüfpunkt, $2: Detailmeldung
    echo "OK: $1 — $2"
    ok_count=$(( ok_count + 1 ))
}

fehler() {
    # $1: Prüfpunkt, $2: Detailmeldung
    echo "FEHLER: $1 — $2"
    err_count=$(( err_count + 1 ))
}

# --- Hilfsmittel -------------------------------------------------------------

# state aus der JSON-Antwort von `llamactl status --json` extrahieren
status_state() {
    local json
    json=$(llamactl status --json 2>/dev/null || true)
    printf '%s' "$json" \
        | grep -o '"state"[[:space:]]*:[[:space:]]*"[a-z]*"' \
        | head -n 1 \
        | sed 's/.*"\(.*\)"/\1/' \
        || true
}

# Warten, bis der Status "ready" ist; gibt 0 zurück, sonst 1
wait_for_ready() {
    local deadline=$(( $(date +%s) + READY_TIMEOUT ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if [ "$(status_state)" = "ready" ]; then
            return 0
        fi
        sleep "$POLL_INTERVAL"
    done
    return 1
}

# Ein Profil durch die komplette Prüfsequenz führen
check_profile() {
    local profile="$1"
    echo ""
    echo "=== Profil: $profile ==="

    # 1) start
    if llamactl start "$profile" >/dev/null 2>&1; then
        ok "start ($profile)" "API-Antwort erhalten, Instanz wird gestartet"
    else
        fehler "start ($profile)" "llamactl start '$profile' fehlgeschlagen (Exit-Code != 0)"
        fehler "ready ($profile)" "nicht geprüft (Start fehlgeschlagen)"
        fehler "status ($profile)" "nicht geprüft (Start fehlgeschlagen)"
        fehler "metrics ($profile)" "nicht geprüft (Start fehlgeschlagen)"
        fehler "stop ($profile)" "nicht geprüft (Start fehlgeschlagen)"
        fehler "status nach Stop ($profile)" "nicht geprüft (Start fehlgeschlagen)"
        return
    fi

    # 2) auf ready warten
    if wait_for_ready; then
        ok "ready ($profile)" "Zustand 'ready' erreicht (Zeitlimit ${READY_TIMEOUT}s)"
    else
        fehler "ready ($profile)" "Zustand 'ready' wurde innerhalb von ${READY_TIMEOUT}s nicht erreicht (State: $(status_state))"
    fi

    # 3) status prüfen — state muss "ready" sein, profile muss stimmen
    local st_state st_profile
    st_state=$(status_state)
    st_profile=$(llamactl status --json 2>/dev/null \
        | grep -o '"profile"[[:space:]]*:[[:space:]]*"[a-z_]*"' \
        | head -n 1 \
        | sed 's/.*"\(.*\)"/\1/' \
        || true)
    if [ "$st_state" = "ready" ] && [ "$st_profile" = "$profile" ]; then
        ok "status ($profile)" "state=ready, profile=$st_profile"
    else
        fehler "status ($profile)" "erwartet state=ready und profile=$profile, bekommen state=$st_state profile=$st_profile"
    fi

    # 4) metrics prüfen — im ready-Zustand müssen vram.used_gib und prompt_tps
    #    vorhanden sein (Modellserver antwortet auf /props und /metrics)
    local metrics_json m_used m_tps
    metrics_json=$(llamactl metrics --json 2>/dev/null || true)
    m_used=$(printf '%s' "$metrics_json" \
        | grep -o '"used_gib":[[:space:]]*[-+0-9.eE]*' \
        | head -n 1 | sed 's/^[^0-9.+-]*//' || true)
    m_tps=$(printf '%s' "$metrics_json" \
        | grep -o '"prompt_tps":[[:space:]]*[-+0-9.eE]*' \
        | head -n 1 | sed 's/^[^0-9.+-]*//' || true)
    if [ -n "$m_used" ] && [ -n "$m_tps" ]; then
        ok "metrics ($profile)" "used_gib=$m_used, prompt_tps=$m_tps"
    else
        fehler "metrics ($profile)" "vram.used_gib oder prompt_tps fehlt in der Metriken-Antwort"
    fi

    # 5) stop
    if llamactl stop >/dev/null 2>&1; then
        ok "stop ($profile)" "API-Antwort erhalten, Instanz wird gestoppt"
    else
        fehler "stop ($profile)" "llamactl stop fehlgeschlagen (Exit-Code != 0)"
    fi

    # 6) status nach Stop — state muss "stopped" sein
    #    Kurz warten, bis der Zustand durchgesetzt ist
    local stop_deadline=$(( $(date +%s) + 30 ))
    while [ "$(date +%s)" -lt "$stop_deadline" ]; do
        st_state=$(status_state)
        if [ "$st_state" = "stopped" ]; then
            break
        fi
        sleep 1
    done
    if [ "$st_state" = "stopped" ]; then
        ok "status nach Stop ($profile)" "state=stopped"
    else
        fehler "status nach Stop ($profile)" "erwartet state=stopped, bekommen state=$st_state"
    fi
}

# --- Hauptteil ----------------------------------------------------------------

echo "Gleichwertigkeitsprüfung der portierten Profile"
echo "API-Server: http://127.0.0.1:8081 (Default LLAMACTL_PORT)"
echo "Profiles: ${PROFILES[*]}"

# API-Server muss erreichbar sein
if ! llamactl profiles >/dev/null 2>&1; then
    fehler "API-Server" "nicht erreichbar unter http://127.0.0.1:8081 — bitte 'systemctl --user status llamactl' prüfen"
    echo ""
    echo "Zusammenfassung: $ok_count OK, $err_count FEHLER"
    exit 1
fi
ok "API-Server" "erreichbar (llamactl profiles lief erfolgreich)"

for p in "${PROFILES[@]}"; do
    check_profile "$p"
done

echo ""
echo "Zusammenfassung: $ok_count OK, $err_count FEHLER"

if [ "$err_count" -eq 0 ]; then
    echo "Alle Prüfungen erfolgreich."
    exit 0
else
    echo "Die Prüfung ist NICHT vollständig erfolgreich."
    exit 1
fi
