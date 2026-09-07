# VRAM-Profile

Die vier Profile in `config/profiles.toml` zielen auf denselben Gesamt-VRAM-Bedarf
ab. Bei 19,98 GiB Gesamt-VRAM ist der Kopfraum sehr klein: alle bisher gemessenen
Profile liegen zwischen 18,17 und 18,99 GiB.

| Profil  | Kontext | Slots | Slot-Kontext | KV-Typ    | Batchgröße | Gemessener VRAM-Spitzenwert     |
|---------|---------|-------|--------------|-----------|------------|---------------------------------|
| fast    | 16384   | 1     | 16384        | f16/f16   | 2048       | 18,17–18,99 GiB (Spanne)        |
| large   | 65536   | 1     | 65536        | q8_0/q8_0 | 1024       | 18,17–18,99 GiB (Spanne)        |
| safe    | 32768   | 1     | 32768        | q8_0/q8_0 | 512        | 18,17–18,99 GiB (Spanne)        |
| shared  | 32768   | 4     | 8192         | q8_0/q8_0 | 512        | noch nicht gemessen             |

- Die Spanne 18,17–18,99 GiB gilt für die Profile `fast`, `large` und `safe`;
  es handelt sich um bekannte Messwerte, keine pro Profil exakt zugeordneten Werte.
- Der Spitzenwert von `shared` ist **noch nicht gemessen** und darf bis zu einer
  Messung nicht als sicher behandelt werden. Gemessen wird er mit
  `scripts/benchmark-vram.sh` (Default-Profil ist `shared`, z. B.
  `DURATION=120 ./scripts/benchmark-vram.sh shared`).
- Hinweis: Die Compute-Puffer wachsen pro Slot mit der Slotanzahl, während die
  KV-Cache-Gesamtgröße gleich bleibt. Das `shared`-Profil teilt zwar das bekannte
  Kontextbudget auf 4 Slots (je 8192 Tokens), der daraus resultierende
  VRAM-Zuschlag ist aber erst durch eine Messung belegt.

## Tuning-Felder

Neben den Sizing-Feldern kennt jedes Profil optionale Tuning-Knöpfe. Sie dürfen
im `[common]`-Block (globaler Default) **und** pro Profil gesetzt werden; ein
Profilwert überschreibt den `[common]`-Wert. Ist ein Feld weder im Profil noch
in `[common]` gesetzt, wird das zugehörige Flag gar nicht gerendert und
llama.cpp verwendet seinen eigenen Default.

| TOML-Feld          | llama.cpp-Flag        | Typ            | Bedeutung                                   |
|--------------------|-----------------------|----------------|---------------------------------------------|
| `batch_size`       | `-b` / `--batch-size` | int (Pflicht)  | logisches Batch-Limit (Default 2048)        |
| `ubatch_size`      | `-ub` / `--ubatch-size` | int          | physische Micro-Batch (Default 512)         |
| `n_gpu_layers`     | `-ngl` / `--n-gpu-layers` | int        | Layer im VRAM (`999` = alle)                |
| `flash_attn`       | `-fa` / `--flash-attn` | `on`/`off`/`auto` | Flash Attention                          |
| `cont_batching`    | `-cb` / `-nocb`       | bool           | Continuous (dynamic) Batching               |
| `cache_reuse`      | `--cache-reuse`       | int            | min. Chunk-Größe für KV-Cache-Reuse (0 = aus) |
| `spec_type`        | `--spec-type`         | str            | Speculative-Decoding-Typ, `draft-mtp` = MTP |
| `spec_draft_n_max` | `--spec-draft-n-max`  | int            | spec-max Tiefe (Draft-Tokens, Default 3)    |
| `spec_draft_n_min` | `--spec-draft-n-min`  | int            | minimale Draft-Tokens                       |

`-b` (logisches Batch) und `-ub` (physisches Micro-Batch) sind **zwei getrennte
Felder** — `batch_size` ist nicht `-ub`.

`cache_reuse` (`--cache-reuse N`) aktiviert das Wiederverwenden von KV-Cache-
Segmenten per KV-Shifting über Requests hinweg (min. Chunk-Größe `N`, Default
`0` = aus; setzt aktiviertes Prompt-Caching voraus).

**MTP (Multi-Token-Prediction):** In der aktuellen llama.cpp ist MTP der
Speculative-Decoding-Typ `draft-mtp` mit Self-Speculation (kein separates
Draft-Modell). Das Modell muss MTP-Layer besitzen (z. B. Qwen3-A3B). Beispiel:

```toml
[profiles.qwen-mtp]
model = "qwen3-a3b.gguf"
ctx_size = 32768
kv_cache_type_k = "q8_0"
kv_cache_type_v = "q8_0"
parallel = 1
batch_size = 512
spec_type = "draft-mtp"
spec_draft_n_max = 4     # spec-max Tiefe
spec_draft_n_min = 0
```

> Historie: Die früher in `extra_args` gepflegten `--draft-max` / `--draft`
> sind in der aktuellen llama.cpp **entfernt** und führen zu einem harten
> Startfehler. Sie wurden durch `spec_draft_n_max` / `spec_draft_n_min` ersetzt.

## Auslesen der effektiven Konfiguration

Die effektiven (Profil über `[common]` gemergten) Werte sind über die API
auslesbar:

- `GET /profiles` — alle Profile mit Sizing- und Tuning-Feldern.
- `GET /profiles/{name}` — ein einzelnes Profil (404 bei unbekanntem Namen).

Auf der Kommandozeile liefert `llamactl profiles --json` denselben Inhalt.

## Profile anlegen / ändern / löschen (CRUD)

Profile können über die API erzeugt, geändert und gelöscht werden. Änderungen
werden **persistent** in die aktive Profildatei (`LLAMACTL_PROFILES`)
geschrieben; vorhandene Kommentare/Notizen bleiben erhalten (Anlegen hängt einen
neuen `[profiles.<name>]`-Block an, Ändern ersetzt nur den betreffenden Block,
Löschen entfernt nur ihn).

- `POST /profiles` — Body sind die Profilfelder **plus** `name`, z. B.
  `{"name":"qwen-mtp","model":"m.gguf","ctx_size":32768,"kv_cache_type_k":"q8_0","kv_cache_type_v":"q8_0","parallel":1,"batch_size":512,"spec_type":"draft-mtp","spec_draft_n_max":4}`.
  `201` bei Erfolg, `409` wenn der Name existiert, `422` bei ungültigen/fehlenden Feldern.
- `PUT /profiles/{name}` — ersetzt die Felder eines Profils. `404` unbekannt,
  `409` wenn es das Profil der **laufenden** Instanz ist (erst stoppen), `422` ungültig.
- `DELETE /profiles/{name}` — löscht ein Profil. `404` unbekannt, `409` wenn in Benutzung.

Auf der Kommandozeile (Felder als JSON via `--fields` oder stdin):

```
llamactl profile-create qwen-mtp --fields '{"model":"m.gguf","ctx_size":32768,
  "kv_cache_type_k":"q8_0","kv_cache_type_v":"q8_0","parallel":1,"batch_size":512,
  "spec_type":"draft-mtp","spec_draft_n_max":4}'
llamactl profile-update safe --fields '{"model":"...","ctx_size":16384, ...}'
llamactl profile-delete qwen-mtp
```

Nur die Sizing-Pflichtfelder (`model`, `ctx_size`, `kv_cache_type_k`,
`kv_cache_type_v`, `parallel`, `batch_size`) sind erforderlich; alle
Tuning-Felder sind optional. Das laufende `manager._profiles` und die Datei
bleiben synchron, sodass ein Dienst-Neustart dieselben Profile lädt.

> Sicherheit: Die Control-API ist unauthentifiziert. Wer sie erreicht, kann
> Profile mit beliebigen `model`-Pfaden und `extra_args` anlegen. Bind an
> `127.0.0.1` binden oder Auth vorschalten, wenn das nicht erwünscht ist.
