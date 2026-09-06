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
