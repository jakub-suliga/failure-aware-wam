# Arbeitsplan hid: versteckte Kalibrierfehler (Stand 18.11.2026)

Gilt für LingBot-VA in RoboTwin 2.0. Ergebnisse: [RESULTS.md](RESULTS.md).

## Entscheidungen und Abweichungen

- **Keine Orakel-Varianten** (Nutzer, 18.11.): C8 entfällt. Stufe A testet die Mechanismen M1–M3 mit dem besten realen Kontext (C3 + `+aux`).
- **Laufende Jobs nicht stoppen** (18.11.). Wegen des Cluster-Ausfalls ab 20.11. sind P7d-Auffüllen, VLM-Benchmark und p0s angehalten, nicht abgebrochen.
- **Minimalpfad vor dem Ausfall:**
  - nur `trans`, 90 Seeds × 7 Trainingsaufgaben
  - C1 × M1 und M3 (C2–C5, mit/ohne `+aux`)
  - Eval nur T1 mit ctx/noctx/shuf
  - die volle Matrix danach
- **Held-out-Aufgaben:** `move_playingcard_away`, `place_object_scale`, `move_pillbottle_pad`. Alle drei haben 12/12 Erfolg ohne Störung und kommen nie im Training vor.

## Umsetzung (Code)

| Teil | Datei |
|---|---|
| Offset o pro Seed, Patch von `setup_demo` / `take_action`, Kompensation des Experten | `fawam/hid_offset.py` |
| Anbindung im Eval-Client | `patches/lingbot-va.patch`: o als h5-Attribut `hid`, Fix bei jedem Seed |
| Task-Configs | `configs/robotwin_task_config/hid_{trans,gain,yaw,none}.yml`, Startpose-Rauschen 0,05 rad |
| Kontext-Features C2–C5 | `fawam/hid_ctx.py`: Fehlerfenster echt − vorgestellt, nur echt, Propriozeption beim Schließen, EE-Trajektorie |
| Leak-Test (3.5) | `fawam/hid_leak.py`: PCA + Ridge, Fehler aus Out-of-Fold-Vorhersagen |
| Smoke-Prüfung | `fawam/hid_check.py`: Ausführung − Befehl gegen o, Erfolg des Experten |
| M3 Residual-Kopf + Gate | `fawam/train_hid.py`, `cluster/train_hid_job.sh` |
| M3 in der Eval | Modus `hidres`, Methoden `hidm3_<C>[_aux03][_noctx\|_shuf]`, Hook auf `action_proj_out` im Server |
| C1 × M1 | Grid `failfix_hid_lora`, Methoden `icl_hid_lora[_noctx\|_shuf]` |
| Sweeps | `p10_hid_smoke`, `p10_hid_data` |

Zu den Bausteinen:
- **Aktionsraum:** LingBot-VA gibt 16-dim EE-Posen relativ zur Startpose aus. Als `state` bekommt es nur seine eigenen Aktionen, keine gemessene Propriozeption. Der „geglaubte Rahmen“ ist damit automatisch erfüllt.
- **M3:** Der Flow-Zielwert ist Rauschen − Aktion. Eine konstante Verschiebung aller Aktionen um d entspricht damit genau einer Geschwindigkeits-Verschiebung um −d. Der Kopf kann so „die ganze Trajektorie um −o verschieben“ ausdrücken.
- **Kosten von M3:** Kein Gradient läuft durch den Transformer. Ein Forward pro Schritt trainiert alle Kontext-Varianten gleichzeitig.

---

## Ursprünglicher Plan (Nutzer, 18.11.2026)

### 0. Ziel und harte Regeln

**Ziel:** Mindestens eine Variante nutzt nach einem Fehlversuch nachweislich den Kontext. Nachweis heißt: Mit Kontext löst sie signifikant mehr Retries als ohne Kontext und als mit vertauschtem Kontext. Das soll gelten:
- auf ungesehenen Aufgaben mit bekanntem Fehlertyp, und
- mindestens schwach auf einem ungesehenen Fehlertyp (abgesenkter Tisch).

**Harte Regeln**
1. **Zur Testzeit nur Kontext, den es auch real gibt:**
   - Kamerabilder und Latents
   - die eigene Video-Vorhersage
   - die Differenz aus Vorhersage und Realität
   - Propriozeption
   - die eigenen Aktionen
   - VLM-Text
   - ein binäres Fehlschlag-Signal
   - andere Signale, die der Roboter real bestimmen kann

   Nicht erlaubt als Eingabe: Objekt- und Tischposen aus der Simulation, Orakel-Δ, `above_mm` und Erfolgs-Flags der Simulation (außer dem einen Fehlschlag-Bit).
2. **Privilegierte Sim-Informationen nur im Training:** für Labels, Datenerzeugung, Experten-Demos und Hilfs-Losses.
3. **Zwei Kontrollen pro Variante** mit identischem Training: `noctx` und `shuf`.
4. **Keine Sim-Evaluation ohne bestandenes Trainings-Gate.**
5. **Konventionen beibehalten:** Sweep-Namen `pNN_*`, `data.js`, Journal, Erfolg ohne „weggeflogen“.

### 1. Bisher bekannt

- **Kein Anreiz, den Kontext zu lesen:** Der Validierungs-Loss mit Kontext ist ≥ dem ohne Kontext.
- **Testtyp im Training:** Mit und ohne Kontext gleich gut, der Adapter lernt die Korrektur auswendig.
- **Held-out-Typ:**
  - Einziges Plus-Signal ist LoRA r16 auf der Attention: 11/25 gegen 2/29 (p = 0,003) bei −5 cm.
  - Bei −10 cm liegen alle Varianten bei 0/30.
  - TTT schadet. AdaLN, (IA)³ und Prompt-Vane können den Kontext ihrer Bauart nach nicht lesen.
- **VLM:** Der Text ist fast konstant und ohne Richtung. Auch fein-getunte Fehler-VLMs unterscheiden Höhenfehler nicht.
- **Sichtbare Störungen** sind aus dem aktuellen Bild erkennbar: für ICL ungeeignet, als Testtypen gut.

**Folgerung:** Der Fehler muss aus dem aktuellen Bild **nicht** erkennbar sein, im Fehlversuch aber **schon**.

### 3. Datensatz `hid`

**Idee:** Ein unsichtbarer, über beide Versuche konstanter Offset o verfälscht die Ausführung, ähnlich einer falschen Hand-Auge-Kalibrierung.

**Offset-Typen** (je 20 % mit o = 0):
- `trans`: z ~ U[−10, 10] cm, x/y ~ U[−5, 5] cm
- `gain`: Skalierung der z-Bewegung U[0,7; 1,3]
- `yaw`: ±25°

**Weitere Regeln**
- Die Startpose wird unabhängig von o randomisiert.
- Propriozeption steht im geglaubten Rahmen.

**Paare:** Versuch 1 mit dem Original-Modell unter o, dazu der Experten-Fix mit kompensierten Zielen, gespeichert im geglaubten Rahmen. Optional sind rund 30 % FAR-Reset-Paare.

**Umfang:** ≥ 300 Paare pro Aufgabe für `trans`, je 150 für `gain` und `yaw`; Held-out sind 3 Aufgaben.

**Leak-Test:**
- o aus dem ersten Frame: R² < 0,1.
- o aus dem Fehlerfenster: R² > 0,5.

### 4. Kontext-Varianten

| ID | Kontext |
|---|---|
| C0 | keiner (`noctx`) |
| C1 | ganzes Video, vorgestellt + echt |
| C2 | Fehlerfenster echt − vorgestellt, Handgelenk-Kameras, 4 vor bis 2 nach dem Schließen, Perceiver auf K Tokens (K ∈ {8, 32}) |
| C3 | C2 + Propriozeption beim Schließen |
| C4 | nur Propriozeption des Fehlversuchs |
| C5 | wie C3, aber nur echte Latents (Ablation: bringt die Vorhersage etwas?) |
| C6 | C3 + VLM-Text („detail“) |
| C7 | nur VLM-Text |

`shuf`: Kontext aus einer anderen Episode mit anderem o.

### 5. Mechanismen

| ID | Mechanismus |
|---|---|
| M1 | LoRA r16 auf der Self-Attention, Kontext als Präfix |
| M2 | gated Cross-Attention-Adapter (Gate mit 0 initialisiert), jeder 4. Block |
| M3 | Residual-Aktionskopf (außen) |
| M4 | Dual-Site: Adapter+ innen + M3 |
| M5 | FiLM aus Kontext + Adapter+ |

`+aux`: Hilfs-Loss, der o bzw. die nötige Korrektur aus dem Kontext vorhersagt.

Gestrichen: AdaLN, (IA)³, Prompt-Vane, TTT.

### 6. Training und Gate

**Training**
- Sequenz: `[Kontext] | [aktuelle Beobachtung] -> Korrektur-Aktionen`.
- In 30 % der Batches vertauschter Kontext (Ziel bleibt die richtige Korrektur), in 10 % Context-Dropout.
- 2 Seeds, 3000 Schritte, Basismodell eingefroren.

**Gate** (Validierungsdaten der Held-out-Aufgaben), alle vier Bedingungen müssen gelten:
1. `L_ctx` < 0,9 · `L_noctx`
2. `L_shuf` ≥ `L_noctx`
3. Korrelation zwischen vorhergesagter und nötiger z-Korrektur: r > 0,5
4. bei `+aux`: R² > 0,5

### 7. Evaluation

**Protokoll:** Versuch 1 mit dem Original-Modell, Retry mit der Variante, 30 Seeds pro Zelle; Bedingungen ctx, noctx, shuf.

**Testsets**
| ID | Was | Typ im Training? |
|---|---|---|
| T1 | hid-Offsets auf den Held-out-Aufgaben | ja (Aufgabe ungesehen) |
| T2 | Tisch −5 / −10 cm | nein |
| T3 | P7d Objektposition ±10 cm | nein |
| T4 | keine Störung, Kontext = Erfolg | Sanity |

**Metriken**
- Anteil gelöster Fehlversuche mit Wilson-Intervall; Fisher-Test ctx gegen noctx und gegen shuf.
- Richtungs-Korrelation (nötige Korrektur gegen tatsächliche Verschiebung, r > 0,3).
- `shift_mm`, `infl_action`, Attention.

**Lebenszeichen**, alle drei Bedingungen:
- T1: ctx − noctx ≥ +20 Prozentpunkte, p < 0,05, und shuf ≤ noctx.
- T2 oder T3: ctx > noctx mit p < 0,05, oder Richtungs-Korrelation r > 0,3.
- T4: kein signifikanter Abfall.

### 8. Reihenfolge

- **Stufe A:** Mechanismus-Diagnose (siehe Entscheidungen oben).
- **Stufe B:** {C1…C5} × {M1…M4} × {±aux}, zuerst nur das Gate.
- **Stufe C:** die besten 3 mit VLM-Text, K, Video-Loss, FAR-Reset und Eval T1–T4.
- **Stufe D** (erst bei Lebenszeichen): `--train_base`, zwei Fehlversuche im Kontext, M5, gain/yaw.

**Abbruchregeln**
- Besteht keine Variante aus Stufe A das Gate: zuerst den Mechanismus reparieren.
- Fehlt die Information im Kontext: zuerst `+aux` und größeres K, danach ein Fehler-Analysator (9).
- Ist C5 gleich gut wie C3: ehrlich berichten.

### 9. Optional: Fehler-Analysator

Ein kleines Netz schätzt aus Fehlerfenster und Propriozeption o bzw. die nötige Korrektur und gibt die Schätzung als Kontext weiter. Trainiert wird es mit Sim-Labels; zur Testzeit gehen nur reale Signale ein.

### 10.–12. Logging, Checkliste, Rückfragen

- **Logging:** Sweeps `p10_hid_data`, `p11_gate`, `p12_eval`; Gate-Tabelle und Lebenszeichen-Ansicht im Artifact.
- **Rückfragen an den Nutzer:**
  - wenn der Aktionsraum einen Punkt aus 3.2 unmöglich macht
  - wenn der Experte mit Kompensation unter 90 % bleibt
  - wenn der Leak-Test zweimal scheitert
  - bevor vom Nutzer gestartete Jobs abgebrochen werden
