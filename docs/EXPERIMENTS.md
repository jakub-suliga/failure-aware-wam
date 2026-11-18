# Experimentplan: Fehler von LingBot-VA erkennen und beheben

> **Stand 2026-11-10: Ziel und aktueller Umfang stehen in [GOAL.md](GOAL.md) und [PLAN.md](PLAN.md).** Dieses Dokument
> beschreibt den Methodenkatalog im Code; der hier skizzierte Sweep-Umfang ist durch PLAN.md ersetzt.

> **Stand 2026-11-04: Der Kern der Arbeit ist jetzt Learn-from-Failure, siehe [METHOD.md](METHOD.md).** Das Modell
> lernt aus seinen eigenen Fehlversuchen: Vorhersagefehler plus VLM- bzw. Orakel-Analyse, gegeben über Adapter,
> Kontext oder beides. A, B und C unten sind Baselines oder Bausteine: A5 Negativ-Prompt, B1 Lücken-Detektor,
> C Analyse. D und E sind die Werkzeuge, auf denen LfF aufbaut. Für die Cluster-Planung haben die P5-Sweeps Vorrang,
> P1 nur als kleine Baseline-Auswahl.

Stand: 2026-11-03. Grundlage sind die Befunde in [FINDINGS.md](FINDINGS.md). Der Code liegt in `fawam/`
(Server, Methoden, Adapter, Daten), die Cluster-Skripte in `cluster/`, die Sweeps in `configs/sweeps/`.

**Kernidee.** LingBot-VA stellt sich unter einer Verschiebung oft den richtigen Griff vor, führt ihn aber nicht aus
(Imagination-Execution-Lücke). Alle Experimente setzen an einer von drei Stellen an:
1. **Vorhersage → Aktion koppeln** (A1, A2, D mit Aktionspfad),
2. **die Lücke online erkennen und reagieren** (B, C, E2),
3. **das Modell mit wenigen Beispielen an die Verschiebung anpassen**: mit Gewichten (D, dein Dual-Site-Ansatz) oder
   ohne Gewichte über den Kontext (E1, E3).

Status-Legende: **getestet** = lief lokal mit dem echten Modell. **Code-getestet** = Unit- oder Replay-Test, aber
nicht in der Simulation. **ungetestet** = geschrieben, nicht ausgeführt.

---

## 1. Methodenkatalog

### A. Trainingsfrei, im Server (pro Chunk)

| ID | Methode | Preset(s) | Idee / Quelle | Hypothese | Status |
|---|---|---|---|---|---|
| A0 | Probe: Greiferhöhe aus vorhergesagtem Latent | (Analyse) | Prüft, ob der Videopfad die Tiefe richtig vorstellt | Vorstellung ≈ richtig, Ausführung falsch | offen: Sim-Labels sind da, Probe fehlt noch |
| A1 | **Imagination Guidance** | `ig1.5` `ig2` `ig3` `ig2_half` | Classifier-free Guidance über die vorhergesagten Frames: Aktion mit vs. ohne die Pred-Frames im KV-Cache, extrapoliert. Analog zu CFG, neu für WAMs. | H1: tiefere Griffe bei −5/−10 cm | getestet (Sim, 1 Ep.) |
| A2 | History-Dropout | `acthist0/2/4` `vidhist0` | Vergangene Aktions-Tokens ausblenden (Attention: 50–59 % auf Aktionshistorie) | H2: weniger Prior, mehr Bild | Code-getestet (Replay) |
| A3 | Best-of-N Imagination | `bon4_medoid` `bon8_medoid` `bon4_random` `bon4_probe` | N Videos samplen, auswählen per Medoid (Konsens), Failure-Probe oder zufällig (Kontrolle). Vgl. World-Coherent Decoding, DreamAvoid. | H6: kleiner Gewinn im Hard-Setting | Code-getestet |
| A4 | Guidance-Sweeps | `vg3` `vg7` `ag2` `ag3` | Text-CFG für Video bzw. Aktion | Kontrolle | Code-getestet |
| A5 | **Failure-aware Negative Prompt** | `neg_missed` `neg_drop` `neg_generic` | Fehlerbeschreibung als CFG-Negativ statt "" (das ist der VLM-Analyse-Einbau ohne Training) | Video weg vom Fehler | Code-getestet |
| A6 | Steering-Vektor | `steer_l19_a1` `steer_l19_a3` | mean(Erfolg) − mean(vor Fehler) der Hidden States, addiert in Layer 19 (vgl. *Steering Robustness into WAMs*, 2607.14943) | schwach, Kontrolle für D | Code-getestet |
| A7 | Mehr Denoising | `steps50` | 50/100 statt 25/50 Schritte | Kontrolle (Rechenaufwand) | Code-getestet |

### B. Online-Erkennung und Eingriff (Client und Server)

Der Server liefert nach jedem Chunk die **Vorhersage-Lücke** (Kosinus pred vs. obs pro Kamera, z-standardisiert mit
`artifacts/gap_stats.json`). Das ist derselbe Detektor wie in FINDINGS Abschnitt 3 (faire AUC 0,84–0,88).
Ausgelöst wird bei z > 2,5 auf einem Handgelenk, höchstens 2 Mal pro Episode mit 2 Chunks Pause dazwischen.
Das ist einstellbar in `fawam/methods.py`, `INTERVENTION_DEFAULTS`.

| ID | Eingriff | Preset | Quelle | Status |
|---|---|---|---|---|
| B1 | nur loggen (immer aktiv) | – | – | getestet |
| B2 | Kontext kürzen | `truncate_on_gap` | FARE (2609.18016): Prefix-Recovery | Code-getestet |
| B3 | Kontext-Reset (Neuplanung ab aktuellem Bild) | `reset_on_gap`, `ig2+reset_on_gap` | FARE: Full Reset | **getestet** (2 Resets ausgelöst) |
| B4 | Re-Prompt mit Korrektur-Template | `reprompt_on_gap` | CorrectVLA (2608.29967) | ungetestet |
| B5 | Negativ-Prompt erst bei Lücke | `neg_on_gap` | A5 + B | ungetestet |
| B6 | Trigger per Sim-Orakel statt Lücke | `reset_on_oracle` | obere Schranke für den Detektor | ungetestet |

### C. Fehleranalyse per VLM bzw. Orakel im Loop

| ID | Methode | Preset | Status |
|---|---|---|---|
| C1 | VLM-Analyse offline auf aufgezeichneten Fails, verglichen mit Sim-Labels (Zuverlässigkeit; MolmoSpaces-CS: VLM-Urteile unzuverlässig) | `fawam/robotwin_hooks.vlm_analysis` auf Frames | Code vorhanden, Auswertungsskript fehlt noch |
| C2 | Bei Lücke: VLM bekommt die echten **und die vorhergesagten** Frames und liefert JSON mit Fehlertyp, Ursache, Korrektur-Instruktion und Negativ-Prompt. Danach Prompt- und Negativ-Update plus Kontext-Reset. | `vlm_on_gap` (Sweep `p1_vlm`, startet vLLM mit Qwen2.5-VL-7B im Job) | ungetestet |
| C2-Orakel | wie C2, aber die Analyse kommt aus dem Sim-Zustand (Greifer x cm über Objekt) | `oracle_on_gap` | ungetestet |

C2 vs. C2-Orakel trennt zwei Fragen: Hilft die *Information* (Orakel) überhaupt, und liefert das VLM sie (C2)?

### D. Few-Shot-Adaption: dein Dual-Site-Adapter auf dem WAM

Übertragung aus *Few-Shot Residual Adaptation of Pretrained Latent World Models*:

| Paper (DUET-DINO + CEM) | Hier (LingBot-VA) |
|---|---|
| eingefrorener Prädiktor, 24 Blöcke × 2 Köpfe | eingefrorenes 30-Block-Backbone, geteilt von **zwei Strömen** (Video, Aktion) |
| Adapter+ nach Attention-Residual und nach Block, pro View | gleich, **pro Strom getrennt** (`inside_streams`); Bottleneck 128 → 96 M Parameter (≈ 2 %) |
| externes Residual-MLP auf vorhergesagtem Latent | Token-MLP (2×2-Patches, 192-d) auf dem entrauschten Video-Latent, **bevor** es in den KV-Cache geht. Der Aktionsstrom sieht also die korrigierte Vorstellung. |
| L1 auf nächstes Latent, 3 Episoden | Flow-Matching-Loss von LingBot-VA (Video und/oder Aktion); außen L1 auf (pred, obs) |
| sequentiell: erst innen, Vorhersagen neu berechnen, dann außen | gleich: `fawam.replay` berechnet die Vorhersagen des adaptierten Modells neu |
| fester CEM-Planer | keine Planung, die Policy handelt direkt. Paper-nah ist D3 mit Best-of-N als „Planer“. |

Wichtiger Unterschied (FINDINGS, Abschnitt 7): Die Video-Vorhersage von LingBot-VA ist ein *Plan*, kein reines
Vorwärtsmodell. Selbstüberwachtes Training auf eigenen **Fail**-Rollouts lehrt den Videopfad, Fehler vorherzusagen.
Für den **Aktionspfad** ist es dagegen korrekt (Hindsight: Die ausgeführten Aktionen haben genau dieses Video
erzeugt). Daher gibt es beide Varianten:

| ID | Adapter (`cluster/adapter_grid.py`) | Daten | Frage |
|---|---|---|---|
| D1 | `d1_dual_m05_n{3,10,30}` | Expert-Replays bei −5 cm, n pro Task | Reichen wenige Demos? (Paper: 3 Episoden) |
| D2 | `d2_inside_both/_action/_video`, `d2_outside_only`, `d2_lora`, `d2_dual_parallel`, `d2_dual_actloss`, `d2_dual_vidloss` | wie D1, n = 10 | Wo muss korrigiert werden? (Paper-RQ3) Bei der Lücke erwartet: Aktionspfad |
| D2-eval | `<adapter>@inside`, `<adapter>@outside` | – | Site-Ablation beim Eval ohne Neutraining |
| D3 | `d3_selfsup_m05_n{3,10,30}`, `d3_selfsup_actloss_m05_n10` | eigene Rollouts (alle Ausgänge) | paper-treu selbstüberwacht; Hindsight-Aktionsloss allein |
| D3-online | `cluster/online_adapt_job.sh` | 3 eigene Episoden, Update nach jeder | exakt das Paper-Protokoll |
| D4 | `d4_filtered_m05` | nur erfolgreiche Rollouts | Self-Training / gefiltertes BC |
| D5 | `d5_multishift_n10` | Demos bei −2,5/−5/−10 cm | Ein Adapter für alle Shifts? Test auf −7,5/−15 cm (nicht gesehen) |
| D6 | `d6_hard_n30` | Expert-Replays im Hard-Setting | Hilft das auch bei handover drops / missed grasps? |

Jede Evaluation läuft zusätzlich auf `table_h_z00`, um das Vergessen zu messen (Paper: 85 → 54 % auf dem Nominal-Task).

Status: Training (innen und außen), Speichern, Laden im Server und Replay sind mit dem echten Modell **getestet**
(20 Schritte, 5 Episoden: Loss fällt, Lücke auf Trainingsdaten 0,098 → 0,072). Kein Eval-Ergebnis bisher.

### E. In-Context-Learning

Bei LingBot-VA ist der **KV-Cache der Kontext**. In-Context-Learning heißt hier: vor der Episode ganze Episoden
(Beobachtungs-Latents + Aktions-Chunks) in den Cache schreiben. Die Episode startet dahinter (RoPE-Zeit läuft weiter,
`attn_window` 72 Latent-Frames ≈ 2–4 Episoden). Das kostet kein Training und keine Bildkodierung, weil die Latents
schon im `episode.h5` stehen.

| ID | Methode | Preset / Grid | Frage | Status |
|---|---|---|---|---|
| E1 | Demo-Prefill, nächste Nachbarn (Kopf-Latent) aus der Demo-Bank des Settings | `icl1_<cfg>`, `icl2_<cfg>`, `icl1gap8_<cfg>` (8 Frames Abstand) | Nutzt das Modell eine passende Demo ohne Training? | **Code-getestet** mit echtem Modell (`fawam/tests/test_prefill.py`), Ergebnis unten |
| E1-Kontrollen | zufällige Demo, Demo der *falschen* Höhe (z00) | `icl1rand_<cfg>`, `icl1wrong_<cfg>` | Inhalt oder nur „irgendein Kontext“? | ungetestet |
| E2 | Retry mit Gedächtnis: nach einem Fail derselbe Seed mit dem Fehlversuch im Kontext | `retry_memory` (Kontrolle: `retry_plain`) | Lernt es aus dem eigenen Fehler im Kontext? | ungetestet |
| E3 | ICL-Finetune: Adapter trainiert auf [Demo gleicher Aufgabe \| Zielsequenz] (Loss nur auf dem Ziel), dann E1 | `e3_icl_multishift` → `e3icl1_<cfg>` vs. `e3noctx_<cfg>` | Bringt man dem WAM bei, Kontext zu nutzen (RICL-artig)? | Trainingsschritt mit Kontext getestet (Unit-Test) |
| E4 | Text-Gedächtnis: Retry mit „im letzten Versuch schloss der Greifer 8 cm über dem Objekt“ im Prompt | `retry_text`, `retry_memory_text` | Reagiert der T5-konditionierte Pfad auf Fehlertext? (Erwartung: kaum) | ungetestet |

**Vorläufig (n = 1, Replay ohne Sim, pick_diverse_bottles z00):** Mit einer Demo derselben Aufgabe und Domäne im
Kontext steigt die Vorhersage-Lücke des ersten Chunks von 0,09 auf 0,48. Danach liegt sie wieder auf dem Niveau ohne
Kontext (0,30 / 0,17 / 0,30 gegen 0,22 / 0,18 / 0,31). Mit einer Demo aus dem randomisierten Setting: 0,36.
LingBot-VA nutzt fremden Kontext also nicht von sich aus. Er stört zunächst eher (vermutlich „setzt“ das Modell die
Demo fort). Das spricht für E3 und für die Varianten mit größerem Abstand.

Muss man für ICL neu trainieren? E1, E2 und E4 brauchen kein Training. Weil LingBot-VA aber nie mit fremden
Episoden im Kontext trainiert wurde, kann es den Kontext ignorieren oder dadurch gestört werden. Genau das misst E1
gegen die Kontrollen. E3 ist die Antwort, falls E1 nicht wirkt: einmal kurz trainieren, danach ist Anpassung reines
ICL zur Testzeit.

---

## 2. Wo die Vorhersagen eingebaut werden

Die dekodierbare Video-Vorhersage ist der Grund für ein WAM. Sie wird an sechs Stellen genutzt:

1. **Detektor (B1):** Lücke zwischen vorhergesagtem und beobachtetem Latent, pro Chunk und Kamera.
2. **Steuerung der Aktion (A1):** Die Aktion wird mit und ohne die vorgestellten Frames berechnet, der Unterschied verstärkt.
3. **Auswahl (A3):** mehrere vorgestellte Zukünfte, Bewertung vor dem Handeln (Konsens oder Failure-Probe auf
   ihren Hidden States, `artifacts/probe/failure_probe.npz`, AUC 0,91 optimistisch).
4. **Korrektur (D, außen):** Das Residual korrigiert die Vorstellung, bevor die Aktion sie liest.
5. **VLM-Input (C2):** Das VLM bekommt neben den echten auch die vorhergesagten Bilder (`vlm.pred_frames`).
   Der Server dekodiert dafür auf Anfrage den letzten vorgestellten Chunk (`{"decode_pred": True}`).
6. **Analyse:** `pred_*.mp4` pro Episode und `fawam.analysis_gap gap` (Lücke am Greifzeitpunkt).

---

## 3. Daten erzeugen

### 3.1 Was schon da ist
Jedes `episode.h5` (Rollout) enthält die Beobachtungs-Latents im Modellraum und die Aktions-Chunks im Cache.
`fawam/latent_data.py` macht daraus direkt Trainingssequenzen. Geprüft: Die Zuordnung Aktionen ↔ Latents ist exakt
(Abweichung 0,0 zu den ausgeführten Aktionen). VAE-Pass und LeRobot-Export sind nicht nötig.

### 3.2 Expert-Demos unter der Verschiebung (`--driver expert_replay`)
Der RoboTwin-Planer löst jede Aufgabe auch bei verschobenem Tisch. Während des ohnehin laufenden Experten-Checks
(`play_once`) wird seine End-Effektor-Trajektorie mitgeschnitten (ein Frame pro `save_freq`, gleiche Granularität
wie die Trainingsdaten von LingBot-VA: 155–159 Frames bei place_mouse_pad, Datensatz 133–178). Dann wird sie
**durch denselben Client- und Server-Pfad wie die Policy** ausgeführt:
- Die Beobachtungen werden mit dem Server-VAE kodiert. Train- und Testverteilung sind also identisch.
- In `actions/planned` stehen die Expert-Chunks im relativen Policy-Format.
- In `actions/model_proposal` steht, was LingBot-VA in derselben Situation getan hätte. Damit lassen sich Demo und
  Policy direkt vergleichen (DAgger-artig).

Geprüft ohne Modell (`fawam/tests/test_expert_replay.py`): Die umgerechnete Trajektorie löst 3/3 (bottles, −5 cm) und
2/2 (mouse pad, −10 cm). Der volle Lauf mit Modell scheiterte lokal am GPU-Speicher (fremde Prozesse auf der 5090)
und ist auf dem Cluster der erste Test.

```bash
DRIVER=expert_replay ANALYZE=0 scripts/run_eval.sh place_mouse_pad 10 table_h_m05 results/expert   # lokal
python cluster/submit_sweep.py configs/sweeps/p2_expert_data.yaml --submit                         # Cluster
```

### 3.3 Text-Embeddings
`python -m fawam.text_cache build <roots> --model_path <ckpt>` (UMT5, 4 MB pro Instruktion). Das erledigt
`cluster/train_job.sh` automatisch.

### 3.4 Neue Shifts
`configs/robotwin_task_config/table_h_{p05,p025,z00,m025,m05,m075,m10,m15}.yml`. `setup.sh` kopiert sie nach RoboTwin.
Weitere Shifts (Reibung, Objektmasse, Kameraposition) wären die Fortsetzung, analog zu deinem Paper (CoM, Reibung).

---

## 4. Training

```bash
# Dual-Site, sequentiell (Paper), 10 Demos pro Task, 4 Tasks
OUT=artifacts/adapters/d1_dual_m05_n10 EPISODES=results/sweeps/p2_expert_data/base/table_h_m05 \
  LIMIT_PER_TASK=10 SUCCESS=true TASKS="move_playingcard_away place_mouse_pad pick_diverse_bottles stack_blocks_two" \
  sbatch cluster/train_job.sh
# oder das ganze Grid
python cluster/adapter_grid.py train --submit
```

Standard-Hyperparameter: Lernraten aus dem Paper (innen 4,343e-4, außen 1,234e-3), AdamW, 400 Schritte innen,
300 außen, Fenster ≤ 24 Latent-Frames, Gradient-Checkpointing.
Lokal gemessen (5090): etwa 0,7 s pro Schritt bei 8 Frames, also ~5–10 min pro Adapter auf einer H100.
Speicher: 7 GB plus Aktivierungen, es passt neben Server und Simulation.

---

## 5. Cluster-Ablauf (HoreKa 2, H100-94GB)

Einmalig: `cluster/setup_hk2.sh repos|envs|curobo|assets|check` (siehe Kopf des Skripts, **auf dem Cluster
ungetestet**). Danach:

| Phase | Sweep / Skript | Zellen × Episoden | Jobs | grobe GPU-h* |
|---|---|---|---|---|
| P0 | `p0_baseline_table` (8 Tasks × 8 Höhen, base) | 64 × 50 | 192 | ~90 |
| P0 | `p0_baseline_hard` (8 Tasks, base) | 8 × 100 | 40 | ~25 |
| P1 | `p1_training_free_table` (22 Methoden × 3 Höhen × 4 Tasks) | 264 × 30 | 528 | ~300 |
| P1 | `p1_training_free_hard` | 36 × 50 | 144 | ~80 |
| P1 | `p1_intervention` | 144 × 30 | 288 | ~150 |
| P1 | `p1_vlm` | 16 × 30 | 48 | ~40 |
| P2 | `p2_expert_data` (8 Tasks × 7 Settings) | 56 × 60 | 112 | ~60 |
| P3 | `p3_adapters` (wird von `adapter_grid.py methods` erzeugt) | ~60 Methoden | ~400 | ~200 |
| P4 | `p4_icl` (wird erzeugt) | ~25 Methoden | ~150 | ~80 |
| D3-online | `cluster/online_adapt_job.sh` (4 Tasks × 2 Höhen) | 8 Jobs | 8 | ~10 |

\* Schätzung: 2–3 min pro Episode auf einer eigenen GPU (lokal gemessen 1,4–3,3 min), 3 Paare pro H100,
Best-of-N und IG 1,5–3× langsamer. Alles zusammen sind das ~1000 GPU-h. **Empfohlene Reihenfolge mit kleinem Start:**

```bash
PY=$PY_RT   # irgendein Python mit pyyaml
# Tier 1 (~60 GPU-h): Baseline und die 6 wichtigsten trainingsfreien Methoden bei -5 cm, 20 Episoden
$PY cluster/submit_sweep.py configs/sweeps/p0_baseline_table.yaml --configs table_h_z00 table_h_m05 table_h_m10 --submit
$PY cluster/submit_sweep.py configs/sweeps/p1_training_free_table.yaml --configs table_h_m05 --episodes 20 \
    --methods base ig2 ig3 acthist2 bon4_medoid neg_missed --submit
$PY cluster/submit_sweep.py configs/sweeps/p1_intervention.yaml --configs table_h_m05 --episodes 20 \
    --methods reset_on_gap oracle_on_gap retry_plain retry_memory --submit
$PY cluster/submit_sweep.py configs/sweeps/p2_expert_data.yaml --configs table_h_m05 table_h_m10 table_h_z00 --submit
# danach: Adapter (D1, D2) + ICL (E1) auf den P2-Daten
$PY cluster/adapter_grid.py train --only d1_ --submit; $PY cluster/adapter_grid.py train --only d2_ --submit
$PY cluster/adapter_grid.py banks && $PY cluster/adapter_grid.py methods
$PY cluster/submit_sweep.py configs/sweeps/p3_adapters.yaml --submit
$PY cluster/submit_sweep.py configs/sweeps/p4_icl.yaml --submit
# Stand pro Zelle / erneut einreichen (überspringt fertige Zellen)
$PY cluster/submit_sweep.py configs/sweeps/p1_training_free_table.yaml --status
```

Alle Methoden eines Sweeps laufen auf **denselben Seeds** (Job j, Paar i → Seed-Index j·n_par+i). Die Vergleiche
sind also gepaart, und McNemar bzw. ein gepaarter Bootstrap sind möglich. Methoden mit Kommas (JSON) gehören als
Datei nach `configs/methods/`, weil `sbatch --export` an Kommas trennt.

## 6. Auswertung

```bash
python -m fawam.analysis_gap compare results/sweeps/p1_training_free_table   # Erfolg pro Methode x Task x Setting, Wilson-CI
python -m fawam.analysis_gap gap results/sweeps/p1_training_free_table/ig2   # Lücke und Greifhöhe am ersten Griff
analysis/analyze.py <cell dir>                                               # die bisherige Fehleranalyse pro Zelle
```
Interventionen, Kandidaten-Scores, Lücken pro Chunk, Prefill-Pfade und die Methode stehen in jedem `episode.h5`
unter `/fawam` und in den Attributen (`interventions`, `fawam_method_full`).
Belastbarkeit: Bei 30 Episoden pro Zelle hat ein Unterschied von 20 % auf 40 % ein Wilson-CI von etwa ±17 %.
Für Aussagen pro Task braucht man also 50–100 Episoden, für Methoden-Rankings reicht Pooling über Tasks.

## 7. Bekannte Lücken und Risiken

- **Cluster-Setup ungetestet**: Vulkan auf den Compute-Nodes ist das größte Risiko. Fallback ist ein Apptainer-Image
  für den Client, wie im Tactile-Projekt.
- **Expert-Replay mit Modell** lief lokal nicht durch (GPU-Speicher durch fremde Prozesse). Konvertierung und
  Ausführung sind ohne Modell geprüft.
- **C2 pred_frames**: Der Decode eines einzelnen Chunks läuft ohne den kausalen VAE-Kontext davor. Für ein VLM reicht
  das, die Bilder sind aber etwas unschärfer als `pred_*.mp4`. Ungetestet.
- **Lücken-z-Score**: Die Statistik stammt aus eval_hard und z00 (2 bzw. 3 Tasks). Für neue Tasks wird der globale
  Wert genutzt. Nach P0 `python -m fawam.probe gap-stats results/sweeps/p0_baseline_table/base/table_h_z00 ...`
  neu erzeugen, sonst triggern B-Methoden auf unbekannten Tasks falsch.
- **Failure-Probe/Steering** sind auf 10 Fails gefittet (eval_hard). Nach P0 neu fitten.
- Kontext-Reset dupliziert das aktuelle Bild als neuen Startframe. Die Frame-Zählung im Client weicht danach um ein
  Latent ab (Events stehen mit env_step im h5).
- A3 mit `probe` braucht pro Kandidat einen Extra-Forward. N = 8 ist auf einer H100 ~3× langsamer.

## 8. Literatur zu den Methoden

- FARE, Causal-History Test-Time Scaling for Failure Recovery in AR WAMs — [2609.18016](https://arxiv.org/pdf/2609.18016) (B2/B3)
- World-Coherent Decoding — [2609.02159](https://arxiv.org/pdf/2609.02159); DreamAvoid — [2605.11750](https://arxiv.org/pdf/2605.11750) (A3)
- CorrectVLA, training-free correction via language feedback — [2608.29967](https://arxiv.org/pdf/2608.29967) (B4, C2)
- Steering Robustness into WAMs via Mechanistic Interpretability — [2607.14943](https://arxiv.org/pdf/2607.14943) (A6)
- WorldScape Policy 2.0 (ICL-WAM) — [2607.18840](https://arxiv.org/abs/2607.18840) (E3, Vergleichsmodell)
- GPT-Policy (frozen VLM in-context) — [2609.19138](https://arxiv.org/pdf/2609.19138) (C)
- FailSafe — [2510.01642](https://arxiv.org/pdf/2510.01642); AHA — [2410.00371](https://arxiv.org/abs/2410.00371); RoboFAC — [2505.12224](https://arxiv.org/pdf/2505.12224) (C1/C2)
- Few-Shot Residual Adaptation of Pretrained Latent World Models (Suliga et al., NeurIPS 2026 WS) (D)
- Aus dem Gedächtnis, nicht neu geprüft: RICL (Retrieval-ICL für π0-FAST), ICRT, Instant Policy (E3)
