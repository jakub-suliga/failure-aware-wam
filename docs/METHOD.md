# Learn-from-Failure (LfF): LingBot-VA lernt aus seinen eigenen Fehlern

> **Stand 2026-11-10: Ziel und aktueller Umfang stehen in [GOAL.md](GOAL.md) und [PLAN.md](PLAN.md).** Dieses Dokument
> beschreibt den Methodenkatalog im Code; der hier skizzierte Sweep-Umfang ist durch PLAN.md ersetzt.

> **Schwerpunkt seit 2026-11-04: In-Context-Learning, siehe [ICL.md](ICL.md).** Die Adapter-Varianten in diesem
> Dokument bleiben unverändert und dienen als Vergleich (dieselbe Fehlerinformation über Gewichte statt über Kontext).

Stand: 2026-11-04 (aktualisiert). Das ist der Kern der Arbeit. Die Methoden A, B und C in [EXPERIMENTS.md](EXPERIMENTS.md) sind ab
jetzt nur noch Baselines oder Bausteine (A5 Negativ-Prompt, B1 Lücken-Detektor). Befunde: [FINDINGS.md](FINDINGS.md).

**Idee in einem Satz.** Das Modell macht eine Episode und scheitert. Aus seinem eigenen Vorhersagefehler und einer
Fehleranalyse (VLM oder Orakel) entsteht ein Lernsignal. Das bekommt das Modell als Adapter-Update, als Kontext
oder beides, damit es denselben Fehler beim nächsten Versuch nicht wiederholt.

## 1. Ablauf

```
Versuch k ──fail──► Analyse ──────────────► Update ─────────────► Versuch k+1 (selber Seed)
                    │ Vorhersage vs. Realität │ (a) Adapter: Flow-Loss │      ... und mit persist:
                    │  → wann / wo            │     auf der Fail-Episode│      nächster Seed startet
                    │ VLM / Orakel            │     + relabelte Aktionen│      schon adaptiert
                    │  → was, warum, Korrektur│     + Gap-Gewichte      │
                    │  (Text + Δ in Metern)   │ (b) Kontext: Fehlversuch│
                    │                         │     (vorgestellt + real)│
                    │                         │     + Korrektur-Prompt  │
```

Code: `fawam/failure_analysis.py` (Analyse) → `fawam/learn.py` (Adapter-Update, Subprozess im lingbot-Env) →
`fawam/robotwin_hooks.LearnFromFailure` (Schleife im Client) → `fawam/server.py` (lädt Adapter, Prefill mit
Vorstellung). Eingeschaltet wird das über `Method.learn` (Presets `lff_*`, Sweeps `configs/sweeps/p5_*.yaml`).

## 2. Die Signale aus der Fail-Episode

| Signal | Woher | Wofür |
|---|---|---|
| **Vorhersagefehler pro Latent** (Kosinus pred vs. obs, Handgelenke) | im `episode.h5` | Lokalisieren, *wann* die Erwartung brach (`gap_peak_latent`). Loss-Gewicht pro Frame: `w = 1 + β·z_robust`, Standard β = 2 |
| **Fehlertyp, Zeit, Arm** | Orakel: Sim-Labels aus `analysis/analyze.py`. VLM: Bilder am Lücken-Peak | Fenster für die Korrektur (`window`: 3 Latents vor dem Fehler), Abschneiden (`truncate_at`) |
| **Korrektur Δ in Metern** | Orakel: Greifer vs. Objekt im Vergleich zu erfolgreichen Griffen (`artifacts/grasp_ref.json`). VLM: `delta_cm` | Relabeling der Aktionen im Fenster (Rampe bis zum vollen Δ) |
| **Korrektur-Text, Negativ-Prompt** | Orakel-Template oder VLM | Prompt des nächsten Versuchs, CFG-Negativ, Text im Meta-ICL |
| **vorgestellte Frames** der Fail-Episode | `model/pred` im h5 | Kontext: das Modell sieht, was es erwartet hat und was passiert ist |

Beispiel (echte Episode, move_playingcard_away bei −10 cm): „the left gripper closed 8 cm above the grasp height
and missed it“, Δz = −8,4 cm über Latents 1–4, abgeschnitten nach Latent 5.

Das Orakel ist die obere Schranke. Es beantwortet, ob die Information überhaupt reicht. Das VLM ist die realistische
Variante. `analyzer="none"` nutzt nur den Vorhersagefehler, ohne Inhalt: Das ist die rein selbstüberwachte Variante.

## 3. Weg (a): Adapter-Update aus dem Fehler

Gleiche Adapter wie im Dual-Site-Paper (Adapter+ in allen 30 Blöcken, getrennt für Video- und Aktionspfad), trainiert
mit dem Flow-Matching-Loss von LingBot-VA auf der Fail-Episode (`fawam/learn.py`, 150 Schritte, ~40 s lokal):

- **Aktionspfad:** Ziel sind die *relabelten* Aktionen, also die ausgeführten plus Korrektur Δ im Fenster. Ohne
  Relabeling würde der Aktionspfad die Fehlaktion nur bestätigen. Weil die nötige Korrektur den Trainings-Aktionsbereich
  verlässt (FINDINGS 5b), wird bei ±3 statt ±1,5 abgeschnitten.
- **Videopfad:** Ziel ist das *beobachtete* Video. Das Weltmodell lernt, was wirklich passiert (Paper: Residuen aus
  dem Vorhersagefehler). Im Korrektur-Fenster ist der Video-Loss standardmäßig aus (`video_in_window=skip`). Dort zeigt
  das Video den Fehler, den die relabelten Aktionen gerade vermeiden sollen.
- **Gewichtung:** Frames mit großem Vorhersagefehler zählen bis zu (1 + β)-fach. Hier lag die Erwartung daneben.
- **Gedächtnis:** Mit `memory=all` werden alle bisherigen Fehlschläge genutzt, mit `persist` auch über Seeds hinweg
  (lebenslanges Lernen). Erfolgreiche eigene Versuche kommen ohne Relabeling dazu, damit nichts verlernt wird.

## 4. Weg (b): der Fehlversuch als Kontext

Der KV-Cache ist der Kontext. Vor dem nächsten Versuch werden in den Cache geschrieben:
1. der Fehlversuch so, **wie das Modell ihn sich vorgestellt hat** (pred-Latents pro Chunk),
2. der Fehlversuch so, **wie er passiert ist** (obs-Latents),
3. die Aktionen dazu.

Der Prompt ist die Instruktion plus die Korrektur. Damit steht der Vorhersagefehler explizit im Kontext.

Ohne Training nutzt LingBot-VA fremden Kontext nicht. Eine passende Demo erhöhte die Lücke des ersten Chunks sogar
(0,09 → 0,48, EXPERIMENTS E1). Deshalb gibt es das **FailFix-Meta-Training**
(`cluster/adapter_grid.py`, Eintrag `failfix_meta`):

- Eine Trainingssequenz ist [Fehlversuch vorgestellt | Fehlversuch real | **Erfolg auf demselben Seed**].
  Der Erfolg kommt aus dem Expert-Replay. P0 und P2 haben dieselben Seeds, die Paarung läuft über Task, Setting und Seed.
- Loss nur auf den Erfolg. Prompt mit Wahrscheinlichkeit 0,5 inklusive Korrektur-Text.
- Das Modell lernt so: „Wenn im Kontext ein Fehlschlag mit dieser Lücke steht, mach es anders.“ Zur Testzeit ist das
  dann reines ICL ohne Gradienten (`lff_icl_meta`).
- Der Expert wird nur offline fürs Meta-Training gebraucht, nicht zur Testzeit.

`lff_both`: Meta-Adapter als Startpunkt, dazu Kontext **und** ein Adapter-Update auf dem Fehler.

## 5. Signal-Ablation: was braucht das Update? (Sweep `p5a_signals`, `p5b_vlm`)

Die Analyse setzt sich aus einzeln schaltbaren Signalen zusammen (`fawam.failure_analysis.analyze_signals`).
Die Rezepte sind identisch: Adapter-Update aus dem Fehlversuch, dann Retry auf demselben Seed.

| Signal | Inhalt | am echten Roboter verfügbar? |
|---|---|---|
| **P** Vorhersagefehler | Lücke pred vs. obs pro Latent → Fehlerzeitpunkt (Beginn der Lücke, Regel des Online-Detektors), Loss-Gewichte, Arm (Handgelenk mit größerer Lücke) | ja |
| **R** Rollout-Infos | Greifer-Zustand und Kommandos (Propriozeption), Kontakt-Flag (Ersatz für Kraft-/Tastsensor) → Fehlertyp missed_grasp/dropped, Zeit, Arm | ja (mit F/T-Sensor) |
| **I** Imaginations-Probe | Ridge-Probe Hidden State → Endeffektor-Position (1,1 cm Fehler). Δ = vorgestellt − real am Fehler → metrische Korrektur aus der eigenen Vorhersage | ja (Labels = Propriozeption) |
| **V** VLM | Fehlertyp, Ursache, Korrektur-Text, Negativ-Prompt, delta_cm. Allein: VLM sucht den Fehler in 6 Keyframes. Mit P/R: Bilder (echt + vorgestellt) am gefundenen Zeitpunkt | ja |
| **O** Orakel | Sim-Wahrheit für alles | nein (obere Schranke) |

| Preset | Signale | Was die Variante prüft |
|---|---|---|
| `retry_plain`, `retry_explore` | – | Kontrollen: neuer Versuch ohne Lernen, mit Aktionsrauschen (FAR-Exploration) |
| `lff_sig_P` | P | **nur Vorhersagefehler**: selbstüberwacht, Gewichte, kein Relabel (kein Δ) |
| `lff_sig_PR` | P + R | **Vorhersagefehler + Rollout-Infos**: genauer Zeitpunkt und Typ, Korrektur nur als Text |
| `lff_sig_PI` | P + I | Vorhersagefehler → metrische Korrektur aus der Vorstellung |
| `lff_sig_PRI` | P + R + I | alles Eigene, ohne VLM |
| `lff_sig_V` | V | **nur VLM** (findet Zeitpunkt selbst, kein Gewicht) |
| `lff_sig_VP` | V + P | VLM am Lückenzeitpunkt, mit echten **und** vorgestellten Bildern |
| `lff_sig_PRIV` | P + R + I + V | alles ohne Privilegien |
| `lff_sig_O`, `lff_sig_OP` | O (+ P) | obere Schranke, mit/ohne Gap-Gewichte |

Was man nach den Offline-Tests schon erwartet (4 echte Fails, Tabelle in OPEN_ITEMS):
- P allein lokalisiert den Fehler zeitlich unscharf (Latent 2/6/21/3 gegen Orakel 13/4/4/4).
- Mit R trifft der Zeitpunkt das Orakel in 4/4 Fällen.
- I liefert für die Tischhöhe ≈ 0, weil die Vorstellung den Greifer nicht tiefer zeigt (FINDINGS A0).
- Eine metrische Korrektur kommt hier also nur von V oder O, oder aus Exploration.

Das ist eine Hypothese, kein Ergebnis.

## 5b. Protokoll aus dem Dual-Site-Paper (Sweep `p5f_paper`)

Genau wie im Paper:

```
Episode 1 → Update → Episode 2 → Update → Episode 3 → Update → Adapter einfrieren → Evaluation (17 Episoden)
```

Jedes Update hat zwei Stufen, wie in Paper Sec. 3.3:
1. **innen:** Adapter+ im Video- und Aktionspfad, 25 Schritte auf allen bisherigen Adaptions-Episoden
   (Erfolge und Fehlschläge).
2. **außen:** Die Vorhersagen werden mit dem adaptierten Modell neu berechnet (`fawam.replay`), dann bekommt das
   Residual-MLP auf dem vorgestellten Latent 25 Schritte L1.

Jeder Client-Lauf ist eine Adaptions-Wiederholung mit eigenen Seeds. 4 Wiederholungen pro Zelle, alle Methoden auf
denselben Seeds. Ausgewertet werden nur die Eval-Episoden:
`python -m fawam.analysis_gap compare results/sweeps/p5f_paper --adapt_episodes 3`.

| Preset | Was |
|---|---|
| `base` | ohne Adaption, dieselben Eval-Seeds |
| `paper_P_dual` | **paper-treu**: nur Vorhersagefehler, selbstüberwacht, innen + außen sequentiell, 25 + 25 Schritte |
| `paper_P_inside`, `paper_P_outside` | nur eine Stelle (Paper-RQ3: innen vs. außen vs. beides) |
| `paper_P_dual_150` | mehr Schritte pro Update (150 + 100), weil wir Batch 1 statt 8 Clips haben |
| `paper_Pw_dual` | + Loss-Gewichte aus dem Vorhersagefehler |
| `paper_PR_dual` | + Rollout-Infos (Zeitpunkt, Typ, Korrektur-Text) |
| `paper_PRI_dual` | + Imaginations-Korrektur (Relabel aus der eigenen Vorstellung) |
| `paper_OP_dual` | + Orakel-Korrektur (Relabel), obere Schranke |
| `paper_VP_dual` (Sweep `p5f_paper_vlm`) | + VLM-Analyse |
| `paper_OP_dual_retry` | Adaptions-Episoden bekommen zusätzlich einen Retry mit Update (mehr Daten pro Seed) |

Unterschiede zum Paper:
- Es gibt keinen CEM-Planer, die Policy handelt direkt. Der Video-Adapter wirkt nur über den KV-Cache auf die Aktionen.
- Das Paper hatte nur eine Szene pro Shift, hier wechseln Objekte und Instruktionen zwischen den Episoden.
- Erwartung nach FINDINGS 5b/A0: `paper_P_*` hilft bei der Tischhöhe wenig, weil ohne Korrektur-Ziel kein Signal
  existiert, das den Aktionsboden verschiebt. `paper_OP_dual` und `paper_PR_dual` sollten den Unterschied zeigen.
  **Genau dieser Vergleich ist die Kernaussage:** Reicht der Vorhersagefehler allein (wie im Paper) oder braucht ein
  WAM zusätzlich Fehlerinformation?

Getestet: Zwei aufeinander folgende Updates nur der äußeren Stufe auf echten Episoden laufen durch, und Runde 2 baut
auf Runde 1 auf (L1 0,28 statt 0,32 unkorrigiert). Innen + außen sequentiell scheiterte lokal am GPU-Speicher.

## 6. Integration aktueller Methoden

Gelesen am 2026-11-04. Die Zahlen sind die der Autoren, viele Arbeiten sind Preprints.

| Paper | Kernidee | berichtetes Ergebnis | in LfF eingebaut als |
|---|---|---|---|
| [T³VF](https://arxiv.org/abs/2605.08215) (Test-Time Training for Visual-Foresight VLAs) | TTT aus Vorhersage vs. später beobachtetem Bild, nur ein kleines Modul, **Filter** über Aktionsvarianz | LIBERO-Plus 49,3 → 52,1. Ohne Filter kaum Gewinn (29,0 → 29,8), mit adaptivem Filter 31,8 | `lff_ttt_filtered`: nur Frames im oberen 30-%-Quantil der Lücke (`gap_filter`) |
| [JEPA-TTT](https://jepa-ttt.github.io/) | persistentes TTT des Latent-Prädiktors unter Dynamik-Shifts, dichter Replay-Buffer | +153 % Planungserfolg (0,267 → 0,678). Persistenz entscheidend (0,289 → 0,729) | `lff_ttt_persist`: Update nach jeder Episode, persistent, nur Video-Loss |
| [WAM-TTT](https://arxiv.org/html/2607.06988) | Fast-Weight-Residuen am Video-Expert eines WAM, **meta-trainiert**, TTT aus Video-Vorhersage | 46,2 vs. 32,5 (LDA). Naives LoRA 30/0, ohne Meta-Training 9/0. Pseudo-Aktions-Loss **schadet** (72 → 29) | Meta-Training → `failfix_meta`. Video-only-Ablation `lff_adapter_videoonly`, Gegenprobe `lff_adapter_actiononly` |
| [RoboRetry](https://jedward225.github.io/RoboRetry/) | eigener Fehlversuch im Kontext für den Retry | schlägt blinden Retry. Gewinn = struktureller Retry-Hinweis + **Korrektur-Text**. Kontakt-Präzision wird durch Text nicht gelöst | `lff_icl` vs. `lff_icl_nocorr` (ohne Text) vs. `lff_icl_noimag` (ohne Vorstellung) |
| [FAR](https://arxiv.org/html/2607.01111v1) (Failure-Aware Retry) | negative Chunks aus dem Fehlschlag (Critic-Wertabfall) vs. positive, Präferenz-Loss, 5–10 Schritte, **Exploration** im Retry | DP 47,0 → 63,4 (sim). Ohne Positive 60,2, ohne Negative 70,6 | `lff_pref`: Diffusion-DPO-Loss, relabelt ≻ ausgeführt im Fehlerfenster. Die **Lücke ersetzt den Critic**. `retry_explore` / `lff_adapter_explore` |
| [FailBank](https://arxiv.org/abs/2609.39820) | Bank eigener Rollouts mit Korrekturen, gewichteter Flow-Loss, **Anker** (eigene Erfolge), frischer LoRA pro Runde, **Guard** | π0.5 66,0 → 74,5. Anker senken Policy-Kosten (38,6 → 16,5) | `lff_bank`: persist, `init=base`, Erfolge als Anker, Guard (Loss-Verhältnis ≤ 1,1 auf eigenen Erfolgen) |
| [RePO-VLA](https://arxiv.org/html/2605.09410) | Recovery-Suffixe als neue Episoden, value-konditioniert, **auf RoboTwin** (FRBench) | π0.5 Fehler-Recovery 15 → 37–43 % | offen: Recovery-Segmente nach Kontext-Reset als Trainingsdaten (OPEN_ITEMS) |
| [FailBench](https://arxiv.org/html/2609.03611) | wie gut VLMs Erfolg/Fehler beurteilen | bestes Modell 0,77 balanced accuracy (Gemini 3 Flash), Qwen3-VL-8B 0,69. Kontaktreiche Aufgaben ≤ 0,60, Bias zu „Erfolg“ | Motivation für V+P (Lücke lokalisiert, VLM beschreibt). VLM-Modell `Qwen3-VL-8B`, austauschbar |
| [FARE](https://arxiv.org/pdf/2609.18016) (Causal-History TTS) | Historie zurücksetzen/kürzen bei Fehlern in AR-WAMs, auf RoboTwin | – | Bausteine B2/B3 (Kontext-Reset), kombinierbar |

Weitere relevante Arbeiten (nicht im Detail gelesen):
- Test-Time-Training, gleiche Familie wie T³VF: [VANE](https://arxiv.org/pdf/2608.09448) (TTT über zukünftige visuelle Repräsentationen), [SALT](https://arxiv.org/html/2610.07946v1) (Adaption innerhalb der Episode aus überlappenden Chunks), [Test-Time Perturbation Tuning](https://arxiv.org/abs/2604.18107) (CVPR 2026)
- Recovery und Selbst-Verbesserung: [Imagining Recovery](https://arxiv.org/pdf/2608.14822) (kontrafaktische Neuausrichtung zur Inferenz), [ViFailback](https://openaccess.thecvf.com/content/CVPR2026/papers/Zeng_Diagnose_Correct_and_Learn_from_Manipulation_Failures_via_Visual_Symbols_CVPR_2026_paper.pdf) (CVPR 2026, VLM-Diagnose mit visuellen Symbolen), [World Action Verifier](https://arxiv.org/html/2604.01985v1)
- Gedächtnis: [MEM](https://arxiv.org/pdf/2603.03596)

**Was LfF davon unterscheidet:**
- Das Lernsignal kommt aus dem eigenen Vorhersagefehler eines WAM. Er lokalisiert den Fehler und gewichtet den Loss, und mit der Imaginations-Probe kann er die Korrektur auch quantifizieren.
- Es wird mit einer semantischen Analyse (VLM) kombiniert.
- Es geht über Adapter **und** über Kontext ins Modell, beides auf demselben WAM.
- Die Paper oben nutzen jeweils nur einen dieser Teile.

## 7. Was jetzt auf dem Cluster läuft (Reihenfolge)

| Schritt | Was | Dauer (grob) |
|---|---|---|
| 0 | Setup + Smoke-Test, `fawam/tests/test_lff_effect.py` auf einer Fail-Episode (lernt der Adapter die Korrektur?) | 1 Tag |
| 1 | P0-Baseline für TABLE4 × {z00, m05, m10, Hard} und P2-Expert-Replays (gleiche Seeds) | ~50 GPU-h |
| 2 | `failure_analysis --grasp_ref` + `pose_probe fit` + `probe gap-stats` auf P0 | CPU, Minuten |
| 3 | **`p5a_signals`** mit `--episodes 20`: Kontrollen, P, PR, PI, PRI, O. Beantwortet „nur Vorhersagefehler vs. + Rollout-Infos vs. Orakel“ | ~150 GPU-h |
| 4 | **`p5b_vlm`**: nur VLM, VLM + P, alles | ~60 GPU-h |
| 5 | `p5c_recipe`: Rezept und Integrationen (T³VF-Filter, FAR-Präferenz, Exploration, video-/action-only) | ~150 GPU-h |
| 6 | `adapter_grid.py train --only failfix_meta`, dann `p5d_icl` (RoboRetry-Ablationen, Meta-ICL, beides) | ~100 GPU-h |
| 7 | `p5e_lifelong`: persistentes Lernen über Seeds (JEPA-TTT, FailBank) | ~80 GPU-h |
| 3b | **`p5f_paper`** (parallel zu 3): Paper-Protokoll, 3 Adaptions-Episoden → einfrieren → 17 Eval-Episoden | ~200 GPU-h |

Kommandos: `$PY_RT cluster/submit_sweep.py configs/sweeps/p5a_signals.yaml --episodes 20 --submit` usw.

**Metriken:** Erfolg im k-ten Versuch (gepaart über Seeds) und Wiederholung desselben Fehlers. Dazu das Vergessen
auf z00 nach dem Lernen, die Anzahl der Updates, die der Guard ablehnt, und die Zeit pro Update.

## 8. Was geprüft ist und was nicht

- **Geprüft offline** (echte Daten oder echtes Modell):
  - Orakel-, P-, PR-, PI- und PRI-Analyse auf 4 echten Fails.
  - Pose-Probe (1,1 cm Fehler).
  - Relabeling.
  - Learner end-to-end (Adapter, 30 Schritte).
  - Präferenz-Loss: erster Schritt korrekt (log 2 beim Start), danach OOM durch fremden GPU-Prozess.
  - Prefill-Pfad.
- **Ungetestet:**
  - Die ganze Retry-Schleife live.
  - VLM-Pfade.
  - Guard.
  - `every_episode`.
  - FailFix-Training.
  - **ob ein Update die Korrektur wirklich lernt** (`test_lff_effect`).
