# In-Context-Learning aus eigenen Fehlern (Schwerpunkt)

Stand: 2026-11-04. Das ist ab jetzt der Fokus der Arbeit. Die Adapter-Varianten ([METHOD.md](METHOD.md)) bleiben
unverändert und dienen als Vergleich: Funktioniert dieselbe Fehlerinformation auch über Gewichte?

**Frage.** LingBot-VA scheitert. Kann es beim nächsten Versuch denselben Fehler vermeiden, *ohne* dass seine Gewichte
geändert werden? Der Fehler wird ihm dafür nur als Kontext gegeben: im KV-Cache als Bild- und Aktions-Tokens und/oder
als Text.

## 1. Wie Kontext bei LingBot-VA funktioniert

LingBot-VA ist autoregressiv. Jeder Chunk sieht über den KV-Cache die bisherigen Beobachtungs- und Aktions-Tokens
(bis 72 Latent-Frames), und über Cross-Attention den Text-Prompt (UMT5). Daraus ergeben sich drei Eingangskanäle:

| Kanal | Wie | Code |
|---|---|---|
| **Erinnerung im KV-Cache** | aufgezeichnete Latents + Aktions-Chunks werden vor dem Versuch in den Cache geschrieben, die Episode läuft dahinter weiter | `fawam/server.py: _prefill` |
| **Prompt** | Instruktion + Korrektur-Text | Reset-Nachricht des Clients |
| **CFG-Negativ** | Fehlerbeschreibung als negatives Text-Embedding | `Method.negative_prompt` |

Das Modell wurde nie mit fremden Episoden im Kontext trainiert. Zwei Fälle werden deshalb getrennt getestet:
- **trainingsfrei** (der Kontext wird einfach gegeben),
- **meta-trainiert** (ein Adapter wurde einmal offline darauf trainiert, diesen Kontext zu nutzen; danach keine Updates mehr).

Erste Beobachtung (n = 1, Replay): Eine fremde Demo im Kontext verschlechtert trainingsfrei die Vorhersage des ersten
Chunks (Lücke 0,09 → 0,48).

## 2. Was in den Kontext kommt (Input-Varianten)

| Variante (Preset) | Inhalt der Erinnerung | Frage |
|---|---|---|
| `retry_plain` | nichts, kein Text | Kontrolle: wie viel bringt ein zweiter Versuch allein? |
| `icl_text_only` | nichts, nur Korrektur-Text | Reicht Sprache? |
| `icl_fail_obs` | Fehlversuch, **wie er passiert ist** | Reicht das eigene Erleben? |
| `icl_fail_pred` | Fehlversuch, **wie das Modell ihn sich vorgestellt hat** | Was bringt die eigene Erwartung allein? |
| `icl_fail_predobs` | vorgestellt + real nacheinander | Hilft der Kontrast Erwartung ↔ Realität (= der Vorhersagefehler im Kontext)? |
| `icl_fail_diff` | **Fehlerkarte** (real − vorgestellt) + real | Der Vorhersagefehler explizit als Token |
| `icl_fail_window` | nur das Fenster um den Fehler (4 Latents davor, 2 danach) | Ganze Episode oder der entscheidende Moment? |
| `icl_fail_corrected` | reale Bilder, aber **korrigierte Aktionen** (was man hätte tun sollen) | Kontrafaktische Erinnerung |
| `icl_demo_same` | Erfolgs-Demo derselben Szene (Expert-Replay) | Obere Schranke einer Demo |
| `icl_demo_nearest` | ähnlichste Erfolgs-Demo aus der Bank | Retrieval statt identischer Szene |
| `icl_fail_demo`, `icl_demo_fail` | Fehlversuch + Erfolgs-Demo (beide Reihenfolgen) | Kontrast „falsch vs. richtig“ |
| `icl_fails_3att` | alle bisherigen Fehlversuche (bis 3 Versuche) | Wächst der Nutzen mit der Erinnerung? |

## 3. Welcher Text dazu kommt und woher die Analyse stammt

| Variante | Text | Analyse |
|---|---|---|
| `icl_fail_predobs` | Korrektur („… 8 cm lower …“) | Orakel + Vorhersagefehler |
| `icl_txt_instruction` | nur die Instruktion | – |
| `icl_txt_negative` | Fehlerbeschreibung als CFG-Negativ | Orakel |
| `icl_txt_corr_neg` | Korrektur + Negativ | Orakel |
| `icl_txt_note` | „In the previous attempt … avoid this.“ | Orakel |
| `icl_fail_predobs_PR`, `icl_fail_window_PR` | Korrektur aus Vorhersagefehler + Rollout-Infos (ohne Simulator) | P + R |
| `icl_fail_predobs_VP` | VLM-Korrektur; VLM sieht echte + vorgestellte Bilder am Lückenzeitpunkt | V + P |
| `icl_fail_predobs_P` | keine Korrektur, nur Erinnerung. Der Fehlerzeitpunkt (fürs Fenster) aus dem Vorhersagefehler | P |

## 4. Wie die Erinnerung eingefügt wird

| Variante | Einfügemethode |
|---|---|
| `icl_fail_predobs` | **sequentiell**: Erinnerung zeitlich *vor* dem Versuch (RoPE-Frames 0 … M), Versuch ab M + 1 |
| `icl_pos_aligned`, `icl_aligned_window` | **phasengleich**: die Erinnerung belegt dieselben Frame-IDs wie der kommende Versuch. Frame k der Erinnerung steht zeitlich „neben“ Frame k des Versuchs (ähnlich der Phasen-Paarung in WAM-TTT) |
| `icl_gap8`, `icl_gap32` | größerer zeitlicher Abstand zwischen Erinnerung und Versuch (8 / 32 Frames) |
| `icl_attn0.5/1.5/2` | Keys der Erinnerungs-Tokens skaliert: Logits ×α, also mehr oder weniger Aufmerksamkeit auf die Erinnerung |
| `icl_trigger`, `icl_trigger_window` | **erst bei Bedarf**: Der Versuch startet ohne Erinnerung. Steigt die Vorhersage-Lücke (z > 2,5), wird der Kontext zurückgesetzt und beginnt mit der Erinnerung („just-in-time“) |
| `icl_pre_and_trigger` | beides |

## 5. Meta-trainiert: dem Modell beibringen, die Erinnerung zu nutzen

FailFix-Training (`cluster/adapter_grid.py`):
- Sequenz = [Erinnerung an den Fehlversuch | **Erfolg auf demselben Seed**]. Der Fehlversuch kommt aus der Baseline P0, der Erfolg aus dem Expert-Replay P2, beide auf denselben Seeds.
- Loss nur auf dem Erfolg. Korrektur-Text mit Wahrscheinlichkeit 0,5.
- Danach wird der Adapter eingefroren. Zur Testzeit ist es reines ICL.
- Ein Adapter pro Erinnerungs-Format, damit Training und Test passen.

| Meta-Adapter | Format | Test-Preset |
|---|---|---|
| `failfix_obs` | real | `icl_meta_obs` |
| `failfix_predobs` | vorgestellt + real | `icl_meta_predobs`, `_notext`, `_PR` |
| `failfix_diff` | Fehlerkarte + real | `icl_meta_diff` |
| `failfix_window` | Fenster um den Fehler | `icl_meta_window` |
| `democtx_meta` | Erfolgs-Demo → Erfolg | `icl_meta_demo` |

Kontrollen:
- `icl_meta_noctx`: Meta-Adapter, aber leere Erinnerung. Kommt der Gewinn vom Kontext oder nur vom Adapter?
- `icl_meta_mismatch`: trainiert auf real, getestet mit vorgestellt + real. Generalisiert das Format?
- Test zusätzlich bei −7,5 cm, das im Meta-Training nicht vorkommt.

Vergleich mit Gewichten: `icl_vs_adapter` (Adapter-Update aus demselben Fehler) und `icl_plus_adapter` (beides).

### 5a. Der Meta-Trainingsdatensatz (Sweep `p7_meta_data`)

Ohne genug Paare lernt das Modell nicht, Kontext zu nutzen. Der Datensatz wird deshalb vorab gezielt erzeugt:

- **Treiber `pair`:** Pro Seed fährt die Policy. Scheitert sie, fährt danach der Experte **denselben Seed** durch exakt
  dieselbe Kamera-/Latent-Pipeline. So entsteht ein Paar (Fehlversuch, Korrektur), Expert-Zeit fällt nur bei Fehlern an.
- **Kontinuierlicher Shift:** `table_h_rand` gibt jedem Seed eine zufällige Tischhöhe in [−15, +5] cm, dazu kommt das
  Hard-Setting. Das Modell sieht viele verschiedene Fehler statt nur bei zwei festen Höhen.
- **8 Tasks**, je 300 Seeds pro Setting. Seeds ab Index 1000, also disjunkt zu allen Test-Sweeps.
- Erwartet werden ~1400 Paare mit zufälliger Höhe (~60 % Fails) plus ~300 im Hard-Setting. Kosten: ~160 Jobs, grob 150 GPU-h.
- Danach `python -m fawam.failure_analysis <p7> --grasp_ref <Erfolge>` (Orakel-Text). Für die simulatorfreie Variante
  zusätzlich `--signals gap+rollout --as_main`.
- Weitere Quelle (optional, ohne Experte): erfolgreiche korrigierte Retries aus den LfF/ICL-Sweeps als Fix.

**Was jedes Trainingsbeispiel enthält** (`FailFixDataset`, Loss nur auf dem Ziel):

| Anteil | Erinnerung im Kontext | Ziel | Zweck |
|---|---|---|---|
| 55 % | eigener Fehlversuch desselben Seeds | Expert-Fix desselben Seeds | den Fehler korrigieren |
| 20 % | Fehlversuch eines **anderen** Seeds derselben Aufgabe/Setting | Fix | Übertragung statt Auswendiglernen der Szene |
| 10 % | ein anderer Erfolg | Fix | Kontext, der kein Fehler ist |
| 15 % | nichts | Fix | die normale Policy darf nicht schlechter werden |

Korrektur-Text im Prompt mit 50 %. Lazy Loading, 10 % der Seeds zurückgehalten zur Validierung. Gemessen wird
der Flow-Loss mit Erinnerung gegen ohne Erinnerung auf denselben Zielen: Meta-Training wirkt, wenn die Erinnerung den
Loss senkt. 12 000 Schritte mit Gradient-Accumulation 4 (≈ 1–2 Tage auf einer H100), Checkpoints alle 2000 Schritte.

**Ablationen des Meta-Trainings** (Test-Presets in `p6d`):
- `failfix_predobs_ownonly`: nur eigene Fehler, ohne Mischung.
- `_notext`: ohne Korrektur-Text.
- `_small`: 3000 Schritte, wie viel Training nötig ist.
- `_holdtask`: ohne `stack_blocks_two` trainiert, auf `stack_blocks_two` getestet. Generalisiert die Fähigkeit auf
  neue Aufgaben?

Getestet auf der CPU: Paarung, Mischung und Laden mit echten Episoden. Die Paare waren dafür künstlich zugeordnet.

## 6. Zwei Testebenen

1. **Offline-Screening (schnell, keine Simulation):** `fawam/icl_screen.py` bzw. `cluster/icl_screen_job.sh`.
   - Für Paare aus Fehlversuch und Erfolg auf demselben Seed wird der Erfolg teacher-forced abgespielt, mit der jeweiligen Erinnerung im Kontext.
   - Gemessen wird, ob die vorgeschlagenen Aktionen näher an die richtigen rücken als ohne Erinnerung, insgesamt und im Fenster um den Fehler.
   - Dazu kommt die Vorhersage-Lücke des ersten Chunks: Stört die Erinnerung?
   - So lassen sich alle Varianten auf 20 Paaren in wenigen GPU-Stunden vergleichen. Nur die vielversprechenden laufen dann in der Simulation.
2. **Simulation (`p6a`–`p6d`):** Erfolg im 2. Versuch gegenüber `retry_plain`, gepaart über Seeds. Dazu:
   - ob derselbe Fehler wiederholt wird,
   - Lücke und Greifhöhe am Fehlerzeitpunkt,
   - Anzahl Erinnerungs-Tokens.

| Sweep | Inhalt | Jobs |
|---|---|---|
| `p6a_icl_content` | Was (Abschnitt 2) | 504 |
| `p6b_icl_text`, `p6b_icl_text_vlm` | Text und Analyse-Quelle (Abschnitt 3) | 192 + 24 |
| `p6c_icl_insert` | Wie (Abschnitt 4) | 264 |
| `p6d_icl_meta` | Meta-trainiert (Abschnitt 5) | 396 |

## 6b. Wie es andere Arbeiten machen: mit oder ohne Training?

| Arbeit | ICL von wem? | Training nötig? | Was sie zeigen |
|---|---|---|---|
| [WAM-TTT](https://arxiv.org/html/2607.06988) (WAM, Video-Kontext/Memory) | die Policy selbst | **ja, Meta-Training** | ohne Meta-Training 9 / 0 statt 100 / 88,9. Eine LingBot-artige ICL-Baseline erreicht 50 vs. 66,7 |
| [MEM](https://arxiv.org/pdf/2603.03596) (π, Video-Gedächtnis) | die Policy | **ja**: Finetuning mit Korrektur-Daten, bei denen der vorherige Fehler im Gedächtnis bleibt | Policy versucht es nach einem Fehler anders |
| [RoboRetry](https://jedward225.github.io/RoboRetry/) (Fehler im Kontext eines VLA) | die Policy | laut Abstract wird mit Fehler-Konditionierung auf einem Fehler-Datensatz (1334 Fails) gearbeitet; Details nicht lesbar | Gewinn = Retry-Hinweis + Korrektur-Text; Kontakt-Präzision bleibt ungelöst |
| [WorldScape Policy 2.0](https://arxiv.org/abs/2607.18840) (WAM mit Video-/Text-Prompts) | die Policy | **ja**, ICL ist Teil des Trainings-Curriculums | One-Shot-Video-Prompts steuern das WAM |
| RICL, ICRT, Instant Policy (aus dem Gedächtnis, nicht neu geprüft) | die Policy | **ja**, ICL-Training mit Kontext-Demos | danach neue Aufgaben ohne Updates |
| [GPT-Policy](https://arxiv.org/pdf/2609.19138), [Harness VLA](https://arxiv.org/html/2607.08448v3), [HELM](https://arxiv.org/pdf/2604.18791), [Recursive Video ICL](https://arxiv.org/html/2610.06843) | ein großes VLM/Agent **über** der Policy | **nein** | die Policy bleibt eingefroren, der Agent plant, wiederholt und wählt Demos |
| [CorrectVLA](https://arxiv.org/pdf/2608.29967) | Sprachkorrektur → Aktionsanpassung | **nein** | moderate Gewinne (46,8 % / 10,1 % Recovery) |
| FAR, FailBank, RePO-VLA, T³VF, JEPA-TTT | – | Finetuning / TTT, kein ICL | siehe METHOD.md |

**Fazit:** Auf Policy-Ebene funktioniert ICL praktisch nur, wenn das Modell vorher darauf trainiert wurde
(Meta-Training oder ICL im Curriculum). Trainingsfrei klappt es, wenn ein großes VLM als Agent über einer eingefrorenen
Policy lernt, nicht die Policy selbst. Das passt zu unserer Beobachtung (Demo im Kontext stört trainingsfrei) und
begründet den Schwerpunkt auf dem Meta-Training. Die trainingsfreien Varianten bleiben als Kontrolle im Plan.

## 7. Ablauf auf dem Cluster

1. P0 (Baseline-Fehler) und P2 (Expert-Replays, gleiche Seeds) für TABLE4 bei −5/−10 cm und Hard.
2. `python -m fawam.failure_analysis --grasp_ref <P0 z00-Erfolge> <P0 Fails>` (Orakel) und `--signals gap+rollout` (ohne
   Simulator). `python cluster/adapter_grid.py banks` (Demo-Banken aus P2).
3. **Offline-Screening** aller Varianten: `FAILS=… FIXES=… OUT=results/icl_screen/m05.csv sbatch cluster/icl_screen_job.sh`.
4. `p6a` mit `--episodes 20`. Danach `p6b` und `p6c` mit den besten Inhalten aus Screening und p6a.
5. **P7 Meta-Datensatz** (`p7_meta_data`, driver pair), Analyse darauf, dann Meta-Training
   `adapter_grid.py train --only failfix_ --submit` und `--only democtx_`. Validierungskurven (Loss mit vs. ohne
   Erinnerung) in `artifacts/adapters/<name>/train_inside.jsonl` prüfen, dann `p6d`.

## 8. Stand

- **Geprüft:**
  - Kontext-Specs und Prompts aller Presets offline.
  - Laden der Formate real, vorgestellt und Fehlerkarte aus echten Episoden.
  - Der frühere Prefill-Pfad mit echtem Modell.
- **Ungetestet:**
  - Der neue Kontext-Builder im Server mit echtem Modell: Fenster, korrigierte Aktionen, Fehlerkarte, phasengleich, Attention-Skalierung. Die lokale GPU war belegt.
  - Der Trigger-Zeitpunkt, das Screening und das Meta-Training pro Format.
- **Erster Test auf dem Cluster:**
  `ANY_SEED=1 FAILS=results/table_height_n10 FIXES=results/table_height/z00 OUT=… sbatch cluster/icl_screen_job.sh`
  läuft auf den vorhandenen Daten (3 Paare). Damit sind alle Einfügewege einmal mit echtem Modell geprüft.
