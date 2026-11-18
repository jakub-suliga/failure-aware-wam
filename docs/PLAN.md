# Plan: was getestet wird

Stand: 2026-11-10. Ziel und Idee stehen in [GOAL.md](GOAL.md). Dieses Dokument listet alle laufenden und geplanten
Experimente, ihre Kontrollen und was gestrichen wurde.

Gemeinsam für alle Experimente:
- **Testaufgabe:** `move_playingcard_away`, Tisch bei −5 cm (`table_h_m05`) und −10 cm (`table_h_m10`).
- **Protokoll:** Versuch 1 mit dem unveränderten Modell. Scheitert er, folgt Analyse, dann Versuch 2 auf demselben
  Seed. Gemessen wird der Anteil der Fehlversuche, die im zweiten Versuch gelingen.
- **Seeds:** Alle Methoden eines Experiments laufen auf denselben Seeds (gepaarter Vergleich).
- Die Pipeline (`cluster/pipeline.py`, alle 20 Minuten per `scrontab`) reicht alles selbst ein. Stand:
  `pipeline/STATUS.md`.

## A. Learn-from-Failure über ein Adapter-Update (Vergleich und obere Schranke)

Nach dem Fehlversuch wird ein Adapter 150 Schritte auf der Fehl-Episode trainiert. Variiert wird, welche Information
die Analyse liefert. Sweeps `p5a_signals` und `p5b_vlm`, 30 Seeds pro Zelle.

| Methode | Analyse-Signal | Frage |
|---|---|---|
| `retry_plain` | – | Kontrolle: zweiter Versuch ohne Lernen |
| `retry_explore` | – | Kontrolle: zweiter Versuch mit Aktionsrauschen |
| `lff_sig_P` | nur Vorhersagefehler | reicht das Selbstüberwachte? |
| `lff_sig_PR`, `_PI`, `_PRI` | + Rollout-Infos, + Imaginations-Probe | hilft Eigenes ohne VLM? |
| `lff_sig_O`, `lff_sig_OP` | Orakel (+ Vorhersagefehler) | obere Schranke |
| `lff_sig_V`, `lff_sig_VP`, `lff_sig_PRIV` | VLM (+ Vorhersagefehler) | liefert das VLM, was das Orakel liefert? |

## B. Dual-Site-Adapter nach dem Paper-Protokoll

3 Adaptions-Episoden, nach jeder ein Update (innen 25 Schritte, dann außen 25 Schritte auf neu berechneten
Vorhersagen), danach eingefroren 17 Eval-Episoden. 4 Wiederholungen pro Zelle. Sweep `p5f_paper`.

| Methode | Was | Frage |
|---|---|---|
| `base` | ohne Adaption | Referenz auf denselben Eval-Seeds |
| `paper_P_dual` | **paper-treu:** nur Vorhersagefehler, innen + außen | überträgt sich der Ansatz auf ein WAM? |
| `paper_P_inside`, `paper_P_outside` | nur eine Stelle | innen, außen oder beides? |
| `paper_P_dual_150` | 150 + 100 Schritte | zu wenig Training? |
| `paper_PR_dual` | + Rollout-Infos | |
| `paper_OP_dual` | + Orakel-Korrektur | obere Schranke |

## C. Kontext ohne Meta-Training (Kontrolle)

Nutzt das eingefrorene WAM den Fehlerkontext von sich aus? Analyse hier vom Orakel, damit nur der Kontext-Weg
getestet wird. Sweep `p6a_icl_content`, 30 Seeds pro Zelle.

| Methode | Kontext |
|---|---|
| `retry_plain` | nichts |
| `icl_text_only` | nur Korrektur-Text im Prompt |
| `icl_fail_predobs` | Fehlversuch vorgestellt + echt, + Text |
| `icl_fail_diff` | Fehlerkarte (echt − vorgestellt), + Text |
| `icl_vs_adapter` | dieselbe Information als Adapter-Update |

## D. Meta-trainierter Kontext mit VLM-Analyse (die Idee)

### D1. Daten und Analyse

- **P7 (`p7_meta_data`):** 7 Trainingsaufgaben × 150 Seeds mit zufälliger Tischhöhe in [−15, +5] cm. Scheitert die
  Policy, fährt der Experte denselben Seed. Ergebnis: Paare (Fehlversuch, Korrektur). Die Testaufgabe ist nicht dabei.
- **VLM-Analyse** (`cluster/vlm_analysis_job.sh`, Qwen3-VL-8B): Das VLM bekommt 6 Keyframes des echten Rollouts (Kopf,
  linkes und rechtes Handgelenk), die Vorhersage des Weltmodells am Fehlerzeitpunkt und den Verlauf des
  Vorhersagefehlers als Text. Es liefert Fehlertyp, Ursache, Korrektur-Anweisung und `delta_cm`.
- **VLM gegen Orakel** (Job `fawam_vlm_vs_oracle_p0`): Auf den 889 Baseline-Fehlversuchen bei −5 und −10 cm werden beide
  VLM-Eingaben (ganzer Rollout bzw. nur der Fehlermoment) mit dem Simulator-Orakel verglichen. Das entscheidet, welche
  Eingabe für D2 genutzt wird und ob das VLM der Engpass ist.

### D2. Meta-Training (ohne die Testaufgabe)

LingBot-VA bleibt eingefroren. Trainiert werden nur die Zusatzmodule, auf Sequenzen
`[Fehlerkontext | Experten-Lösung desselben Seeds]`, Loss nur auf der Lösung. Mischung pro Beispiel: 55 % eigener
Fehlversuch, 20 % Fehlversuch eines anderen Seeds, 10 % ein Erfolg, 15 % kein Kontext. Erster Durchgang: 3000 Schritte
× Gradient-Accumulation 4, Validierung alle 500 Schritte (Loss mit Kontext gegen ohne Kontext auf zurückgehaltenen
Seeds). Text und Fehlervektor stammen aus der VLM-Analyse.

**Wie der Kontext ins Modell kommt** (ein Adapter pro Zeile, `cluster/adapter_grid.py`, Suffix `_vlm`):

| Adapter | Mechanismus |
|---|---|
| `type_adapterplus` | Kontext als Tokens im KV-Cache (Self-Attention), Adapter+ in allen Blöcken lernt, ihn zu nutzen |
| `type_lora_r16_attn` | wie oben, LoRA auf der Self-Attention |
| `type_adaln` | gelernter Offset der Norm-Modulation (statisch) |
| `type_ia3_adaln` | (IA)³-Skalierung + AdaLN |
| `type_film_memory` | FiLM/AdaLN-Zero: Modulation abhängig vom Mittel der Kontext-Tokens |
| `type_film_error` | FiLM abhängig vom Fehlervektor (VLM-Δ, Arm, Fehlertyp, Lückenstatistik) |
| `type_film_gapcurve` | FiLM abhängig vom Fehlervektor **+ Verlauf des Vorhersagefehlers** (16 Stützstellen) |
| `type_film_memory_plus_adapterplus` | FiLM + Adapter+ |
| `type_prompt_vane` | gelernte Prompt-Tokens aus einer Bank |
| `type_ttt_wam` | Fast-Weights, ein analytischer Schritt auf den Kontext-Tokens |

In allen Zeilen steht der Fehlversuch (vorgestellt + echt) als Tokens im Kontext, und der VLM-Text geht über die
Text-Cross-Attention ein.

**Was im Kontext steht** (Adapter+ als Mechanismus):

| Adapter | Kontext |
|---|---|
| `type_adapterplus` | ganzer Fehlversuch, vorgestellt + echt |
| `failfix_diff` | Fehlerkarte (echt − vorgestellt) + echt |
| `failfix_window` | nur das Fenster um den Fehler (4 Latents davor, 2 danach) |
| `failfix_textonly` | nur der VLM-Text, keine Rollout-Tokens |

**Obere Schranke mit Orakel-Analyse:** `type_adapterplus`, `type_film_error`, `type_film_gapcurve` (ohne `_vlm`).

### D3. Auswertung (`p8_icl_vlm_meta`, `p8_icl_oracle_meta`)

30 Seeds pro Zelle. Zur Testzeit analysiert dasselbe VLM den Fehlversuch live.

| Methode | Zweck |
|---|---|
| `retry_plain` | Kontrolle: kein Lernen |
| `icl_fail_predobs_VP` | Kontrolle: derselbe Kontext ohne Meta-Training |
| `icl_type_<typ>_VP` (10 Typen) | die Mechanismen aus D2 |
| `icl_type_<typ>_VP_noctx` (nur −5 cm) | **Kontrolle:** derselbe Adapter ohne Kontext und ohne Text |
| `icl_meta_diff_VP`, `icl_meta_window_VP`, `icl_meta_textonly_VP` | Kontext-Inhalte aus D2 |
| `icl_type_adapterplus`, `_film_error`, `_film_gapcurve` (Orakel) | obere Schranke |

## Reihenfolge und Umfang

| Schritt | Jobs | hängt ab von |
|---|---|---|
| A ohne VLM (Rest von `p5a`) | wenige | – |
| B Dual-Site (`p5f`) | 28 | – |
| C Kontext ohne Training (`p6a`) | 30 | – |
| VLM gegen Orakel | 1 | – |
| P7 Daten | 32 offen | – |
| A mit VLM (`p5b`) | 18 | VLM-Test bestanden |
| Analysen auf P7 (Orakel, VLM) | 2 | P7 fertig |
| Meta-Training | 13 + 3 | Analysen |
| D3 Auswertung | 120 + 27 | Meta-Training |

## Gestrichen

- Alle anderen Aufgaben in den Auswertungen (vorher 4 Aufgaben).
- `p2_expert_data`, Dual-Site mit Experten-Demos (D1/D2) und `p3`: Die Frage ist, ob der Adapter **nur mit
  Vorhersagefehler** funktioniert.
- `p6b`, `p6c` (Text- und Einfüge-Varianten ohne Training), `p6d`, `p6e`, `p6f`, `p5c`, `p5e`, `p1_*`, Hard-Setting.
- Demo-Kontext (Experten-Demos im Kontext): gehört nicht zur Idee.

## Offene Risiken

- **VLM ungetestet:** Der VLM-Server ist auf dem Cluster zweimal an der Infrastruktur gescheitert; der dritte Test
  läuft. Alle VLM-Teile warten darauf.
- **Neuer Code nur teilweise geprüft:** Die Rollout-Analyse des VLM ist ohne Server getestet (Anfrage wird korrekt
  gebaut). `type_film_gapcurve` und `failfix_textonly` sind noch in keinem Training gelaufen.
- **P7 ist unausgewogen:** Bisher stammen etwa die Hälfte der Paare von `move_can_pot`. `lift_pot` hat keine
  Objektposen, dort gibt es keine Orakel-Analyse (VLM geht).
- **Transfer auf eine neue Aufgabe** ist die härtere Variante. Gelingt D3 nicht, ist ein Lauf mit der Testaufgabe im
  Training der nächste Diagnoseschritt (trennt „Idee funktioniert nicht“ von „generalisiert nicht über Aufgaben“).
- **Kurzes Meta-Training:** 3000 Schritte sind ein erster Durchgang. Die besten Varianten bekommen danach mehr.
- **Rechenzeit:** Jobs warten 1–2 Tage in der Queue. Die Jobs laufen über zwei Projektkonten.

## Ergänzung vom 2026-11-10 (Nachmittag): breiter testen

- **C breiter, mit VLM und ohne Training** (`p6v_icl_free_vlm`, 15 Methoden): VLM-Ausgabe als Prompt, als
  Negativ-Prompt, beides, als Notiz, als korrigierte Aktionen im Fehlversuch. Kontext-Inhalt: nur echt, vorgestellt +
  echt, Fehlerkarte, Fenster, die 6 Frames mit dem höchsten Vorhersagefehler. Einfügen: doppelte Attention, erst am
  Lücken-Trigger, bis zu 3 Versuche. Kontrolle: Rauschen statt Rollout. Jede Methode gibt es per `+VP` mit VLM statt
  Orakel.
- **D, neue Mechanismen:** `type_film_text` (FiLM aus dem Embedding des VLM-Texts, also die VLM-Ausgabe als Latent)
  und `failfix_gapkeys` (nur die Frames mit dem höchsten Vorhersagefehler). Echte VLM-Hidden-States als Kontext-Tokens
  sind noch nicht gebaut (vLLM gibt sie nicht aus).
- **E. Verwandte Arbeiten nachgebaut** (`p9_related`): FARE (Kontext-Reset), T³VF (gefiltertes Test-Time-Training),
  WAM-TTT-Stil (nur Video-Loss), FAR (Präferenz-Loss), FailBank. RoboRetry-artige Varianten stehen in C.
- **F. Verallgemeinern:** Sobald eine Methode ein Lebenszeichen gibt, ihren Namen in `pipeline/gates/generalize`
  schreiben. Die Pipeline testet sie dann auf `handover_block`, `move_can_pot`, `place_shoe` und
  `pick_diverse_bottles`, bei −5 cm und im Hard-Setting. Für meta-trainierte Methoden muss die jeweilige Aufgabe dafür
  aus dem Training genommen werden (noch nicht automatisiert). Ein zweites Modell ist noch nicht ausgewählt.
- **Analysen pro Variante** (`fawam/report.py`, stündlich): Ergebnis mit Intervall und gepaartem Test gegen die
  Kontrolle, Greifhöhe Versuch 1 gegen 2, Kontext-Nutzung (Attention, Einfluss, Verschiebung der Aktion, je Block und je
  Kontext-Teil), Analyse gegen Orakel, Vorhersagefehler am Griff, Videos. Ausgabe in `~/dev/fawam-report`, daraus wird
  die Übersichtsseite veröffentlicht.

## Entscheidung vom 2026-11-12: kein Orakel mehr

- Die Methode muss ohne Simulator-Wissen und ohne menschliches Eingreifen funktionieren (Ziel: echter Roboter).
- Es laufen **keine weiteren Experimente mit Orakel-Analyse**. Die vorhandenen Orakel-Ergebnisse (A, B, C, E) bleiben
  als Referenz stehen, mehr nicht.
- Gestrichen: Meta-Training mit Orakel-Analyse (`failfix_type_*` ohne `_vlm`) und `p8_icl_oracle_meta`.
- Analyse, Korrektur und Kontext kommen vom VLM: Es bestimmt zuerst, **an welcher Stelle** der Versuch scheitert,
  und dann an dieser Stelle, **was zu ändern ist** (z. B. wie viel tiefer). Das gilt für das Adapter-Update und für
  alle Kontext-Varianten (ICL).
- Erlaubt ist nur, was Policy und Roboter selbst liefern (Beobachtungen, eigene Vorhersage, Vorhersagefehler,
  Greiferstellung) oder was ein VLM/LLM/anderes Modell daraus erzeugt.

### B2. Dual-Site-Varianten, nur Vorhersagefehler (eingereicht 2026-11-12, Stufe `dual_variants`)

Gleiches Protokoll wie B (4 Wiederholungen × 20 Episoden, −5 und −10 cm), Vergleich gegen `base` und `paper_P_dual`.

| Methode | Änderung gegenüber `paper_P_dual` | Frage |
|---|---|---|
| `paper_Pw_dual` | Frames nach Vorhersagefehler gewichtet | hilft die Gewichtung? |
| `paper_P_dual_s10`, `_s50` | 10 + 10 bzw. 50 + 50 Schritte | wo liegt das Optimum (150 war schlechter)? |
| `paper_P_dual_video` | nur Video-Loss | kommt der Gewinn aus der Video-Vorhersage? |
| `paper_P_dual_action` | nur Aktions-Loss | Kontrolle: nur Nachahmen der eigenen Aktionen |
| `paper_P_dual_vidtok` | Adapter nur auf Video-Tokens | reicht es, das Weltmodell anzupassen? |
| `paper_P_dual_n1`, `_n6` | 1 bzw. 6 Adaptions-Episoden | wie viele Daten braucht es? |
| `paper_P_dual_filt` | nur die 30 % Frames mit dem höchsten Fehler | |
| `paper_P_dual_lr3` | 3× Lernrate | größere statt mehr Schritte |
