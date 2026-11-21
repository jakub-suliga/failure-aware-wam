# Failure-Aware WAM – Gesamtüberblick

Stand: 21.11.2026. Dieses Dokument erklärt das Projekt für Leser, die nicht dabei waren: was das Ziel ist, welche
Stränge verfolgt wurden, was getestet wurde, was herauskam, welche Schlüsse gezogen wurden und warum die aktuellen
Experimente so aussehen. Detailtabellen aller Läufe stehen in [RESULTS.md](RESULTS.md) (automatisch erzeugt), der
aktuelle Arbeitsplan in [PLAN_FAWAM_V2.md](PLAN_FAWAM_V2.md). Ältere Planungsdokumente ([GOAL.md](GOAL.md),
[PLAN.md](PLAN.md), [PLAN_HID.md](PLAN_HID.md), [ICL.md](ICL.md), [METHOD.md](METHOD.md)) sind historisch.

---

## 1. Worum es geht

**Ziel:** Ein *World-Action-Model* (WAM) soll aus dem eigenen Fehlversuch lernen, ohne dass ein Mensch oder der
Simulator ihm die Korrektur vorgibt. Ein WAM sagt die Zukunft als Video voraus und leitet daraus Aktionen ab. Wenn es
scheitert, hat es zwei Dinge, die eine normale Policy nicht hat: seine eigene **Vorstellung** (was es erwartet hat)
und die **Realität** (was passiert ist). Die Abweichung zwischen beiden – der **Vorhersagefehler** – zeigt, wann und wo
das Modell die Welt falsch eingeschätzt hat. Die Idee: Diese Information (plus ggf. eine Analyse durch ein
Vision-Language-Modell, VLM) reicht, damit das Modell im nächsten Versuch richtig handelt.

Zwei Wege, die Information zurück ins Modell zu bringen:

1. **In-Context-Learning (ICL):** Der Fehlversuch steht beim nächsten Versuch im Kontext (KV-Cache) des Modells. Kein
   Gradientenschritt zur Testzeit. Damit das Modell Kontext überhaupt nutzt, wird ein kleines Zusatzmodul
   *meta-trainiert* – aber nie auf der Testaufgabe.
2. **Test-Time-Training (TTT):** Nach dem Fehlversuch wird ein kleiner Adapter auf den eigenen Episoden dieser einen
   Szene trainiert (z. B. nur mit dem Vorhersagefehler), dann wird erneut versucht.

**Harte Randbedingungen** (Entscheidungen des Projektverantwortlichen, siehe Abschnitt 7):
- Zur Testzeit nur Information, die ein echter Roboter auch hätte: Kamerabilder, eigene Vorhersage, Vorhersagefehler,
  eigene Aktionen/Greiferzustand und was ein VLM daraus macht. **Kein Simulator-Orakel** (keine wahre Δz, kein
  Fehlerzeitpunkt aus dem Simulator) – außer in ausdrücklich als Diagnose markierten Tests.
- Die Testaufgabe kommt im Training nie vor.
- Basismodell (LingBot-VA) bleibt eingefroren; trainiert werden nur Zusatzmodule.

## 2. Aufbau und Begriffe

- **LingBot-VA** (Policy-Server, `lingbot-va/`, Checkpoint `lingbot-va-posttrain-robotwin`): autoregressives
  Video-Action-Modell auf Wan-Basis. Pro *Chunk* (2 latente Frames, 32 Aktionen) entrauscht es erst das Video der
  nächsten Frames, dann die Aktionen, die auf dieses vorgestellte Video schauen. Danach werden die *echten* Beobachtungen
  und die ausgeführten Aktionen in den KV-Cache geschrieben. Video- und Aktions-Tokens laufen durch **dieselben** 30
  Transformer-Blöcke (kein getrennter „Video-Expert“) – deshalb sind „Video-only“-Adapter hier Adapter, die nur im
  Video-Durchlauf aktiv sind.
- **RoboTwin 2.0** (Simulations-Client, `RoboTwin/`): bimanueller Aloha-AgileX-Roboter, viele Aufgaben.
- **Testaufgabe:** `move_playingcard_away` (Kartenbox greifen und wegschieben). Basis bei normaler Tischhöhe 72/72,
  bei −5 cm 1/72, bei −10 cm 0/72. **Trainingsaufgaben** (7): place_mouse_pad, pick_diverse_bottles, stack_blocks_two,
  place_shoe, move_can_pot, lift_pot, handover_block.
- **Erfolg (korrigiert):** RoboTwin wertet eine weggeschleuderte Box als Erfolg. Wir zählen nur Episoden, in denen das
  Objekt auf Starthöhe (±5 cm) und innerhalb von 0,6 m bleibt („weggeflogen“ = Fehlschlag).
- **Retry-Protokoll:** Versuch 1 = Basis. Nur Seeds, deren Versuch 1 scheitert, zählen für die Methode; gemessen wird,
  wie viele davon im zweiten (oder 2.–4.) Versuch gelingen. Kontrolle: derselbe Retry ohne Lernen (`retry_plain` /
  `ttt_retry`). Alle Methoden laufen auf denselben Seeds; Statistik mit Fisher-Test.
- **Orakel / tf_replan:** Diagnose-Werkzeuge mit Simulator-Wissen. *tf_replan* ersetzt die vorgestellten Frames
  durch echte Frames einer Planer-Fortsetzung ab dem aktuellen Zustand („perfektes Video“); ausgeführt werden nur die
  Aktionen des Modells.
- **Pipeline:** Alle Läufe sind Sweeps (`configs/sweeps/*.yaml`) aus SLURM-Jobs auf HoreKa. Ein Cron
  (`cluster/pipeline.py`, alle 20 min) reicht Jobs nach Priorität ein, prüft Gates und startet abhängige Stufen.
  Ergebnisse landen im Report (Artifact „Failure-Aware WAM“) und in `docs/RESULTS.md`.

## 3. Was wir über den Fehler wissen (Diagnose, Nov 2026)

Quelle: [FINDINGS.md](FINDINGS.md), p13/p14/p15.

1. **Fehler sind reproduzierbar und aus der eigenen Vorhersage erkennbar.** Die Distanz zwischen vorgestelltem und
   echtem Handgelenk-Latent erkennt Fehlschläge mit AUC 0,84–0,88, etwa 3 Latents vor dem Fehler.
2. **Tisch tiefer = Greifer schließt zu hoch.** Die kommandierte Greifhöhe geht nur 2–4 cm mit; bei −10 cm müsste die
   Aktion außerhalb des Trainingsbereichs liegen (Sättigung am Rand des Aktionsraums).
3. **Die Vorstellung „glaubt“ an den Erfolg.** Das Modell stellt sich den Griff auf derselben zu hohen Höhe vor, aber
   danach eine angehobene Karte. Der Vorhersagefehler sagt also *wann/wo*, aber nicht *wie* korrigiert werden muss.
4. **Ein perfektes Video reicht oft (p13).** Mit tf_replan: −5 cm 27/27 (Basis 1/72), −10 cm 8/27, 0 cm 26/27. Der
   Aktionsteil folgt einem guten Video – bei −10 cm nicht weit genug. Leicht verrauschtes Video (bis 25 % Rauschen,
   p14) hilft genauso; ab 50 % bricht es ein (13/27, 0/27). **Schluss: Wenn sich die Video-Vorhersage verbessern
   lässt, verbessern sich die Aktionen mit.** Das ist der Grund für den aktuellen Plan (nur der Video-Teil wird
   angepasst).
5. **Andere Störungen scheitern auf der Testaufgabe kaum (p15):** Objektposition ±10 cm 25/27 ok, Drehung ±90° 27/27,
   Kamera 23/27, Startpose 14/27 (tf_replan löst davon nur 46 %).

## 4. Die Stränge bisher – Frage, Test, Ergebnis, Schluss

Zahlen: gelöste Fehlversuche / versuchte Fehlversuche auf der Testaufgabe, −5 cm | −10 cm, Erfolg korrigiert.

### A. Adapter-Update nach dem Fehlversuch (Learn-from-Failure, p5a/p5b)
- **Frage:** Reicht ein kleines Gewichts-Update nach einem Fehlversuch? Welches Signal braucht es?
- **Ergebnis:** Mit Orakel-Korrektur 21–23/25 | 19–20/30. Nur mit Vorhersagefehler (und Rollout-/Vorstellungs-Signalen)
  1–3/25 | 0/30. Mit VLM-Analyse 2/28 | 0/20. Retry ohne Lernen 0/26 | 0/30.
- **Schluss:** Das Modell kann eine *richtige* Korrektur sofort umsetzen. Der Engpass ist, wo die Korrektur
  herkommt – weder Vorhersagefehler allein noch das VLM liefern sie.

### B. Dual-Site-Adapter (Übertragung des eigenen Papers, p5f)
- **Frage:** Funktioniert *Few-Shot Residual Adaptation* (Adapter innen + Residual auf dem vorhergesagten Latent,
  trainiert nur mit dem Vorhersagefehler aus 3 Episoden) auf einem WAM?
- **Ergebnis:** Paper-Protokoll (Erfolg nach Adaption): Basis 2/68 | 0/68; `paper_P_dual` 22/68 | 0/68; beste
  Varianten nur 1 Adaptions-Episode 33/76, nur Video-Tokens 30/68 (je ~40 %) | 0. Mit Orakel 25/68 | 13/68.
- **Schluss:** Bei −5 cm hilft reiner Vorhersagefehler deutlich (3 % → ~40 %), bei −10 cm gar nicht. Die
  Video-Tokens-Variante ist so gut wie die volle – ein Hinweis, dass der Effekt über die Vorhersage läuft.

### C. Kontext ohne Training (p6a, p6v)
- **Frage:** Nutzt LingBot-VA den Fehlversuch im Kontext von sich aus?
- **Ergebnis:** 0–1 von ~25 mit Orakel-Text, 0 mit VLM-Text; Kontrolle 0/24.
- **Schluss:** Nein. Kontext-Nutzung muss trainiert werden.

### D. Meta-trainierte Kontext-Adapter (p8, p8f, p8d, p8t, p8s, p8c)
- **Frage:** Lernt ein Adapter, der auf `[Fehlerkontext | erfolgreicher Versuch]` trainiert wird, den Kontext zu
  lesen?
- **Testtyp im Training (p8, Tischhöhe):** 24–28/30 gelöst – aber dieselben Adapter *ohne* Kontext genauso gut, reines
  Fine-Tuning 23/24 | 30/30. **Schluss:** Der Adapter lernt „tiefer greifen“ auswendig, der Kontext ist überflüssig.
- **Ungesehener Fehlertyp** (Training auf anderen Störungen, Test auf Tisch): p8d (Startpose + Kamera) bis 16/26 mit,
  12/23 ohne Kontext; p8t („7 Typen“) LoRA **11/25 mit gegen 2/29 ohne Kontext (p = 0,003)** – einziges positives
  Zeichen, aber TTT-Adapter umgekehrt (0/24 gegen 9/25); p8s (stärkere Störungen P7d, detaillierter VLM-Text, Basis
  mittrainiert) mit ≈ ohne Kontext; p8c (kleine Kontexte, nur Fenster um den Fehler) mit ≤ ohne Kontext
  (10/25 vs 13/23 usw.). −10 cm durchgehend 0–1/30.
- **Schluss:** In keinem Datensatz wurde der Kontext verlässlich genutzt. Ursache: In den Daten ist die Korrektur
  entweder im aktuellen Bild sichtbar oder immer gleich, also muss das Modell den Fehlversuch nicht lesen
  (Memorisierung, vgl. Yin et al. 2020).

### E. Versteckte Kalibrier-Offsets „hid“ (p10–p12)
- **Warum:** Daten, bei denen der Fehler *nur* im Fehlversuch erkennbar ist: pro Seed ein unsichtbarer Versatz o der
  ausgeführten Endeffektor-Position (Translation, Gain, Drehung), in beiden Versuchen gleich; Experten-Fix kompensiert.
- **Ergebnis:** Leak-Test: o aus dem ersten Bild kaum ablesbar (R² ≤ 0,12), aus dem Fehlerfenster gut (z 0,57).
  Bester Mechanismus (Residualkopf C2, Mismatch im Fenster) senkt den Loss um 9,5 %, ändert in der Simulation aber
  nichts (C2 4/65, ohne Kontext 3/63, fremder Kontext 3/63, Retry 2/63). LoRA C1: Kontext ungenutzt.
- **Schluss:** Selbst wenn die Information nur im Fehlversuch steckt, liest ihn bisher kein Mechanismus so, dass sich
  das Verhalten ändert. Die Aktionsänderung war zu klein (|dv| 0,009).

### F. VLM als Fehleranalyse
- **Ergebnis:** Qwen3-VL erkennt „missed grasp“ in ~80 %, nennt aber fast nie „zu hoch“ oder eine Korrektur (Δz ≈ 0
  in 94–97 %). Erst Zeitpunkt finden, dann fragen: Zeitpunkt nur in 23–35 % richtig. Fein-getunte Fehler-VLMs
  (ViFailback, RoboFAC, Cosmos-Reason1) sagen „tiefer“ ebenso oft bei Fällen ohne Höhenfehler.
- **Schluss:** Das VLM wird „so wie es ist“ genutzt (nur Fehlertyp/Beschreibung); eine Korrekturrichtung liefert es
  nicht.

### Gesamtschluss bis 20.11.
- Eine richtige Korrektur wird umgesetzt (Orakel-Updates), aber keine Quelle ohne Orakel liefert sie zuverlässig.
- Kontext wird nicht gelesen, solange die Daten das nicht erzwingen.
- Ein besseres *Video* führt zu besseren Aktionen (p13/p14). → **Neuer Fokus: nur die Video-Vorhersage verbessern**
  (Aktionsteil eingefroren), entweder per ICL (Stufe 1) oder per TTT auf der eigenen Szene.

## 5. Aktueller Plan v2 (seit 21.11.) – was jetzt getestet wird und warum

Vollständiger Plan: [PLAN_FAWAM_V2.md](PLAN_FAWAM_V2.md). Drei Linien:

### p20 Diagnose (Grundlage für die Entscheidungen)
| ID | Frage | Umsetzung |
|---|---|---|
| D1 | Folgt das vorhergesagte Video den Aktionen (Prognose) oder ist es ein Plan? | `fawam/d1_condition.py`: KV-Cache einer echten Episode bis Chunk c nachbauen, Chunk c mit Original / z ±2 cm / Zufalls-Aktionen schreiben, Chunk c+1 mit gleichem Rauschen vorhersagen; Latent-Änderung und vorgestellte Greiferhöhe (Pose-Sonde) gegen Rausch-Untergrenze. Gate `results/gates/p20_d1.json` |
| D2 | Wo scheitert die Basis? | Sweep `p20_d2_base`, 16 Einstellungen × 21 Seeds |
| D3 | Rettet der Experte ab kurz vor dem Fehler? | Sweep `p20_d3_snap` (Snapshot-Treiber, s. u.): von 1/2/3 Chunks vor dem Greifer-Schließen (a) Planer-Trajektorie, (b) Policy mit tf_replan bis Ende |
| D4 | Reicht ein perfektes Video in den neuen Einstellungen? | Sweep `p20_d4_tf` (tf_replan) |
| D5 | Ist der versteckte Versatz im ersten Bild sichtbar? | `fawam/hid_leak.py` auf den S-B-Daten |
| D6 | Orakel-Phasen | `fawam/phases.py`: approach/lower/close/lift/transport/release aus Greiferbefehl, EE-Höhe, Objektpose |
| D7 | Erkennt ein VLM die Phasen? | `fawam/vlm_phase.py` (Qwen3-VL-32B) gegen D6, ±1 Frame Toleranz |

### p21 ICL Stufe 1 – „Liest das Modell den Fehlversuch, wenn der Parameter unbekannt ist?“
- **Einstellungen:** S-A Tisch ±3–6 cm (`icl_sa`), S-B versteckter seitlicher Versatz y ±3–5 cm, x auf der Hälfte der
  Seeds (`icl_sb`); Vorzeichen über die Seed-Parität balanciert; `*_v1` = gleiche Seeds (gleiche Szene) mit
  umgekehrtem Vorzeichen → **Gegenbeispiele**: gleicher Startzustand, anderes Ziel. S-B ist der strengere Test (kein
  Kontakt-Hinweis wie „absenken bis Tisch“, im ersten Bild kaum ablesbar) und zählt für die Aussage.
- **Warum Snapshot-Daten:** Bisher war das Ziel eine komplette Experten-Episode vom Start an; das Modell konnte die
  Korrektur aus der ganzen Episode auswendig lernen. Jetzt ist die *Vergangenheit* der eigene Versuch bis kurz vor dem
  Fehler und das *Ziel* nur die Experten-Fortsetzung ab dort. Den Fehler selbst sieht das Modell nur im Kontext.
- **Daten** (`p21_data_sa/sb`, Treiber `snap`): Basis-Versuch mit Snapshot pro Chunk; ab 1/2/3 Chunks vor dem
  Greifer-Schließen spielt der Planer (`play_once` ab aktuellem Zustand) die Fortsetzung ab, kompensiert für den
  Versatz; der Server kodiert nur (`skip_generation`). Ein Paar pro erfolgreicher Fortsetzung; gelungene Versuche
  liefern „irrelevanten Kontext“ (10 % der Samples, Ziel ohne Korrektur). 7 Aufgaben × 2 Vorzeichen × 48 Seeds.
- **Sequenz im Training** (`fawam/icl_snap.py`): `[Kontext des Fehlversuchs | eigene Vergangenheit | Experten-Fortsetzung]`,
  Loss nur Video-Flow-Matching auf der Fortsetzung, 1500 Schritte, Checkpoint alle 250 Schritte.
- **Kontexte:** C0 keiner (eigenes Training) · C1 echt − vorgestellt im 7-Frame-Fenster um das Greifer-Schließen · C2
  + echte Frames · C3 + vorgestellte Frames · C4 nur skalarer Fehler pro Frame · C5 C2 + Greiferzustand · C6 C2 +
  Phasen-Text · C7 C2 + Orakel-Text · C8 nur Orakel-Text (C6–C8 sind Diagnosen).
- **Architekturen:** A1 Adapter+ nur im Video-Durchlauf (Standard) · A2 LoRA r16 nur Video · A3 Gated
  Cross-Attention (Flamingo-artig, jeder 3. Block, tanh-Gates mit 0 initialisiert, Kontext nur über die
  Cross-Attention) · T1 Adapter nur, wo der zeitlich ausgerichtete Fehler des letzten Versuchs über dem Median lag ·
  T2 Adapter-Stärke σ(a·e+b), a, b gelernt · T4 tanh-Gate pro Layer aus dem gepoolten Kontext · T5 wie T2, Ausrichtung
  per nächstem Nachbarn im Latentraum · Chunk-Ablation (Fenster 50 % / Loss ×4 um den Fehlerzeitpunkt).
- **Test** (Client `_icl_rewind`, Presets `icl2_*`): Versuch 1 = Basis; bei Fehlschlag Rücksprung in den Snapshot 2
  Chunks vor dem Greifer-Schließen, Server bekommt Adapter + Kontext + eigene Vergangenheit, Retry bis Ende. Auch
  Basis-Erfolge bekommen diesen Retry (misst Schaden). `_foreign`: Kontext aus einem Fehlversuch mit umgekehrtem
  Vorzeichen (Bank `p21_bank`).
- **Kette:** Daten fertig + Einstellungs-Gate (D2 ≥ 70 % Fehlschläge, D4 ≥ 70 % tf_replan) → Training → Negativ-Filter
  (Val-Loss eigen < C0 und fremd > C0 auf zurückgehaltenen Seeds) → Screening `p21_screen` (40 Seeds = 20 je
  Vorzeichen; C0 läuft immer) → Vollauswertung (27 je Vorzeichen + C0 + fremder Kontext), wenn Variante − C0 ≥ 25 pp.

### p22 TTT – „Kann ein Adapter, nur auf dieser Szene trainiert, den Fehler beheben?“
- **Warum:** Dual-Site (B) zeigte bei −5 cm ~40 % nur mit Vorhersagefehler. Jetzt systematisch, Video-only, mit bis
  zu 3 Retries und mehreren Fehlerarten: Tisch −5 cm (Plan-Fehler), Gain 0,7 / 1,3 (Dynamik-Fehler), Versatz y ±5 cm.
- **Varianten** (`fawam/methods.py`, `ttt_*`): P0 Vorhersagefehler auf der ganzen Episode (Raster 10/50/200 Schritte ×
  1e-4/1e-3) · P1 nur Chunks mit Fehler über Median · P2 Adapter-Stärke pro Chunk = Fehler des letzten Versuchs · P3
  8 Video-Pläne im Fehlerfenster, gewählt der unähnlichste zur Realität des Fehlversuchs · P4 P3 + Adapter auf das echte
  Video des ersten gelungenen Retrys, danach neue Seeds ohne Exploration · P5 P4 + Hinge, der die Vorhersage im
  Fehlerfenster vom Fehlversuch wegdrückt · P6 Adapter über die Retries weitertrainiert · LoRA- und Dual-Site-Vergleich
  · Kontrollen `ttt_retry` (gleiche Retries ohne Lernen) und das frühere `lff_ttt_filtered` (4/27).
- **Erwartung (vorab festgehalten):** P0 hilft bei Gain (das Modell lernt, wie sich die Welt bewegt), kaum beim Tisch
  (der Adapter lernt „Karte bleibt liegen“, nicht „tiefer greifen“). Hilft P0 beim Tisch doch, ist das ein wichtiger
  Befund.
- **Kette:** Screening 20 Seeds → Vollauswertung 27 Seeds, wenn Variante − Retry ≥ 25 pp.

### Erste Beobachtungen (21.11., noch unvollständig)
- D2: Tisch +3/+5 cm und Versatz ±3 cm scheitern auf der Testaufgabe kaum (0–1 von 5–11 Seeds), y ±5 cm in 3 von 4–5,
  −5 cm Tisch wie bekannt fast immer. Das Einstellungs-Gate poolt über die Stufen einer Einstellung; liegt S-A oder S-B
  unter 70 %, werden deren Trainings automatisch übersprungen (Regel aus dem Plan; als „offen (Nutzer)“ im Journal).
- Smoke-Tests: TTT-Lerner 50 Schritte ≈ 75 s, Adapter+ (Video) 47,9 M Parameter, LoRA 11,8 M; P2-Gate und
  P3-Exploration greifen; Snapshot-Experte rettet ab 1 und 2 Chunks vor dem Schließen; Rewind-Retry schreibt Kontext +
  Vergangenheit korrekt in den Cache.

## 6. Umsetzung von Plan v2 – Abweichungen und offene Punkte

- **Abhängigkeiten:** statt reiner `sbatch --dependency`-Ketten über den bestehenden Pipeline-Cron
  (`cluster/pipeline_v2.py`): Das Account-Limit von 100 Jobs erlaubt nicht, ~500 Jobs sofort mit Abhängigkeiten
  einzureichen. Gates werden wie geplant als `results/gates/<id>.json` geschrieben; Trainings lesen ihr Gate zusätzlich
  beim Start (`cluster/p21_train_job.sh`) und enden mit „skipped“, wenn es nicht bestanden ist.
- **Wiederaufsetzen:** Trainings setzen am letzten 250er-Checkpoint fort; der Client überspringt Seeds, deren erster
  Versuch schon aufgezeichnet ist; die Pipeline reicht unvollständige Zellen einmal nach.
- **Nicht umgesetzt:** A4 (VACE-Kontextblöcke: VACE addiert Hinweise tokenweise und braucht einen räumlich-zeitlich
  ausgerichteten Kontext), T3 (Gate aus einem Hilfskopf), T6 (Ausrichtung per Phase), P7 (Prognose-Modus; wird vom
  D1-Gate ohnehin übersprungen, falls das Video nicht aktionskonditioniert ist), G2 (Offline-Greifposition, nur Info).
- **Abweichung T1–T5:** Kontext ist das Fehlerfenster wie C2, nicht pro Chunk ein eigener Ausschnitt; das Gate nutzt
  den ausgerichteten Fehler des letzten Versuchs.
- **Offen (Nutzer):** Gate-Pooling (s. o.), Lernrate ICL (gesetzt 4,3e-4 wie bisherige Meta-Trainings), Reset-Abstand
  im Test (k = 2, bis D3 vorliegt), Lesart „fremder Kontext darüber“ (umgesetzt als fremd > C0), Cosmos-Reason2-Lizenz.
- **Kapazität:** ~30 GPU-Jobs laufen gleichzeitig; die eingereihten ~460 Jobs brauchen grob 30 h. Vor der Wartung am
  23.11. realistisch: Diagnose, Daten, ICL-Kerntrainings mit Filter und Screening, TTT-Kern und Raster. Unsicher: alle
  Vollauswertungen, TTT P3–P6, die Prio-4-Varianten.

## 7. Grundsatzentscheidungen und warum

| Entscheidung | Warum |
|---|---|
| Kein Orakel in Methoden (12.11.) | Die Methode muss auf einem echten Roboter ohne Simulator-Wissen funktionieren. Orakel nur als Diagnose (p13/p14, C7/C8). |
| VLM so nutzen, wie es ist (13.11.) | Es lieferte keine Korrekturrichtung; statt darauf zu warten, werden nur die verlässlichen Teile (Fehlertyp, Beschreibung) genutzt. |
| Viele Fehlertypen mit wenig Episoden (17.11.) | Mit nur einem Fehlertyp lernt der Adapter die Korrektur auswendig statt den Kontext zu lesen. |
| Testaufgabe nie im Training | Sonst ist nicht zu trennen, ob der Kontext oder das Training hilft. |
| „Ohne Kontext“-Kontrolle für jeden Adapter | Die entscheidende Kontrolle: trennt „nutzt den Kontext“ von „wurde durch Training besser“. |
| Erfolg korrigiert (15.11.) | Sonst zählen weggeschleuderte Boxen als Erfolg (10 von 12 „Erfolgen“ bei −5 cm). |
| Nur Video anpassen (20./21.11.) | p13/p14: Mit gutem Video folgen die Aktionen. Der Aktionsteil bleibt eingefroren, damit klar ist, dass der Effekt über die Vorhersage läuft. |
| Adapter+ als Standard | War im eigenen Paper besser als LoRA; LoRA läuft als Vergleich. |

## 8. Wo was liegt (Plan v2)

| Was | Datei |
|---|---|
| Neue Störungen (fester/zufälliger Versatz, Gain, Tisch mit Vorzeichen) | `fawam/hid_offset.py`, `fawam/sim_disturb.py`, `configs/robotwin_task_config/{table_h_*,hid_*,icl_s*}.yml` |
| Snapshot-Treiber, Rewind-Retry, Überspringen fertiger Seeds | `patches/lingbot-va.patch` (Client `eval_polict_client_openpi.py`: `_snap_continuations`, `_icl_rewind`) |
| Generischer Neuplaner | `fawam/teacher_force.py` (`replan_poses(generic=True)`) |
| ICL-Datensatz, Kontexte, Rewind-Spezifikation | `fawam/icl_snap.py` |
| Training (Snapshot-Paare, A3, Gates, Validierung eigen/fremd/ohne, Resume, P5-Hinge) | `fawam/train_adapter.py`, `fawam/learn.py`, `cluster/p21_train_job.sh` |
| Adapter: Video-only-LoRA, Gated Cross-Attention, Gates | `fawam/adapters.py` |
| Server: Gate pro Chunk (P2, T1/T2/T5), Exploration (P3), Prefill-Bereiche, Cross-Attention-Kontext | `fawam/server.py` |
| Methoden-Presets `ttt_*`, `icl2_*` | `fawam/methods.py` |
| Diagnosen | `fawam/d1_condition.py`, `fawam/phases.py`, `fawam/vlm_phase.py`, `fawam/hid_leak.py` |
| Gates, Zusammenfassung für den Report | `fawam/gates_v2.py` → `results/gates/*.json`, `results/v2_summary.json` |
| Ketten | `cluster/pipeline_v2.py` (in `cluster/pipeline.py` eingehängt) |
| Sweeps | `configs/sweeps/p20_*.yaml`, `p21_*.yaml`, `p22_*.yaml` |
| Code-Check vor dem Einreichen | `cluster/dev_check.sh` (alle neuen Pfade einmal kurz, auf `dev-gpu-h100`) |
| Journal (Probleme, Analysen, offene Punkte) | `analysis/journal.json` → Report-Tab „Probleme & Analysen“, Tab „Plan v2“ |
