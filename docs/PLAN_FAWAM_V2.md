# PLAN_FAWAM_V2 – ICL Stufe 1 und TTT-Linie

Stand: 21.11.2026. Ersetzt PLAN_STUFE1.md. Gilt für LingBot-VA auf RoboTwin; Testaufgabe bleibt move_playingcard_away,
Training auf den 7 bisherigen Trainingsaufgaben. (Vom Nutzer vorgegebener Plan, unverändert übernommen. Wie er
umgesetzt wurde und wo davon abgewichen wurde: [OVERVIEW.md](OVERVIEW.md), Abschnitt 6.)

## 0. Regeln für die Umsetzung
- **Nichts Neues in p8c/p8s/hid einreihen.** Laufende Jobs auslaufen lassen, Ergebnisse wie gewohnt in den Report.
- Neue Sweeps mit Präfix `p20_` (Diagnose), `p21_` (ICL), `p22_` (TTT). Bestehende Ergebnisse nicht anfassen.
- **Aktionsteil immer eingefroren.** Jeder Adapter ist nur im Video-Durchlauf aktiv (`action_mode=False`); der
  Aktions-Durchlauf rechnet mit Original-Gewichten und liest nur die veränderten Video-K/V. Kein `--train_base`.
- **Standard-Adapter ist Adapter+** (war im eigenen Paper besser als LoRA). LoRA r16 läuft als Vergleich mit.
- Erfolg korrigiert wie bisher (weggeflogen = Fehlschlag).
- Statistik: Fisher-Test, alle Methoden auf denselben Seeds.
- **Screening vor Vollauswertung:** erst 20 Seeds auf der Testaufgabe; nur klar vorne liegende Varianten bekommen 27
  Seeds pro Einstellung. 20 Seeds erkennen nur große Effekte (4/20 vs 10/20: p = 0,10; 2/20 vs 12/20: p = 0,002).
- **Alles sofort einreichen (21.11.), als Abhängigkeitskette.** Ziel: so viel wie möglich vor der HoreKa-Wartung am
  23.11. fertig. Nichts wartet auf eine manuelle Entscheidung; wo der Plan früher „nach Ergebnis X“ sagte, gilt jetzt
  eine automatische Regel (siehe Abschnitt 4).
- Abhängigkeiten über `sbatch --dependency=afterok:<id>`; Gates als kleine CPU-Jobs, die ihr Ergebnis in eine
  Gate-Datei schreiben (`results/gates/<id>.json`). Abhängige Jobs lesen die Datei beim Start und beenden sich sofort
  mit Status „übersprungen“, wenn ihr Gate nicht bestanden ist.
- Job-Limit 100 pro Account: Job-Arrays verwenden. Was nicht ins Limit passt, reiht der bestehende Pipeline-Cron nach
  Priorität nach (Abschnitt 4).
- Wartung: Reservierung mit `scontrol show res` prüfen; `--time` so setzen, dass Jobs vor Beginn der Wartung enden
  können. Alle Trainings und Evals müssen nach Abbruch wieder aufsetzen können (Checkpoint pro 250 Schritte, Eval
  überspringt fertige Seeds).
- Jede Entscheidung, die hier offen ist, als Eintrag „offen (Nutzer)“ ins Journal, nicht selbst festlegen.
- Report: neue Abschnitte „Diagnose p20“, „ICL Stufe 1 p21“, „TTT p22“ im bestehenden Artifact-Report.

## 1. Diagnose (p20)
| ID | Frage | Vorgehen | Entscheidet |
|---|---|---|---|
| D1 | Ist die Video-Vorhersage auf Aktionen konditioniert (Prognose) oder ein Plan? | Gleicher Zustand und Kontext; im KV-Cache verschiedene Aktions-Chunks (Original, ±2 cm in z, Zufall); Änderung des nächsten Video-Chunks messen (Latent-L2, Greiferposition per Sonde). | Ob TTT-P7 als Prognose-Modell taugt |
| D2 | Wo scheitert die Basis? | 20 Seeds je Einstellung: Tisch −5/−3/+3/+5 cm; versteckter Versatz y ±3/±5, x ±3/±5 cm; Gain 0,7/0,85/1,15/1,3. | Einstellungen für Stufe 1 und TTT (≥ 70 % Fehlschläge) |
| D3 | Rettet der Experte ab kurz vor dem Fehler? | Fehlversuche aus D2: Snapshot 1/2/3 Chunks vor Greifer-Schließen, dann Neuplaner (tf_replan) bis Ende. | Reset-Abstand |
| D4 | Reicht ein perfektes Video in den neuen Einstellungen? | tf_replan für +3/+5 cm, y ±5 cm, Gain 0,7/1,3. | Nur Einstellungen mit ≥ 70 % kommen in den Video-Weg |
| D5 | Leak | Linearer Probe des versteckten Parameters aus dem ersten Bild. | Info |
| D6 | Orakel-Phasen | Annähern → Absenken → Schließen → Heben → Transport → Ablegen aus Sim-Ereignissen. | Grundlage für T6 / VLM-Test |
| D7 | Erkennt ein VLM die Phasen? | Qwen3-VL-32B labelt je Frame die Phase; Genauigkeit gegen D6 (±1 Frame). | Ob T6 ohne Orakel laufen kann |

## 2. ICL-Linie, Stufe 1 (p21)
**Frage:** Liest das Modell den Fehlversuch, wenn der Fehlertyp bekannt, der Parameter aber unbekannt ist?
- Einstellungen: **S-A** Tisch ±3–6 cm; **S-B** versteckter seitlicher Versatz y (und x) ±3–5 cm (zählt für die
  Aussage). Beide Vorzeichen balanciert. Gate nach D2/D4: < 70 % Basis-Fehlschläge oder < 70 % tf_replan → Trainings
  dieser Einstellung übersprungen.
- Daten: Basis unter verstecktem Parameter, nur Fehlversuche; Ziel = Experte ab 1/2/3 Chunks vor dem
  Greifer-Schließen; Vergangenheit = eigener Versuch bis zum Reset-Punkt; Gegenbeispiele (gleicher Start, anderer
  Parameter); 10 % irrelevanter Kontext (gelungener Versuch, Ziel ohne Korrektur); ≥ 600 Paare pro Einstellung.
- Kontexte C0 (keiner, eigenes Training), C1 Mismatch im Fenster (7 Frames), C2 C1 + echte Frames, C3 C1 + echt +
  vorgestellt; Zusätze (S-B, A1): C4 skalarer Fehler, C5 C2 + Greifer, C6 C2 + Phase, C7 C2 + Orakel-Text, C8 nur
  Orakel-Text.
- Architekturen: A1 Adapter+ (Video, Kontext als Tokens), A2 LoRA r16, A3 Gated Cross-Attention (Flamingo), A4
  Kontext-Blöcke (VACE).
- Zeitlich ausgerichteter Mismatch (S-B, A1 + C2): T1 harter Schalter, T2 weiches Gate σ(a|e|+b), T3 Gate aus
  Hilfskopf, T4 kontextabhängiges tanh-Gate pro Layer, T5 Ausrichtung per nächstem Nachbarn, T6 per Orakel-Phase.
- Training: nur Flow-Matching auf Video-Tokens, 1500 Schritte, gleiche Lernrate; Ablation Chunk-Ziehung.
- Auswertung: Negativ-Filter (Val-Loss mit Kontext < C0, mit fremdem Kontext darüber) → G2 offline (Info) →
  Sim-Screening (20 Seeds je Einstellung und Vorzeichen, C0 immer) → Vollauswertung bei ≥ 25 pp über C0 (27 Seeds,
  Kontrollen C0 und fremder Kontext; bestanden: ≥ 20 pp gepoolt, p < 0,05, beide Vorzeichen, fremd ≤ C0) → Schaden auf
  Basis-Erfolgs-Seeds.

## 3. TTT-Linie (p22)
**Frage:** Kann ein Adapter, der nur auf den Episoden dieser einen Szene trainiert wird, die Fehler beheben?
Max. 3 Retries pro Seed. Einstellungen Tisch −5 cm, Gain 0,7 und 1,3, Versatz y ±5 cm. Adapter+ (Standard), LoRA r16,
Dual-Site. Raster Schritte {10, 50, 200} × Lernrate {1e-4, 1e-3} nur für P0. 20 Seeds Screening, 27 bei ≥ 25 pp über
„Retry ohne Lernen“.

| ID | Idee |
|---|---|
| P0 | Reiner Vorhersagefehler: Flow-Matching auf die echten Frames der Fehl-Episode, dann Retry |
| P1 | Wie P0, Loss nur auf Chunks mit Fehler über dem Median |
| P2 | Wie P0, Adapter-Stärke pro Chunk = normierter Fehler des letzten Versuchs |
| P3 | Exploration in der Vorstellung: 8 Video-Pläne, gewählt der, der sich im Fehlerfenster am stärksten vom Fehlversuch unterscheidet |
| P4 | P3, nach dem ersten gelungenen Retry Adapter auf dessen echtes Video; dann neue Seeds ohne Exploration |
| P5 | P4 + Hinge-Term, der die Vorhersage im Fehlerfenster vom Fehlversuch wegdrückt |
| P6 | Adapter über Retry 1–3 weitertrainiert |
| P7 | Prognose-Modus + Auswahl unter N Aktions-Chunks (hängt am D1-Gate) |

Erwartung: P0 hilft bei Gain (Dynamik), kaum beim Tisch (Plan-Fehler).

## 4. Einreichen – Prioritäten
Prio 1: D1–D7, TTT P0 (+ Raster), P1, P2, Retry ohne Lernen, ICL-Datensammlung · Prio 2: Gates, ICL-Kernsatz A1 × C0–C3
× S-A/S-B, TTT P3–P6 und Adapter-Vergleich · Prio 3: ICL A2–A4, Chunk-Ablation, TTT P7 · Prio 4: T1–T6, C4–C8. An jedem
Training: Negativ-Filter → G2 + Screening → ggf. Vollauswertung.

## 5. Offen (Nutzer)
- Cosmos-Reason2-Lizenz und `hf auth login`.
- Übertragung auf VPP2/RoboDojo erst, wenn eine TTT-Variante auf RoboTwin funktioniert (laut Nutzer vorerst egal).
