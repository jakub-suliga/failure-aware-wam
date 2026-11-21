# Ergebnisse Failure-Aware WAM

Automatisch erzeugt aus dem Report (`fawam/report_md.py`) · Stand der Daten: 2026-11-21 16:07 · Testaufgabe `move_playingcard_away`.

Erfolg zählt nur, wenn das Objekt auf dem Tisch bleibt („weggeflogen“ = vom Simulator als Erfolg gewertet, hier Fehlschlag). Brüche: gelöst / versucht. Ziele und Plan: [GOAL.md](GOAL.md), [PLAN.md](PLAN.md), [PLAN_HID.md](PLAN_HID.md); ältere Befunde: [FINDINGS.md](FINDINGS.md).

## Probleme, Analysen, laufende Arbeit

Stand: 2026-11-21 16:10

### Was gerade läuft

| Schritt | Status |
|---|---|
| Plan v2 eingereicht 21.11. (PLAN_FAWAM_V2): p8c/p8s/hid nichts Neues; neue Sweeps p20_/p21_/p22_ | läuft |
| p20 D2 Basis (16 Einstellungen × 21 Seeds): Jobs 139120–127, 139153–160 | läuft, Ende ~16:45 |
| p20 D4 tf_replan (9 Einstellungen): Jobs 139134–141, 139161 | läuft, Ende ~17:30 |
| p20 D1 Aktions-Konditionierung (offline, 24 Episoden × 4 Chunks × 5 Varianten) | Queue Prio −4 (als nächstes), Ende ~19:00 |
| p20 D3 Snapshot-Experte + Policy/tf ab 1/2/3 Chunks (8 Einstellungen, 24 Jobs) | Queue Prio −2, Ende ~22.11. 02:00 |
| p21 Daten S-B (56 Jobs, 7 Aufgaben × 2 Vorzeichen-Varianten × 48 Seeds), danach S-A (56) | S-B wartet auf GPUs (Ende ~19:30); S-A danach (Ende ~23:30) |
| p21 Fremdkontext-Bank (Testaufgabe, S-A/S-B, 2 Jobs) | Queue Prio −2, Ende ~22:00 |
| p22 TTT 16 Varianten × 5 Einstellungen × 20 Seeds (320 Jobs): Kern (Retry, P0, P1, P2) Prio −2; Raster, P3–P6, LoRA, Dual-Site, lff_ttt Prio −1 | Queue; Kern (Retry/P0/P1/P2, 80 Jobs) bis ~22.11. 06:00, Rest (240 Jobs) bis ~22.11. 20:00 |
| Automatische Ketten (cluster/pipeline_v2.py): D2+D4 → Einstellungs-Gate + D6 → D7; Daten+Gate → Trainings (A1×C0–C3, A2, Chunk-Ablation, C4–C8) → Negativ-Filter → Screening → Vollauswertung; p22 → Screening-Gate → Vollauswertung; D1 → P7-Gate | aktiv (Pipeline-Cron alle 20 min) |
| Kapazität 21.11. 15:50: ~30 GPU-Jobs laufen gleichzeitig, 68 warten (Priorität = Alter). ~460 Jobs in der Pipeline-Queue ≈ 30 h GPU-Zeit bei 30 GPUs. Keine Wartungsreservierung eingetragen (scontrol show res leer); Jobs, die nicht mehr vor die Wartung passen, warten und starten danach. | Info |
| Realistisch vor der Wartung (23.11.): D1–D7, Einstellungs-Gate, S-B- und S-A-Daten, ICL-Kerntrainings A1×C0–C3 (+A2/A3, Ablation) mit Negativ-Filter, Screening der bestandenen Varianten, TTT-Kern + Raster. Unsicher: TTT P3–P6/LoRA/Dual-Site, C4–C8/T1–T5-Trainings, alle Vollauswertungen. | Schätzung |

### Analysen

**Plan v2: Smoke-Tests 21.11.**

- TTT (hid y +5 cm, 1 Seed): P0, LoRA, P2, P3, P4, Dual-Site laufen bis Retry 3/4 ohne Fehler. Lerner: 50 Schritte ≈ 75 s; Adapter+ nur im Video-Durchlauf 47,9 M Parameter, LoRA r16 (nur Video) 11,8 M.
- P2: Adapter-Stärke pro Chunk = normierter Vorhersagefehler des letzten Versuchs (gemessen 0,0–1,0). P3: im Fehlerfenster 8 Video-Pläne, gewählt der am weitesten von der Realität des Fehlversuchs.
- Snapshot-Treiber: Testaufgabe unter Versatz y +5 cm, Basis scheitert; Experte ab 1 und 2 Chunks vor dem Greifer-Schließen erfolgreich (Schließen in Chunk 2).
- ICL-Datensatz: Kontexte C0–C8 bauen korrekt ([Kontext 7 Frames je Gruppe | eigene Vergangenheit | Experten-Fortsetzung], Loss nur auf der Fortsetzung). Rewind-Retry mit Null-Adapter läuft als Smoke.
- Code-Check im Dev-Job (cluster/dev_check.sh): Kompilieren, Importe, Gates, Datensatz (C0–C8 × Kontext-Modi × Gates × Chunk-Ziehung) ok; TTT-Lerner mit P5-Hinge und Video-LoRA ok; Trainings A3 (126 M Parameter) und T4 (48 M) laufen. Gefunden und behoben: Absturz, wenn unter den Validierungs-Seeds kein Fehlversuch ist.

**Schritt 0 (p15): Welche Fehler behebt das Planer-Video? (4 von 6 Typen fertig)**

- move_playingcard_away, 27 Seeds, Erfolg korrigiert. Basis | Planer-Video: Objektposition ±10 cm 25/27 | 22/27 · Drehung ±90° 27/27 | 25/27 · Kamera 23/27 | 21/27 · Startpose 14/27 | 18/27.
- Auf dieser Aufgabe erzeugen Objektposition, Drehung und Kamera kaum Fehlschläge (0–4 von 27). Damit ist dort nicht messbar, ob das Video sie behebt.
- Startpose: Von 13 Fehlschlägen der Basis löst das Planer-Video 6 (46 %, unter der 70-%-Schwelle).
- Auffällig: Wo die Basis schon gelingt, verliert das Planer-Video 2–4 von 23–27. Das fremde Video stört den Aktionsteil also etwas.
- hid nur z und Objektgröße liefen nicht (Konfigurationsdateien fehlten in RoboTwin/task_config) → behoben, neu eingereicht 21.11. 06:25.
- Vorläufiges Fazit: Außer Tisch erfüllt bisher kein Typ das Kriterium. Für Objektposition, Drehung und Kamera müsste Schritt 0 auf den Trainingsaufgaben laufen, wo sie öfter scheitern; dafür fehlt der Neuplaner für diese Aufgaben.

**P8c kleine Kontexte: kein Kontext-Gebrauch**

- Gelöste Fehlversuche bei −5 cm, mit gegen ohne Kontext (gleicher Adapter): Differenzkarte im Fenster 10/25 gegen 13/23 · + echtes Bild 9/24 gegen 16/27 · vorgestellt + echt im Fenster 10/24 gegen 12/27 · 6 Keyframes 9/27 gegen 12/27. Kein Fisher-Test mit Kontext besser (p 0,69–0,97).
- −10 cm: alle 0–1/30.
- Ohne Kontext lösen diese Adapter bei −5 cm 44–59 %, mehr als die p8t-LoRA ohne Kontext (2/29). Die Gewinne kommen aus dem Adapter selbst, nicht aus dem Fehlversuch.
- Die beiden Varianten mit Detail-Text (Kontext) brachen ab (zu langer Dateiname) → behoben, neu eingereicht. Ohne Kontext: 8/26 und 4/27.
- Fazit: Kleinerer Kontext macht den Fehlversuch nicht nutzbar; das p8t-Ergebnis (11/25 gegen 2/29) wiederholt sich so nicht.

**p8s mehr Fehlschläge (P7b+c+d): erst Kontrollen fertig**

- Ohne Kontext bei −5 cm: LoRA 6/27, LoRA + Basis 1/22, FiLM+Adapter+ 9/29, FiLM+Adapter+ + Basis 3/25; Fine-Tuning mit Basis 3/27. −10 cm 0–1/30.
- Die Varianten mit Kontext brachen am selben Dateinamen-Fehler ab → neu eingereicht 21.11. 06:25.

**Video-Plan, Schritt 0: Wie stellt sich das Basismodell den Griff vor? (−5 cm)**

- 6 Seeds bei −5 cm, Basismodell scheitert. Pro Video oben echt, Mitte die eigene Vorstellung des Modells, unten das Planer-Video (p13 tf_replan, gleicher Seed); links Kopfkamera, rechts Handgelenk des greifenden Arms.
- Sichtprüfung an 2 Seeds (10000, 40000): Die Vorstellung zeigt den Griff etwa auf derselben, zu hohen Höhe wie die Realität, aber danach eine angehobene Karte im Greifer. In der Realität bleibt die Karte liegen und das Handgelenk-Bild ist leer.
- Vorläufige Deutung: Das Modell stellt sich die normale Bewegung vor und 'glaubt', dass sie gelingt. Die Vorhersage korrigiert die Höhe nicht. Genau das soll der Video-Plan ändern.
- Einschränkung: Sichtprüfung an Einzelbildern, nicht gemessen. Eine Messung der vorgestellten Greifhöhe kommt mit der Aktions-Sonde (Offline-Gate).

**hid C2 in der Simulation (p12_hid_c2): kein Effekt**

- 3 ungesehene Aufgaben × 30 Seeds, versteckter Versatz. Gelöste Fehlversuche im 2. Versuch: Retry ohne Lernen 2/63 · C2 mit eigenem Fehlversuch 4/65 · C2 ohne Kontext 3/63 · C2 mit fremdem Fehlversuch 3/63.
- Grund: Der Residualkopf verschiebt die Aktion kaum (Median |dv| 0,009, max 0,028 in normierten Einheiten), der Versatz beträgt bis zu 10 cm. Die 9,5 % besserer Loss im Training reichen nicht für eine sichtbare Korrektur.
- LoRA C1 (hid): Validierungs-Loss mit Kontext 0,00754 gegen ohne 0,00752 → Kontext ungenutzt, Gate nicht bestanden, keine Simulation.
- Fazit hid bisher: Kein Mechanismus liest den Versatz aus dem Fehlversuch so, dass sich das Verhalten ändert.

**Teacher Forcing (p13): Wird die Aktion besser, wenn das Video perfekt ist?**

- Test: Basis-LingBot-VA; die vorhergesagte Zukunft wird durch echte Frames ersetzt. Ausgeführt werden nur die Aktionen des Modells.
- Version 2 (tf_replan): Vor jedem Chunk plant der RoboTwin-Planer ab dem aktuellen Zustand neu (greifen / weiterschieben / loslassen) → Snapshot → Planer-Aktionen einen Chunk lang ausführen, Frames aufnehmen → zurücksetzen. Kein Sprung im Video, Neuplanung in 100 % erfolgreich.
- Ergebnis (Erfolg korrigiert): 0 cm 26/27 (Basis 72/72) · −5 cm 27/27 (Basis 1/72) · −10 cm 8/27 (Basis 0/72).
- Abweichung Aktion ↔ Orakel (Median): 0 cm 0,6 cm · −5 cm 1,0 cm · −10 cm 2,1 cm. Vor jeder Neuplanung liegt der Roboter bei −10 cm im Median 2,8 cm neben dem Plan.
- Fazit: Bei −5 cm reicht ein perfektes Video. Bei −10 cm folgt der Aktionsteil dem Video nicht weit genug; das Video allein reicht dort nicht.
- Version 1 (Expertenpfad ab nächstem Punkt): −5 cm 22/24, −10 cm 0/22. Bei −10 cm ungültig, weil das Video zurück auf den Pfad springt.
- Kontrollen: Simulator nach jedem Vorausblick exakt zurückgesetzt (Restfehler 0); Experte allein −5 und −10 cm je 12/12.
- Videos: oben echte Kamera (Kopf | Handgelenk), unten das Orakel-Video, das der Aktionsteil als Vorhersage bekommt.

**Verrauschtes Orakel-Video (p14): Reicht ein nicht perfektes Video?**

- Wie tf_replan, aber die Orakel-Latents werden vor dem Aktionsteil verrauscht: (1−s)·Orakel + s·Rauschen (erstes Bild bleibt sauber). Latents haben etwa Einheitsskala.
- −5 cm: s=0 27/27 · s=0,10 27/27 · s=0,25 26/27 · s=0,50 13/27 · s=0,75 0/27.
- −10 cm: s=0 8/27 · s=0,10 8/27 · s=0,25 5/27 · s=0,50 3/27 · s=0,75 0/27.
- Abweichung Aktion ↔ Orakel bei −5 cm (Median): 1,0 → 1,0 → 1,2 → 2,3 → 5,7 cm.
- Fazit: Leicht gestörte Videos (s ≤ 0,25) helfen genauso wie perfekte; der Einbruch liegt zwischen 0,25 und 0,5.
- Einschränkung: Gauß-Rauschen ist eine andere Art Fehler als eine falsche Vorhersage. Nächster Schritt wäre: eigene Vorhersage des Modells mit dem Orakel mischen.

**Ist der Kontext zu groß? (p8t LoRA, hid M3)**

- p8t LoRA (11/25 mit gegen 2/29 ohne Kontext bei −5 cm) bekam ~7000 Kontext-Tokens: ganzer Fehlversuch, vorgestellt + echt, 3 Kameras, dazu den fast immer gleichen VLM-Satz.
- Ein Kontext-Token bekommt halb so viel Attention wie ein normales (0,53); nur 27 % der Kontext-Attention liegt im Fenster um den Fehler. Ohne Kontext ändert sich die Aktion um 11 %, aber die Greifhöhe nicht nach unten (+1 bzw. +2,6 mm).
- hid M3 (fertig 20.11.): kleinster Kontext am besten. C2 = nur echt − vorgestellt in 7 Frames um den Fehler: Loss −9,5 % gegenüber ohne Kontext (Gate verlangt −10 %), gemischter Kontext schlechter als keiner, z-Korrelation 0,59. C3 (+ Greiferzustand) −5 %, C5 (nur echtes Bild) ±0, C4 (Trajektorie) ±0. Keine Variante hat das Gate formal bestanden.
- Kleine Kontexte beim ungesehenen Fehlertyp wurden nie getestet → P8c (6 Varianten) und hid C2 in der Simulation (als Ausnahme).

**Neuer Plan: versteckte Kalibrier-Offsets (hid)**

- LingBot-VA gibt EE-Posen relativ zur Startpose aus und bekommt als state nur die eigenen Aktionen: Der geglaubte Rahmen ist automatisch erfüllt.
- o pro Seed deterministisch, in beiden Versuchen gleich: trans (z ±10 cm, x/y ±5 cm), gain (0,7–1,3), yaw (±25°), je 20 % o = 0.
- Experten-Fix mit kompensierten Zielen, gespeichert als die Aktionen, die die Policy ausgeben müsste.
- Mechanismen: M1 LoRA + Kontext, M2 gated Cross-Attention, M3 Residual-Kopf (konstante Geschwindigkeits-Verschiebung = Aktions-Verschiebung bei Rectified Flow).
- Keine Orakel-Varianten (Entscheidung 18.11.); Stufe A = C3 + aux × M1–M3.
- Daten p10: 465 Fehlversuche mit o ≠ 0, 580 (Fehlversuch, Fix)-Paare über 7 Trainingsaufgaben.
- Leak-Test: erstes Bild R² x 0,105 / y 0,04 / z 0,121 (Ziel < 0,1, knapp verfehlt); Fehlerfenster x 0,15 / y 0,20 / z 0,57.
- Kette: Leak-Test → Training (M3 C2–C5 ± aux, LoRA C1) → Trainings-Gate → Simulation nur für bestandene Varianten.

**Ungesehener Fehlertyp, 2 Trainingstypen (p8d)**

- Training nur auf Startpose- und Kamera-Fehlern, Test auf abgesenktem Tisch.
- Bestes Ergebnis: FiLM-memory + Adapter+ mit 16/26 gelöst bei −5 cm. Bei −10 cm löst keine Variante mehr als 1/30.
- Mit und ohne Kontext gleich: Der Kontext wird nicht genutzt.

**Ungesehener Fehlertyp, „7 Typen“ (p8t)**

- LoRA: 11/25 mit Kontext gegen 2/29 ohne bei −5 cm (Fisher p = 0,003). Das ist das erste Zeichen, dass ein Modell den Kontext nutzt.
- TTT umgekehrt: 0/24 mit gegen 9/25 ohne Kontext (p = 0,002). Hier schadet der Kontext.
- FiLM-memory + Adapter+ 3/25 gegen 2/27, Adapter+ 0/21 gegen 1/26, Fine-Tuning 4/25.
- Bei −10 cm bleibt alles bei 0/30.
- Einschränkung: Die Daten waren wegen P7c fast nur P7b, also kaum mehr Typen als in p8d.

**Zum Vergleich: Training mit Tischhöhe (p8)**

- Wenn der Testtyp im Training vorkommt: 28/30 gelöst auf beiden Höhen.
- Aber auch ohne Kontext ähnlich: Der Adapter lernt „tiefer greifen“ einfach auswendig.

**Was das VLM auf der Testaufgabe sagt**

- In rund 80 % der Fälle „missed grasp“, das stimmt: Der Greifer schließt 3 cm (−5) bzw. 8 cm (−10) zu hoch.
- Rund 20 % sind falsch („grasped but failed to lift“), obwohl nie gegriffen wurde.
- Die Texte sind fast identisch und sagen weder „zu hoch“ noch wie man es behebt.

**Erst Fehlerzeitpunkt finden, dann fragen (two_stage)**

- Zeitpunkt richtig in nur 23 % (8B) bzw. 35 % (32B) der Fälle, 17–27 % Ausfälle.
- „Tiefer greifen“ kommt in rund 78 % der Höhenfehler, aber genauso in rund 70 % der Fälle ohne Höhenfehler.
- Ergebnis: hilft nicht, das „tiefer“ kommt vom Prompt, nicht vom Bild.

**Fein-getunte Fehler-VLMs (vorläufig, Wiederholung läuft)**

- Getestet: ViFailback-8B (CVPR 2026, ALOHA AgileX), RoboFAC-7B, Cosmos-Reason1-7B, jeweils auch im eigenen Prompt-Format.
- Keines unterscheidet Höhenfehler zuverlässig: RoboFAC (eigenes Format) sagt „tiefer“ bei 74 % der Höhenfehler, aber auch bei 55 % der anderen.
- Qwen3-VL mit Checkliste: 86–100 % „tiefer“, aber 97–100 % auch dort, wo es falsch ist.
- Tabelle unten. Die Werte von ViFailback und dem 8B-detail-Prompt sind wegen der Port-Kollision noch unvollständig.

**Basismodell mittrainieren (wie WAM-TTT)**

- WAM-TTT trainiert beim Meta-Training das WAM selbst mit; beim Test bleibt alles eingefroren.
- Umgesetzt als --train_base: 5,09 Mrd. Parameter, Lernrate 1e-5, fp32-Masterkopie → nur auf H200.
- Smoke-Test ok: 40 Schritte, ~5 s pro Schritt, base.safetensors (10 GB) gespeichert.
- In der Eval nutzt nur der Retry die neuen Gewichte; der erste Versuch läuft mit dem Original-Modell, damit der Vergleich gleich bleibt.

**Neue Störungen P7d, erster Durchgang (12 Episoden pro Aufgabe)**

- Objektposition ±10 cm: 36 Fehlschläge (~43 %).
- Objektdrehung ±90°: 17 Fehlschläge (~20 %).
- Schwere Objekte: 2, schwacher Greifer: 4 → gestrichen.

### Probleme

| Problem | Ursache | Maßnahme | Status |
|---|---|---|---|
| p15 hid_trans_z: 0 Episoden | fawam/hid_offset.py: TYPES enthielt 'trans_z' nicht → AssertionError beim Start (beide Jobs). | TYPES erweitert (trans_z, lat, fixed). p15 hid_trans_z nicht neu eingereicht (Plan v2 ersetzt Schritt 0). | behoben |
| 9 Jobs von D2/D4 nach 2 min abgebrochen (21.11. 15:24) | Versions-Race: die Server luden methods.py, während neue Method-Felder ergänzt wurden; der Client schickte die neuen Felder. | Neu eingereicht (139153–161); Code-Änderungen ab da atomar (tmp + rename), keine neuen Method-Felder mehr. | behoben |
| Snap-Smoke: tf-Fortsetzung lief nicht | FAWAM_SNAP_MODES='expert,tf' – das Komma zerlegt sbatch --export. | Trenner '+', D3-Queue-Einträge umgeschrieben, Nach-Smoke p20_d3_smoke. | behoben, Smoke läuft |
| Karte: Greifer schließt schon in Chunk 2 | k = 3 Chunks vor dem Schließen liegt vor dem Episodenstart. | Reset-Punkt 0 erlaubt (Vergangenheit leer = volle Experten-Episode); im Test k = 2 → Reset bei Chunk 1. | behoben |
| Evals mit VLM-Detail-Text abgebrochen (20.11.) | Der Prompt wird als Ordnername für Visualisierungen benutzt (wan_va_server._reset); der lange Detail-Text überschritt das 255-Byte-Limit → OSError. Betroffen: alle Kontext-Varianten von p8s (24 Jobs) und 2 von p8c (12 Jobs). | Ordnername gekürzt + Hash; fehlende Zellen neu eingereicht 21.11. 06:25. | behoben, läuft neu |
| p15: neue Konfigurationen fehlten | hid_trans_z.yml und dist_objsize30.yml lagen nur in configs/, nicht in RoboTwin/task_config/ (setup.sh kopiert sie). | Kopiert, 12 Jobs + Smoke neu eingereicht. | behoben, läuft neu |
| LingBot-VA hat keinen getrennten Video-Expert | Video- und Aktions-Tokens laufen durch dieselben 30 Blöcke (wan_va/modules/model.py: ein self.blocks, nur Embedder, Zeit-Konditionierung und Ausgabe-Kopf sind pro Modus getrennt). Der Plan verlangt 'nur Parameter des Video-Expert trainierbar'. | Vorschlag: Adapter (LoRA/FiLM/Cross-Attention), die nur im Video-Durchlauf aktiv sind (action_mode=False). Die geteilten Gewichte bleiben eingefroren, der Aktions-Durchlauf rechnet unverändert, sieht aber die durch den Adapter veränderten Video-K/V im Cache. Entscheidung beim Nutzer. | offen (Nutzer) |
| P8c: kleine Kontexte beim ungesehenen Fehlertyp | Gleiche Daten, gleicher Adapter (LoRA r16) und gleiche VLM-Texte wie p8t, nur der Kontext ist kleiner: (1) echt − vorgestellt im Fenster um den Fehler, (2) dasselbe + echt, (3) vorgestellt + echt im Fenster, (4) 6 Frames mit höchstem Vorhersagefehler, (5) wie 1 mit detailliertem VLM-Text inkl. Fix-Satz, (6) nur der detaillierte VLM-Text. | 6 Trainings eingereiht 20.11. 10:01 (137739–744); danach Eval p8c_small_ctx automatisch, jede Variante mit und ohne Kontext. Kontrolle mit fremdem Fehlversuch entfällt: bei der Testaufgabe scheitern alle Seeds gleich (zu hoch gegriffen), der fremde Versuch wäre fast derselbe. | Eval läuft |
| hid-Trainings nach <1 min abgestürzt (19.11. 22:53 und 20.11. 04:05) | M3: Das Erfolgs-Flag in den Daten ist ein Bool, train_hid verglich es mit dem Text "True" → 0 Trainingspaare. LoRA (C1): Konfiguration „combo“ ohne LoRA-Schalter → 0 trainierbare Parameter. | Vergleich korrigiert (580 Paare), LoRA-Konfiguration auf „lora“ gestellt; Pipeline-Stufen zurückgesetzt, neu gestartet 20.11. 09:46 (Jobs 137719, 137720). | erledigt (behoben, Trainings liefen 20.11.) |
| hid Leak-Test knapp nicht bestanden | Der Versatz o ist im ersten Bild schwach ablesbar: R² x 0,105 / y 0,04 / z 0,121 (Ziel < 0,1). Im Fehlerfenster gut ablesbar (z 0,57), das Signal ist also da. | Entscheidung 19.11.: Grenze auf 0,15 gelockert und testen. Die Kontrollen ohne Kontext und mit gemischtem Kontext zeigen, ob das Modell o aus dem ersten Bild liest. Wenn ja: Ursache suchen. | zurückgestellt (hid hängt an Schritt 0) |
| Oracle-Video springt bei großem Abstand zum Expertenpfad (p13) | Der aufgezeichnete Expertenpfad wird nur ab dem nächstgelegenen Punkt abgespielt; bei −10 cm liegt der Roboter ~4 cm daneben. | tf_replan: Planer läuft vor jedem Chunk ab dem aktuellen Zustand neu. Nebenbei: Snapshot setzt jetzt auch den Python-Greiferwert zurück (fehlte, Einfluss gering). | erledigt (tf_replan) |
| hid-Jobs nach 45 s abgestürzt (19.11. 00:22–00:43) | Alle 41 Jobs (Smoke + Daten) bekamen nachts eine GPU, aber fawam/hid_offset.py patchte Base_Task.setup_demo; das gibt es nur in den Aufgaben-Unterklassen → AttributeError beim Start des Clients. | Hook auf Base_Task._init_task_env_ umgestellt (wird von jedem setup_demo mit dem Seed aufgerufen); 6 Smoke- und 35 Daten-Jobs um 08:10 neu eingereiht. | erledigt |
| Wartung HoreKa am 23.11. | Wartung ist am 23.11. (nicht 20.11.). Alles soll vorher durchlaufen. | Angehaltene P7d- und p0s-Jobs am 19.11. freigegeben (mit niedrigerer Priorität, damit hid und p14 zuerst laufen). | Evals laufen; Zeitlimits auf 2:30 / 3 h gesenkt |
| Daten lehren keinen Kontext-Gebrauch | In allen bisherigen Daten ist der Fehler im aktuellen Bild sichtbar oder die Korrektur immer gleich: Validierungs-Loss mit Kontext ≥ ohne Kontext, der Adapter lernt die Korrektur auswendig. | Datensatz hid: pro Seed ein unsichtbarer Kalibrier-Offset o, nur aus dem Fehlversuch erkennbar (PLAN_HID.md). | offen: hid hat es nicht gelöst (C2 ohne Effekt) |
| P7c-Störungen erzeugen kaum Fehlschläge | Licht, Hintergrund, Clutter, Objektposition 5 cm und Drehung ±60° geben nur 1–10 % Fehlschläge (27 insgesamt). Das Basismodell ist darauf schon randomisiert trainiert. | Das Training mit „7 Typen“ bestand zu 86 % aus P7b-Daten. Neuer Datensatz P7d mit stärkeren Störungen. | erledigt (P7d gesammelt) |
| Schwere Objekte und schwacher Greifer wirken nicht | Masse ×10–40: 2 Fehlschläge, Greifkraft ×0,15: 4 Fehlschläge (je ~84 Episoden). Die Patches greifen laut Log, aber der Griff in der Simulation bleibt stabil. | Beide Störungen gestrichen; Objektposition ±10 cm und Drehung ±90° werden auf 36 Episoden pro Aufgabe aufgefüllt. | erledigt (nur Position/Drehung aufgefüllt) |
| VLM-Text ist fast immer gleich | „The left arm failed to grasp the box“: kein Hinweis, dass der Greifer zu hoch ist, kein Fix. Für das Modell ist der Kontext damit fast konstant. | Neuer VLM-Modus „detail“ (Phase, wo der Greifer beim Schließen war, was das Objekt tat, ein Fix-Satz). Dazu Test fein-getunter Fehler-VLMs. | Test läuft (Detail-Text in P8c und p8s) |
| Kontext wird oft nicht genutzt | Mit und ohne Kontext gleich gut: Der Adapter lernt die Korrektur selbst, ohne den Kontext zu lesen. | Mehr verschiedene Fehlertypen (P7d) und Basismodell mittrainieren (wie WAM-TTT, Option --train_base). | Test läuft (P8c kleine Kontexte, p8s Basis mittrainiert) |
| VLM-Benchmark: Anfragen an den falschen Server | Mehrere Jobs auf einem Knoten nutzten alle Port 8001, 40–58 von 105 Anfragen bekamen 404. Gültige Antworten sind korrekt (Modellname muss passen). | Port pro Job (auch im VLM-Analyse-Job), fehlgeschlagene Anfragen werden wiederholt. | erledigt (Port pro Job) |
| Pipeline tickt nicht automatisch | Die scrontab wurde beim VS-Code-Start deaktiviert (#DISABLED). | scrontab neu installiert, Ticks laufen jetzt auf der Partition cpu (nicht agent, nicht dev-cpu). | erledigt |
| Job-Limit 100 pro Account | Die p0s-Jobs sind inzwischen gelaufen (19 fertig, 2 Zeitlimit). | Abbrechen, wenn nicht mehr gebraucht. | erledigt |
| Cosmos-Reason2 nicht ladbar | Das Modell ist auf Hugging Face zugangsbeschränkt. | Lizenz auf huggingface.co/nvidia/Cosmos-Reason2-8B akzeptieren und auf dem Cluster „hf auth login“ ausführen. Getestet wird vorerst Cosmos-Reason1-7B. | offen (Nutzer) |
| Weggeflogene Box zählt als Erfolg | RoboTwin prüft nur |x| > 0,23 m mit offenen Greifern. | Erfolg nur, wenn das Objekt auf Starthöhe ±5 cm und innerhalb von 0,6 m bleibt; Spalte „weggeflogen“. | erledigt |

## Ergebnisse je Sweep

### P0 · Basismodell, Tischhöhe

`p0_baseline_table` · Erfolg im ersten Versuch

| Methode | 0 cm | −5 cm | −10 cm |
|---|---|---|---|
| `base` | 72/72 (100 %) | 1/72 (1 %), 6 weggeflogen | 0/72 (0 %) |

### A · Adapter-Update nach dem Fehlversuch, verschiedene Signale

`p5a_signals` · gelöste Fehlversuche im zweiten Versuch

| Methode | −5 cm | −10 cm |
|---|---|---|
| `lff_sig_O` | 21/26 (81 %), 1 weggeflogen | 19/30 (63 %), 3 weggeflogen |
| `lff_sig_OP` | 23/25 (92 %) | 20/30 (67 %), 1 weggeflogen |
| `lff_sig_P` | 1/24 (4 %), 4 weggeflogen | 0/30 (0 %) |
| `lff_sig_PI` | 2/26 (8 %) | 0/30 (0 %) |
| `lff_sig_PR` | 1/29 (3 %), 3 weggeflogen | 0/30 (0 %) |
| `lff_sig_PRI` | 3/25 (12 %) | 0/30 (0 %) |
| `retry_explore` | 4/27 (15 %), 7 weggeflogen | 0/30 (0 %), 1 weggeflogen |
| `retry_plain` | 0/26 (0 %), 2 weggeflogen | 0/30 (0 %) |

### A · Adapter-Update mit VLM

`p5b_vlm` · gelöste Fehlversuche im zweiten Versuch

| Methode | −5 cm | −10 cm |
|---|---|---|
| `lff_sig_VP` | 2/28 (7 %), 3 weggeflogen | 0/20 (0 %) |

### B · Dual-Site-Adapter (Paper-Protokoll: Erfolg nach Adaption)

`p5f_paper` · Erfolg nach 3 Adaptions-Episoden

| Methode | −5 cm | −10 cm |
|---|---|---|
| `base` | 2/68 (3 %), 9 weggeflogen | 0/68 (0 %) |
| `paper_OP_dual` | 25/68 (37 %), 6 weggeflogen | 13/68 (19 %), 4 weggeflogen |
| `paper_PR_dual` | 13/68 (19 %), 10 weggeflogen | 0/75 (0 %), 1 weggeflogen |
| `paper_P_dual` | 22/68 (32 %), 8 weggeflogen | 0/68 (0 %), 5 weggeflogen |
| `paper_P_dual_150` | 0/68 (0 %), 9 weggeflogen | – |
| `paper_P_dual_action` | 0/68 (0 %) | 0/68 (0 %) |
| `paper_P_dual_filt` | 13/51 (25 %), 5 weggeflogen | 1/68 (1 %), 4 weggeflogen |
| `paper_P_dual_lr3` | 13/68 (19 %), 6 weggeflogen | 0/68 (0 %), 1 weggeflogen |
| `paper_P_dual_n1` | 33/76 (43 %), 12 weggeflogen | 0/76 (0 %), 1 weggeflogen |
| `paper_P_dual_n1vidtok` | 31/76 (41 %), 11 weggeflogen | 0/76 (0 %), 1 weggeflogen |
| `paper_P_dual_n6` | 15/42 (36 %), 7 weggeflogen | 0/56 (0 %), 1 weggeflogen |
| `paper_P_dual_s10` | 9/68 (13 %), 9 weggeflogen | 1/51 (2 %) |
| `paper_P_dual_s50` | 10/68 (15 %), 12 weggeflogen | 0/68 (0 %), 1 weggeflogen |
| `paper_P_dual_video` | 4/68 (6 %), 11 weggeflogen | 1/68 (1 %), 4 weggeflogen |
| `paper_P_dual_vidtok` | 30/68 (44 %), 6 weggeflogen | 0/68 (0 %), 2 weggeflogen |
| `paper_P_inside` | 13/68 (19 %), 7 weggeflogen | 1/68 (1 %), 6 weggeflogen |
| `paper_P_outside` | 2/68 (3 %), 9 weggeflogen | 0/68 (0 %) |
| `paper_Pw_dual` | 13/68 (19 %), 11 weggeflogen | 0/51 (0 %), 2 weggeflogen |

### C · Kontext ohne Training (Orakel-Text)

`p6a_icl_content` · gelöste Fehlversuche im zweiten Versuch

| Methode | −5 cm | −10 cm |
|---|---|---|
| `icl_fail_diff` | 1/26 (4 %) | 0/30 (0 %) |
| `icl_fail_predobs` | 0/24 (0 %), 1 weggeflogen | 0/30 (0 %) |
| `icl_text_only` | 1/25 (4 %), 4 weggeflogen | 0/30 (0 %) |
| `icl_vs_adapter` | 20/26 (77 %), 1 weggeflogen | 25/30 (83 %) |
| `retry_plain` | 0/24 (0 %), 4 weggeflogen | 0/30 (0 %) |

### C · Kontext ohne Training (VLM)

`p6v_icl_free_vlm` · gelöste Fehlversuche im zweiten Versuch

| Methode | −5 cm | −10 cm |
|---|---|---|
| `icl_fail_obs+VP` | 0/26 (0 %), 3 weggeflogen | 0/30 (0 %) |
| `icl_fail_predobs+VP` | 0/27 (0 %), 2 weggeflogen | 0/30 (0 %) |
| `icl_manip_noise+VP` | 0/26 (0 %) | 0/30 (0 %) |
| `icl_text_only+VP` | 0/26 (0 %), 4 weggeflogen | 0/30 (0 %) |
| `icl_txt_negative+VP` | 0/25 (0 %) | 0/30 (0 %) |
| `retry_plain` | 0/24 (0 %), 1 weggeflogen | 0/30 (0 %) |

### D · Meta-trainierte Kontext-Adapter, Testtyp (Tischhöhe) im Training

`p8_icl_vlm_meta` · gelöste Fehlversuche im zweiten Versuch

| Methode | −5 cm | −10 cm |
|---|---|---|
| `icl_fail_predobs_VP` | 0/29 (0 %) | 0/31 (0 %) |
| `icl_meta_diff_VP` | 25/28 (89 %) | 28/31 (90 %), 1 weggeflogen |
| `icl_meta_gapkeys_VP` | 28/28 (100 %) | 23/30 (77 %), 1 weggeflogen |
| `icl_meta_textonly_VP` | 27/30 (90 %), 1 weggeflogen | 20/30 (67 %), 1 weggeflogen |
| `icl_meta_window_VP` | 24/28 (86 %) | 25/30 (83 %), 2 weggeflogen |
| `icl_type_adaln_VP` | 17/26 (65 %), 1 weggeflogen | 7/31 (23 %) |
| `icl_type_adaln_VP_noctx` | 25/30 (83 %) | – |
| `icl_type_adapterplus_VP` | 26/27 (96 %), 1 weggeflogen | 11/20 (55 %), 2 weggeflogen |
| `icl_type_adapterplus_VP_noctx` | 23/26 (88 %), 1 weggeflogen | – |
| `icl_type_film_error_VP` | 17/26 (65 %) | 10/30 (33 %), 4 weggeflogen |
| `icl_type_film_error_VP_noctx` | 21/25 (84 %) | – |
| `icl_type_film_gapcurve_VP` | 19/28 (68 %), 1 weggeflogen | 8/30 (27 %), 2 weggeflogen |
| `icl_type_film_gapcurve_VP_noctx` | 23/27 (85 %), 1 weggeflogen | – |
| `icl_type_film_memory_VP` | 17/26 (65 %) | 13/30 (43 %), 1 weggeflogen |
| `icl_type_film_memory_VP_noctx` | 1/26 (4 %), 1 weggeflogen | – |
| `icl_type_film_memory_plus_adapterplus_VP` | 21/22 (95 %), 1 weggeflogen | 28/30 (93 %), 1 weggeflogen |
| `icl_type_film_memory_plus_adapterplus_VP_noctx` | 27/28 (96 %) | – |
| `icl_type_film_text_VP` | 8/29 (28 %) | 5/30 (17 %) |
| `icl_type_film_text_VP_noctx` | 25/30 (83 %) | – |
| `icl_type_ia3_adaln_VP` | 18/26 (69 %), 2 weggeflogen | 7/31 (23 %), 4 weggeflogen |
| `icl_type_ia3_adaln_VP_noctx` | 24/30 (80 %), 1 weggeflogen | – |
| `icl_type_lora_r16_attn_VP` | 28/30 (93 %) | 26/30 (87 %) |
| `icl_type_lora_r16_attn_VP_noctx` | 28/29 (97 %) | – |
| `icl_type_prompt_vane_VP` | 7/26 (27 %), 3 weggeflogen | 1/30 (3 %), 2 weggeflogen |
| `icl_type_prompt_vane_VP_noctx` | 24/27 (89 %) | – |
| `icl_type_ttt_wam_VP` | 19/27 (70 %), 4 weggeflogen | 14/30 (47 %), 1 weggeflogen |
| `icl_type_ttt_wam_VP_noctx` | 28/28 (100 %) | – |
| `retry_plain` | 1/27 (4 %), 2 weggeflogen | 0/30 (0 %) |

### D · Nur Fine-Tuning

`p8f_finetune` · gelöste Fehlversuche im zweiten Versuch

| Methode | −5 cm | −10 cm |
|---|---|---|
| `ft_adapterplus` | 23/24 (96 %) | 30/30 (100 %) |

### D · Ungesehener Fehlertyp, Training auf 2 Typen (Startpose, Kamera)

`p8d_heldout_type` · gelöste Fehlversuche im zweiten Versuch

| Methode | −5 cm | −10 cm |
|---|---|---|
| `ft_dist_adapterplus` | 3/24 (12 %), 1 weggeflogen | 0/30 (0 %), 1 weggeflogen |
| `icl_dist_adapterplus_VP` | 6/19 (32 %) | 0/30 (0 %), 1 weggeflogen |
| `icl_dist_adapterplus_VP_noctx` | 10/25 (40 %) | 1/30 (3 %), 1 weggeflogen |
| `icl_dist_film_memory_plus_adapterplus_VP` | 16/26 (62 %), 1 weggeflogen | 1/30 (3 %), 6 weggeflogen |
| `icl_dist_film_memory_plus_adapterplus_VP_noctx` | 12/23 (52 %), 1 weggeflogen | 0/30 (0 %), 2 weggeflogen |
| `icl_dist_lora_r16_attn_VP` | 6/24 (25 %), 2 weggeflogen | 0/30 (0 %) |
| `icl_dist_lora_r16_attn_VP_noctx` | 7/28 (25 %), 3 weggeflogen | 0/30 (0 %) |
| `icl_dist_ttt_wam_VP` | 2/26 (8 %), 4 weggeflogen | 0/30 (0 %) |
| `icl_dist_ttt_wam_VP_noctx` | 2/26 (8 %), 3 weggeflogen | 0/30 (0 %) |

### D · Ungesehener Fehlertyp, Training auf „7 Typen“ (P7b + P7c)

`p8t_heldout_types` · gelöste Fehlversuche im zweiten Versuch

| Methode | −5 cm | −10 cm |
|---|---|---|
| `ft_types_adapterplus` | 4/25 (16 %), 2 weggeflogen | 0/30 (0 %), 2 weggeflogen |
| `icl_types_adapterplus_VP` | 0/21 (0 %), 3 weggeflogen | 0/30 (0 %) |
| `icl_types_adapterplus_VP_noctx` | 1/26 (4 %), 2 weggeflogen | 0/30 (0 %) |
| `icl_types_film_memory_plus_adapterplus_VP` | 3/25 (12 %), 4 weggeflogen | 0/30 (0 %) |
| `icl_types_film_memory_plus_adapterplus_VP_noctx` | 2/27 (7 %), 4 weggeflogen | 0/30 (0 %) |
| `icl_types_lora_r16_attn_VP` | 11/25 (44 %), 3 weggeflogen | 0/30 (0 %) |
| `icl_types_lora_r16_attn_VP_noctx` | 2/29 (7 %), 2 weggeflogen | 0/30 (0 %) |
| `icl_types_ttt_wam_VP` | 0/24 (0 %) | 0/30 (0 %) |
| `icl_types_ttt_wam_VP_noctx` | 9/25 (36 %), 2 weggeflogen | 0/30 (0 %) |

### D · Mehr Fehlschläge (P7d), VLM detail, mit/ohne Basis-Training

`p8s_strong` · gelöste Fehlversuche im zweiten Versuch

| Methode | −5 cm | −10 cm |
|---|---|---|
| `ft_strong_base` | 3/27 (11 %), 3 weggeflogen | 0/30 (0 %) |
| `icl_strong_film_memory_plus_adapterplus_VPD` | 10/27 (37 %), 1 weggeflogen | 1/30 (3 %), 1 weggeflogen |
| `icl_strong_film_memory_plus_adapterplus_VPD_noctx` | 9/29 (31 %), 5 weggeflogen | 0/30 (0 %), 1 weggeflogen |
| `icl_strong_film_memory_plus_adapterplus_base_VPD` | 2/29 (7 %), 5 weggeflogen | 0/30 (0 %) |
| `icl_strong_film_memory_plus_adapterplus_base_VPD_noctx` | 3/25 (12 %), 4 weggeflogen | 0/30 (0 %), 1 weggeflogen |
| `icl_strong_lora_r16_attn_VPD` | 5/26 (19 %), 5 weggeflogen | 0/30 (0 %) |
| `icl_strong_lora_r16_attn_VPD_noctx` | 6/27 (22 %), 3 weggeflogen | 1/30 (3 %) |
| `icl_strong_lora_r16_attn_base_VPD` | 6/25 (24 %), 5 weggeflogen | 1/30 (3 %), 1 weggeflogen |
| `icl_strong_lora_r16_attn_base_VPD_noctx` | 1/22 (5 %), 5 weggeflogen | 0/30 (0 %) |

### E · Verwandte Arbeiten nachgebaut

`p9_related` · gelöste Fehlversuche im zweiten Versuch

| Methode | −5 cm | −10 cm |
|---|---|---|
| `lff_adapter_videoonly` | 1/24 (4 %), 2 weggeflogen | 0/30 (0 %) |
| `lff_bank` | 8/20 (40 %) | 7/21 (33 %), 1 weggeflogen |
| `lff_pref` | 21/28 (75 %) | 21/30 (70 %), 1 weggeflogen |
| `lff_ttt_filtered` | 4/27 (15 %), 2 weggeflogen | 0/30 (0 %) |
| `reset_on_gap` | – | – |
| `retry_plain` | 1/28 (4 %), 3 weggeflogen | 0/30 (0 %) |

## Meta-Training: Validierungs-Loss mit gegen ohne Kontext

| Adapter | Schritte | Val-Loss mit Kontext | ohne Kontext | Kontext hilft? |
|---|---|---|---|---|
| `failfix_diff_vlm` | 1500 | 0.0044 | 0.0028 | nein |
| `failfix_dist_adapterplus_vlm` | 1500 | 0.0063 | 0.006 | nein |
| `failfix_dist_film_memory_plus_adapterplus_vlm` | 1500 | 0.0064 | 0.0051 | nein |
| `failfix_dist_ft_adapterplus` | 1500 | 0.1324 | 0.0031 | nein |
| `failfix_dist_lora_r16_attn_vlm` | 1500 | 0.005 | 0.0036 | nein |
| `failfix_dist_ttt_wam_vlm` | 1500 | 0.0087 | 0.0053 | nein |
| `failfix_ft_adapterplus` | 1500 | 0.116 | 0.0021 | nein |
| `failfix_gapkeys_vlm` | 1500 | 0.003 | 0.0023 | nein |
| `failfix_hid_lora` | 1500 | 0.0075 | 0.0075 | nein |
| `failfix_sc_diffobswin` | 1500 | 0.0036 | 0.0034 | nein |
| `failfix_sc_diffwin` | 1500 | 0.0034 | 0.0029 | nein |
| `failfix_sc_diffwin_vlmd` | 1500 | 0.0032 | 0.0029 | nein |
| `failfix_sc_gapkeys` | 1500 | 0.0038 | 0.0032 | nein |
| `failfix_sc_predobswin` | 1500 | 0.0035 | 0.0031 | nein |
| `failfix_sc_textonly_vlmd` | 1500 | 0.0027 | 0.0027 | nein |
| `failfix_strong_film_memory_plus_adapterplus_vlmd` | 1500 | 0.0056 | 0.0054 | nein |
| `failfix_strong_film_memory_plus_adapterplus_vlmd_base` | 1500 | 0.0087 | 0.0049 | nein |
| `failfix_strong_ft_base` | 1500 | 0.1415 | 0.0031 | nein |
| `failfix_strong_lora_r16_attn_vlmd` | 1500 | 0.0043 | 0.004 | nein |
| `failfix_strong_lora_r16_attn_vlmd_base` | 1500 | 0.0069 | 0.0043 | nein |
| `failfix_textonly_vlm` | 1500 | 0.002 | 0.002 | nein |
| `failfix_type_adaln_vlm` | 1500 | 0.0066 | 0.0038 | nein |
| `failfix_type_adapterplus_vlm` | 1500 | 0.0053 | 0.0036 | nein |
| `failfix_type_film_error_vlm` | 1500 | 0.0051 | 0.0145 | ja |
| `failfix_type_film_gapcurve_vlm` | 1500 | 0.0054 | 0.0145 | ja |
| `failfix_type_film_memory_plus_adapterplus_vlm` | 1500 | 0.0032 | 0.0032 | nein |
| `failfix_type_film_memory_vlm` | 1500 | 0.0047 | 0.0145 | ja |
| `failfix_type_film_text_vlm` | 1500 | 0.0052 | 0.0032 | nein |
| `failfix_type_ia3_adaln_vlm` | 1500 | 0.0059 | 0.0034 | nein |
| `failfix_type_lora_r16_attn_vlm` | 1500 | 0.0031 | 0.0026 | nein |
| `failfix_type_prompt_vane_vlm` | 1500 | 0.0573 | 0.0107 | nein |
| `failfix_type_ttt_wam_vlm` | 1500 | 0.0044 | 0.0028 | nein |
| `failfix_types_adapterplus_vlm` | 1500 | 0.0061 | 0.0043 | nein |
| `failfix_types_film_memory_plus_adapterplus_vlm` | 1500 | 0.0047 | 0.0042 | nein |
| `failfix_types_ft_adapterplus` | 1500 | 0.162 | 0.0031 | nein |
| `failfix_types_lora_r16_attn_vlm` | 1500 | 0.0049 | 0.0037 | nein |
| `failfix_types_ttt_wam_vlm` | 1500 | 0.0076 | 0.0036 | nein |
| `failfix_window_vlm` | 1500 | 0.0029 | 0.0025 | nein |

## VLM

### VLM gegen Orakel (Qwen3-VL-8B, offline)

| Tisch / Eingabe / Aufgaben | n | Typ richtig | Richtung richtig | Δz-Fehler |
|---|---|---|---|---|
| table_h_m05|rollout|test | 65 | 91 % | 34 % | 3.13 cm |
| table_h_m05|rollout|all | 281 | 52 % | 32 % | 2.74 cm |
| table_h_m05|frame|test | 65 | 66 % | 8 % | 3.12 cm |
| table_h_m05|frame|all | 281 | 33 % | 8 % | 2.67 cm |
| table_h_m10|rollout|test | 72 | 29 % | 26 % | 7.83 cm |
| table_h_m10|rollout|all | 487 | 12 % | 28 % | 7.39 cm |
| table_h_m10|frame|test | 72 | 25 % | 7 % | 7.84 cm |
| table_h_m10|frame|all | 487 | 9 % | 11 % | 7.35 cm |

### VLM-Benchmark: 105 Fehlschläge (fawam/vlm_bench.py)

tiefer = sagt „tiefer greifen“, wo der Greifer zu hoch war · tiefer falsch = sagt es ohne Höhenfehler

| Modell | Prompt | n | Ausfälle | Typ | tiefer | tiefer falsch | Lokalisierung |
|---|---|---|---|---|---|---|---|
| MINT-SJTU_RoboFAC-7B | checklist | 105 | 0 | 37 % | 0 % | 10 % | – |
| MINT-SJTU_RoboFAC-7B | detail | 105 | 0 | 94 % | 8 % | 17 % | – |
| MINT-SJTU_RoboFAC-7B | native_robofac | 105 | 0 | 74 % | 74 % | 57 % | – |
| MINT-SJTU_RoboFAC-7B | native_vifailback | 105 | 0 | 11 % | 0 % | 0 % | – |
| MINT-SJTU_RoboFAC-7B | open | 105 | 0 | 94 % | 6 % | 20 % | – |
| Qwen_Qwen3-VL-32B-Instruct | checklist | 105 | 0 | 94 % | 100 % | 97 % | – |
| Qwen_Qwen3-VL-32B-Instruct | dense12 | 105 | 0 | 90 % | 26 % | 17 % | – |
| Qwen_Qwen3-VL-32B-Instruct | detail | 105 | 0 | 92 % | 86 % | 87 % | – |
| Qwen_Qwen3-VL-32B-Instruct | legacy_cm | 105 | 0 | 68 % | 11 % | 20 % | – |
| Qwen_Qwen3-VL-32B-Instruct | native_robofac | 105 | 0 | 89 % | 17 % | 33 % | – |
| Qwen_Qwen3-VL-32B-Instruct | native_vifailback | 105 | 0 | 86 % | 15 % | 17 % | – |
| Qwen_Qwen3-VL-32B-Instruct | open | 105 | 0 | 93 % | 31 % | 33 % | – |
| Qwen_Qwen3-VL-32B-Instruct | open_nocurve | 105 | 0 | 93 % | 4 % | 13 % | – |
| Qwen_Qwen3-VL-32B-Instruct | pred_pairs | 105 | 0 | 92 % | 0 % | 0 % | – |
| Qwen_Qwen3-VL-32B-Instruct | steps | 105 | 0 | 70 % | 18 % | 27 % | – |
| Qwen_Qwen3-VL-32B-Instruct | success_ref | 105 | 0 | 94 % | 31 % | 33 % | – |
| Qwen_Qwen3-VL-32B-Instruct | two_stage | 105 | 18 | 87 % | 80 % | 67 % | 34 % |
| Qwen_Qwen3-VL-32B-Instruct | video | 105 | 0 | 91 % | 0 % | 7 % | – |
| Qwen_Qwen3-VL-32B-Instruct | wrist_zoom | 105 | 0 | 94 % | 12 % | 0 % | – |
| Qwen_Qwen3-VL-32B-Instruct | zoom_pairs_steps | 105 | 0 | 89 % | 3 % | 3 % | – |
| Qwen_Qwen3-VL-8B-Instruct | checklist | 105 | 0 | 94 % | 86 % | 100 % | – |
| Qwen_Qwen3-VL-8B-Instruct | dense12 | 105 | 0 | 94 % | 3 % | 0 % | – |
| Qwen_Qwen3-VL-8B-Instruct | detail | 105 | 0 | 94 % | 18 % | 43 % | – |
| Qwen_Qwen3-VL-8B-Instruct | legacy_cm | 105 | 0 | 82 % | 6 % | 0 % | – |
| Qwen_Qwen3-VL-8B-Instruct | native_robofac | 105 | 0 | 91 % | 10 % | 23 % | – |
| Qwen_Qwen3-VL-8B-Instruct | native_vifailback | 105 | 0 | 30 % | 6 % | 17 % | – |
| Qwen_Qwen3-VL-8B-Instruct | open | 105 | 0 | 93 % | 3 % | 0 % | – |
| Qwen_Qwen3-VL-8B-Instruct | open_nocurve | 105 | 0 | 93 % | 8 % | 0 % | – |
| Qwen_Qwen3-VL-8B-Instruct | pred_pairs | 105 | 0 | 92 % | 6 % | 0 % | – |
| Qwen_Qwen3-VL-8B-Instruct | steps | 105 | 0 | 84 % | 33 % | 27 % | – |
| Qwen_Qwen3-VL-8B-Instruct | success_ref | 105 | 0 | 94 % | 1 % | 0 % | – |
| Qwen_Qwen3-VL-8B-Instruct | two_stage | 105 | 28 | 73 % | 76 % | 70 % | 23 % |
| Qwen_Qwen3-VL-8B-Instruct | video | 105 | 0 | 85 % | 0 % | 3 % | – |
| Qwen_Qwen3-VL-8B-Instruct | wrist_zoom | 105 | 0 | 94 % | 17 % | 13 % | – |
| Qwen_Qwen3-VL-8B-Instruct | zoom_pairs_steps | 105 | 0 | 70 % | 22 % | 20 % | – |
| nvidia_Cosmos-Reason1-7B | checklist | 105 | 0 | 94 % | 25 % | 27 % | – |
| nvidia_Cosmos-Reason1-7B | detail | 105 | 0 | 36 % | 4 % | 7 % | – |
| nvidia_Cosmos-Reason1-7B | native_robofac | 105 | 0 | 85 % | 4 % | 17 % | – |
| nvidia_Cosmos-Reason1-7B | native_vifailback | 105 | 0 | 18 % | 0 % | 7 % | – |
| nvidia_Cosmos-Reason1-7B | open | 105 | 0 | 94 % | 0 % | 0 % | – |
| sii-rhos-ai_ViFailback-8B | checklist | 105 | 0 | 7 % | 8 % | 20 % | – |
| sii-rhos-ai_ViFailback-8B | detail | 105 | 0 | 41 % | 24 % | 37 % | – |
| sii-rhos-ai_ViFailback-8B | native_robofac | 105 | 0 | 91 % | 18 % | 27 % | – |
| sii-rhos-ai_ViFailback-8B | native_vifailback | 105 | 0 | 82 % | 0 % | 0 % | – |
| sii-rhos-ai_ViFailback-8B | open | 105 | 0 | 23 % | 12 % | 37 % | – |
