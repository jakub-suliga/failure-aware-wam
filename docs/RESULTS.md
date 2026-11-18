# Ergebnisse Failure-Aware WAM

Automatisch erzeugt aus dem Report (`fawam/report_md.py`) · Stand der Daten: 2026-11-18 15:54 · Testaufgabe `move_playingcard_away`.

Erfolg zählt nur, wenn das Objekt auf dem Tisch bleibt („weggeflogen“ = vom Simulator als Erfolg gewertet, hier Fehlschlag). Brüche: gelöst / versucht. Ziele und Plan: [GOAL.md](GOAL.md), [PLAN.md](PLAN.md), [PLAN_HID.md](PLAN_HID.md); ältere Befunde: [FINDINGS.md](FINDINGS.md).

## Probleme, Analysen, laufende Arbeit

Stand: 2026-11-18 21:00

### Was gerade läuft

| Schritt | Status |
|---|---|
| Minimalpfad vor dem Cluster-Ausfall (20.11.): Datensatz hid_trans (p10_hid_data, 90 Seeds × 7 Aufgaben, Paare Fehlversuch + kompensierter Experte) | wartet auf GPU (35 Jobs) |
| Smoke-Test hid (p10_hid_smoke): Policy-Fehlerrate unter o, Experte mit Kompensation ≥ 90 %, Ausführung = Ziel + o | wartet auf GPU (6 Jobs) |
| Dev-Runner (4 GPUs, dev-gpu-h100) für Smoke + Daten | wartet, Knoten hkn0901 nicht verfügbar |
| Leak-Test (fawam/hid_leak.py): o aus dem ersten Frame R² < 0,1, aus dem Fehlerfenster > 0,5 | wartet auf Daten |
| Training C1×M1 (LoRA r16, ganzer Fehlversuch im Kontext, 30 % fremder Kontext) und M3 (Residual-Kopf, C2–C5 × aux in einem Job) + Trainings-Gate | Code fertig, wartet auf Daten |
| Eval T1: 3 Held-out-Aufgaben (move_playingcard_away, place_object_scale, move_pillbottle_pad) × ctx / noctx / shuf | Code fertig, wartet |
| Angehalten bis nach dem Ausfall: P7d-Auffüllen, VLM-Benchmark-Wiederholung, P8s-Kette, p0s-Screening | angehalten |

### Analysen

**Neuer Plan: versteckte Kalibrier-Offsets (hid)**

- LingBot-VA gibt EE-Posen relativ zur Startpose aus und bekommt als state nur die eigenen Aktionen: Der geglaubte Rahmen ist automatisch erfüllt.
- o pro Seed deterministisch, in beiden Versuchen gleich: trans (z ±10 cm, x/y ±5 cm), gain (0,7–1,3), yaw (±25°), je 20 % o = 0.
- Experten-Fix mit kompensierten Zielen, gespeichert als die Aktionen, die die Policy ausgeben müsste.
- Mechanismen: M1 LoRA + Kontext, M2 gated Cross-Attention, M3 Residual-Kopf (konstante Geschwindigkeits-Verschiebung = Aktions-Verschiebung bei Rectified Flow).
- Keine Orakel-Varianten (Entscheidung 18.11.); Stufe A = C3 + aux × M1–M3.

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
| Keine GPU vor dem Ausfall | Normale GPU-Partitionen voll: Startschätzung für unsere ältesten Jobs 20.11. 01:50, für neue Jobs 23.11.–30.11.; Dev-Knoten hkn0901 nicht verfügbar. | Alles eingereiht und Code fertig, damit jede GPU-Minute genutzt wird; andere GPU-Maschine wäre die Alternative. | offen |
| Daten lehren keinen Kontext-Gebrauch | In allen bisherigen Daten ist der Fehler im aktuellen Bild sichtbar oder die Korrektur immer gleich: Validierungs-Loss mit Kontext ≥ ohne Kontext, der Adapter lernt die Korrektur auswendig. | Datensatz hid: pro Seed ein unsichtbarer Kalibrier-Offset o, nur aus dem Fehlversuch erkennbar (PLAN_HID.md). | in Arbeit |
| P7c-Störungen erzeugen kaum Fehlschläge | Licht, Hintergrund, Clutter, Objektposition 5 cm und Drehung ±60° geben nur 1–10 % Fehlschläge (27 insgesamt). Das Basismodell ist darauf schon randomisiert trainiert. | Das Training mit „7 Typen“ bestand zu 86 % aus P7b-Daten. Neuer Datensatz P7d mit stärkeren Störungen. | in Arbeit |
| Schwere Objekte und schwacher Greifer wirken nicht | Masse ×10–40: 2 Fehlschläge, Greifkraft ×0,15: 4 Fehlschläge (je ~84 Episoden). Die Patches greifen laut Log, aber der Griff in der Simulation bleibt stabil. | Beide Störungen gestrichen; Objektposition ±10 cm und Drehung ±90° werden auf 36 Episoden pro Aufgabe aufgefüllt. | in Arbeit |
| VLM-Text ist fast immer gleich | „The left arm failed to grasp the box“: kein Hinweis, dass der Greifer zu hoch ist, kein Fix. Für das Modell ist der Kontext damit fast konstant. | Neuer VLM-Modus „detail“ (Phase, wo der Greifer beim Schließen war, was das Objekt tat, ein Fix-Satz). Dazu Test fein-getunter Fehler-VLMs. | in Arbeit |
| Kontext wird oft nicht genutzt | Mit und ohne Kontext gleich gut: Der Adapter lernt die Korrektur selbst, ohne den Kontext zu lesen. | Mehr verschiedene Fehlertypen (P7d) und Basismodell mittrainieren (wie WAM-TTT, Option --train_base). | in Arbeit |
| VLM-Benchmark: Anfragen an den falschen Server | Mehrere Jobs auf einem Knoten nutzten alle Port 8001, 40–58 von 105 Anfragen bekamen 404. Gültige Antworten sind korrekt (Modellname muss passen). | Port pro Job (auch im VLM-Analyse-Job), fehlgeschlagene Anfragen werden wiederholt. | in Arbeit |
| Pipeline tickt nicht automatisch | Die scrontab wurde beim VS-Code-Start deaktiviert (#DISABLED). | scrontab neu installiert, Ticks laufen jetzt auf der Partition cpu (nicht agent, nicht dev-cpu). | erledigt |
| Job-Limit 100 pro Account | 21 angehaltene p0s_task_screen-Jobs belegen Plätze. | Abbrechen, wenn nicht mehr gebraucht. | offen (Nutzer) |
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
| MINT-SJTU_RoboFAC-7B | native_robofac | 105 | 4 | 76 % | 74 % | 55 % | – |
| MINT-SJTU_RoboFAC-7B | native_vifailback | 105 | 0 | 11 % | 0 % | 0 % | – |
| MINT-SJTU_RoboFAC-7B | open | 105 | 0 | 94 % | 6 % | 20 % | – |
| Qwen_Qwen3-VL-32B-Instruct | checklist | 105 | 0 | 94 % | 100 % | 97 % | – |
| Qwen_Qwen3-VL-32B-Instruct | dense12 | 105 | 0 | 90 % | 26 % | 17 % | – |
| Qwen_Qwen3-VL-32B-Instruct | detail | 105 | 44 | 93 % | 90 % | 90 % | – |
| Qwen_Qwen3-VL-32B-Instruct | legacy_cm | 105 | 0 | 68 % | 11 % | 20 % | – |
| Qwen_Qwen3-VL-32B-Instruct | native_robofac | 105 | 0 | 89 % | 17 % | 33 % | – |
| Qwen_Qwen3-VL-32B-Instruct | native_vifailback | 105 | 7 | 87 % | 15 % | 17 % | – |
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
| Qwen_Qwen3-VL-8B-Instruct | detail | 105 | 58 | 89 % | 17 % | 38 % | – |
| Qwen_Qwen3-VL-8B-Instruct | legacy_cm | 105 | 0 | 82 % | 6 % | 0 % | – |
| Qwen_Qwen3-VL-8B-Instruct | native_robofac | 105 | 50 | 94 % | 16 % | 20 % | – |
| Qwen_Qwen3-VL-8B-Instruct | native_vifailback | 105 | 55 | 32 % | 3 % | 15 % | – |
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
| nvidia_Cosmos-Reason1-7B | native_robofac | 105 | 6 | 84 % | 4 % | 18 % | – |
| nvidia_Cosmos-Reason1-7B | native_vifailback | 105 | 0 | 18 % | 0 % | 7 % | – |
| nvidia_Cosmos-Reason1-7B | open | 105 | 0 | 94 % | 0 % | 0 % | – |
| sii-rhos-ai_ViFailback-8B | checklist | 105 | 53 | 8 % | 10 % | 33 % | – |
| sii-rhos-ai_ViFailback-8B | detail | 105 | 5 | 42 % | 24 % | 38 % | – |
| sii-rhos-ai_ViFailback-8B | native_robofac | 105 | 57 | 90 % | 19 % | 27 % | – |
| sii-rhos-ai_ViFailback-8B | native_vifailback | 105 | 39 | 76 % | 0 % | 0 % | – |
| sii-rhos-ai_ViFailback-8B | open | 105 | 55 | 22 % | 14 % | 36 % | – |
