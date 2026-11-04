# Offene Punkte

Stand: 2026-11-04. Das sind die Dinge, die geschrieben, aber noch nicht geprüft sind, oder die noch fehlen.
Details zu den Methoden stehen in [EXPERIMENTS.md](EXPERIMENTS.md), die Befunde in [FINDINGS.md](FINDINGS.md).

## 1. Blockiert, bevor auf dem Cluster gerechnet werden kann

- [ ] **Repo committen und pushen.** Seit dem Initial-Commit ist nichts committet: `fawam/`, `cluster/`, `docs/`,
      `configs/sweeps/`, neue Tischhöhen-Configs, `artifacts/gap_stats.json` + `artifacts/probe/`, aktualisierter Patch.
- [ ] **Cluster-Setup ausführen** (`cluster/setup_hk2.sh repos|envs|curobo|assets|check`). Das ist komplett ungetestet.
  - [ ] Environments aus `cluster/envs/*_pip_freeze.txt` lassen sich installieren.
  - [ ] cuRobo baut für H100/B200 (`TORCH_CUDA_ARCH_LIST="9.0;10.0"`).
  - [ ] **Vulkan-Rendering auf den Compute-Nodes** (`RoboTwin/script/test_render.py`). Das ist das größte Risiko.
        Fallback: Client in einem Apptainer-Image mit Vulkan, Server auf dem Host (wie im Tactile-Projekt).
  - [ ] Checkpoint herunterladen und `attn_mode` in `transformer/config.json` auf `torch` setzen.
- [ ] **Smoke-Job auf dem Cluster:** `cluster/setup_hk2.sh check` (1 Episode `pick_diverse_bottles`, z00).
- [ ] **Mehrere Paare pro GPU:** Prüfen, ob 3 Server/Sim-Paare (`N_PAR=3`) auf einer H100-94GB ohne OOM laufen und
      wie viel schneller das ist als 1 Paar. Davon hängen die GPU-h-Schätzungen ab.

## 1b. Learn-from-Failure (Kern, [METHOD.md](METHOD.md))

- [ ] **Lernt der Adapter die Korrektur?** `python -m fawam.tests.test_lff_effect <fail ep> <adapter>` vergleicht die
      vorgeschlagene Greiferhöhe von Basis und Adapter auf der Fail-Episode. Lokal scheiterte es am GPU-Speicher
      (fremder Prozess, 17 GB). Erster Test auf dem Cluster. Zum Beispiel mit
      `results/table_height_n10/m10/episodes/fail/move_playingcard_away/ep0001_seed10001` (analysis.json liegt dort).
- [ ] Die LfF-Schleife im Client (Versuch → Analyse → Learner-Subprozess → Versuch 2) einmal live laufen lassen.
      Server, Sim und Learner brauchen zusammen ~35 GB. Prüfen: Ordner `..._attempt2`, `lff/<task>/learner.log`, `lff` in meta.json.
- [ ] Prefill mit `include_pred` (vorgestellter + realer Fehlversuch im Kontext) mit echtem Modell testen.
- [ ] `failfix_meta` trainieren. Braucht P0-Fails mit `analysis.json` und P2-Expert-Replays auf denselben Seeds.
      Prüfen, wie viele Paare zusammenkommen.
- [ ] Hyperparameter des Updates: Schritte (150), Lernrate, β, Fensterlänge (3 Latents), Rampe, `lambda_video`,
      `action_clip` (3). Kleiner Sweep auf 1–2 Tasks, bevor `p5_lff` voll läuft.
- [ ] VLM-Analyse liefert `delta_cm`: prüfen, ob die Zahlen brauchbar sind (C1), sonst nur Richtung plus feste Schrittweite.
- [ ] Orakel-Heuristik: bei mehreren Objekten wird das nächste in xy genommen, Drops bekommen kein Δ (nur Text und
      Gewichte). Für handover-Drops braucht es eine andere Korrektur (z. B. Greifer länger geschlossen, Übergabepose).
- [ ] `artifacts/grasp_ref.json` aus mehr Erfolgen neu bauen (aktuell table_height z00 + eval_hard).
- [ ] Analyse-Skript für Sättigung am Aktionsrand (FINDINGS 5b) in `fawam/analysis_gap.py` übernehmen.
- [ ] Hinweis: `analysis.json` wurde am 2026-11-04 in 4 aufgezeichnete Fail-Ordner unter `results/` geschrieben
      (table_height_n10 m10/m05, eval_hard handover ep0002). Es wurde nichts gelöscht.

### Paper-Protokoll (`p5f_paper`)

- [ ] Ein Lauf `paper_P_dual` mit `NUM=5` (3 Adaption + 2 Eval) live: 3 Adapter-Ordner `seed*_every_v1..3`, ab
      Episode 4 `lff.phase = eval` in meta.json, kein weiterer Learner-Aufruf.
- [ ] Innen + außen sequentiell (Replay im Learner lädt ein zweites Modell): Speicher auf der H100 prüfen
      (Server 17 + Sim 5 + Learner ~12 + Replay 17 GB). Bei Engpass `outside=parallel`.
- [ ] 25 Schritte mit Batch 1 sind deutlich weniger Daten als im Paper (8 Clips × 8 Frames). Gradient-Accumulation
      oder Batch > 1 ergänzen, falls `paper_P_dual` und `paper_P_dual_150` stark auseinanderliegen.
- [ ] `cluster/online_adapt_job.sh` ist durch `p5f_paper` überholt (gleiches Protokoll, aber ohne Signale und
      ohne äußere Stufe). Nur noch als Fallback.

### Signal-Ablation und Integrationen (2026-11-04, [METHOD.md](METHOD.md) Abschnitte 5–6)

Offline-Ergebnis auf 4 echten Fails, Fehlerzeitpunkt als Latent:

| Episode | Orakel | P | P + R | Δz (Orakel / I) |
|---|---|---|---|---|
| handover ep0002 (drop) | 13 | 2 | 13 | 0 / 0 |
| card m10 ep0000 | 4 | 6 | 4 | −8,5 / 0 cm |
| card m10 ep0001 | 4 | 21 | 4 | −8,4 / 0 cm |
| bottles m05 ep0000 | 4 | 3 | 4 | 0 / 0 |

- [ ] `pose_probe fit` und `probe gap-stats` auf P0 neu laufen lassen (aktuell aus den Tischhöhen-Läufen und eval_hard).
- [ ] P-Zeitpunkt (Beginn der Lücke) kalibrieren: Schwelle z = 2,5 aus `gap_stats.json`. Bei unbekannten Tasks gilt der globale Wert.
- [ ] `pref_beta` (500) und `lambda_pref` einstellen: Die Flow-Losses sind ~0,05, Grad-Norm im ersten Schritt 244 (wird auf 1 geclippt).
- [ ] Guard (`lff_bank`) und `every_episode` (`lff_ttt_persist`) einmal live laufen lassen.
- [ ] vLLM-Version mit Qwen3-VL-Support im `vllm`-Env (für `p5b_vlm`). Fallback: Qwen2.5-VL-7B (`VLM_MODEL=` im Sweep).
- [ ] T³VF filtert über **Aktionsvarianz** (K = 5 Samples), wir über die Lücke. Variante mit Aktionsvarianz als Filter
      fehlt noch. Dafür braucht es K Aktions-Samples pro Chunk im Replay.
- [ ] RePO-VLA-Idee: Recovery-Segmente (nach Kontext-Reset bis Erfolg) als eigene Episoden ab t = 0 ins Training.
      `latent_data.load_episode` schneidet aktuell beim ersten Reset ab.
- [ ] Imaginations-Probe liefert für die Tischhöhe Δ ≈ 0 (FINDINGS A0). Bei anderen Fehlertypen (handover, Platzieren)
      prüfen, ob sie dort Information trägt.

## 2. Code, der noch nie gelaufen ist

| Punkt | Wie prüfen | Priorität |
|---|---|---|
| Expert-Replay **mit Modell** (`DRIVER=expert_replay`). Ohne Modell geprüft: 3/3 und 2/2 gelöst. Lokal scheiterte es am GPU-Speicher. | 1 Episode, danach `actions/planned` vs. `actions/model_proposal` im h5 ansehen | hoch (P2 hängt daran) |
| `reprompt_on_gap`, `neg_on_gap` (Prompt-Update mitten in der Episode) | 1 Episode, Event `update_prompt` im h5 | mittel |
| `oracle_on_gap`, `reset_on_oracle` (Sim-Analyse) | 1 Episode, `interventions` im h5 enthält `analysis` | mittel |
| `vlm_on_gap` + vLLM-Start im Job + `decode_pred` | 1 Job aus `p1_vlm`, `logs/<job>/vllm.log` + Latenz in `analysis.latency_s` | mittel |
| `retry_plain`, `retry_memory`, `retry_text` (zweiter Versuch, Prefill des Fehlversuchs). Prefill selbst ist geprüft. | 1 Fail-Episode, Ordner `..._attempt2` muss entstehen | mittel |
| `cluster/train_job.sh` mit `OUTSIDE=sequential` (Replay → Pairs → außen) | ein Adapter aus `adapter_grid.py train --only d1_dual_m05_n3` | hoch |
| `cluster/online_adapt_job.sh` (D3-online, 3 Runden) | 1 Job, 3 Adapter-Ordner `adapter_r1..3` | mittel |
| `cluster/adapter_grid.py banks/methods` (erzeugt `p3_adapters.yaml`, `p4_icl.yaml`) | nach P2 laufen lassen, Dry-Run von `submit_sweep.py` | hoch |
| ICL-Training E3 auf dem echten Modell (nur der Unit-Test mit kleinem Modell lief) | `adapter_grid.py train --only e3_` | mittel |
| `fawam.analysis_gap compare` auf echten Sweep-Ergebnissen | nach den ersten Jobs | hoch |

## 3. Nachkalibrieren, sobald P0 (Baseline) fertig ist

- [ ] **Gap-Statistik neu erzeugen.** Sie stammt aktuell nur aus eval_hard und z00 (2–3 Tasks). Für unbekannte Tasks
      greift der globale Wert, und die B-Methoden lösen dann zu früh oder zu spät aus.
      `python -m fawam.probe gap-stats results/sweeps/p0_baseline_table/base/table_h_z00 results/sweeps/p0_baseline_hard`
- [ ] **Trigger-Schwelle wählen** (z = 2,5 ist geraten). Aus P0 eine ROC der Online-Lücke **vor** dem Fehler
      berechnen und eine Schwelle mit z. B. ≤ 10 % Fehlalarmen auf Erfolgen nehmen. In der Smoke-Episode löste es schon
      bei Chunk 2 aus.
- [ ] **Failure-Probe und Steering-Vektor neu fitten** (aktuell aus 10 Fails, Probe-AUC 0,91 nur per Chunk-Split, also
      optimistisch). Dafür zuerst `analysis/analyze.py` auf die P0-Zellen laufen lassen (liefert `failure_latent`),
      dann `fawam.probe failure-probe` / `steer`. Für eine ehrliche AUC per Episode splitten.

## 4. Fehlende Analysen

- [x] **A0 (erledigt 2026-11-04, FINDINGS Abschnitt 5):** Die Vorstellung zeigt den Greifer nicht tiefer als real.
      Folgefrage offen: Objekt-Probe ist zu ungenau (4,9 cm). Bessere Objekt-Probe nur auf den Kopfkamera-Latents testen.
- [ ] ~~A0: Stellt sich das Modell die richtige Tiefe vor?~~ Eine Probe, die aus dem vorhergesagten
      Handgelenk-Latent die Greiferhöhe schätzt, trainiert auf beobachteten Latents mit Sim-Labels (`sim/*_ee_pose`).
      Ohne sie bleibt die Imagination-Execution-Lücke eine Interpretation aus 2 angesehenen Episoden plus der
      Distanz-Statistik.
- [ ] **C1: Zuverlässigkeit der VLM-Fehleranalyse offline.** VLM auf aufgezeichneten Fails und Erfolgen gegen die
      Sim-Labels aus `analyze.py`. Code für die Abfrage existiert (`vlm_analysis`), das Auswertungsskript fehlt.
- [ ] Gepaarte Statistik (McNemar / gepaarter Bootstrap über Seeds) in `analysis_gap compare` ergänzen. Aktuell
      gibt es nur Wilson-CIs pro Zelle.
- [ ] Tischhöhen-Analyse auf Aufgaben jenseits von Greifen erweitern (Platzieren, Übergabe).

## 5. Bekannte Schwächen der Implementierung

- Kontext-Reset (B3) kodiert das aktuelle Bild als neuen Startframe. Die Frame-Zählung des Clients
  (`frames/latent_index`) weicht danach um ein Latent ab. `analyze.py` weiß davon nichts.
- `latent_data.load_episode` schneidet Episoden am ersten Kontext-Reset ab. Rollouts mit B3 sind also nur bis dahin
  Trainingsdaten.
- Der `decode_pred` eines einzelnen Chunks läuft ohne kausalen VAE-Kontext. Die Bilder fürs VLM sind etwas unschärfer.
- Methoden-Specs mit Kommas (JSON) dürfen nicht direkt in `submit_sweep.py`, weil `sbatch --export` an Kommas trennt.
  Solche Specs gehören als Datei nach `configs/methods/`.
- Bei `steps50` und `bon8_*` stimmen die Zeitschätzungen nicht (~3× langsamer). `time` im Sweep ggf. erhöhen.
- Lokale 5090 wird mit Nutzer `blank` geteilt (5–13 GB). Live-Tests lokal vorher mit `nvidia-smi` prüfen.

## 6. Später

- Weitere Verschiebungen analog zum Paper: Reibung, Objektmasse/Schwerpunkt, Kameraposition.
- WorldScape Policy 2.0 als ICL-natives Vergleichsmodell.
- Übertragung auf die anderen Benchmarks (RoboDojo über XPolicyLab; RoboLab und MolmoSpaces-CS brauchen einen
  DROID-Checkpoint).
- Aufräumen: `results/_smoke` und `results/_smoke_expert` (unvollständiger `_running`-Ordner) sind Testläufe vom
  2026-11-03. Löschen erst nach deinem OK.
