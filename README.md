# failure-aware-wam

Testbed for a failure-aware world-action model: LingBot-VA (policy server) evaluated in RoboTwin (simulation client),
with per-episode recording of latents, hidden states, actions and sim state, offline analysis, and (`fawam/`)
test-time methods, few-shot adapters and in-context learning on top of the frozen model.

- **Goal and current plan: [docs/GOAL.md](docs/GOAL.md), [docs/PLAN.md](docs/PLAN.md)** (2026-11-10; they replace the scope of the
  older experiment plans below)
- Findings so far: [docs/FINDINGS.md](docs/FINDINGS.md) (the imagination-execution gap)
- Experiment plan, methods, data generation, training, cluster workflow: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)

## Layout
- `fawam/` – `server.py` (LingBot-VA server + methods; default method = bit-identical to upstream), `methods.py`
  (method presets), `adapters.py` (dual-site Adapter+ / external residual / LoRA / steering), `train_adapter.py`,
  `replay.py` (teacher-forced replay), `latent_data.py` (episode.h5 -> training sequences), `robotwin_hooks.py`
  (expert replay, online intervention, oracle/VLM analysis), `probe.py`, `demo_bank.py`, `analysis_gap.py`,
  `text_cache.py`, `tests/`
- `analysis/` – `analyze.py` (tables, plots, videos, `index.html` viewer), `attention_replay.py` / `attention_render.py`
  (offline replay of `episode.h5` recording action→video attention), `table_height.py`
- `scripts/` – local runs: `start_server.sh` (fawam server), `stop_server.sh`, `run_eval.sh` (METHOD=, DRIVER=, SEED=),
  `run_all.sh`, `run_table_height.sh`, `analyze_results.sh`, …
- `cluster/` – SLURM: `eval_job.sh`, `train_job.sh`, `online_adapt_job.sh`, `submit_sweep.py`, `adapter_grid.py`,
  `setup_hk2.sh`, `env.sh`, cluster profiles, env freezes
- `configs/` – `robotwin_task_config/table_h_*.yml` (fixed table height offsets), `sweeps/*.yaml`, `methods/*.json`
- `artifacts/` – `gap_stats.json`, `probe/` (tracked); `text_emb/`, `adapters/`, `demo_bank/` (local)
- `patches/lingbot-va.patch` – recording hooks + fawam client in `wan_va_server.py`, `eval_polict_client_openpi.py`,
  `modules/model.py` (incl. the `key_exclude` hook used by imagination guidance / history dropout)

## Setup
```bash
./setup.sh   # clones lingbot-va @ 7c6ffa9 (+ patch) and RoboTwin @ 2eeec32, copies task configs
```
Then download the LingBot-VA checkpoints into `lingbot-va/checkpoints/` and the RoboTwin assets, and build cuRobo.
On the cluster: `cluster/setup_hk2.sh` (see its header).

Environments (RTX 5090 / sm_120):
- `lingbot` – server + training, torch 2.9 cu128
- `robotwin` – client, torch 2.8 cu129, cuRobo v0.7.8 built with `TORCH_CUDA_ARCH_LIST=12.0`
  (env's CUDA 12.9 nvcc + gcc13, `CPATH`/`LIBRARY_PATH=$ENV/targets/x86_64-linux/{include,lib}`)

Quick checks: `python -m fawam.tests.test_adapters` (lingbot env, GPU), `fawam/tests/test_expert_replay.py`
(robotwin env), `python -m fawam.replay --episodes <recorded episode> --out x.npz` (replay must give
`replay_vs_recorded=0`).

`results/` and `logs/` are not tracked.
