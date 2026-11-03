# failure-aware-wam

Testbed for a failure-aware world-action model: LingBot-VA (policy server) evaluated in RoboTwin (simulation client),
with per-episode recording of latents, hidden states, actions and sim state, plus offline analysis.

## Layout
- `analysis/` – `analyze.py` (tables, plots, videos, `index.html` viewer), `attention_replay.py` / `attention_render.py`
  (offline replay of `episode.h5` recording action→video attention), `table_height.py`
- `scripts/` – `start_server.sh`, `stop_server.sh`, `run_eval.sh`, `run_all.sh`, `run_table_height.sh`, `analyze_results.sh`, …
- `patches/lingbot-va.patch` – recording hooks in `wan_va_server.py`, `eval_polict_client_openpi.py`, `modules/model.py`
- `configs/robotwin_task_config/` – `table_h_{p05,z00,m05,m10}.yml` (fixed `table_height_bias`)

## Setup
```bash
./setup.sh   # clones lingbot-va @ 7c6ffa9 (+ patch) and RoboTwin @ 2eeec32, copies task configs
```
Then download the LingBot-VA checkpoints into `lingbot-va/checkpoints/` and the RoboTwin assets, and build cuRobo.

Environments (RTX 5090 / sm_120):
- `lingbot` – server, torch 2.9 cu128
- `robotwin` – client, torch 2.8 cu129, cuRobo v0.7.8 built with `TORCH_CUDA_ARCH_LIST=12.0`
  (env's CUDA 12.9 nvcc + gcc13, `CPATH`/`LIBRARY_PATH=$ENV/targets/x86_64-linux/{include,lib}`)

`results/` and `logs/` are not tracked.
