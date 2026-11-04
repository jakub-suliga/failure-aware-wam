#!/bin/bash
# Start the LingBot-VA RoboTwin policy server (fawam.server: upstream inference + the test-time methods of
# docs/EXPERIMENTS.md; with the default method bit-identical to wan_va_server.py) in the background
# (T5 + VAE offloaded to CPU, ~11 GB VRAM). The method is chosen per episode by the client (METHOD in run_eval.sh).
# Usage: start_server.sh [port]   -> waits until the server listens; log in logs/lingbot_server.log
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PORT=${1:-29056}
cd "$ROOT/lingbot-va"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/lingbot_server.log"
PYTHONPATH="$ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} nohup ${PY_LB:-~/miniconda3/envs/lingbot/bin/python} -m torch.distributed.run \
    --nproc_per_node 1 --master_port $((PORT + 5)) -m fawam.server --config-name robotwin --port "$PORT" \
    --save_root visualization/ --model_path checkpoints/lingbot-va-posttrain-robotwin --enable_offload \
    > "$LOG" 2>&1 &
echo $! > "$ROOT/logs/lingbot_server.pid"
until grep -q "server listening\|Traceback" "$LOG"; do sleep 5; done
tail -1 "$LOG"
