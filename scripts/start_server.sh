#!/bin/bash
# Start the LingBot-VA RoboTwin policy server in the background (T5 + VAE offloaded to CPU, ~11 GB VRAM).
# Usage: start_server.sh [port]   -> waits until the server listens; log in logs/lingbot_server.log
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PORT=${1:-29056}
cd "$ROOT/lingbot-va"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/lingbot_server.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} nohup ~/miniconda3/envs/lingbot/bin/python -m torch.distributed.run \
    --nproc_per_node 1 --master_port $((PORT + 5)) wan_va/wan_va_server.py --config-name robotwin --port "$PORT" \
    --save_root visualization/ --model_path checkpoints/lingbot-va-posttrain-robotwin --enable_offload \
    > "$LOG" 2>&1 &
echo $! > "$ROOT/logs/lingbot_server.pid"
until grep -q "server listening\|Traceback" "$LOG"; do sleep 5; done
tail -1 "$LOG"
