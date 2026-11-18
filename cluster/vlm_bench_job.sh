#!/bin/bash
# VLM failure-analysis benchmark (fawam/vlm_bench.py): one VLM on this GPU, all input/prompt variants on recorded
# failures, scored against the oracle. Env: VLM_MODEL [VARIANTS="a b"] [MAXLEN=32768]
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
set -uo pipefail
source "${FAWAM_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}/cluster/env.sh"
: "${VLM_MODEL:?}"
cd "$FAWAM_ROOT"
# per-job port: several VLM jobs can share a node (8001 for all of them sent requests to the wrong server)
VLM_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 20000))
LOGDIR=logs/vlm_bench_${SLURM_JOB_ID:-$$}; mkdir -p "$LOGDIR"
HF_HUB_OFFLINE=1 "$PY_VLLM" -m vllm.entrypoints.openai.api_server --model "$VLM_MODEL" --port "$VLM_PORT" \
    --gpu-memory-utilization 0.92 --max-model-len "${MAXLEN:-32768}" > "$LOGDIR/vllm.log" 2>&1 &
VPID=$!
trap 'kill $VPID 2>/dev/null' EXIT
until grep -q "Application startup complete\|Traceback" "$LOGDIR/vllm.log"; do sleep 10; done
grep -q Traceback "$LOGDIR/vllm.log" && { tail -30 "$LOGDIR/vllm.log"; exit 1; }
"$PY_RT" -m fawam.vlm_bench run --model "$VLM_MODEL" --endpoint "http://127.0.0.1:$VLM_PORT/v1" ${VARIANTS:+--variants $VARIANTS}
"$PY_RT" -m fawam.vlm_bench summary
