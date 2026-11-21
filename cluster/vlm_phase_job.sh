#!/bin/bash
# p20 D7: VLM phase labelling (own vLLM port per job). Env: VLM_MODEL (default Qwen3-VL-32B), ROOTS (episode roots).
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
set -uo pipefail
source "${FAWAM_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}/cluster/env.sh"
cd "$FAWAM_ROOT"
VLM_MODEL=${VLM_MODEL:-Qwen/Qwen3-VL-32B-Instruct}
ROOTS=${ROOTS:-results/sweeps/p20_d2_base results/sweeps/p20_d4_tf}
VLM_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 20000))
LOGDIR=logs/vlm_phase_${SLURM_JOB_ID:-$$}; mkdir -p "$LOGDIR"
HF_HUB_OFFLINE=1 "$PY_VLLM" -m vllm.entrypoints.openai.api_server --model "$VLM_MODEL" --port "$VLM_PORT" \
    --gpu-memory-utilization 0.92 --max-model-len 16384 > "$LOGDIR/vllm.log" 2>&1 &
VPID=$!
trap 'kill $VPID 2>/dev/null' EXIT
until grep -q "Application startup complete\|Traceback" "$LOGDIR/vllm.log"; do sleep 10; done
grep -q Traceback "$LOGDIR/vllm.log" && { tail -30 "$LOGDIR/vllm.log"; exit 1; }
"$PY_RT" -m fawam.vlm_phase run $ROOTS --endpoint "http://127.0.0.1:$VLM_PORT/v1" --model "$VLM_MODEL" --limit 60
