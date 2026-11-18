#!/bin/bash
# Offline VLM failure analysis (signals gap+vlm -> analysis_gap+vlm.json next to every failure), for the meta-ICL
# training data. Starts a local vLLM server with the VLM on this GPU. Env: ROOTS (space-separated episode roots)
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
set -uo pipefail
source "${FAWAM_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}/cluster/env.sh"
: "${ROOTS:?}"
cd "$FAWAM_ROOT"
# per-job port: several VLM jobs can share a node (8001 for all of them sent requests to the wrong server)
VLM_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 20000))
MODEL=${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
LOGDIR=logs/vlm_analysis_${SLURM_JOB_ID:-$$}; mkdir -p "$LOGDIR"
HF_HUB_OFFLINE=1 "$PY_VLLM" -m vllm.entrypoints.openai.api_server --model "$MODEL" --port "$VLM_PORT" \
    --gpu-memory-utilization 0.85 --max-model-len 16384 > "$LOGDIR/vllm.log" 2>&1 &
VPID=$!
trap 'kill $VPID 2>/dev/null' EXIT
until grep -q "Application startup complete\|Traceback" "$LOGDIR/vllm.log"; do sleep 10; done
grep -q Traceback "$LOGDIR/vllm.log" && { tail -30 "$LOGDIR/vllm.log"; exit 1; }
# VLM_MODES: rollout (keyframes of the whole rollout + prediction-error curve -> analysis_gap+vlm.json) and/or
# frame (only the failure moment -> analysis_gap+vlm_frame.json)
for mode in ${VLM_MODES:-rollout}; do
    "$PY_RT" -m fawam.failure_analysis --signals gap+vlm --vlm_mode "$mode" --model "$MODEL" --endpoint "http://127.0.0.1:$VLM_PORT/v1" ${VLM_DETAIL:+--vlm_detail} $ROOTS | tail -n 3
done
n=$(find $ROOTS -path '*/fail/*' -name "analysis_gap+vlm${VLM_DETAIL:+_detail}.json" | wc -l)
bad=$(find $ROOTS -path '*/fail/*' -name 'analysis_gap+vlm*.json' -print0 | xargs -0 -r grep -l '"vlm_error"' | wc -l)
echo "VLM analyses: $n, with a failed VLM call: $bad"
[ "$n" -gt 0 ] && [ "$bad" -lt $((n / 10 + 1)) ]
