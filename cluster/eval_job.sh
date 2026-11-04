#!/bin/bash
# One SLURM job = one GPU running N_PAR (server, simulator) pairs side by side, each on its own seed range.
# Submitted by cluster/submit_sweep.py; can also be run by hand on an interactive GPU node.
# Env: TASK CONFIG METHOD NUM (episodes per pair) SEED0 (first seed index, pair i uses SEED0+i) N_PAR OUT DRIVER
#      VLM=1 starts a local vLLM server for analyzer=vlm methods (model: VLM_MODEL, port 8001)
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
set -uo pipefail
# sbatch runs a copy of this script from the spool dir: locate the repo via FAWAM_ROOT (exported by submit_sweep.py)
source "${FAWAM_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}/cluster/env.sh"
: "${TASK:?}" "${CONFIG:?}" "${OUT:?}"
METHOD=${METHOD:-base}; NUM=${NUM:-10}; SEED0=${SEED0:-0}; N_PAR=${N_PAR:-2}; DRIVER=${DRIVER:-policy}
JOB=${SLURM_JOB_ID:-local$$}
LOGDIR=$OUT/logs/$JOB
mkdir -p "$LOGDIR"
BASE_PORT=$(( 20000 + (${SLURM_JOB_ID:-$$} % 400) * 20 ))
echo "[$(date)] job $JOB on $(hostname): $TASK $CONFIG method=$METHOD driver=$DRIVER num=$NUM seeds=$SEED0..$((SEED0+N_PAR-1)) out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT

if [ "${VLM:-0}" = 1 ]; then
    "$PY_VLLM" -m vllm.entrypoints.openai.api_server --model "${VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}" \
        --port 8001 --gpu-memory-utilization 0.22 --max-model-len 16384 > "$LOGDIR/vllm.log" 2>&1 &
    pids+=($!)
fi

cd "$FAWAM_ROOT/lingbot-va" || exit 1
for i in $(seq 0 $((N_PAR - 1))); do
    port=$((BASE_PORT + 2 * i))
    "$PY_LB" -m torch.distributed.run --nproc_per_node 1 --master_port $((port + 1)) -m fawam.server \
        --config-name robotwin --port "$port" --save_root "$LOGDIR/vis" \
        --model_path "${MODEL_PATH:-checkpoints/lingbot-va-posttrain-robotwin}" --enable_offload \
        > "$LOGDIR/server_$i.log" 2>&1 &
    pids+=($!)
done
for i in $(seq 0 $((N_PAR - 1))); do
    until grep -q "server listening\|Traceback" "$LOGDIR/server_$i.log" 2>/dev/null; do sleep 10; done
    grep -q Traceback "$LOGDIR/server_$i.log" && { echo "server $i failed"; tail -30 "$LOGDIR/server_$i.log"; exit 1; }
done
if [ "${VLM:-0}" = 1 ]; then
    until grep -q "Uvicorn running\|Application startup complete\|Traceback" "$LOGDIR/vllm.log"; do sleep 10; done
fi

export ROBOTWIN_ROOT="$FAWAM_ROOT/RoboTwin" PYTHONWARNINGS=ignore::UserWarning
cpids=()
for i in $(seq 0 $((N_PAR - 1))); do
    port=$((BASE_PORT + 2 * i))
    "$PY_RT" -m evaluation.robotwin.eval_polict_client_openpi \
        --config policy/ACT/deploy_policy.yml --overrides --task_name "$TASK" --task_config "$CONFIG" \
        --train_config_name 0 --model_name 0 --ckpt_setting 0 --seed $((SEED0 + i)) --policy_name ACT \
        --method "$METHOD" --driver "$DRIVER" --save_root "$OUT" --video_guidance_scale 5 --action_guidance_scale 1 \
        --test_num "$NUM" --port "$port" > "$LOGDIR/client_$i.log" 2>&1 &
    cpids+=($!)
done
rc=0
for p in "${cpids[@]}"; do wait "$p" || rc=1; done
for i in $(seq 0 $((N_PAR - 1))); do grep -a "Success rate" "$LOGDIR/client_$i.log" | tail -1; done
rm -rf "$LOGDIR/vis"
echo "[$(date)] done rc=$rc"
exit $rc
