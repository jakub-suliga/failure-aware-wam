#!/bin/bash
# Adapter training job (D): inside stage, optional replay + outside stage (sequential dual-site, as in the paper).
# Env: OUT EPISODES (space-separated roots) and any of
#   KIND=dual|inside|lora|outside  STREAMS="video action"  LIMIT=<n episodes>  LIMIT_PER_TASK=<n>  SETTINGS="table_h_m05"  SUCCESS=any|true|false
#   TASKS="a b" (default: all in EPISODES)  STEPS=400  LAMBDA_V=1 LAMBDA_A=1  OUTSIDE=sequential|parallel|none  CONTEXT_K=0  EXTRA="<more train args>"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
set -euo pipefail
# sbatch runs a copy of this script from the spool dir: locate the repo via FAWAM_ROOT (exported by submit_sweep.py)
source "${FAWAM_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}/cluster/env.sh"
: "${OUT:?}" "${EPISODES:?}"
KIND=${KIND:-dual}; STREAMS=${STREAMS:-video action}; STEPS=${STEPS:-400}; OUTSIDE=${OUTSIDE:-sequential}
SEL=(--episodes $EPISODES --success "${SUCCESS:-any}")
[ -n "${LIMIT:-}" ] && SEL+=(--limit "$LIMIT")
[ -n "${LIMIT_PER_TASK:-}" ] && SEL+=(--limit_per_task "$LIMIT_PER_TASK")
[ -n "${LIMIT_PER_CELL:-}" ] && SEL+=(--limit_per_cell "$LIMIT_PER_CELL")
[ -n "${SETTINGS:-}" ] && SEL+=(--settings $SETTINGS)
[ -n "${TASKS:-}" ] && SEL+=(--tasks $TASKS)
[ -n "${DRIVERS:-}" ] && SEL+=(--drivers $DRIVERS)
cd "$FAWAM_ROOT"
mkdir -p "$OUT"
MODEL=${MODEL_PATH:-$FAWAM_ROOT/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin}
"$PY_LB" -m fawam.text_cache build $EPISODES --model_path "$MODEL" ${TEXT_EXTRA:+--extra "$TEXT_EXTRA"}
if [ "$KIND" != outside ]; then
    "$PY_LB" -m fawam.train_adapter inside "${SEL[@]}" --kind "$KIND" --inside_streams $STREAMS --steps "$STEPS" \
        --lambda_video "${LAMBDA_V:-1}" --lambda_action "${LAMBDA_A:-1}" --context_k "${CONTEXT_K:-0}" \
        --model_path "$MODEL" --out "$OUT" ${EXTRA:-}
fi
INIT=(); ADAPTER=()
if [ "$KIND" != outside ]; then INIT=(--init "$OUT"); ADAPTER=(--adapter "$OUT"); fi
case "$OUTSIDE" in
    sequential)  # paper: outside residual trained on predictions of the adapted model
        "$PY_LB" -m fawam.replay "${SEL[@]}" "${ADAPTER[@]}" --model_path "$MODEL" --out "$OUT/pairs.npz"
        "$PY_LB" -m fawam.train_adapter outside "${INIT[@]}" --pairs "$OUT/pairs.npz" --out "$OUT" ;;
    parallel)    # ablation: outside residual on the recorded base-model predictions
        "$PY_LB" -m fawam.train_adapter outside "${SEL[@]}" "${INIT[@]}" --out "$OUT" ;;
    none) ;;
esac
echo "[$(date)] adapter ready: $OUT"
