#!/bin/bash
# p21 ICL stage-1 training (plan v2, 2026-11-21). Env: NAME SETTING (sa|sb) CTX (C0..C8) ARCH (A1 = Adapter+ video,
# A2 = LoRA r16 video, A3 = gated cross-attention,
# T1/T2/T4/T5 = A1 with an adapter gate) SAMPLING (uniform|win50|w4) STEPS (1500). Reads its gate results/gates/p21_setting_<S>.json at
# start and ends at once with status "skipped" if the setting did not pass. Resumes from the last checkpoint (250 steps).
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
set -euo pipefail
source "${FAWAM_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}/cluster/env.sh"
cd "$FAWAM_ROOT"
: "${NAME:?}" "${SETTING:?}" "${CTX:?}"
OUT=artifacts/adapters/$NAME
mkdir -p "$OUT"
G=results/gates/p21_setting_${SETTING}.json
if [ -f "$G" ] && ! "$PY_RT" -c "import json,sys; sys.exit(0 if json.load(open('$G'))['passed'] else 1)"; then
    echo "{\"skipped\": \"gate p21_setting_${SETTING}\"}" > "$OUT/skipped.json"; echo "skipped: gate $G"; exit 0
fi
case "${ARCH:-A1}" in
    A1) KIND=(--kind inside --inside_streams video) ;;
    A2) KIND=(--kind lora --inside_streams video --lora_rank 16) ;;
    A3) KIND=(--kind combo --xattn_every 3 --ctx_mode separate) ;;  # Flamingo gated cross-attention only
    T1) KIND=(--kind inside --inside_streams video --gate t1) ;;
    T2) KIND=(--kind inside --inside_streams video --gate t2) ;;
    T4) KIND=(--kind inside --inside_streams video --gate t4 --adapter_up_init normal --ctx_mode both) ;;
    T5) KIND=(--kind inside --inside_streams video --gate t5) ;;
    *) echo "architecture ${ARCH} not implemented"; echo "{\"skipped\": \"arch ${ARCH} not implemented\"}" > "$OUT/skipped.json"; exit 0 ;;
esac
ROOTS=${ROOTS:-"results/sweeps/p21_data_${SETTING}/base/icl_${SETTING} results/sweeps/p21_data_${SETTING}/base/icl_${SETTING}_v1"}
"$PY_LB" -m fawam.train_adapter inside --episodes $ROOTS --snap_roots $ROOTS --icl_ctx "$CTX" \
    --sampling "${SAMPLING:-uniform}" "${KIND[@]}" --steps "${STEPS:-1500}" --lambda_video 1 --lambda_action 0 \
    --val_every 250 --save_every 250 --max_hours "${MAX_HOURS:-3.6}" --out "$OUT" ${EXTRA:-}
echo "[$(date)] adapter ready: $OUT"
