#!/bin/bash
# M3 residual head on the hid pairs + training gate (fawam/train_hid.py). Env: ROOTS OUT [VAL_ROOTS] [STEPS] [EXTRA]
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
set -euo pipefail
source "${FAWAM_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}/cluster/env.sh"
: "${ROOTS:?}" "${OUT:?}"
cd "$FAWAM_ROOT"
MODEL=${MODEL_PATH:-$FAWAM_ROOT/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin}
"$PY_LB" -m fawam.text_cache build $ROOTS ${VAL_ROOTS:-} --model_path "$MODEL"
"$PY_LB" -m fawam.train_hid --roots $ROOTS ${VAL_ROOTS:+--val_roots $VAL_ROOTS} --out "$OUT" \
    --steps "${STEPS:-1500}" --model_path "$MODEL" ${EXTRA:-}
