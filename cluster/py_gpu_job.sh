#!/bin/bash
# Generic one-GPU python job (plan v2 diagnostics): CMD="python module args" run in the lingbot env from the repo root.
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
set -euo pipefail
source "${FAWAM_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}/cluster/env.sh"
cd "$FAWAM_ROOT"
: "${CMD:?}"
echo "[$(date)] $(hostname): $CMD"
eval "$PY_LB $CMD"
echo "[$(date)] done"
