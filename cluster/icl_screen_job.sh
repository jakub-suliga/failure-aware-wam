#!/bin/bash
# Offline ICL screening (docs/ICL.md): one GPU, no simulator. Env: FAILS FIXES OUT [VARIANTS LIMIT=20 ANY_SEED=0]
#   FAILS = P0 roots with analysis.json next to the failures; FIXES = P2 expert-replay roots (same seeds)
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
set -euo pipefail
source "${FAWAM_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}/cluster/env.sh"
: "${FAILS:?}" "${FIXES:?}" "${OUT:?}"
cd "$FAWAM_ROOT"
ARGS=(--fails $FAILS --fixes $FIXES --limit "${LIMIT:-20}" --out "$OUT")
[ -n "${VARIANTS:-}" ] && ARGS+=(--variants $VARIANTS)
[ "${ANY_SEED:-0}" = 1 ] && ARGS+=(--any_seed)
"$PY_LB" -m fawam.icl_screen "${ARGS[@]}"
