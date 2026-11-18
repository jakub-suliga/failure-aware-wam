#!/bin/bash
# Prediction comparison base model vs. adapted model (fawam/compare_pred.py) for one paper-protocol cell.
# Env: METHOD (e.g. paper_P_dual), SETTING (table_h_m05), N (episodes per repeat, default 1).
# Episodes: the base run's first eval episodes of each repeat; adapter: the frozen v3 adapter of the same repeat.
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
set -uo pipefail
source "${FAWAM_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}/cluster/env.sh"
: "${METHOD:?}" "${SETTING:?}"
cd "$FAWAM_ROOT"
S=results/sweeps/p5f_paper
pairs=()
for rep in 10000 20000 30000 40000; do
    ad=$S/$METHOD/$SETTING/lff/move_playingcard_away/seed$((rep + 2))_every_v3
    [ -f "$ad/adapter.pt" ] || { echo "no adapter $ad"; continue; }
    for i in $(seq 3 $((2 + ${N:-1}))); do
        ep=$(ls -d $S/base/$SETTING/episodes/{fail,success}/move_playingcard_away/ep*_seed$((rep + i)) 2>/dev/null | head -1)
        [ -n "$ep" ] && pairs+=("$ep=$ad")
    done
done
echo "${#pairs[@]} pair(s)"
"$PY_LB" -m fawam.compare_pred --pairs "${pairs[@]}" --out "results/analysis/compare_pred/${METHOD}_$SETTING" 2>&1 |
    grep --line-buffered -v "it/s\|s/it\|FRAME START\|INFO"
