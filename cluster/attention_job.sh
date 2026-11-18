#!/bin/bash
# Spatial attention of the action tokens (analysis/attention_replay.py + attention_render.py) for the report's sample
# episodes of variants that run the unchanged weights without a memory. Env: TODO = file with one episode dir per line
# (fawam/report.py writes results/report/attention_todo.txt). Videos land in <cell>/analysis/attention/videos/.
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
set -uo pipefail
source "${FAWAM_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}/cluster/env.sh"
: "${TODO:?}"
cd "$FAWAM_ROOT"
for root in $(sed 's|/episodes/.*||' "$TODO" | sort -u); do
    eps=$(grep "^$root/episodes/" "$TODO")
    echo "[$(date)] $root: $(echo "$eps" | wc -l) episode(s)"
    [ -f "$root/analysis/episodes.csv" ] || "$PY_RT" analysis/analyze.py "$root" || { echo "analyze failed for $root"; continue; }
    (cd lingbot-va && "$PY_LB" ../analysis/attention_replay.py $eps --out "$root/analysis/attention" 2>&1 |
        grep --line-buffered -v "it/s\|s/it\|FRAME START\|INFO")
    "$PY_RT" analysis/attention_render.py "$root" || echo "render failed for $root"
done
find $(sed 's|/episodes/.*||' "$TODO" | sort -u) -path '*/analysis/attention/videos/*.mp4' | wc -l
