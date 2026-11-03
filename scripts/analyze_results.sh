#!/bin/bash
# Full analysis of result roots (each contains episodes/): analyze.py, then attention replay + render into
# <root>/analysis/. The replay loads the model (~17 GB VRAM) and so waits until the policy server from
# start_server.sh is stopped. Episodes already replayed are skipped, so re-running after new evals only adds the new ones.
# Runs one at a time (lock). Called automatically by run_eval.sh / run_all.sh / run_table_height.sh.
# Usage: analyze_results.sh <results root> [...]
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY_RT=~/miniconda3/envs/robotwin/bin/python
PY_LB=~/miniconda3/envs/lingbot/bin/python
PID_FILE="$ROOT/logs/lingbot_server.pid"
ROOTS=()
for r in "$@"; do ROOTS+=("$(realpath -m "$r")"); done

exec 9>"$ROOT/logs/analysis.lock"
flock 9

for r in "${ROOTS[@]}"; do
    echo "[$(date)] analyze $r"
    $PY_RT "$ROOT/analysis/analyze.py" "$r" || echo "analyze $r failed"
done

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[$(date)] waiting for the policy server to stop before attention replay"
    while [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; do sleep 30; done
fi

for r in "${ROOTS[@]}"; do
    echo "[$(date)] attention replay $r"
    eps=$(ls -d "$r"/episodes/{success,fail}/*/* 2>/dev/null)
    [ -n "$eps" ] || { echo "no episodes in $r"; continue; }
    (cd "$ROOT/lingbot-va" && PYTORCH_ALLOC_CONF=expandable_segments:True $PY_LB ../analysis/attention_replay.py $eps \
        --out "$r/analysis/attention" 2>&1 | grep --line-buffered -v "it/s\|s/it\|FRAME START\|INFO")
    echo "[$(date)] attention render $r"
    $PY_RT "$ROOT/analysis/attention_render.py" "$r" || echo "render $r failed"
done
echo "[$(date)] analysis DONE: ${ROOTS[*]}"
