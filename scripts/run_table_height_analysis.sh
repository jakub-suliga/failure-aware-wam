#!/bin/bash
# Full analysis (analyze.py + attention replay/render) for each table-height result root, like results/eval_hard/analysis.
# Start detached:  setsid nohup scripts/run_table_height_analysis.sh > logs/table_height_analysis.log 2>&1 < /dev/null &
# Env override: ROOTS="results/table_height_n10/m05 ..."
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ROOTS=${ROOTS:-"results/table_height_n10/m05 results/table_height_n10/m10 results/table_height/z00 results/table_height/p05"}
PY_RT=~/miniconda3/envs/robotwin/bin/python
PY_LB=~/miniconda3/envs/lingbot/bin/python

for r in $ROOTS; do
    echo "[$(date)] analyze $r"
    $PY_RT "$ROOT/analysis/analyze.py" "$ROOT/$r" || echo "analyze $r failed"
done
for r in $ROOTS; do
    echo "[$(date)] attention replay $r"
    eps=$(ls -d "$ROOT/$r"/episodes/{success,fail}/*/* 2>/dev/null)
    (cd "$ROOT/lingbot-va" && PYTORCH_ALLOC_CONF=expandable_segments:True $PY_LB ../analysis/attention_replay.py $eps \
        --out "$ROOT/$r/analysis/attention" 2>&1 | grep --line-buffered -v "it/s\|s/it\|FRAME START\|INFO")
    echo "[$(date)] attention render $r"
    $PY_RT "$ROOT/analysis/attention_render.py" "$ROOT/$r" || echo "render $r failed"
done
echo "[$(date)] DONE"
