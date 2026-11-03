#!/bin/bash
# Full pipeline: (re)start the server, record eval episodes, stop the server, build the analysis (incl. attention).
# Start detached so it survives closing the editor/terminal:
#   setsid nohup scripts/run_all.sh > logs/run_all.log 2>&1 < /dev/null &
# Env overrides: TASKS="a b", NUM=10, CONFIG=demo_clean, SAVE_ROOT=results/eval
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TASKS=${TASKS:-"handover_block pick_diverse_bottles"}
NUM=${NUM:-10}
CONFIG=${CONFIG:-demo_clean}
SAVE_ROOT=${SAVE_ROOT:-$ROOT/results/eval}

echo "[$(date)] start: tasks=$TASKS num=$NUM config=$CONFIG save_root=$SAVE_ROOT"
"$ROOT/scripts/stop_server.sh"
sleep 15
"$ROOT/scripts/start_server.sh" || { echo "server failed to start"; exit 1; }
for task in $TASKS; do
    echo "[$(date)] eval $task"
    ANALYZE=0 "$ROOT/scripts/run_eval.sh" "$task" "$NUM" "$CONFIG" "$SAVE_ROOT" > "$ROOT/logs/eval_$task.log" 2>&1 \
        || echo "eval $task exited with an error, see logs/eval_$task.log"
    grep -a "Success rate" "$ROOT/logs/eval_$task.log" | tail -1
done
"$ROOT/scripts/stop_server.sh"
echo "[$(date)] analysis"
"$ROOT/scripts/analyze_results.sh" "$SAVE_ROOT"
echo "[$(date)] DONE"
