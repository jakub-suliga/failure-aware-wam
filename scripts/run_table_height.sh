#!/bin/bash
# Table-height sweep: same tasks and seeds at fixed table offsets (RoboTwin/task_config/table_h_*.yml, demo_clean
# otherwise). Table and objects move together, robot and cameras stay fixed.
# Start detached:  setsid nohup scripts/run_table_height.sh > logs/run_table_height.log 2>&1 < /dev/null &
# Env overrides: TASKS="a b", HEIGHTS="z00 m10", NUM=8, SAVE_BASE=results/table_height, LOG_DIR=logs
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TASKS=${TASKS:-"move_playingcard_away place_mouse_pad pick_diverse_bottles"}
HEIGHTS=${HEIGHTS:-"z00 m10 p05 m05"}
NUM=${NUM:-8}
SAVE_BASE=${SAVE_BASE:-$ROOT/results/table_height}
LOG_DIR=${LOG_DIR:-$ROOT/logs}
mkdir -p "$LOG_DIR"

echo "[$(date)] start: tasks=$TASKS heights=$HEIGHTS num=$NUM save_base=$SAVE_BASE"
"$ROOT/scripts/stop_server.sh"
sleep 15
"$ROOT/scripts/start_server.sh" || { echo "server failed to start"; exit 1; }
for h in $HEIGHTS; do
    for task in $TASKS; do
        echo "[$(date)] eval $task table_h_$h"
        log="$LOG_DIR/eval_table_h_${h}_$task.log"
        ANALYZE=0 "$ROOT/scripts/run_eval.sh" "$task" "$NUM" "table_h_$h" "$SAVE_BASE/$h" > "$log" 2>&1 \
            || echo "eval $task table_h_$h exited with an error, see $log"
        grep -a "Success rate" "$log" | tail -1
    done
done
"$ROOT/scripts/stop_server.sh"
echo "[$(date)] analysis"
"$ROOT/scripts/analyze_results.sh" $(for h in $HEIGHTS; do echo "$SAVE_BASE/$h"; done)
echo "[$(date)] DONE"
