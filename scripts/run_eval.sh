#!/bin/bash
# Run RoboTwin episodes against a running server and record everything.
# Usage: run_eval.sh <task_name> <num_episodes> [task_config=demo_clean] [save_root=results/eval] [port=29056]
# METHOD=<fawam method spec, e.g. ig2 or ig2+reset_on_gap> (default base), DRIVER=policy|expert_replay,
# SEED=<k> (episodes start at seed 10000*(1+k); different k = disjoint seeds for parallel jobs)
# Episodes land in <save_root>/episodes/{success,fail}/<task>/epXXXX_seedY/
# Afterwards scripts/analyze_results.sh <save_root> is started in the background (log: logs/analysis.log); its
# attention replay waits until the server is stopped. ANALYZE=0 skips this (pipelines analyze once at the end).
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TASK=$1; NUM=$2; CONFIG=${3:-demo_clean}; SAVE_ROOT=$(realpath -m "${4:-$ROOT/results/eval}"); PORT=${5:-29056}
cd "$ROOT/lingbot-va" || exit 1
export ROBOTWIN_ROOT="$ROOT/RoboTwin" LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH PYTHONWARNINGS=ignore::UserWarning
${PY_RT:-~/miniconda3/envs/robotwin/bin/python} -m evaluation.robotwin.eval_polict_client_openpi \
    --config policy/ACT/deploy_policy.yml --overrides --task_name "$TASK" --task_config "$CONFIG" \
    --train_config_name 0 --model_name 0 --ckpt_setting 0 --seed "${SEED:-0}" --policy_name ACT \
    --method "${METHOD:-base}" --driver "${DRIVER:-policy}" \
    --save_root "$SAVE_ROOT" --video_guidance_scale 5 --action_guidance_scale 1 --test_num "$NUM" --port "$PORT"
rc=$?
if [ "${ANALYZE:-1}" != 0 ]; then
    setsid nohup "$ROOT/scripts/analyze_results.sh" "$SAVE_ROOT" >> "$ROOT/logs/analysis.log" 2>&1 < /dev/null &
    echo "analysis started in the background, log: $ROOT/logs/analysis.log"
fi
exit $rc
