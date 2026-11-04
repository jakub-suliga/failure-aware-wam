#!/bin/bash
# D3-online: the paper's protocol on the WAM. Three adaptation episodes on the shifted task; after each one the adapters
# are updated from ALL episodes so far (self-supervised: the policy's own rollouts, video + hindsight action loss);
# then the frozen adapter is evaluated on EVAL_NUM fresh seeds. Server, simulator and training share one GPU
# (~17 + 5 + 20 GB on an H100-94GB). Adapter versions are separate dirs, so the server reloads them.
# Env: TASK CONFIG OUT [ROUNDS=3 STEPS=150 EVAL_NUM=20 KIND=dual LAMBDA_V=1 LAMBDA_A=1 SOURCE=self|expert]
#   SOURCE=expert: the adaptation episodes are expert replays instead of own rollouts (few-shot demos, online).
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
set -uo pipefail
source "${FAWAM_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}/cluster/env.sh"
: "${TASK:?}" "${CONFIG:?}" "${OUT:?}"
ROUNDS=${ROUNDS:-3}; STEPS=${STEPS:-150}; EVAL_NUM=${EVAL_NUM:-20}; KIND=${KIND:-dual}; SOURCE=${SOURCE:-self}
MODEL=${MODEL_PATH:-$FAWAM_ROOT/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin}
mkdir -p "$OUT/logs" "$OUT/methods"
PORT=$(( 21000 + (${SLURM_JOB_ID:-$$} % 400) * 10 ))
cd "$FAWAM_ROOT/lingbot-va" || exit 1
"$PY_LB" -m torch.distributed.run --nproc_per_node 1 --master_port $((PORT + 1)) -m fawam.server --config-name robotwin \
    --port "$PORT" --save_root "$OUT/logs/vis" --model_path "$MODEL" --enable_offload > "$OUT/logs/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
until grep -q "server listening\|Traceback" "$OUT/logs/server.log"; do sleep 10; done
export ROBOTWIN_ROOT="$FAWAM_ROOT/RoboTwin" PYTHONWARNINGS=ignore::UserWarning

client() {  # $1 save_root  $2 method  $3 num  $4 seed index  $5 driver
    "$PY_RT" -m evaluation.robotwin.eval_polict_client_openpi --config policy/ACT/deploy_policy.yml --overrides \
        --task_name "$TASK" --task_config "$CONFIG" --train_config_name 0 --model_name 0 --ckpt_setting 0 \
        --seed "$4" --policy_name ACT --method "$2" --driver "$5" --save_root "$1" --video_guidance_scale 5 \
        --action_guidance_scale 1 --test_num "$3" --port "$PORT" >> "$OUT/logs/client.log" 2>&1
}

method=base
for r in $(seq 1 "$ROUNDS"); do
    driver=policy; [ "$SOURCE" = expert ] && driver=expert_replay
    client "$OUT/adapt" "$method" 1 $((100 + r)) "$driver"
    A=$OUT/adapter_r$r
    INIT=(); [ "$r" -gt 1 ] && INIT=(--init "$OUT/adapter_r$((r - 1))")
    (cd "$FAWAM_ROOT" && "$PY_LB" -m fawam.text_cache build "$OUT/adapt" --model_path "$MODEL" &&
     "$PY_LB" -m fawam.train_adapter inside --episodes "$OUT/adapt" --kind "$KIND" --steps "$STEPS" "${INIT[@]}" \
        --lambda_video "${LAMBDA_V:-1}" --lambda_action "${LAMBDA_A:-1}" --model_path "$MODEL" --out "$A" &&
     if [ "$KIND" = dual ]; then "$PY_LB" -m fawam.train_adapter outside --init "$A" --episodes "$OUT/adapt" \
        --steps_outside 100 --out "$A"; fi) >> "$OUT/logs/train.log" 2>&1 || { echo "training round $r failed"; exit 1; }
    printf '{"name": "online_r%s", "adapter_path": "%s"}\n' "$r" "$A" > "$OUT/methods/r$r.json"
    method=$OUT/methods/r$r.json
    echo "[$(date)] round $r done"
done
client "$OUT/eval" "$method" "$EVAL_NUM" 7 policy
grep -a "Success rate" "$OUT/logs/client.log" | tail -1
