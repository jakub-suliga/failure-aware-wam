#!/bin/bash
# One dev-partition job with several GPUs that works through pipeline/dev_queue.jsonl (cluster/dev_queue.py), one
# worker per GPU. Submit:  sbatch --partition=dev-gpu-h100 --time=01:00:00 --gres=gpu:4 cluster/dev_runner.sh  (B200 dev nodes: from Dec 2026)
# Stop early with scancel: tasks that were running go back to the normal queue (their held SLURM jobs are released).
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
set -uo pipefail
source "${FAWAM_ROOT:?}/cluster/env.sh"
cd "$FAWAM_ROOT"
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
LIMIT=$(squeue -h -j "$SLURM_JOB_ID" -o %l | awk -F'[-:]' '{n=NF; s=$n+60*$(n-1)+3600*$(n-2); if (n>3) s+=86400*$(n-3); print s}')
END=$(( $(date +%s) + ${LIMIT:-3600} - 600 ))   # stop taking tasks that would not finish 10 min before the limit
echo "[$(date)] dev runner $SLURM_JOB_ID on $(hostname): $NGPU GPU(s), until $(date -d @$END)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pids=()
trap 'kill -TERM "${pids[@]}" 2>/dev/null; wait' TERM INT
for g in $(seq 0 $((NGPU - 1))); do
    "$PY_RT" cluster/dev_queue.py work --gpu "$g" --end "$END" & pids+=($!)
    sleep 20   # stagger the start-up
done
wait
"$PY_RT" cluster/dev_queue.py status
# CHAIN=<n>: submit the next run while tasks are left (dev-gpu-h100 allows 1 h per job); stop the chain with
# `scancel -n fawam_dev_runner` or by deleting pipeline/dev_queue.jsonl
if [ "${CHAIN:-0}" -gt 0 ] && grep -q '"state": "todo"' pipeline/dev_queue.jsonl 2>/dev/null; then
    env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --job-name=fawam_dev_runner --partition="$SLURM_JOB_PARTITION" \
        --gres="gpu:$NGPU" --time=01:00:00 --account="$SLURM_JOB_ACCOUNT" --output="$FAWAM_ROOT/logs/dev_runner_%j.out" \
        --chdir="$FAWAM_ROOT" --export=ALL,FAWAM_ROOT="$FAWAM_ROOT",CHAIN=$((CHAIN - 1)) cluster/dev_runner.sh
fi
echo "[$(date)] dev runner done"
