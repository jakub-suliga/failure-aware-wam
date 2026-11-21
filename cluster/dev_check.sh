#!/bin/bash
# Plan v2 code check on one GPU (dev-gpu-h100, <= 1 h): every new code path once, short. Lines "CHECK <name> ok|FAIL".
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
set -uo pipefail
source "${FAWAM_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}/cluster/env.sh"
cd "$FAWAM_ROOT"
OUT=$FAWAM_ROOT/results/_devcheck/${SLURM_JOB_ID:-local}; mkdir -p "$OUT"
chk() { local n=$1; shift; if "$@" > "$OUT/$n.log" 2>&1; then echo "CHECK $n ok"; else echo "CHECK $n FAIL (rc $?)"; tail -15 "$OUT/$n.log"; fi; }
SM=results/_smoke/p21_snap/base; ROOTS="$SM/icl_sb $SM/hid_y_p5 results/_smoke/p20_d3/base/hid_y_p5"
# 1 compile + imports
chk compile "$PY_RT" -m py_compile fawam/*.py cluster/pipeline.py cluster/pipeline_v2.py lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py
chk import_rt "$PY_RT" -c "import sys; sys.path[:0]=['.','cluster']; import fawam.gates_v2, fawam.icl_snap, fawam.phases, fawam.vlm_phase, fawam.methods, pipeline, pipeline_v2; pipeline.stages(pipeline.squeue())"
chk import_lb "$PY_LB" -c "import sys; sys.path.insert(0,'.'); import fawam.server, fawam.train_adapter, fawam.learn, fawam.d1_condition, fawam.adapters"
chk gates "$PY_RT" -m fawam.gates_v2 summary
# 2 dataset, all contexts / modes / gates (CPU)
chk dataset "$PY_LB" - <<PY
import sys; sys.path.insert(0,'.')
from fawam.paths import add_lingbot_to_path; add_lingbot_to_path()
from configs import VA_CONFIGS
from fawam.latent_data import ActionNorm
from fawam import icl_snap
import torch
pairs = icl_snap.collect_pairs("$ROOTS".split()); assert pairs, "no pairs"
class TC:
    def get(self, t): return torch.zeros(512, 4096)
norm = ActionNorm(VA_CONFIGS["robotwin_train"])
for C in ["C0","C1","C2","C3","C4","C5","C6","C7","C8"]:
    for cm, g in (("prepend", None), ("separate", None), ("both", None), ("prepend", "t1"), ("prepend", "t5")):
        for samp in ("uniform", "win50", "w4"):
            ds = icl_snap.SnapPairDataset(pairs, norm, TC(), ctx=C, sampling=samp, ctx_mode=cm, gate=g)
            for mode in ("own", "foreign", "none"):
                ds.mode = mode; s = ds[0]
                F = s["latents"].shape[1]
                assert s["actions"].shape[1] == F and len(s["weight_video"]) == F
                if g: assert len(s["gate_err"]) == F, (len(s["gate_err"]), F)
print("dataset ok", len(pairs))
PY
# 3 trainings (4 steps, validation, checkpoint) and resume
tr() { local A=$1 C=$2 N=p21smoke_$1_$2; rm -rf artifacts/adapters/$N
  NAME=$N SETTING=smoke CTX=$C ARCH=$A STEPS=4 ROOTS="$ROOTS" EXTRA="--val_every 2 --save_every 2 --val_frac 0.4" bash cluster/p21_train_job.sh &&
  test -f artifacts/adapters/$N/adapter.pt && grep -q val_video_own artifacts/adapters/$N/train_inside.jsonl; }
chk train_A3 tr A3 C2
chk train_T4 tr T4 C2
# 4 rewind retry in the simulator with the A3 / T4 adapters (background, 1 episode each, retry always runs: harm)
ev() { TASK=move_playingcard_away CONFIG=icl_sb METHOD=$1 NUM=1 SEED0=0 N_PAR=1 OUT=$OUT/eval_$1 BASE_PORT=$2 OFFLOAD=1 \
       bash cluster/eval_job.sh && ls $OUT/eval_$1/episodes/*/move_playingcard_away/*_attempt2/episode.h5; }
( chk eval_A3 ev icl2_smoke_A3 23100 ) > $OUT/ev_A3.res &
( chk eval_T4 ev icl2_smoke_T4 23200 ) > $OUT/ev_T4.res &
for v in "A1 C2" "A2 C2" "T1 C2" "T2 C2" "T5 C2" "A1 C0" "A1 C4" "A1 C7"; do set -- $v; chk train_$1_$2 tr $1 $2; done
chk resume bash -c "NAME=p21smoke_A1_C2 SETTING=smoke CTX=C2 ARCH=A1 STEPS=6 ROOTS='$ROOTS' EXTRA='--val_every 2 --save_every 2 --val_frac 0.4' bash cluster/p21_train_job.sh | grep -q 'resuming'"
# 5 TTT learner: video-only Adapter+, LoRA video-only, P5 hinge
S=$(ls -d results/sweeps/p20_d2_base/base/hid_y_m3/episodes/success/move_playingcard_away/* | head -1)
F=results/_smoke/p22_ttt/ttt_p4/hid_y_p5/episodes/fail/move_playingcard_away/ep0000_seed10000
chk learn_hinge "$PY_LB" -m fawam.learn --episodes $S --out $OUT/hinge --steps 4 --no_relabel --gap_beta 0 --lambda_action 0 --kind inside --inside_streams video --lr 1e-4 --hinge_episode $F --hinge_beta 1.0
chk learn_lora "$PY_LB" -m fawam.learn --episodes $F --out $OUT/lora --steps 4 --no_relabel --gap_beta 0 --lambda_action 0 --kind lora --inside_streams video --lr 1e-4
wait
cat $OUT/ev_A3.res $OUT/ev_T4.res
# 6 D1 on one episode, one chunk
chk d1 "$PY_LB" -m fawam.d1_condition --episodes results/sweeps/p0_baseline_table/base/table_h_m05 --limit 1 --chunks 1 --out $OUT/d1
rm -f results/gates/p20_d1.json  # the check must not open the real D1 gate
echo "[$(date)] dev check done"
