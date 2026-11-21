#!/bin/bash
# Plan v2 code check, part 2 (after dev_check.sh): validation path of every p21 architecture, rewind retry in the
# simulator with the A3 / T4 / T5 smoke adapters, D1. Lines "CHECK <name> ok|FAIL".
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
set -uo pipefail
source "${FAWAM_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}/cluster/env.sh"
cd "$FAWAM_ROOT"
OUT=$FAWAM_ROOT/results/_devcheck/${SLURM_JOB_ID:-local}; mkdir -p "$OUT"
chk() { local n=$1; shift; if "$@" > "$OUT/$n.log" 2>&1; then echo "CHECK $n ok"; else echo "CHECK $n FAIL (rc $?)"; tail -15 "$OUT/$n.log"; fi; }
ev() { TASK=move_playingcard_away CONFIG=icl_sb METHOD=$1 NUM=1 SEED0=0 N_PAR=1 OUT=$OUT/eval_$1 BASE_PORT=$2 OFFLOAD=1 \
       timeout 2400 bash cluster/eval_job.sh && ls $OUT/eval_$1/episodes/*/move_playingcard_away/*_attempt2/episode.h5; }
( chk eval_A3 ev icl2_smoke_A3 23100 ) > $OUT/ev_A3.res &
( chk eval_T4 ev icl2_smoke_T4 23200 ) > $OUT/ev_T4.res &
( chk eval_T5 ev icl2_smoke_T5 23300 ) > $OUT/ev_T5.res &
chk validate "$PY_LB" - <<'PY'
import sys, argparse; sys.path.insert(0, ".")
import torch
from fawam import train_adapter as T, adapters as fa, icl_snap
from fawam.latent_data import ActionNorm
from fawam.text_cache import TextCache
cfg = T.VA_CONFIGS["robotwin_train"]
ps = icl_snap.collect_pairs(["results/_smoke/p21_snap/base/icl_sb", "results/_smoke/p21_snap/base/hid_y_p5",
                             "results/_smoke/p20_d3/base/hid_y_p5"])
tf = T.load_frozen_transformer(str(T.LINGBOT / "checkpoints" / "lingbot-va-posttrain-robotwin"), torch.device("cuda"))
sv = T.FlowMatchScheduler(shift=cfg.snr_shift, sigma_min=0.0, extra_one_step=True); sv.set_timesteps(1000, training=True)
sa = T.FlowMatchScheduler(shift=cfg.action_snr_shift, sigma_min=0.0, extra_one_step=True); sa.set_timesteps(1000, training=True)
a = argparse.Namespace(icl_gap=1)
for arch, mode, gate in (("A1", "prepend", None), ("A2", "prepend", None), ("A3", "separate", None),
                         ("T2", "prepend", "t2"), ("T4", "both", None), ("T5", "prepend", "t5"), ("T1", "prepend", "t1")):
    aset = fa.AdapterSet.load(f"artifacts/adapters/p21smoke_{arch}_C2", device="cuda")
    fa.attach(tf, aset)
    ds = icl_snap.SnapPairDataset(ps, ActionNorm(cfg), TextCache(), ctx="C2", ctx_mode=mode, gate=gate)
    ds.p_irr, ds.cfg_prob = 0.0, 0.0
    r = T.validate_snap(tf, ds, a, sv, sa, tuple(cfg.patch_size), torch.device("cuda"), n=2)
    print(arch, r, flush=True)
    assert all(v == v for v in r.values()), r
PY
wait
cat $OUT/ev_*.res
for m in A3 T4 T5; do grep -ah "ICL rewind\|context for\|adapter_gate" $OUT/eval_icl2_smoke_$m/logs/*/client_0.log $OUT/eval_icl2_smoke_$m/logs/*/server_0.log 2>/dev/null | cut -c1-200 | tail -2; done
chk d1 "$PY_LB" -m fawam.d1_condition --episodes results/sweeps/p0_baseline_table/base/table_h_m05 --limit 1 --chunks 1 --out $OUT/d1
rm -f results/gates/p20_d1.json
echo "[$(date)] dev check 2 done"
