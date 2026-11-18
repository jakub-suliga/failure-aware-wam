#!/bin/bash
# CPU preparation on the P0 baseline (docs/METHOD.md, section 7, step 2): grasp reference from the z00 successes,
# oracle analysis.json next to every P0 failure, imagination pose probe fitted on P0. Previous artifacts are kept as
# *.bak. Run via sbatch on a CPU partition from the repo root.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
set -euo pipefail
source "${FAWAM_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}/cluster/env.sh"
cd "$FAWAM_ROOT"
P0=results/sweeps/p0_baseline_table/base
for f in artifacts/grasp_ref.json artifacts/probe/pose_probe.npz; do [ -f "$f.bak" ] || cp "$f" "$f.bak"; done
"$PY_RT" -m fawam.failure_analysis --grasp_ref $P0/table_h_z00 -- $P0/table_h_m05 $P0/table_h_m10 | tail -n 5
"$PY_RT" -m fawam.pose_probe fit $P0/table_h_z00 $P0/table_h_m05 $P0/table_h_m10
n=$(find $P0 -path '*/fail/*' -name analysis.json | wc -l)
echo "analysis.json written for $n P0 failures"
[ "$n" -gt 0 ]
