"""Checks for the hid data (plan 2026-11-18, checklist 2-3):
  * executed = commanded + o: recorded sim EE positions vs the commanded (believed) EE actions, median over the
    steps where the command has been still for a while (the arm has settled), against the drawn offset
  * policy failure rate under o, success rate of the compensated expert fix (goal >= 90 %)

  python -m fawam.hid_check results/sweeps/p10_hid_smoke/base
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


def settled_offset(f, settle=8):
    """median(sim EE position - commanded position) over steps whose command was unchanged for `settle` steps."""
    cmd = f["actions/executed_ee_action"][()]
    step = f["actions/executed_env_step"][()]
    ok = f["actions/executed_executed"][()]
    sim_step = f["sim/env_step"][()]
    out = []
    for arm, (sl, key) in {"left": (slice(0, 3), "sim/left_ee_pose"), "right": (slice(8, 11), "sim/right_ee_pose")}.items():
        pos = f[key][()][:, :3]
        d = []
        for i in range(settle, len(cmd)):
            if not ok[i] or np.abs(cmd[i - settle:i + 1, sl] - cmd[i, sl]).max() > 1e-4:
                continue
            j = min(int(np.searchsorted(sim_step, step[i])), len(pos) - 1)
            d.append(pos[j] - cmd[i, sl])
        out.append(np.median(d, axis=0) if d else np.full(3, np.nan))
    return out


def main(root):
    rows = defaultdict(lambda: dict(policy=[], fix=[], err=[], n_still=0))
    for h5p in sorted(Path(root).glob("*/episodes/*/*/*/episode.h5")):
        with h5py.File(h5p, "r") as f:
            a = dict(f.attrs)
            if "success" not in a:  # still being written
                continue
            hid = json.loads(a.get("hid", "null") or "null")
            cfg = a.get("task_config")
            r = rows[cfg]
            succ = str(a["success"]) == "True"
            if a["driver"] == "expert_replay":
                r["fix"].append(succ)
            else:
                r["policy"].append(succ)
            if hid and hid["type"] == "trans" and a["driver"] == "policy":
                for d in settled_offset(f):
                    if not np.isnan(d).any():
                        r["err"].append((np.asarray(hid["o"]), d))
    for cfg, r in sorted(rows.items()):
        p, x = r["policy"], r["fix"]
        print(f"{cfg}: policy success {sum(p)}/{len(p)}, compensated expert fix {sum(x)}/{len(x)}"
              f" ({100 * sum(x) / max(1, len(x)):.0f} %)")
        if r["err"]:
            o = np.array([e[0] for e in r["err"]])
            d = np.array([e[1] for e in r["err"]])
            res = d - o
            print(f"  executed - commanded vs o (n={len(o)} arm-episodes): corr z {np.corrcoef(o[:, 2], d[:, 2])[0, 1]:.2f}, "
                  f"x {np.corrcoef(o[:, 0], d[:, 0])[0, 1]:.2f}, y {np.corrcoef(o[:, 1], d[:, 1])[0, 1]:.2f}; "
                  f"median |residual| {np.median(np.abs(res), axis=0).round(4)} m")


if __name__ == "__main__":
    main(sys.argv[1])
