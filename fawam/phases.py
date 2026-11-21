"""Oracle task phases from simulator events (plan v2 2026-11-21, D6). Analysis / training labels only, never a
test-time input of a method (except the explicitly oracle diagnostics T6 / C6).

Phases per recorded frame (frames/env_step -> sim record of that step):
  approach  before the gripper closes, end effector more than 5 cm above the object
  lower     before the close, end effector < 5 cm above its height at the close and not rising (flange pose)
  close     the gripper command goes closed (first drop below 0.8) .. 12 sim steps later
  lift      closed, object z rising (> 1 cm above its start height) or end effector rising
  transport closed, object lifted or moved > 1 cm horizontally since the close, end effector not rising
  release   from the moment the gripper opens again after the close
The active arm is the one whose gripper closes first. Without any close the episode is approach/lower only.

  python -m fawam.phases <episode roots ...> [--out results/p20_d6_phases.json]   # labels for every episode
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

PHASES = ["approach", "lower", "close", "lift", "transport", "release"]
THR = 0.8


def phases_of(h5):
    """-> dict(frames=[phase per frame], close_step, open_step, arm, close_frame, close_latent)."""
    f = h5 if isinstance(h5, h5py.File) else h5py.File(h5, "r")
    try:
        st = f["sim/env_step"][()]
        gl, gr = f["sim/left_gripper"][()], f["sim/right_gripper"][()]
        obj = f["sim/object_pose"][()][:, 0, :3] if "sim/object_pose" in f and f["sim/object_pose"].shape[1] else None
        el, er = f["sim/left_ee_pose"][()][:, :3], f["sim/right_ee_pose"][()][:, :3]
        fst = f["frames/env_step"][()]
        fli = f["frames/latent_index"][()]
    finally:
        if not isinstance(h5, h5py.File):
            f.close()
    cl = np.where((gl[:-1] >= THR) & (gl[1:] < THR))[0]
    cr = np.where((gr[:-1] >= THR) & (gr[1:] < THR))[0]
    first = [(c[0] + 1, a) for c, a in ((cl, "left"), (cr, "right")) if len(c)]
    if first:
        i_close, arm = min(first)
    else:
        i_close, arm = None, ("left" if obj is not None and obj[0, 0] < 0 else "right")
    g, ee = (gl, el) if arm == "left" else (gr, er)
    i_open = None
    if i_close is not None:
        op = np.where((g[i_close:-1] < THR) & (g[i_close + 1:] >= THR))[0]
        i_open = i_close + 1 + int(op[0]) if len(op) else None
    z0 = obj[0, 2] if obj is not None else None
    lab = []
    for i in range(len(st)):
        if i_close is None or i < i_close:
            # "< 5 cm above the object": the EE pose is the flange, so measured against the height at the close
            ref_z = ee[i_close, 2] if i_close is not None else (obj[i, 2] if obj is not None else -1e9)
            near = ee[i, 2] < ref_z + 0.05 and (i < 2 or ee[i, 2] <= ee[max(0, i - 5), 2] + 1e-3)
            lab.append("lower" if near else "approach")
        elif i_open is not None and i >= i_open:
            lab.append("release")
        elif i < i_close + 12:
            lab.append("close")
        else:
            lifted = obj is not None and obj[i, 2] - z0 > 0.01
            rising = i >= 3 and ee[i, 2] - ee[i - 3, 2] > 0.002
            moved = obj is not None and np.linalg.norm(obj[i, :2] - obj[i_close, :2]) > 0.01
            if rising or (lifted and obj[i, 2] - obj[max(0, i - 3), 2] > 0.001):
                lab.append("lift")
            elif lifted or moved:
                lab.append("transport")
            else:
                lab.append("close")
    idx = np.clip(np.searchsorted(st, fst), 0, len(st) - 1)
    frames = [lab[i] for i in idx]
    close_frame = int(np.searchsorted(fst, st[i_close])) if i_close is not None else None
    return dict(frames=frames, latent_index=[int(x) for x in fli], arm=arm,
                close_step=None if i_close is None else int(st[i_close]),
                open_step=None if i_open is None else int(st[i_open]),
                close_frame=close_frame,
                close_latent=None if close_frame is None else int(fli[min(close_frame, len(fli) - 1)]))


def latent_phases(ph):
    """Phase per latent frame (majority of its raw frames)."""
    out = {}
    for p, l in zip(ph["frames"], ph["latent_index"]):
        out.setdefault(int(l), []).append(p)
    return {l: max(set(v), key=v.count) for l, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default="results/p20_d6_phases.json")
    a = ap.parse_args()
    res, n_close = {}, 0
    for root in a.roots:
        for h5 in sorted(Path(root).glob("**/episodes/*/*/*/episode.h5")):
            try:
                ph = phases_of(h5)
            except Exception as e:  # incomplete episode
                res[str(h5.parent)] = dict(error=repr(e)[:200])
                continue
            n_close += ph["close_step"] is not None
            res[str(h5.parent)] = ph
            (h5.parent / "phases.json").write_text(json.dumps(ph))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(n=len(res), n_with_close=n_close,
                                           counts={p: sum(v.get("frames", []).count(p) for v in res.values())
                                                   for p in PHASES}), indent=1))
    print(f"{len(res)} episodes, {n_close} with a gripper close -> {a.out}")


if __name__ == "__main__":
    main()
