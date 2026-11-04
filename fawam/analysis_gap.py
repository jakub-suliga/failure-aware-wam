"""Imagination-execution gap at the first grasp (docs/FINDINGS.md, section 5) and method comparison tables.

  python -m fawam.analysis_gap gap results/table_height results/table_height_n10      -> gap table per setting
  python -m fawam.analysis_gap compare results/sweeps/<sweep>                         -> success per method x setting

`gap`: for every episode, the first time a gripper is commanded closed; at that latent the cosine distance between the
predicted and the observed latent of that arm's wrist camera, plus the commanded vs. object height.
`compare`: success rates with Wilson 95% intervals per (method, task, setting) from episode.h5 attributes, written to
<sweep>/summary.csv and printed as a pivot.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from fawam.latent_data import find_episodes


def first_close_gap(f):
    raw = f["actions/executed_raw_action"][()]
    li = f["actions/executed_latent_index"][()]
    closes = []
    for arm, ch in ((0, 7), (1, 15)):
        hit = np.where(raw[:, ch] < 0.5)[0]
        closes.append(hit[0] if len(hit) else None)
    valid = [(k, arm) for arm, k in enumerate(closes) if k is not None]
    if not valid:
        return None
    k, arm = min(valid)
    L = int(li[k])
    cam = ["cam_left_wrist", "cam_right_wrist"][arm]
    pi = {int(x): i for i, x in enumerate(f["model/pred/latent_index"][()])}
    oi = {int(x): i for i, x in enumerate(f["model/obs/latent_index"][()])}
    if L not in pi or L not in oi:
        return None
    p = f[f"model/pred/latents/{cam}"][pi[L]].astype(np.float32).ravel()
    o = f[f"model/obs/latents/{cam}"][oi[L]].astype(np.float32).ravel()
    row = dict(close_step=int(f["actions/executed_env_step"][k]), close_latent=L, arm=arm,
               cos_at_grasp=float(1 - p @ o / (np.linalg.norm(p) * np.linalg.norm(o) + 1e-8)))
    if "sim/left_ee_pose" in f and "sim/object_pose" in f:
        env_step = f["sim/env_step"][()]
        j = int(np.searchsorted(env_step, row["close_step"]))
        j = min(j, len(env_step) - 1)
        ee = f[f"sim/{['left', 'right'][arm]}_ee_pose"][j][:3]
        objs = f["sim/object_pose"][j][:, :3]
        o_near = objs[np.argmin(np.linalg.norm(objs[:, :2] - ee[:2], axis=1))]
        row.update(ee_z=float(ee[2]), obj_z=float(o_near[2]), dz=float(ee[2] - o_near[2]))
    return row


def gap(a):
    rows = []
    for e in find_episodes(a.roots):
        with h5py.File(e / "episode.h5", "r") as f:
            r = first_close_gap(f)
            if r is None:
                continue
            r.update(task=str(f.attrs["task_name"]), setting=str(f.attrs["task_config"]),
                     success=bool(f.attrs["success"]), method=str(f.attrs.get("fawam_method", "base")), path=str(e))
            rows.append(r)
    df = pd.DataFrame(rows)
    if a.out:
        df.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200):
        print(df.groupby(["setting", "success"])[["cos_at_grasp", "dz"]].agg(["mean", "std", "count"]).round(3))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def compare(a):
    rows = []
    for e in find_episodes([a.sweep]):
        with h5py.File(e / "episode.h5", "r") as f:
            at = f.attrs
            rows.append(dict(method=str(at.get("fawam_method", "base")), task=str(at["task_name"]),
                             setting=str(at["task_config"]), success=bool(at["success"]),
                             attempt=int(at.get("attempt", 1)), seed=int(at["seed"]),
                             interventions=len(json.loads(at.get("interventions", "[]"))),
                             phase=(json.loads(at.get("lff", "{}")) or {}).get("phase"),
                             episode=int(at.get("episode", -1)),
                             env_steps=int(at["env_steps"])))
    df = pd.DataFrame(rows)
    if df.empty:
        print("no episodes")
        return
    # paper protocol (adapt_episodes): only the episodes after the adaptation phase count; for methods without
    # phases, the same episode indices are dropped so the comparison stays on identical seeds
    n_adapt = int(a.adapt_episodes or 0)
    if n_adapt and (df.phase == "adapt").any():
        print(f"evaluation phase only: dropping the first {n_adapt} episodes of every client run (adaptation)")
    first = df[(df.attempt == 1) & ((df.phase == "eval") | (df.phase.isna() & (df.episode >= n_adapt)) |
                                    ((df.phase != "adapt") & (n_adapt == 0)))]
    g = first.groupby(["method", "task", "setting"]).success.agg(["sum", "count"]).reset_index()
    g["rate"] = g["sum"] / g["count"]
    g[["lo", "hi"]] = [wilson(k, n) for k, n in zip(g["sum"], g["count"])]
    retry = df.groupby(["method", "task", "setting", "seed"]).success.max().groupby(
        ["method", "task", "setting"]).mean().rename("rate_with_retry")
    g = g.join(retry, on=["method", "task", "setting"])
    out = Path(a.sweep) / "summary.csv"
    g.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(g.pivot_table(index="method", columns=["task", "setting"], values="rate").round(2))
    print(f"wrote {out}")


def memory(a):
    """Does the model use the in-context memory? Per method: attention ratio on memory tokens (mass / token share;
    1 = like any other token, < 1 = ignored), per-layer profile, influence (relative change of the prediction when the
    memory is hidden), optional full counterfactual (mean |delta| of the arm positions in mm), and whether these
    differ between successful and failed attempts."""
    rows, layers = [], {}
    for e in find_episodes([a.sweep]):
        with h5py.File(e / "episode.h5", "r") as f:
            if "fawam/memory" not in f:
                continue
            g = f["fawam/memory"]
            at = f.attrs
            meth = str(at.get("fawam_method", "base"))
            aa, sa = g["attn_action"][()], g["share_action"][()]
            r = dict(method=meth, task=str(at["task_name"]), success=bool(at["success"]),
                     attn_ratio_action=float(np.nanmean(np.nanmean(aa, 1) / np.maximum(sa, 1e-8))),
                     infl_action=float(np.nanmean(g["infl_action"][()])), infl_video=float(np.nanmean(g["infl_video"][()])),
                     memory_tokens=float(np.nanmean(g["memory_tokens"][()])))
            if "attn_video" in g:
                r["attn_ratio_video"] = float(np.nanmean(np.nanmean(g["attn_video"][()], 1) /
                                                         np.maximum(g["share_video"][()], 1e-8)))
            if "cf_dxyz_left" in g:
                r["cf_dxyz_mm"] = float(np.nanmean(np.maximum(g["cf_dxyz_left"][()], g["cf_dxyz_right"][()]))) * 1000
            if "ttt_update" in f["fawam"].attrs:
                r["ttt_update"] = float(np.mean(list(json.loads(f["fawam"].attrs["ttt_update"]).values())))
            rows.append(r)
            layers.setdefault(meth, []).append(np.nanmean(aa / np.maximum(sa, 1e-8)[:, None], 0))
    df = pd.DataFrame(rows)
    if df.empty:
        print("no episodes with a memory")
        return
    cols = [c for c in ("attn_ratio_action", "attn_ratio_video", "infl_action", "infl_video", "cf_dxyz_mm",
                        "ttt_update", "memory_tokens") if c in df]
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(df.groupby("method")[cols].mean().round(3))
        print("\nsuccess vs. failure (does using the memory go with success?)")
        print(df.groupby(["method", "success"])[["attn_ratio_action", "infl_action"]].mean().unstack().round(3))
    out = Path(a.sweep)
    df.to_csv(out / "memory_usage.csv", index=False)
    prof = pd.DataFrame({m: np.nanmean(np.stack(v), 0) for m, v in layers.items()})
    prof.index.name = "layer"
    prof.to_csv(out / "memory_attention_per_layer.csv")
    print(f"wrote {out / 'memory_usage.csv'} and memory_attention_per_layer.csv")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gap")
    g.add_argument("roots", nargs="+")
    g.add_argument("--out")
    c = sub.add_parser("compare")
    c.add_argument("sweep")
    c.add_argument("--adapt_episodes", type=int, default=0,
                   help="paper protocol: drop the first N episodes per client run (adaptation phase)")
    mm = sub.add_parser("memory")
    mm.add_argument("sweep")
    a = ap.parse_args()
    {"gap": gap, "compare": compare, "memory": memory}[a.cmd](a)


if __name__ == "__main__":
    main()
