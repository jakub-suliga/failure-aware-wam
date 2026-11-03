"""Table-height sweep analysis: does the policy adapt its grasp height to a shifted table?

Reads every <run>/<offset>/episodes/{success,fail}/<task>/<episode>/episode.h5 under the given result roots
(e.g. results/table_height results/table_height_n10) and writes to --out:

  episodes.csv        one row per episode: run, offset, task, seed, success, failure type, first-grasp geometry
  success_rates.csv   successes / episodes per task and offset, all runs pooled
  seed_pairs.csv      scene seeds evaluated more than once at the same offset, with their outcomes and noise seeds
  grasp_height.png    gripper height above the target object at the first gripper close, per task and offset

Run with the robotwin env:
  ~/miniconda3/envs/robotwin/bin/python analysis/table_height.py results/table_height results/table_height_n10
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze import GRIPPER_CLOSED, label_failures  # noqa: E402

OFFSETS = {"m10": -0.10, "m05": -0.05, "z00": 0.0, "p05": 0.05}


def load_episode(ep_dir):
    with h5py.File(ep_dir / "episode.h5", "r") as f:
        meta = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        return dict(dir=ep_dir, meta=meta, success=bool(meta["success"]),
                    sim={k: f["sim"][k][()] for k in f["sim"]},
                    object_names=list(f["sim"].attrs["object_names"]),
                    noise_seeds=f["model"]["chunks"]["noise_seed"][()].tolist())


def first_grasp(ep):
    """Geometry at the first gripper close of either arm: EE height above / horizontal distance to the nearest object."""
    sim = ep["sim"]
    best = None
    for arm in ("left", "right"):
        closed = sim[f"{arm}_gripper"] < GRIPPER_CLOSED
        starts = np.where(closed[1:] & ~closed[:-1])[0] + 1
        if len(starts) and (best is None or starts[0] < best[1]):
            best = (arm, int(starts[0]))
    if best is None:
        return dict(grasp_arm=None, grasp_step=None, grasp_dz=np.nan, grasp_dxy=np.nan, grasp_contact=False)
    arm, t = best
    ee = sim[f"{arm}_ee_pose"][t, :3]
    obj = sim["object_pose"][t, :, :3]
    dxy = np.linalg.norm(obj[:, :2] - ee[:2], axis=1)
    i = int(dxy.argmin())
    contact = bool((sim[f"contact_{arm}_gripper"][t:t + 8] > 1e-6).any())
    return dict(grasp_arm=arm, grasp_step=int(sim["env_step"][t]), grasp_object=ep["object_names"][i],
                grasp_dz=float(ee[2] - obj[i, 2]), grasp_dxy=float(dxy[i]), grasp_contact=contact)


def collect(roots):
    rows = []
    for root in roots:
        for off_dir in sorted(p for p in root.iterdir() if p.name in OFFSETS):
            for h5 in sorted((off_dir / "episodes").glob("*/*/*/episode.h5")):
                if h5.parts[-4] == "_running":
                    continue
                ep = load_episode(h5.parent)
                _, ftype, fstep = label_failures(ep)
                sim = ep["sim"]
                rows.append(dict(run=root.name, offset=off_dir.name, offset_m=OFFSETS[off_dir.name],
                                 task=ep["meta"]["task_name"], seed=int(ep["meta"]["seed"]), success=ep["success"],
                                 failure_type=ftype, failure_step=fstep, env_steps=int(ep["meta"]["env_steps"]),
                                 obj_z0=float(sim["object_pose"][0, :, 2].min()),
                                 min_ee_z=float(min(sim["left_ee_pose"][:, 2].min(), sim["right_ee_pose"][:, 2].min())),
                                 noise_seed0=ep["noise_seeds"][0], **first_grasp(ep)))
    return pd.DataFrame(rows)


def plot_grasp_height(df, out):
    tasks = sorted(df.task.unique())
    order = [o for o in OFFSETS if o in set(df.offset)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.5 * len(tasks), 4), sharey=True)
    for ax, task in zip(np.atleast_1d(axes), tasks):
        d = df[(df.task == task) & df.grasp_dz.notna()]
        for k, off in enumerate(order):
            for ok, color in ((True, "#2a9d8f"), (False, "#e76f51")):
                v = d[(d.offset == off) & (d.success == ok)].grasp_dz * 100
                ax.scatter(np.full(len(v), k) + np.random.uniform(-0.15, 0.15, len(v)), v, s=18, color=color,
                           label=("success" if ok else "fail") if k == 0 else None, alpha=0.8)
        ref = d[(d.offset == "z00") & d.success].grasp_dz.median() * 100
        if np.isfinite(ref):
            ax.axhline(ref, color="gray", ls="--", lw=1, label="z00 success median")
        ax.set_xticks(range(len(order)), [f"{OFFSETS[o] * 100:+.0f} cm" for o in order])
        ax.set_title(task)
        ax.set_xlabel("table offset")
    np.atleast_1d(axes)[0].set_ylabel("EE z − object z at first close [cm]")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/table_height_analysis"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = collect(args.roots)
    df.to_csv(args.out / "episodes.csv", index=False)

    rates = df.groupby(["task", "offset_m"]).success.agg(["sum", "count"]).reset_index()
    rates.to_csv(args.out / "success_rates.csv", index=False)

    multi = df.groupby(["task", "offset", "seed"]).filter(lambda g: len(g) > 1)
    pairs = multi.groupby(["task", "offset", "seed"]).agg(
        outcomes=("success", lambda s: "".join("S" if x else "F" for x in s)),
        noise_seeds=("noise_seed0", lambda s: json.dumps(sorted(set(s))))).reset_index()
    pairs.to_csv(args.out / "seed_pairs.csv", index=False)

    plot_grasp_height(df, args.out / "grasp_height.png")

    pd.set_option("display.width", 200)
    print(rates.pivot(index="task", columns="offset_m", values=["sum", "count"]))
    print("\nfailure types:\n", df[~df.success].groupby(["task", "offset_m"]).failure_type.value_counts().unstack(fill_value=0))
    print("\nfirst-grasp dz [cm] median (n):")
    g = df.groupby(["task", "offset_m", "success"]).grasp_dz
    print(pd.concat([(g.median() * 100).round(1), g.count()], axis=1, keys=["dz_cm", "n"]))
    print("\nfirst-grasp contact rate:\n", df.groupby(["task", "offset_m"]).grasp_contact.mean().round(2).unstack())
    print("\nobj_z0 / min_ee_z median:\n", df.groupby(["task", "offset_m"])[["obj_z0", "min_ee_z"]].median().round(3))
    print("\nrepeated seeds:", len(pairs), "consistent:", (pairs.outcomes.str.len() == pairs.outcomes.apply(lambda s: max(s.count("S"), s.count("F")))).sum())
    print("noise seed differs between repeats:", (pairs.noise_seeds.apply(lambda s: len(json.loads(s)) > 1)).sum())


if __name__ == "__main__":
    main()
