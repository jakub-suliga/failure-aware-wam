"""Demo bank for in-context prefill (docs/EXPERIMENTS.md, E1/E3).

  python -m fawam.demo_bank build data/expert/table_h_m05 --out artifacts/demo_bank_m05 [--success true]

Copies nothing: index.json lists the episode dirs (relative to the bank via symlinks) with task, setting and
instruction; init_latents.npy holds the initial head-camera latent of each (the server's nearest-neighbour key).
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from fawam.latent_data import canvas, find_episodes


def build(a):
    out = Path(a.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    index, feats = [], []
    for e in find_episodes(a.roots, success=a.success, settings=a.settings, drivers=a.drivers):
        with h5py.File(e / "episode.h5", "r") as f:
            lat = canvas(f)  # C, F, 24, 20
            at = f.attrs
            name = f"{at['task_name']}__{at['task_config']}__{e.name}"
            link = out / "episodes" / name
            if not link.exists():
                os.symlink(e.resolve(), link)
            index.append(dict(path=f"episodes/{name}", task=str(at["task_name"]), setting=str(at["task_config"]),
                              instruction=str(at["instruction"]), success=bool(at["success"]), seed=int(at["seed"]),
                              driver=str(at.get("driver", "policy")), frames=int(lat.shape[1])))
            feats.append(lat[:, 0, 8:].reshape(-1).astype(np.float32))
    (out / "index.json").write_text(json.dumps(index, indent=1))
    np.save(out / "init_latents.npy", np.stack(feats))
    print(f"{len(index)} episodes -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("roots", nargs="+")
    b.add_argument("--out", required=True)
    b.add_argument("--success", type=lambda s: None if s == "any" else s == "true", default=True)
    b.add_argument("--settings", nargs="*")
    b.add_argument("--drivers", nargs="*")
    a = ap.parse_args()
    build(a)


if __name__ == "__main__":
    main()
