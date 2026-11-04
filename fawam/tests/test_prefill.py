"""In-context prefill through the real server without the simulator (lingbot env, ~17 GB GPU, run from lingbot-va/):

  PYTHONPATH=.. python -m fawam.tests.test_prefill <target episode dir> <demo bank dir>

Feeds the target episode's recorded camera frames (real_*.mp4) and action chunks, once without and once with a
demo prefilled from the bank (nearest neighbour), and reports the per-chunk prediction gap of both runs.
"""
import os
import sys
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np

from fawam.paths import LINGBOT
from fawam.server import FawamServer

sys.path.insert(0, str(LINGBOT / "wan_va"))
from configs import VA_CONFIGS  # noqa: E402
from distributed.util import init_distributed  # noqa: E402

KEYS = {"cam_high": "observation.images.cam_high", "cam_left_wrist": "observation.images.cam_left_wrist",
        "cam_right_wrist": "observation.images.cam_right_wrist"}


def frames(ep):
    return {k: iio.imread(ep / f"real_{k}.mp4") for k in KEYS}


def run(server, ep, method, n_chunks=4):
    fr = frames(ep)
    with h5py.File(ep / "episode.h5", "r") as f:
        prompt, seed, task = str(f.attrs["instruction"]), int(f.attrs["seed"]), str(f.attrs["task_name"])
        planned = f["actions/planned"][()]
    server.infer(dict(reset=True, prompt=prompt, seed=seed, method=method, task_name=task))
    obs0 = {v: fr[k][0] for k, v in KEYS.items()}
    out = server.infer(dict(obs=obs0))
    gaps, raw_i = [], 1
    for c in range(n_chunks):
        if c > 0:
            out = server.infer(dict(obs=obs0))
        n_obs = 4 if c == 0 else 8
        keys = [{v: fr[k][min(raw_i + j, len(fr[k]) - 1)] for k, v in KEYS.items()} for j in range(n_obs)]
        raw_i += n_obs
        r = server.infer(dict(obs=keys, compute_kv_cache=True, state=planned[c]))
        gaps.append(round(max(r["gap"]["cam_left_wrist"], r["gap"]["cam_right_wrist"]), 3))
    return out["frame_st_id"], gaps, server.episode_log["prefill"]


def main(target, bank):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29811")
    init_distributed(1, 0, 0)
    cfg = VA_CONFIGS["robotwin"]
    cfg.wan22_pretrained_model_name_or_path = str(LINGBOT / "checkpoints" / "lingbot-va-posttrain-robotwin")
    cfg.enable_offload = True
    cfg.rank, cfg.local_rank, cfg.world_size = 0, 0, 1
    server = FawamServer(cfg)
    ep = Path(target)
    for name, method in (("no context", "base"),
                         ("1 demo (nearest)", dict(name="icl", prefill=dict(bank=bank, k=1, select="nearest", gap=1)))):
        fst, gaps, pre = run(server, ep, method)
        print(f"{name:18s} last chunk frame_st_id={fst} wrist gap per chunk={gaps} prefill={[Path(p).parent.name for p in pre]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
