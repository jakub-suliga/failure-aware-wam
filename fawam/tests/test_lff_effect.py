"""Did a Learn-from-Failure update learn its correction? Teacher-forced replay of the failed episode (recorded
observations + action chunks) through the base model and through the adapted model; compares the z of the failed
arm's proposed actions in the analysis window with the recorded actions and the relabeled target
(lingbot env, run from the repo root):

  python -m fawam.tests.test_lff_effect <failed episode dir> <adapter dir> [--vg 5]
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from fawam.latent_data import ARM_XYZ
from fawam.paths import LINGBOT
from fawam.server import FawamServer, canvas_from_h5
from fawam.paths import add_lingbot_to_path

add_lingbot_to_path()
from configs import VA_CONFIGS  # noqa: E402
from distributed.util import init_distributed  # noqa: E402


@torch.no_grad()
def proposals(server, ep, method):
    with h5py.File(ep / "episode.h5", "r") as f:
        lat = canvas_from_h5(f).to(server.device, server.dtype)
        planned = f["actions/planned"][()]
        oc = f["model/obs/chunk_id"][()]
        fst = f["model/chunks/frame_st_id"][()]
        prompt, seed, task = str(f.attrs["instruction"]), int(f.attrs["seed"]), str(f.attrs["task_name"])
    server._reset(prompt=prompt, seed=seed, method=method, task=task)
    server.pending_prefill = None
    out = {}
    for c in range(len(planned)):
        rows = np.where(oc == c)[0]
        if c == 0:
            server.init_latent, server.init_ready = lat[:, :, 0:1], True
        act, _ = server._infer(None, frame_st_id=server.frame_st_id)
        out[int(fst[c])] = act  # (16, 2, 16) raw
        server._commit_observation(lat[:, :, rows], planned[c][:, :len(rows)])
    server.transformer.clear_cache(server.cache_name)
    return out, planned, fst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("adapter")
    ap.add_argument("--vg", type=float, default=5.0)
    a = ap.parse_args()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29813")
    init_distributed(1, 0, 0)
    cfg = VA_CONFIGS["robotwin"]
    cfg.wan22_pretrained_model_name_or_path = str(LINGBOT / "checkpoints" / "lingbot-va-posttrain-robotwin")
    cfg.enable_offload = True
    cfg.rank, cfg.local_rank, cfg.world_size = 0, 0, 1
    server = FawamServer(cfg)
    ep = Path(a.episode)
    ana = json.loads((ep / "analysis.json").read_text())
    zc = ARM_XYZ[ana["arm"]].start + 2
    l0, l1 = ana["window"]
    base, planned, fst = proposals(server, ep, dict(name="base", video_guidance_scale=a.vg))
    adap, _, _ = proposals(server, ep, dict(name="lff", video_guidance_scale=a.vg,
                                            adapter_path=str(Path(a.adapter).resolve())))
    print(f"arm {ana['arm']}, correction dz {ana['delta_m'][2]:+.3f} m over latents {l0}..{l1}")
    print("latent | recorded z | relabeled target z | base proposal z | adapted proposal z   (relative to start, m)")
    for c, st in enumerate(fst):
        for i in range(2):
            k = int(st) + i
            if k < 1 or k > ana["truncate_at"] + 1:
                continue
            rec = planned[c][zc, i].mean()
            frac = 0 if k < l0 else min(1.0, (k - l0 + 1) / (l1 - l0 + 1))
            tgt = rec + frac * ana["delta_m"][2]
            print(f"{k:6d} | {rec:+.3f} | {tgt:+.3f} | {base[st][zc, i].mean():+.3f} | {adap[st][zc, i].mean():+.3f}")


if __name__ == "__main__":
    main()
