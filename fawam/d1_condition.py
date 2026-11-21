"""p20 D1 (plan v2, 2026-11-21): is LingBot-VA's video prediction conditioned on actions (a forecast) or a plan?

LingBot-VA denoises the video of chunk c+1 before its actions, so actions can only reach the predicted video through
the action tokens of the PAST chunks in the KV cache. Test: rebuild the cache of a recorded episode up to chunk c
(observed latents + executed actions, exactly as during the rollout), write chunk c with a modified action chunk, then
predict chunk c+1 with the SAME noise. Variants of the chunk-c actions:
  orig       the executed actions
  orig_n2    the executed actions, other denoising noise   -> sampling noise floor
  zp2 / zm2  both arms' targets 2 cm higher / lower
  rand       the action chunk of another (random) chunk of the same episode + 3 cm Gaussian noise
Measures per variant: relative latent change of the predicted chunk vs orig, and the imagined end-effector shift
(pose probe on the predicted frames' hidden states, artifacts/probe/pose_probe.npz) in z.

Decision (gate results/gates/p20_d1.json, used by TTT P7): conditioned = the imagined z shift follows the sign of the
+-2 cm action change in >= 70 % of the samples AND the median latent change of zp2/zm2 exceeds 1.5x the noise floor.

  python -m fawam.d1_condition --episodes <roots> --limit 24 --out results/p20_d1
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from fawam.latent_data import find_episodes
from fawam.paths import LINGBOT, ROOT, add_lingbot_to_path
from fawam.server import FawamServer, canvas_from_h5

add_lingbot_to_path()
from configs import VA_CONFIGS  # noqa: E402
from distributed.util import init_distributed  # noqa: E402


@torch.no_grad()
def predict_after(server, lat, planned, obs_chunk, c, act_c, meta, noise_shift=0):
    """Cache = chunks 0..c-1 as recorded + chunk c observed with `act_c`; returns the predicted chunk c+1 latents
    (C, F, H, W) and the probe xyz of its frames (F, 6) or None."""
    server._reset(prompt=meta["prompt"], seed=meta["seed"] + noise_shift, method=dict(name="d1"), task=meta["task"])
    server.pending_prefill = None
    for cc in range(c + 1):
        rows = np.where(obs_chunk == cc)[0]
        if cc == 0:
            server.init_latent = lat[:, :, 0:1]
            server.init_ready = True
        server._commit_observation(lat[:, :, rows], act_c if cc == c else planned[cc][:, :len(rows)])
    server.episode_log["pred"].clear()
    server.episode_log["seed"] = meta["seed"] + noise_shift
    _, pred = server._infer(None, frame_st_id=server.frame_st_id)
    hid = server.episode_log["pred"][-1].get("hidden")
    pp = server._pose_probe()
    xyz = pp(hid.float().numpy()) if (pp is not None and hid is not None) else None
    server.transformer.clear_cache(server.cache_name)
    return pred[0].float().cpu(), xyz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--model_path", default=str(LINGBOT / "checkpoints" / "lingbot-va-posttrain-robotwin"))
    ap.add_argument("--out", default=str(ROOT / "results" / "p20_d1"))
    a = ap.parse_args()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(__import__("fawam.paths", fromlist=["free_port"]).free_port()))
    init_distributed(1, 0, 0)
    cfg = VA_CONFIGS["robotwin"]
    cfg.wan22_pretrained_model_name_or_path = a.model_path
    cfg.enable_offload = True
    cfg.rank, cfg.local_rank, cfg.world_size = 0, 0, 1
    server = FawamServer(cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    eps = find_episodes(a.episodes, limit=a.limit, seed=0)
    rows_out = []
    for e in eps:
        with h5py.File(Path(e) / "episode.h5", "r") as f:
            meta = dict(prompt=str(f.attrs.get("prompt", f.attrs["instruction"])), seed=int(f.attrs["seed"]),
                        task=str(f.attrs["task_name"]))
            lat = canvas_from_h5(f).to(server.device, server.dtype)
            planned = f["actions/planned"][()]
            obs_chunk = f["model/obs/chunk_id"][()]
        n_ch = int(obs_chunk.max()) + 1
        for c in a.chunks:
            if c + 1 >= n_ch:
                continue
            nr = int((obs_chunk == c).sum())
            base = planned[c][:, :nr].copy()
            variants = dict(orig=base)
            for name, dz in (("zp2", 0.02), ("zm2", -0.02)):
                v = base.copy()
                v[[2, 10]] += dz  # relative EE targets of both arms, metres
                variants[name] = v
            other = int(rng.choice([x for x in range(n_ch) if x != c]))
            rv = planned[other][:, :nr].copy()
            rv[[0, 1, 2, 8, 9, 10]] += rng.normal(0, 0.03, size=(6, 1, 1)).astype(np.float32)
            variants["rand"] = rv
            res = {}
            for name, act in variants.items():
                res[name] = predict_after(server, lat, planned, obs_chunk, c, act, meta)
            res["orig_n2"] = predict_after(server, lat, planned, obs_chunk, c, base, meta, noise_shift=7)
            p0, x0 = res["orig"]
            row = dict(episode=str(e), task=meta["task"], chunk=c)
            for name, (p, x) in res.items():
                if name == "orig":
                    continue
                row[f"dlat_{name}"] = float((p - p0).norm() / (p0.norm() + 1e-8))
                if x is not None and x0 is not None:
                    d = np.asarray(x) - np.asarray(x0)  # F, 6 (left xyz, right xyz)
                    row[f"dz_{name}"] = float(np.mean(d[:, [2, 5]]))
            rows_out.append(row)
            print(json.dumps(row), flush=True)
    (out / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_out))
    med = lambda k: float(np.nanmedian([r.get(k, np.nan) for r in rows_out])) if rows_out else float("nan")
    sign_ok = [(r.get("dz_zp2", 0) > 0) for r in rows_out if "dz_zp2" in r] + \
              [(r.get("dz_zm2", 0) < 0) for r in rows_out if "dz_zm2" in r]
    frac_sign = float(np.mean(sign_ok)) if sign_ok else float("nan")
    floor = med("dlat_orig_n2")
    dl = float(np.nanmedian([r[k] for r in rows_out for k in ("dlat_zp2", "dlat_zm2") if k in r])) if rows_out else 0
    summ = dict(n=len(rows_out), noise_floor=floor, dlat_z2=dl, dlat_rand=med("dlat_rand"),
                dz_zp2_cm=100 * med("dz_zp2"), dz_zm2_cm=100 * med("dz_zm2"), dz_rand_cm=100 * med("dz_rand"),
                frac_sign_follows=frac_sign,
                conditioned=bool(frac_sign >= 0.7 and dl > 1.5 * floor),
                rule="imagined z follows the sign of +-2 cm in >= 70 % and latent change > 1.5 x noise floor")
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    gates = ROOT / "results" / "gates"
    gates.mkdir(parents=True, exist_ok=True)
    (gates / "p20_d1.json").write_text(json.dumps(dict(passed=summ["conditioned"], **summ), indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
