"""Does an adapter change what the model predicts? Teacher-forced replay of the same recorded episodes under several
conditions, with identical history (recorded observations + actions) and identical noise seeds:

  base      unchanged model
  reseed    unchanged model, other noise seeds            (reference: how much do two samples of the same model differ)
  inside    adapter, inner site only
  dual      adapter, inner + outer site (as evaluated)

Per chunk it stores the imagined latents and the predicted action chunk of every condition and writes
  <out>/compare.npz    per-chunk metrics (see `rows`)
  <out>/summary.json   means per setting
  <out>/img/*.png      decoded frames: real | base | dual | |dual - base| for the chunk where the arm is lowest

  python -m fawam.compare_pred --pairs <episode dir>=<adapter dir> ... --out results/analysis/compare_pred/x
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from fawam.paths import LINGBOT, add_lingbot_to_path
from fawam.server import FawamServer, canvas_from_h5, cos_dist

add_lingbot_to_path()
from configs import VA_CONFIGS  # noqa: E402
from distributed.util import init_distributed  # noqa: E402

ARMS = dict(left=slice(0, 3), right=slice(8, 11))


@torch.no_grad()
def replay(server, ep_dir, method, seed_shift=0):
    """-> list over chunks of dict(pred (C, f, 24, 20), obs (same), actions (16, F, H) raw)."""
    with h5py.File(Path(ep_dir) / "episode.h5", "r") as f:
        prompt = str(f.attrs.get("prompt", f.attrs["instruction"]))
        seed = int(f.attrs["seed"]) + seed_shift
        task = str(f.attrs["task_name"])
        lat = canvas_from_h5(f).to(server.device, server.dtype)
        planned = f["actions/planned"][()]
        obs_chunk = f["model/obs/chunk_id"][()]
    server._reset(prompt=prompt, seed=seed, method=method, task=task)
    server.pending_prefill = None
    out = []
    for c in range(len(planned)):
        rows = np.where(obs_chunk == c)[0]
        if len(rows) == 0:
            break
        if c == 0:
            server.init_latent = lat[:, :, 0:1]
            server.init_ready = True
        actions, pred = server._infer(None, frame_st_id=server.frame_st_id)
        server._commit_observation(lat[:, :, rows], planned[c][:, :len(rows)])
        start = 1 if c == 0 else 0
        out.append(dict(pred=pred[0, :, start:len(rows)].float().cpu(), obs=lat[0, :, rows[start:]].float().cpu(),
                        actions=np.asarray(actions, dtype=np.float32), planned=planned[c]))
    server.transformer.clear_cache(server.cache_name)
    return out


def episode_info(ep_dir):
    with h5py.File(Path(ep_dir) / "episode.h5", "r") as f:
        z = {a: f[f"sim/{a}_ee_pose"][:, 2] for a in ARMS}
        arm = max(z, key=lambda a: np.ptp(z[a]))
        step_low = int(np.argmin(z[arm]))
        chunk_low = int(np.searchsorted(f["actions/planned_env_step"][()], step_low, side="right") - 1)
        return dict(arm=arm, chunk_low=chunk_low, success=bool(f.attrs["success"]), seed=int(f.attrs["seed"]),
                    setting=str(f.attrs["setting"]))


def frame_gap(server, p, o):
    """Cosine distance per camera, mean over the chunk's frames; p, o: (C, f, 24, 20)."""
    res = {}
    for cam, (hs, ws) in server.LATENT_CAM_SLICES.items():
        res[cam] = float(cos_dist(p[:, :, hs, ws].permute(1, 0, 2, 3), o[:, :, hs, ws].permute(1, 0, 2, 3)).mean())
    return res


def decode(server, lat):
    """(C, f, 24, 20) -> {cam: uint8 image of the last frame}."""
    return {cam: np.asarray(server._decode_latents_uint8(l.to(server.device, server.dtype))[-1])
            for cam, l in server._split_cams(lat).items()}


def montage(server, chunk, info, path):
    from PIL import Image
    imgs = {k: decode(server, chunk[k]) for k in ("obs", "base", "dual")}
    cams = ["cam_high", f"cam_{info['arm']}_wrist"]
    rows = []
    for cam in cams:
        d = np.abs(imgs["dual"][cam].astype(np.int16) - imgs["base"][cam].astype(np.int16))
        d = np.clip(d * 4, 0, 255).astype(np.uint8)
        rows.append(np.concatenate([imgs["obs"][cam], imgs["base"][cam], imgs["dual"][cam], d], axis=1))
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="<episode dir>=<adapter dir>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_reseed", action="store_true")
    ap.add_argument("--model_path", default=str(LINGBOT / "checkpoints" / "lingbot-va-posttrain-robotwin"))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(__import__("fawam.paths", fromlist=["free_port"]).free_port()))
    init_distributed(1, 0, 0)
    cfg = VA_CONFIGS["robotwin"]
    cfg.wan22_pretrained_model_name_or_path = a.model_path
    cfg.enable_offload = False
    cfg.rank, cfg.local_rank, cfg.world_size = 0, 0, 1
    server = FawamServer(cfg)

    rows = []
    for pair in a.pairs:
        ep, adapter = pair.split("=")
        info = episode_info(ep)
        ad = str(Path(adapter).resolve())
        conds = dict(base=(dict(name="replay"), 0),
                     inside=(dict(name="replay", adapter_path=ad, adapter_use="inside"), 0),
                     dual=(dict(name="replay", adapter_path=ad, adapter_use="all"), 0))
        if not a.no_reseed:
            conds["reseed"] = (dict(name="replay"), 1)
        R = {k: replay(server, ep, m, s) for k, (m, s) in conds.items()}
        arm = ARMS[info["arm"]]
        for c, b in enumerate(R["base"]):
            if b["pred"].shape[1] == 0:
                continue
            row = dict(episode=str(ep), adapter=ad, chunk=c, **info)
            row["planned_vs_base_mm"] = float(np.abs(b["actions"][arm] - b["planned"][arm]).mean() * 1000)
            for k, r in R.items():
                p = r[c]["pred"]
                for cam, g in frame_gap(server, p, b["obs"]).items():
                    row[f"gap_{k}_{cam}"] = g
                row[f"l1_{k}"] = float((p - b["obs"]).abs().mean())
                row[f"z_{k}_mm"] = float(r[c]["actions"][arm][2].mean() * 1000)      # mean height of the chunk
                row[f"zmin_{k}_mm"] = float(r[c]["actions"][arm][2].min() * 1000)
                if k == "base":
                    continue
                row[f"latent_change_{k}"] = float((p - b["pred"]).norm() / (b["pred"].norm() + 1e-8))
                for cam, g in frame_gap(server, p, b["pred"]).items():
                    row[f"cos_to_base_{k}_{cam}"] = g
                d = (r[c]["actions"] - b["actions"])[arm] * 1000                     # 3, F, H in mm
                row[f"dz_{k}_mm"] = float(d[2].mean())
                row[f"dz_end_{k}_mm"] = float(d[2, -1, -4:].mean())
                row[f"dxyz_{k}_mm"] = float(np.linalg.norm(d, axis=0).mean())
            rows.append(row)
            if c == info["chunk_low"]:
                try:
                    montage(server, dict(obs=b["obs"], base=b["pred"], dual=R["dual"][c]["pred"]), info,
                            out / "img" / f"{info['setting']}_seed{info['seed']}_chunk{c}.png")
                except Exception as e:  # the comparison numbers do not depend on the pictures
                    print(f"montage failed: {e!r}", flush=True)
        low = [r for r in rows if r["episode"] == str(ep) and r["chunk"] == info["chunk_low"]]
        print(f"{ep}: chunks={len(R['base'])} arm={info['arm']} low_chunk={info['chunk_low']} " +
              (f"dz_dual={low[0]['dz_dual_mm']:.1f}mm change_dual={low[0]['latent_change_dual']:.3f}" if low else ""),
              flush=True)
        np.savez_compressed(out / "compare.npz", rows=np.array([json.dumps(r) for r in rows]))

    summary = {}
    num = [k for k in rows[0] if isinstance(rows[0][k], float)]
    for setting in sorted({r["setting"] for r in rows}):
        for where, sel in (("all_chunks", lambda r: True), ("low_chunk", lambda r: r["chunk"] == r["chunk_low"]),
                           ("before_low", lambda r: r["chunk"] <= r["chunk_low"])):
            rs = [r for r in rows if r["setting"] == setting and sel(r)]
            summary[f"{setting}|{where}"] = dict(n=len(rs), **{k: round(float(np.mean([r[k] for r in rs])), 4)
                                                               for k in num})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
