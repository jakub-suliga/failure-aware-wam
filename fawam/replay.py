"""Offline teacher-forced replay of recorded episodes through the (adapted) model.

For every chunk the model imagines the next frames exactly as during the rollout (same per-chunk noise seeds), then
the *recorded* observation latents and action chunk are written into the KV cache. Outputs:

  --out pairs.npz   pred / obs latent canvases (C, N, 24, 20) of all frames k >= 1, for the external residual
                    (sequential dual-site training, paper Sec. 3.3), plus per-frame cosine gaps per camera.

With no adapter this reproduces the recorded predictions (fidelity check: `replay_vs_recorded` in the npz, ~0 when
the run was recorded with the same code and GPU type).

  python -m fawam.replay --episodes data/expert/table_h_m05 --limit 10 --adapter artifacts/adapters/x --out x.npz
"""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from fawam.latent_data import find_episodes
from fawam.paths import LINGBOT, add_lingbot_to_path
from fawam.server import FawamServer, canvas_from_h5, cos_dist

add_lingbot_to_path()
from configs import VA_CONFIGS  # noqa: E402
from distributed.util import init_distributed  # noqa: E402


@torch.no_grad()
def replay_episode(server, ep_dir, method):
    with h5py.File(Path(ep_dir) / "episode.h5", "r") as f:
        prompt = str(f.attrs.get("prompt", f.attrs["instruction"]))
        seed = int(f.attrs["seed"])
        task = str(f.attrs["task_name"])
        lat = canvas_from_h5(f).to(server.device, server.dtype)
        planned = f["actions/planned"][()]
        obs_chunk = f["model/obs/chunk_id"][()]
        rec_pred = canvas_from_h5(f, "pred") if "model/pred" in f else None
    server._reset(prompt=prompt, seed=seed, method=method, task=task)
    server.pending_prefill = None
    preds, obs = [], []
    for c in range(len(planned)):
        rows = np.where(obs_chunk == c)[0]
        if len(rows) == 0:
            break
        if c == 0:
            server.init_latent = lat[:, :, 0:1]
            server.init_ready = True
        _, pred = server._infer(None, frame_st_id=server.frame_st_id)
        server._commit_observation(lat[:, :, rows], planned[c][:, :len(rows)])
        start = 1 if c == 0 else 0
        preds.append(pred[0, :, start:len(rows)].float().cpu())
        obs.append(lat[0, :, rows[start:]].float().cpu())
    P, O = torch.cat(preds, dim=1), torch.cat(obs, dim=1)
    fid = None
    if rec_pred is not None and rec_pred.shape[2] >= P.shape[1] + 1:
        R = rec_pred[0, :, 1:P.shape[1] + 1]
        fid = float(((P - R) ** 2).mean() / (R ** 2).mean())
    server.transformer.clear_cache(server.cache_name)
    return P, O, fid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--limit_per_task", type=int, help="few-shot n per task (applied before --limit)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--success", type=lambda s: None if s == "any" else s == "true", default=None)
    ap.add_argument("--settings", nargs="*")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--drivers", nargs="*")
    ap.add_argument("--adapter", help="adapter dir (only its inside part is used)")
    ap.add_argument("--method", default=None, help="extra method spec (default: base)")
    ap.add_argument("--model_path", default=str(LINGBOT / "checkpoints" / "lingbot-va-posttrain-robotwin"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29700 + os.getpid() % 200))
    init_distributed(1, 0, 0)
    cfg = VA_CONFIGS["robotwin"]
    cfg.wan22_pretrained_model_name_or_path = a.model_path
    cfg.enable_offload = True
    cfg.rank, cfg.local_rank, cfg.world_size = 0, 0, 1
    server = FawamServer(cfg)
    method = dict(name="replay")
    if a.method:
        from fawam.methods import resolve
        method = resolve(a.method).to_dict()
    if a.adapter:
        method.update(adapter_path=str(Path(a.adapter).resolve()), adapter_use="inside")
    eps = find_episodes(a.episodes, success=a.success, tasks=a.tasks, settings=a.settings, limit=a.limit, seed=a.seed,
                        drivers=a.drivers, limit_per_task=a.limit_per_task)
    P, O, fids, gaps = [], [], [], []
    for e in eps:
        p, o, fid = replay_episode(server, e, method)
        P.append(p)
        O.append(o)
        fids.append(np.nan if fid is None else fid)
        gaps.append(float(cos_dist(p.permute(1, 0, 2, 3), o.permute(1, 0, 2, 3)).mean()))
        print(f"{e}: frames={p.shape[1]} mean_cos_gap={gaps[-1]:.3f} replay_vs_recorded={fids[-1]:.2e}", flush=True)
    np.savez_compressed(a.out, pred=torch.cat(P, 1).half().numpy(), obs=torch.cat(O, 1).half().numpy(),
                        replay_vs_recorded=np.array(fids), episode_gap=np.array(gaps),
                        episodes=np.array([str(e) for e in eps]))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
