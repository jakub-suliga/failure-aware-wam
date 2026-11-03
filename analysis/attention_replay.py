"""Replay recorded episodes through LingBot-VA and record where the action tokens attend.

The server is deterministic given the episode: per-chunk noise seeds, observed latents and the executed action chunks
are all in episode.h5, so the KV cache and every denoising step can be rebuilt offline. While the action chunk is
denoised, the self-attention of its 32 action tokens over the KV cache (past observed frames, the chunk's predicted
frames, past actions) is computed explicitly (conditional CFG branch, mean over heads and action tokens) for every
layer at a few denoising steps, and summed per video token.

Output per episode, <out>/<episode>.h5:
  chunk_frame_st_id (N,)            first latent index the chunk predicts
  frames/<c>  (nF, 2) int           (latent index, is_pred) of the video frames in the cache at chunk c
  maps/<c>    (L, nF, 12, 10) f16   attention mass per video token (token grid of the T-shaped canvas)
  action_mass (N, L) f32            mass on action tokens
  replay_pred_err (N,) f32          relative MSE of replayed vs. recorded predicted latents (fidelity check)

Run from lingbot-va/ with the lingbot env:
  python ../analysis/attention_replay.py <episode_dir> [...] --out <dir>
"""
import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import torch

LINGBOT = Path(__file__).resolve().parents[1] / "lingbot-va"
sys.path.insert(0, str(LINGBOT / "wan_va"))
from configs import VA_CONFIGS  # noqa: E402
from wan_va_server import VA_Server  # noqa: E402

CAMS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
ACTION_STEPS = [0, 12, 25, 37, 49]  # of 50 (+1 cache write step)
TOKEN_H, TOKEN_W = 12, 10


def canvas(f, rows):
    """Per-camera latents (stored split) -> T-shaped canvas (1, C, F, 24, 20) as the server encodes it."""
    hi = f["model/obs/latents/cam_high"][rows]
    lw = f["model/obs/latents/cam_left_wrist"][rows]
    rw = f["model/obs/latents/cam_right_wrist"][rows]
    top = np.concatenate([lw, rw], axis=-1)  # F, C, 8, 20
    full = np.concatenate([top, hi], axis=-2)  # F, C, 24, 20
    return torch.from_numpy(full.astype(np.float32)).permute(1, 0, 2, 3)[None]


class AttentionRecorder:
    """Tracks what every KV-cache slot holds and computes action->cache attention in the wrapped attn ops."""

    def __init__(self, transformer, cache_name="pos"):
        self.tf, self.cache_name = transformer, cache_name
        self.call = None  # meta of the tokens in the current forward
        self.active = False
        self.acc = None
        attn0 = transformer.blocks[0].attn1
        orig_update = attn0.update_cache

        def update_cache(cache_name, key, value, is_pred):
            slots = orig_update(cache_name, key, value, is_pred)
            s = slots.cpu().numpy()
            self.kind[s], self.frame[s], self.pos[s] = self.call["kind"], self.call["frame"], self.call["pos"]
            self.pred[s] = bool(is_pred)
            return slots

        attn0.update_cache = update_cache
        for li, blk in enumerate(transformer.blocks):
            blk.attn1.attn_op = partial(self._attn, li, blk.attn1, blk.attn1.attn_op)

    def reset(self):
        """After the server re-created the KV cache (new episode)."""
        n = self.tf.blocks[0].attn1.attn_caches[self.cache_name]["k"].shape[1]
        self.kind = np.full(n, -1)  # 0 video, 1 action
        self.frame = np.full(n, -1)
        self.pos = np.full(n, -1)
        self.pred = np.zeros(n, bool)

    def set_call(self, action_mode, frame_st_id, num_frames, per_frame):
        f = np.repeat(np.arange(frame_st_id, frame_st_id + num_frames), per_frame)
        p = np.tile(np.arange(per_frame), num_frames)
        self.call = dict(kind=int(action_mode), frame=f, pos=p)

    def _attn(self, li, attn, op, q, k, v):
        out = op(q, k, v)
        if self.active:
            mask = attn.attn_caches[self.cache_name]["mask"]
            valid = mask.nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
            qq, kk = q[0].float(), k[0].float()  # conditional branch; (Lq, H, D), (Lk, H, D)
            scores = torch.einsum("qhd,khd->hqk", qq, kk) / qq.shape[-1] ** 0.5
            w = scores.softmax(-1).mean(dim=(0, 1)).cpu().numpy()  # (Lk,)
            self.acc[li].append((valid, w))
        return out

    def stop(self):
        """-> frames (nF, 2), maps (L, nF, 12, 10), action mass (L,), all averaged over the captured steps."""
        self.active = False
        L = len(self.tf.blocks)
        steps = len(self.acc[0])
        valid0 = self.acc[0][0][0]
        vid = valid0[self.kind[valid0] == 0]
        keys = sorted({(int(self.frame[s]), int(self.pred[s])) for s in vid})
        fidx = {key: i for i, key in enumerate(keys)}
        maps = np.zeros((L, len(keys), TOKEN_H * TOKEN_W), np.float32)
        amass = np.zeros(L, np.float32)
        for li in range(L):
            for valid, w in self.acc[li]:
                isvid = self.kind[valid] == 0
                sv = valid[isvid]
                fi = np.array([fidx[(int(self.frame[s]), int(self.pred[s]))] for s in sv])
                np.add.at(maps[li], (fi, self.pos[sv]), w[isvid] / steps)
                amass[li] += w[~isvid].sum() / steps
        return np.array(keys), maps.reshape(L, len(keys), TOKEN_H, TOKEN_W), amass


def replay(server, rec, ep_dir, out_path):
    f = h5py.File(ep_dir / "episode.h5", "r")
    prompt, seed = f.attrs["instruction"], int(f.attrs["seed"])
    frame_st = f["model/chunks/frame_st_id"][:]
    planned = f["actions/planned"][:]  # N, 16, F, 16
    obs_chunk = f["model/obs/chunk_id"][:]
    pred_chunk = f["model/pred/chunk_id"][:]
    server._reset(prompt=prompt, seed=seed)
    rec.reset()

    # Feed recorded observations instead of encoding images
    queue = []
    server._encode_obs = lambda obs: queue.pop(0).to(server.device).to(server.dtype)
    orig_tf = server.transformer.forward

    capture = {"on": False, "step": 0}

    def tf_forward(input_dict, update_cache=0, cache_name="pos", action_mode=False, **kw):
        g = input_dict["grid_id"]
        g = g[0] if g.dim() == 3 else g
        nf = input_dict["noisy_latents"].shape[2]
        per_frame = input_dict["noisy_latents"].shape[3] if action_mode else TOKEN_H * TOKEN_W
        rec.set_call(action_mode, int(g[0].min().floor()), nf, per_frame)
        rec.active = False
        if action_mode and capture["on"]:  # action denoising step counter within the chunk
            rec.active = capture["step"] in ACTION_STEPS
            capture["step"] += 1
        return orig_tf(input_dict, update_cache=update_cache, cache_name=cache_name, action_mode=action_mode, **kw)

    server.transformer.forward = tf_forward

    out = h5py.File(out_path, "w")
    out.attrs.update(episode=ep_dir.name, instruction=prompt, seed=seed, action_steps=ACTION_STEPS,
                     task=ep_dir.parent.name, success=bool(f.attrs["success"]))
    amass_all, errs = [], []
    for c, st in enumerate(frame_st):
        if c == 0:
            queue.append(canvas(f, np.where(obs_chunk == 0)[0][:1]))
        rec.acc = {li: [] for li in range(len(server.transformer.blocks))}
        capture["on"], capture["step"] = True, 0
        _, latents = server._infer(dict(obs=[None]), frame_st_id=int(st))
        capture["on"] = False
        frames, maps, amass = rec.stop()
        out[f"frames/{c}"] = frames
        out.create_dataset(f"maps/{c}", data=maps.astype(np.float16), compression="gzip")
        amass_all.append(amass)
        # fidelity: replayed vs. recorded predicted latents of this chunk
        rows = np.where(pred_chunk == c)[0]
        if len(rows):
            ref = canvas_pred(f, rows)
            got = latents[0].float().cpu()
            errs.append(float(((got - ref) ** 2).mean() / (ref ** 2).mean()))
        else:
            errs.append(np.nan)
        rows = np.where(obs_chunk == c)[0]
        if len(rows) and c < len(planned):
            queue.append(canvas(f, rows[1:] if c == 0 else rows))  # chunk 0: the server prepends the init frame
            server._compute_kv_cache(dict(obs=[None], state=planned[c].astype(np.float32)))
        print(f"  chunk {c} frame_st {st} pred_err {errs[-1]:.2e} action_mass L15 {amass[15]:.3f}", flush=True)
    out["chunk_frame_st_id"] = frame_st
    out["action_mass"] = np.array(amass_all)
    out["replay_pred_err"] = np.array(errs, np.float32)
    out.close()
    server.transformer.forward = orig_tf


def canvas_pred(f, rows):
    hi = f["model/pred/latents/cam_high"][rows]
    top = np.concatenate([f["model/pred/latents/cam_left_wrist"][rows],
                          f["model/pred/latents/cam_right_wrist"][rows]], axis=-1)
    return torch.from_numpy(np.concatenate([top, hi], axis=-2).astype(np.float32)).permute(1, 0, 2, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model_path", default=str(LINGBOT / "checkpoints/lingbot-va-posttrain-robotwin"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = VA_CONFIGS["robotwin"]
    cfg.wan22_pretrained_model_name_or_path = args.model_path
    cfg.enable_offload = True
    cfg.rank = cfg.local_rank = 0
    cfg.world_size = 1
    cfg.save_root = str(args.out / "_server")
    torch.cuda.set_device(0)
    server = VA_Server(cfg)
    server._reset(prompt=None)
    rec = AttentionRecorder(server.transformer)
    for ep_dir in args.episodes:
        task = ep_dir.parent.name
        out_path = args.out / f"{task}__{ep_dir.name}.h5"
        if out_path.exists():
            print("skip", out_path.name)
            continue
        print("replay", task, ep_dir.name, flush=True)
        replay(server, rec, ep_dir, out_path.with_suffix(".tmp"))
        out_path.with_suffix(".tmp").rename(out_path)


if __name__ == "__main__":
    main()
