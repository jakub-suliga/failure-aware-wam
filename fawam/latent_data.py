"""Training sequences from recorded episodes (docs/EXPERIMENTS.md, D/E3).

Every episode.h5 written by the (fawam) server + client already holds the observation latents in the model's own
normalized latent space (`model/obs/latents/<cam>`, T-shaped canvas when concatenated) and the action chunks that went
into the KV cache (`actions/planned`: the policy's chunks for rollouts, the expert's for `--driver expert_replay`).
That is exactly what LingBot-VA's forward_train consumes, so no VAE pass and no LeRobot export is needed:

  latents (48, F, 24, 20)      obs latent canvas, frame 0 = initial observation
  actions (30, F, 16, 1)       normalized like the server's preprocess_action (frame 0 = zeros, as at inference)
  actions_mask (30, F, 16, 1)  used action channels
  text                         instruction (embedded by fawam.text_cache)

For a rollout the (video, action) pairs are always consistent (hindsight: the executed actions produced the observed
video), so the action loss on failed rollouts trains inverse dynamics, not the failure. The video loss on failed
rollouts trains the model to *predict* failures; whether that is wanted is the D3 vs D1 question.
"""
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch

from fawam.paths import CAMS


def canvas(f, group="obs"):
    g = f[f"model/{group}/latents"]
    hi, lw, rw = g["cam_high"][()], g["cam_left_wrist"][()], g["cam_right_wrist"][()]
    full = np.concatenate([np.concatenate([lw, rw], axis=-1), hi], axis=-2)  # F, C, 24, 20
    return full.transpose(1, 0, 2, 3)  # C, F, 24, 20


class ActionNorm:
    """Same mapping as VA_Server.preprocess_action (16 raw channels -> 30 normalized, unused channels masked)."""

    def __init__(self, cfg):
        self.inv = np.array(cfg.inverse_used_action_channel_ids)
        self.q01 = np.array(cfg.norm_stat["q01"], dtype=np.float32)[:, None, None]
        self.q99 = np.array(cfg.norm_stat["q99"], dtype=np.float32)[:, None, None]
        self.mask = np.zeros(cfg.action_dim, bool)
        self.mask[cfg.used_action_channel_ids] = True

    def __call__(self, raw):
        """raw (16, F, H) -> normalized (30, F, H), mask (30, F, H)."""
        padded = np.concatenate([raw, np.zeros_like(raw[:1])], axis=0)[self.inv]
        x = (padded - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0
        x = np.clip(x, -1.5, 1.5)
        m = np.broadcast_to(self.mask[:, None, None], x.shape)
        return (x * m).astype(np.float32), m.copy()


def load_episode(path, norm: ActionNorm):
    path = Path(path)
    h5 = path / "episode.h5" if path.is_dir() else path
    with h5py.File(h5, "r") as f:
        lat = canvas(f)
        li = f["model/obs/latent_index"][()]
        chunk_of_obs = f["model/obs/chunk_id"][()]
        planned = f["actions/planned"][()]  # n_chunks, 16, 2, 16
        meta = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items() if k != "task_args"}
        events = json.loads(f["fawam"].attrs.get("events", "[]")) if "fawam" in f else []
    # keep only the first context segment if the episode had a context reset (frame ids continue across it)
    resets = [e["frame"] for e in events if e.get("kind") == "context_reset"]
    stop = len(li)
    if resets:
        stop = int(np.searchsorted(li, min(resets)))  # frame ids are absolute (continue across resets)
    raw = np.zeros((16, stop, planned.shape[-1]), np.float32)
    chunk_first = {}
    for r in range(stop):
        c = int(chunk_of_obs[r])
        chunk_first.setdefault(c, r)
        raw[:, r] = planned[c][:, r - chunk_first[c]]
    acts, mask = norm(raw)
    acts[:, 0] = 0.0  # frame 0: conditioning frame, zero action as at inference
    return dict(latents=lat[:, :stop].astype(np.float32), actions=acts[..., None], actions_mask=mask[..., None],
                text=str(meta.get("instruction", "")), prompt=str(meta.get("prompt", meta.get("instruction", ""))),
                task=str(meta.get("task_name", "")), success=bool(meta.get("success", False)),
                setting=str(meta.get("task_config", "")), path=str(h5.parent))


def find_episodes(roots, success=None, tasks=None, settings=None, limit=None, seed=0, drivers=None,
                  limit_per_task=None):
    """Episode dirs under <root>/episodes/{success,fail}/<task>/<ep>/ (or roots that are episode dirs)."""
    eps = []
    for root in roots:
        root = Path(root)
        if (root / "episode.h5").exists():
            eps.append(root)
            continue
        for h5 in sorted(root.glob("**/episode.h5")):
            if "_running" in h5.parts:
                continue
            eps.append(h5.parent)
    out = []
    for e in eps:
        if success is not None or tasks or settings or drivers:
            with h5py.File(e / "episode.h5", "r") as f:
                at = f.attrs
                if success is not None and bool(at.get("success", False)) != success:
                    continue
                if tasks and str(at.get("task_name")) not in tasks:
                    continue
                if settings and str(at.get("task_config")) not in settings:
                    continue
                if drivers and str(at.get("driver", "policy")) not in drivers:
                    continue
        out.append(e)
    rng = random.Random(seed)
    if limit_per_task is not None:
        by_task = {}
        for e in out:
            with h5py.File(e / "episode.h5", "r") as f:
                by_task.setdefault(str(f.attrs.get("task_name")), []).append(e)
        out = []
        for t in sorted(by_task):
            rng.shuffle(by_task[t])
            out += by_task[t][:limit_per_task]
        out = sorted(out)
    if limit is not None:
        rng.shuffle(out)
        out = sorted(out[:limit])
    return out


class EpisodeDataset(torch.utils.data.Dataset):
    """Random windows of recorded episodes. Windows start at frame 0 (with probability p0) or at a chunk start, so the
    first frame is either the true initial frame or a cached-context frame, as at inference.

    context_k > 0 (E3, in-context training): each sample is [context episode(s) of the same task | target window],
    concatenated in time, so the model learns to use a prefilled demonstration; the loss mask covers only the target
    (context frames get timestep 0 = clean, see train_adapter)."""

    def __init__(self, episodes, norm, text_cache, max_frames=24, p_start0=0.5, context_k=0, context_pool=None,
                 seed=0, cfg_prob=0.1):
        self.items = [load_episode(e, norm) for e in episodes]
        self.text_cache = text_cache
        self.max_frames = max_frames
        self.p_start0 = p_start0
        self.context_k = context_k
        self.pool = [load_episode(e, norm) for e in context_pool] if context_pool else self.items
        self.rng = random.Random(seed)
        self.cfg_prob = cfg_prob

    def __len__(self):
        return len(self.items) * 50  # windows per epoch (sampling is random anyway)

    def _window(self, ep, max_frames):
        F = ep["latents"].shape[1]
        if F <= max_frames or self.rng.random() < self.p_start0:
            s = 0
        else:
            s = 2 * self.rng.randrange(1, max(2, (F - max_frames) // 2 + 1))
        e = min(F, s + max_frames)
        return {k: (v[:, s:e] if isinstance(v, np.ndarray) else v) for k, v in ep.items()}

    def __getitem__(self, idx):
        ep = self.items[idx % len(self.items)]
        tgt = self._window(ep, self.max_frames)
        ctx_frames = 0
        if self.context_k:
            same = [c for c in self.pool if c["task"] == ep["task"] and c["path"] != ep["path"]]
            ctx = [self.rng.choice(same) for _ in range(self.context_k)] if same else []
            parts = [c for c in ctx] + [tgt]
            ctx_frames = sum(c["latents"].shape[1] for c in ctx)
            tgt = {k: (np.concatenate([p[k] for p in parts], axis=1) if isinstance(tgt[k], np.ndarray) else tgt[k])
                   for k in tgt}
        text = "" if self.rng.random() < self.cfg_prob else ep["text"]
        return dict(latents=torch.from_numpy(tgt["latents"]), actions=torch.from_numpy(tgt["actions"]),
                    actions_mask=torch.from_numpy(tgt["actions_mask"]), text_emb=self.text_cache.get(text),
                    context_frames=torch.tensor(ctx_frames))


def build_pairs(episodes, group_pred="pred"):
    """(pred, obs) latent canvases aligned by latent index, for the external residual (paper's output-site MLP)."""
    P, O = [], []
    for e in episodes:
        h5 = Path(e) / "episode.h5" if Path(e).is_dir() else Path(e)
        with h5py.File(h5, "r") as f:
            if f"model/{group_pred}" not in f:
                continue
            pi = {int(k): i for i, k in enumerate(f[f"model/{group_pred}/latent_index"][()])}
            oi = {int(k): i for i, k in enumerate(f["model/obs/latent_index"][()])}
            common = [k for k in sorted(pi) if k in oi and k >= 1]
            if not common:
                continue
            p, o = canvas(f, group_pred), canvas(f, "obs")
            P.append(p[:, [pi[k] for k in common]])
            O.append(o[:, [oi[k] for k in common]])
    return np.concatenate(P, axis=1), np.concatenate(O, axis=1)  # C, N, 24, 20 each
