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

    def __init__(self, cfg, clip=1.5):
        self.clip = clip  # upstream clips at 1.5; corrections beyond the training range need more (FINDINGS 5b)
        self.inv = np.array(cfg.inverse_used_action_channel_ids)
        self.q01 = np.array(cfg.norm_stat["q01"], dtype=np.float32)[:, None, None]
        self.q99 = np.array(cfg.norm_stat["q99"], dtype=np.float32)[:, None, None]
        self.mask = np.zeros(cfg.action_dim, bool)
        self.mask[cfg.used_action_channel_ids] = True

    def __call__(self, raw):
        """raw (16, F, H) -> normalized (30, F, H), mask (30, F, H)."""
        padded = np.concatenate([raw, np.zeros_like(raw[:1])], axis=0)[self.inv]
        x = (padded - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0
        x = np.clip(x, -self.clip, self.clip)
        m = np.broadcast_to(self.mask[:, None, None], x.shape)
        return (x * m).astype(np.float32), m.copy()


ARM_XYZ = {"left": slice(0, 3), "right": slice(8, 11)}  # raw 16-d action: [l xyz quat grip, r xyz quat grip]


def relabel_planned(planned, chunk_of, li, ana):
    """planned action chunks with the analysis correction applied (same ramp as load_episode(relabel=True)).
    Used to write *corrected* actions next to the observed failure into the in-context memory."""
    out = planned.copy()
    if not (ana and ana.get("failed") and ana.get("window") and ana.get("arm") in ARM_XYZ):
        return out
    delta = np.asarray(ana.get("delta_m", [0, 0, 0]), np.float32)
    if np.abs(delta).max() == 0:
        return out
    l0, l1 = ana["window"]
    first = {}
    for r in range(len(li)):
        c = int(chunk_of[r])
        first.setdefault(c, r)
        k = int(li[r])
        if k < l0:
            continue
        frac = min(1.0, (k - l0 + 1) / (l1 - l0 + 1))
        out[c][ARM_XYZ[ana["arm"]], r - first[c]] += frac * delta[:, None]
    return out


def gap_weights(gap, li, beta):
    """Per-frame loss weight from the prediction error: 1 + beta * (robust z of the gap, clipped to [0, 3]) / 3."""
    g = np.array([gap.get(str(int(k)), np.nan) for k in li], dtype=np.float64)
    ok = ~np.isnan(g)
    w = np.ones(len(li), np.float32)
    if beta and ok.sum() > 2:
        med = np.median(g[ok])
        mad = np.median(np.abs(g[ok] - med)) * 1.4826 + 1e-6
        z = np.clip((g - med) / mad, 0, 3) / 3
        w[ok] = 1 + beta * z[ok]
    return w


def load_episode(path, norm: ActionNorm, relabel=False, gap_beta=0.0, video_in_window="real",
                 analysis_name="analysis.json", group="obs", gap_filter=0.0):
    """One episode as training arrays. Learn-from-Failure options (docs/METHOD.md):
    relabel      apply the analysis' correction delta_m to the failed arm's actions over the analysis window (linear
                 ramp up to the full correction at the failure latent) and cut the episode at truncate_at
    gap_beta     up-weight frames where the imagination diverged from reality (prediction error), see gap_weights
    video_in_window  'real': video loss on the observed (failed) frames everywhere; 'skip': no video loss inside the
                 corrected window (the observed video there shows the failure the relabeled actions avoid)
    group        'obs' (observed latents) or 'pred' (the model's own imagination, for imagined-context segments)
    gap_filter   T3VF-style sample selection: only frames whose prediction gap is in the top (1 - gap_filter) of the
                 episode get a loss (0 = all frames)"""
    path = Path(path)
    h5 = path / "episode.h5" if path.is_dir() else path
    with h5py.File(h5, "r") as f:
        src = "pred" if group == "pred" else "obs"
        lat = canvas(f, src)
        li = f[f"model/{src}/latent_index"][()]
        chunk_of_obs = f[f"model/{src}/chunk_id"][()]
        if group == "diff":  # where the imagination was wrong: observed - imagined latent
            pidx = {int(k): i for i, k in enumerate(f["model/pred/latent_index"][()])}
            keep = np.array([int(k) in pidx for k in li])
            pred = canvas(f, "pred")
            lat = lat[:, keep] - pred[:, [pidx[int(k)] for k in li[keep]]]
            li, chunk_of_obs = li[keep], chunk_of_obs[keep]
        planned = f["actions/planned"][()]  # n_chunks, 16, 2, 16
        meta = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items() if k != "task_args"}
        events = json.loads(f["fawam"].attrs.get("events", "[]")) if "fawam" in f else []
    ana_path = h5.parent / analysis_name
    ana = json.loads(ana_path.read_text()) if ana_path.exists() else None
    # keep only the first context segment if the episode had a context reset (frame ids continue across it)
    resets = [e["frame"] for e in events if e.get("kind") == "context_reset"]
    stop = len(li)
    if resets:
        stop = int(np.searchsorted(li, min(resets)))  # frame ids are absolute (continue across resets)
    if relabel and ana and ana.get("failed") and ana.get("truncate_at") is not None:
        stop = min(stop, int(np.searchsorted(li, ana["truncate_at"], side="right")))
    raw = np.zeros((16, stop, planned.shape[-1]), np.float32)
    chunk_first = {}
    for r in range(stop):
        c = int(chunk_of_obs[r])
        chunk_first.setdefault(c, r)
        raw[:, r] = planned[c][:, r - chunk_first[c]]
    in_window = np.zeros(stop, bool)
    relabeled = False
    if relabel and ana and ana.get("failed") and ana.get("window") and ana.get("arm") in ARM_XYZ:
        delta = np.asarray(ana.get("delta_m", [0, 0, 0]), np.float32)
        if np.abs(delta).max() > 0:
            l0, l1 = ana["window"]
            for r in range(stop):
                k = int(li[r])
                if k < l0:
                    continue
                frac = min(1.0, (k - l0 + 1) / (l1 - l0 + 1))
                raw[ARM_XYZ[ana["arm"]], r] += frac * delta[:, None]
                in_window[r] = True
            relabeled = True
    acts, mask = norm(raw)
    acts[:, 0] = 0.0  # frame 0: conditioning frame, zero action as at inference
    neg = None
    if relabeled:  # the executed (failed) actions = negatives for the failure-contrastive loss
        raw_neg = raw.copy()
        for r in range(stop):
            if in_window[r]:
                k = int(li[r])
                frac = min(1.0, (k - ana["window"][0] + 1) / (ana["window"][1] - ana["window"][0] + 1))
                raw_neg[ARM_XYZ[ana["arm"]], r] -= frac * np.asarray(ana["delta_m"], np.float32)[:, None]
        neg, _ = norm(raw_neg)
        neg[:, 0] = 0.0
    w = gap_weights(ana["gap"], li[:stop], gap_beta) if (ana and gap_beta) else np.ones(stop, np.float32)
    if gap_filter and ana and ana.get("gap"):
        g = np.array([ana["gap"].get(str(int(k)), np.nan) for k in li[:stop]])
        if np.isfinite(g).sum() > 2:
            thr = np.nanquantile(g, gap_filter)
            w = np.where(np.isfinite(g) & (g >= thr), w, 0.0).astype(np.float32)
    wv = w.copy()
    if relabeled and video_in_window == "skip":
        wv[in_window] = 0.0
    text = str(meta.get("instruction", ""))
    return dict(latents=lat[:, :stop].astype(np.float32), actions=acts[..., None], actions_mask=mask[..., None],
                actions_neg=(neg if neg is not None else acts)[..., None], window_mask=in_window[None].astype(np.float32),
                weight_video=wv[None], weight_action=w[None], latent_index=li[:stop][None].astype(np.int64),
                text=text, prompt=str(meta.get("prompt", text)), correction=(ana or {}).get("correction") or text,
                task=str(meta.get("task_name", "")), success=bool(meta.get("success", False)),
                setting=str(meta.get("task_config", "")), seed=int(meta.get("seed", -1)), path=str(h5.parent),
                relabeled=relabeled)


def find_episodes(roots, success=None, tasks=None, settings=None, limit=None, seed=0, drivers=None,
                  limit_per_task=None, limit_per_cell=None):
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
    if limit_per_cell is not None:  # at most n per (task, task_config): many failure types with few episodes each
        by_cell = {}
        for e in out:
            with h5py.File(e / "episode.h5", "r") as f:
                by_cell.setdefault((str(f.attrs.get("task_name")), str(f.attrs.get("task_config"))), []).append(e)
        out = []
        for k in sorted(by_cell):
            rng.shuffle(by_cell[k])
            out += by_cell[k][:limit_per_cell]
        out = sorted(out)
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
                 seed=0, cfg_prob=0.1, load_kw=None, text_mode="instruction"):
        self.load_kw = load_kw or {}
        self.text_mode = text_mode  # instruction | correction (failure analysis appended to the instruction)
        self.items = [load_episode(e, norm, **self.load_kw) for e in episodes]
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
        text = ep["correction"] if self.text_mode == "correction" else ep["text"]
        text = "" if self.rng.random() < self.cfg_prob else text
        return dict(latents=torch.from_numpy(tgt["latents"]), actions=torch.from_numpy(tgt["actions"]),
                    actions_mask=torch.from_numpy(tgt["actions_mask"]), text_emb=self.text_cache.get(text),
                    context_frames=torch.tensor(ctx_frames), weight_video=torch.from_numpy(tgt["weight_video"][0]),
                    weight_action=torch.from_numpy(tgt["weight_action"][0]),
                    actions_neg=torch.from_numpy(tgt["actions_neg"]), window_mask=torch.from_numpy(tgt["window_mask"][0]),
                    relabeled=torch.tensor(bool(ep["relabeled"])))


class FailFixDataset(EpisodeDataset):
    """Meta-training for in-context learning from failures (docs/ICL.md, section 5).

    Pairs (failed attempt, fix on the SAME seed) are matched on (task, task_config, seed); the fix is an expert replay
    (driver pair / P2) or a successful corrected retry. Each sample is [memory | target], loss only on the target.
    Mixture (probabilities, the rest = own failure of the same seed):
      p_cross    memory = a failure of ANOTHER seed of the same task/setting (generalise beyond the identical scene)
      p_success  memory = another success of the task (context that is not a failure)
      p_noctx    no memory at all (keeps the plain policy intact)
    Memory format: context_groups (obs | pred | diff), optional window around the analysed failure.
    Episodes are loaded lazily (LRU), so thousands of pairs fit in memory."""

    def __init__(self, fails, fixes, norm, text_cache, include_pred=True, p_text=0.5, max_frames=24, seed=0,
                 cfg_prob=0.1, context_groups=None, window=None, p_cross=0.0, p_success=0.0, p_noctx=0.0,
                 cache_size=256, analysis_name="analysis.json", text_only=False,
                 keyframes=0):
        from functools import lru_cache
        self.norm, self.text_cache, self.max_frames, self.p_start0 = norm, text_cache, max_frames, 1.0
        self.rng, self.cfg_prob, self.p_text = random.Random(seed), cfg_prob, p_text
        self.groups = context_groups or (["pred", "obs"] if include_pred else ["obs"])
        self.window, self.p_cross, self.p_success, self.p_noctx = window, p_cross, p_success, p_noctx
        self.analysis_name = analysis_name  # which failure analysis feeds text and FiLM-error (oracle / VLM)
        self.text_only = text_only  # no rollout tokens in the context: only the analysis text (and FiLM-error)
        self.keyframes = keyframes  # > 0: only the k latents with the highest prediction error
        self._load = lru_cache(maxsize=cache_size)(
            lambda path, group: load_episode(path, norm, group=group, analysis_name=analysis_name))
        key = lambda e: self._key(e)
        fix_by_key = {}
        for e in fixes:
            fix_by_key.setdefault(key(e), str(e))
        self.pairs = [(str(e), fix_by_key[key(e)]) for e in fails if key(e) in fix_by_key]
        if not self.pairs:
            raise ValueError("no (failure, fix) pairs with matching task/setting/seed")
        self.fails_by_ts, self.fixes_by_ts = {}, {}
        for f_, x_ in self.pairs:
            k = key(f_)[:2]
            self.fails_by_ts.setdefault(k, []).append(f_)
            self.fixes_by_ts.setdefault(k, []).append(x_)
        self.items = self.pairs

    _keys = {}

    @classmethod
    def _key(cls, e):
        e = str(e)
        if e not in cls._keys:
            with h5py.File(Path(e) / "episode.h5", "r") as f:
                cls._keys[e] = (str(f.attrs["task_name"]), str(f.attrs["task_config"]), int(f.attrs["seed"]))
        return cls._keys[e]

    def _memory(self, fail_path):
        if self.text_only:
            return []
        parts = [self._load(fail_path, g) for g in self.groups]
        if self.keyframes:
            ana_p = Path(fail_path) / self.analysis_name
            gap = (json.loads(ana_p.read_text()).get("gap") or {}) if ana_p.exists() else {}
            top = sorted(gap, key=gap.get, reverse=True)[:self.keyframes]
            if top:
                top = np.array([int(k) for k in top])
                parts = [{k: (v[:, np.isin(p["latent_index"][0], top)] if isinstance(v, np.ndarray) else v)
                          for k, v in p.items()} for p in parts]
        if self.window:
            ana_p = Path(fail_path) / self.analysis_name
            ana = json.loads(ana_p.read_text()) if ana_p.exists() else None
            if ana and ana.get("latent") is not None:
                L, b, a = int(ana["latent"]), self.window.get("before", 4), self.window.get("after", 2)
                parts = [{k: (v[:, (p["latent_index"][0] >= L - b) & (p["latent_index"][0] <= L + a)]
                              if isinstance(v, np.ndarray) else v) for k, v in p.items()} for p in parts]
        return parts

    def __getitem__(self, idx):
        fail_path, fix_path = self.pairs[idx % len(self.pairs)]
        fix = self._load(fix_path, "obs")
        ts = self._key(fail_path)[:2]
        r = self.rng.random()
        mem_path = fail_path  # the failure whose memory is in the context (its analysis drives FiLM-error)
        if r < self.p_noctx:
            parts, kind = [], "noctx"
        elif r < self.p_noctx + self.p_success:
            others = [x for x in self.fixes_by_ts[ts] if x != fix_path]
            parts, kind = ([self._load(self.rng.choice(others), "obs")] if others else []), "success"
        elif r < self.p_noctx + self.p_success + self.p_cross:
            others = [f_ for f_ in self.fails_by_ts[ts] if f_ != fail_path]
            mem_path = self.rng.choice(others) if others else fail_path
            parts, kind = self._memory(mem_path), "cross"
        else:
            parts, kind = self._memory(fail_path), "own"
        tgt = self._window(fix, self.max_frames)
        seq = parts + [tgt]
        ctx_frames = sum(p["latents"].shape[1] for p in parts)
        cat = {k: np.concatenate([p[k] for p in seq], axis=1) for k in ("latents", "actions", "actions_mask",
                                                                         "weight_video", "weight_action")}
        fail_ep = self._load(fail_path, "obs")
        from fawam.adapters import error_vector
        ana_p = Path(mem_path) / self.analysis_name
        err = error_vector(json.loads(ana_p.read_text())) if (kind in ("own", "cross") and ana_p.exists()) else None
        use_corr = kind in ("own",) and self.rng.random() < self.p_text
        text = fail_ep["correction"] if use_corr else fail_ep["text"]
        text = "" if self.rng.random() < self.cfg_prob else text
        return dict(latents=torch.from_numpy(cat["latents"]), actions=torch.from_numpy(cat["actions"]),
                    actions_mask=torch.from_numpy(cat["actions_mask"]), text_emb=self.text_cache.get(text),
                    context_frames=torch.tensor(ctx_frames), weight_video=torch.from_numpy(cat["weight_video"][0]),
                    weight_action=torch.from_numpy(cat["weight_action"][0]),
                    actions_neg=torch.from_numpy(cat["actions"]), window_mask=torch.zeros(cat["latents"].shape[1]),
                    relabeled=torch.tensor(False), kind=kind,
                    error_vec=torch.from_numpy(err if err is not None else np.zeros(32, np.float32)),
                    has_error=torch.tensor(err is not None))


def split_pairs_by_seed(ds, val_frac=0.1, seed=0):
    """Hold out whole seeds (both the failure and its fix) for validation."""
    rng = random.Random(seed)
    seeds = sorted({ds._key(f)[2] for f, _ in ds.pairs})
    val = set(rng.sample(seeds, max(1, int(len(seeds) * val_frac)))) if len(seeds) > 1 else set()
    tr = [p for p in ds.pairs if ds._key(p[0])[2] not in val]
    va = [p for p in ds.pairs if ds._key(p[0])[2] in val]
    return tr, va


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
