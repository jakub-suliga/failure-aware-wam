"""p21 ICL stage 1 (plan v2, 2026-11-21): training pairs from the snap driver.

Data (driver snap, lingbot-va client `_snap_continuations`): per seed the base policy's attempt (hidden parameter
drawn from the seed), and from the snapshot k = 1, 2, 3 chunks before the gripper closes the planner continues to the
end (expert, replayed through the EE interface, compensated under a hidden offset). One pair per successful
continuation:

  sequence = [context of the failed attempt | own attempt up to the reset point (past) | expert continuation (target)]

Loss: video flow matching on the target window only (context and past are conditioning). The failure itself is only in
the context. Context variants (window = 7 latent frames around the latent in which the gripper closed):
  C0 none                          C1 mismatch obs - pred           C2 C1 + real frames
  C3 C1 + real + imagined frames   C4 scalar error per frame only (|obs - pred| broadcast, no direction)
  C5 C2 + gripper state (gripper channels of the executed actions next to the context latents)
  C6 C2 + phase text (oracle phases, fawam/phases.py)   C7 C2 + oracle text   C8 oracle text only
Context latents carry NO actions except in C5 (zero action tokens), so the context content is only what the variant
says. Mixture: 10 % of samples use a SUCCESSFUL attempt of the same scene as context (target = expert without
correction), the rest the own failure. Foreign context (validation / gate): a failure of the same task and setting
with the opposite sign of the hidden parameter.

Labels for analysis (never inputs, except the oracle texts of C7/C8): hid offset (S-B) / table bias (S-A).
"""
import json
import random
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import torch

from fawam.latent_data import canvas

CTX_GROUPS = {"C0": [], "C1": ["diff"], "C2": ["diff", "obs"], "C3": ["diff", "obs", "pred"], "C4": ["diffmag"],
              "C5": ["diff", "obs"], "C6": ["diff", "obs"], "C7": ["diff", "obs"], "C8": []}
CTX_TEXT = {"C6": "phase", "C7": "oracle", "C8": "oracle"}
WIN = (3, 3)  # latent frames before / after the close latent


def _attrs(h5):
    with h5py.File(h5, "r") as f:
        return {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items() if k != "task_args"}


def param_of(meta):
    """(kind, value vector, sign) of the hidden parameter: S-B lateral offset (sign of y) / S-A table bias."""
    hid = meta.get("hid")
    hid = json.loads(hid) if isinstance(hid, str) else hid
    if hid and hid.get("type") == "trans":
        o = [float(x) for x in hid["o"]]
        return "lat", o, (1 if o[1] > 0 else -1 if o[1] < 0 else 0)
    if hid and hid.get("type") == "gain":
        g = float(hid["o"][0])
        return "gain", [g], (1 if g > 1 else -1)
    tb = meta.get("table_bias")
    tb = json.loads(tb) if isinstance(tb, str) and tb not in ("None", "") else tb
    if tb not in (None, "None"):
        return "table", [float(tb)], (1 if float(tb) > 0 else -1)
    return "none", [], 0


def oracle_text(meta, instruction):
    """C7/C8: deviation at the close and the correction, from the simulator label (oracle diagnostic)."""
    kind, v, _ = param_of(meta)
    if kind == "lat":
        parts = [f"{abs(v[i]) * 100:.0f} cm along {'+' if v[i] > 0 else '-'}{ax}" for i, ax in ((1, "y"), (0, "x"))
                 if abs(v[i]) >= 0.005]
        corr = [f"{abs(v[i]) * 100:.0f} cm along {'-' if v[i] > 0 else '+'}{ax}" for i, ax in ((1, "y"), (0, "x"))
                if abs(v[i]) >= 0.005]
        return f"{instruction} In the previous attempt the gripper closed {' and '.join(parts)} off the target; " \
               f"move {' and '.join(corr)}."
    if kind == "table":
        d = abs(v[0]) * 100
        return (f"{instruction} In the previous attempt the gripper closed {d:.0f} cm above the object; grasp {d:.0f} cm "
                f"lower." if v[0] < 0 else f"{instruction} In the previous attempt the gripper hit the object {d:.0f} "
                f"cm too low; grasp {d:.0f} cm higher.")
    if kind == "gain":
        return f"{instruction} In the previous attempt every vertical motion was scaled by {v[0]:.2f}."
    return instruction


def collect_pairs(roots):
    """All (parent attempt, successful expert continuation) pairs below the sweep roots."""
    out = []
    for root in roots:
        for info_p in sorted(Path(root).glob("**/episodes/*/*/*/snap_info.json")):
            parent = info_p.parent
            if "_attempt" in parent.name:
                continue
            info = json.loads(info_p.read_text())
            meta = _attrs(parent / "episode.h5") if (parent / "episode.h5").exists() else None
            if meta is None:
                continue
            for c in info.get("conts", []):
                e = c.get("expert") or {}
                if not e.get("success") or not e.get("dir") or not (Path(e["dir"]) / "episode.h5").exists():
                    continue
                kind, val, sign = param_of(meta)
                out.append(dict(parent=str(parent), cont=e["dir"], k=int(c["k"]), reset_chunk=int(c["reset_chunk"]),
                                close_chunk=int(info["close_chunk"]), task=str(meta.get("task_name")),
                                config=str(meta.get("task_config")), seed=int(meta.get("seed", -1)),
                                parent_success=bool(info.get("parent_success")), kind=kind, param=val, sign=sign,
                                instruction=str(meta.get("instruction", ""))))
    return out


def summary(pairs):
    by = {}
    for p in pairs:
        key = (p["config"], "succ" if p["parent_success"] else "fail", p["sign"])
        by[key] = by.get(key, 0) + 1
    return {" ".join(map(str, k)): v for k, v in sorted(by.items())}


@lru_cache(maxsize=512)
def _episode(path):
    """obs/pred canvases, latent index, chunk ids, planned actions of one episode (numpy)."""
    h5 = Path(path) / "episode.h5"
    with h5py.File(h5, "r") as f:
        d = dict(obs=canvas(f, "obs").astype(np.float32), li=f["model/obs/latent_index"][()],
                 chunk=f["model/obs/chunk_id"][()], planned=f["actions/planned"][()])
        if "model/pred" in f:
            d.update(pred=canvas(f, "pred").astype(np.float32), pli=f["model/pred/latent_index"][()])
    return d


def _raw_actions(ep, rows):
    """16 raw action channels per latent row (from the chunk that produced that latent)."""
    first = {}
    for r, c in enumerate(ep["chunk"]):
        first.setdefault(int(c), r)
    raw = np.zeros((16, len(rows), ep["planned"].shape[-1]), np.float32)
    for i, r in enumerate(rows):
        c = int(ep["chunk"][r])
        raw[:, i] = ep["planned"][c][:, min(r - first[c], ep["planned"].shape[2] - 1)]
    return raw


def close_latent(ep, close_chunk):
    rows = np.where(ep["chunk"] == close_chunk)[0]
    return int(ep["li"][rows[-1]]) if len(rows) else int(ep["li"][-1])


class SnapPairDataset:
    """Samples [context | past | target window] for train_adapter (same dict keys as FailFixDataset)."""

    def __init__(self, pairs, norm, text_cache, ctx="C2", max_past=8, target_frames=12, sampling="uniform",
                 p_irrelevant=0.1, cfg_prob=0.1, seed=0, phase_text=None, ctx_mode="prepend", gate=None):
        self.norm, self.tc, self.ctx = norm, text_cache, ctx
        self.ctx_mode = ctx_mode  # prepend (A1/A2) | separate (A3 cross-attention) | both (T4: prepend + pooled)
        self.gate = gate          # t1 | t2 | t5: per-frame aligned error of the last attempt (gate_err)
        self.max_past, self.T, self.sampling = max_past, target_frames, sampling
        self.rng, self.cfg_prob = random.Random(seed), cfg_prob
        self.fail = [p for p in pairs if not p["parent_success"]]
        self.succ = [p for p in pairs if p["parent_success"]]
        self.p_irr = p_irrelevant if self.succ else 0.0
        self.items = self.fail
        self.mode = "own"  # own | foreign | none (validation switches)
        self.by_cell = {}
        for p in self.fail:
            self.by_cell.setdefault((p["task"], p["config"]), []).append(p)
        if not self.fail:
            raise ValueError("no failed attempts with a successful expert continuation")

    def __len__(self):
        return len(self.items)

    # ------------------------------------------------------------------ pieces
    def context_parts(self, src):
        """Context latents (C, F, H, W) of a source attempt (`src` = pair dict of the attempt whose context is shown)."""
        groups = CTX_GROUPS[self.ctx]
        if not groups:
            return []
        ep = _episode(src["parent"])
        L = close_latent(ep, src["close_chunk"])
        rows = np.where((ep["li"] >= L - WIN[0]) & (ep["li"] <= L + WIN[1]))[0]
        if "pred" in ep:
            pidx = {int(k): i for i, k in enumerate(ep["pli"])}
            rows = np.array([r for r in rows if int(ep["li"][r]) in pidx and int(ep["li"][r]) > 0])
        if not len(rows):
            return []
        obs = ep["obs"][:, rows]
        pred = ep["pred"][:, [pidx[int(ep["li"][r])] for r in rows]] if "pred" in ep else np.zeros_like(obs)
        parts = []
        for g in groups:
            lat = {"obs": obs, "pred": pred, "diff": obs - pred}.get(g)
            if g == "diffmag":  # scalar error per frame, no direction
                mag = np.sqrt(((obs - pred) ** 2).mean(axis=(0, 2, 3)))
                lat = np.broadcast_to(mag[None, :, None, None], obs.shape).astype(np.float32).copy()
            acts, mask = self.norm(_raw_actions(ep, rows))
            if self.ctx == "C5":  # keep only the gripper channels of the executed actions
                raw = _raw_actions(ep, rows)
                keep = np.zeros_like(raw)
                keep[[7, 15]] = raw[[7, 15]]
                acts, mask = self.norm(keep)
                acts = acts * 0 + acts  # normalised; non-gripper channels sit at their normalised zero
            else:
                acts = np.zeros_like(acts)
            parts.append(dict(latents=lat, actions=acts[..., None], actions_mask=mask[..., None],
                              w=np.zeros(lat.shape[1], np.float32)))
        return parts

    def text(self, src, instruction):
        t = CTX_TEXT.get(self.ctx)
        if t == "oracle":
            return oracle_text(_attrs(Path(src["parent"]) / "episode.h5"), instruction)
        if t == "phase":
            ph = Path(src["parent"]) / "phases.json"
            if not ph.exists():
                from fawam.phases import phases_of
                ph.write_text(json.dumps(phases_of(Path(src["parent"]) / "episode.h5")))
            from fawam.phases import latent_phases
            lp = latent_phases(json.loads(ph.read_text()))
            ep = _episode(src["parent"])
            L = close_latent(ep, src["close_chunk"])
            seq = [lp.get(l) for l in range(L - WIN[0], L + WIN[1] + 1) if lp.get(l)]
            dedup = [p for i, p in enumerate(seq) if i == 0 or p != seq[i - 1]]
            return f"{instruction} Previous attempt, phases around the failure: {', '.join(dedup)}."
        return instruction

    def past_and_target(self, p):
        par, con = _episode(p["parent"]), _episode(p["cont"])
        prow = np.where(par["chunk"] < p["reset_chunk"])[0]
        crow = np.arange(len(con["li"]))
        # window in the continuation: uniform / 50 % starting at the reset (covers the failure time) / weighted
        n_c = len(crow)
        T = min(self.T, n_c)
        if self.sampling == "win50" and self.rng.random() < 0.5:
            s = 0
        else:
            s = self.rng.randrange(0, max(1, n_c - T + 1))
        past_c = crow[:s]  # continuation frames before the window are past too
        tgt = crow[s:s + T]
        pl, pa = par["obs"][:, prow], _raw_actions(par, prow)
        if len(past_c):
            pl = np.concatenate([pl, con["obs"][:, past_c]], axis=1)
            pa = np.concatenate([pa, _raw_actions(con, past_c)], axis=1)
        if pl.shape[1] > self.max_past:
            pl, pa = pl[:, -self.max_past:], pa[:, :, -self.max_past:]
        tl, ta = con["obs"][:, tgt], _raw_actions(con, tgt)
        w = np.ones(len(tgt), np.float32)
        if s == 0:
            w[0] = 0.0  # frame 0 of the continuation = the observation at the reset (conditioning frame)
        reset_li = int(par["li"][prow[-1]]) if len(prow) else 0  # continuation frame 0 = parent latent reset_li
        if self.sampling == "w4":  # loss x4 on the frames around the original failure time
            L = close_latent(par, p["close_chunk"])
            for i, r in enumerate(tgt):
                if abs(reset_li + int(con["li"][r]) - L) <= 3:
                    w[i] *= 4.0
        self._last = dict(reset_li=reset_li, tgt=tgt, con=con, past_n=pl.shape[1])
        return (pl, pa), (tl, ta, w)

    def gate_err(self, src, n_ctx, n_past):
        """T1/T2/T5: normalised error |obs - pred| of the attempt `src` for every frame of the sequence; -1 for
        context / past frames (adapter fully on there). t1/t2: time-aligned (parent latent = reset + continuation
        index); t5: the parent frame nearest (cosine, latent space) to the previous observed continuation frame, +1."""
        ep = _episode(src["parent"])
        e = np.zeros(int(ep["li"].max()) + 1, np.float32)
        if "pred" in ep:
            pidx = {int(k): i for i, k in enumerate(ep["pli"])}
            for r, k in enumerate(ep["li"]):
                if int(k) in pidx and int(k) > 0:
                    e[int(k)] = float(np.sqrt(((ep["obs"][:, r] - ep["pred"][:, pidx[int(k)]]) ** 2).mean()))
        e = e / (e.max() + 1e-8)
        L = self._last
        out = [-1.0] * (n_ctx + n_past)
        if self.gate == "t5":
            P = ep["obs"].reshape(ep["obs"].shape[0], ep["obs"].shape[1], -1).transpose(1, 0, 2).reshape(ep["obs"].shape[1], -1)
            P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
            for j, r in enumerate(L["tgt"]):
                q = L["con"]["obs"][:, max(0, int(r) - 1)].reshape(-1)
                nn_i = int(np.argmax(P @ (q / (np.linalg.norm(q) + 1e-8))))
                li = int(ep["li"][min(nn_i + (1 if r > 0 else 0), len(ep["li"]) - 1)])
                out.append(float(e[min(li, len(e) - 1)]))
        else:
            for r in L["tgt"]:
                out.append(float(e[min(L["reset_li"] + int(L["con"]["li"][r]), len(e) - 1)]))
        return np.array(out, np.float32), float(np.median(e[1:])) if len(e) > 1 else 0.5

    def foreign(self, p):
        cands = [q for q in self.by_cell.get((p["task"], p["config"]), []) if q["sign"] == -p["sign"] and q["sign"]]
        return self.rng.choice(cands) if cands else None

    # ------------------------------------------------------------------ sample
    def __getitem__(self, idx):
        irr = self.mode == "own" and self.p_irr and self.rng.random() < self.p_irr
        p = self.rng.choice(self.succ) if irr else self.items[idx % len(self.items)]
        (pl, pa), (tl, ta, w) = self.past_and_target(p)  # first rng use = the window (same in every val mode)
        src, kind = p, ("irrelevant" if irr else "own")  # irrelevant: context = the successful attempt itself
        if self.mode == "foreign":
            src, kind = (self.foreign(p) or p), "foreign"
        ctx = [] if self.mode == "none" else self.context_parts(src)
        sep = None
        if self.ctx_mode in ("separate", "both") and ctx:
            sep = np.concatenate([x["latents"] for x in ctx], axis=1)
            if self.ctx_mode == "separate":
                ctx = []
        pa_n, pm = self.norm(pa)
        ta_n, tm = self.norm(ta)
        pa_n[:, 0] = 0.0 if not ctx else pa_n[:, 0]
        seq = ctx + [dict(latents=pl, actions=pa_n[..., None], actions_mask=pm[..., None],
                          w=np.zeros(pl.shape[1], np.float32)),
                     dict(latents=tl, actions=ta_n[..., None], actions_mask=tm[..., None], w=w)]
        seq = [x for x in seq if x["latents"].shape[1]]
        cat = {k: np.concatenate([x[k] for x in seq], axis=1) for k in ("latents", "actions", "actions_mask")}
        wv = np.concatenate([x["w"] for x in seq])
        ctx_frames = sum(x["latents"].shape[1] for x in ctx)
        instr = p["instruction"]
        text = instr if self.mode == "none" else self.text(src, instr)
        if self.mode == "own" and self.rng.random() < self.cfg_prob:
            text = ""
        extra = {}
        if sep is not None:
            extra["ctx_latents"] = torch.from_numpy(np.ascontiguousarray(sep))
        if self.gate:
            ge, thr = self.gate_err(src, ctx_frames, pl.shape[1])
            extra.update(gate_err=torch.from_numpy(ge), gate_thr=torch.tensor(thr))
        return dict(**extra, latents=torch.from_numpy(np.ascontiguousarray(cat["latents"])),
                    actions=torch.from_numpy(np.ascontiguousarray(cat["actions"])),
                    actions_mask=torch.from_numpy(np.ascontiguousarray(cat["actions_mask"])),
                    text_emb=self.tc.get(text), text=text, context_frames=torch.tensor(ctx_frames),
                    weight_video=torch.from_numpy(wv), weight_action=torch.zeros(len(wv)),
                    actions_neg=torch.from_numpy(np.ascontiguousarray(cat["actions"])),
                    window_mask=torch.zeros(len(wv)), relabeled=torch.tensor(False), kind=kind,
                    error_vec=torch.zeros(32), has_error=torch.tensor(False))

    def all_texts(self):
        """Every prompt this dataset can produce (for the text-embedding cache)."""
        out = {""}
        for p in self.fail + self.succ:
            out.add(p["instruction"])
            if CTX_TEXT.get(self.ctx):
                out.add(self.text(p, p["instruction"]))
        return out


def close_from_h5(ep_dir, thr=0.8):
    """(close chunk, close latent) of a recorded attempt: first commanded gripper drop below thr (sim record) -> the
    executed chunk at that env step -> the last observed latent of that chunk. Fallback: chunk with the largest gap."""
    with h5py.File(Path(ep_dir) / "episode.h5", "r") as f:
        st, gl, gr = f["sim/env_step"][()], f["sim/left_gripper"][()], f["sim/right_gripper"][()]
        g = np.minimum(gl, gr)
        hit = np.where((g[:-1] >= thr) & (g[1:] < thr))[0]
        ech, est = f["actions/executed_chunk_id"][()], f["actions/executed_env_step"][()]
        li, ch = f["model/obs/latent_index"][()], f["model/obs/chunk_id"][()]
        if len(hit):
            step = st[hit[0] + 1]
            j = np.where(est >= step)[0]
            c = int(ech[j[0]]) if len(j) else int(ech[-1])
        else:
            gaps = [f[f"fawam/gap/{cam}"][()] for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist")
                    if f"fawam/gap/{cam}" in f]
            c = int(np.nanargmax(np.nanmax(np.stack(gaps), 0))) if gaps else int(ch.max())
    rows = np.where(ch == c)[0]
    return c, int(li[rows].max()) if len(rows) else int(li.max())


def rewind_spec(ctx, ctx_src, L_ctx, parent, reset_chunk, max_past=8, gap=1, ctx_mode="prepend"):
    """Server prefill spec for the p21 retry: [context of ctx_src around L_ctx | own past (chunks < reset)], the
    same layout as the training sequence (context groups contiguous, `gap` RoPE frames, past contiguous with the
    episode). Context latents carry zero action tokens (C5: gripper channels only)."""
    groups = CTX_GROUPS[ctx]
    items = []
    for i, g in enumerate(groups):
        items.append(dict(path=str(ctx_src), groups=[g], segment={"li": [L_ctx - WIN[0], L_ctx + WIN[1]], "min_li": 1},
                          manip="grip_only" if ctx == "C5" else "no_actions", role="failure",
                          gap=gap if i == len(groups) - 1 else 0,
                          # A3: context only through the gated cross-attention (not in the KV cache); T4: also pooled
                          xattn=ctx_mode == "separate", pool=ctx_mode == "both"))
    items.append(dict(path=str(parent), groups=["obs"], segment={"chunk_lt": int(reset_chunk), "last": int(max_past)},
                      role="other", gap=0))
    return dict(items=items, gap=gap, position="sequential", attn_scale=1.0)


def reset_offset(parent, reset_chunk):
    """Parent latent index of the retry's frame 0 (= the last observed latent before the reset point)."""
    with h5py.File(Path(parent) / "episode.h5", "r") as f:
        li, ch = f["model/obs/latent_index"][()], f["model/obs/chunk_id"][()]
    rows = np.where(ch < int(reset_chunk))[0]
    return int(li[rows].max()) if len(rows) else 0


def prompt_for(ctx, ctx_src, L_ctx, instruction):
    """Text channel of the p21 variant at test time (C6 phases / C7, C8 oracle text; else the instruction)."""
    t = CTX_TEXT.get(ctx)
    if not t:
        return instruction
    if t == "oracle":
        return oracle_text(_attrs(Path(ctx_src) / "episode.h5"), instruction)
    from fawam.phases import latent_phases, phases_of
    lp = latent_phases(phases_of(Path(ctx_src) / "episode.h5"))
    seq = [lp.get(l) for l in range(L_ctx - WIN[0], L_ctx + WIN[1] + 1) if lp.get(l)]
    dedup = [q for i, q in enumerate(seq) if i == 0 or q != seq[i - 1]]
    return f"{instruction} Previous attempt, phases around the failure: {', '.join(dedup)}."


def foreign_failure(bank_roots, sign, task, seed):
    """A failed attempt of the same task in the bank with the OPPOSITE sign of the hidden parameter."""
    cands = []
    for root in bank_roots:
        for h5 in sorted(Path(root).glob(f"**/episodes/fail/{task}/*/episode.h5")):
            if "_attempt" in h5.parent.name:
                continue
            _, _, sg = param_of(_attrs(h5))
            if sg and sg == -sign:
                cands.append(h5.parent)
    return cands[(int(seed) * 7919) % len(cands)] if cands else None


def split_by_seed(pairs, val_frac=0.1, seed=0):
    rng = random.Random(seed)
    keys = sorted({(p["task"], p["seed"] % 100000) for p in pairs})
    val = set(rng.sample(keys, max(1, int(len(keys) * val_frac)))) if len(keys) > 1 else set()
    return [p for p in pairs if (p["task"], p["seed"] % 100000) not in val], \
        [p for p in pairs if (p["task"], p["seed"] % 100000) in val]


if __name__ == "__main__":
    import sys
    ps = collect_pairs(sys.argv[1:])
    print(len(ps), "pairs")
    print(json.dumps(summary(ps), indent=1))
