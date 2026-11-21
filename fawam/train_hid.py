"""M3 (residual action head) on the hid pairs, with the training gate (plan 2026-11-18, sections 4-6).

The frozen LingBot-VA predicts the flow velocity of the target actions (expert fix in the believed frame); a small
context encoder reads the FAILED attempt's real signals (hid_ctx: real − imagined wrist window, proprio at the close,
commanded EE trajectory) and a head adds a constant velocity offset Δv to every action token. With the flow target
noise − x0, a constant shift of the actions by d is exactly Δv = −d, so the head can express "move the whole
trajectory by −o" -- the correction a hidden translation offset needs. Gradients never pass through the transformer:
one no-grad forward per step serves every configuration, so all context variants (C2, C3, C4, C5) × (with/without
+aux) train side by side in one job.

Per step and configuration: loss = 0.6 L(own context) + 0.3 L(shuf: context of another pair, same target)
+ 0.1 L(no context: learned null vector) + λ_aux · MSE(aux head(context), o) for +aux.
Gate on validation pairs (held-out seeds; --val_roots for held-out tasks): L_ctx < 0.9 L_noctx, L_shuf >= L_noctx,
corr(Δv_z, o_z) > 0.5, aux R² > 0.5 (+aux only). Written to <out>/<config>/gate.json.

  python -m fawam.train_hid --roots results/sweeps/p10_hid_data/base/hid_trans --out artifacts/hid_m3 --steps 1500
"""
import argparse
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from fawam import hid_ctx
from fawam.latent_data import ActionNorm, load_episode
from fawam.paths import add_lingbot_to_path
from fawam.text_cache import TextCache

add_lingbot_to_path()
from configs import VA_CONFIGS  # noqa: E402
from utils import FlowMatchScheduler  # noqa: E402

from fawam.train_adapter import add_noise, flow_losses, load_frozen_transformer  # noqa: E402

CTX = ("C2", "C3", "C4", "C5")


# ------------------------------------------------------------------------------------------------ data
def pairs(roots, tasks=None):
    """(attempt-1 policy episode, expert fix episode, o) on the same (task, config, seed); fix must have succeeded."""
    att, fix = {}, {}
    for root in roots:
        for h5p in Path(root).glob("episodes/*/*/*/episode.h5"):
            with h5py.File(h5p, "r") as f:
                a = f.attrs
                key = (str(a["task_name"]), str(a["task_config"]), int(a["seed"]))
                if tasks and key[0] not in tasks:
                    continue
                if a.get("driver") == "policy" and str(a.get("attempt")) == "1":
                    att[key] = h5p.parent
                elif a.get("driver") == "expert_replay" and bool(a.get("success")):
                    fix[key] = h5p.parent
    out = []
    for k in sorted(att):
        if k in fix:
            lab = hid_ctx.labels(att[k])
            if lab is not None:
                out.append(dict(key=k, att=att[k], fix=fix[k], o=lab[0]))
    return out


def ctx_tensor(feats, kind, device):
    """Context features of one attempt for configuration kind -> dict of tensors (batch 1)."""
    t = lambda x: torch.from_numpy(np.asarray(x, np.float32))[None].to(device)
    if kind == "C2":
        return dict(win=t(feats["win_diff"]))
    if kind == "C3":
        return dict(win=t(feats["win_diff"]), close=t(feats["close"]))
    if kind == "C4":
        return dict(traj=t(feats["traj"]), close=t(feats["close"]))
    if kind == "C5":
        return dict(win=t(feats["win_real"]), close=t(feats["close"]))
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------------ model
class CtxEncoder(nn.Module):
    """Perceiver: K learned queries cross-attend to the window patches (+ proprio tokens) -> K tokens, pooled."""

    def __init__(self, K=16, d=256, heads=4, layers=2):
        super().__init__()
        self.win_in = nn.Linear(48, d)
        self.win_pos = nn.Parameter(torch.randn(7 * 2 * 80, d) * 0.02)
        self.close_in = nn.Sequential(nn.Linear(12, d), nn.GELU(), nn.Linear(d, d))
        self.traj_in = nn.Linear(8, d)
        self.traj_pos = nn.Parameter(torch.randn(16, d) * 0.02)
        self.q = nn.Parameter(torch.randn(K, d) * 0.02)
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                                  for _ in range(layers)])
        self.norm_kv = nn.LayerNorm(d)
        self.null = nn.Parameter(torch.zeros(d))  # "no context"

    def forward(self, c):
        toks = []
        if "win" in c:  # (B, 7, 2, 48, 8, 10) -> (B, 1120, 48)
            w = c["win"].permute(0, 1, 2, 4, 5, 3).reshape(c["win"].shape[0], -1, 48)
            toks.append(self.win_in(w) + self.win_pos)
        if "traj" in c:
            toks.append(self.traj_in(c["traj"]) + self.traj_pos)
        if "close" in c:
            toks.append(self.close_in(c["close"])[:, None])
        kv = self.norm_kv(torch.cat(toks, 1))
        q = self.q[None].expand(kv.shape[0], -1, -1)
        for att, mlp in zip(self.attn, self.mlp):
            q = q + att(q, kv, kv, need_weights=False)[0]
            q = q + mlp(q)
        return q, q.mean(1)


class ResidualHead(nn.Module):
    """pooled context -> constant velocity offset on the 30 action channels (zero at init); aux head -> o."""

    def __init__(self, d=256, action_dim=30):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, action_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.aux = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))

    def forward(self, pooled):
        return self.net(pooled)


class HidM3(nn.Module):
    def __init__(self, kind, K=16, d=256):
        super().__init__()
        self.kind, self.enc, self.head = kind, CtxEncoder(K, d), ResidualHead(d)

    def delta(self, c):
        """c: context dict or None (no context) -> (Δv (B, 30), pooled)."""
        if c is None:
            pooled = self.enc.null[None]
        else:
            _, pooled = self.enc(c)
        return self.head(pooled), pooled


# ------------------------------------------------------------------------------------------------ train
def make_input(tf_cfg, s, device, patch, sched_v, sched_a, rng, noise_seed=None):
    lat = s["latents"][None].to(device)
    act = s["actions"][None].to(device)
    am = s["actions_mask"][None].to(device)
    g = torch.Generator(device=device)
    if noise_seed is not None:
        g.manual_seed(noise_seed)
    nl = torch.randn(lat.shape, generator=g, device=device) if noise_seed is not None else None
    na = torch.randn(act.shape, generator=g, device=device) if noise_seed is not None else None
    ld = add_noise(lat, sched_v, device, patch, noisy_cond_prob=0.5 if noise_seed is None else 0.0, noise=nl)
    ad = add_noise(act, sched_a, device, patch, action_mask=am, action_mode=True, noise=na)
    ld["text_emb"] = s["text_emb"][None].to(device, torch.bfloat16)
    ad["text_emb"] = ld["text_emb"]
    ad["actions_mask"] = am
    return dict(latent_dict=ld, action_dict=ad, chunk_size=int(rng.integers(1, 5)) if noise_seed is None else 2,
                window_size=int(rng.integers(4, 65)) if noise_seed is None else 32)


def action_loss(inp, pred, dv, patch, sched_v, sched_a):
    lat_pred, act_pred = pred
    return flow_losses(inp, (lat_pred, act_pred.float() + dv[:, None, :].float()), patch, sched_v, sched_a)[1]


def sample(item, norm, tcache, max_frames, rng):
    ep = item.get("_ep")
    if ep is None:
        ep = item["_ep"] = load_episode(item["fix"], norm)
    F = ep["latents"].shape[1]
    s = 0 if F <= max_frames or rng.random() < 0.5 else 2 * rng.integers(1, max(2, (F - max_frames) // 2 + 1))
    e = min(F, s + max_frames)
    w = {k: (v[:, s:e] if isinstance(v, np.ndarray) else v) for k, v in ep.items()}
    return dict(latents=torch.from_numpy(w["latents"]), actions=torch.from_numpy(w["actions"]),
                actions_mask=torch.from_numpy(w["actions_mask"]), text_emb=tcache.get(ep["text"]))


@torch.no_grad()
def gate(models, val, feats, norm, tcache, tf, a, device, patch, sched_v, sched_a, z_ch):
    rng = np.random.default_rng(123)
    res = {k: dict(own=[], null=[], shuf=[], dz=[], oz=[], aux=[], o=[]) for k in models}
    for i, it in enumerate(val):
        s = sample(it, norm, tcache, a.max_frames, rng)
        inp = make_input(None, s, device, patch, sched_v, sched_a, rng, noise_seed=1000 + i)
        pred = tf(inp, train_mode=True)
        other = val[(i + 1 + int(rng.integers(len(val) - 1))) % len(val)] if len(val) > 1 else it
        for k, m in models.items():
            kind = m.kind
            d_own, pooled = m.delta(ctx_tensor(feats(it["att"]), kind, device))
            d_null, _ = m.delta(None)
            d_shuf, _ = m.delta(ctx_tensor(feats(other["att"]), kind, device))
            r = res[k]
            r["own"].append(float(action_loss(inp, pred, d_own, patch, sched_v, sched_a)))
            r["null"].append(float(action_loss(inp, pred, d_null, patch, sched_v, sched_a)))
            r["shuf"].append(float(action_loss(inp, pred, d_shuf, patch, sched_v, sched_a)))
            r["dz"].append(float((d_own - d_null)[0, z_ch].mean()))
            r["oz"].append(float(it["o"][2]))
            r["aux"].append(m.head.aux(pooled)[0].float().cpu().numpy() / 10.0)
            r["o"].append(it["o"])
    out = {}
    for k, r in res.items():
        L = {x: float(np.mean(r[x])) for x in ("own", "null", "shuf")}
        corr = float(np.corrcoef(r["dz"], r["oz"])[0, 1]) if np.std(r["dz"]) > 0 else 0.0
        A, O = np.array(r["aux"]), np.array(r["o"])
        aux_r2 = float(1 - ((A - O) ** 2).sum() / ((O - O.mean(0)) ** 2).sum())
        g = dict(L_ctx=L["own"], L_noctx=L["null"], L_shuf=L["shuf"], corr_z=corr, aux_R2=aux_r2, n=len(val),
                 c1=L["own"] < 0.9 * L["null"], c2=L["shuf"] >= L["null"], c3=corr > 0.5,
                 c4=(aux_r2 > 0.5) if models[k].aux_w > 0 else None)
        g["passed"] = bool(g["c1"] and g["c2"] and g["c3"] and (g["c4"] is not False))
        out[k] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--val_roots", nargs="*", help="held-out task pairs for the gate (else held-out seeds)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--ctx", nargs="+", default=list(CTX))
    ap.add_argument("--aux", nargs="+", type=float, default=[0.0, 0.3])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_frames", type=int, default=20)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_hours", type=float, default=0)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--text_cache", default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    prng = random.Random(a.seed)
    device = torch.device("cuda")
    cfg = VA_CONFIGS["robotwin_train"]
    norm = ActionNorm(cfg)
    tcache = TextCache(a.text_cache) if a.text_cache else TextCache()
    P = pairs(a.roots, a.tasks)
    if a.val_roots:
        tr, val = P, pairs(a.val_roots)
    else:
        seeds = sorted({p["key"] for p in P})
        vs = set(prng.sample(seeds, max(1, int(len(seeds) * a.val_frac))))
        tr, val = [p for p in P if p["key"] not in vs], [p for p in P if p["key"] in vs]
    print(f"{len(tr)} train / {len(val)} val pairs", flush=True)
    feats = lambda ep: hid_ctx.extract(ep)
    for p in tr + val:
        feats(p["att"])  # cache the context features once
    inv = np.array(cfg.inverse_used_action_channel_ids)
    used = set(cfg.used_action_channel_ids)
    z_ch = [j for j in range(len(inv)) if j in used and int(inv[j]) in (2, 10)]
    print("z channels", z_ch, flush=True)

    model_path = a.model_path or str(Path(__file__).resolve().parents[1] / "lingbot-va/checkpoints/lingbot-va-posttrain-robotwin")
    tf = load_frozen_transformer(model_path, device)
    patch = tuple(cfg.patch_size)
    sched_v = FlowMatchScheduler(shift=cfg.snr_shift, sigma_min=0.0, extra_one_step=True)
    sched_v.set_timesteps(1000, training=True)
    sched_a = FlowMatchScheduler(shift=cfg.action_snr_shift, sigma_min=0.0, extra_one_step=True)
    sched_a.set_timesteps(1000, training=True)

    models = {}
    for c in a.ctx:
        for w in a.aux:
            k = f"{c}_M3" + (f"_aux{w:g}" if w > 0 else "")
            m = HidM3(c, a.K).to(device)
            m.aux_w = w
            models[k] = m
    opts = {k: torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4) for k, m in models.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.jsonl", "a")
    t0 = time.time()
    gates = {}
    for step in range(1, a.steps + 1):
        it = tr[int(rng.integers(len(tr)))]
        other = tr[int(rng.integers(len(tr)))]
        s = sample(it, norm, tcache, a.max_frames, rng)
        inp = make_input(None, s, device, patch, sched_v, sched_a, rng)
        with torch.no_grad():
            pred = tf(inp, train_mode=True)
        pred = (pred[0].detach(), pred[1].detach())
        rec = dict(step=step)
        for k, m in models.items():
            d_own, pooled = m.delta(ctx_tensor(feats(it["att"]), m.kind, device))
            d_shuf, _ = m.delta(ctx_tensor(feats(other["att"]), m.kind, device))
            d_null, _ = m.delta(None)
            l_own = action_loss(inp, pred, d_own, patch, sched_v, sched_a)
            loss = 0.6 * l_own + 0.3 * action_loss(inp, pred, d_shuf, patch, sched_v, sched_a) + \
                0.1 * action_loss(inp, pred, d_null, patch, sched_v, sched_a)
            if m.aux_w > 0:
                o = torch.from_numpy(it["o"] * 10.0).to(device)[None]
                loss = loss + m.aux_w * ((m.head.aux(pooled) - o) ** 2).mean()
            opts[k].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opts[k].step()
            rec[k] = round(float(l_own), 4)
        if step % 50 == 0 or step == 1:
            rec["s"] = round(time.time() - t0, 1)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        last = step == a.steps or (a.max_hours and time.time() - t0 > a.max_hours * 3600)
        if val and (step % a.val_every == 0 or last):
            for m in models.values():
                m.eval()
            gates = gate(models, val, feats, norm, tcache, tf, a, device, patch, sched_v, sched_a, z_ch)
            for m in models.values():
                m.train()
            print(json.dumps(dict(step=step, gate=gates)), flush=True)
            log.write(json.dumps(dict(step=step, gate=gates)) + "\n")
            log.flush()
        if last:
            break
    for k, m in models.items():
        d = out / k
        d.mkdir(exist_ok=True)
        torch.save(m.state_dict(), d / "hid_m3.pt")
        (d / "config.json").write_text(json.dumps(dict(kind=m.kind, K=a.K, aux_w=m.aux_w, mechanism="M3")))
        if k in gates:
            (d / "gate.json").write_text(json.dumps(dict(gates[k], step=step), indent=1))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
