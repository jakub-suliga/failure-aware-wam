"""Few-shot residual adaptation of the frozen LingBot-VA (docs/EXPERIMENTS.md, D; paper: dual-site adaptation).

Stage `inside`: Adapter+ (or LoRA) inside the 30 blocks, trained with LingBot-VA's own flow-matching losses on
recorded episodes (video loss and/or action loss; teacher-forced, block-causal FlexAttention as in upstream training).
Stage `outside`: the external residual MLP on the predicted clean latent, trained with an L1 loss on
(predicted, observed) latent pairs. As in the paper it is trained *after* the inside adapters, on predictions of the
adapted model (`--pairs` from fawam.replay); without `--pairs` it uses the predictions recorded in the episodes
(= base model predictions; the "parallel" ablation).

Examples (lingbot env, one GPU, from the repo root):
  # D1: dual-site on 10 expert replays at -5 cm, video + action loss
  python -m fawam.train_adapter inside --episodes data/expert/table_h_m05 --limit 10 --out artifacts/adapters/d1_m05_n10
  python -m fawam.replay --adapter artifacts/adapters/d1_m05_n10 --episodes data/expert/table_h_m05 --limit 10 \
      --out artifacts/adapters/d1_m05_n10/pairs.npz
  python -m fawam.train_adapter outside --init artifacts/adapters/d1_m05_n10 \
      --pairs artifacts/adapters/d1_m05_n10/pairs.npz --out artifacts/adapters/d1_m05_n10
"""
import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fawam import adapters as fa
from fawam.latent_data import ActionNorm, EpisodeDataset, FailFixDataset, build_pairs, find_episodes
from fawam.paths import LINGBOT, ROOT, add_lingbot_to_path
from fawam.text_cache import TextCache

add_lingbot_to_path()
from configs import VA_CONFIGS  # noqa: E402
from utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id, sample_timestep_id  # noqa: E402

DEFAULT_MODEL = LINGBOT / "checkpoints" / "lingbot-va-posttrain-robotwin"


# ------------------------------------------------------------------------------------------ upstream-equivalent bits

def add_noise(latent, sched, device, patch, action_mask=None, action_mode=False, noisy_cond_prob=0.0,
              clean_frames=0, frame_gap=0, noise=None, tid=None):
    """wan_va/train.py Trainer._add_noise, plus: the first `clean_frames` frames (in-context demos) get clean
    conditioning, and frames from `clean_frames` on are shifted by `frame_gap` in RoPE time (as the server's prefill)."""
    B, C, Fr, H, W = latent.shape
    tid = sample_timestep_id(batch_size=Fr, num_train_timesteps=sched.num_train_timesteps) if tid is None else tid
    noise = torch.zeros_like(latent).normal_() if noise is None else noise
    timesteps = sched.timesteps[tid].to(device)
    noisy = sched.add_noise(latent, noise, timesteps, t_dim=2)
    targets = sched.training_target(latent, noise, timesteps)
    pf, ph, pw = (1, 1, 1) if action_mode else patch
    grid = get_mesh_id(Fr // pf, H // ph, W // pw, t=1 if action_mode else 0, f_w=1, f_shift=0,
                       action=action_mode).to(device)
    if frame_gap and clean_frames:
        grid[0] = torch.where(grid[0] >= clean_frames, grid[0] + frame_gap, grid[0])
    grid = grid[None].repeat(B, 1, 1)
    cond = latent
    if torch.rand(1).item() < noisy_cond_prob:
        ctid = sample_timestep_id(batch_size=Fr, min_timestep_bd=0.5, max_timestep_bd=1.0,
                                  num_train_timesteps=sched.num_train_timesteps)
        cond_t = sched.timesteps[ctid].to(device)
        if clean_frames:
            cond_t[:clean_frames] = 0
        cond = sched.add_noise(latent, torch.zeros_like(latent).normal_(), cond_t, t_dim=2)
    else:
        cond_t = torch.zeros_like(timesteps)
    if action_mask is not None:
        m = action_mask.float()
        noisy, targets, cond = noisy * m, targets * m, cond * m
    return dict(timesteps=timesteps[None].repeat(B, 1), noisy_latents=noisy, targets=targets, latent=cond,
                cond_timesteps=cond_t[None].repeat(B, 1), grid_id=grid, _noise=noise, _tid=tid)


def flow_losses(input_dict, pred, patch, sched_v, sched_a, frame_weight=None, frame_weight_action=None):
    """wan_va/train.py Trainer.compute_loss (frame-wise weighted MSE), with optional per-frame loss weights
    (video, action; the action weight defaults to the video weight)."""
    latent_pred, action_pred = pred
    tv, ta = input_dict["latent_dict"]["targets"], input_dict["action_dict"]["targets"]
    action_pred = action_pred.reshape(action_pred.shape[0], tv.shape[2], -1, action_pred.shape[-1])
    action_pred = action_pred.permute(0, 3, 1, 2)[..., None]  # b c f n 1
    latent_pred = data_seq_to_patch(patch, latent_pred, tv.shape[-3], tv.shape[-2], tv.shape[-1],
                                    batch_size=latent_pred.shape[0])
    B, Fr = input_dict["latent_dict"]["timesteps"].shape
    wv = sched_v.training_weight(input_dict["latent_dict"]["timesteps"].flatten()).reshape(B, Fr)
    wa = sched_a.training_weight(input_dict["action_dict"]["timesteps"].flatten()).reshape(B, Fr)
    fw = torch.ones(B, Fr, device=tv.device) if frame_weight is None else frame_weight[None].to(tv.device)
    fwa = fw if frame_weight_action is None else frame_weight_action[None].to(tv.device)

    lv = F.mse_loss(latent_pred.float(), tv.float(), reduction="none") * wv[:, None, :, None, None]
    lv = lv.permute(0, 2, 3, 4, 1).flatten(2).mean(2)  # B, F
    lv = (lv * fw).sum() / fw.sum().clamp(min=1)

    am = input_dict["action_dict"]["actions_mask"].float()
    la = F.mse_loss(action_pred.float(), ta.float(), reduction="none") * wa[:, None, :, None, None] * am
    la = la.permute(0, 2, 3, 4, 1).flatten(2).sum(2) / (am.permute(0, 2, 3, 4, 1).flatten(2).sum(2) + 1e-6)
    la = (la * fwa).sum() / fwa.sum().clamp(min=1)
    return lv, la


# ------------------------------------------------------------------------------------------ stages

def load_frozen_transformer(model_path, device):
    from distributed.fsdp import apply_ac
    from modules.utils import load_transformer
    tf = load_transformer(str(Path(model_path) / "transformer"), torch_dtype=torch.bfloat16, torch_device=device,
                          attn_mode="flex")
    tf.requires_grad_(False)
    tf.eval()
    apply_ac(tf)
    return tf


def train_inside(a):
    device = torch.device("cuda")
    cfg = VA_CONFIGS["robotwin_train"]
    eps = find_episodes(a.episodes, success=a.success, tasks=a.tasks, settings=a.settings, limit=a.limit,
                        seed=a.seed, drivers=a.drivers, limit_per_task=a.limit_per_task,
                        limit_per_cell=a.limit_per_cell)
    if not eps:
        raise SystemExit("no episodes found")
    ctx_pool = find_episodes(a.context_pool, success=True, tasks=a.tasks) if a.context_pool else None
    print(f"{len(eps)} training episodes")
    load_kw = dict(relabel=a.relabel, gap_beta=a.gap_beta, video_in_window=a.video_in_window,
                   gap_filter=a.gap_filter)
    if a.fixes:  # meta-ICL from failures: eps are the failed attempts, fixes the successes on the same seeds
        fixes = find_episodes(a.fixes, success=True, tasks=a.tasks)
        ds = FailFixDataset(eps, fixes, ActionNorm(cfg, a.action_clip), TextCache(a.text_cache), include_pred=a.include_pred,
                            p_text=a.p_text, max_frames=a.max_frames, seed=a.seed, cfg_prob=a.cfg_prob,
                            context_groups=a.context_groups,
                            window=dict(before=a.window_before, after=a.window_after) if a.context_window else None,
                            p_cross=a.p_cross, p_success=a.p_success, p_noctx=a.p_noctx, analysis_name=a.analysis_name,
                            text_only=a.text_only, keyframes=a.context_keyframes)
        from fawam.latent_data import split_pairs_by_seed
        tr, va = split_pairs_by_seed(ds, a.val_frac, a.seed) if a.val_frac > 0 else (ds.pairs, [])
        val_ds = None
        if va:
            import copy
            val_ds = copy.copy(ds)
            val_ds.pairs, val_ds.items = va, va
            val_ds.p_noctx = val_ds.p_success = val_ds.p_cross = 0.0  # validate on own-failure memory only
            val_ds.rng = __import__("random").Random(999)
        ds.pairs = ds.items = tr
        print(f"{len(tr)} train / {len(va)} val (failure, fix) pairs on identical seeds")
    else:
        val_ds = None
        ds = EpisodeDataset(eps, ActionNorm(cfg, a.action_clip), TextCache(a.text_cache), max_frames=a.max_frames,
                            context_k=a.context_k, context_pool=ctx_pool, seed=a.seed, cfg_prob=a.cfg_prob,
                            load_kw=load_kw, text_mode=a.text_mode)
        n_rel = sum(it["relabeled"] for it in ds.items)
        if a.relabel:
            print(f"{n_rel} of {len(ds.items)} episodes relabeled with their failure analysis")
    tf = load_frozen_transformer(a.model_path, device)
    if a.init:
        aset = fa.AdapterSet.load(a.init, device=device)
    else:
        acfg = fa.AdapterConfig(kind=a.kind, inside_streams=a.inside_streams, bottleneck=a.bottleneck,
                                lora_rank=a.lora_rank, lora_alpha=a.lora_alpha or float(a.lora_rank),
                                blocks=a.blocks, sites=a.sites, lora_targets=a.lora_targets,
                                adapter_learn_scale=not a.no_adapter_scale, use_adapterplus=a.use_adapterplus,
                                use_lora=a.use_lora, ia3=a.ia3, adaln=a.adaln, adapter_norm=not a.no_adapter_norm,
                                prompt_tokens=a.prompt_tokens, prompt_bank=a.prompt_bank, prompt_topk=a.prompt_topk,
                                ttt_dim=a.ttt_dim, ttt_eta=a.ttt_eta, ttt_streams=a.ttt_streams,
                                svf_blocks=a.svf_blocks, svf_targets=a.svf_targets, dense_blocks=a.dense_blocks,
                                dense_targets=a.dense_targets, outside_type=a.outside_type,
                                film_source=a.film_source, film_rank=a.film_rank, error_dim=a.error_dim)
        aset = fa.AdapterSet(acfg).to(device)
    fa.attach(tf, aset)
    params = aset.inside_parameters()
    for p in params:
        p.requires_grad_(True)
    print(f"trainable inside params: {sum(p.numel() for p in params) / 1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.weight_decay)
    base = None
    if a.train_base:  # also train the base WAM (WAM-TTT style meta-training; fawam/base_train.py)
        from fawam.base_train import BaseTrainer
        base = BaseTrainer(tf, lr=a.base_lr, weight_decay=a.weight_decay, blocks=a.base_blocks)
    sched_v = FlowMatchScheduler(shift=cfg.snr_shift, sigma_min=0.0, extra_one_step=True)
    sched_v.set_timesteps(1000, training=True)
    sched_a = FlowMatchScheduler(shift=cfg.action_snr_shift, sigma_min=0.0, extra_one_step=True)
    sched_a.set_timesteps(1000, training=True)
    patch = tuple(cfg.patch_size)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_inside.jsonl", "a")
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        s = ds[int(rng.integers(len(ds)))]
        lat = s["latents"][None].to(device)
        act = s["actions"][None].to(device)
        am = s["actions_mask"][None].to(device)
        ctx = int(s["context_frames"])
        ld = add_noise(lat, sched_v, device, patch, noisy_cond_prob=0.5, clean_frames=ctx, frame_gap=a.icl_gap if ctx else 0)
        ad = add_noise(act, sched_a, device, patch, action_mask=am, action_mode=True, clean_frames=ctx,
                       frame_gap=a.icl_gap if ctx else 0)
        ld["text_emb"] = s["text_emb"][None].to(device, torch.bfloat16)
        ad["text_emb"] = ld["text_emb"]
        ad["actions_mask"] = am
        window = 128 if ctx else int(rng.integers(4, 65))
        input_dict = dict(latent_dict=ld, action_dict=ad, chunk_size=int(rng.integers(1, 5)), window_size=window)
        fw, fwa = s["weight_video"].clone().float(), s["weight_action"].clone().float()
        if ctx:  # in-context frames are conditioning only
            fw[:ctx] = 0
            fwa[:ctx] = 0
        tf._fawam.ctx.ctx_frames = ctx  # TTT branches: the first ctx frames are the in-context memory
        tf._fawam.ctx.film_error = s["error_vec"].to(device) if bool(s.get("has_error", False)) else None
        pred = tf(input_dict, train_mode=True)
        lv, la = flow_losses(input_dict, pred, patch, sched_v, sched_a, fw, fwa)
        loss = a.lambda_video * lv + a.lambda_action * la
        lp = torch.zeros((), device=device)
        if a.pref_beta and bool(s.get("relabeled", False)) and s["window_mask"].sum() > 0:
            # FAR-style failure-contrastive preference (Diffusion-DPO form): relabeled actions preferred over the
            # executed failed ones on the failure window, relative to the frozen base model.
            wm = s["window_mask"].float()
            neg = s["actions_neg"][None].to(device)
            ad_neg = add_noise(neg, sched_a, device, patch, action_mask=am, action_mode=True, noise=ad["_noise"],
                               tid=ad["_tid"])
            ad_neg["text_emb"], ad_neg["actions_mask"] = ad["text_emb"], am
            in_neg = dict(latent_dict=ld, action_dict=ad_neg, chunk_size=input_dict["chunk_size"], window_size=window)
            pred_neg = tf(in_neg, train_mode=True)
            with torch.no_grad():
                tf._fawam.ctx.enabled = False
                ref_pos = tf(input_dict, train_mode=True)
                ref_neg = tf(in_neg, train_mode=True)
                tf._fawam.ctx.enabled = True
            e = lambda inp, pr: flow_losses(inp, pr, patch, sched_v, sched_a, wm, wm)[1]
            margin = (e(input_dict, pred) - e(input_dict, ref_pos)) - (e(in_neg, pred_neg) - e(in_neg, ref_neg))
            lp = -torch.nn.functional.logsigmoid(-a.pref_beta * margin)
            loss = loss + a.lambda_pref * lp
        if loss.requires_grad:  # FiLM-only variants: a sample without context/error input touches no trainable param
            (loss / a.grad_accum).backward()
        gn = torch.tensor(0.0)
        if step % a.grad_accum == 0:
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if base is not None:
                base.step(lr_scale=min(1.0, (step // a.grad_accum) / max(1, a.base_warmup)))
        if val_ds is not None and a.val_every and step % a.val_every == 0:
            vl = validate(tf, val_ds, a, sched_v, sched_a, patch, device)
            print(json.dumps(dict(step=step, val_action=vl[1], val_video=vl[0], val_action_noctx=vl[2])), flush=True)
            log.write(json.dumps(dict(step=step, val_action=vl[1], val_video=vl[0], val_action_noctx=vl[2])) + "\n")
            log.flush()
        if step % a.log_every == 0 or step == 1:
            rec = dict(step=step, loss=float(loss), video=float(lv), action=float(la), pref=float(lp), grad_norm=float(gn),
                       frames=int(lat.shape[2]), ctx=ctx, s=round(time.time() - t0, 1))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if a.save_every and step % a.save_every == 0:
            aset.save(out / f"step{step}")
        if a.max_hours and time.time() - t0 > a.max_hours * 3600:  # short SLURM limits start sooner: stop in time, save
            print(f"time budget of {a.max_hours} h reached at step {step}/{a.steps}", flush=True)
            (out / "stopped_early.json").write_text(json.dumps(dict(step=step, steps=a.steps)))
            break
    if a.guard_episodes:
        ratio = guard_ratio(tf, aset, a, cfg, sched_v, sched_a, patch, device)
        ok = ratio <= a.guard_tau
        (out / "guard.json").write_text(json.dumps(dict(loss_ratio=ratio, tau=a.guard_tau, accepted=ok)))
        print(f"guard: anchor loss ratio {ratio:.3f} (tau {a.guard_tau}) -> {'accepted' if ok else 'REJECTED'}")
    aset.save(out)
    if base is not None:
        base.save(out)
    (out / "train_args_inside.json").write_text(json.dumps({**vars(a), "episodes_used": [str(e) for e in eps]},
                                                           indent=2, default=str))
    print(f"saved {out}")


@torch.no_grad()
def validate(tf, ds, a, sched_v, sched_a, patch, device, n=24):
    """Flow loss on held-out (failure, fix) seeds: with the failure memory, and the same targets without memory
    (val_action_noctx). Meta-training works if the memory lowers the loss below the no-memory value."""
    out = []
    for use_ctx in (True, False):
        tv = ta = 0.0
        for i in range(n):
            torch.manual_seed(5000 + i)
            ds.rng.seed(5000 + i)
            ds.p_noctx = 0.0 if use_ctx else 1.0
            smp = ds[i]
            lat, act = smp["latents"][None].to(device), smp["actions"][None].to(device)
            am = smp["actions_mask"][None].to(device)
            ctx = int(smp["context_frames"])
            ld = add_noise(lat, sched_v, device, patch, clean_frames=ctx, frame_gap=a.icl_gap if ctx else 0)
            ad = add_noise(act, sched_a, device, patch, action_mask=am, action_mode=True, clean_frames=ctx,
                           frame_gap=a.icl_gap if ctx else 0)
            ld["text_emb"] = smp["text_emb"][None].to(device, torch.bfloat16)
            ad["text_emb"], ad["actions_mask"] = ld["text_emb"], am
            inp = dict(latent_dict=ld, action_dict=ad, chunk_size=2, window_size=128)
            fw = torch.ones(lat.shape[2])
            fw[:ctx] = 0
            tf._fawam.ctx.ctx_frames = ctx
            tf._fawam.ctx.film_error = smp["error_vec"].to(device) if bool(smp.get("has_error", False)) else None
            lv, la = flow_losses(inp, tf(inp, train_mode=True), patch, sched_v, sched_a, fw, fw)
            tv, ta = tv + float(lv), ta + float(la)
        out.append((tv / n, ta / n))
    ds.p_noctx = 0.0
    return out[0][0], out[0][1], out[1][1]


@torch.no_grad()
def guard_ratio(tf, aset, a, cfg, sched_v, sched_a, patch, device, n=16):
    """FailBank-style acceptance test: flow loss on held-out anchor episodes (own successes) with / without the
    adapter, same noise. ratio > tau means the update broke behaviour that worked."""
    eps = find_episodes(a.guard_episodes)
    ds = EpisodeDataset(eps, ActionNorm(cfg, a.action_clip), TextCache(a.text_cache), max_frames=a.max_frames,
                        seed=123, cfg_prob=0.0)
    tot = {True: 0.0, False: 0.0}
    for i in range(n):
        torch.manual_seed(1000 + i)
        smp = ds[i]
        lat, act = smp["latents"][None].to(device), smp["actions"][None].to(device)
        am = smp["actions_mask"][None].to(device)
        ld = add_noise(lat, sched_v, device, patch)
        ad = add_noise(act, sched_a, device, patch, action_mask=am, action_mode=True)
        ld["text_emb"] = smp["text_emb"][None].to(device, torch.bfloat16)
        ad["text_emb"], ad["actions_mask"] = ld["text_emb"], am
        inp = dict(latent_dict=ld, action_dict=ad, chunk_size=2, window_size=32)
        for enabled in (True, False):
            tf._fawam.ctx.enabled = enabled
            lv, la = flow_losses(inp, tf(inp, train_mode=True), patch, sched_v, sched_a)
            tot[enabled] += float(lv + la)
    tf._fawam.ctx.enabled = True
    return tot[True] / max(tot[False], 1e-8)


def train_outside(a):
    device = torch.device("cuda")
    if a.pairs:
        d = np.load(a.pairs)
        P, O = d["pred"], d["obs"]
    else:
        eps = find_episodes(a.episodes, success=a.success, tasks=a.tasks, settings=a.settings, limit=a.limit,
                            seed=a.seed, drivers=a.drivers, limit_per_task=a.limit_per_task,
                        limit_per_cell=a.limit_per_cell)
        P, O = build_pairs(eps)
    print(f"{P.shape[1]} (pred, obs) latent frames")
    P = torch.from_numpy(P).float().permute(1, 0, 2, 3)  # N, C, 24, 20
    O = torch.from_numpy(O).float().permute(1, 0, 2, 3)
    if a.init:
        aset = fa.AdapterSet.load(a.init, device=device)
        if aset.outside is None:
            aset.cfg.kind = "dual" if aset.cfg.has_inside else "outside"
            aset.outside = fa.ExternalResidual(aset.cfg.latent_channels, aset.cfg.patch,
                                               aset.cfg.outside_hidden).to(device)
    else:
        aset = fa.AdapterSet(fa.AdapterConfig(kind="outside", outside_type=a.outside_type)).to(device)
    params = aset.outside_parameters()
    opt = torch.optim.AdamW(params, lr=a.lr_outside, weight_decay=a.weight_decay)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_outside.jsonl", "a")
    rng = np.random.default_rng(a.seed)
    n = P.shape[0]
    base_l1 = float((P - O).abs().mean())
    for step in range(1, a.steps_outside + 1):
        idx = torch.from_numpy(rng.choice(n, size=min(a.batch_outside, n), replace=n < a.batch_outside))
        p, o = P[idx].to(device), O[idx].to(device)
        corr = aset.outside(p[:, :, None])[:, :, 0]
        loss = (corr - o).abs().mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % a.log_every == 0 or step == 1:
            rec = dict(step=step, l1=float(loss), l1_uncorrected=base_l1)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
    aset.save(out)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["inside", "outside"])
    ap.add_argument("--episodes", nargs="*", default=[])
    ap.add_argument("--success", type=lambda s: None if s == "any" else s == "true", default=None,
                    help="any | true | false")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--settings", nargs="*", help="task_config names to keep, e.g. table_h_m05")
    ap.add_argument("--drivers", nargs="*", help="policy | expert_replay")
    ap.add_argument("--limit_per_task", type=int, help="few-shot n per task (applied before --limit)")
    ap.add_argument("--limit_per_cell", type=int, help="at most n episodes per (task, task_config)")
    ap.add_argument("--limit", type=int, help="number of episodes (few-shot n)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_path", default=str(DEFAULT_MODEL))
    ap.add_argument("--text_cache", default=str(ROOT / "artifacts" / "text_emb"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", help="existing adapter dir to continue from / to add the outside residual to")
    ap.add_argument("--kind", default="dual", choices=["dual", "inside", "lora", "outside", "ia3", "adaln", "combo"])
    ap.add_argument("--sites", nargs="+", default=["attn", "block"], help="Adapter+ sites")
    ap.add_argument("--lora_targets", nargs="+", default=["attn"], help="attn | cross | ffn")
    ap.add_argument("--lora_alpha", type=float, default=None, help="default = rank")
    ap.add_argument("--no_adapter_scale", action="store_true", help="Houlsby-style Adapter (fixed scale 1)")
    ap.add_argument("--use_adapterplus", action="store_true", help="combo: Adapter+")
    ap.add_argument("--use_lora", action="store_true", help="combo: LoRA")
    ap.add_argument("--ia3", action="store_true", help="add (IA)^3")
    ap.add_argument("--adaln", action="store_true", help="add AdaLN-modulation tuning")
    ap.add_argument("--no_adapter_norm", action="store_true", help="Adapter without LayerNorm (plain residual MLP)")
    ap.add_argument("--prompt_tokens", type=int, default=0, help="VANE/T3VF prompt tokens per prompt (0 = off)")
    ap.add_argument("--prompt_bank", type=int, default=1)
    ap.add_argument("--prompt_topk", type=int, default=1)
    ap.add_argument("--ttt_dim", type=int, default=0, help="WAM-TTT fast-weight branch width (0 = off)")
    ap.add_argument("--ttt_eta", type=float, default=0.1)
    ap.add_argument("--ttt_streams", nargs="+", default=["video"])
    ap.add_argument("--svf_blocks", type=int, nargs="*", help="SALT singular-value scaling in these blocks")
    ap.add_argument("--svf_targets", nargs="+", default=["self_k"])
    ap.add_argument("--dense_blocks", type=int, nargs="*", help="full fine-tuning (dense deltas) of these blocks")
    ap.add_argument("--dense_targets", nargs="+", default=["attn"])
    ap.add_argument("--outside_type", default="mlp", choices=["mlp", "channel"])
    ap.add_argument("--film_source", default="none", choices=["none", "memory", "error", "text"])
    ap.add_argument("--film_rank", type=int, default=64)
    ap.add_argument("--error_dim", type=int, default=16, help="FiLM-error input: 16 = failure vector, 32 = + gap curve")
    ap.add_argument("--text_only", action="store_true", help="meta-ICL: analysis text only, no rollout in the context")
    ap.add_argument("--context_keyframes", type=int, default=0,
                    help="meta-ICL: only the k latents with the highest prediction error as memory (0 = off)")
    ap.add_argument("--inside_streams", nargs="+", default=["video", "action"])
    ap.add_argument("--blocks", type=int, nargs="*", help="block indices with adapters (default: all 30)")
    ap.add_argument("--bottleneck", type=int, default=128)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=4.343e-4)
    ap.add_argument("--train_base", action="store_true",
                    help="also train the base transformer (fp32 master weights on the GPU: needs an H200)")
    ap.add_argument("--base_lr", type=float, default=1e-5)
    ap.add_argument("--base_warmup", type=int, default=200, help="optimizer steps of linear base-lr warmup")
    ap.add_argument("--base_blocks", type=int, nargs="*", help="only these blocks of the base (default: everything)")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--lambda_video", type=float, default=1.0)
    ap.add_argument("--lambda_action", type=float, default=1.0)
    ap.add_argument("--max_frames", type=int, default=24)
    ap.add_argument("--cfg_prob", type=float, default=0.1)
    ap.add_argument("--context_k", type=int, default=0, help="E3: demos concatenated in front of each sample")
    ap.add_argument("--context_pool", nargs="*", help="E3: episode roots the context demos are drawn from")
    ap.add_argument("--icl_gap", type=int, default=1)
    # Learn-from-Failure (docs/METHOD.md)
    ap.add_argument("--relabel", action="store_true", help="apply analysis.json corrections to failed episodes")
    ap.add_argument("--action_clip", type=float, default=1.5, help="normalized action clip (upstream 1.5)")
    ap.add_argument("--gap_beta", type=float, default=0.0, help="up-weight frames with high prediction error")
    ap.add_argument("--video_in_window", default="real", choices=["real", "skip"])
    ap.add_argument("--text_mode", default="instruction", choices=["instruction", "correction"])
    ap.add_argument("--gap_filter", type=float, default=0.0, help="T3VF: train only on frames above this gap quantile")
    ap.add_argument("--pref_beta", type=float, default=0.0, help="FAR: failure-contrastive preference loss (0 = off)")
    ap.add_argument("--lambda_pref", type=float, default=1.0)
    ap.add_argument("--guard_episodes", nargs="*", help="FailBank guard: anchor episodes for the acceptance test")
    ap.add_argument("--guard_tau", type=float, default=1.1)
    ap.add_argument("--fixes", nargs="*", help="meta-ICL: successful episodes on the same seeds as --episodes")
    ap.add_argument("--analysis_name", default="analysis.json",
                    help="meta-ICL: failure analysis per episode (analysis.json = oracle, analysis_gap+vlm.json = VLM)")
    ap.add_argument("--include_pred", action="store_true", help="meta-ICL: imagined segment in the context")
    ap.add_argument("--context_groups", nargs="*", help="meta-ICL memory format: obs | pred | diff (overrides include_pred)")
    ap.add_argument("--context_window", action="store_true", help="meta-ICL: only the latents around the failure")
    ap.add_argument("--window_before", type=int, default=4)
    ap.add_argument("--window_after", type=int, default=2)
    ap.add_argument("--p_text", type=float, default=0.5, help="meta-ICL: prob. of the correction text in the prompt")
    ap.add_argument("--p_cross", type=float, default=0.2, help="meta-ICL: memory = failure of another seed")
    ap.add_argument("--p_success", type=float, default=0.1, help="meta-ICL: memory = another success")
    ap.add_argument("--p_noctx", type=float, default=0.15, help="meta-ICL: no memory")
    ap.add_argument("--val_frac", type=float, default=0.1, help="meta-ICL: held-out seeds")
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--steps_outside", type=int, default=300)
    ap.add_argument("--lr_outside", type=float, default=1.234e-3)
    ap.add_argument("--batch_outside", type=int, default=32)
    ap.add_argument("--pairs", help="npz with pred/obs canvases from fawam.replay")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--max_hours", type=float, default=0, help="stop the inside training after this time and save")
    a = ap.parse_args()
    if a.stage == "inside":
        train_inside(a)
    else:
        train_outside(a)


if __name__ == "__main__":
    main()
