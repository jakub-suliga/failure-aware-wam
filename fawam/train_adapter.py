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
from fawam.latent_data import ActionNorm, EpisodeDataset, build_pairs, find_episodes
from fawam.paths import LINGBOT, ROOT, add_lingbot_to_path
from fawam.text_cache import TextCache

add_lingbot_to_path()
from configs import VA_CONFIGS  # noqa: E402
from utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id, sample_timestep_id  # noqa: E402

DEFAULT_MODEL = LINGBOT / "checkpoints" / "lingbot-va-posttrain-robotwin"


# ------------------------------------------------------------------------------------------ upstream-equivalent bits

def add_noise(latent, sched, device, patch, action_mask=None, action_mode=False, noisy_cond_prob=0.0,
              clean_frames=0, frame_gap=0):
    """wan_va/train.py Trainer._add_noise, plus: the first `clean_frames` frames (in-context demos) get clean
    conditioning, and frames from `clean_frames` on are shifted by `frame_gap` in RoPE time (as the server's prefill)."""
    B, C, Fr, H, W = latent.shape
    tid = sample_timestep_id(batch_size=Fr, num_train_timesteps=sched.num_train_timesteps)
    noise = torch.zeros_like(latent).normal_()
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
                cond_timesteps=cond_t[None].repeat(B, 1), grid_id=grid)


def flow_losses(input_dict, pred, patch, sched_v, sched_a, frame_weight=None):
    """wan_va/train.py Trainer.compute_loss (frame-wise weighted MSE), with an optional per-frame loss weight."""
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

    lv = F.mse_loss(latent_pred.float(), tv.float(), reduction="none") * wv[:, None, :, None, None]
    lv = lv.permute(0, 2, 3, 4, 1).flatten(2).mean(2)  # B, F
    lv = (lv * fw).sum() / fw.sum().clamp(min=1)

    am = input_dict["action_dict"]["actions_mask"].float()
    la = F.mse_loss(action_pred.float(), ta.float(), reduction="none") * wa[:, None, :, None, None] * am
    la = la.permute(0, 2, 3, 4, 1).flatten(2).sum(2) / (am.permute(0, 2, 3, 4, 1).flatten(2).sum(2) + 1e-6)
    la = (la * fw).sum() / fw.sum().clamp(min=1)
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
                        seed=a.seed, drivers=a.drivers, limit_per_task=a.limit_per_task)
    if not eps:
        raise SystemExit("no episodes found")
    ctx_pool = find_episodes(a.context_pool, success=True, tasks=a.tasks) if a.context_pool else None
    print(f"{len(eps)} training episodes")
    ds = EpisodeDataset(eps, ActionNorm(cfg), TextCache(a.text_cache), max_frames=a.max_frames,
                        context_k=a.context_k, context_pool=ctx_pool, seed=a.seed, cfg_prob=a.cfg_prob)
    tf = load_frozen_transformer(a.model_path, device)
    if a.init:
        aset = fa.AdapterSet.load(a.init, device=device)
    else:
        acfg = fa.AdapterConfig(kind=a.kind, inside_streams=a.inside_streams, bottleneck=a.bottleneck,
                                lora_rank=a.lora_rank, blocks=a.blocks)
        aset = fa.AdapterSet(acfg).to(device)
    fa.attach(tf, aset)
    params = aset.inside_parameters()
    for p in params:
        p.requires_grad_(True)
    print(f"trainable inside params: {sum(p.numel() for p in params) / 1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.weight_decay)
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
        fw = None
        if ctx:
            fw = torch.ones(lat.shape[2])
            fw[:ctx] = 0
        pred = tf(input_dict, train_mode=True)
        lv, la = flow_losses(input_dict, pred, patch, sched_v, sched_a, fw)
        loss = a.lambda_video * lv + a.lambda_action * la
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % a.log_every == 0 or step == 1:
            rec = dict(step=step, loss=float(loss), video=float(lv), action=float(la), grad_norm=float(gn),
                       frames=int(lat.shape[2]), ctx=ctx, s=round(time.time() - t0, 1))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if a.save_every and step % a.save_every == 0:
            aset.save(out / f"step{step}")
    aset.save(out)
    (out / "train_args_inside.json").write_text(json.dumps({**vars(a), "episodes_used": [str(e) for e in eps]},
                                                           indent=2, default=str))
    print(f"saved {out}")


def train_outside(a):
    device = torch.device("cuda")
    if a.pairs:
        d = np.load(a.pairs)
        P, O = d["pred"], d["obs"]
    else:
        eps = find_episodes(a.episodes, success=a.success, tasks=a.tasks, settings=a.settings, limit=a.limit,
                            seed=a.seed, drivers=a.drivers, limit_per_task=a.limit_per_task)
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
        aset = fa.AdapterSet(fa.AdapterConfig(kind="outside")).to(device)
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
    ap.add_argument("--limit", type=int, help="number of episodes (few-shot n)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_path", default=str(DEFAULT_MODEL))
    ap.add_argument("--text_cache", default=str(ROOT / "artifacts" / "text_emb"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", help="existing adapter dir to continue from / to add the outside residual to")
    ap.add_argument("--kind", default="dual", choices=["dual", "inside", "lora", "outside"])
    ap.add_argument("--inside_streams", nargs="+", default=["video", "action"])
    ap.add_argument("--blocks", type=int, nargs="*", help="block indices with adapters (default: all 30)")
    ap.add_argument("--bottleneck", type=int, default=128)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=4.343e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--lambda_video", type=float, default=1.0)
    ap.add_argument("--lambda_action", type=float, default=1.0)
    ap.add_argument("--max_frames", type=int, default=24)
    ap.add_argument("--cfg_prob", type=float, default=0.1)
    ap.add_argument("--context_k", type=int, default=0, help="E3: demos concatenated in front of each sample")
    ap.add_argument("--context_pool", nargs="*", help="E3: episode roots the context demos are drawn from")
    ap.add_argument("--icl_gap", type=int, default=1)
    ap.add_argument("--steps_outside", type=int, default=300)
    ap.add_argument("--lr_outside", type=float, default=1.234e-3)
    ap.add_argument("--batch_outside", type=int, default=32)
    ap.add_argument("--pairs", help="npz with pred/obs canvases from fawam.replay")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "inside":
        train_inside(a)
    else:
        train_outside(a)


if __name__ == "__main__":
    main()
