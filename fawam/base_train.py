"""Train the base WAM together with the adapters (user 2026-11-18; as in WAM-TTT, where meta-training optimises "the
WAM parameters, the TTT slow projections and W_init", and nothing is updated at test time).

Training (fawam/train_adapter.py --train_base): the transformer stays in bf16 for the forward/backward; this module
keeps fp32 master weights and AdamW moments on the GPU and applies the update one tensor at a time (no full fp32
gradient copy). Memory for LingBot-VA (4.7 B parameters): bf16 weights + bf16 grads 19 GB, fp32 master + 2 moments
57 GB, so it needs an H200 (141 GB), not an H100.

The trained weights are saved next to the adapter as base.safetensors (bf16, names without wrapper prefixes). The
evaluation server (fawam/server.py) swaps them in whenever the active adapter directory has that file and restores
the original weights otherwise (first attempts run the original model).
"""
from pathlib import Path

import torch

BASE_FILE = "base.safetensors"
_WRAPPERS = ("_checkpoint_wrapped_module.", "_fsdp_wrapped_module.", "_orig_mod.")


def clean(name):
    for w in _WRAPPERS:
        name = name.replace(w, "")
    return name


def base_params(tf, blocks=None):
    """(clean name, parameter) of the transformer itself. Adapters are attached through hooks, never registered as
    submodules, so every named parameter is a base weight. blocks: only these block indices (+ nothing outside)."""
    out = []
    for n, p in tf.named_parameters():
        c = clean(n)
        if blocks is not None:
            if not c.startswith("blocks.") or int(c.split(".")[1]) not in blocks:
                continue
        out.append((c, p))
    return out


class BaseTrainer:
    def __init__(self, tf, lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8, clip=1.0, blocks=None):
        self.named = base_params(tf, blocks)
        self.params = [p for _, p in self.named]
        for p in self.params:
            p.requires_grad_(True)
        self.master = [p.detach().float().clone() for p in self.params]
        self.m = [torch.zeros_like(x) for x in self.master]
        self.v = [torch.zeros_like(x) for x in self.master]
        self.lr, self.wd, self.b1, self.b2, self.eps, self.clip = lr, weight_decay, betas[0], betas[1], eps, clip
        self.t = 0
        print(f"train_base: {sum(p.numel() for p in self.params) / 1e9:.2f} B base parameters, lr {lr}", flush=True)

    @torch.no_grad()
    def step(self, lr_scale=1.0):
        grads = [p.grad for p in self.params if p.grad is not None]
        if not grads:
            return 0.0
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in grads])).item()
        scale = min(1.0, self.clip / (norm + 1e-6)) if self.clip else 1.0
        self.t += 1
        lr = self.lr * lr_scale
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, w, m, v in zip(self.params, self.master, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad.float().mul_(scale)
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            if self.wd:
                w.mul_(1 - lr * self.wd)
            w.addcdiv_(m / bc1, (v / bc2).sqrt_().add_(self.eps), value=-lr)
            p.copy_(w)
            p.grad = None
        return norm

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def save(self, path):
        from safetensors.torch import save_file
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file({n: p.detach().to(torch.bfloat16).contiguous().cpu() for n, p in self.named}, str(path / BASE_FILE))


class BaseSwapper:
    """Evaluation: keep the original weights on the CPU and swap trained base weights in/out of the live model."""

    def __init__(self, tf):
        self.tf = tf
        # FSDP2 (the server shards even with one GPU): parameters are DTensors; with world size 1 the local shard
        # is the full tensor
        self.live = {clean(n): (p.to_local() if hasattr(p, "to_local") else p).data for n, p in tf.named_parameters()}
        self.orig = {}
        self.cache = {}
        self.current = None  # None = original weights

    @torch.no_grad()
    def use(self, adapter_dir):
        """adapter_dir with base.safetensors -> its weights; anything else -> the original weights."""
        f = Path(adapter_dir) / BASE_FILE if adapter_dir else None
        key = str(f) if f is not None and f.exists() else None
        if key == self.current:
            return
        if key is None:
            for n, t in self.orig.items():
                self.live[n].copy_(t, non_blocking=True)
        else:
            if key not in self.cache:
                from safetensors.torch import load_file
                self.cache = {key: load_file(key)}  # one trained base at a time (9.5 GB each)
                for n in self.cache[key]:
                    if n not in self.live:
                        raise RuntimeError(f"{key}: unknown base parameter {n}")
            sd = self.cache[key]
            for n, t in sd.items():
                if n not in self.orig:
                    self.orig[n] = self.live[n].detach().to("cpu", copy=True)
                self.live[n].copy_(t, non_blocking=True)
            for n, t in self.orig.items():  # parameters of an earlier base that this one does not cover
                if n not in sd:
                    self.live[n].copy_(t, non_blocking=True)
        torch.cuda.synchronize()
        self.current = key
        print(f"base weights: {'original' if key is None else key}", flush=True)
