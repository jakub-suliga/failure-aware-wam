"""Residual adapters for the frozen LingBot-VA transformer (dual-site adaptation, see docs/EXPERIMENTS.md, D).

Transfer of "Few-Shot Residual Adaptation of Pretrained Latent World Models" (Suliga et al., 2026) to the WAM:

* inside the predictor: Adapter+ modules (LN -> down -> GELU -> up, learned channel scale, zero-init up-projection)
  after the self-attention residual and after the completed block, in every one of the 30 blocks. LingBot-VA uses ONE
  backbone for both streams, so every site has a separate adapter per stream ("video", "action"); which streams are
  adapted is a config choice (the paper had separate view heads; the streams are the WAM's analogue).
* outside the predictor: a token-wise residual MLP on the predicted clean video latent (2x2 patches of the 48-channel
  latent = 192-d tokens), applied before the predicted frames are written into the KV cache, so the action stream
  conditions on the corrected imagination. Zero-initialised last layer: the adapted model starts identical to the base.
* LoRA on the self-attention projections as the parameter-efficient baseline (paper: best single-site internal method).
* steering vectors (training-free): a fixed vector added to the block output of one layer.

All modules are float32 and run on bf16 activations by casting. The block forward is re-implemented here (identical
maths to upstream WanTransformerBlock.forward) with the two adapter sites; a per-forward context says which stream the
tokens belong to, or, for training (video and action tokens concatenated), the segment layout of the sequence.
"""
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

SITES = ("attn", "block")
STREAMS = ("video", "action")


# ------------------------------------------------------------------------------------------------ config

@dataclass
class AdapterConfig:
    kind: str = "dual"  # dual | inside | outside | lora | none
    inside_streams: list = field(default_factory=lambda: ["video", "action"])  # streams with internal adapters
    sites: list = field(default_factory=lambda: list(SITES))
    blocks: list = None  # block indices with internal adapters (None = all)
    bottleneck: int = 128
    lora_rank: int = 16
    lora_alpha: float = 16.0
    outside_hidden: int = 1024
    dim: int = 3072
    num_blocks: int = 30
    latent_channels: int = 48
    patch: int = 2

    @property
    def has_inside(self):
        return self.kind in ("dual", "inside")

    @property
    def has_outside(self):
        return self.kind in ("dual", "outside")

    @property
    def has_lora(self):
        return self.kind == "lora"

    def block_ids(self):
        return list(range(self.num_blocks)) if self.blocks is None else list(self.blocks)


# ------------------------------------------------------------------------------------------------ modules

class AdapterPlus(nn.Module):
    """h + c * W_up GELU(W_down LN(h)) (Steitz & Roth 2024), zero-initialised W_up."""

    def __init__(self, dim, bottleneck):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.scale = nn.Parameter(torch.ones(dim))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h):
        x = h.float()
        return (x + self.scale * self.up(F.gelu(self.down(self.norm(x))))).type_as(h)


class ExternalResidual(nn.Module):
    """Token-wise MLP on 2x2 patches of the predicted latent: z + R(z). Sees only the predicted token (paper)."""

    def __init__(self, channels=48, patch=2, hidden=1024):
        super().__init__()
        d = channels * patch * patch
        self.patch = patch
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latents):
        """latents (B, C, F, H, W) normalized latent canvas -> corrected latents, same shape and dtype."""
        p = self.patch
        x = rearrange(latents.float(), "b c f (h p1) (w p2) -> b f h w (c p1 p2)", p1=p, p2=p)
        x = x + self.net(x)
        out = rearrange(x, "b f h w (c p1 p2) -> b c f (h p1) (w p2)", p1=p, p2=p)
        return out.to(latents.dtype)


class LoRADelta(nn.Module):
    """Low-rank update (alpha / r) B A x of a frozen linear, applied as a forward hook on that linear (so FSDP-managed
    base weights are never touched or re-registered). B is zero-initialised."""

    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, x):
        return (x.float() @ self.A.t() @ self.B.t()) * self.scaling


class AdapterSet(nn.Module):
    """All trainable parts of one adaptation: internal adapters per stream/block/site, external residual, LoRA."""

    def __init__(self, cfg: AdapterConfig):
        super().__init__()
        self.cfg = cfg
        self.inside = nn.ModuleDict()
        if cfg.has_inside:
            for stream in cfg.inside_streams:
                self.inside[stream] = nn.ModuleDict({
                    str(b): nn.ModuleDict({s: AdapterPlus(cfg.dim, cfg.bottleneck) for s in cfg.sites})
                    for b in cfg.block_ids()})
        self.outside = ExternalResidual(cfg.latent_channels, cfg.patch, cfg.outside_hidden) if cfg.has_outside else None
        self.lora = nn.ModuleDict()  # filled by install() (needs the base linears)

    def inside_parameters(self):
        return list(self.inside.parameters()) + list(self.lora.parameters())

    def outside_parameters(self):
        return [] if self.outside is None else list(self.outside.parameters())

    def get(self, stream, block, site):
        s = self.inside[stream] if stream in self.inside else None
        if s is None or str(block) not in s or site not in s[str(block)]:
            return None
        return s[str(block)][site]

    # -------------------------------------------------------------- io
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        torch.save(state, path / "adapter.pt")
        (path / "adapter_config.json").write_text(json.dumps(asdict(self.cfg), indent=2))

    @classmethod
    def load(cls, path, transformer=None, device="cpu"):
        path = Path(path)
        cfg = AdapterConfig(**json.loads((path / "adapter_config.json").read_text()))
        aset = cls(cfg)
        if cfg.has_lora:  # create the LoRA deltas (shapes from the config) before loading their weights
            for b in cfg.block_ids():
                for name in ("to_q", "to_k", "to_v", "to_out"):
                    aset.lora[f"{b}_{name}"] = LoRADelta(cfg.dim, cfg.dim, cfg.lora_rank, cfg.lora_alpha)
        state = torch.load(path / "adapter.pt", map_location="cpu", weights_only=True)
        missing, unexpected = aset.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"adapter state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        return aset.to(device)


# ------------------------------------------------------------------------------------------------ context

class AdapterContext:
    """What the tokens of the current transformer forward are.

    stream: "video" | "action" for inference forwards; segments: [(n_tokens, stream or None), ...] for training
    forwards that concatenate both streams (None = padding, never adapted). enabled=False bypasses all adapters
    (used to get base-model outputs from the same module, e.g. for replay comparisons).
    """

    def __init__(self):
        self.stream = "video"
        self.segments = None
        self.enabled = True
        self.steer = {}  # block index -> (vector (dim,), alpha, streams)


def _apply_segments(h, segments, fn_for_stream):
    outs, start = [], 0
    for n, stream in segments:
        part = h[:, start:start + n]
        fn = fn_for_stream(stream) if stream is not None else None
        outs.append(fn(part) if fn is not None else part)
        start += n
    if start < h.shape[1]:
        outs.append(h[:, start:])
    return torch.cat(outs, dim=1)


def _site(h, aset, ctx, block_id, site):
    if aset is None or not ctx.enabled:
        return h
    if ctx.segments is None:
        mod = aset.get(ctx.stream, block_id, site)
        return h if mod is None else mod(h)
    return _apply_segments(h, ctx.segments, lambda s: aset.get(s, block_id, site))


def _steer(h, ctx, block_id):
    if block_id not in ctx.steer:
        return h
    vec, alpha, streams = ctx.steer[block_id]
    if ctx.segments is None:
        return (h.float() + alpha * vec).type_as(h) if ctx.stream in streams else h
    return _apply_segments(h, ctx.segments,
                           lambda s: (lambda x: (x.float() + alpha * vec).type_as(x)) if s in streams else None)


def adapted_block_forward(block, block_id, holder, hidden_states, encoder_hidden_states, temb, rotary_emb,
                          update_cache=0, cache_name="pos"):
    """Upstream WanTransformerBlock.forward + adapter site after the self-attention residual and after the block."""
    aset, ctx = holder.adapters, holder.ctx
    temb_scale_shift_table = block.scale_shift_table[None] + temb.float()
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = \
        rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(6, dim=1)
    shift_msa, scale_msa, gate_msa = shift_msa.squeeze(1), scale_msa.squeeze(1), gate_msa.squeeze(1)
    c_shift_msa, c_scale_msa, c_gate_msa = c_shift_msa.squeeze(1), c_scale_msa.squeeze(1), c_gate_msa.squeeze(1)

    norm_hidden_states = (block.norm1(hidden_states.float()) * (1. + scale_msa) + shift_msa).type_as(hidden_states)
    attn_output = block.attn1(norm_hidden_states, norm_hidden_states, norm_hidden_states, rotary_emb,
                              update_cache=update_cache, cache_name=cache_name)
    hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
    hidden_states = _site(hidden_states, aset, ctx, block_id, "attn")

    norm_hidden_states = block.norm2(hidden_states.float()).type_as(hidden_states)
    attn_output = block.attn2(norm_hidden_states, encoder_hidden_states, encoder_hidden_states, None,
                              update_cache=0, cache_name=cache_name)
    hidden_states = hidden_states + attn_output

    norm_hidden_states = (block.norm3(hidden_states.float()) * (1. + c_scale_msa) + c_shift_msa).type_as(hidden_states)
    ff_output = block.ffn(norm_hidden_states)
    hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
    hidden_states = _site(hidden_states, aset, ctx, block_id, "block")
    return _steer(hidden_states, ctx, block_id)


class AdapterHolder:
    """Attached to the transformer as `_fawam`: the active AdapterSet (or None) and the forward context."""

    def __init__(self):
        self.adapters = None
        self.ctx = AdapterContext()


def _unwrap(block):
    return getattr(block, "_checkpoint_wrapped_module", block)


def install(transformer):
    """Route every block through adapted_block_forward and every forward through the stream context. Idempotent.
    With no adapters and no steering set, outputs are identical to the upstream model."""
    if hasattr(transformer, "_fawam"):
        return transformer._fawam
    holder = AdapterHolder()
    transformer._fawam = holder
    for i, blk in enumerate(transformer.blocks):
        inner = _unwrap(blk)
        inner.forward = (lambda b, bid: (lambda *a, **k: adapted_block_forward(b, bid, holder, *a, **k)))(inner, i)

    orig_forward = transformer.forward

    def forward(input_dict, update_cache=0, cache_name="pos", action_mode=False, train_mode=False):
        if train_mode:
            holder.ctx.segments = train_segments(input_dict, transformer.patch_size)
        else:
            holder.ctx.segments = None
            holder.ctx.stream = "action" if action_mode else "video"
        return orig_forward(input_dict, update_cache=update_cache, cache_name=cache_name, action_mode=action_mode,
                            train_mode=train_mode)

    transformer.forward = forward
    return holder


def train_segments(input_dict, patch_size):
    """Token layout of WanTransformer3DModel.forward_train: [noisy video | clean video | noisy action | clean action]
    (batch flattened into the sequence; padding after). Only batch size 1 is supported by the adapter trainer."""
    lat = input_dict["latent_dict"]["noisy_latents"]
    act = input_dict["action_dict"]["noisy_latents"]
    B, _, Fl, H, W = lat.shape
    assert B == 1, "adapter training uses batch size 1 (upstream flattens the batch into the sequence)"
    n_lat = Fl * (H // patch_size[1]) * (W // patch_size[2])
    n_act = act.shape[2] * act.shape[3] * act.shape[4]
    return [(n_lat, "video"), (n_lat, "video"), (n_act, "action"), (n_act, "action")]


def add_lora(transformer, aset):
    """LoRA on attn1 q/k/v/out of the adapted blocks (shared weights: affects both streams), as forward hooks."""
    cfg = aset.cfg
    holder = install(transformer)
    if getattr(holder, "lora_hooks", None):
        return
    holder.lora_hooks = []
    for b in cfg.block_ids():
        attn = _unwrap(transformer.blocks[b]).attn1
        for name, lin in (("to_q", attn.to_q), ("to_k", attn.to_k), ("to_v", attn.to_v), ("to_out", attn.to_out[0])):
            key = f"{b}_{name}"
            if key not in aset.lora:
                aset.lora[key] = LoRADelta(lin.in_features, lin.out_features, cfg.lora_rank, cfg.lora_alpha)

            def hook(mod, inp, out, key=key):
                if holder.adapters is None or not holder.ctx.enabled or key not in holder.adapters.lora:
                    return out
                return out + holder.adapters.lora[key](inp[0]).to(out.dtype)

            holder.lora_hooks.append(lin.register_forward_hook(hook))


def attach(transformer, aset):
    """Install the routing (if needed) and activate `aset` (None detaches)."""
    holder = install(transformer)
    if aset is not None and aset.cfg.has_lora:
        add_lora(transformer, aset)
    holder.adapters = aset
    return holder
