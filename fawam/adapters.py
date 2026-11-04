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

import numpy as np
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
    """kind: dual | inside | outside | lora | ia3 | adaln | combo | none.
    combo = any union of the components below (use_adapterplus, use_lora, ia3, adaln, outside via kind dual/outside is
    separate). Components (docs/ICL.md, adapter types):
      Adapter+ (bottleneck, sites, inside_streams, blocks, adapter_learn_scale=False -> Houlsby-style fixed scale)
      LoRA on lora_targets: attn (self-attn q/k/v/o, both streams), cross (text cross-attn q/k/v/o), ffn (in/out)
      (IA)^3: learned scales on self-attn keys/values and on the FFN hidden activation
      AdaLN: learned offset of every block's modulation table (scale/shift/gate of norms)"""
    kind: str = "dual"
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
    ffn_dim: int = 14336
    lora_targets: list = field(default_factory=lambda: ["attn"])
    adapter_learn_scale: bool = True
    use_adapterplus: bool = False  # combo
    use_lora: bool = False         # combo
    ia3: bool = False
    adaln: bool = False
    adapter_norm: bool = True       # False (+ adapter_learn_scale False) = plain residual MLP (paper baseline)
    # VANE / T3VF: learnable prompt tokens read by every block through a gated attention (prompt bank + top-k router)
    prompt_tokens: int = 0
    prompt_bank: int = 1
    prompt_topk: int = 1
    # WAM-TTT: fast-weight residual branch; one analytic inner SGD step on the in-context memory tokens
    ttt_dim: int = 0
    ttt_eta: float = 0.1
    ttt_streams: list = field(default_factory=lambda: ["video"])
    # SALT: singular-value scaling (log-scale beta, clipped) of selected projections
    svf_blocks: list = None
    svf_targets: list = field(default_factory=lambda: ["self_k"])  # self_k | self_v | cross_k
    svf_clip: float = 0.15
    # dual-site paper baseline: dense (full-rank) deltas = full fine-tuning of selected linears
    dense_blocks: list = None
    dense_targets: list = field(default_factory=lambda: ["attn"])  # attn | ffn
    outside_type: str = "mlp"       # mlp (residual MLP) | channel (channel-wise scale + shift, paper baseline)
    # FiLM / conditional AdaLN: per-block offsets of the modulation table (scale/shift/gate) computed from
    #   memory: the pooled in-context memory tokens at that block | error: a compact failure vector from the analysis
    film_source: str = "none"       # none | memory | error
    film_rank: int = 64
    error_dim: int = 16

    @property
    def has_inside(self):
        return self.kind in ("dual", "inside") or (self.kind == "combo" and self.use_adapterplus)

    @property
    def has_outside(self):
        return self.kind in ("dual", "outside")

    @property
    def has_lora(self):
        return self.kind == "lora" or (self.kind == "combo" and self.use_lora)

    @property
    def has_ia3(self):
        return self.kind == "ia3" or self.ia3

    @property
    def has_adaln(self):
        return self.kind == "adaln" or self.adaln

    @property
    def has_film(self):
        return self.film_source in ("memory", "error")

    @property
    def has_prompt(self):
        return self.prompt_tokens > 0

    @property
    def has_ttt(self):
        return self.ttt_dim > 0

    @property
    def has_svf(self):
        return bool(self.svf_blocks)

    @property
    def has_dense(self):
        return bool(self.dense_blocks)

    def block_ids(self):
        return list(range(self.num_blocks)) if self.blocks is None else list(self.blocks)


# ------------------------------------------------------------------------------------------------ modules

class AdapterPlus(nn.Module):
    """h + c * W_up GELU(W_down LN(h)) (Steitz & Roth 2024), zero-initialised W_up."""

    def __init__(self, dim, bottleneck, learn_scale=True, norm=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if norm else nn.Identity()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.scale = nn.Parameter(torch.ones(dim), requires_grad=learn_scale)
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


class IA3(nn.Module):
    """(IA)^3 (Liu et al. 2022): learned rescaling of self-attention keys/values and the FFN hidden activation."""

    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.k = nn.Parameter(torch.ones(dim))
        self.v = nn.Parameter(torch.ones(dim))
        self.ff = nn.Parameter(torch.ones(ffn_dim))


class ChannelScale(nn.Module):
    """Outside site, paper baseline: channel-wise scale and shift of the imagined latent, zero-initialised."""

    def __init__(self, channels=48):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(channels))
        self.b = nn.Parameter(torch.zeros(channels))

    def forward(self, latents):
        x = latents.float()
        sh = (1, -1) + (1,) * (x.dim() - 2)
        return (x * (1 + self.s.view(sh)) + self.b.view(sh)).to(latents.dtype)


class PromptMemory(nn.Module):
    """VANE (latent-prompt bank with top-k routing) / T3VF (learnable query tokens), adapted to a frozen block:
    h + g * Attn(W_q LN(h), P, P), with a bank of prompts P_i (n tokens each) mixed by a router on the mean hidden state.
    g is zero-initialised (identical to the base at init)."""

    def __init__(self, dim, n_tokens, bank=1, topk=1):
        super().__init__()
        self.prompts = nn.Parameter(torch.randn(bank, n_tokens, dim) * 0.02)
        self.router = nn.Linear(dim, bank) if bank > 1 else None
        self.topk = min(topk, bank)
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(dim))
        self.heads = 24 if dim % 24 == 0 else 1  # LingBot-VA: 24 heads of 128

    def forward(self, h):
        x = self.norm(h.float())  # B, L, D
        if self.router is not None:
            w = self.router(x.mean(1)).softmax(-1)  # B, bank
            top = w.topk(self.topk, dim=-1)
            mix = torch.zeros_like(w).scatter(-1, top.indices, top.values)
            mix = mix / mix.sum(-1, keepdim=True)
            P = torch.einsum("bk,knd->bnd", mix, self.prompts)
        else:
            P = self.prompts[:1].expand(x.shape[0], -1, -1)
        B, L, D = x.shape
        hd = D // self.heads
        q = self.q(x).view(B, L, self.heads, hd).transpose(1, 2)
        k = P.reshape(B, -1, self.heads, hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, k).transpose(1, 2).reshape(B, L, D)
        return (h.float() + self.gate * o).type_as(h)


class TTTBranch(nn.Module):
    """WAM-TTT-style fast-weight residual branch (linear fast weight, Spatial-TTT form):
        dz = W_O f_W(W_Q LN(z)),  f_W(x) = W x
    The fast weight W starts at W_init and takes ONE inner SGD step on the key-value reconstruction loss
    ||W k - v||^2 over the in-context memory tokens (k = W_K LN(h), v = W_V LN(h)); for a linear f_W the step is
    analytic from two running sums (S_kk = sum k k^T, S_vk = sum v k^T), so the server can accumulate them while it
    prefills the memory and gradients flow through it in meta-training. W_O zero-initialised: identity at init."""

    def __init__(self, dim, dh, eta=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.Wq, self.Wk, self.Wv = (nn.Linear(dim, dh, bias=False) for _ in range(3))
        self.Wo = nn.Linear(dh, dim, bias=False)
        nn.init.zeros_(self.Wo.weight)
        self.W_init = nn.Parameter(torch.eye(dh))
        self.eta = eta

    def stats(self, h_ctx):
        x = self.norm(h_ctx.float()).reshape(-1, h_ctx.shape[-1])
        k, v = F.normalize(self.Wk(x), dim=-1), self.Wv(x)
        return k.t() @ k, v.t() @ k, x.shape[0]

    def fast_weight(self, stats):
        if stats is None or stats[2] == 0:
            return self.W_init
        S_kk, S_vk, n = stats
        return self.W_init - self.eta * 2.0 / n * (self.W_init @ S_kk - S_vk)

    def forward(self, h, stats=None):
        W = self.fast_weight(stats)
        q = F.normalize(self.Wq(self.norm(h.float())), dim=-1)
        return (h.float() + self.Wo(q @ W.t())).type_as(h)


class FiLMHead(nn.Module):
    """Low-rank map condition -> offset of one block's modulation table (1, 6, dim); last layer zero-initialised
    (AdaLN-Zero style), so the adapted block equals the base block at init."""

    def __init__(self, cond_dim, dim, rank):
        super().__init__()
        self.norm = nn.LayerNorm(cond_dim)
        self.down = nn.Linear(cond_dim, rank)
        self.up = nn.Linear(rank, 6 * dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.dim = dim

    def forward(self, c):
        return self.up(F.gelu(self.down(self.norm(c.float())))).view(-1, 6, self.dim)


ERROR_TYPES = ["missed_grasp", "misaligned_grasp", "dropped", "misplaced", "unknown"]


def error_vector(ana, dim=16):
    """Compact failure description for FiLM-error: delta_m (3, in dm), arm one-hot (2), failure type one-hot (5),
    gap statistics (max, mean, onset position), padded to dim."""
    v = np.zeros(dim, np.float32)
    if not ana or not ana.get("failed"):
        return v
    v[0:3] = np.asarray(ana.get("delta_m", [0, 0, 0]), np.float32) * 10
    if ana.get("arm") in ("left", "right"):
        v[3 + ("left", "right").index(ana["arm"])] = 1
    ft = ana.get("failure_type", "unknown")
    v[5 + (ERROR_TYPES.index(ft) if ft in ERROR_TYPES else 4)] = 1
    g = np.array(list((ana.get("gap") or {}).values()), np.float32)
    if len(g):
        v[10], v[11] = g.max(), g.mean()
        lat = ana.get("latent")
        v[12] = (lat / max(len(g), 1)) if lat is not None else 0
    return v


class SVFDelta(nn.Module):
    """SALT: W(beta) = U diag(sigma * exp(beta)) V^T of a frozen linear, beta clipped to [-c, c]; only beta is trained.
    The SVD of the base weight is computed when attached (not saved)."""

    def __init__(self, rank, clip=0.15):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(rank))
        self.clip = clip
        self.register_buffer("U", torch.empty(0), persistent=False)
        self.register_buffer("S", torch.empty(0), persistent=False)
        self.register_buffer("Vh", torch.empty(0), persistent=False)

    def set_base(self, weight):
        w = weight.full_tensor() if hasattr(weight, "full_tensor") else weight
        U, S, Vh = torch.linalg.svd(w.detach().float(), full_matrices=False)
        self.U, self.S, self.Vh = U, S, Vh

    def forward(self, x):
        scale = self.S * (torch.exp(self.beta.clamp(-self.clip, self.clip)) - 1)
        return ((x.float() @ self.Vh.t()) * scale) @ self.U.t()


class DenseDelta(nn.Module):
    """Full-rank delta of a frozen linear (= full fine-tuning of that weight), zero-initialised."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.D = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x):
        return x.float() @ self.D.t()


SVF_PATHS = {"self_k": ("attn1", "to_k"), "self_v": ("attn1", "to_v"), "cross_k": ("attn2", "to_k")}


def dense_specs(cfg):
    out = []
    d, f = cfg.dim, cfg.ffn_dim
    for b in cfg.dense_blocks or []:
        if "attn" in cfg.dense_targets:
            out += [(f"{b}_{n}", ("attn1", n), d, d) for n in ("to_q", "to_k", "to_v", "to_out")]
        if "ffn" in cfg.dense_targets:
            out += [(f"{b}_ff_in", ("ffn", "in"), d, f), (f"{b}_ff_out", ("ffn", "out"), f, d)]
    return out


def lora_specs(cfg):
    """(key, module path, in, out) of every LoRA delta for the configured targets."""
    out = []
    d, f = cfg.dim, cfg.ffn_dim
    for b in cfg.block_ids():
        if "attn" in cfg.lora_targets:
            out += [(f"{b}_{n}", ("attn1", n), d, d) for n in ("to_q", "to_k", "to_v", "to_out")]
        if "cross" in cfg.lora_targets:
            out += [(f"{b}_x_{n}", ("attn2", n), d, d) for n in ("to_q", "to_k", "to_v", "to_out")]
        if "ffn" in cfg.lora_targets:
            out += [(f"{b}_ff_in", ("ffn", "in"), d, f), (f"{b}_ff_out", ("ffn", "out"), f, d)]
    return out


def _linear(block, path):
    mod, name = path
    if mod == "ffn":
        return block.ffn.net[0].proj if name == "in" else block.ffn.net[2]
    attn = getattr(block, mod)
    return attn.to_out[0] if name == "to_out" else getattr(attn, name)


class AdapterSet(nn.Module):
    """All trainable parts of one adaptation: internal adapters per stream/block/site, external residual, LoRA."""

    def __init__(self, cfg: AdapterConfig):
        super().__init__()
        self.cfg = cfg
        self.inside = nn.ModuleDict()
        if cfg.has_inside:
            for stream in cfg.inside_streams:
                self.inside[stream] = nn.ModuleDict({
                    str(b): nn.ModuleDict({s: AdapterPlus(cfg.dim, cfg.bottleneck, cfg.adapter_learn_scale,
                                                          cfg.adapter_norm) for s in cfg.sites})
                    for b in cfg.block_ids()})
        self.outside = (ChannelScale(cfg.latent_channels) if cfg.outside_type == "channel" else
                        ExternalResidual(cfg.latent_channels, cfg.patch, cfg.outside_hidden)) if cfg.has_outside else None
        self.lora = nn.ModuleDict({k: LoRADelta(i, o, cfg.lora_rank, cfg.lora_alpha)
                                   for k, _, i, o in lora_specs(cfg)}) if cfg.has_lora else nn.ModuleDict()
        self.ia3 = nn.ModuleDict({str(b): IA3(cfg.dim, cfg.ffn_dim) for b in cfg.block_ids()}) if cfg.has_ia3 \
            else nn.ModuleDict()
        self.adaln = nn.ParameterDict({str(b): nn.Parameter(torch.zeros(1, 6, cfg.dim)) for b in cfg.block_ids()}) \
            if cfg.has_adaln else nn.ParameterDict()
        self.prompt = nn.ModuleDict({str(b): PromptMemory(cfg.dim, cfg.prompt_tokens, cfg.prompt_bank, cfg.prompt_topk)
                                     for b in cfg.block_ids()}) if cfg.has_prompt else nn.ModuleDict()
        self.ttt = nn.ModuleDict({st: nn.ModuleDict({str(b): TTTBranch(cfg.dim, cfg.ttt_dim, cfg.ttt_eta)
                                                     for b in cfg.block_ids()}) for st in cfg.ttt_streams}) \
            if cfg.has_ttt else nn.ModuleDict()
        self.svf = nn.ModuleDict({f"{b}_{t}": SVFDelta(cfg.dim, cfg.svf_clip) for b in (cfg.svf_blocks or [])
                                  for t in cfg.svf_targets}) if cfg.has_svf else nn.ModuleDict()
        self.dense = nn.ModuleDict({k: DenseDelta(i, o) for k, _, i, o in dense_specs(cfg)}) if cfg.has_dense \
            else nn.ModuleDict()
        cond = cfg.dim if cfg.film_source == "memory" else cfg.error_dim
        self.film = nn.ModuleDict({str(b): FiLMHead(cond, cfg.dim, cfg.film_rank) for b in cfg.block_ids()}) \
            if cfg.has_film else nn.ModuleDict()

    def inside_parameters(self):
        """Trainable non-outside parameters (Houlsby-style adapters keep their fixed scale out)."""
        out = []
        for name, p in self.named_parameters():
            if name.startswith("outside."):
                continue
            if name.endswith(".scale") and name.startswith("inside.") and not self.cfg.adapter_learn_scale:
                continue
            out.append(p)
        return out

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
        state = {k: v.detach().cpu() for k, v in self.state_dict().items()}  # SVD buffers are not persistent
        torch.save(state, path / "adapter.pt")
        (path / "adapter_config.json").write_text(json.dumps(asdict(self.cfg), indent=2))

    @classmethod
    def load(cls, path, transformer=None, device="cpu"):
        path = Path(path)
        cfg = AdapterConfig(**json.loads((path / "adapter_config.json").read_text()))
        aset = cls(cfg)  # creates every component (LoRA shapes from the config) before loading the weights
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
        self.ttt_accumulate = False  # server prefill: TTT branches / FiLM collect memory statistics
        self.film_error = None  # FiLM-error condition (error_dim,) tensor
        self.ctx_frames = 0  # training: in-context frames at the start of the sequence (TTT memory)
        self.n_frames = 1


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
    table = block.scale_shift_table
    if aset is not None and ctx.enabled and str(block_id) in aset.adaln:
        table = table + aset.adaln[str(block_id)]
    if aset is not None and ctx.enabled and str(block_id) in aset.film:
        off = _film_offset(hidden_states, aset, ctx, block_id, holder)
        if off is not None:
            table = table + off[:1].to(table.dtype)
    temb_scale_shift_table = table[None] + temb.float()
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = \
        rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(6, dim=1)
    shift_msa, scale_msa, gate_msa = shift_msa.squeeze(1), scale_msa.squeeze(1), gate_msa.squeeze(1)
    c_shift_msa, c_scale_msa, c_gate_msa = c_shift_msa.squeeze(1), c_scale_msa.squeeze(1), c_gate_msa.squeeze(1)

    norm_hidden_states = (block.norm1(hidden_states.float()) * (1. + scale_msa) + shift_msa).type_as(hidden_states)
    attn_output = block.attn1(norm_hidden_states, norm_hidden_states, norm_hidden_states, rotary_emb,
                              update_cache=update_cache, cache_name=cache_name)
    hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
    hidden_states = _site(hidden_states, aset, ctx, block_id, "attn")
    if aset is not None and ctx.enabled:
        if str(block_id) in aset.prompt:
            hidden_states = aset.prompt[str(block_id)](hidden_states)
        if len(aset.ttt):
            hidden_states = _ttt(hidden_states, aset, ctx, block_id, holder)

    norm_hidden_states = block.norm2(hidden_states.float()).type_as(hidden_states)
    attn_output = block.attn2(norm_hidden_states, encoder_hidden_states, encoder_hidden_states, None,
                              update_cache=0, cache_name=cache_name)
    hidden_states = hidden_states + attn_output

    norm_hidden_states = (block.norm3(hidden_states.float()) * (1. + c_scale_msa) + c_shift_msa).type_as(hidden_states)
    ff_output = block.ffn(norm_hidden_states)
    hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
    hidden_states = _site(hidden_states, aset, ctx, block_id, "block")
    return _steer(hidden_states, ctx, block_id)


def _film_offset(h, aset, ctx, block_id, holder):
    """FiLM condition for this block. memory: mean of the block-input hidden states of the in-context memory tokens
    (training: clean context frames of the sequence; inference: summed while the server prefills the memory).
    error: the failure vector set by the client/trainer. None -> no offset (no memory / no failure)."""
    head = aset.film[str(block_id)]
    if aset.cfg.film_source == "error":
        e = ctx.film_error
        return None if e is None else head(e.to(h.device)[None])
    b = str(block_id)
    if ctx.segments is not None:  # training
        if ctx.ctx_frames <= 0:
            return None
        nf = max(ctx.n_frames, 1)
        parts, start = [], 0
        for i, (n, stream) in enumerate(ctx.segments):
            if i % 2 == 1:  # clean segments
                parts.append(h[:, start:start + (n // nf) * ctx.ctx_frames])
            start += n
        return head(torch.cat(parts, dim=1).float().mean(1))
    if ctx.ttt_accumulate:  # prefill: collect, no offset yet
        s_, n_ = holder.film_stats.get(b, (0.0, 0))
        holder.film_stats[b] = (s_ + h[:1].float().sum(1), n_ + h.shape[1])
        return None
    st = holder.film_stats.get(b)
    return None if st is None else head(st[0] / st[1])


def _ttt(h, aset, ctx, block_id, holder):
    """Fast-weight branch. Training (segments): stats from the clean in-context frames of the same stream in this
    sequence. Inference: while the server prefills the memory (ctx.ttt_accumulate) the stats are summed per block and
    stream; afterwards every forward uses the stepped fast weight."""
    b = str(block_id)
    if ctx.segments is None:
        br = aset.ttt[ctx.stream][b] if ctx.stream in aset.ttt and b in aset.ttt[ctx.stream] else None
        if br is None:
            return h
        key = (ctx.stream, b)
        if ctx.ttt_accumulate:
            S_kk, S_vk, n = br.stats(h[:1])
            old = holder.ttt_stats.get(key)
            holder.ttt_stats[key] = (S_kk, S_vk, n) if old is None else (old[0] + S_kk, old[1] + S_vk, old[2] + n)
            return br(h, None)
        return br(h, holder.ttt_stats.get(key))
    # training: [noisy video | clean video | noisy action | clean action], context = first ctx_frames frames
    outs, start = [], 0
    nf = max(ctx.n_frames, 1)
    for i, (n, stream) in enumerate(ctx.segments):
        part = h[:, start:start + n]
        br = aset.ttt[stream][b] if (stream in aset.ttt and b in aset.ttt[stream]) else None
        if br is not None:
            clean = h[:, start + (n if i % 2 == 0 else 0):start + (2 * n if i % 2 == 0 else n)]
            n_ctx = (n // nf) * ctx.ctx_frames
            stats = br.stats(clean[:, :n_ctx]) if n_ctx > 0 else None
            part = br(part, stats)
        outs.append(part)
        start += n
    if start < h.shape[1]:
        outs.append(h[:, start:])
    return torch.cat(outs, dim=1)


class AdapterHolder:
    """Attached to the transformer as `_fawam`: the active AdapterSet (or None) and the forward context."""

    def __init__(self):
        self.adapters = None
        self.ctx = AdapterContext()
        self.ttt_stats = {}  # (stream, block) -> (S_kk, S_vk, n) of the prefilled in-context memory
        self.film_stats = {}  # block -> (sum of memory hidden states, count) for FiLM-memory


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
            holder.ctx.n_frames = int(input_dict["latent_dict"]["noisy_latents"].shape[2])
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
    """Forward hooks for LoRA deltas and (IA)^3 scales (FSDP-managed base weights are never touched). Hooks are
    installed once per linear and look the active AdapterSet up at call time, so adapters can be swapped."""
    holder = install(transformer)
    hooked = holder.__dict__.setdefault("hooked", set())
    for key, path, _, _ in lora_specs(aset.cfg) if aset.cfg.has_lora else []:
        if key in hooked:
            continue
        lin = _linear(_unwrap(transformer.blocks[int(key.split("_")[0])]), path)

        def hook(mod, inp, out, key=key):
            a = holder.adapters
            if a is None or not holder.ctx.enabled or key not in a.lora:
                return out
            return out + a.lora[key](inp[0]).to(out.dtype)

        lin.register_forward_hook(hook)
        hooked.add(key)
    for key, mod in list(aset.svf.items()) + list(aset.dense.items()):
        b, rest = key.split("_", 1)
        blk = _unwrap(transformer.blocks[int(b)])
        lin = _linear(blk, SVF_PATHS[rest] if isinstance(mod, SVFDelta) else
                      next(p for k_, p, _, _ in dense_specs(aset.cfg) if k_ == key))
        if isinstance(mod, SVFDelta) and mod.S.numel() == 0:
            mod.set_base(lin.weight)
            mod.to(next(mod.parameters()).device)
        hk = f"{'svf' if isinstance(mod, SVFDelta) else 'dense'}_{key}"
        if hk in hooked:
            continue

        def hook(mod_, inp, out, key=key, which="svf" if isinstance(mod, SVFDelta) else "dense"):
            a = holder.adapters
            table = getattr(a, which, {}) if a is not None else {}
            if a is None or not holder.ctx.enabled or key not in table:
                return out
            return out + table[key](inp[0]).to(out.dtype)

        lin.register_forward_hook(hook)
        hooked.add(hk)
    for b in aset.cfg.block_ids() if aset.cfg.has_ia3 else []:
        blk = _unwrap(transformer.blocks[b])
        for which, mod in (("k", blk.attn1.to_k), ("v", blk.attn1.to_v), ("ff", blk.ffn.net[0])):
            key = f"ia3_{b}_{which}"
            if key in hooked:
                continue

            def hook(mod_, inp, out, b=b, which=which):
                a = holder.adapters
                if a is None or not holder.ctx.enabled or str(b) not in a.ia3:
                    return out
                return out * getattr(a.ia3[str(b)], which).to(out.dtype)

            mod.register_forward_hook(hook)
            hooked.add(key)


def attach(transformer, aset):
    """Install the routing (if needed) and activate `aset` (None detaches)."""
    holder = install(transformer)
    if aset is not None and (aset.cfg.has_lora or aset.cfg.has_ia3 or aset.cfg.has_svf or aset.cfg.has_dense):
        add_lora(transformer, aset)
    holder.ttt_stats = {}
    holder.film_stats = {}
    holder.adapters = aset
    return holder
