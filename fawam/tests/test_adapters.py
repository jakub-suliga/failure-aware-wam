"""Small-model checks of the adapter routing, cache exclusion and adapter training (lingbot env, one GPU):

  python -m fawam.tests.test_adapters
"""
import copy

import torch

from fawam import adapters as fa
from fawam.paths import add_lingbot_to_path

add_lingbot_to_path()
from modules.model import WanTransformer3DModel  # noqa: E402
from utils import FlowMatchScheduler, get_mesh_id  # noqa: E402

DEV = "cuda"
KW = dict(patch_size=[1, 2, 2], num_attention_heads=2, attention_head_dim=32, in_channels=48, out_channels=48,
          action_dim=30, text_dim=64, freq_dim=32, ffn_dim=128, num_layers=3, cross_attn_norm=True, eps=1e-6,
          rope_max_seq_len=1024)


def make(attn_mode="torch"):
    torch.manual_seed(0)
    m = WanTransformer3DModel(**KW, attn_mode=attn_mode).to(DEV, torch.bfloat16).eval()
    return m


def video_input(F=2, H=24, W=20, frame_st=0):
    g = torch.Generator(device=DEV).manual_seed(1)
    x = torch.randn(1, 48, F, H, W, generator=g, device=DEV, dtype=torch.bfloat16)
    return dict(noisy_latents=x, timesteps=torch.full((1, F), 500.0, device=DEV),
                grid_id=get_mesh_id(F, H // 2, W // 2, 0, 1, frame_st).to(DEV)[None],
                text_emb=torch.randn(1, 16, 64, generator=g, device=DEV, dtype=torch.bfloat16))


def test_identity():
    base, routed = make(), make()
    routed.load_state_dict(base.state_dict())
    for m in (base, routed):
        m.create_empty_cache("pos", 8, 240, 32, device=DEV, dtype=torch.bfloat16, batch_size=1)
    fa.install(routed)
    inp = video_input()
    with torch.no_grad():
        a = base(copy.deepcopy(inp), update_cache=0)
        b = routed(copy.deepcopy(inp), update_cache=0)
    assert torch.equal(a, b), "routing without adapters must be bit-identical"
    cfg = fa.AdapterConfig(kind="dual", dim=64, num_blocks=3, bottleneck=8)
    aset = fa.AdapterSet(cfg).to(DEV)
    fa.attach(routed, aset)
    with torch.no_grad():
        c = routed(copy.deepcopy(inp), update_cache=0)
    assert torch.equal(a, c), "zero-initialised adapters must be a no-op"
    with torch.no_grad():
        for p in aset.inside.parameters():
            p.add_(0.05 * torch.randn_like(p))
        d = routed(copy.deepcopy(inp), update_cache=0)
    assert not torch.equal(a, d), "non-zero adapters must change the output"
    lat = torch.randn(1, 48, 2, 24, 20, device=DEV)
    assert torch.equal(aset.outside(lat), lat.to(lat.dtype)), "zero-init external residual must be identity"
    print("identity / zero-init: ok")


def test_exclusion():
    m = make()
    m.create_empty_cache("pos", 8, 240, 32, device=DEV, dtype=torch.bfloat16, batch_size=1)
    fa.install(m)
    with torch.no_grad():
        m(video_input(frame_st=0), update_cache=2)  # context frames 0-1
        m(video_input(frame_st=2), update_cache=1)  # predicted frames 2-3 (is_pred)
        q = video_input(frame_st=4)
        full = m(copy.deepcopy(q), update_cache=0)
        attn = [fa._unwrap(b).attn1 for b in m.blocks]
        c = attn[0].attn_caches["pos"]
        excl = c["mask"] & c["is_pred"]
        for a in attn:
            a.key_exclude = excl.clone()
        hidden = m(copy.deepcopy(q), update_cache=0)
        for a in attn:
            a.key_exclude = None
        # reference: drop the predicted slots from the cache for real
        for a in attn:
            a.clear_pred_cache("pos")
        ref = m(copy.deepcopy(q), update_cache=0)
    assert not torch.equal(full, hidden)
    assert torch.equal(hidden, ref), "key_exclude must equal removing the slots"
    print("cache exclusion: ok")


def test_train_step():
    from fawam.train_adapter import add_noise, flow_losses
    m = make("flex")
    m.requires_grad_(False)  # as fawam.train_adapter.load_frozen_transformer
    fa.install(m)
    cfg = fa.AdapterConfig(kind="dual", dim=64, num_blocks=3, bottleneck=8)
    aset = fa.AdapterSet(cfg).to(DEV)
    fa.attach(m, aset)
    for p in aset.inside_parameters():
        p.requires_grad_(True)
    sv = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    sv.set_timesteps(1000, training=True)
    sa = FlowMatchScheduler(shift=1.0, sigma_min=0.0, extra_one_step=True)
    sa.set_timesteps(1000, training=True)
    Fr = 6
    lat = torch.randn(1, 48, Fr, 24, 20, device=DEV)
    act = torch.randn(1, 30, Fr, 16, 1, device=DEV).clamp(-1, 1)
    am = torch.ones_like(act, dtype=torch.bool)
    ld = add_noise(lat, sv, DEV, (1, 2, 2), noisy_cond_prob=0.5, clean_frames=2, frame_gap=1)
    ad = add_noise(act, sa, DEV, (1, 2, 2), action_mask=am, action_mode=True, clean_frames=2, frame_gap=1)
    ld["text_emb"] = torch.randn(1, 512, 64, device=DEV, dtype=torch.bfloat16)
    ad["text_emb"] = ld["text_emb"]
    ad["actions_mask"] = am
    inp = dict(latent_dict=ld, action_dict=ad, chunk_size=2, window_size=16)
    pred = m(inp, train_mode=True)
    fw = torch.ones(Fr)
    fw[:2] = 0
    lv, la = flow_losses(inp, pred, (1, 2, 2), sv, sa, fw)
    (lv + la).backward()
    up = [aset.get(s, b, "block").up.weight.grad for s in ("video", "action") for b in range(3)]
    assert all(g is not None for g in up), "adapters must receive gradients"
    assert all(p.grad is None for p in m.parameters()), "base model must stay frozen"
    print(f"train step: ok (video {lv.item():.3f}, action {la.item():.3f})")


if __name__ == "__main__":
    test_identity()
    test_exclusion()
    test_train_step()
