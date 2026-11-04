"""Logic test of the memory-usage probe and the memory manipulations without loading LingBot-VA
(lingbot env, CPU): python -m fawam.tests.test_memory_probe"""
import types

import numpy as np
import torch

from fawam.server import MEM_CAMS, MEM_PARTS, FawamServer


def fake_server(n=40):
    s = types.SimpleNamespace()
    s.cache_name = "pos"
    s.ctx_slots = torch.zeros(n, dtype=torch.bool)
    s.ctx_slots[:20] = True                       # slots 0-19 = memory
    s._mem_part = torch.full((n,), len(MEM_PARTS) - 1)
    s._mem_part[:10], s._mem_part[10:20] = 0, 1    # imagined / observed failure
    s._mem_cam = torch.zeros(n, dtype=torch.long)
    s._mem_cam[15:20] = 3                         # action tokens
    s._mem_win = torch.zeros(n, dtype=torch.long)
    s._mem_win[5:10] = 1
    s._probe = [[]]
    return s


def test_probe():
    s = fake_server()
    attn = types.SimpleNamespace(key_exclude=None, attn_caches={"pos": {"mask": torch.ones(40, dtype=torch.bool)}})
    q = torch.zeros(1, 4, 2, 8)
    k = torch.zeros(1, 40, 2, 8)
    k[0, 10:20] = 5.0  # strong keys on the observed-failure memory tokens
    q[0, :, :, :] = 1.0
    out = FawamServer._memory_probe_op(s, 0, attn, lambda q, k, v: q, q, k, k)
    mass, share, brk = s._probe[0][0]
    assert abs(share - 0.5) < 1e-6
    assert mass > 0.95, mass  # nearly all attention on memory
    assert brk["part"][1] > 0.9 * mass and brk["part"][0] < 0.05, brk["part"]
    print(f"probe: mass on memory {mass:.3f}, share {share:.2f}, part={np.round(brk['part'], 3)}, "
          f"cam={np.round(brk['cam'], 3)}, window={np.round(brk['win'], 3)}: ok")
    # excluded memory keys must not count
    attn.key_exclude = s.ctx_slots.clone()
    s._probe = [[]]
    FawamServer._memory_probe_op(s, 0, attn, lambda q, k, v: q, q, k[:, 20:], k[:, 20:])
    assert s._probe[0][0][0] == 0.0
    print("probe with the memory hidden: 0 mass: ok")


def test_manip():
    s = types.SimpleNamespace(LATENT_CAM_SLICES=FawamServer.LATENT_CAM_SLICES)
    lat = torch.randn(1, 48, 6, 24, 20)
    rows = np.arange(1, 5)
    for mp in ["shuffle_time", "reverse_time", "noise", "blank_head", "blank_wrists", "blank_left"]:
        out = FawamServer._manipulate(s, lat, rows, mp)
        assert torch.equal(out[:, :, [0, 5]], lat[:, :, [0, 5]]), f"{mp}: rows outside the segment changed"
        assert not torch.equal(out[:, :, rows], lat[:, :, rows]), f"{mp}: no change"
    b = FawamServer._manipulate(s, lat, rows, "blank_head")
    assert (b[:, :, rows, 8:] == 0).all() and torch.equal(b[:, :, rows, :8], lat[:, :, rows, :8])
    r = FawamServer._manipulate(s, lat, rows, "reverse_time")
    assert torch.equal(r[:, :, 1], lat[:, :, 4])
    print("manipulations: ok")


if __name__ == "__main__":
    test_probe()
    test_manip()
