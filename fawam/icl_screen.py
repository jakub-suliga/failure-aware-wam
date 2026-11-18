"""Offline screening of in-context memory variants (docs/ICL.md, no simulator, one GPU, lingbot env).

For pairs (own failed attempt F, successful expert replay S on the SAME seed and setting; P0 and P2 share seeds),
S is replayed teacher-forced (its observations and expert action chunks go into the KV cache) while the model sees F
as in-context memory in the format of each variant. Per chunk we compare the model's proposed actions with the expert
actions it is about to be shown. If a memory format carries usable information, the proposals move toward the expert
(especially around the time F failed) compared with no memory. Also reported: the prediction gap of the first chunk
(does the memory confuse the model?) and of all chunks.

  python -m fawam.icl_screen --fails results/sweeps/p0_baseline_table/base/table_h_m05 \
      --fixes results/sweeps/p2_expert_data/base/table_h_m05 --variants base icl_fail_obs icl_fail_predobs ... \
      --limit 20 --out results/icl_screen/m05.csv

Variants are method presets (or files); their learn.icl config is applied offline, meta_adapter/adapter_path loaded.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from fawam.latent_data import find_episodes
from fawam.methods import resolve
from fawam.paths import LINGBOT, ROOT, add_lingbot_to_path
from fawam.robotwin_hooks import icl_prompt, make_context_spec
from fawam.server import FawamServer, canvas_from_h5, cos_dist

add_lingbot_to_path()
from configs import VA_CONFIGS  # noqa: E402
from distributed.util import init_distributed  # noqa: E402


def pairs(fails, fixes, limit, seed=0, any_seed=False):
    """same-seed pairs; any_seed=True pairs a failure with any success of the same task (smoke tests without P2)."""
    fix = {}
    for e in fixes:
        with h5py.File(e / "episode.h5", "r") as f:
            k = (str(f.attrs["task_name"]),) if any_seed else \
                (str(f.attrs["task_name"]), str(f.attrs["task_config"]), int(f.attrs["seed"]))
            fix.setdefault(k, e)
    out = []
    for e in fails:
        if not (e / "analysis.json").exists():
            continue
        with h5py.File(e / "episode.h5", "r") as f:
            k = (str(f.attrs["task_name"]), str(f.attrs["task_config"]), int(f.attrs["seed"]))
        kk = k[:1] if any_seed else k
        if kk in fix:
            out.append((e, fix[kk], k))
    rng = np.random.default_rng(seed)
    rng.shuffle(out)
    return out[:limit] if limit else out


def memory_usage(records):
    """Mean over chunks: attention ratio on the memory (attention mass / memory share of the tokens; 1 = no
    preference, 0 = ignored) for the action and video tokens, and the relative change of the prediction when the
    memory is hidden."""
    if not records:
        return dict(attn_ratio_action=np.nan, attn_ratio_video=np.nan, infl_action=np.nan, infl_video=np.nan)
    ra = [np.nanmean(r["attn_action"]) / max(r["share_action"], 1e-8) for r in records]
    rv = [np.nanmean(r["attn_video"]) / max(r["share_video"], 1e-8) for r in records if r.get("attn_video") is not None]
    f = lambda k: float(np.nanmean([np.nan if r.get(k) is None else r[k] for r in records]))
    return dict(attn_ratio_action=float(np.mean(ra)), attn_ratio_video=float(np.mean(rv)) if rv else np.nan,
                infl_action=f("infl_action"), infl_video=f("infl_video"))


@torch.no_grad()
def run_variant(server, fail_dir, fix_dir, key, name):
    m = resolve(name, root=ROOT)
    ana = json.loads((fail_dir / "analysis.json").read_text())
    with h5py.File(fix_dir / "episode.h5", "r") as f:
        lat = canvas_from_h5(f).to(server.device, server.dtype)
        planned = f["actions/planned"][()]
        oc = f["model/obs/chunk_id"][()]
        li_fix = f["model/obs/latent_index"][()]
        instr = str(f.attrs["instruction"])
    spec, prompt, neg, adapter = None, instr, None, m.adapter_path
    if m.learn and m.learn.get("icl"):
        icl = m.learn["icl"]
        spec = make_context_spec(icl, fail_dir, key[2], key[1], [str(fail_dir)], [])
        from fawam.adapters import error_vector
        spec["error_vec"] = error_vector(ana).tolist()
        prompt, neg = icl_prompt(icl, ana, instr)
        adapter = m.learn.get("meta_adapter") or adapter
    method = dict(name=name, adapter_path=adapter, negative_prompt=neg)
    server._reset(prompt=prompt, seed=key[2], method=method, task=key[0])
    server.pending_prefill = None
    server.init_latent, server.init_ready = lat[:, :, 0:1], True
    if spec:
        server._prefill(spec)
    norm = lambda a: server.preprocess_action(a)[0, :, :, :, 0].float()  # 30, F, 16 normalized
    used = torch.as_tensor(server.action_mask)
    err, err_win, gaps = [], [], []
    L = ana.get("latent")
    for c in range(len(planned)):
        rows = np.where(oc == c)[0]
        if len(rows) == 0:
            break
        act, _ = server._infer(None, frame_st_id=server.frame_st_id)
        n = len(rows)
        e = (norm(act[:, :n]) - norm(planned[c][:, :n]))[used].pow(2).mean().item()
        if c == 0:
            e = (norm(act[:, 1:n]) - norm(planned[c][:, 1:n]))[used].pow(2).mean().item()  # frame 0 = conditioning
        err.append(e)
        k0 = int(li_fix[rows[0]])
        if L is not None and L - 4 <= k0 <= L + 2:
            err_win.append(e)
        r = server._commit_observation(lat[:, :, rows], planned[c][:, :n])
        g = r.get("gap", {})
        gaps.append(max(g.get("cam_left_wrist", 0), g.get("cam_right_wrist", 0)))
    mem = server.episode_log.get("memory", [])
    use = memory_usage(mem)
    server.transformer.clear_cache(server.cache_name)
    return dict(**use, action_mse=float(np.mean(err)), action_mse_window=float(np.mean(err_win)) if err_win else np.nan,
                gap_first=float(gaps[0]) if gaps else np.nan, gap_mean=float(np.mean(gaps)) if gaps else np.nan,
                memory_tokens=int(server.ctx_slots.sum()) if spec and getattr(server, "ctx_slots", None) is not None else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fails", nargs="+", required=True)
    ap.add_argument("--fixes", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["base", "icl_text_only", "icl_fail_obs", "icl_fail_pred",
                                                       "icl_fail_predobs", "icl_fail_diff", "icl_fail_window",
                                                       "icl_fail_corrected", "icl_demo_same", "icl_fail_demo",
                                                       "icl_pos_aligned", "icl_gap8", "icl_attn1.5", "icl_attn0.5"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--any_seed", action="store_true", help="smoke test: pair failures with any success of the task")
    ap.add_argument("--model_path", default=str(LINGBOT / "checkpoints" / "lingbot-va-posttrain-robotwin"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(__import__("fawam.paths", fromlist=["free_port"]).free_port()))
    init_distributed(1, 0, 0)
    cfg = VA_CONFIGS["robotwin"]
    cfg.wan22_pretrained_model_name_or_path = a.model_path
    cfg.enable_offload = True
    cfg.rank, cfg.local_rank, cfg.world_size = 0, 0, 1
    server = FawamServer(cfg)
    P = pairs(find_episodes(a.fails, success=False), find_episodes(a.fixes, success=True), a.limit,
              any_seed=a.any_seed)
    print(f"{len(P)} (failure, fix) pairs on identical seeds")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = None
        for fail_dir, fix_dir, key in P:
            for v in a.variants:
                r = dict(task=key[0], setting=key[1], seed=key[2], variant=v, **run_variant(server, fail_dir, fix_dir, key, v))
                print(json.dumps(r), flush=True)
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(r))
                    w.writeheader()
                w.writerow(r)
                fh.flush()
    print(f"wrote {a.out}; summarise: pandas.read_csv(...).groupby('variant')[['action_mse','action_mse_window']].mean()")


if __name__ == "__main__":
    main()
