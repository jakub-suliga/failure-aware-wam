"""LingBot-VA policy server with the fawam test-time methods (docs/EXPERIMENTS.md, A-E).

`FawamServer` subclasses the (recording-patched) upstream `VA_Server`. With the default `Method()` every chunk runs
the same operations in the same order as upstream, so the base results stay bit-comparable with the earlier runs.

Per episode the client sends the method with the reset message (`{"reset": True, "prompt": ..., "method": "ig2"}`).
Extra messages on top of the upstream protocol:
  {"update_prompt": True, "prompt": str|None, "negative_prompt": str|None}   re-encode the text conditioning
  {"context_op": "reset"}            drop the KV cache, the next infer is a first chunk again (frame ids continue)
  {"context_op": "truncate", "keep": k}   drop cached tokens older than k latent frames
  {"event": {...}}                   append a client event (intervention, analysis) to the episode record
  {"decode_pred": True}              last imagined frame per camera (uint8), for the VLM analyzer
compute_kv_cache replies carry the prediction gap of the chunk: {"gap": {cam: cos}, "gap_z": {cam: z}}.

Run (from the repo root, lingbot env):
  python -m torch.distributed.run --nproc_per_node 1 --master_port 29061 -m fawam.server --port 29056 \
      --model_path lingbot-va/checkpoints/lingbot-va-posttrain-robotwin --enable_offload
"""
import argparse
import json
import os
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from fawam import adapters as fa
from fawam.methods import Method, resolve
from fawam.paths import CAMS, ROOT, add_lingbot_to_path

add_lingbot_to_path()
from configs import VA_CONFIGS  # noqa: E402
from distributed.util import init_distributed  # noqa: E402
from utils import data_seq_to_patch, init_logger, logger  # noqa: E402
from utils.sever_utils import run_async_server_mode  # noqa: E402
from wan_va_server import VA_Server  # noqa: E402

KIND_VIDEO, KIND_ACTION = 0, 1


def canvas_from_h5(f, group="obs", rows=None):
    """Per-camera latents stored split in episode.h5 -> T-shaped canvas (1, C, F, 24, 20), float32."""
    g = f[f"model/{group}/latents"]
    sel = slice(None) if rows is None else rows
    hi, lw, rw = g["cam_high"][sel], g["cam_left_wrist"][sel], g["cam_right_wrist"][sel]
    full = np.concatenate([np.concatenate([lw, rw], axis=-1), hi], axis=-2)  # F, C, 24, 20
    return torch.from_numpy(full.astype(np.float32)).permute(1, 0, 2, 3)[None]


def cos_dist(a, b):
    a, b = a.flatten(1).float(), b.flatten(1).float()
    return 1 - (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)


class FawamServer(VA_Server):

    def __init__(self, job_config):
        super().__init__(job_config)
        self.defaults = {k: job_config[k] for k in ("guidance_scale", "action_guidance_scale", "num_inference_steps",
                                                     "action_num_inference_steps")}
        self.holder = fa.install(self.transformer)
        self.method = Method()
        self.adapter_cache = {}
        self.steer_cache = {}
        self.probe = None
        self.ep_frame0 = 0
        self.init_ready = False
        self.pending_prefill = None
        self.task = None
        self._call = None
        self._slot_kind = self._slot_frame = self._excl = None
        self._attn1 = [fa._unwrap(b).attn1 for b in self.transformer.blocks]
        for i, attn in enumerate(self._attn1):
            attn.key_exclude = None
            attn.update_cache = partial(self._tracked_update_cache, attn.update_cache, i == 0)
        stats_path = ROOT / "artifacts" / "gap_stats.json"
        self.gap_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    # ------------------------------------------------------------------------------------------ KV cache tracking

    def _tracked_update_cache(self, orig, first_block, cache_name, key, value, is_pred):
        slots = orig(cache_name, key, value, is_pred)
        if self._excl is not None:
            self._excl[slots] = False
        if first_block and self._slot_kind is not None and self._call is not None:
            kind, frames = self._call
            self._slot_kind[slots] = kind
            self._slot_frame[slots] = frames[:len(slots)] if frames.numel() >= len(slots) else frames[-1]
        return slots

    def _cache(self):
        return self._attn1[0].attn_caches[self.cache_name]

    def _tf(self, input_dict, update_cache, action_mode, frame_st, num_frames):
        """Transformer call with slot bookkeeping (kind + latent frame per written token)."""
        per = self.action_per_frame if action_mode else (self.latent_height // 2) * (self.latent_width // 2)
        frames = torch.arange(frame_st, frame_st + num_frames, device=self.device).repeat_interleave(per)
        self._call = (KIND_ACTION if action_mode else KIND_VIDEO, frames)
        try:
            return self.transformer(input_dict, update_cache=update_cache, cache_name=self.cache_name,
                                    action_mode=action_mode)
        finally:
            self._call = None

    def _exclusion(self, *, pred_video=False, action_before=None):
        """Bool mask over cache slots to hide for the next forward(s)."""
        if self._excl is None:
            return None
        c = self._cache()
        occ = c["mask"]
        m = torch.zeros_like(occ)
        if pred_video:
            m |= occ & c["is_pred"] & (self._slot_kind == KIND_VIDEO)
        if action_before is not None:
            m |= occ & ~c["is_pred"] & (self._slot_kind == KIND_ACTION) & (self._slot_frame < action_before)
        return m if bool(m.any()) else None

    @contextmanager
    def _excluding(self, mask):
        if mask is None:
            yield
            return
        self._excl = mask.clone()
        for attn in self._attn1:
            attn.key_exclude = self._excl
        try:
            yield
        finally:
            for attn in self._attn1:
                attn.key_exclude = None
            self._excl = torch.zeros_like(mask)

    def _new_slot_meta(self):
        n = self._cache()["k"].shape[1]
        self._slot_kind = torch.full((n,), -1, device=self.device, dtype=torch.long)
        self._slot_frame = torch.full((n,), -1, device=self.device, dtype=torch.long)
        self._excl = torch.zeros(n, device=self.device, dtype=torch.bool)

    # ------------------------------------------------------------------------------------------ method setup

    def _apply_method(self, method):
        m = resolve(method, root=ROOT)
        self.method = m
        cfg = self.job_config
        for k, v in self.defaults.items():
            cfg[k] = v
        if m.video_guidance_scale is not None:
            cfg.guidance_scale = m.video_guidance_scale
        if m.action_guidance_scale is not None:
            cfg.action_guidance_scale = m.action_guidance_scale
        if m.num_inference_steps:
            cfg.num_inference_steps = m.num_inference_steps
        if m.action_num_inference_steps:
            cfg.action_num_inference_steps = m.action_num_inference_steps

        aset = None
        if m.adapter_path:
            if m.adapter_path not in self.adapter_cache:
                logger.info(f"loading adapter {m.adapter_path}")
                while len(self.adapter_cache) >= 2:  # Learn-from-Failure writes a new adapter dir per update
                    self.adapter_cache.pop(next(iter(self.adapter_cache)))
                self.adapter_cache[m.adapter_path] = fa.AdapterSet.load(m.adapter_path, device=self.device).eval()
            aset = self.adapter_cache[m.adapter_path]
        fa.attach(self.transformer, aset)
        self.use_inside = aset is not None and m.adapter_use in ("all", "inside")
        self.holder.ctx.enabled = self.use_inside
        self.use_outside = aset is not None and aset.outside is not None and m.adapter_use in ("all", "outside")

        self.holder.ctx.steer = {}
        if m.steer_path and m.steer_alpha:
            if m.steer_path not in self.steer_cache:
                self.steer_cache[m.steer_path] = dict(np.load(m.steer_path, allow_pickle=True))
            s = self.steer_cache[m.steer_path]
            layer = int(m.steer_layer if m.steer_layer is not None else s["layer"])
            key = f"dir_L{layer}"
            vec = torch.from_numpy(s[key]).float().to(self.device)
            self.holder.ctx.steer = {layer: (vec, float(m.steer_alpha), list(m.steer_streams))}

        self.probe = None
        if m.candidate_score == "probe" and m.probe_path:
            self.probe = {k: torch.from_numpy(np.asarray(v)).float().to(self.device) if np.asarray(v).dtype != object
                          else v for k, v in np.load(m.probe_path, allow_pickle=True).items()}
        return m

    def _encode_negative(self, text):
        if not self.use_cfg:
            return
        self.negative_prompt_embeds = self._get_t5_prompt_embeds(prompt=[text or ""], max_sequence_length=512)

    def _reset(self, prompt=None, seed=None, method=None, task=None):
        self._apply_method(method)
        super()._reset(prompt=prompt, seed=seed)
        if self.method.negative_prompt:
            self._encode_negative(self.method.negative_prompt)
        self.prompt = prompt
        self.task = task
        self.ep_frame0 = 0
        self.init_ready = False
        self._new_slot_meta()
        self.episode_log.update(method=self.method.short(), events=[], gaps=[], cand=[], prefill=[])
        self.pending_prefill = self.method.prefill
        rng_seed = 0 if seed is None else int(seed)
        self._rng = np.random.default_rng(rng_seed + 12345)

    # ------------------------------------------------------------------------------------------ protocol

    @torch.no_grad()
    def infer(self, obs):
        if obs.get("reset", False):
            logger.info("******************* Reset server ******************")
            self._reset(prompt=obs.get("prompt"), seed=obs.get("seed"), method=obs.get("method"),
                        task=obs.get("task_name"))
            if obs.get("prefill"):
                self.pending_prefill = obs["prefill"]
            elif obs.get("prefill_paths"):
                self.pending_prefill = dict(paths=list(obs["prefill_paths"]), gap=int(obs.get("prefill_gap", 1)),
                                            include_pred=bool(obs.get("prefill_include_pred", False)))
            return dict(method=self.method.short())
        if obs.get("update_prompt", False):
            if obs.get("prompt"):
                self.prompt_embeds, _ = self.encode_prompt(prompt=obs["prompt"], do_classifier_free_guidance=False,
                                                           max_sequence_length=512, device=self.device, dtype=self.dtype)
            if "negative_prompt" in obs:
                self._encode_negative(obs["negative_prompt"])
            self.episode_log["events"].append(dict(kind="update_prompt", frame=int(self.frame_st_id),
                                                   prompt=obs.get("prompt"),
                                                   negative_prompt=obs.get("negative_prompt"), t=time.time()))
            return dict()
        if obs.get("context_op"):
            return self._context_op(obs["context_op"], obs)
        if obs.get("decode_pred"):
            return self._decode_last_pred()
        if obs.get("event"):
            self.episode_log["events"].append(dict(obs["event"], frame=int(self.frame_st_id), t=time.time()))
            return dict()
        if obs.get("finalize", False) or obs.get("compute_kv_cache", False):
            if obs.get("compute_kv_cache", False):
                return self._compute_kv_cache(obs) or dict()
            return super().infer(obs)
        # one chunk
        if self.pending_prefill is not None and self.frame_st_id == self.ep_frame0 and not self.init_ready:
            self.init_latent = self._encode_obs(obs)
            self.init_ready = True
            self._prefill(self.pending_prefill)
            self.pending_prefill = None
        frame_st_id = self.frame_st_id
        action, _ = self._infer(obs, frame_st_id=frame_st_id)
        return dict(action=action, frame_st_id=frame_st_id)

    def _context_op(self, op, obs):
        if op == "reset":
            # Fresh context, same episode: frame ids continue so the episode record stays monotonic
            self.transformer.clear_cache(self.cache_name)
            patch = self.job_config.patch_size
            lat_tok = (self.job_config.frame_chunk_size * self.latent_height * self.latent_width) // (
                patch[0] * patch[1] * patch[2])
            self.transformer.create_empty_cache(self.cache_name, self.job_config.attn_window, lat_tok,
                                                self.job_config.frame_chunk_size * self.action_per_frame,
                                                dtype=self.dtype, device=self.device,
                                                batch_size=2 if self.use_cfg else 1)
            self.streaming_vae.clear_cache()
            if self.streaming_vae_half is not None:
                self.streaming_vae_half.clear_cache()
            self._new_slot_meta()
            self.ep_frame0 = self.frame_st_id
            self.init_ready = False
            if obs.get("prefill"):  # reset-with-memory: the next first chunk starts after this in-context memory
                self.pending_prefill = obs["prefill"]
        elif op == "truncate":
            keep = int(obs.get("keep", 2))
            c = self._cache()
            drop = c["mask"] & (self._slot_frame < self.frame_st_id - keep) & (self._slot_frame >= 0)
            for attn in self._attn1:
                cc = attn.attn_caches[self.cache_name]
                cc["mask"][drop] = False
                cc["id"][drop] = -1
        else:
            raise ValueError(f"unknown context_op {op}")
        self.episode_log["events"].append(dict(kind=f"context_{op}", frame=int(self.frame_st_id), t=time.time()))
        return dict(frame_st_id=int(self.frame_st_id))

    def _decode_last_pred(self):
        """Last frame of the most recent imagined chunk per camera, decoded to uint8 (for the VLM analyzer)."""
        if not self.episode_log['pred']:
            return dict(frames=[])
        lat = self.episode_log['pred'][-1]['latents']  # C, F, H, W
        frames = []
        for cam, l in self._split_cams(lat).items():
            frames.append(self._decode_latents_uint8(l)[-1])
        return dict(frames=frames)

    # ------------------------------------------------------------------------------------------ in-context prefill

    def _bank_paths(self, item):
        """Resolve a bank item {"bank", "select": nearest|random|same_seed, "k", "seed"} to episode paths."""
        bank = Path(item["bank"])
        index = json.loads((bank / "index.json").read_text())
        cands = [e for e in index if self.task is None or e["task"] == self.task]
        if item.get("setting"):
            cands = [e for e in cands if e.get("setting") == item["setting"]]
        if not cands:
            logger.warning(f"demo bank {bank} has no episode for task {self.task}; no prefill")
            return []
        k = int(item.get("k", 1))
        sel = item.get("select", "nearest")
        if sel == "same_seed":
            hit = [e for e in cands if int(e.get("seed", -1)) == int(item.get("seed", -2))]
            return [str(bank / e["path"]) for e in hit[:k]]
        if sel == "random":
            idx = self._rng.choice(len(cands), size=min(k, len(cands)), replace=False)
            return [str(bank / cands[i]["path"]) for i in idx]
        q = self.init_latent[0, :, 0, 8:].float().flatten()  # nearest: initial head-camera latent
        feats = torch.from_numpy(np.load(bank / "init_latents.npy")).to(q.device)
        sel_i = torch.tensor([i for i, e in enumerate(index) if e in cands], device=q.device)
        sims = F.cosine_similarity(feats[sel_i], q[None], dim=1)
        return [str(bank / index[i]["path"]) for i in sel_i[sims.argsort(descending=True)[:k]].tolist()]

    def _context_items(self, spec):
        """Normalise a prefill spec to a list of items {path, groups, segment, actions, role}.
        Legacy keys (paths / bank / k / select / include_pred) are still accepted."""
        items = []
        for it in spec.get("items", []):
            if it.get("bank"):
                for p in self._bank_paths(it):
                    items.append({**it, "path": p})
            else:
                items.append(dict(it))
        legacy = list(spec.get("paths", []))
        if spec.get("bank") and not spec.get("items"):
            legacy += self._bank_paths(spec)
        groups = ["pred", "obs"] if spec.get("include_pred") else ["obs"]
        items += [dict(path=p, groups=groups) for p in legacy]
        return items

    def _segment_rows(self, li, item, ana):
        """Rows of an episode to write: all, or a window around the analysed failure ({"before": b, "after": a})."""
        seg = item.get("segment", "full")
        if seg == "full" or not ana or ana.get("latent") is None:
            return np.arange(len(li))
        b, a = (seg.get("before", 4), seg.get("after", 2)) if isinstance(seg, dict) else (4, 2)
        L = int(ana["latent"])
        return np.where((li >= L - b) & (li <= L + a))[0]

    def _prefill(self, spec):
        """In-context memory (docs/ICL.md): write recorded episodes into the KV cache before the episode (or, after a
        context reset, before the next chunk). Per item:
          groups   which latents: obs (what happened), pred (what the model imagined), diff (obs - pred: where the
                   imagination was wrong); several groups = several consecutive segments
          segment  full | {"before": b, "after": a} latents around the analysed failure (analysis.json)
          actions  executed | corrected (analysis correction applied: what should have been done)
        Global: gap (RoPE frames between segments and before the episode), position sequential | aligned (each segment
        occupies the same frame ids the coming episode will use = phase-aligned memory), attn_scale (keys of the
        context tokens scaled: >1 makes the model attend to the memory more)."""
        from fawam.latent_data import relabel_planned
        items = self._context_items(spec)
        gap = int(spec.get("gap", 1))
        aligned = spec.get("position", "sequential") == "aligned"
        start = int(self.frame_st_id)
        base = start
        mask_before = self._cache()["mask"].clone()
        for it in items:
            ep = Path(it["path"])
            ep = ep / "episode.h5" if ep.is_dir() else ep
            ana_p = ep.parent / it.get("analysis", "analysis.json")
            ana = json.loads(ana_p.read_text()) if ana_p.exists() else None
            for group in it.get("groups", ["obs"]):
                with h5py.File(ep, "r") as f:
                    src = "pred" if group == "pred" else "obs"
                    if f"model/{src}" not in f:
                        continue
                    lat = canvas_from_h5(f, src)
                    li = f[f"model/{src}/latent_index"][()]
                    chunk_of = f[f"model/{src}/chunk_id"][()]
                    planned = f["actions/planned"][()]
                    if group == "diff":
                        pl = f["model/pred/latent_index"][()]
                        pidx = {int(k): i for i, k in enumerate(pl)}
                        pred = canvas_from_h5(f, "pred")
                        keep = np.array([int(k) in pidx for k in li])
                        lat = lat[:, :, keep] - pred[:, :, [pidx[int(k)] for k in li[keep]]]
                        li, chunk_of = li[keep], chunk_of[keep]
                if it.get("actions") == "corrected":
                    planned = relabel_planned(planned, chunk_of, li, ana)
                rows = self._segment_rows(li, it, ana)
                if len(rows) == 0:
                    continue
                lat = lat.to(self.device, self.dtype)
                li0, li1 = int(li[rows[0]]), int(li[rows[-1]])
                seg0 = start if aligned else base
                for c in np.unique(chunk_of[rows]):
                    crow = rows[chunk_of[rows] == c]
                    first_row = np.where(chunk_of == c)[0][0]
                    lat_c = lat[:, :, crow]
                    act = planned[c][:, crow - first_row]
                    act_c = self.preprocess_action(act).to(lat_c)
                    fst = seg0 + int(li[crow[0]]) - li0
                    input_dict = self._prepare_latent_input(lat_c, act_c, frame_st_id=fst)
                    self._tf(self._repeat_input_for_cfg(input_dict['latent_res_lst']), 2, False, fst, len(crow))
                    self._tf(self._repeat_input_for_cfg(input_dict['action_res_lst']), 2, True, fst, len(crow))
                if not aligned:
                    base += li1 - li0 + 1 + gap
                self.episode_log["prefill"].append(dict(path=str(ep), group=group, actions=it.get("actions", "executed"),
                                                        frames=[li0, li1], role=it.get("role")))
        new_slots = self._cache()["mask"] & ~mask_before
        alpha = float(spec.get("attn_scale", 1.0))
        if alpha != 1.0 and new_slots.any():
            for attn in self._attn1:  # q.k logits of the memory tokens scale with alpha (keys are post-norm/RoPE)
                attn.attn_caches[self.cache_name]["k"][:, new_slots] *= alpha
        self.ctx_slots = new_slots
        self.ep_frame0 = self.frame_st_id = start if aligned else base
        logger.info(f"in-context memory: {len(items)} item(s), {int(new_slots.sum())} tokens, episode continues at "
                    f"frame {self.frame_st_id} ({'aligned' if aligned else 'sequential'})")

    # ------------------------------------------------------------------------------------------ inference

    def _video_step(self, latents, t, latent_cond, frame_st_id, update_cache):
        input_dict = self._prepare_latent_input(latents, None, t, t, latent_cond, None, frame_st_id=frame_st_id)
        return self._tf(self._repeat_input_for_cfg(input_dict['latent_res_lst']), update_cache, False, frame_st_id,
                        self.job_config.frame_chunk_size)

    def _denoise_video(self, latents, timesteps, latent_cond, frame_st_id, first):
        cfg = self.job_config
        fcs = cfg.frame_chunk_size
        records = []
        excl = self._exclusion(action_before=None if self.method.video_history_keep is None
                               else frame_st_id - self.method.video_history_keep)
        with self._excluding(excl):
            for i, t in enumerate(timesteps[:-1]):
                pred = self._video_step(latents, t, latent_cond, frame_st_id, 0)
                pred = data_seq_to_patch(cfg.patch_size, pred, fcs, self.latent_height, self.latent_width,
                                         batch_size=2 if self.use_cfg else 1)
                if cfg.guidance_scale > 1:
                    pred = pred[1:] + cfg.guidance_scale * (pred[:1] - pred[1:])
                else:
                    pred = pred[:1]
                if i in self.record_denoise_steps:
                    sigma = float(t) / self.scheduler.num_train_timesteps
                    x0_hat = latents.float() - sigma * pred.float()
                    records.append(dict(step=i, timestep=float(t), sigma=sigma, x_t=latents[0].half().cpu(),
                                        x0_hat=x0_hat[0].half().cpu()))
                latents = self.scheduler.step(pred, t, latents, return_dict=False)
                latents[:, :, 0:1] = latent_cond if first else latents[:, :, 0:1]
        return latents, records

    def _cam_region(self, latents, cams):
        return torch.cat([latents[..., hs, ws].flatten(1) for cam, (hs, ws) in self.LATENT_CAM_SLICES.items()
                          if cam in cams], dim=1)

    def _score_candidates(self, cands, latent_cond, frame_st_id, noise_seed):
        m = self.method
        n = len(cands)
        if n == 1 or m.candidate_score == "first":
            return 0, [0.0] * n
        if m.candidate_score == "random":
            return int(np.random.default_rng(noise_seed).integers(n)), [0.0] * n
        if m.candidate_score == "medoid":
            x = torch.stack([self._cam_region(c[0, :, -1:].float(), m.score_cams).flatten() for c in cands])
            x = F.normalize(x, dim=1)
            d = 1 - x @ x.t()
            scores = d.sum(1)
            return int(scores.argmin()), scores.tolist()
        if m.candidate_score == "probe":
            p = self.probe
            scores = []
            t0 = torch.zeros((), device=self.device)
            for c in cands:
                _, hidden = self._run_with_hidden_capture(
                    self.job_config.frame_chunk_size, partial(self._video_step, c, t0, latent_cond, frame_st_id, 0))
                li = int(p["layer_index"])
                feat = hidden[:, li, :, :].float().mean(0)[p["cams"].long().cpu()].flatten().to(self.device)
                z = (feat - p["mean"]) / p["std"]
                scores.append(float(torch.sigmoid(z @ p["w"] + p["b"])))
            return int(np.argmin(scores)), scores
        raise ValueError(f"unknown candidate_score {m.candidate_score}")

    def _action_noise(self, actions, t, action_cond, frame_st_id, update_cache, excl):
        cfg = self.job_config
        input_dict = self._prepare_latent_input(None, actions, t, t, None, action_cond, frame_st_id=frame_st_id)
        with self._excluding(excl):
            out = self._tf(self._repeat_input_for_cfg(input_dict['action_res_lst']), update_cache, True, frame_st_id,
                           cfg.frame_chunk_size)
        if update_cache == 1:
            return None
        out = rearrange(out, 'b (f n) c -> b c f n 1', f=cfg.frame_chunk_size)
        if cfg.action_guidance_scale > 1:
            return out[1:] + cfg.action_guidance_scale * (out[:1] - out[1:])
        return out[:1]

    def _infer(self, obs, frame_st_id=0):
        m, cfg = self.method, self.job_config
        fcs = cfg.frame_chunk_size
        first = frame_st_id == self.ep_frame0
        assert cfg.video_exec_step == -1, "fawam server assumes full video denoising (video_exec_step=-1)"
        if first and not self.init_ready:
            self.init_latent = self._encode_obs(obs)
        self.init_ready = False  # consumed: the next first chunk (after a context reset) encodes again
        init_latent = self.init_latent

        noise_seed = (self.episode_log['seed'] * 10007 + len(self.episode_log['pred'])) % 2**31
        generator = torch.Generator(device=self.device).manual_seed(noise_seed)
        shape_v = (1, 48, fcs, self.latent_height, self.latent_width)
        latents = torch.randn(*shape_v, generator=generator, device=self.device, dtype=self.dtype)
        actions = torch.randn(1, cfg.action_dim, fcs, self.action_per_frame, 1, generator=generator,
                              device=self.device, dtype=self.dtype)
        cands = [latents]
        for k in range(1, max(1, m.num_candidates)):
            g = torch.Generator(device=self.device).manual_seed((noise_seed + 7919 * k) % 2**31)
            cands.append(torch.randn(*shape_v, generator=g, device=self.device, dtype=self.dtype))

        self.scheduler.set_timesteps(cfg.num_inference_steps)
        self.action_scheduler.set_timesteps(cfg.action_num_inference_steps)
        timesteps = F.pad(self.scheduler.timesteps, (0, 1), mode='constant', value=0)
        action_timesteps = F.pad(self.action_scheduler.timesteps, (0, 1), mode='constant', value=0)
        latent_cond = init_latent[:, :, 0:1].to(self.dtype) if first else None

        with torch.no_grad():
            results = [self._denoise_video(c, timesteps, latent_cond, frame_st_id, first) for c in cands]
            idx, scores = self._score_candidates([r[0] for r in results], latent_cond, frame_st_id, noise_seed)
            latents, denoise_records = results[idx]
            if self.use_outside:
                with torch.autocast("cuda", enabled=False):
                    latents = self.holder.adapters.outside(latents.float()).to(self.dtype)
                if first:
                    latents[:, :, 0:1] = latent_cond
            # commit the (chosen, corrected) imagination into the KV cache; pooled hidden states for the record
            _, hidden = self._run_with_hidden_capture(
                fcs, partial(self._video_step, latents, timesteps[-1], latent_cond, frame_st_id, 1))
            latents[:, :, 0:1] = latent_cond if first else latents[:, :, 0:1]

            hist = None if m.action_history_keep is None else frame_st_id - m.action_history_keep
            excl_c = self._exclusion(action_before=hist)
            use_ig = m.imagination_guidance != 1.0
            n_steps = len(action_timesteps)
            for i, t in enumerate(action_timesteps):
                last = i == n_steps - 1
                action_cond = torch.zeros([1, cfg.action_dim, 1, self.action_per_frame, 1], device=self.device,
                                          dtype=self.dtype) if first else None
                if last:
                    self._action_noise(actions, t, action_cond, frame_st_id, 1, excl_c)
                else:
                    pred = self._action_noise(actions, t, action_cond, frame_st_id, 0, excl_c)
                    if use_ig and i < m.imagination_guidance_until * (n_steps - 1):
                        excl_u = self._exclusion(pred_video=True, action_before=hist)
                        pred_u = self._action_noise(actions, t, action_cond, frame_st_id, 0, excl_u)
                        pred = pred_u + m.imagination_guidance * (pred - pred_u)
                    actions = self.action_scheduler.step(pred, t, actions, return_dict=False)
                actions[:, :, 0:1] = action_cond if first else actions[:, :, 0:1]

        if m.action_noise:
            g = torch.Generator(device=self.device).manual_seed((noise_seed + 31337) % 2**31)
            first_ok = 1 if first else 0  # never perturb the conditioning frame
            actions[:, :, first_ok:] += m.action_noise * torch.randn(actions[:, :, first_ok:].shape, generator=g,
                                                                     device=self.device, dtype=actions.dtype)
        actions[:, ~self.action_mask] *= 0
        self.episode_log['pred'].append(dict(frame_st_id=frame_st_id, wall_time=time.time(), noise_seed=noise_seed,
                                             latents=latents[0].detach().cpu(), denoise=denoise_records,
                                             hidden=hidden))
        self.episode_log['cand'].append(dict(frame_st_id=frame_st_id, chosen=idx, scores=scores))
        actions = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        return actions, latents

    def _compute_kv_cache(self, obs):
        self.transformer.clear_pred_cache(self.cache_name)
        latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == self.ep_frame0:
            latent_model_input = torch.cat([self.init_latent, latent_model_input], dim=2) \
                if latent_model_input is not None else self.init_latent
        return self._commit_observation(latent_model_input, obs['state'])

    def _commit_observation(self, latent_model_input, state):
        """Write observed latents (already including the initial frame for a first chunk) and the executed action
        chunk into the KV cache, replacing the chunk's imagination. Returns the prediction gap of the chunk."""
        self.transformer.clear_pred_cache(self.cache_name)
        obs_record = dict(frame_st_id=self.frame_st_id, wall_time=time.time(),
                          latents=latent_model_input[0].detach().cpu())
        self.episode_log['obs'].append(obs_record)
        reply = self._gap(latent_model_input)

        action_model_input = self.preprocess_action(state).to(latent_model_input)
        input_dict = self._prepare_latent_input(latent_model_input, action_model_input, frame_st_id=self.frame_st_id)
        n = latent_model_input.shape[2]
        with torch.no_grad():
            _, obs_record['hidden'] = self._run_with_hidden_capture(
                n, partial(self._tf, self._repeat_input_for_cfg(input_dict['latent_res_lst']), 2, False,
                           self.frame_st_id, n))
            self._tf(self._repeat_input_for_cfg(input_dict['action_res_lst']), 2, True, self.frame_st_id, n)
        torch.cuda.empty_cache()
        self.frame_st_id += n
        return reply

    def _gap(self, obs_latents):
        """Cosine distance predicted vs. observed latent per camera (max over the chunk's new frames), z-scored with
        the success statistics of the task (artifacts/gap_stats.json, from fawam.probe gap-stats)."""
        if not self.episode_log['pred']:
            return dict()
        pred = self.episode_log['pred'][-1]
        if pred['frame_st_id'] != self.frame_st_id:
            return dict()
        p = pred['latents'][None].float()
        o = obs_latents.detach().cpu().float()
        nf = min(p.shape[2], o.shape[2])
        start = 1 if self.frame_st_id == self.ep_frame0 else 0  # frame 0 of a first chunk is the conditioning frame
        gap, gap_z = {}, {}
        stats = self.gap_stats.get(self.task) or self.gap_stats.get("_all", {})
        for cam, (hs, ws) in self.LATENT_CAM_SLICES.items():
            d = cos_dist(p[0, :, start:nf, hs, ws].permute(1, 0, 2, 3), o[0, :, start:nf, hs, ws].permute(1, 0, 2, 3))
            if d.numel() == 0:
                continue
            gap[cam] = float(d.max())
            if cam in stats:
                gap_z[cam] = (gap[cam] - stats[cam]["mean"]) / max(stats[cam]["std"], 1e-6)
        self.episode_log['gaps'].append(dict(frame_st_id=int(self.frame_st_id), gap=gap, gap_z=gap_z))
        return dict(gap=gap, gap_z=gap_z)

    # ------------------------------------------------------------------------------------------ record

    def _finalize_episode(self, episode_dir):
        log = self.episode_log
        out = super()._finalize_episode(episode_dir)
        if not out.get("saved"):
            return out
        with h5py.File(os.path.join(episode_dir, 'episode.h5'), 'a') as f:
            g = f.require_group('fawam')
            g.attrs['method'] = json.dumps(log.get('method', {}), default=str)
            g.attrs['events'] = json.dumps(log.get('events', []), default=str)
            g.attrs['prefill'] = json.dumps(log.get('prefill', []))
            g.attrs['ep_frame0'] = int(self.ep_frame0)
            cand = log.get('cand', [])
            if cand:
                g['cand_chosen'] = np.array([c['chosen'] for c in cand])
                n = max(len(c['scores']) for c in cand)
                g['cand_scores'] = np.array([c['scores'] + [np.nan] * (n - len(c['scores'])) for c in cand])
            gaps = log.get('gaps', [])
            if gaps:
                g['gap_frame_st_id'] = np.array([x['frame_st_id'] for x in gaps])
                for cam in CAMS:
                    g[f'gap/{cam}'] = np.array([x['gap'].get(cam, np.nan) for x in gaps])
                    g[f'gap_z/{cam}'] = np.array([x['gap_z'].get(cam, np.nan) for x in gaps])
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="robotwin")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--enable_offload", action="store_true")
    parser.add_argument("--save_root", default=None)
    args = parser.parse_args()
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    if args.model_path is not None:
        config.wan22_pretrained_model_name_or_path = args.model_path
    if args.enable_offload:
        config.enable_offload = True
    rank, local_rank = int(os.getenv("RANK", 0)), int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank, config.local_rank, config.world_size = rank, local_rank, world_size
    model = FawamServer(config)
    logger.info("****************************** fawam server ******************************")
    run_async_server_mode(model, local_rank, config.host, port)


if __name__ == "__main__":
    init_logger()
    main()
