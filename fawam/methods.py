"""Test-time method configuration shared by the policy server (fawam/server.py) and the RoboTwin client hooks
(fawam/robotwin_hooks.py). One `Method` describes everything that differs from plain LingBot-VA inference.

A method is given as: a preset name ("ig2"), several presets joined with "+" ("ig2+reset_on_gap"), a JSON/YAML file,
a JSON string, or a dict. Later parts override earlier ones; dict values are merged one level deep.
The experiment IDs (A1, B3, ...) refer to docs/EXPERIMENTS.md.
"""
import os
import copy
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

WRIST = ["cam_left_wrist", "cam_right_wrist"]


@dataclass
class Method:
    name: str = "base"

    # --- sampling and guidance (server) -------------------------------------------------------------------------
    num_candidates: int = 1            # A3 best-of-N video samples per chunk
    candidate_score: str = "medoid"    # medoid | probe | random | first
    score_cams: list = field(default_factory=lambda: list(WRIST))
    probe_path: str = None             # fawam.probe output (.npz) for candidate_score=probe
    video_guidance_scale: float = None  # None = server config (5)
    action_guidance_scale: float = None  # None = server config (1)
    imagination_guidance: float = 1.0  # A1: a = a(no pred frames) + w * (a(pred frames) - a(no pred frames))
    imagination_guidance_until: float = 1.0  # fraction of the action denoising steps that use it
    action_history_keep: int = None    # A2: latent frames of past action tokens visible to the action stream
    video_history_keep: int = None     # A2: latent frames of past action tokens visible to the video stream
    negative_prompt: str = None        # A5: failure description used as the CFG negative
    num_inference_steps: int = None
    action_num_inference_steps: int = None
    action_noise: float = 0.0          # Gaussian noise (normalized action units) on the executed actions (exploration)
    # teacher forcing (diagnostic, user request 2026-11-19): "oracle" = the client renders the expert's next chunk
    # from the CURRENT sim state (look-ahead, then restore) and the server uses those latents instead of the imagined
    # video, so the action stream sees a perfect prediction of the correct continuation (fawam/teacher_force.py)
    teacher_force: str = None
    # p14 (user request 2026-11-19): oracle latents are corrupted before use, flow-matching style
    # lat = (1 - s) * lat + s * eps (eps ~ N(0, 1), first conditioning frame untouched) -> "slightly imperfect video"
    teacher_force_noise: float = 0.0

    # --- steering (server, A6) ------------------------------------------------------------------------------------
    steer_path: str = None             # fawam.probe steering output (.npz)
    steer_alpha: float = 0.0
    steer_layer: int = None            # None = layer stored in the file
    steer_streams: list = field(default_factory=lambda: ["video", "action"])

    # --- adaptation (server, D) -------------------------------------------------------------------------------------
    adapter_path: str = None           # fawam.train_adapter output dir
    adapter_use: str = "all"           # all | inside | outside
    hid_residual: dict = None          # M3 (fawam/train_hid.py): {"path": head dir, "ctx_ep": failed episode or None}
    # plan v2 (2026-11-21) TTT P2: adapter strength per chunk = normalised prediction gap of the previous attempt at the
    # same chunk index (time-aligned): {"ref_ep": failed episode dir}
    adapter_gate: dict = None
    # plan v2 TTT P3: exploration in imagination. Inside the failure window of the previous attempt (latent frames
    # [latent - before, latent + after] around its gap peak) sample n video plans and keep the one FARTHEST from what
    # really happened there: {"ref_ep", "latent", "n": 8, "before": 4, "after": 2}
    explore: dict = None
    skip_generation: bool = False      # p21 data driver: no denoising at all (expert continuations only need encoding)

    # --- in-context (server + client, E) --------------------------------------------------------------------------
    prefill: dict = None               # {"bank": dir, "k": 1, "select": nearest|random|same_task, "gap": 1}
    retry: bool = False                # E2/E4 baseline (client): after a failure, rerun the same seed once more
    retry_with_memory: bool = False    # E2: ... with the failed attempt prefilled into the KV cache
    text_memory: bool = False          # E4: ... with a description of the previous failure appended to the prompt

    # --- Learn-from-Failure (client + learner, docs/METHOD.md) ----------------------------------------------------
    learn: dict = None                 # see LEARN_DEFAULTS

    trigger_prefill: dict = None       # memory spec inserted by the intervention action "reset_memory"
    # does the model use the memory? (docs/ICL.md, section 6c) - measured whenever a memory is in the KV cache
    measure_memory: bool = True        # attention mass on memory tokens per layer + influence at one denoising step
    memory_counterfactual: str = "step"  # step | full (whole action chunk again without memory: delta in metres) | off
    memory_imagination: bool = False   # also denoise the imagined frames without memory; pose probe on both

    # --- online intervention (client, B/C) ---------------------------------------------------------------------
    intervention: dict = None          # see INTERVENTION_DEFAULTS
    analyzer: str = None               # None | oracle | vlm
    vlm: dict = None                   # {"endpoint": "http://host:port/v1", "model": "...", "pred_frames": true}

    @property
    def uses_client_logic(self):
        return bool(self.intervention or self.retry or self.analyzer or self.learn)

    def to_dict(self):
        return asdict(self)

    def short(self):
        """Non-default fields only, for logs and episode metadata."""
        base = Method()
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) != getattr(base, f.name) or f.name == "name"}


INTERVENTION_DEFAULTS = dict(
    trigger=dict(kind="gap", z=2.5, cams=list(WRIST), min_chunk=1, consecutive=1),  # gap | oracle | always_after
    action="reset_context",      # reset_context | truncate | reprompt | negative_prompt | analyze
    truncate_keep=2,             # latent frames kept by `truncate`
    reprompt_template="{instruction} The previous grasp missed: move the gripper lower and grasp the object again.",
    negative_template="The gripper closes in the air above the object without touching it.",
    max_interventions=2,
    cooldown_chunks=2,
)

LEARN_DEFAULTS = dict(
    mode="adapter",          # adapter | icl | both  (how the failure is given to the model for the next attempt)
    analyzer="oracle",       # legacy: oracle | vlm | none; superseded by `signals`
    signals=None,            # subset of gap, rollout, imagine, vlm, oracle (fawam.failure_analysis.analyze_signals)
    init="prev",             # prev: continue the last adapter | base: fresh adapter from the base each update (FailBank)
    gap_filter=0.0,          # T3VF: train only on frames above this prediction-gap quantile
    pref_beta=0.0,           # FAR: failure-contrastive preference loss (relabeled vs executed actions); ~500
    guard=False,             # FailBank: accept an update only if it does not hurt own successful episodes
    retry_action_noise=0.0,  # FAR: Gaussian exploration noise (normalized action units) in the retry
    every_episode=False,     # JEPA-TTT: update after every episode (success or failure), persistently, no retry
    adapt_episodes=None,     # paper protocol: with every_episode, update only after the first N episodes, then freeze
                             # the adapter and evaluate the remaining episodes of the run (meta lff.phase adapt|eval)
    outside="none",          # external residual on the imagined latent: none | parallel (recorded predictions) |
                             # sequential (paper: predictions recomputed with the adapted model, fawam.replay)
    outside_steps=100,
    max_attempts=2,          # attempts per seed (attempt 1 + max_attempts-1 updates)
    persist=False,           # keep the adapter / memory across seeds (lifelong: the next seed starts adapted)
    memory="all",            # adapter training data: all | last failed episode(s) of the memory
    kind=None,               # learner adapter kind: None = dual (Adapter+ inside) | inside | lora
    inside_streams=None,     # None = video + action tokens | ["video"] | ["action"]
    lr=None,                 # inside learning rate (None = train_adapter default 4.3e-4)
    steps=150, gap_beta=2.0, relabel=True, video_in_window="skip", lambda_video=1.0, lambda_action=1.0,
    include_pred=True,       # icl: imagined + observed failed attempt in the context
    prompt="correction",     # next attempt's prompt: correction (analysis) | instruction
    negative=False,          # next attempt's CFG negative = analysis.negative_prompt
    hid_head=None,           # mode hidres: M3 head dir (fawam/train_hid.py output/<config>)
    hid_ctx="own",           # mode hidres: own (the failed attempt) | shuf (a failure of another seed) | none
    hid_shuf_root="results/sweeps/p10_hid_data/base/hid_trans",  # failures for hid_ctx shuf (other seeds, other o)
    meta_adapter=None,       # icl/both: adapter meta-trained to use failures in context (FailFix training)
    icl=None,                # in-context memory for the next attempt, see ICL_DEFAULTS (docs/ICL.md)
    keep_successes=True,     # persist: successful attempts join the memory (trained without relabel)
)

ICL_DEFAULTS = dict(
    content=["fail"],        # ordered memory items: fail (last own failure) | fails (all failures of the seed, up to
                             # max_items) | demo_same_seed | demo_nearest | demo_random (expert replays from demo_bank)
                             # | success_own (own earlier successes, persist mode)
    groups=["pred", "obs"],  # for failure items: obs (what happened) | pred (what was imagined) | diff (obs - pred)
    segment="full",          # full | window (latents around the analysed failure)
    window=dict(before=4, after=2),
    actions="executed",      # failure items: executed | corrected (analysis correction written as the actions)
    position="sequential",   # sequential (memory before the episode in time) | aligned (same frame ids = phase-aligned)
    gap=1,                   # RoPE frames between memory segments / before the episode
    attn_scale=1.0,          # >1: scale the memory keys (more attention on the memory)
    text="correction",       # correction | instruction | negative | correction+negative | failure_note
    timing="pre",            # pre (memory before the retry) | trigger (insert at the prediction-gap trigger during
                             # the retry: context reset + memory) | both
    demo_bank="artifacts/demo_bank/{setting}",
    max_items=3,
    counterfactual="full",   # memory usage measurement in the attempt with memory: step | full | full_video | off
    manip=None,              # content manipulation of the failure memory: shuffle_time | reverse_time | noise |
                             # blank_head | blank_wrists | no_actions (docs/ICL.md 6a)
    fail_bank="artifacts/fail_bank/{setting}",  # failures of other seeds (content fail_other_seed)
)

# ---------------------------------------------------------------------------------------------- presets
PRESETS = {
    "base": {},
    "tf_oracle": dict(teacher_force="oracle"),
    "tf_replan": dict(teacher_force="replan"),  # oracle = task planner re-run from the current state before EVERY chunk
    **{f"tf_replan_n{int(round(_s * 100)):02d}": dict(teacher_force="replan", teacher_force_noise=_s)
       for _s in (0.1, 0.25, 0.5, 0.75)},
    # A1 imagination guidance
    "ig1.5": dict(imagination_guidance=1.5),
    "ig2": dict(imagination_guidance=2.0),
    "ig3": dict(imagination_guidance=3.0),
    "ig2_half": dict(imagination_guidance=2.0, imagination_guidance_until=0.5),
    # A2 history dropout
    "acthist0": dict(action_history_keep=0),
    "acthist2": dict(action_history_keep=2),
    "acthist4": dict(action_history_keep=4),
    "vidhist0": dict(video_history_keep=0),
    # A3 best-of-N
    "bon4_medoid": dict(num_candidates=4, candidate_score="medoid"),
    "bon8_medoid": dict(num_candidates=8, candidate_score="medoid"),
    "bon4_random": dict(num_candidates=4, candidate_score="random"),
    "bon4_probe": dict(num_candidates=4, candidate_score="probe", probe_path="artifacts/probe/failure_probe.npz"),
    # A4 guidance sweeps
    "vg3": dict(video_guidance_scale=3.0),
    "vg7": dict(video_guidance_scale=7.0),
    "ag2": dict(action_guidance_scale=2.0),
    "ag3": dict(action_guidance_scale=3.0),
    # A5 failure-aware negative prompt
    "neg_missed": dict(negative_prompt="The gripper closes in the air above the object without touching it."),
    "neg_drop": dict(negative_prompt="The object slips out of the gripper and falls onto the table."),
    "neg_generic": dict(negative_prompt="The robot fails: it misses the object, drops it or knocks it over."),
    # A6 steering
    "steer_l19_a1": dict(steer_path="artifacts/probe/steer.npz", steer_layer=19, steer_alpha=1.0),
    "steer_l19_a3": dict(steer_path="artifacts/probe/steer.npz", steer_layer=19, steer_alpha=3.0),
    # A7 more compute in the denoiser
    "steps50": dict(num_inference_steps=50, action_num_inference_steps=100),
    # B online intervention on a prediction gap
    "reset_on_gap": dict(intervention=dict(action="reset_context")),
    "truncate_on_gap": dict(intervention=dict(action="truncate")),
    "reprompt_on_gap": dict(intervention=dict(action="reprompt")),
    "neg_on_gap": dict(intervention=dict(action="negative_prompt")),
    "reset_on_oracle": dict(intervention=dict(action="reset_context", trigger=dict(kind="oracle"))),
    # C VLM / oracle analysis on a gap
    "oracle_on_gap": dict(analyzer="oracle", intervention=dict(action="analyze")),
    "vlm_on_gap": dict(analyzer="vlm", intervention=dict(action="analyze"),
                       vlm=dict(endpoint="http://127.0.0.1:8001/v1", model="Qwen/Qwen2.5-VL-7B-Instruct",
                                pred_frames=True)),
    # E in-context
    "icl_demo1": dict(prefill=dict(bank="artifacts/demo_bank", k=1, select="nearest", gap=1)),
    "icl_demo2": dict(prefill=dict(bank="artifacts/demo_bank", k=2, select="nearest", gap=1)),
    "icl_demo1_random": dict(prefill=dict(bank="artifacts/demo_bank", k=1, select="random", gap=1)),
    "retry_plain": dict(retry=True),
    "retry_memory": dict(retry=True, retry_with_memory=True),
    "retry_text": dict(retry=True, text_memory=True),
    "retry_memory_text": dict(retry=True, retry_with_memory=True, text_memory=True),
    # Learn-from-Failure (docs/METHOD.md): fail -> analyse -> adapt -> retry
    "lff_adapter": dict(learn=dict(mode="adapter", signals=["oracle", "gap"])),
    # signal ablation (what goes into the update): P = prediction gap, R = rollout (proprio + contact),
    # I = imagination pose probe, V = VLM, O = oracle. Adapter mode, same recipe otherwise.
    "lff_sig_O": dict(learn=dict(mode="adapter", signals=["oracle"])),
    "lff_sig_OP": dict(learn=dict(mode="adapter", signals=["oracle", "gap"])),
    "lff_sig_V": dict(learn=dict(mode="adapter", signals=["vlm"]), vlm=dict(endpoint="http://127.0.0.1:8001/v1",
                      model="Qwen/Qwen3-VL-8B-Instruct", pred_frames=False)),
    "lff_sig_VP": dict(learn=dict(mode="adapter", signals=["vlm", "gap"]), vlm=dict(endpoint="http://127.0.0.1:8001/v1",
                       model="Qwen/Qwen3-VL-8B-Instruct", pred_frames=True)),
    "lff_sig_P": dict(learn=dict(mode="adapter", signals=["gap"], relabel=False)),
    "lff_sig_PR": dict(learn=dict(mode="adapter", signals=["gap", "rollout"])),
    "lff_sig_PI": dict(learn=dict(mode="adapter", signals=["gap", "imagine"])),
    "lff_sig_PRI": dict(learn=dict(mode="adapter", signals=["gap", "rollout", "imagine"])),
    "lff_sig_PRIV": dict(learn=dict(mode="adapter", signals=["gap", "rollout", "imagine", "vlm"]),
                         vlm=dict(endpoint="http://127.0.0.1:8001/v1", model="Qwen/Qwen3-VL-8B-Instruct",
                                  pred_frames=True)),
    # recipe ablations (oracle + gap signals)
    "lff_adapter_noweight": dict(learn=dict(mode="adapter", signals=["oracle"], gap_beta=0.0)),
    "lff_adapter_norelabel": dict(learn=dict(mode="adapter", signals=["oracle", "gap"], relabel=False)),
    "lff_adapter_instr": dict(learn=dict(mode="adapter", signals=["oracle", "gap"], prompt="instruction")),
    "lff_adapter_videoonly": dict(learn=dict(mode="adapter", signals=["oracle", "gap"], lambda_action=0.0)),  # WAM-TTT
    "lff_adapter_actiononly": dict(learn=dict(mode="adapter", signals=["oracle", "gap"], lambda_video=0.0)),
    "lff_adapter_3att": dict(learn=dict(mode="adapter", signals=["oracle", "gap"], max_attempts=3)),
    # integrations of current methods (docs/METHOD.md, section 8)
    "lff_ttt_filtered": dict(learn=dict(mode="adapter", signals=["gap"], relabel=False, gap_filter=0.7)),   # T3VF
    "lff_ttt_persist": dict(learn=dict(mode="adapter", signals=["gap"], relabel=False, persist=True,
                                       lambda_action=0.0, max_attempts=1, every_episode=True)),                                    # JEPA-TTT
    "lff_pref": dict(learn=dict(mode="adapter", signals=["oracle", "gap"], pref_beta=500.0)),                # FAR
    "lff_pref_PRI": dict(learn=dict(mode="adapter", signals=["gap", "rollout", "imagine"], pref_beta=500.0)),
    "lff_bank": dict(learn=dict(mode="adapter", signals=["oracle", "gap"], persist=True, init="base",
                                guard=True, keep_successes=True)),                                            # FailBank
    "retry_explore": dict(learn=dict(mode="none", retry_action_noise=0.1)),                                   # FAR exploration
    "lff_adapter_explore": dict(learn=dict(mode="adapter", signals=["oracle", "gap"], retry_action_noise=0.1)),
    # in-context (RoboRetry: structural retry cue vs corrective content)
    "lff_icl": dict(learn=dict(mode="icl", signals=["oracle", "gap"])),
    "lff_icl_nocorr": dict(learn=dict(mode="icl", signals=["oracle", "gap"], prompt="instruction")),
    "lff_icl_noimag": dict(learn=dict(mode="icl", signals=["oracle", "gap"], include_pred=False)),
    "lff_icl_meta": dict(learn=dict(mode="icl", signals=["oracle", "gap"], meta_adapter="artifacts/adapters/failfix_meta")),
    "lff_both": dict(learn=dict(mode="both", signals=["oracle", "gap"], meta_adapter="artifacts/adapters/failfix_meta")),
    # Paper protocol (Few-Shot Residual Adaptation, Suliga et al.): 3 adaptation episodes, update after each, freeze,
    # evaluate. Run with NUM = 3 + eval episodes per client (sweep p5f_paper). Inner/outer 25 steps as in the paper.
    "paper_P_dual": dict(learn=dict(mode="adapter", signals=["gap"], relabel=False, every_episode=True, persist=True,
                                    max_attempts=1, adapt_episodes=3, outside="sequential", steps=25, outside_steps=25,
                                    gap_beta=0.0)),
    "paper_P_inside": dict(learn=dict(mode="adapter", signals=["gap"], relabel=False, every_episode=True, persist=True,
                                      max_attempts=1, adapt_episodes=3, outside="none", steps=25, gap_beta=0.0)),
    "paper_P_outside": dict(learn=dict(mode="adapter", signals=["gap"], relabel=False, every_episode=True,
                                       persist=True, max_attempts=1, adapt_episodes=3, outside="parallel",
                                       outside_steps=25, steps=0, gap_beta=0.0)),
    "paper_P_dual_150": dict(learn=dict(mode="adapter", signals=["gap"], relabel=False, every_episode=True,
                                        persist=True, max_attempts=1, adapt_episodes=3, outside="sequential",
                                        steps=150, outside_steps=100, gap_beta=0.0)),
    # Dual-site variants, prediction error only (no oracle, no VLM): what matters in the recipe? (2026-11-12)
    **{f"paper_P_dual_{k}": dict(learn=dict(mode="adapter", signals=["gap"], relabel=False, every_episode=True,
                                            persist=True, max_attempts=1, outside="sequential",
                                            **{**dict(adapt_episodes=3, steps=25, outside_steps=25, gap_beta=0.0), **v}))
       for k, v in dict(
           s10=dict(steps=10, outside_steps=10), s50=dict(steps=50, outside_steps=50),   # between 25 and 150
           video=dict(lambda_action=0.0),        # video-prediction loss only (the pure world-model signal)
           action=dict(lambda_video=0.0),        # control: action loss only (imitating the own executed actions)
           vidtok=dict(inside_streams=["video"]),  # adapter on the video tokens only (world model adapted, policy not)
           n1=dict(adapt_episodes=1), n6=dict(adapt_episodes=6),   # fewer / more adaptation episodes
           filt=dict(gap_filter=0.7),            # train only on the 30 % frames with the highest prediction error
           lr3=dict(lr=1.3e-3),                  # 3x learning rate
           n1vidtok=dict(adapt_episodes=1, inside_streams=["video"]),  # best two of 2026-11-13 combined
       ).items()},
    "paper_Pw_dual": dict(learn=dict(mode="adapter", signals=["gap"], relabel=False, every_episode=True, persist=True,
                                     max_attempts=1, adapt_episodes=3, outside="sequential", steps=25,
                                     outside_steps=25)),                                  # + prediction-error weights
    "paper_PR_dual": dict(learn=dict(mode="adapter", signals=["gap", "rollout"], every_episode=True, persist=True,
                                     max_attempts=1, adapt_episodes=3, outside="sequential", steps=25,
                                     outside_steps=25)),
    "paper_PRI_dual": dict(learn=dict(mode="adapter", signals=["gap", "rollout", "imagine"], every_episode=True,
                                      persist=True, max_attempts=1, adapt_episodes=3, outside="sequential", steps=25,
                                      outside_steps=25)),
    "paper_OP_dual": dict(learn=dict(mode="adapter", signals=["oracle", "gap"], every_episode=True, persist=True,
                                     max_attempts=1, adapt_episodes=3, outside="sequential", steps=25,
                                     outside_steps=25)),
    "paper_VP_dual": dict(learn=dict(mode="adapter", signals=["vlm", "gap"], every_episode=True, persist=True,
                                     max_attempts=1, adapt_episodes=3, outside="sequential", steps=25,
                                     outside_steps=25),
                          vlm=dict(endpoint="http://127.0.0.1:8001/v1", model="Qwen/Qwen3-VL-8B-Instruct",
                                   pred_frames=True)),
    "paper_OP_dual_retry": dict(learn=dict(mode="adapter", signals=["oracle", "gap"], every_episode=False,
                                           persist=True, max_attempts=2, adapt_episodes=3, outside="sequential",
                                           steps=25, outside_steps=25)),       # adaptation episodes also get a retry
    # ===================== In-Context Learning from failures (docs/ICL.md) =====================
    # all: attempt 1 fails -> analysis (oracle+gap unless noted) -> attempt 2 with the memory, no weight update.
    # -- WHAT goes into the context (text = correction, timing = pre, sequential, gap 1)
    "icl_text_only": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(content=[]))),
    "icl_fail_obs": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(groups=["obs"]))),
    "icl_fail_pred": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(groups=["pred"]))),
    "icl_fail_predobs": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(groups=["pred", "obs"]))),
    "icl_fail_diff": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(groups=["diff", "obs"]))),
    "icl_fail_window": dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                       icl=dict(groups=["pred", "obs"], segment="window"))),
    "icl_fail_corrected": dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                          icl=dict(groups=["obs"], actions="corrected"))),
    "icl_demo_same": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(content=["demo_same_seed"]))),
    "icl_demo_nearest": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(content=["demo_nearest"]))),
    "icl_fail_demo": dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                     icl=dict(content=["fail", "demo_same_seed"], groups=["obs"]))),
    "icl_demo_fail": dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                     icl=dict(content=["demo_same_seed", "fail"], groups=["obs"]))),
    "icl_fails_3att": dict(learn=dict(mode="icl", signals=["oracle", "gap"], max_attempts=3,
                                      icl=dict(content=["fails"], groups=["obs"]))),
    # -- same, with non-privileged analysis (only matters for text / window / corrected)
    "icl_fail_predobs_PR": dict(learn=dict(mode="icl", signals=["gap", "rollout"], icl=dict(groups=["pred", "obs"]))),
    "icl_fail_window_PR": dict(learn=dict(mode="icl", signals=["gap", "rollout"],
                                          icl=dict(groups=["pred", "obs"], segment="window"))),
    "icl_fail_predobs_VP": dict(learn=dict(mode="icl", signals=["vlm", "gap"], icl=dict(groups=["pred", "obs"])),
                                vlm=dict(endpoint="http://127.0.0.1:8001/v1", model="Qwen/Qwen3-VL-8B-Instruct",
                                         pred_frames=True)),
    "icl_fail_predobs_P": dict(learn=dict(mode="icl", signals=["gap"], icl=dict(groups=["pred", "obs"], text="instruction"))),
    # -- WHICH TEXT accompanies the memory (memory = imagined + observed failure)
    "icl_txt_instruction": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(text="instruction"))),
    "icl_txt_negative": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(text="negative"))),
    "icl_txt_corr_neg": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(text="correction+negative"))),
    "icl_txt_note": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(text="failure_note"))),
    # -- HOW it is inserted
    "icl_pos_aligned": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(position="aligned"))),
    "icl_gap8": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(gap=8))),
    "icl_gap32": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(gap=32))),
    "icl_attn0.5": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(attn_scale=0.5))),
    "icl_attn1.5": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(attn_scale=1.5))),
    "icl_attn2": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(attn_scale=2.0))),
    "icl_trigger": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(timing="trigger"))),
    "icl_trigger_window": dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                          icl=dict(timing="trigger", segment="window"))),
    "icl_pre_and_trigger": dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(timing="both"))),
    "icl_aligned_window": dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                          icl=dict(position="aligned", segment="window"))),
    # -- META-TRAINED: the model was taught to use this memory format (FailFix adapters, cluster/adapter_grid.py)
    "icl_meta_obs": dict(learn=dict(mode="icl", signals=["oracle", "gap"], meta_adapter="artifacts/adapters/failfix_obs",
                                    icl=dict(groups=["obs"]))),
    "icl_meta_predobs": dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                        meta_adapter="artifacts/adapters/failfix_predobs", icl=dict(groups=["pred", "obs"]))),
    "icl_meta_diff": dict(learn=dict(mode="icl", signals=["oracle", "gap"], meta_adapter="artifacts/adapters/failfix_diff",
                                     icl=dict(groups=["diff", "obs"]))),
    "icl_meta_window": dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                       meta_adapter="artifacts/adapters/failfix_window",
                                       icl=dict(groups=["pred", "obs"], segment="window"))),
    "icl_meta_predobs_notext": dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                               meta_adapter="artifacts/adapters/failfix_predobs",
                                               icl=dict(groups=["pred", "obs"], text="instruction"))),
    "icl_meta_predobs_PR": dict(learn=dict(mode="icl", signals=["gap", "rollout"],
                                           meta_adapter="artifacts/adapters/failfix_predobs",
                                           icl=dict(groups=["pred", "obs"]))),
    "icl_meta_mismatch": dict(learn=dict(mode="icl", signals=["oracle", "gap"],   # trained on obs, tested with pred+obs
                                         meta_adapter="artifacts/adapters/failfix_obs", icl=dict(groups=["pred", "obs"]))),
    "icl_meta_demo": dict(learn=dict(mode="icl", signals=["oracle", "gap"], meta_adapter="artifacts/adapters/democtx_meta",
                                     icl=dict(content=["demo_same_seed"]))),
    # meta-training ablations (same test format pred+obs): only own-failure samples / no text / fewer steps /
    # an unseen task (move_can_pot was held out of meta-training)
    "icl_meta_ownonly": dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                        meta_adapter="artifacts/adapters/failfix_predobs_ownonly")),
    "icl_meta_trained_notext": dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                               meta_adapter="artifacts/adapters/failfix_predobs_notext",
                                               icl=dict(text="instruction"))),
    "icl_meta_small": dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                      meta_adapter="artifacts/adapters/failfix_predobs_small")),
    "icl_meta_holdtask": dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                         meta_adapter="artifacts/adapters/failfix_predobs_holdtask")),
    "icl_meta_noctx": dict(learn=dict(mode="icl", signals=["oracle", "gap"],   # meta adapter, but empty memory
                                      meta_adapter="artifacts/adapters/failfix_predobs", icl=dict(content=[]))),
    # -- for comparison: the same failure through the weights (adapters, unchanged) and through both
    "icl_vs_adapter": dict(learn=dict(mode="adapter", signals=["oracle", "gap"])),
    "icl_plus_adapter": dict(learn=dict(mode="both", signals=["oracle", "gap"],
                                        meta_adapter="artifacts/adapters/failfix_predobs")),
    "cf_full": dict(memory_counterfactual="full"),
    # imagination vs. action: also denoise the imagined frames without the memory (pose probe on both, ~+60% time)
    "imagcf": dict(learn=dict(icl=dict(counterfactual="full_video"))),
    "cf_off": dict(memory_counterfactual="off", measure_memory=False),
    "lff_adapter_lifelong": dict(learn=dict(mode="adapter", signals=["oracle", "gap"], persist=True)),
    "lff_lifelong_PRI": dict(learn=dict(mode="adapter", signals=["gap", "rollout", "imagine"], persist=True)),
}


# test presets for the adapter-type comparison of ICL meta-training (cluster/adapter_grid.py ADAPTER_TYPES)
META_TYPES = ["adapterplus", "adapterplus_b32", "adapterplus_b512", "houlsby", "residual_mlp", "adapterplus_video",
              "adapterplus_action", "adapterplus_first10", "adapterplus_last10", "lora_r16_attn", "lora_r16_attn_ffn",
              "lora_r64_all", "ia3", "adaln", "ia3_adaln", "prompt_t3vf", "prompt_vane", "ttt_wam", "ttt_wam_both",
              "svf_salt", "dense_last2", "ttt_plus_adapterplus", "lora_plus_prompt"]
# content manipulations of the memory (training-free and meta-trained, memory = imagined + observed failure)
MANIPS = ["shuffle_time", "reverse_time", "noise", "blank_head", "blank_wrists", "no_actions"]
for _mp in MANIPS:
    PRESETS[f"icl_manip_{_mp}"] = dict(learn=dict(mode="icl", signals=["oracle", "gap"], icl=dict(manip=_mp)))
    PRESETS[f"icl_meta_manip_{_mp}"] = dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                                       meta_adapter="artifacts/adapters/failfix_predobs",
                                                       icl=dict(manip=_mp)))
PRESETS["icl_manip_otherseed"] = dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                                 icl=dict(content=["fail_other_seed"])))
PRESETS["icl_meta_manip_otherseed"] = dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                                      meta_adapter="artifacts/adapters/failfix_predobs",
                                                      icl=dict(content=["fail_other_seed"])))
META_TYPES += ["film_memory", "film_error", "film_gapcurve", "film_text", "film_memory_plus_adapterplus"]
for _t in META_TYPES:
    PRESETS[f"icl_type_{_t}"] = dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                                meta_adapter=f"artifacts/adapters/failfix_type_{_t}",
                                                icl=dict(groups=["pred", "obs"])))
    PRESETS[f"icl_type_{_t}_noctx"] = dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                                      meta_adapter=f"artifacts/adapters/failfix_type_{_t}",
                                                      icl=dict(content=[])))

# meta-ICL with the VLM analysis at test time (adapters meta-trained on analysis_gap+vlm.json without the test
# task, cluster/adapter_grid.py). <name>_noctx = the same adapter without memory and without the analysis text:
# what the adapter does on its own (fine-tuning effect), the control every memory variant is compared to.
_VLM = dict(endpoint="http://127.0.0.1:8001/v1", model="Qwen/Qwen3-VL-8B-Instruct", pred_frames=True)
VLM_META_TYPES = ["adapterplus", "lora_r16_attn", "adaln", "ia3_adaln", "film_memory", "film_error", "film_gapcurve",
                  "film_text", "film_memory_plus_adapterplus", "prompt_vane", "ttt_wam"]
# "<any icl/lff preset>+VP": the same method with the VLM analysis (rollout + prediction-error curve) instead of the oracle
PRESETS["VP"] = dict(learn=dict(signals=["vlm", "gap"]), vlm=_VLM)
PRESETS["icl_fail_gapkeys"] = dict(learn=dict(mode="icl", signals=["oracle", "gap"],
                                              icl=dict(groups=["pred", "obs"], segment="gaptop")))
PRESETS["icl_meta_gapkeys_VP"] = dict(learn=dict(mode="icl", signals=["vlm", "gap"],
                                                 meta_adapter="artifacts/adapters/failfix_gapkeys_vlm",
                                                 icl=dict(groups=["pred", "obs"], segment="gaptop")), vlm=_VLM)
for _t in VLM_META_TYPES:
    _ad = f"artifacts/adapters/failfix_type_{_t}_vlm"
    PRESETS[f"icl_type_{_t}_VP"] = dict(learn=dict(mode="icl", signals=["vlm", "gap"], meta_adapter=_ad,
                                                   icl=dict(groups=["pred", "obs"])), vlm=_VLM)
    PRESETS[f"icl_type_{_t}_VP_noctx"] = dict(learn=dict(mode="icl", signals=["vlm", "gap"], meta_adapter=_ad,
                                                         icl=dict(content=[], text="instruction")), vlm=_VLM)
# plain fine-tuning baseline: Adapter+ trained on the 7 training tasks without memory or analysis text
# (cluster/adapter_grid.py failfix_ft_adapterplus); retry of the failed seed with that adapter, nothing in context
PRESETS["ft_adapterplus"] = dict(learn=dict(mode="icl", signals=["gap"],
                                            meta_adapter="artifacts/adapters/failfix_ft_adapterplus",
                                            icl=dict(content=[], text="instruction")))
# held-out failure type (cluster/adapter_grid.py failfix_dist_*): meta-trained on start-posture + camera failures only
for _t in ["film_memory_plus_adapterplus", "lora_r16_attn", "adapterplus", "ttt_wam"]:
    _ad = f"artifacts/adapters/failfix_dist_{_t}_vlm"
    PRESETS[f"icl_dist_{_t}_VP"] = dict(learn=dict(mode="icl", signals=["vlm", "gap"], meta_adapter=_ad,
                                                   icl=dict(groups=["pred", "obs"])), vlm=_VLM)
    PRESETS[f"icl_dist_{_t}_VP_noctx"] = dict(learn=dict(mode="icl", signals=["vlm", "gap"], meta_adapter=_ad,
                                                         icl=dict(content=[], text="instruction")), vlm=_VLM)
for _t in ["film_memory_plus_adapterplus", "lora_r16_attn", "adapterplus", "ttt_wam"]:  # 7 failure types (P7b+P7c)
    _ad = f"artifacts/adapters/failfix_types_{_t}_vlm"
    PRESETS[f"icl_types_{_t}_VP"] = dict(learn=dict(mode="icl", signals=["vlm", "gap"], meta_adapter=_ad,
                                                    icl=dict(groups=["pred", "obs"])), vlm=_VLM)
    PRESETS[f"icl_types_{_t}_VP_noctx"] = dict(learn=dict(mode="icl", signals=["vlm", "gap"], meta_adapter=_ad,
                                                          icl=dict(content=[], text="instruction")), vlm=_VLM)
PRESETS["ft_types_adapterplus"] = dict(learn=dict(mode="icl", signals=["gap"],
                                                  meta_adapter="artifacts/adapters/failfix_types_ft_adapterplus",
                                                  icl=dict(content=[], text="instruction")))
PRESETS["ft_dist_adapterplus"] = dict(learn=dict(mode="icl", signals=["gap"],
                                                 meta_adapter="artifacts/adapters/failfix_dist_ft_adapterplus",
                                                 icl=dict(content=[], text="instruction")))
# P8s (user 2026-11-18): adapters on P7b+c+d with the detailed VLM analysis (+ fix sentence); _base = the base model
# was trained too (its weights are swapped in with the adapter, fawam/base_train.py)
_VLMD = dict(_VLM, detail=True)
for _t in ["lora_r16_attn", "film_memory_plus_adapterplus"]:
    for _b in ("", "_base"):
        _ad = f"artifacts/adapters/failfix_strong_{_t}_vlmd{_b}"
        PRESETS[f"icl_strong_{_t}{_b}_VPD"] = dict(learn=dict(mode="icl", signals=["vlm", "gap"], meta_adapter=_ad,
                                                              icl=dict(groups=["pred", "obs"])), vlm=_VLMD)
        PRESETS[f"icl_strong_{_t}{_b}_VPD_noctx"] = dict(learn=dict(mode="icl", signals=["vlm", "gap"], meta_adapter=_ad,
                                                                    icl=dict(content=[], text="instruction")), vlm=_VLMD)
PRESETS["ft_strong_base"] = dict(learn=dict(mode="icl", signals=["gap"],
                                            meta_adapter="artifacts/adapters/failfix_strong_ft_base",
                                            icl=dict(content=[], text="instruction")))
# p12 hid eval (plan 2026-11-18): C1 x M1 trained on hidden offsets; real signals only (memory = imagined + real
# failed attempt, instruction as text). ctx / noctx / shuf (memory = a failure of another seed, other offset)
_HIDL = "artifacts/adapters/failfix_hid_lora"
PRESETS["icl_hid_lora"] = dict(learn=dict(mode="icl", signals=["gap"], meta_adapter=_HIDL,
                                          icl=dict(groups=["pred", "obs"], text="instruction")))
PRESETS["icl_hid_lora_noctx"] = dict(learn=dict(mode="icl", signals=["gap"], meta_adapter=_HIDL,
                                                icl=dict(content=[], text="instruction")))
PRESETS["icl_hid_lora_shuf"] = dict(learn=dict(mode="icl", signals=["gap"], meta_adapter=_HIDL,
                                               icl=dict(groups=["pred", "obs"], text="instruction",
                                                        content=["fail_other_seed"],
                                                        fail_bank="artifacts/fail_bank/hid_trans_eval")))
# p12 hid eval, M3 residual head (fawam/train_hid.py): context = real signals of the failed attempt (C2-C5)
for _c in ("C2", "C3", "C4", "C5"):
    for _x in ("", "_aux0.3"):
        _h = f"artifacts/hid_m3/{_c}_M3{_x}"
        for _mode, _suf in (("own", ""), ("none", "_noctx"), ("shuf", "_shuf")):
            PRESETS[f"hidm3_{_c}{_x.replace('.', '')}{_suf}"] = dict(learn=dict(mode="hidres", signals=["gap"],
                                                                               hid_head=_h, hid_ctx=_mode))
# P8c small contexts (cluster/adapter_grid.py SMALL_CTX); _noctx = same adapter, instruction only, no memory
_W = dict(segment="window", window=dict(before=4, after=2))
for _n, _icl, _v in (("diffwin", dict(groups=["diff"], **_W), _VLM), ("diffobswin", dict(groups=["diff", "obs"], **_W), _VLM),
                     ("predobswin", dict(groups=["pred", "obs"], **_W), _VLM),
                     ("gapkeys", dict(groups=["pred", "obs"], segment="gaptop"), _VLM),
                     ("diffwin_vlmd", dict(groups=["diff"], **_W), _VLMD), ("textonly_vlmd", dict(content=[]), _VLMD)):
    _ad = f"artifacts/adapters/failfix_sc_{_n}"
    PRESETS[f"icl_sc_{_n}"] = dict(learn=dict(mode="icl", signals=["vlm", "gap"], meta_adapter=_ad, icl=_icl), vlm=_v)
    PRESETS[f"icl_sc_{_n}_noctx"] = dict(learn=dict(mode="icl", signals=["vlm", "gap"], meta_adapter=_ad,
                                                    icl=dict(content=[], text="instruction")), vlm=_v)
PRESETS["icl_meta_diff_VP"] =dict(learn=dict(mode="icl", signals=["vlm", "gap"],
                                              meta_adapter="artifacts/adapters/failfix_diff_vlm",
                                              icl=dict(groups=["diff", "obs"])), vlm=_VLM)
PRESETS["icl_meta_window_VP"] = dict(learn=dict(mode="icl", signals=["vlm", "gap"],
                                                meta_adapter="artifacts/adapters/failfix_window_vlm",
                                                icl=dict(groups=["pred", "obs"], segment="window")), vlm=_VLM)
PRESETS["icl_meta_textonly_VP"] = dict(learn=dict(mode="icl", signals=["vlm", "gap"],
                                                  meta_adapter="artifacts/adapters/failfix_textonly_vlm",
                                                  icl=dict(content=[])), vlm=_VLM)


def _merge(base: dict, over: dict):
    """Recursive merge (so "icl_x+imagcf" only changes learn.icl.counterfactual)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_part(part, root=None):
    if isinstance(part, dict):
        return part
    part = str(part).strip()
    if part.startswith("{"):
        return json.loads(part)
    p = Path(part)
    if not p.is_absolute() and not p.exists():
        p = Path(root or Path(__file__).resolve().parents[1]) / p
    if p.suffix in (".json", ".yaml", ".yml") and p.exists():
        if p.suffix == ".json":
            return json.loads(p.read_text())
        import yaml
        return yaml.safe_load(p.read_text())
    if part in PRESETS:
        return dict(PRESETS[part])
    raise KeyError(f"unknown method part {part!r}; presets: {sorted(PRESETS)}")


def resolve(spec=None, root=None) -> Method:
    """spec: None | Method | dict | str ("a+b", path, JSON). Relative artifact paths resolve against `root`."""
    if isinstance(spec, Method):
        return spec
    if spec is None or spec == "":
        return Method()
    if isinstance(spec, dict) and "name" in spec and set(spec) <= {f.name for f in fields(Method)}:
        parts, name = [spec], spec["name"]
    elif isinstance(spec, dict):
        parts, name = [spec], "custom"
    elif str(spec).strip().startswith("{"):
        parts, name = [spec], "custom"
    else:
        parts, name = str(spec).split("+"), str(spec)
    merged = {}
    for part in parts:
        merged = _merge(merged, _load_part(part, root))
    if merged.get("intervention") is not None:
        merged["intervention"] = _merge(INTERVENTION_DEFAULTS, merged["intervention"])
    if merged.get("learn") is not None:
        merged["learn"] = _merge(LEARN_DEFAULTS, merged["learn"])
        if merged["learn"].get("mode") in ("icl", "both"):
            merged["learn"]["icl"] = _merge(ICL_DEFAULTS, merged["learn"].get("icl") or {})
    merged.setdefault("name", name)
    unknown = set(merged) - {f.name for f in fields(Method)}
    if unknown:
        raise KeyError(f"unknown method fields {sorted(unknown)}")
    _port = os.environ.get("FAWAM_VLM_PORT")  # several eval jobs on one node (cluster/dev_runner.sh): one vLLM port each
    if _port and isinstance(merged.get("vlm"), dict) and merged["vlm"].get("endpoint"):
        merged["vlm"] = dict(merged["vlm"], endpoint=merged["vlm"]["endpoint"].replace(":8001", f":{_port}"))
    m = Method(**merged)
    if root is not None:
        root = Path(root)
        for key in ("probe_path", "steer_path", "adapter_path"):
            v = getattr(m, key)
            if v and not Path(v).is_absolute():
                setattr(m, key, str(root / v))
        if m.learn and m.learn.get("meta_adapter") and not Path(m.learn["meta_adapter"]).is_absolute():
            m.learn = {**m.learn, "meta_adapter": str(root / m.learn["meta_adapter"])}
        if m.prefill and m.prefill.get("bank") and not Path(m.prefill["bank"]).is_absolute():
            m.prefill = {**m.prefill, "bank": str(root / m.prefill["bank"])}
    return m

# ===================== plan v2 (2026-11-21), TTT line p22: video-only adaptation on this one scene =====================
# Adapter only in the video pass (inside_streams video / LoRA lora_streams video), video flow-matching loss on the own
# failed episode (teacher forcing on the real past), action loss 0, no relabel, no oracle. Up to 3 retries per seed.
_TTT = dict(mode="adapter", signals=["gap"], relabel=False, gap_beta=0.0, lambda_action=0.0, video_in_window="real",
            kind="inside", inside_streams=["video"], steps=50, lr=1e-4, init="base", memory="last", max_attempts=4)
PRESETS["ttt_p0"] = dict(learn=dict(_TTT))                                            # P0 pure prediction error
for _st in (10, 50, 200):
    for _lr, _ln in ((1e-4, "1e-4"), (1e-3, "1e-3")):
        PRESETS[f"ttt_p0_s{_st}_lr{_ln}"] = dict(learn=dict(_TTT, steps=_st, lr=_lr))  # P0 grid (Adapter+)
PRESETS["ttt_p1"] = dict(learn=dict(_TTT, gap_filter=0.5))                            # P1 loss only above the median gap
PRESETS["ttt_p2"] = dict(learn=dict(_TTT, gate="prev_gap"))                           # P2 adapter strength = last gap
PRESETS["ttt_p3"] = dict(learn=dict(mode="none", signals=["gap"], max_attempts=4, explore=True, explore_n=8))
PRESETS["ttt_p4"] = dict(learn=dict(_TTT, mode="none", explore=True, explore_n=8, learn_from_success=True,
                                    persist=True))                                    # P4 own success as the target
PRESETS["ttt_p5"] = dict(learn=dict(_TTT, mode="none", explore=True, explore_n=8, learn_from_success=True,
                                    persist=True, hinge_beta=1.0))                    # P5 + video hinge vs the failure
PRESETS["ttt_p6"] = dict(learn=dict(_TTT, init="prev"))                               # P6 keep training across retries
PRESETS["ttt_p0_lora"] = dict(learn=dict(_TTT, kind="lora"))                          # adapter comparison: LoRA r16
PRESETS["ttt_p0_dual"] = dict(learn=dict(_TTT, kind=None, outside="sequential", outside_steps=50))  # dual-site
PRESETS["ttt_retry"] = dict(learn=dict(mode="none", signals=["gap"], max_attempts=4))  # control: retry, no learning

# ===================== plan v2 (2026-11-21), ICL stage 1 p21: rewind retry with a meta-trained video adapter ===========
# Adapter artifacts/adapters/p21_<setting>_<arch>_<ctx>[_<sampling>] (cluster/p21_train_job.sh); test protocol in the
# client (_icl_rewind): reset k=2 chunks before the gripper closed, context of the failed attempt + own past in the KV
# cache. _foreign: context = a failure with the opposite sign of the hidden parameter (bank sweep p21_bank).
P21_VARIANTS = [(s, a, c, x) for s in ("sa", "sb") for a, c, x in
                [("A1", c, "") for c in ("C0", "C1", "C2", "C3")] + [("A2", "C2", ""), ("A3", "C2", "")]] + \
               [("sb", "A1", c, "") for c in ("C4", "C5", "C6", "C7", "C8")] + \
               [("sb", "A1", "C2", x) for x in ("_win50", "_w4")] + \
               [("sb", t, "C2", "") for t in ("T1", "T2", "T4", "T5")]
_P21_EXTRA = {"A3": dict(ctx_mode="separate"), "T1": dict(gate="t1"), "T2": dict(gate="t2"),
              "T4": dict(ctx_mode="both"), "T5": dict(gate="t5")}
for _s, _a, _c, _x in P21_VARIANTS:
    _n = f"p21_{_s}_{_a}_{_c}{_x}"
    _base = dict(mode="icl_rewind", signals=["gap"], meta_adapter=f"artifacts/adapters/{_n}", ctx=_c, k=2, harm=True,
                 max_past=8, foreign_bank=[f"results/sweeps/p21_bank/base/icl_{_s}"], **_P21_EXTRA.get(_a, {}))
    PRESETS[f"icl2_{_s}_{_a}_{_c}{_x}"] = dict(learn=dict(_base))
    PRESETS[f"icl2_{_s}_{_a}_{_c}{_x}_foreign"] = dict(learn=dict(_base, foreign=True, harm=False))
# smoke of the p21 rewind protocol: zero-initialised Adapter+ (= base), context C3 / C5 / C7
for _c in ("C3", "C5", "C7"):
    PRESETS[f"icl2_smoke_{_c}"] = dict(learn=dict(mode="icl_rewind", signals=["gap"], ctx=_c, k=2, harm=True,
                                                  max_past=8, meta_adapter="artifacts/adapters/p21_smoke_A1",
                                                  foreign_bank=["results/sweeps/p21_bank/base/icl_sb"]))
PRESETS["icl2_smoke_C3_foreign"] = dict(learn=dict(PRESETS["icl2_smoke_C3"]["learn"], foreign=True, harm=False))
# smoke of A3 / T gates at inference (adapters from the 20-step training smokes)
for _a, _x in (("A3", dict(ctx_mode="separate")), ("T1", dict(gate="t1")), ("T2", dict(gate="t2")),
               ("T4", dict(ctx_mode="both")), ("T5", dict(gate="t5")), ("A1", {}), ("A2", {})):
    PRESETS[f"icl2_smoke_{_a}"] = dict(learn=dict(mode="icl_rewind", signals=["gap"], ctx="C2", k=2, harm=True,
                                                  max_past=8, meta_adapter=f"artifacts/adapters/p21smoke_{_a}_C2",
                                                  foreign_bank=["results/sweeps/p21_bank/base/icl_sb"], **_x))
