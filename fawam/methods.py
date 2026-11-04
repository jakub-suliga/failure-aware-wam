"""Test-time method configuration shared by the policy server (fawam/server.py) and the RoboTwin client hooks
(fawam/robotwin_hooks.py). One `Method` describes everything that differs from plain LingBot-VA inference.

A method is given as: a preset name ("ig2"), several presets joined with "+" ("ig2+reset_on_gap"), a JSON/YAML file,
a JSON string, or a dict. Later parts override earlier ones; dict values are merged one level deep.
The experiment IDs (A1, B3, ...) refer to docs/EXPERIMENTS.md.
"""
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

    # --- steering (server, A6) ------------------------------------------------------------------------------------
    steer_path: str = None             # fawam.probe steering output (.npz)
    steer_alpha: float = 0.0
    steer_layer: int = None            # None = layer stored in the file
    steer_streams: list = field(default_factory=lambda: ["video", "action"])

    # --- adaptation (server, D) -------------------------------------------------------------------------------------
    adapter_path: str = None           # fawam.train_adapter output dir
    adapter_use: str = "all"           # all | inside | outside

    # --- in-context (server + client, E) --------------------------------------------------------------------------
    prefill: dict = None               # {"bank": dir, "k": 1, "select": nearest|random|same_task, "gap": 1}
    retry: bool = False                # E2/E4 baseline (client): after a failure, rerun the same seed once more
    retry_with_memory: bool = False    # E2: ... with the failed attempt prefilled into the KV cache
    text_memory: bool = False          # E4: ... with a description of the previous failure appended to the prompt

    # --- Learn-from-Failure (client + learner, docs/METHOD.md) ----------------------------------------------------
    learn: dict = None                 # see LEARN_DEFAULTS

    trigger_prefill: dict = None       # memory spec inserted by the intervention action "reset_memory"

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
    steps=150, gap_beta=2.0, relabel=True, video_in_window="skip", lambda_video=1.0, lambda_action=1.0,
    include_pred=True,       # icl: imagined + observed failed attempt in the context
    prompt="correction",     # next attempt's prompt: correction (analysis) | instruction
    negative=False,          # next attempt's CFG negative = analysis.negative_prompt
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
)

# ---------------------------------------------------------------------------------------------- presets
PRESETS = {
    "base": {},
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
    # an unseen task (stack_blocks_two was held out of meta-training)
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
    "lff_adapter_lifelong": dict(learn=dict(mode="adapter", signals=["oracle", "gap"], persist=True)),
    "lff_lifelong_PRI": dict(learn=dict(mode="adapter", signals=["gap", "rollout", "imagine"], persist=True)),
}


def _merge(base: dict, over: dict):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
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
