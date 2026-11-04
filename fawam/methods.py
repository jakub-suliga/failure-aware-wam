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

    # --- online intervention (client, B/C) ---------------------------------------------------------------------
    intervention: dict = None          # see INTERVENTION_DEFAULTS
    analyzer: str = None               # None | oracle | vlm
    vlm: dict = None                   # {"endpoint": "http://host:port/v1", "model": "...", "pred_frames": true}

    @property
    def uses_client_logic(self):
        return bool(self.intervention or self.retry or self.analyzer)

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
        if m.prefill and m.prefill.get("bank") and not Path(m.prefill["bank"]).is_absolute():
            m.prefill = {**m.prefill, "bank": str(root / m.prefill["bank"])}
    return m
