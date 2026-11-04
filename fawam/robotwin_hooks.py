"""Client-side pieces for the RoboTwin eval loop (lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py).

* ExpertTrajectory: records the planner's end-effector trajectory while the eval runs its expert check (play_once),
  and converts it to the policy's relative 16-d action format. Used by `--driver expert_replay` to record demos that
  go through exactly the same observation/latent pipeline as the policy (docs/EXPERIMENTS.md, D0).
* Orchestrator: online intervention on a prediction gap (B), with oracle / VLM failure analysis (C).
* failure description for text memory (E4).

Only numpy / scipy / stdlib here: the module is imported inside the robotwin env.
"""
import base64
import io
import json
import time
import urllib.request

import numpy as np
from scipy.spatial.transform import Rotation as R

from fawam.methods import INTERVENTION_DEFAULTS

# ------------------------------------------------------------------------------------------ action format

def _rel_pose(pose, init):
    """Inverse of the client's add_eef_pose: absolute (xyz, quat) -> relative to the initial pose."""
    rot = R.from_quat(init[3:7]).inv() * R.from_quat(pose[3:7])
    return np.concatenate([pose[:3] - init[:3], rot.as_quat()])


def to_relative(abs16, init16):
    """[l xyz quat grip, r xyz quat grip] absolute -> the policy's relative action (same layout)."""
    l = _rel_pose(abs16[:7], init16[:7])
    r = _rel_pose(abs16[8:15], init16[8:15])
    return np.concatenate([l, abs16[7:8], r, abs16[15:16]])


class ExpertTrajectory:
    """Hooks TASK_ENV._take_picture during play_once: one entry per saved frame (every save_freq physics steps),
    the same granularity as the RoboTwin demos LingBot-VA was trained on. Action t = pose of frame t + 1."""

    def __init__(self, task_env):
        self.env = task_env
        self.poses = []

    def __enter__(self):
        env = self.env
        self._save_data, self._take = getattr(env, "save_data", False), env._take_picture
        env.save_data = True

        def take_picture():
            self.poses.append(self.current_pose(env))

        env._take_picture = take_picture
        return self

    def __exit__(self, *exc):
        self.env.save_data = self._save_data
        self.env._take_picture = self._take
        return False

    @staticmethod
    def current_pose(env):
        return np.concatenate([np.asarray(env.get_arm_pose("left"), dtype=np.float64),
                               [env.robot.get_left_gripper_val()],
                               np.asarray(env.get_arm_pose("right"), dtype=np.float64),
                               [env.robot.get_right_gripper_val()]])

    def relative_actions(self, init16):
        """(T, 16) relative actions; the last pose is held."""
        if len(self.poses) < 2:
            return np.zeros((0, 16))
        targets = np.stack(self.poses[1:])
        return np.stack([to_relative(p, init16) for p in targets])


def expert_chunk(rel_actions, step0, n_frames, actions_per_frame=16, first=False):
    """Model-format action chunk (16, n_frames, actions_per_frame) starting at executed step `step0`.
    For a first chunk, frame 0 is the conditioning frame and stays zero (as in the policy's own chunk 0)."""
    chunk = np.zeros((16, n_frames, actions_per_frame), dtype=np.float32)
    k = step0
    for i in range(1 if first else 0, n_frames):
        for j in range(actions_per_frame):
            a = rel_actions[min(k, len(rel_actions) - 1)] if len(rel_actions) else np.zeros(16)
            chunk[:, i, j] = a
            k += 1
    return chunk


# ------------------------------------------------------------------------------------------ failure analysis

def oracle_analysis(env, instruction):
    """Ground-truth failure description from the simulator (upper bound for a VLM analyzer)."""
    arms = []
    for arm in ("left", "right"):
        ee = np.asarray(env.get_arm_pose(arm))[:3]
        grip = env.robot.get_left_gripper_val() if arm == "left" else env.robot.get_right_gripper_val()
        arms.append((arm, ee, grip))
    objs = [a for a in env.scene.get_all_actors() if a.get_name() not in ("", "table", "wall", "ground")]
    if not objs:
        return dict(failure_type="unknown", cause="no objects found", correction=instruction)
    # the arm whose gripper is most closed is the one that tried to grasp
    arm, ee, grip = min(arms, key=lambda x: x[2])
    obj = min(objs, key=lambda o: np.linalg.norm(np.asarray(o.get_pose().p)[:2] - ee[:2]))
    op = np.asarray(obj.get_pose().p)
    dz = float(ee[2] - op[2])
    dxy = float(np.linalg.norm(ee[:2] - op[:2]))
    if grip < 0.5 and dz > 0.08:
        ftype, cause = "missed_grasp", f"the {arm} gripper closed {dz * 100:.0f} cm above the object"
        corr = f"Lower the {arm} gripper about {dz * 100:.0f} cm further down and grasp the object again."
    elif dxy > 0.05:
        ftype, cause = "misaligned", f"the {arm} gripper is {dxy * 100:.0f} cm to the side of the object"
        corr = f"Move the {arm} gripper over the object, then grasp it."
    else:
        ftype, cause = "unclear", "the gripper is near the object"
        corr = instruction
    return dict(failure_type=ftype, cause=cause, correction=f"{instruction} {corr}",
                negative_prompt=f"The {arm} gripper closes in the air above the object without touching it.")


VLM_PROMPT = """You watch a bimanual robot doing: "{instruction}".
Image 1-3: the current camera views (head, left wrist, right wrist).{pred_note}
Did the robot just fail? Answer with JSON only:
{{"failed": true|false, "failure_type": "missed_grasp|dropped|misplaced|collision|none",
  "cause": "<one sentence, concrete, with directions/distances>",
  "correction": "<one imperative instruction for the robot to recover>",
  "negative_prompt": "<one sentence describing the failure to avoid>"}}"""


def _jpeg_b64(img):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def vlm_analysis(cfg, instruction, frames, pred_frames=None, timeout=120):
    """OpenAI-compatible chat endpoint (e.g. vLLM serving Qwen2.5-VL). frames: [head, left, right] HxWx3 uint8."""
    note = ""
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(f)}"}} for f in frames]
    if pred_frames:
        note = "\nImage 4-6: what the robot's own world model predicted these views would look like."
        content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(f)}"}}
                    for f in pred_frames]
    content.append({"type": "text", "text": VLM_PROMPT.format(instruction=instruction, pred_note=note)})
    body = json.dumps(dict(model=cfg.get("model"), messages=[{"role": "user", "content": content}],
                           temperature=0.0, max_tokens=300)).encode()
    req = urllib.request.Request(cfg["endpoint"].rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = json.loads(r.read())["choices"][0]["message"]["content"]
    try:
        out = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except ValueError:
        out = dict(failed=None, raw=text)
    out["latency_s"] = time.time() - t0
    if out.get("correction"):
        out["correction"] = f"{instruction} {out['correction']}"
    return out


def failure_text(env, meta_events, instruction):
    """E4: prompt suffix describing the previous attempt."""
    a = oracle_analysis(env, instruction)
    return f"{instruction} In the previous attempt {a['cause']}; avoid this."


# ------------------------------------------------------------------------------------------ orchestrator

class Orchestrator:
    """Decides after every chunk whether to intervene, and talks to the server. The eval loop calls
    `after_chunk(reply, env, obs_frames)` with the compute_kv_cache reply and reacts to the returned dict:
    {"first": True} -> the next model call is a first chunk again (context reset)."""

    def __init__(self, method, model, instruction):
        self.m, self.model, self.instruction = method, model, instruction
        self.cfg = {**INTERVENTION_DEFAULTS, **(method.intervention or {})}
        self.trig = {**INTERVENTION_DEFAULTS["trigger"], **self.cfg.get("trigger", {})}
        self.n_chunks = 0
        self.n_done = 0
        self.last_at = -10**9
        self.streak = 0
        self.log = []

    def _triggered(self, reply, env):
        kind = self.trig["kind"]
        if kind == "gap":
            z = reply.get("gap_z", {}) if reply else {}
            vals = [z[c] for c in self.trig["cams"] if c in z]
            hit = bool(vals) and max(vals) > self.trig["z"]
        elif kind == "oracle":
            hit = oracle_analysis(env, self.instruction)["failure_type"] == "missed_grasp"
        elif kind == "always_after":
            hit = self.n_chunks >= int(self.trig.get("chunk", 4))
        else:
            raise ValueError(f"unknown trigger {kind}")
        self.streak = self.streak + 1 if hit else 0
        return self.streak >= int(self.trig.get("consecutive", 1))

    def after_chunk(self, reply, env, frames=None):
        self.n_chunks += 1
        if not self.m.intervention or self.n_chunks < int(self.trig.get("min_chunk", 1)):
            return {}
        if self.n_done >= self.cfg["max_interventions"] or self.n_chunks - self.last_at < self.cfg["cooldown_chunks"]:
            return {}
        if not self._triggered(reply, env):
            return {}
        self.n_done += 1
        self.last_at = self.n_chunks
        action = self.cfg["action"]
        event = dict(kind="intervention", action=action, chunk=self.n_chunks, step=int(env.take_action_cnt),
                     gap_z=(reply or {}).get("gap_z"))
        result = {}
        if action == "analyze":
            if self.m.analyzer == "vlm":
                try:
                    pred = None
                    if (self.m.vlm or {}).get("pred_frames"):
                        pred = (self.model.infer(dict(decode_pred=True)) or {}).get("frames") or None
                    analysis = vlm_analysis(self.m.vlm, self.instruction, frames or [], pred_frames=pred)
                except Exception as e:  # keep the episode going if the VLM endpoint is down
                    analysis = dict(failed=None, error=str(e)[-300:])
            else:
                analysis = oracle_analysis(env, self.instruction)
            event["analysis"] = analysis
            if analysis.get("correction"):
                self.model.infer(dict(update_prompt=True, prompt=analysis["correction"],
                                      negative_prompt=analysis.get("negative_prompt")))
            self.model.infer(dict(context_op="reset"))
            result = {"first": True}
        elif action == "reset_context":
            self.model.infer(dict(context_op="reset"))
            result = {"first": True}
        elif action == "truncate":
            self.model.infer(dict(context_op="truncate", keep=int(self.cfg["truncate_keep"])))
        elif action == "reprompt":
            self.model.infer(dict(update_prompt=True,
                                  prompt=self.cfg["reprompt_template"].format(instruction=self.instruction)))
        elif action == "negative_prompt":
            self.model.infer(dict(update_prompt=True, negative_prompt=self.cfg["negative_template"]))
        else:
            raise ValueError(f"unknown intervention action {action}")
        self.model.infer(dict(event=event))
        self.log.append(event)
        return result
