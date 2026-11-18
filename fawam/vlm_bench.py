"""Which VLM setup gives a usable failure analysis? Offline benchmark on recorded failures against the simulator oracle.

The VLM is asked what went wrong and how to fix it (no metric depth). Varied: what it sees (keyframes, dense keyframes,
zoom on the acting wrist around the grasp, real-vs-imagined pairs, a successful rollout as reference, the whole rollout
as a video) and how it is asked (structured answer, step by step, checklist). Scored per variant and model:
  type       failure class as the oracle (grasp miss / dropped / misplaced)
  arm        acting arm as the oracle
  fix        where the oracle says "the gripper was too high": the answer says to go lower (and not higher);
             as a check against always answering "lower": how often it says "lower" where the oracle sees no height error

  python -m fawam.vlm_bench run --model Qwen/Qwen3-VL-8B-Instruct [--variants ...]   # needs a vLLM server on :8001
  python -m fawam.vlm_bench summary                                                  # -> results/vlm_bench/summary.json
"""
import argparse
import base64
import io
import json
import re
import subprocess
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from fawam import failure_analysis as fa
from fawam.paths import ROOT

P0 = ROOT / "results/sweeps/p0_baseline_table/base"
OUT = ROOT / "results/vlm_bench"
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
# (setting, task, n): the test task at both shifts, plus tasks with other failure types
EPISODES = [("table_h_m05", "move_playingcard_away", 30), ("table_h_m10", "move_playingcard_away", 30),
            ("table_h_m05", "handover_block", 15), ("table_h_m05", "move_can_pot", 15), ("table_h_m05", "place_shoe", 15)]

SCHEMA = ('{"failure_type": "missed_grasp|misaligned_grasp|dropped|misplaced|collision|other", "arm": "left|right", '
          '"cause": "<one sentence: where was the gripper relative to the object when it failed>", '
          '"fix": "<one imperative sentence: what the robot must do differently in the next attempt>", '
          '"fix_direction": "left|right|forward|backward|higher|lower|hold_longer|other"}')
ASK = {
    "legacy": ('Did the robot fail? Answer with JSON only: {"failure_type": "missed_grasp|dropped|misplaced|collision|none", '
               '"arm": "left|right", "cause": "<one sentence, concrete, with directions/distances>", '
               '"delta_cm": {"x": <forward +>, "y": <left +>, "z": <up +>}, "correction": "<one imperative instruction>"}'),
    "open": "Explain what went wrong and how the robot must act differently. Answer with JSON only: " + SCHEMA,
    "steps": ("First write one short line per image: where is the gripper of the acting arm relative to the object (above "
              "it, at it, beside it; touching it or not; fingers open or closed). Then decide what went wrong and how to "
              "fix it, and end with this JSON: " + SCHEMA),
    "checklist": ("Answer these questions one by one, each in one short line:\n1. Which arm tried to grasp?\n2. When its "
                  "fingers closed, was the object between the fingers?\n3. If not: was the gripper too high above the "
                  "object, too low, or beside it (which side)?\n4. Did the object move with the gripper afterwards?\n"
                  "Then end with this JSON: " + SCHEMA),
    # 2026-11-18: the prompt of the pipeline's VLM "detail" mode (fawam/failure_analysis.vlm_rollout)
    "detail": ('Look closely at the image where the gripper closes and the ones just before it. Answer with JSON only: '
               '{"failure_type": "missed_grasp|dropped|misplaced|collision", "arm": "left|right", '
               '"phase": "approach|grasp|lift|transport|place|release", "what_happened": "<two concrete sentences: where '
               'exactly the gripper was relative to the object when it closed or let go (above / beside / in front of / '
               'behind it, touching it or not), and what the object did>", "fix": "<one imperative sentence: what the '
               'robot must do differently in the next attempt>"}'),
    # native formats of fine-tuned failure VLMs (paraphrased from their repos, 2026-11-18)
    "vifailback": ("These images are captured by the head camera of the dual-arm robot in chronological order. Answer "
                   "step by step: 1. Was the task 'Successful' or 'Failed'? 2. At which image (keyframe) does the "
                   "failure happen? 3. Failure type: one of 'gripper state', 'gripper 6d-pose', 'task planning', "
                   "'human intervention'. 4. Low-level command for the correction: name the gripper (left/right) and "
                   "the move, rotate or grasp action, with the direction. Think inside <Think></Think>, then give the "
                   "result inside <Answer></Answer>."),
    "robofac": ("Why did the robot fail this task? Explain the failure in detail. Then give a correction suggestion: in "
                "which direction, relative to the robot arm, and how much should the gripper move to succeed?"),
}
# variant -> (visual input, question, give the prediction-error curve)
VARIANTS = {
    "legacy_cm":        ("keys6", "legacy", True),      # the setup used so far (asks for centimetres)
    "open":             ("keys6", "open", True),
    "open_nocurve":     ("keys6", "open", False),
    "steps":            ("keys6", "steps", True),
    "checklist":        ("keys6", "checklist", True),
    "dense12":          ("keys12", "open", True),
    "wrist_zoom":       ("zoom", "open", True),
    "pred_pairs":       ("pairs", "open", True),
    "success_ref":      ("ref", "open", True),
    "video":            ("video", "open", True),
    "zoom_pairs_steps": ("zoom+pairs", "steps", True),
    # two questions (user 2026-11-13): 1. at which image does the attempt go wrong (the VLM localises it itself, no
    # hint when the gripper closes), 2. enlarged views around THAT moment + real vs predicted: what to change, how much
    "two_stage":        ("two_stage", "open", True),
    "detail":           ("keys6", "detail", True),
    "native_vifailback": ("head15", "vifailback", False),
    "native_robofac":   ("frames12", "robofac", False),
}
ASK_LOCATE = ('Step 1 of 2: find the moment of the failure. Answer with JSON only: {"failure_image": <number of the image '
              'at which the attempt goes wrong>, "arm": "left|right", "failure_type": "missed_grasp|misaligned_grasp|'
              'dropped|misplaced|collision|other", "what_happened": "<one short sentence>"}')
ASK_FIX = ("Step 2 of 2: look at image 3. Where is the gripper relative to the object: too high above it, too low, beside "
           "it (which side), or at it? How must the robot act differently at this moment? Answer with JSON only: " +
           SCHEMA[:-1] + ', "amount": "slightly|clearly|a lot"}')


# ------------------------------------------------------------------------------------------------ inputs
def jpeg(img, scale=1):
    from PIL import Image
    im = Image.fromarray(np.asarray(img, dtype=np.uint8))
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}


def load(ep, kind="real"):
    import imageio.v3 as iio
    return {c: iio.imread(ep / f"{kind}_{c}.mp4") for c in CAMS}


def hcat(imgs):
    h = max(i.shape[0] for i in imgs)
    return np.concatenate([np.pad(i, ((0, h - i.shape[0]), (0, 0), (0, 0))) for i in imgs], axis=1)


def episode_info(ep):
    """Everything a real robot also has: instruction, gripper closing time and arm (proprioception), gap curve."""
    with h5py.File(ep / "episode.h5", "r") as f:
        gap = fa.gap_curve(f)
        step, arm = fa.first_close(f)
        fe = f["frames/env_step"][()]
        close = int(np.searchsorted(fe, step)) if step is not None else len(fe) // 3
        return dict(instr=str(f.attrs["instruction"]), gap=gap, arm=arm or "left", close=close,
                    latent=int(fa.step_to_latent(f, step)) if step is not None else 4, task=str(f.attrs["task_name"]))


def build(ep, visual, info, ref_ep=None):
    """-> (content parts, description of the images for the prompt)."""
    real = load(ep)
    T = min(len(v) for v in real.values())
    close = min(info["close"], T - 1)
    wrist = f"cam_{info['arm']}_wrist"
    parts, notes = [], []

    def keys(n):
        idx = sorted(set(np.linspace(2, T - 1, n - 1).astype(int).tolist()) | {close})
        parts.extend(jpeg(hcat([real[c][i] for c in CAMS])) for i in idx)
        notes.append(f"Images 1-{len(idx)}: keyframes of the rollout in time order, each showing the head camera, the "
                     f"left wrist camera and the right wrist camera side by side. The gripper closes at image "
                     f"{idx.index(close) + 1}.")

    for v in visual.split("+"):
        n0 = len(parts)
        if v in ("keys6", "keys12"):
            keys(int(v[4:]))
        elif v == "head15":  # ViFailback: separate head-camera frames in time order
            idx = np.linspace(0, T - 1, 15).astype(int).tolist()
            parts.extend(jpeg(real["cam_high"][i]) for i in idx)
            notes.append(f'The task is: "{info["instr"]}".')
        elif v == "frames12":  # RoboFAC: frames of the video (about 1 per second), all three views
            idx = np.linspace(0, T - 1, 12).astype(int).tolist()
            parts.extend(jpeg(hcat([real[c][i] for c in CAMS])) for i in idx)
            notes.append(f'The images are frames of a video of a bimanual robot doing the task "{info["instr"]}" '
                         f'(head camera, left wrist camera, right wrist camera side by side).')
        elif v == "zoom":
            idx = [min(max(close + d, 0), T - 1) for d in (-6, -3, -1, 0, 2, 5)]
            parts.extend(jpeg(hcat([real["cam_high"][i], real[wrist][i]]), scale=2) for i in idx)
            notes.append(f"Images {n0 + 1}-{n0 + 6}: enlarged views around the grasp in time order (left: head camera, "
                         f"right: wrist camera of the {info['arm']} arm). The gripper closes at image {n0 + 4}.")
        elif v == "pairs":
            pred = load(ep, "pred")
            idx = [min(4 * (info["latent"] + d), T - 1, len(pred[wrist]) - 1) for d in (0, 1)]
            parts.extend(jpeg(hcat([real[wrist][i], pred[wrist][i], real["cam_high"][i], pred["cam_high"][i]]), scale=2)
                         for i in idx)
            notes.append(f"Images {n0 + 1}-{n0 + 2}: at and just after the grasp, from left to right: REAL wrist view, the "
                         f"wrist view the robot's world model PREDICTED, REAL head view, PREDICTED head view. Differences "
                         f"between real and predicted show where the robot's expectation was wrong.")
        elif v == "ref":
            keys(6)
            if ref_ep is not None:
                rinfo = episode_info(ref_ep)
                rr = load(ref_ep)
                rc = min(rinfo["close"], min(len(x) for x in rr.values()) - 1)
                idx = [max(rc - 4, 0), rc, min(rc + 4, len(rr["cam_high"]) - 1)]
                n1 = len(parts)
                parts.extend(jpeg(hcat([rr[c][i] for c in CAMS])) for i in idx)
                notes.append(f"Images {n1 + 1}-{n1 + 3}: for comparison, a SUCCESSFUL execution of the same task (before, "
                             f"at and after the grasp).")
        elif v == "video":
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                from fawam.report import ffmpeg_bin
                cmd = [ffmpeg_bin(), "-y", "-loglevel", "error"]
                for c in CAMS:
                    cmd += ["-i", str(ep / f"real_{c}.mp4")]
                cmd += ["-filter_complex", "hstack=inputs=3:shortest=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", tmp.name]
                subprocess.run(cmd, check=True)
                b64 = base64.b64encode(Path(tmp.name).read_bytes()).decode()
            parts.append({"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + b64}})
            notes.append("The video shows the whole rollout: head camera, left wrist camera and right wrist camera side "
                         "by side.")
    return parts, " ".join(notes)


def prompt(info, note, ask, curve):
    if ask in ("vifailback", "robofac"):  # native formats: no failure given away, no prediction-error curve
        return note + "\n" + ASK[ask]
    t = f'A bimanual robot tried: "{info["instr"]}" and failed. {note}'
    if curve:
        g = info["gap"]
        t += ("\nPrediction error of the robot's world model per time step (0 = the world behaved as it expected, higher = "
              "reality differed from its expectation): " + ", ".join(f"t{k}:{g[k]:.2f}" for k in sorted(g)) +
              f". The gripper closes at t{info['latent']}.")
    return t + "\n" + ASK[ask]


def two_stage(ep, info, a, curve=True):
    """-> (answer of step 2, answer of step 1, frame the VLM picked)."""
    real = load(ep)
    T = min(len(v) for v in real.values())
    idx = np.linspace(2, T - 1, 12).astype(int)
    t = (f'A bimanual robot tried: "{info["instr"]}" and failed. Images 1-12: keyframes of the rollout in time order, each '
         f'showing the head camera, the left wrist camera and the right wrist camera side by side, at time steps '
         f'{", ".join("t%d" % (i // 4) for i in idx)}.')
    if curve:
        g = info["gap"]
        t += ("\nPrediction error of the robot's world model per time step (0 = the world behaved as it expected, higher "
              "= reality differed from its expectation): " + ", ".join(f"t{k}:{g[k]:.2f}" for k in sorted(g)) + ".")
    t1 = chat(a.endpoint, a.model, [jpeg(hcat([real[c][i] for c in CAMS])) for i in idx] +
              [{"type": "text", "text": t + "\n" + ASK_LOCATE}], 200)
    a1 = parse(t1)
    try:
        k = min(max(int(a1.get("failure_image")) - 1, 0), 11)
    except (TypeError, ValueError):
        k = 5
    fi = int(idx[k])
    arm = a1.get("arm") if a1.get("arm") in ("left", "right") else info["arm"]
    wrist = f"cam_{arm}_wrist"
    pred = load(ep, "pred")
    parts = [jpeg(hcat([real["cam_high"][i], real[wrist][i]]), scale=2)
             for i in (min(max(fi + d, 0), T - 1) for d in (-6, -3, 0, 3, 6))]
    j = min(fi, len(pred[wrist]) - 1)
    parts.append(jpeg(hcat([real[wrist][min(j, T - 1)], pred[wrist][j]]), scale=2))
    t2 = (f'A bimanual robot tried: "{info["instr"]}" and failed. You found the moment at which it goes wrong'
          f'{" (" + str(a1["what_happened"]) + ")" if a1.get("what_happened") else ""}. Images 1-5: enlarged views around '
          f'that moment in time order (left: head camera, right: wrist camera of the {arm} arm); image 3 is the moment '
          f'itself. Image 6: at that moment, left the REAL wrist view, right the wrist view the robot\'s world model '
          f'PREDICTED; differences show where its expectation was wrong.\n' + ASK_FIX)
    return chat(a.endpoint, a.model, parts + [{"type": "text", "text": t2}], 350), t1, fi


# ------------------------------------------------------------------------------------------------ run
def chat(endpoint, model, content, max_tokens):
    body = json.dumps(dict(model=model, messages=[{"role": "user", "content": content}], temperature=0.0,
                           max_tokens=max_tokens)).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def episodes():
    out = []
    for setting, task, n in EPISODES:
        eps = [e for e in sorted((P0 / setting / "episodes/fail" / task).glob("*")) if (e / "analysis.json").exists()]
        out += [(setting, e) for e in eps[:n]]
    return out


def reference(task):
    d = P0 / "table_h_z00/episodes/success" / task
    eps = sorted(d.glob("*")) if d.exists() else []
    return eps[0] if eps else None


def run(a):
    out = OUT / re.sub(r"[^A-Za-z0-9.-]+", "_", a.model)
    out.mkdir(parents=True, exist_ok=True)
    eps = episodes()
    print(f"{len(eps)} failed episodes, model {a.model}")
    for name in a.variants or VARIANTS:
        visual, ask, curve = VARIANTS[name]
        path = out / f"{name}.jsonl"
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
        if any("error" in r for r in recs):  # retry failed calls: keep only the answered ones
            recs = [r for r in recs if "error" not in r]
            path.write_text("".join(json.dumps(r) + "\n" for r in recs))
        done = {r["ep"] for r in recs}

        def one(item):
            setting, ep = item
            if str(ep) in done:
                return None
            t0 = time.time()
            rec = dict(ep=str(ep), setting=setting, variant=name)
            try:
                info = episode_info(ep)
                if visual == "two_stage":
                    rec["text"], rec["stage1"], rec["frame"] = two_stage(ep, info, a, curve)
                    rec["seconds"] = round(time.time() - t0, 1)
                    return rec
                parts, note = build(ep, visual, info, reference(info["task"]) if "ref" in visual else None)
                rec["text"] = chat(a.endpoint, a.model, parts + [{"type": "text", "text": prompt(info, note, ask, curve)}],
                                   (900 if ask in ("steps", "checklist", "vifailback", "robofac") else 350) *
                                   (3 if "cosmos" in a.model.lower() else 1))  # reasoning model: <think> first
            except Exception as e:  # a variant the server cannot handle (e.g. video) must not stop the benchmark
                rec["error"] = repr(e)[-300:]
            rec["seconds"] = round(time.time() - t0, 1)
            return rec

        t0, n_err = time.time(), 0
        with ThreadPoolExecutor(a.workers) as pool, open(path, "a") as f:
            for rec in pool.map(one, eps):
                if rec:
                    n_err += "error" in rec
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
        print(f"{name}: {time.time() - t0:.0f} s, {n_err} errors", flush=True)


# ------------------------------------------------------------------------------------------------ scoring
LOWER = ("too high", "above the", "above its", "lower", "descend", "move down", "further down", "not low enough",
         "did not go low", "didn't go low", "did not reach down", "deeper", "closer to the table")
HIGHER = ("too low", "raise the", "move up", "higher before", "lift the gripper", "go higher", "below the")
GRASP = {"missed_grasp", "misaligned_grasp"}


def parse(text):
    try:
        return json.loads(text[_last_obj(text):text.rindex("}") + 1])
    except (ValueError, TypeError):
        return {}


def _last_obj(text):
    """Start of the last top-level JSON object (answers with reasoning may contain braces before it)."""
    depth, end = 0, text.rindex("}")
    for i in range(end, -1, -1):
        depth += text[i] == "}"
        depth -= text[i] == "{"
        if depth == 0:
            return i
    raise ValueError


def vertical(ans, text):
    """lower | higher | none: what the answer says about the height of the gripper."""
    d = ans.get("fix_direction")
    if d in ("lower", "higher"):
        return d
    if isinstance(ans.get("delta_cm"), dict):
        z = float(ans["delta_cm"].get("z") or 0)
        if abs(z) >= 0.5:
            return "lower" if z < 0 else "higher"
    t = " ".join(str(ans.get(k) or "") for k in ("cause", "what_happened", "fix", "correction")).lower().strip() \
        or text.lower()[-600:]
    lo, hi = any(w in t for w in LOWER), any(w in t for w in HIGHER)
    return "lower" if lo and not hi else "higher" if hi and not lo else "none"


def text_type(ans, text):
    """failure type from the JSON, else from free text (native formats of fine-tuned VLMs)."""
    if ans.get("failure_type"):
        return ans["failure_type"]
    t = text.lower()
    if "<answer>" in t:
        t = t[t.rindex("<answer>"):]
    if any(w in t for w in ("6d-pose", "6d pose", "miss", "fail to grasp", "failed to grasp", "not grasp", "grasp")):
        return "missed_grasp"
    if any(w in t for w in ("drop", "slip", "fell", "fall")):
        return "dropped"
    if any(w in t for w in ("place", "position", "target")):
        return "misplaced"
    return None


def summary(a):
    rows = []
    for mdir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        for path in sorted(mdir.glob("*.jsonl")):
            recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            s = dict(model=mdir.name, variant=path.stem, n=len(recs), errors=sum("error" in r for r in recs),
                     seconds=round(float(np.median([r["seconds"] for r in recs])), 1) if recs else None)
            ok = [r for r in recs if "text" in r]
            sc = {k: [] for k in ("type", "arm", "lower", "higher", "lower_false", "parsed", "loc")}
            examples = []
            for r in ok:
                o = json.loads((Path(r["ep"]) / "analysis.json").read_text())
                ans = parse(r["text"])
                sc["parsed"].append(bool(ans))
                if "frame" in r and o.get("latent") is not None:  # two_stage: picked moment within 2 latents of the oracle's
                    sc["loc"].append(abs(r["frame"] // 4 - int(o["latent"])) <= 2)
                cls = lambda t: "grasp" if t in GRASP else t
                sc["type"].append(cls(text_type(ans, r["text"])) == cls(o.get("failure_type")))
                if o.get("arm"):
                    sc["arm"].append(ans.get("arm") == o["arm"])
                if (o.get("delta_m") or [0, 0, 0])[2] < -0.015:  # the oracle says: the gripper was too high
                    v = vertical(ans, r["text"])
                    sc["lower"].append(v == "lower")
                    sc["higher"].append(v == "higher")
                    if len(examples) < 2:
                        ex = ans.get("fix") or ans.get("correction") or r["text"][-300:]
                        examples.append(f"{ans.get('what_happened') or ''} | {ex}".strip(" |"))
                elif abs((o.get("delta_m") or [0, 0, 0])[2]) < 0.005:  # no height error: "lower" would be wrong
                    sc["lower_false"].append(vertical(ans, r["text"]) == "lower")
            s.update({k: (round(float(np.mean(v)), 3) if v else None) for k, v in sc.items()})
            s.update(n_height=len(sc["lower"]), examples=examples)
            rows.append(s)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1))
    print(f"{'model':28s} {'variant':18s} {'n':>4s} {'err':>4s} {'type':>6s} {'arm':>6s} {'lower':>6s} {'higher':>7s} {'lowFP':>6s} {'loc':>6s} {'s':>5s}")
    for s in rows:
        f = lambda x: "   –  " if x is None else f"{100 * x:5.0f}%"
        print(f"{s['model'][:28]:28s} {s['variant']:18s} {s['n']:4d} {s['errors']:4d} {f(s['type'])} {f(s['arm'])} "
              f"{f(s['lower'])} {f(s['higher'])} {f(s['lower_false'])} {f(s['loc'])} {s['seconds'] or 0:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    r.add_argument("--variants", nargs="*")
    r.add_argument("--workers", type=int, default=6)
    sub.add_parser("summary")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else summary(a)


if __name__ == "__main__":
    main()
