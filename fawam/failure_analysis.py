"""Failure analysis of one recorded episode -> <episode>/analysis.json (Learn-from-Failure, docs/METHOD.md).

Combines the two signals the method is built on:
  1. prediction error: per latent frame, cosine distance of the imagined vs. the observed wrist latents. Its peak
     localises *when* the model's expectation broke; it is also stored as a per-frame weight for adaptation.
  2. failure analysis by an analyzer:
       oracle  ground truth from the simulator (failure type/time from analysis/analyze.py's sim labels; correction
               from the gripper-object geometry vs. successful grasps of the same task) - the upper bound
       vlm     a VLM gets the real and the imagined views at the failure time and returns type, cause, a
               correction instruction, a negative prompt and a metric correction (delta_cm)
Output (all analyzers): failed, failure_type, arm, step, latent, gap (per latent), gap_peak_latent, cause,
correction (text), negative_prompt, delta_m [dx, dy, dz] for the arm (robot base frame), window [l0, l1] (latents
whose actions the delta is applied to), truncate_at (latent after which the rollout is no useful training data).

  python -m fawam.failure_analysis <episode dirs or roots> [--analyzer oracle|vlm] [--force]
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from fawam.paths import ROOT, WRIST_CAMS

sys.path.insert(0, str(ROOT / "analysis"))
GRASP_REF = ROOT / "artifacts" / "grasp_ref.json"
DEFAULT_DZ = 0.14  # gripper link above the object's centre at a successful grasp (median over z00 successes)


def gap_curve(f):
    """{latent: max cosine distance over wrist cams} for observed latents k >= 1."""
    pi = {int(k): i for i, k in enumerate(f["model/pred/latent_index"][()])}
    oi = {int(k): i for i, k in enumerate(f["model/obs/latent_index"][()])}
    out = {}
    for k in sorted(set(pi) & set(oi)):
        if k < 1:
            continue
        v = []
        for cam in WRIST_CAMS:
            p = f[f"model/pred/latents/{cam}"][pi[k]].astype(np.float32).ravel()
            o = f[f"model/obs/latents/{cam}"][oi[k]].astype(np.float32).ravel()
            v.append(1 - p @ o / (np.linalg.norm(p) * np.linalg.norm(o) + 1e-8))
        out[k] = float(max(v))
    return out


def step_to_latent(f, step):
    li, es = f["frames/latent_index"][()], f["frames/env_step"][()]
    hit = np.where(es >= step)[0]
    return int(li[hit[0]]) if len(hit) else int(li[-1])


def sim_label(ep_dir):
    """Failure type, step and arm from the simulator ground truth (analysis/analyze.py)."""
    import analyze
    ep = analyze.load_episode(Path(ep_dir))
    events, ftype, fstep = analyze.label_failures(ep)
    arm = None
    for e in events:
        if e["type"] == ftype and e["step"] == fstep:
            arm = e["arm"]
    return ftype, fstep, arm


def grasp_reference(task, arm):
    """(ee - object centre) at the first close of successful grasps: [dx, dy, dz] for this task and arm."""
    default = [0.0, 0.0, DEFAULT_DZ]
    if not GRASP_REF.exists():
        return np.array(default)
    ref = json.loads(GRASP_REF.read_text())
    r = ref.get(f"{task}|{arm}") or ref.get(task) or ref.get("_all") or default
    return np.array(r if isinstance(r, list) else [0.0, 0.0, r])


def first_close(f, arm=None):
    raw = f["actions/executed_raw_action"][()]
    steps = f["actions/executed_env_step"][()]
    cand = []
    for a, ch in (("left", 7), ("right", 15)):
        if arm not in (None, a):
            continue
        hit = np.where(raw[:, ch] < 0.5)[0]
        if len(hit):
            cand.append((int(steps[hit[0]]), a))
    return min(cand) if cand else (None, None)


def geometry_at(f, step, arm):
    """End effector at `step`, nearest object (in xy) at the episode start (objects can already be knocked over)."""
    sim_steps = f["sim/env_step"][()]
    j = min(int(np.searchsorted(sim_steps, step)), len(sim_steps) - 1)
    ee = f[f"sim/{arm}_ee_pose"][j][:3]
    objs = f["sim/object_pose"][0][:, :3]
    obj = objs[np.argmin(np.linalg.norm(objs[:, :2] - ee[:2], axis=1))]
    return ee, obj


def oracle(ep_dir, f, task):
    ftype, fstep, arm = sim_label(ep_dir)
    out = dict(failure_type=ftype, step=fstep, arm=arm, delta_m=[0.0, 0.0, 0.0])
    instr = str(f.attrs["instruction"])
    if ftype == "missed_grasp":
        ee, obj = geometry_at(f, fstep, arm)
        target = obj + grasp_reference(task, arm)
        dz = float(target[2] - ee[2])
        dxy = (target[:2] - ee[:2]).astype(float)
        side = float(np.linalg.norm(dxy)) > 0.03
        parts = []
        if side:
            parts.append(f"{np.linalg.norm(dxy) * 100:.0f} cm to the side of the object")
        if abs(dz) > 0.02:
            parts.append(f"{abs(dz) * 100:.0f} cm {'above' if dz < 0 else 'below'} the grasp height")
        where = " and ".join(parts) or "at the object but without a stable grasp"
        moves = []
        if side:
            moves.append("over the object")
        if abs(dz) > 0.02:
            moves.append(f"{abs(dz) * 100:.0f} cm {'lower' if dz < 0 else 'higher'}")
        out.update(delta_m=[float(dxy[0]) if side else 0.0, float(dxy[1]) if side else 0.0, dz if abs(dz) > 0.02 else 0.0],
                   failure_type="missed_grasp" if not side else "misaligned_grasp",
                   cause=f"the {arm} gripper closed {where} and missed it",
                   correction=f"{instr} Move the {arm} gripper {' and '.join(moves) or 'precisely onto the object'} "
                              f"before closing it.",
                   negative_prompt=f"The {arm} gripper closes in the air next to or above the object without touching it.")
    elif ftype == "dropped":
        out.update(cause=f"the object slipped out of the {arm} gripper",
                   correction=f"{instr} Keep the {arm} gripper firmly closed until the object is placed.",
                   negative_prompt="The object slips out of the gripper and falls onto the table.")
    elif ftype in ("misplaced", "timeout"):
        out.update(cause="the object was not placed at the target" if ftype == "misplaced"
                   else "the robot never grasped an object",
                   correction=instr, negative_prompt="The robot fails to complete the task.")
    return out


def vlm(ep_dir, f, cfg, latent):
    import imageio.v3 as iio
    from fawam.robotwin_hooks import vlm_analysis
    idx = 4 * latent
    real = [iio.imread(Path(ep_dir) / f"real_{c}.mp4", index=min(idx, iio.improps(Path(ep_dir) / f"real_{c}.mp4").shape[0] - 1))
            for c in ("cam_high", "cam_left_wrist", "cam_right_wrist")]
    pred = [iio.imread(Path(ep_dir) / f"pred_{c}.mp4", index=min(idx, iio.improps(Path(ep_dir) / f"pred_{c}.mp4").shape[0] - 1))
            for c in ("cam_high", "cam_left_wrist", "cam_right_wrist")] if cfg.get("pred_frames", True) else None
    a = vlm_analysis(cfg, str(f.attrs["instruction"]), real, pred)
    d = a.get("delta_cm") or {}
    arm = a.get("arm") if a.get("arm") in ("left", "right") else None
    return dict(failure_type=a.get("failure_type", "unknown"), arm=arm, step=None, cause=a.get("cause"),
                correction=a.get("correction"), negative_prompt=a.get("negative_prompt"),
                delta_m=[float(d.get(k, 0) or 0) / 100 for k in ("x", "y", "z")], vlm_raw=a)


def analyze_episode(ep_dir, analyzer="oracle", vlm_cfg=None, window=3, after=1, force=False):
    ep_dir = Path(ep_dir)
    out_path = ep_dir / ("analysis.json" if analyzer == "oracle" else f"analysis_{analyzer}.json")
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())
    with h5py.File(ep_dir / "episode.h5", "r") as f:
        task = str(f.attrs["task_name"])
        failed = not bool(f.attrs["success"])
        gap = gap_curve(f)
        res = dict(failed=failed, task=task, analyzer=analyzer, gap={str(k): v for k, v in gap.items()})
        if failed:
            if analyzer == "oracle":
                res.update(oracle(ep_dir, f, task))
            else:
                # time from the prediction gap: its peak after the first close (the VLM gets that moment)
                close_step, _ = first_close(f)
                start = step_to_latent(f, close_step) - 2 if close_step is not None else 1
                cands = {k: v for k, v in gap.items() if k >= start} or gap
                peak = max(cands, key=cands.get)
                res.update(vlm(ep_dir, f, vlm_cfg or {}, peak))
                res["step"] = int(f["frames/env_step"][()][min(4 * peak, len(f["frames/env_step"]) - 1)])
            if res.get("arm") is None:
                _, res["arm"] = first_close(f)
            if res.get("step") is not None:
                L = step_to_latent(f, res["step"])
                res.update(latent=L, window=[max(1, L - window), L], truncate_at=L + after)
            L0 = res.get("latent", 1)
            after_close = {k: v for k, v in gap.items() if L0 - window <= k <= L0 + 2}
            res["gap_peak_latent"] = max(after_close, key=after_close.get) if after_close else None
    out_path.write_text(json.dumps(res, indent=1, default=float))
    return res


def build_grasp_ref(roots):
    """Median gripper height above the object centre at the first close of successful episodes, per task."""
    from fawam.latent_data import find_episodes
    per = {}
    for e in find_episodes(roots, success=True):
        with h5py.File(e / "episode.h5", "r") as f:
            step, arm = first_close(f)
            if step is None:
                continue
            ee, obj = geometry_at(f, step, arm)
            t = str(f.attrs["task_name"])
            for key in (f"{t}|{arm}", t):
                per.setdefault(key, []).append((ee - obj).astype(float))
    ref = {k: np.median(np.stack(v), axis=0).tolist() for k, v in per.items()}
    ref["_all"] = [0.0, 0.0, float(np.median([x[2] for k, v in per.items() if "|" in k for x in v]))] if per \
        else [0.0, 0.0, DEFAULT_DZ]
    GRASP_REF.parent.mkdir(parents=True, exist_ok=True)
    GRASP_REF.write_text(json.dumps(ref, indent=1))
    return ref


# ------------------------------------------------------------------------------------------ composable signals
SIGNALS = ("gap", "rollout", "imagine", "vlm", "oracle")
LEGACY = {"oracle": ["oracle"], "vlm": ["gap", "vlm"], "none": ["gap"]}


def rollout_event(f):
    """Failure type/time/arm from the policy's own rollout: gripper state (proprioception) plus a contact flag
    (stand-in for force/tactile sensing), for both arms. No object positions. Earliest event wins:
    missed_grasp = a close with no contact within 25 steps; dropped = contact lost while the gripper stays closed."""
    st = f["sim/env_step"][()]
    events = []
    for arm in ("left", "right"):
        c = f[f"sim/contact_{arm}_gripper"][()].sum(1) > 1e-6
        g = f[f"sim/{arm}_gripper"][()] < 0.5
        for j in np.where(g[1:] & ~g[:-1])[0] + 1:
            if not c[j:j + 25].any():
                events.append((int(st[j]), "missed_grasp", arm))
        held = c & g
        for j in np.where(held[:-1] & ~held[1:])[0] + 1:
            if g[j] and j - np.argmax(held[:j][::-1] == 0) > 5:  # still closed after >= 5 held steps
                events.append((int(st[j]), "dropped", arm))
    if not events:
        step, arm = first_close(f)
        return dict(failure_type="unknown" if step is not None else "no_grasp", step=None, arm=arm)
    step, ftype, arm = min(events)
    return dict(failure_type=ftype, step=step, arm=arm)


def gap_onset(f, gap, task, z=2.5):
    """First latent whose prediction gap exceeds the task's success statistics by z (the online detector's rule);
    falls back to the peak. The peak itself is late: after a failure everything diverges."""
    stats_p = ROOT / "artifacts" / "gap_stats.json"
    if stats_p.exists() and gap:
        st = json.loads(stats_p.read_text())
        s = st.get(task) or st.get("_all") or {}
        ref = [s[c] for c in WRIST_CAMS if c in s]
        if ref:
            mu, sd = max(r["mean"] for r in ref), max(r["std"] for r in ref)
            for k in sorted(gap):
                if (gap[k] - mu) / max(sd, 1e-6) > z:
                    return int(k)
    return int(max(gap, key=gap.get)) if gap else None


def vlm_keyframes(ep_dir, f, cfg, n=6):
    """VLM-only analysis: the VLM sees n keyframes of the whole episode (head | left | right side by side) and has to
    find the failure itself (no prediction gap, no simulator)."""
    import imageio.v3 as iio
    from fawam.robotwin_hooks import _jpeg_b64
    import urllib.request
    vids = {c: iio.imread(Path(ep_dir) / f"real_{c}.mp4") for c in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
    T = min(len(v) for v in vids.values())
    idx = np.linspace(4, T - 1, n).astype(int)
    imgs = [np.concatenate([vids[c][i] for c in vids], axis=1) for i in idx]
    instr = str(f.attrs["instruction"])
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(im)}"}} for im in imgs]
    content.append({"type": "text", "text": (
        f'A bimanual robot tried: "{instr}" and failed. The {n} images are keyframes in time order (each: head, left '
        'wrist, right wrist camera). Answer with JSON only: {"failure_frame": <1..%d>, "failure_type": "missed_grasp|'
        'dropped|misplaced|collision", "arm": "left|right", "cause": "<one sentence with directions/distances>", '
        '"correction": "<one imperative instruction>", "negative_prompt": "<the failure to avoid>", '
        '"delta_cm": {"x": <forward +>, "y": <left +>, "z": <up +>}}' % n)})
    body = json.dumps(dict(model=cfg.get("model"), messages=[{"role": "user", "content": content}], temperature=0.0,
                           max_tokens=300)).encode()
    req = urllib.request.Request(cfg["endpoint"].rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        text = json.loads(r.read())["choices"][0]["message"]["content"]
    try:
        a = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except ValueError:
        a = dict(raw=text)
    k = int(a.get("failure_frame", n)) - 1
    frame = int(idx[min(max(k, 0), n - 1)])
    a["latent"] = (frame + 3) // 4
    if a.get("correction"):
        a["correction"] = f"{instr} {a['correction']}"
    return a


def analyze_signals(ep_dir, signals, vlm_cfg=None, window=3, after=1, force=False, delta_min=0.015):
    """Learn-from-Failure analysis from any subset of signals (docs/METHOD.md, section 2):
      gap      prediction error: failure time = its peak, per-frame loss weights (use_gap_weights)
      rollout  policy's gripper commands + contact flag: failure type/time/arm
      imagine  imagination pose probe: metric correction = imagined - observed ee position at the failure
      vlm      VLM: type, cause, correction text, negative prompt, delta_cm (at the time from gap/rollout if
               present, else from keyframes of the whole episode)
      oracle   simulator ground truth for everything (upper bound)
    Priority when several give the same field: time oracle > rollout > gap > vlm; delta oracle > imagine > vlm;
    text oracle > vlm > imagine/rollout templates. Writes analysis_<signals>.json and returns the dict."""
    signals = LEGACY.get(signals, signals) if isinstance(signals, str) else list(signals)
    tag = "+".join(s for s in SIGNALS if s in signals)
    ep_dir = Path(ep_dir)
    out_path = ep_dir / f"analysis_{tag}.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())
    with h5py.File(ep_dir / "episode.h5", "r") as f:
        task, instr = str(f.attrs["task_name"]), str(f.attrs["instruction"])
        failed = not bool(f.attrs["success"])
        gap = gap_curve(f)
        res = dict(failed=failed, task=task, signals=tag, use_gap_weights="gap" in signals,
                   gap={str(k): v for k, v in gap.items()}, delta_m=[0.0, 0.0, 0.0], correction=None,
                   negative_prompt=None, failure_type="unknown", arm=None, latent=None)
        if not failed:
            out_path.write_text(json.dumps(res, default=float))
            return res
        # --- when / which arm
        if "oracle" in signals:
            res.update(oracle(ep_dir, f, task))
            res["latent"] = step_to_latent(f, res["step"]) if res.get("step") is not None else None
        else:
            if "rollout" in signals:
                ev = rollout_event(f)
                res.update(failure_type=ev["failure_type"], arm=ev["arm"])
                if ev["step"] is not None:
                    res["latent"] = step_to_latent(f, ev["step"])
            if res["latent"] is None and "gap" in signals and gap:
                peak = gap_onset(f, gap, task)
                res["latent"] = int(peak)
                if res["arm"] is None:  # the wrist whose view diverged more
                    pi = {int(k): i for i, k in enumerate(f["model/pred/latent_index"][()])}
                    oi = {int(k): i for i, k in enumerate(f["model/obs/latent_index"][()])}
                    d = {}
                    for cam, arm in (("cam_left_wrist", "left"), ("cam_right_wrist", "right")):
                        p_ = f[f"model/pred/latents/{cam}"][pi[peak]].astype(np.float32).ravel()
                        o_ = f[f"model/obs/latents/{cam}"][oi[peak]].astype(np.float32).ravel()
                        d[arm] = 1 - p_ @ o_ / (np.linalg.norm(p_) * np.linalg.norm(o_) + 1e-8)
                    res["arm"] = max(d, key=d.get)
            # --- what / how
            if "vlm" in signals:
                try:
                    if res["latent"] is not None:
                        v = vlm(ep_dir, f, vlm_cfg or {}, res["latent"])
                    else:
                        v = vlm_keyframes(ep_dir, f, vlm_cfg or {})
                        v["delta_m"] = [float((v.get("delta_cm") or {}).get(k, 0) or 0) / 100 for k in "xyz"]
                        res["latent"] = v.get("latent")
                    res.update({k: v[k] for k in ("cause", "correction", "negative_prompt") if v.get(k)})
                    if res["failure_type"] == "unknown" and v.get("failure_type"):
                        res["failure_type"] = v["failure_type"]
                    res["arm"] = res["arm"] or (v.get("arm") if v.get("arm") in ("left", "right") else None)
                    res["vlm_delta_m"] = v.get("delta_m", [0, 0, 0])
                    res["vlm_raw"] = v.get("vlm_raw", v)
                except Exception as e:  # endpoint down: keep the other signals
                    res["vlm_error"] = str(e)[-300:]
            if "imagine" in signals and res["latent"] is not None and res["arm"]:
                from fawam.pose_probe import PoseProbe
                d = PoseProbe().imagination_delta(f, int(res["latent"]), res["arm"])
                if d is not None:
                    d = np.where(np.abs(d) >= delta_min, d, 0.0)
                    res["imagine_delta_m"] = [float(x) for x in d]
            if "imagine" in signals and res.get("imagine_delta_m"):
                res["delta_m"] = res["imagine_delta_m"]
            elif "vlm" in signals and res.get("vlm_delta_m"):
                res["delta_m"] = res["vlm_delta_m"]
            if res["correction"] is None and res["arm"]:
                dz = res["delta_m"][2]
                if abs(dz) >= delta_min:
                    res["correction"] = f"{instr} Move the {res['arm']} gripper {abs(dz) * 100:.0f} cm " \
                                        f"{'lower' if dz < 0 else 'higher'} before closing it."
                elif res["failure_type"] == "missed_grasp":
                    res["correction"] = f"{instr} The {res['arm']} gripper closed without touching the object; " \
                                        f"grasp the object properly."
                elif res["failure_type"] == "dropped":
                    res["correction"] = f"{instr} Keep the {res['arm']} gripper closed until the object is placed."
            if res["negative_prompt"] is None and res["failure_type"] in ("missed_grasp", "misaligned_grasp"):
                res["negative_prompt"] = "The gripper closes in the air without touching the object."
            elif res["negative_prompt"] is None and res["failure_type"] == "dropped":
                res["negative_prompt"] = "The object slips out of the gripper and falls onto the table."
        L = res.get("latent")
        if L is not None:
            res.update(window=[max(1, int(L) - window), int(L)], truncate_at=int(L) + after)
    out_path.write_text(json.dumps(res, indent=1, default=float))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--analyzer", default="oracle", choices=["oracle", "vlm"])
    ap.add_argument("--signals", help="composable analysis, e.g. gap+rollout+imagine (overrides --analyzer)")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--grasp_ref", nargs="*", help="build artifacts/grasp_ref.json from these roots first")
    a = ap.parse_args()
    if a.grasp_ref:
        print("grasp ref:", build_grasp_ref(a.grasp_ref))
    from fawam.latent_data import find_episodes
    for e in find_episodes(a.roots, success=False):
        cfg = dict(endpoint=a.endpoint, model=a.model)
        r = analyze_signals(e, a.signals.split("+"), cfg, force=a.force) if a.signals else \
            analyze_episode(e, a.analyzer, cfg, force=a.force)
        print(f"{e.parent.name}/{e.name}: {r.get('failure_type')} arm={r.get('arm')} latent={r.get('latent')} "
              f"delta_z={r.get('delta_m', [0, 0, 0])[2]:+.3f} gap_peak={r.get('gap_peak_latent')}")


if __name__ == "__main__":
    main()
