"""Variant report: the same analyses for every tested model variant (docs/PLAN.md), with the focus on whether the
in-context memory is used. Runs hourly from scrontab (cluster/report_cron.sh) and is incremental (one cached record
per episode).

  python -m fawam.report            # -> results/report/data.js, report.json, media/*.mp4

Per variant (sweep / method) and setting, on the test task:
  outcome    first-attempt success, share of failures fixed in attempt 2 (Wilson CI), paired test against retry_plain
  behaviour  gripper height above the grasp height at the first close, attempt 1 vs. 2 (does it go lower?)
  context    attention on the memory tokens relative to their token share (per layer, per memory part), influence
             (relative change of the prediction when the memory is hidden), action shift caused by the memory in mm
  analysis   failure analysis used for the retry (text, delta) against the simulator oracle
  gap        prediction error at the grasp, attempt 1 vs. 2
  samples    2 failed + 2 successful rollouts as videos (real head | real wrist | imagined head) with their curves
Plus: meta-training curves (validation loss with vs. without memory) and the offline VLM-vs-oracle comparison.
"""
import argparse
import hashlib
import json
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from fawam import failure_analysis as fa
from fawam.paths import ROOT

TEST_TASK = "move_playingcard_away"
SETTINGS = ["table_h_m05", "table_h_m10"]
SWEEPS = ["p0_baseline_table", "p5a_signals", "p5b_vlm", "p5f_paper", "p6a_icl_content", "p6v_icl_free_vlm",
          "p9_related", "p8_icl_vlm_meta", "p8f_finetune", "p8d_heldout_type", "p8_icl_oracle_meta", "p8t_heldout_types", "p8s_strong"]
CACHE_VERSION = 5
OUT = ROOT / "results" / "report"


# ------------------------------------------------------------------------------------------------ per episode
def episode_record(ep):
    cache = ep / "report_cache.json"
    if cache.exists():
        r = json.loads(cache.read_text())
        if r.get("v") == CACHE_VERSION:
            return r
    meta = json.loads((ep / "meta.json").read_text()) if (ep / "meta.json").exists() else {}
    m = re.match(r"ep(\d+)_seed(\d+)(?:_attempt(\d+))?", ep.name)
    r = dict(v=CACHE_VERSION, ep_index=int(m.group(1)), seed=int(m.group(2)), attempt=int(m.group(3) or 1))
    with h5py.File(ep / "episode.h5", "r") as f:
        task = str(f.attrs["task_name"])
        # RoboTwin's check only tests |x| > 0.23 with open grippers, so a card that is knocked off the table counts as
        # solved. A success is kept only if the object ends where it started in height (+-5 cm) and within 0.6 m; the
        # simulator's own flag stays in sim_success, flung marks the rejected ones. (Measured 2026-11-15: real
        # successes end at dz ~ 0 and |x| ~ 0.38 m, the others 0.6-0.7 m lower or metres away.)
        sim_ok, flung = bool(f.attrs["success"]), False
        op = f["sim/object_pose"][()]
        if sim_ok and op.shape[1] > 0:
            a, b = op[0].reshape(-1)[:3], op[-1].reshape(-1)[:3]
            flung = bool(abs(float(b[2] - a[2])) > 0.05 or float(np.linalg.norm(b[:2] - a[:2])) > 0.6)
        r.update(task=task, success=sim_ok and not flung, sim_success=sim_ok, flung=flung)
        gap = fa.gap_curve(f)
        r["gap"] = [round(gap[k], 3) for k in sorted(gap)]
        step, arm = fa.first_close(f)
        r["arm"] = arm
        if step is not None and f["sim/object_pose"].shape[1] > 0:
            ee, obj = fa.geometry_at(f, step, arm)
            ref = fa.grasp_reference(task, arm)
            r["above_mm"] = round(float(ee[2] - obj[2] - ref[2]) * 1000, 1)  # gripper above the grasp height
            L = fa.step_to_latent(f, step)
            r["grasp_latent"] = int(L)
            near = [gap[k] for k in gap if L - 1 <= k <= L + 2]
            r["gap_grasp"] = round(float(max(near)), 3) if near else None
        if not r["success"] and f["sim/object_pose"].shape[1] > 0:
            try:
                o = fa.oracle(ep, f, task)
                r["oracle"] = dict(type=o.get("failure_type"), dz_cm=round(100 * (o.get("delta_m") or [0, 0, 0])[2], 1),
                                   arm=o.get("arm"))
            except Exception:  # no usable oracle for this episode
                pass
        if "fawam/memory" in f:
            g = f["fawam/memory"]
            aa, sa = g["attn_action"][()], np.maximum(g["share_action"][()], 1e-8)
            mem = dict(tokens=int(np.nanmean(g["memory_tokens"][()])),
                       attn_action=[round(float(x), 3) for x in np.nanmean(aa, 1) / sa],
                       layer_action=[round(float(x), 3) for x in np.nanmean(aa / sa[:, None], 0)],
                       heat=np.round(aa / sa[:, None], 2).tolist(),  # chunk x block: attention on the memory
                       infl_action=[round(float(x), 3) for x in g["infl_action"][()]],
                       infl_video=[round(float(x), 3) for x in g["infl_video"][()]])
            if "attn_video" in g:
                av, sv = g["attn_video"][()], np.maximum(g["share_video"][()], 1e-8)
                mem["attn_video"] = [round(float(x), 3) for x in np.nanmean(av, 1) / sv]
                mem["layer_video"] = [round(float(x), 3) for x in np.nanmean(av / sv[:, None], 0)]
            side = arm or "left"
            if f"cf_sdz_{side}" in g:  # action with memory minus action without memory, z of the grasping arm
                mem["shift_mm"] = [round(float(x) * 1000, 2) for x in g[f"cf_sdz_{side}"][()]]
            if "brk_action_part" in g and "parts" in g.attrs:
                pm, ps = g["brk_action_part"][()].mean(0), g["brk_action_part_share"][()].mean(0)
                mem["parts"] = {n: round(float(pm[i] / ps[i]), 3) for i, n in enumerate(json.loads(g.attrs["parts"]))
                                if ps[i] > 0}
                wm = g["brk_action_win"][()].mean(0)
                mem["window_frac"] = round(float(wm[1] / (wm.sum() + 1e-12)), 3)
            r["memory"] = mem
    lff = meta.get("lff") or {}
    r["phase"] = lff.get("phase")
    ana = lff.get("analysis") or {}
    if ana:
        r["analysis"] = dict(signals=lff.get("signals"), type=ana.get("failure_type"), cause=ana.get("cause"),
                             correction=ana.get("correction"),
                             dz_cm=round(100 * (ana.get("delta_m") or [0, 0, 0])[2], 1))
    r["prompt"] = meta.get("prompt")
    try:
        cache.write_text(json.dumps(r))
    except OSError:
        pass
    return r


def cell_episodes(sweep, method, setting):
    base = ROOT / "results" / "sweeps" / sweep / method / setting / "episodes"
    out = []
    for kind in ("success", "fail"):
        d = base / kind / TEST_TASK
        if d.exists():
            for ep in sorted(x for x in d.iterdir() if "_snap_" not in x.name):  # plan v2 snap continuations
                if (ep / "episode.h5").exists():
                    try:
                        out.append((ep, episode_record(ep)))
                    except Exception as e:  # half-written episode of a running job
                        print(f"skip {ep}: {e!r}")
    return out


# ------------------------------------------------------------------------------------------------ statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(c - h), 3), round(float(c + h), 3)]


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return round(float(np.median(xs)), 2) if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return round(float(np.mean(xs)), 3) if xs else None


def paired_p(final_a, final_b):
    """Exact McNemar on the seeds both methods ran: a solved / b not vs. b solved / a not."""
    from scipy.stats import binomtest
    common = set(final_a) & set(final_b)
    b = sum(final_a[s] and not final_b[s] for s in common)
    c = sum(final_b[s] and not final_a[s] for s in common)
    return dict(n=len(common), better=b, worse=c, p=round(float(binomtest(b, b + c, 0.5).pvalue), 4) if b + c else None)


# ------------------------------------------------------------------------------------------------ per cell
N_ADAPT = 3  # paper protocol: adaptation episodes per client run of the variant being summarized (set in main)


def adapt_episodes(method):
    try:
        from fawam.methods import resolve
        return int((resolve(method).learn or {}).get("adapt_episodes") or 3)
    except Exception:
        return 3


def summarize(eps, protocol):
    by_seed = defaultdict(dict)
    for ep, r in eps:
        by_seed[r["seed"]][r["attempt"]] = (ep, r)
    s = dict(n_seeds=len(by_seed))
    if protocol == "paper":  # 3 adaptation episodes per client run, then frozen evaluation
        ev = [r for _, r in eps if r["ep_index"] >= N_ADAPT and r["attempt"] == 1]
        ad = [r for _, r in eps if r["ep_index"] < N_ADAPT and r["attempt"] == 1]
        s.update(eval_n=len(ev), eval_ok=sum(r["success"] for r in ev), eval_ci=wilson(sum(r["success"] for r in ev), len(ev)),
                 adapt_n=len(ad), adapt_ok=sum(r["success"] for r in ad),
                 above_eval_mm=med([r.get("above_mm") for r in ev]), above_adapt_mm=med([r.get("above_mm") for r in ad]),
                 gap_eval=med([r.get("gap_grasp") for r in ev]), gap_adapt=med([r.get("gap_grasp") for r in ad]))
        s["final"] = {str(r["seed"]): r["success"] for r in ev}
        s["flung"] = sum(bool(r.get("flung")) for r in ev)
        return s
    first = {sd: v[1][1] for sd, v in by_seed.items() if 1 in v}
    fails = [sd for sd, r in first.items() if not r["success"]]
    retried = [sd for sd in fails if 2 in by_seed[sd]]
    fixed = [sd for sd in retried if by_seed[sd][2][1]["success"]]
    s.update(first_ok=sum(r["success"] for r in first.values()), first_n=len(first), fails=len(fails),
             retried=len(retried), fixed=len(fixed), fix_ci=wilson(len(fixed), len(retried)))
    s["final"] = {str(sd): bool(r["success"] or (sd in fixed)) for sd, r in first.items()}
    s["flung"] = sum(bool(by_seed[sd][2][1].get("flung")) for sd in retried)  # retries the simulator counted as solved
    s["flung_first"] = sum(bool(r.get("flung")) for r in first.values())
    a1 = [by_seed[sd][1][1] for sd in retried]
    a2 = [by_seed[sd][2][1] for sd in retried]
    s["above1_mm"], s["above2_mm"] = med([r.get("above_mm") for r in a1]), med([r.get("above_mm") for r in a2])
    s["lower_mm"] = med([x.get("above_mm") - y.get("above_mm") for x, y in zip(a1, a2)
                         if x.get("above_mm") is not None and y.get("above_mm") is not None])
    s["gap1"], s["gap2"] = med([r.get("gap_grasp") for r in a1]), med([r.get("gap_grasp") for r in a2])
    s["same_failure"] = sum((not y["success"]) and (y.get("oracle") or {}).get("type") == (x.get("oracle") or {}).get("type")
                            for x, y in zip(a1, a2))
    # context usage: attempts that ran with a memory
    mem = [r["memory"] for r in a2 if r.get("memory")]
    if mem:
        early = lambda xs: xs[:4]  # the chunks up to the grasp decide the outcome
        c = dict(n=len(mem), tokens=int(np.median([m["tokens"] for m in mem])),
                 attn_action=mean([np.mean(early(m["attn_action"])) for m in mem]),
                 attn_video=mean([np.mean(early(m["attn_video"])) for m in mem if "attn_video" in m]),
                 infl_action=mean([np.mean(early(m["infl_action"])) for m in mem]),
                 infl_video=mean([np.mean(early(m["infl_video"])) for m in mem]),
                 shift_mm=mean([np.mean(early(m["shift_mm"])) for m in mem if "shift_mm" in m]),
                 layer_action=[round(float(x), 3) for x in np.mean([m["layer_action"] for m in mem], 0)],
                 window_frac=mean([m.get("window_frac") for m in mem]))
        hs = [np.array(m["heat"])[:8] for m in mem if m.get("heat") and len(m["heat"]) >= 8]
        if hs:
            c["heat"] = np.round(np.mean(hs, 0), 2).tolist()  # first 8 chunks x 30 blocks
        if any("layer_video" in m for m in mem):
            c["layer_video"] = [round(float(x), 3) for x in np.mean([m["layer_video"] for m in mem if "layer_video" in m], 0)]
        parts = defaultdict(list)
        for m in mem:
            for k, v in (m.get("parts") or {}).items():
                parts[k].append(v)
        c["parts"] = {k: round(float(np.mean(v)), 3) for k, v in parts.items()}
        ok = [np.mean(early(m["infl_action"])) for r, m in ((r, r["memory"]) for r in a2 if r.get("memory")) if r["success"]]
        ko = [np.mean(early(m["infl_action"])) for r, m in ((r, r["memory"]) for r in a2 if r.get("memory")) if not r["success"]]
        c["infl_action_success"], c["infl_action_fail"] = mean(ok), mean(ko)
        s["context"] = c
    # the analysis the retry was based on, against the oracle of the failed first attempt
    an = [(by_seed[sd][2][1].get("analysis"), by_seed[sd][1][1].get("oracle")) for sd in retried]
    an = [(a, o) for a, o in an if a]
    if an:
        both = [(a, o) for a, o in an if o]
        s["analysis"] = dict(
            source=an[0][0].get("signals"), n=len(an), types=dict(Counter(a.get("type") for a, _ in an)),
            dz_cm=med([a.get("dz_cm") for a, _ in an]), oracle_dz_cm=med([o.get("dz_cm") for _, o in both]),
            type_agree=mean([a.get("type") == o.get("type") for a, o in both]),
            sign_agree=mean([np.sign(a.get("dz_cm") or 0) == np.sign(o.get("dz_cm") or 0) for a, o in both if o.get("dz_cm")]),
            abs_err_cm=med([abs((a.get("dz_cm") or 0) - o.get("dz_cm")) for a, o in both]),
            examples=[a.get("correction") for a, _ in an if a.get("correction")][:3])
    return s


# ------------------------------------------------------------------------------------------------ samples
def ffmpeg_bin():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def make_video(ep, arm, out):
    """Top row: real head | real wrist of the grasping arm. Bottom row: what the model imagined for both. 640x480."""
    if out.exists():
        return True
    wrist = f"cam_{arm or 'left'}_wrist"
    ins = [ep / "real_cam_high.mp4", ep / f"real_{wrist}.mp4", ep / "pred_cam_high.mp4", ep / f"pred_{wrist}.mp4"]
    if not all(p.exists() for p in ins):
        return False
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error"]
    for p in ins:
        cmd += ["-i", str(p)]
    pad = "scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2"
    cmd += ["-filter_complex", f"[0:v]{pad}[a];[1:v]{pad}[b];[2:v]{pad}[c];[3:v]{pad}[d];"
                               "[a][b]hstack=shortest=1[t];[c][d]hstack=shortest=1[u];[t][u]vstack=shortest=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30", "-movflags", "+faststart", "-an", str(out)]
    return subprocess.run(cmd).returncode == 0 and out.exists()


def samples(eps, protocol, media, key):
    if protocol == "paper":
        pool = [(ep, r) for ep, r in eps if r["ep_index"] >= N_ADAPT]
    elif protocol == "single":
        pool = eps
    else:
        pool = [(ep, r) for ep, r in eps if r["attempt"] == 2]
    out = []
    # one real success, one failure, one card that left the table (if any): the artifact holds at most 511 files
    for want in ("ok", "fail", "flung"):
        sel = [x for x in pool if (("flung" if x[1].get("flung") else "ok" if x[1]["success"] else "fail") == want)]
        for ep, r in sel[:1]:
            name = hashlib.sha1(f"v2/{key}/{ep.name}".encode()).hexdigest()[:12] + ".mp4"
            if not make_video(ep, r.get("arm"), media / name):
                continue
            s = dict(video=f"media/{name}", success=r["success"], flung=bool(r.get("flung")), seed=r["seed"], attempt=r["attempt"],
                     above_mm=r.get("above_mm"), gap=r["gap"], grasp_latent=r.get("grasp_latent"),
                     prompt=r.get("prompt"), cause=(r.get("analysis") or {}).get("cause"))
            if r.get("memory"):
                s["attn"], s["infl"] = r["memory"]["attn_action"], r["memory"]["infl_action"]
                s["shift"], s["heat"] = r["memory"].get("shift_mm"), r["memory"].get("heat")
            # spatial attention overlay (analysis/attention_replay.py + attention_render.py), base weights only
            att = ep.parents[3] / "analysis" / "attention" / "videos" / f"{r['task']}__{ep.name}.mp4"
            if att.exists():
                an = name.replace(".mp4", "_att.mp4")
                if not (media / an).exists():
                    (media / an).write_bytes(att.read_bytes())
                s["att_video"] = f"media/{an}"
            s["ep"] = str(ep)
            if r["attempt"] == 2:  # the failed first attempt of the same seed, for comparison
                first = [x for _, x in eps if x["seed"] == r["seed"] and x["attempt"] == 1]
                if first:
                    s["gap_first"], s["above_first_mm"] = first[0]["gap"], first[0].get("above_mm")
            out.append(s)
    return out


# ------------------------------------------------------------------------------------------------ extras
def training_curves():
    out = {}
    for d in sorted((ROOT / "artifacts" / "adapters").glob("failfix_*")):
        log = d / "train_inside.jsonl"
        if not log.exists():
            continue
        rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        val = [r for r in rows if "val_action" in r]
        tr = [r for r in rows if "action" in r and "val_action" not in r]
        out[d.name] = dict(done=(d / "adapter.pt").exists(), last_step=rows[-1]["step"] if rows else 0,
                           steps=[r["step"] for r in val], val=[round(r["val_action"], 4) for r in val],
                           val_noctx=[round(r["val_action_noctx"], 4) for r in val],
                           train_steps=[r["step"] for r in tr[::10]], train=[round(r["action"], 4) for r in tr[::10]])
    return out


def vlm_vs_oracle():
    """Offline: VLM analysis (rollout / single-frame input) of the P0 failures against the simulator oracle."""
    out = {}
    for setting in SETTINGS:
        base = ROOT / "results/sweeps/p0_baseline_table/base" / setting / "episodes/fail"
        for mode, fname in (("rollout", "analysis_gap+vlm.json"), ("frame", "analysis_gap+vlm_frame.json")):
            for scope in ("test", "all"):
                rows = []
                for ep in base.glob(f"{TEST_TASK if scope == 'test' else '*'}/*"):
                    v, o = ep / fname, ep / "analysis.json"
                    if not (v.exists() and o.exists()):
                        continue
                    v, o = json.loads(v.read_text()), json.loads(o.read_text())
                    if "vlm_raw" not in v or o.get("analyzer", "oracle") != "oracle" and "oracle" not in str(o.get("signals")):
                        continue
                    vdz, odz = 100 * (v.get("vlm_delta_m") or [0, 0, 0])[2], 100 * (o.get("delta_m") or [0, 0, 0])[2]
                    rows.append(dict(type=v.get("vlm_raw", {}).get("failure_type") == o.get("failure_type"),
                                     sign=(np.sign(vdz) == np.sign(odz)) if odz else None, err=abs(vdz - odz),
                                     vdz=vdz, odz=odz, text=v.get("correction")))
                if rows:
                    out[f"{setting}|{mode}|{scope}"] = dict(
                        n=len(rows), type_agree=mean([r["type"] for r in rows]),
                        sign_agree=mean([r["sign"] for r in rows if r["sign"] is not None]),
                        abs_err_cm=med([r["err"] for r in rows]), vlm_dz_cm=med([r["vdz"] for r in rows]),
                        oracle_dz_cm=med([r["odz"] for r in rows]), examples=[r["text"] for r in rows if r["text"]][:3])
    return out


def cluster_status():
    """Queue snapshot for the page's "Stand" section: fawam jobs per sweep and state, last finished job, next start."""
    import getpass

    def run(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.splitlines()
        except Exception:
            return []
    user, jobs, nxt = getpass.getuser(), defaultdict(Counter), None
    for l in run("squeue", "-u", user, "-h", "-o", "%j|%T|%S"):
        name, state, start = (l.split("|") + ["", ""])[:3]
        if not name.startswith("fawam_"):
            continue
        jobs[name[6:]][state] += 1
        if state == "PENDING" and re.match(r"\d{4}-", start) and "cron" not in name:
            nxt = min(nxt or start, start)
    done, last = defaultdict(Counter), None
    for l in run("sacct", "-u", user, "-X", "-n", "-P", "-S", "now-2days", "-o", "jobname,state,end"):
        name, state, end = (l.split("|") + ["", ""])[:3]
        state = state.split()[0] if state else ""
        if not name.startswith("fawam_") or "cron" in name or state not in ("COMPLETED", "FAILED", "TIMEOUT"):
            continue
        done[name[6:]][state] += 1
        last = max(last or end, end)
    cron = {}
    for key, f in (("pipeline", "cron.log"), ("report", "report.log")):
        p = ROOT / "pipeline" / f
        if p.exists():
            cron[key] = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
    stages = {}
    if (ROOT / "pipeline" / "stages.json").exists():
        stages = {k: v.get("at") for k, v in json.loads((ROOT / "pipeline" / "stages.json").read_text()).items()}
    gates = sorted(p.name for p in (ROOT / "pipeline" / "gates").glob("*")) if (ROOT / "pipeline" / "gates").exists() else []
    return dict(jobs={k: dict(v) for k, v in jobs.items()}, done_48h={k: dict(v) for k, v in done.items()},
                last_end=last and last.replace("T", " ")[:16], next_start=nxt and nxt.replace("T", " ")[:16],
                cron=cron, stages=stages, gates=gates)


def journal_videos(journal, media):
    """Analyses in analysis/journal.json may list example episodes: "v": [{"ep": <episode dir>, "cap": ...}].
    Each becomes media/j_<sweep>_<method>_<setting>_<seed>.mp4 (same 2x2 layout as the samples), "src" is filled in."""
    names = set()
    for a in (journal or {}).get("analyses", []):
        for v in a.get("v", []):
            if v.get("file"):  # a prebuilt video (e.g. analysis/p15_ownpred/*.mp4), copied as is
                src = ROOT / v["file"]
                name = "j_" + "_".join(Path(v["file"]).with_suffix("").parts[-2:]) + ".mp4"
                if src.exists():
                    if not (media / name).exists() or (media / name).stat().st_mtime < src.stat().st_mtime:
                        (media / name).write_bytes(src.read_bytes())
                    v["src"] = f"media/{name}"
                    names.add(name)
                continue
            ep = ROOT / v["ep"]
            parts = ep.relative_to(ROOT / "results/sweeps").parts
            name = "j_" + "_".join([parts[0], parts[1], parts[2].replace("table_h_", ""), ep.name.split("_", 1)[1]]) + ".mp4"
            try:
                arm = episode_record(ep).get("arm")
            except Exception:
                arm = None
            if make_video(ep, arm, media / name):
                v["src"] = f"media/{name}"
                names.add(name)
    return names


def _json(p):
    """problems / analyses / running work (analysis/journal.json, written by hand) and the VLM benchmark summary"""
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _v2():
    """Plan v2 (p20 diagnosis, p21 ICL stage 1, p22 TTT) tables, fawam/gates_v2.summary()."""
    try:
        from fawam import gates_v2
        return json.loads(json.dumps(gates_v2.summary(), default=float))
    except Exception as e:
        print(f"v2 summary failed: {e!r}")
        return dict(error=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    variants = {}
    for sweep in SWEEPS:
        root = ROOT / "results" / "sweeps" / sweep
        if not root.exists():
            continue
        for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
            method = mdir.name
            global N_ADAPT
            N_ADAPT = adapt_episodes(method)
            protocol = "single" if sweep == "p0_baseline_table" else "paper" if sweep.startswith("p5f") else "retry"
            cells = {}
            for setting in (["table_h_z00"] if protocol == "single" else []) + SETTINGS:
                eps = cell_episodes(sweep, method, setting)
                if not eps:
                    continue
                if protocol == "single":
                    ok = sum(r["success"] for _, r in eps)
                    c = dict(n_seeds=len(eps), first_ok=ok, first_n=len(eps), first_ci=wilson(ok, len(eps)),
                             above1_mm=med([r.get("above_mm") for _, r in eps]), gap1=med([r.get("gap_grasp") for _, r in eps]),
                             final={str(r["seed"]): r["success"] for _, r in eps},
                             flung=sum(bool(r.get("flung")) for _, r in eps))
                else:
                    c = summarize(eps, protocol)
                c["samples"] = samples(eps, protocol, media, f"{sweep}/{method}/{setting}")
                cells[setting] = c
            if cells:
                variants[f"{sweep}/{method}"] = dict(sweep=sweep, method=method, protocol=protocol, settings=cells)
    # paired comparison against the control of the same sweep
    for key, v in variants.items():
        ctrl = variants.get(f"{v['sweep']}/{'base' if v['protocol'] == 'paper' else 'retry_plain'}")
        for setting, c in v["settings"].items():
            if ctrl and ctrl is not v and setting in ctrl["settings"]:
                fa_, fb_ = ({int(k): x for k, x in d["final"].items()} for d in (c, ctrl["settings"][setting]))
                c["vs_control"] = paired_p(fa_, fb_)
    for v in variants.values():
        for c in v["settings"].values():
            c.pop("final", None)
    allsamples = [(v, s) for v in variants.values() for c in v["settings"].values() for s in c.get("samples", [])]
    used = {s[k].split("/")[-1] for _, s in allsamples for k in ("video", "att_video") if s.get(k)}
    # episodes for the spatial attention replay: only variants that run the unchanged weights without a memory
    todo = sorted({s["ep"] for v, s in allsamples if v["method"] in ("base", "retry_plain") and not s.get("att_video")})
    (out / "attention_todo.txt").write_text("\n".join(todo) + ("\n" if todo else ""))
    for _, s in allsamples:
        s.pop("ep", None)
    journal = _json(ROOT / "analysis" / "journal.json")
    used |= journal_videos(journal, media)
    for f in media.glob("*.mp4"):
        if f.name not in used:
            f.unlink()
    data = dict(generated=time.strftime("%Y-%m-%d %H:%M"), test_task=TEST_TASK, variants=variants,
                training=training_curves(), vlm_vs_oracle=vlm_vs_oracle(), status=cluster_status(),
                journal=journal, vlm_bench=_json(ROOT / "results/vlm_bench/summary.json"), v2=_v2())
    (out / "report.json").write_text(json.dumps(data, indent=1))
    (out / "data.js").write_text("window.REPORT = " + json.dumps(data) + ";\n")
    page = ROOT / "analysis" / "report_page.html"
    if page.exists():
        (out / "index.html").write_text(page.read_text())
    print(f"{len(variants)} variants, {len(used)} videos, {time.time() - t0:.0f} s -> {out}")
    try:  # markdown copy for the repo (docs/RESULTS.md, docs/results_summary.json)
        from fawam import report_md
        report_md.main(out / "report.json")
    except Exception as e:
        print(f"report_md failed: {e!r}")


if __name__ == "__main__":
    main()
