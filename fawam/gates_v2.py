"""Automatic gates of plan v2 (2026-11-21). Every gate writes results/gates/<id>.json = {"passed": bool, ...}; dependent
jobs read it at start (cluster/p21_train_job.sh) or the pipeline (cluster/pipeline_v2.py) only submits what passed and
records the rest as "skipped: gate <id>".

  python -m fawam.gates_v2 setting          # after D2 + D4: which ICL settings (S-A, S-B) go on
  python -m fawam.gates_v2 filter <adapter>  # p21 negative filter from the training's validation losses
  python -m fawam.gates_v2 screen_icl        # p21 screening -> full evaluation (variant - C0 >= 25 pp)
  python -m fawam.gates_v2 screen_ttt        # p22 screening -> full evaluation (variant - retry >= 25 pp)
  python -m fawam.gates_v2 summary           # tables for the report (results/v2_summary.json)

Success is the corrected one (a flung object is a failure). Statistics: Fisher exact test on the same seeds.
"""
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "results" / "gates"
SW = ROOT / "results" / "sweeps"
TEST = "move_playingcard_away"
SA_D2, SA_D4 = ["table_h_m05", "table_h_m03", "table_h_p03", "table_h_p05"], ["table_h_m03", "table_h_p03", "table_h_p05"]
SB_D2 = ["hid_y_p3", "hid_y_m3", "hid_y_p5", "hid_y_m5", "hid_x_p3", "hid_x_m3", "hid_x_p5", "hid_x_m5"]
SB_D4 = ["hid_y_p5", "hid_y_m5", "hid_x_p5", "hid_x_m5"]


def write_gate(name, d):
    GATES.mkdir(parents=True, exist_ok=True)
    (GATES / f"{name}.json").write_text(json.dumps(d, indent=1, default=float))
    return d


def ep_success(ep):
    """Corrected success: simulator success and the object did not fly away (> 5 cm height change or > 0.6 m)."""
    with h5py.File(Path(ep) / "episode.h5", "r") as f:
        ok = bool(f.attrs["success"])
        op = f["sim/object_pose"][()]
        if ok and op.shape[1] > 0:
            a, b = op[0].reshape(-1)[:3], op[-1].reshape(-1)[:3]
            if abs(float(b[2] - a[2])) > 0.05 or float(np.linalg.norm(b[:2] - a[:2])) > 0.6:
                ok = False
    return ok


def _name(ep):
    m = re.match(r"ep(\d+)_seed(\d+)(?:_attempt(\d+))?", ep.name)
    return int(m.group(2)), int(m.group(3) or 1)


def cell(path, task=TEST):
    """{seed: {attempt: success}} of a sweep cell dir (…/<method>/<config>); snap continuations are skipped."""
    out = {}
    for kind in ("success", "fail"):
        d = Path(path) / "episodes" / kind / task
        if not d.exists():
            continue
        for ep in d.iterdir():
            if "_snap_" in ep.name or not (ep / "episode.h5").exists():
                continue
            try:
                seed, att = _name(ep)
                out.setdefault(seed, {})[att] = ep_success(ep)
            except Exception:
                continue
    return out


def first_attempt(c):
    return {s: a[1] for s, a in c.items() if 1 in a}


def fisher(k1, n1, k2, n2):
    from scipy.stats import fisher_exact
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]) if n1 and n2 else float("nan")


# ------------------------------------------------------------------------------------------------ setting gate
def setting():
    res = {}
    for S, d2c, d4c in (("sa", SA_D2, SA_D4), ("sb", SB_D2, SB_D4)):
        fails = n = 0
        per = {}
        for c in d2c:
            fa = first_attempt(cell(SW / "p20_d2_base/base" / c))
            k = sum(not v for v in fa.values())
            per[f"D2 {c}"] = f"{k}/{len(fa)} failed"
            fails, n = fails + k, n + len(fa)
        tf_ok = tf_n = 0
        for c in d4c:
            fa = first_attempt(cell(SW / "p20_d4_tf/tf_replan" / c))
            k = sum(fa.values())
            per[f"D4 {c}"] = f"{k}/{len(fa)} tf_replan ok"
            tf_ok, tf_n = tf_ok + k, tf_n + len(fa)
        if S == "sa":  # table -5 cm with tf_replan is known from p13 (27/27)
            fa = first_attempt(cell(SW / "p13_tf_replan/tf_replan/table_h_m05"))
            per["p13 table_h_m05"] = f"{sum(fa.values())}/{len(fa)} tf_replan ok"
            tf_ok, tf_n = tf_ok + sum(fa.values()), tf_n + len(fa)
        fr, tr = (fails / n if n else 0.0), (tf_ok / tf_n if tf_n else 0.0)
        res[S] = write_gate(f"p21_setting_{S}", dict(passed=bool(fr >= 0.7 and tr >= 0.7), base_fail_rate=fr,
                                                     tf_replan_rate=tr, n_base=n, n_tf=tf_n, cells=per,
                                                     rule="base failures >= 70 % and tf_replan success >= 70 %"))
    # which single settings qualify (for TTT / reporting)
    per_cfg = {}
    for c in SA_D2 + SB_D2 + ["hid_g070", "hid_g085", "hid_g115", "hid_g130"]:
        fa = first_attempt(cell(SW / "p20_d2_base/base" / c))
        tf = first_attempt(cell(SW / "p20_d4_tf/tf_replan" / c))
        per_cfg[c] = dict(base_fail=sum(not v for v in fa.values()), n=len(fa),
                          tf_ok=sum(tf.values()) if tf else None, n_tf=len(tf) if tf else None)
    write_gate("p20_d2d4", dict(passed=True, settings=per_cfg))
    return res


# ------------------------------------------------------------------------------------------------ negative filter
def _last_val(name):
    p = ROOT / "artifacts" / "adapters" / name / "train_inside.jsonl"
    if not p.exists():
        return None
    v = [json.loads(l) for l in p.read_text().splitlines() if l.strip() and "val_video_own" in l]
    return v[-1] if v else None


def filter_gate(name):
    """p21 negative filter: video val loss with the own context < C0 of the same setting; with a foreign context
    (opposite sign) > C0. C0 itself always passes (its sim screening always runs)."""
    m = re.match(r"p21_(s[ab])_(A\d)_(C\d)(.*)", name)
    S, A, C, x = m.groups()
    if C == "C0":
        return write_gate(f"p21_filter_{name}", dict(passed=True, note="C0 is the control"))
    v, c0 = _last_val(name), _last_val(f"p21_{S}_A1_C0")
    if v is None or c0 is None:
        return write_gate(f"p21_filter_{name}", dict(passed=False, reason="missing validation", own=v, c0=c0))
    own, foreign, l0 = v["val_video_own"], v["val_video_foreign"], c0["val_video_own"]
    return write_gate(f"p21_filter_{name}", dict(passed=bool(own < l0 and foreign > l0), own=own, foreign=foreign,
                                                 none=v.get("val_video_none"), c0=l0, step=v.get("step"),
                                                 rule="own < C0 and foreign > C0 (video val loss, held-out seeds)"))


# ------------------------------------------------------------------------------------------------ screening
def retry_rate(c, harm=False):
    """p21: success of the rewind retry on seeds whose first attempt failed (harm=True: on base successes)."""
    sel = [a.get(2) for s, a in c.items() if 1 in a and (a[1] == harm) and 2 in a]
    return sum(bool(x) for x in sel), len(sel)


def screen_icl(sweep="p21_screen"):
    out = {}
    base = SW / sweep
    for cfg in ("icl_sa", "icl_sb"):
        S = cfg[-2:]
        c0 = base / f"icl2_{S}_A1_C0" / cfg
        k0, n0 = retry_rate(cell(c0))
        for d in sorted(base.glob(f"icl2_{S}_*")):
            if d.name.endswith("_foreign") or d.name.endswith("_C0"):
                continue
            k, n = retry_rate(cell(d / cfg))
            diff = (k / n if n else 0) - (k0 / n0 if n0 else 0)
            out[d.name] = write_gate(f"p21_full_{d.name}", dict(passed=bool(n >= 10 and n0 >= 10 and diff >= 0.25),
                                                                 k=k, n=n, k_c0=k0, n_c0=n0, diff_pp=100 * diff,
                                                                 p=fisher(k, n, k0, n0),
                                                                 rule="screening: variant - C0 >= 25 pp"))
    return out


def screen_ttt(sweep="p22_ttt"):
    """p22: success within 3 retries on seeds whose first attempt failed; full evaluation if >= 25 pp over ttt_retry."""
    out = {}
    base = SW / sweep
    for cfg_dir in sorted((base / "ttt_retry").glob("*")):
        cfg = cfg_dir.name
        ctrl = cell(cfg_dir)
        fails = {s for s, a in ctrl.items() if a.get(1) is False}

        def rate(c):
            sel = [s for s, a in c.items() if a.get(1) is False]
            return sum(any(v for k, v in c[s].items() if k > 1) for s in sel), len(sel)
        k0, n0 = rate(ctrl)
        for d in sorted(base.glob("*")):
            if d.name == "ttt_retry" or not (d / cfg).exists():
                continue
            k, n = rate(cell(d / cfg))
            diff = (k / n if n else 0) - (k0 / n0 if n0 else 0)
            out[f"{d.name}/{cfg}"] = write_gate(f"p22_full_{d.name}_{cfg}",
                                                dict(passed=bool(n >= 8 and diff >= 0.25), k=k, n=n, k_retry=k0,
                                                     n_retry=n0, diff_pp=100 * diff, p=fisher(k, n, k0, n0),
                                                     rule="screening: variant - retry without learning >= 25 pp"))
    return out


def _j(p):
    p = Path(p)
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:
        return None


def _jobs():
    import subprocess
    r = subprocess.run(["squeue", "--me", "-h", "-o", "%j|%T"], capture_output=True, text=True)
    out = {}
    for l in r.stdout.splitlines():
        n, st = l.split("|")
        out.setdefault(n, []).append(st)
    q = ROOT / "pipeline" / "queue.jsonl"
    queued = [json.loads(l).get("label", "") for l in q.read_text().splitlines() if l.strip()] if q.exists() else []
    return out, queued


def d3_table():
    out = {}
    for info_p in sorted((SW / "p20_d3_snap/base").glob("*/episodes/*/*/*/snap_info.json")):
        cfg = info_p.parts[-6]
        info = json.loads(info_p.read_text())
        if info.get("parent_success"):
            continue
        row = out.setdefault(cfg, dict(failed=0, no_close=0))
        row["failed"] += 1
        if info.get("close_chunk") is None:
            row["no_close"] += 1
        for c in info.get("conts", []):
            for mode in ("expert", "tf"):
                if mode in c and "success" in c[mode]:
                    key = f"{mode}_k{c['k']}"
                    ok = ep_success(c[mode]["dir"]) if Path(c[mode]["dir"], "episode.h5").exists() else False
                    k, n = row.get(key, (0, 0))
                    row[key] = (k + int(ok), n + 1)
    return out


def p21_data():
    from fawam import icl_snap
    res = {}
    for S in ("sa", "sb"):
        roots = [SW / f"p21_data_{S}/base/icl_{S}", SW / f"p21_data_{S}/base/icl_{S}_v1"]
        att = fails = 0
        conts = {}
        for r in roots:
            for info_p in r.glob("episodes/*/*/*/snap_info.json"):
                info = json.loads(info_p.read_text())
                att += 1
                fails += not info.get("parent_success")
                for c in info.get("conts", []):
                    e = c.get("expert") or {}
                    k, n = conts.get(f"k{c['k']}", (0, 0))
                    conts[f"k{c['k']}"] = (k + int(bool(e.get("success"))), n + int("success" in e))
        pairs = icl_snap.collect_pairs([str(r) for r in roots])
        res[S] = dict(attempts=att, failed=fails, expert_by_k=conts, pairs=len(pairs),
                      pairs_fail=sum(not p["parent_success"] for p in pairs),
                      by_sign={str(sg): sum(p["sign"] == sg for p in pairs) for sg in (-1, 1)},
                      by_task={t: sum(p["task"] == t for p in pairs) for t in sorted({p["task"] for p in pairs})})
    return res


def p21_table(jobs, queued):
    import importlib.util
    spec = importlib.util.spec_from_file_location("pipeline_v2", ROOT / "cluster" / "pipeline_v2.py")
    pv2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pv2)
    rows = []
    for name, S, arch, ctx, samp, prio in pv2.TRAIN:
        d = ROOT / "artifacts" / "adapters" / name
        jn = f"fawam_p21train_{name}"
        st = "skipped: " + _j(d / "skipped.json")["skipped"] if (d / "skipped.json").exists() else \
            ("läuft" if "RUNNING" in jobs.get(jn, []) else "wartet (SLURM)" if jn in jobs else
             "in der Queue" if f"p21train {name}" in queued else "fertig" if (d / "adapter.pt").exists() and
             not (d / "resume.json").exists() or (_j(d / "resume.json") or {}).get("step") == 1500 else
             "teilweise" if (d / "adapter.pt").exists() else "wartet auf Daten/Gate")
        v = _last_val(name)
        m = "icl2_" + name[len("p21_"):]
        cfg = f"icl_{S}"
        sc = cell(SW / "p21_screen" / m / cfg) if (SW / "p21_screen" / m / cfg).exists() else {}
        fo = cell(SW / "p21_screen" / f"{m}_foreign" / cfg) if (SW / "p21_screen" / f"{m}_foreign" / cfg).exists() else {}
        rows.append(dict(name=name, setting=S, arch=arch, ctx=ctx, sampling=samp, status=st,
                         val=v and {k: v.get(k) for k in ("step", "val_video_own", "val_video_foreign", "val_video_none")},
                         filter=_j(GATES / f"p21_filter_{name}.json"), retry=retry_rate(sc), harm=retry_rate(sc, True),
                         foreign=retry_rate(fo), full=_j(GATES / f"p21_full_{m}.json")))
    return rows


def p22_table():
    rows = []
    base = SW / "p22_ttt"
    for d in sorted(base.glob("*")) if base.exists() else []:
        for cd in sorted(d.glob("*")):
            c = cell(cd)
            first_fail = [s for s, a in c.items() if a.get(1) is False]
            by_r = [sum(any(c[s].get(k) for k in range(2, r + 2)) for s in first_fail) for r in (1, 2, 3)]
            secs = []
            for lg in cd.glob("lff/*/learner.log"):
                secs += [json.loads(l)["seconds"] for l in lg.read_text().splitlines() if l.startswith('{"adapter"')]
            rows.append(dict(method=d.name, config=cd.name, seeds=len(c), first_ok=sum(bool(a.get(1)) for a in c.values()),
                             first_fail=len(first_fail), retry_ok=by_r, learn_s=float(np.median(secs)) if secs else None,
                             full=_j(GATES / f"p22_full_{d.name}_{cd.name}.json")))
    return rows


def summary():
    jobs, queued = _jobs()
    out = dict(
        d1=_j(ROOT / "results/p20_d1/summary.json"),
        d2d4=(_j(GATES / "p20_d2d4.json") or {}).get("settings") or {
            c: dict(base_fail=sum(not v for v in first_attempt(cell(SW / "p20_d2_base/base" / c)).values()),
                    n=len(first_attempt(cell(SW / "p20_d2_base/base" / c))),
                    tf_ok=sum(first_attempt(cell(SW / "p20_d4_tf/tf_replan" / c)).values()),
                    n_tf=len(first_attempt(cell(SW / "p20_d4_tf/tf_replan" / c))))
            for c in SA_D2 + SB_D2 + ["hid_g070", "hid_g085", "hid_g115", "hid_g130"]},
        d3=d3_table(), d5=_j(GATES / "p20_d5_leak.json"), d6=_j(ROOT / "results/p20_d6_phases.json"),
        d7=_j(ROOT / "results/p20_d7/summary.json"),
        settings={S: _j(GATES / f"p21_setting_{S}.json") for S in ("sa", "sb")},
        data=p21_data(), p21=p21_table(jobs, queued), p22=p22_table(),
        p7=_j(GATES / "p22_p7.json"),
        jobs={k: dict(running=v.count("RUNNING"), pending=v.count("PENDING")) for k, v in jobs.items()
              if k.startswith(("fawam_p20", "fawam_p21", "fawam_p22"))},
        queued={k: sum(q.startswith(k) for q in queued) for k in ("p20", "p21", "p22")})
    (ROOT / "results" / "v2_summary.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def main():
    cmd = sys.argv[1]
    if cmd == "summary":
        print(json.dumps(summary(), indent=1, default=float)[:3000])
        return
    if cmd == "setting":
        print(json.dumps(setting(), indent=1))
    elif cmd == "filter":
        print(json.dumps(filter_gate(sys.argv[2]), indent=1))
    elif cmd == "screen_icl":
        print(json.dumps(screen_icl(), indent=1))
    elif cmd == "screen_ttt":
        print(json.dumps(screen_ttt(), indent=1))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
