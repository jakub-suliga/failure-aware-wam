"""Stages of plan v2 (docs/PLAN_FAWAM_V2.md, 2026-11-21) for cluster/pipeline.py: p20 diagnosis, p21 ICL stage 1,
p22 TTT. Everything was queued on 2026-11-21; the chains below fire automatically (no manual decisions):

  D2 + D4 idle            -> setting gate (results/gates/p21_setting_{sa,sb}.json, inline) + D6 phases (cpu job)
  D6 done                 -> D7 VLM phase labelling (Qwen3-VL-32B, gpu job)
  p21 data S idle + gate  -> trainings of setting S (prio: core A1 x C0-C3, then A2 + chunk ablation, then C4-C8);
                             a setting that failed its gate gets no trainings (artifacts/adapters/<n>/skipped.json)
  p21 data S-B idle       -> D5 leak probe (cpu job)
  training finished       -> negative filter (inline, results/gates/p21_filter_<n>.json) -> sim screening p21_screen
                             (C0 always); G2 offline is not run (info only, see journal)
  p21_screen idle         -> screening gate -> full evaluation (54 seeds = 27 per sign, + C0 + foreign context)
  p22_ttt idle            -> screening gate -> full evaluation (27 seeds, + ttt_retry)
  D1 gate                 -> P7 (not implemented in this submission: recorded as open / skipped)
Priorities in the feeder queue: lower = earlier (data -3, D1 -4, Prio-1 items -2, Prio-2 -1, Prio-3 0, Prio-4 1).
"""
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ADAPTERS = ROOT / "artifacts" / "adapters"
GATES = ROOT / "results" / "gates"

# (name, setting, arch, ctx, sampling, prio)
TRAIN = ([(f"p21_{s}_A1_{c}", s, "A1", c, "uniform", -1) for s in ("sa", "sb") for c in ("C0", "C1", "C2", "C3")] +
         [(f"p21_{s}_A2_C2", s, "A2", "C2", "uniform", 0) for s in ("sa", "sb")] +
         [("p21_sb_A1_C2_win50", "sb", "A1", "C2", "win50", 0), ("p21_sb_A1_C2_w4", "sb", "A1", "C2", "w4", 0)] +
         [(f"p21_{s}_A3_C2", s, "A3", "C2", "uniform", 0) for s in ("sa", "sb")] +
         [(f"p21_sb_A1_{c}", "sb", "A1", c, "uniform", 1) for c in ("C4", "C5", "C6", "C7", "C8")] +
         [(f"p21_sb_{t}_C2", "sb", t, "C2", "uniform", 1) for t in ("T1", "T2", "T4", "T5")])
NOT_IMPLEMENTED = ["p21_{s}_A4_C2 (VACE context blocks)", "p21_sb_T3 (gate from an auxiliary error head)",
                   "p21_sb_T6 (phase-aligned context)", "p22 P7 (forecast mode + selection)", "G2 offline"]


def _gate(name):
    p = GATES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _method(name):
    return "icl2_" + name[len("p21_"):]


def stages(jobs):
    import pipeline as P
    idle = lambda s: P.idle(jobs, s) and not any(x.get("label", "").startswith(s) for x in P.read_queue())
    out = [
        # manual stages of 2026-11-21 (kept here so the pipeline re-queues unfinished cells once a sweep is idle)
        ("v2_data", lambda: False, lambda: P.sweep("p21_data_sa", prio=-3) + P.sweep("p21_data_sb", prio=-3)),
        ("v2_bank", lambda: False, lambda: P.sweep("p21_bank", prio=-2)),
        ("v2_d3", lambda: False, lambda: P.sweep("p20_d3_snap", prio=-2)),
        ("v2_d2d4", lambda: False, lambda: P.sweep("p20_d2_base", prio=-2) + P.sweep("p20_d4_tf", prio=-2)),
        ("v2_ttt", lambda: False, lambda: P.sweep("p22_ttt", prio=-1)),
        # setting gate + D6 once D2 and D4 are through
        ("v2_gate_setting", lambda: idle("p20_d2_base") and idle("p20_d4_tf"), _setting_gate),
        ("v2_d6", lambda: idle("p20_d2_base") and idle("p20_d4_tf") and idle("p20_d3_snap"),
         lambda: P.cpu_job("p20_d6", "$PY_RT -m fawam.phases results/sweeps/p20_d2_base results/sweeps/p20_d4_tf "
                                     "results/sweeps/p20_d3_snap --out results/p20_d6_phases.json", prio=-2)),
        ("v2_d7", lambda: P._stage_done("v2_d6") and P.sacct_state("fawam_p20_d6") == "COMPLETED",
         lambda: P.gpu_job("p20_d7", dict(CMD="", VLM_MODEL="Qwen/Qwen3-VL-32B-Instruct", HF_HUB_OFFLINE="1"),
                           "vlm_phase_job.sh", prio=-2, time="03:00:00")),
        ("v2_d5", lambda: idle("p21_data_sb") and idle("p20_d2_base"),
         lambda: P.cpu_job("p20_d5", "$PY_RT -m fawam.hid_leak results/sweeps/p21_data_sb/base/icl_sb "
                                     "results/sweeps/p21_data_sb/base/icl_sb_v1 && cp results/sweeps/p21_data_sb/base/"
                                     "icl_sb/leak_test.json results/gates/p20_d5_leak.json || true", prio=-2)),
        ("v2_d1_gate", lambda: _gate("p20_d1") is not None, _p7),
        ("v2_ttt_full", lambda: P._stage_done("v2_ttt") and idle("p22_ttt"), _ttt_full),
        ("v2_icl_full", lambda: all(P._stage_done(f"v2_screen_{n}") or P._stage_done(f"v2_train_{n}_skipped")
                                    for n, *_ in TRAIN if n.endswith("C0")) and
         any(P._stage_done(f"v2_screen_{n}") for n, *_ in TRAIN) and idle("p21_screen") and
         all(not P.active(jobs, f"fawam_p21train_{n}") for n, *_ in TRAIN), _icl_full),
    ]
    for name, S, arch, ctx, samp, prio in TRAIN:
        data_ready = lambda S=S: P._stage_done("v2_gate_setting") and idle(f"p21_data_{S}")
        out.append((f"v2_train_{name}", data_ready,
                    lambda name=name, S=S, arch=arch, ctx=ctx, samp=samp, prio=prio: _train(name, S, arch, ctx, samp,
                                                                                            prio)))
        out.append((f"v2_screen_{name}",
                    lambda name=name: P._stage_done(f"v2_train_{name}") and
                    not P.active(jobs, f"fawam_p21train_{name}") and (ADAPTERS / name / "adapter.pt").exists() and
                    (P.sacct_state(f"fawam_p21train_{name}") in ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED")),
                    lambda name=name: _screen(name)))
    return out


def _log_skip(name, why):
    d = ADAPTERS / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "skipped.json").write_text(json.dumps(dict(skipped=why, at=f"{datetime.datetime.now():%Y-%m-%d %H:%M}")))


def _setting_gate():
    from fawam import gates_v2
    r = gates_v2.setting()
    import pipeline as P
    P.log(f"v2 setting gate: " + ", ".join(f"{k} {'passed' if v['passed'] else 'NOT passed'} "
                                          f"(fail {v['base_fail_rate']:.0%}, tf {v['tf_replan_rate']:.0%})"
                                          for k, v in r.items()))
    return 0


def _train(name, S, arch, ctx, samp, prio):
    import pipeline as P
    g = _gate(f"p21_setting_{S}")
    if g is not None and not g.get("passed"):
        _log_skip(name, f"gate p21_setting_{S}")
        return 0
    argv = ["sbatch", f"--job-name=fawam_p21train_{name}", f"--partition={P.PARTITION}", "--gres=gpu:1",
            "--time=04:00:00", f"--output={ROOT}/logs/p21train_{name}_%j.out", f"--chdir={ROOT}",
            "cluster/p21_train_job.sh"]
    env = dict(FAWAM_ROOT=str(ROOT), NAME=name, SETTING=S, CTX=ctx, ARCH=arch, SAMPLING=samp, STEPS="1500")
    return P.enqueue([dict(argv=argv, env=env, label=f"p21train {name}")], prio)


def _screen(name):
    import pipeline as P
    from fawam import gates_v2
    g = gates_v2.filter_gate(name)
    if not g.get("passed"):
        P.log(f"v2 negative filter: {name} NOT passed -> no sim screening ({g})")
        return 0
    S = name.split("_")[1]
    P.log(f"v2 negative filter: {name} passed -> screening")
    return P.sweep("p21_screen", "--methods", _method(name), "--configs", f"icl_{S}", prio=-1)


def _icl_full():
    import pipeline as P
    from fawam import gates_v2
    r = gates_v2.screen_icl()
    n = 0
    for m, g in r.items():
        if g.get("passed"):
            S = m.split("_")[1]
            n += P.sweep("p21_screen", "--methods", m, f"{m}_foreign", f"icl2_{S}_A1_C0", "--configs", f"icl_{S}",
                         "--episodes", "54", prio=-1)
    P.log(f"v2 ICL screening gate: {sum(g['passed'] for g in r.values())} of {len(r)} variants -> full evaluation")
    return n


def _ttt_full():
    import pipeline as P
    from fawam import gates_v2
    r = gates_v2.screen_ttt()
    n = 0
    for key, g in r.items():
        if g.get("passed"):
            m, cfg = key.split("/")
            n += P.sweep("p22_ttt", "--methods", m, "ttt_retry", "--configs", cfg, "--episodes", "27", prio=-1)
    P.log(f"v2 TTT screening gate: {sum(g['passed'] for g in r.values())} of {len(r)} -> full evaluation")
    return n


def _p7():
    import pipeline as P
    g = _gate("p20_d1")
    note = ("D1: conditioned -> P7 would run, but P7 is not implemented in this submission (open)" if g.get("passed")
            else "D1: not conditioned -> P7 skipped (gate p20_d1)")
    (GATES / "p22_p7.json").write_text(json.dumps(dict(passed=False, note=note, d1=g), indent=1))
    P.log(note)
    return 0
