#!/usr/bin/env python3
"""Unattended experiment pipeline for the Learn-from-Failure / ICL study (runs from scrontab, see cluster/pipeline_cron.sh).

Every tick:
  1. stages whose prerequisites are met put their jobs into pipeline/queue.jsonl (each stage runs once; a sweep is
     re-emitted once after it finished, which re-submits only unfinished cells),
  2. the feeder submits queued jobs by priority while both project accounts stay under the per-account submit limit
     (100 jobs incl. pending), picking the account with more free slots,
  3. pipeline/STATUS.md is rewritten.

  python cluster/pipeline.py tick        # what the cron job runs
  python cluster/pipeline.py status      # print STATUS.md
  touch pipeline/gates/<name>            # open a manual gate (e.g. film_fixed)
"""
import argparse
import datetime
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "pipeline"
QUEUE = STATE / "queue.jsonl"
DONE = STATE / "stages.json"
GATES = STATE / "gates"
LOG = STATE / "pipeline.log"
ACCOUNTS = ["hk-project-p0029595"]  # p0024638 retired 2026-11-17 (user)
CAP = 97  # per-account submit limit is 100 (pending + running); keep room for the cron job and manual jobs
PARTITION = "gpu-h100,gpu-h200"
P0 = ROOT / "results/sweeps/p0_baseline_table/base"
PY = os.environ.get("PY_RT", sys.executable)
VLM_ENV = {"VLM_MODEL": "Qwen/Qwen3-VL-8B-Instruct", "HF_HUB_OFFLINE": "1"}

# meta-training waves with the oracle analysis (cluster/adapter_grid.py names)
META_MAIN = ["failfix_predobs", "failfix_obs", "failfix_diff", "failfix_window", "democtx_meta"]
META_ABL = ["failfix_predobs_ownonly", "failfix_predobs_notext", "failfix_predobs_small", "failfix_predobs_holdtask"]


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def clean_env(extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))}
    env.update(extra or {})
    return env


def sh(cmd, env=None):
    return subprocess.run(cmd, capture_output=True, text=True, env=clean_env(env), cwd=ROOT)


# ------------------------------------------------------------------------------------------------ slurm state
def squeue():
    r = sh(["squeue", "--me", "-h", "-o", "%i|%a|%j|%T"])
    return [dict(zip(("id", "account", "name", "state"), l.split("|"))) for l in r.stdout.splitlines() if l]


def sacct_state(name):
    """State of the newest job with this name (None if never run)."""
    r = sh(["sacct", "-u", os.environ.get("USER", ""), "-X", "-n", "-P", "-S", "2026-11-01", "--name", name, "-o", "JobID,State"])
    rows = [l.split("|") for l in r.stdout.splitlines() if l]
    return rows[-1][1].split()[0] if rows else None


def read_queue():
    return [json.loads(l) for l in QUEUE.read_text().splitlines() if l.strip()] if QUEUE.exists() else []


def write_queue(items):
    tmp = QUEUE.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(x) + "\n" for x in items))
    tmp.replace(QUEUE)


def active(jobs, prefix):
    """Jobs (queued in SLURM or still in our queue) whose name / label starts with prefix."""
    n = sum(j["name"].startswith(prefix) for j in jobs)
    return n + sum(x.get("label", "").startswith(prefix.replace("fawam_", "")) for x in read_queue())


# ------------------------------------------------------------------------------------------------ enqueue helpers
def enqueue(items, prio):
    q = read_queue()
    seq = max([x.get("seq", 0) for x in q] + [0])
    for i, x in enumerate(items):
        x.update(prio=prio, seq=seq + i + 1)
    write_queue(q + items)
    return len(items)


def emit(cmd, env=None):
    """Run submit_sweep.py / adapter_grid.py in queue mode and return the emitted job specs."""
    with tempfile.NamedTemporaryFile("r", suffix=".jsonl", dir=STATE, delete=False) as f:
        path = f.name
    r = sh(cmd, {"FAWAM_QUEUE": path, "PARTITION": PARTITION, **(env or {})})
    if r.returncode:
        log(f"emit failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    items = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    os.unlink(path)
    return items


def sweep(yaml_name, *args, prio=2, env=None):
    items = emit([PY, "cluster/submit_sweep.py", f"configs/sweeps/{yaml_name}.yaml", *args, "--submit"], env)
    for x in items:
        x["env"] = {**(env or {}), **x.get("env", {})}
    return enqueue(items, prio)


def train(names, prio=2, env=None):
    items = []
    for n in names:
        items += emit([PY, "cluster/adapter_grid.py", "train", "--only", n, "--submit"], env)
    seen, keep = {x.get("label") for x in read_queue()}, []
    for x in items:  # --only is a prefix match (d3_..._n3 also emits _n30): keep each requested adapter once
        if x["label"][len("train_"):] in names and x["label"] not in seen:
            seen.add(x["label"])
            keep.append(x)
    items = keep
    return enqueue(items, prio)


def cpu_job(name, script, prio=0):
    argv = ["sbatch", f"--job-name=fawam_{name}", "--partition=cpu", "--time=03:00:00", "--cpus-per-task=8",
            "--mem=60G", f"--output={ROOT}/logs/{name}_%j.out", f"--chdir={ROOT}",
            f"--wrap=source cluster/env.sh && {script}"]
    return enqueue([dict(argv=argv, env={}, label=name)], prio)


def gpu_job(name, script_env, script, prio=1, time="12:00:00"):
    argv = ["sbatch", f"--job-name=fawam_{name}", f"--partition={PARTITION}", "--gres=gpu:1", f"--time={time}",
            f"--output={ROOT}/logs/{name}_%j.out", f"--chdir={ROOT}", f"cluster/{script}"]
    return enqueue([dict(argv=argv, env={"FAWAM_ROOT": str(ROOT), **script_env}, label=name)], prio)


# ------------------------------------------------------------------------------------------------ gates
def gate(name):
    return (GATES / name).exists()


def smoke_ok(name):
    """Smoke test passed: SLURM state and the expected artifact. Can be forced with pipeline/gates/smoke_<name>."""
    if gate(f"smoke_{name}"):
        return True
    if gate(f"smoke_{name}_failed") or sacct_state(f"fawam_smoke_{name}") != "COMPLETED":
        return False
    s = ROOT / "results/_smoke" / name
    ok = {
        "lff": lambda: any(s.glob("episodes/*/*/*_attempt2/episode.h5")),
        "icl": lambda: any(s.glob("episodes/*/*/*_attempt2/episode.h5")),
        "paper": lambda: any('"eval"' in p.read_text() for p in s.glob("episodes/*/*/*/meta.json")),
        "pair": lambda: any(s.glob("episodes/*/*/*")),
        "vlm": lambda: any((json.loads(m.read_text()).get("lff") or {}).get("analysis", {}).get("cause")
                           for m in s.glob("episodes/*/*/*_attempt2/meta.json")),  # the VLM really answered
    }[name]()
    if ok:
        (GATES / f"smoke_{name}").touch()
    return ok


def prep_ok():
    probe = ROOT / "artifacts/probe/pose_probe.npz"
    bak = probe.with_suffix(".npz.bak")
    return any(P0.glob("*/episodes/fail/*/*/analysis.json")) and bak.exists() and \
        probe.stat().st_mtime > bak.stat().st_mtime


def idle(jobs, sweep_name):
    return active(jobs, f"fawam_{sweep_name}") == 0


def adapters_exist(names):
    return [n for n in names if (ROOT / "artifacts/adapters" / n / "adapter.pt").exists()]


def top_types(k=5):
    """Meta-trained adapter types ranked by how much the memory lowers the held-out action loss."""
    scores = {}
    for d in (ROOT / "artifacts/adapters").glob("failfix_type_*"):
        val = [json.loads(l) for l in (d / "train_inside.jsonl").read_text().splitlines()
               if l.strip() and "val_action" in l] if (d / "train_inside.jsonl").exists() else []
        if val and (d / "adapter.pt").exists():
            v = val[-1]
            scores[d.name[len("failfix_type_"):]] = v["val_action_noctx"] - v["val_action"]
    ranked = sorted(scores, key=scores.get, reverse=True)
    (STATE / "type_ranking.json").write_text(json.dumps({t: scores[t] for t in ranked}, indent=1))
    return ranked[:k]


# ------------------------------------------------------------------------------------------------ stages
# Plan: docs/GOAL.md and docs/PLAN.md. One test task; the meta-training (P7) never contains it.
TEST_TASK = "move_playingcard_away"
T = ["--tasks", TEST_TASK, "--configs", "table_h_m05", "table_h_m10"]
CORE_TYPES = ["adapterplus", "lora_r16_attn", "adaln", "ia3_adaln", "film_memory", "film_error", "film_gapcurve",
              "film_text", "film_memory_plus_adapterplus", "prompt_vane", "ttt_wam"]
META_VLM = ["failfix_diff_vlm", "failfix_window_vlm", "failfix_textonly_vlm", "failfix_gapkeys_vlm"] + \
    [f"failfix_type_{t}_vlm" for t in CORE_TYPES]
# First wave (user 2026-11-13: results quickly, the VLM as it is = what failed, no correction direction): five
# mechanisms, short meta-training, evaluated as soon as they are trained; the other META_VLM adapters follow.
META_FAST = ["failfix_type_adapterplus_vlm", "failfix_type_lora_r16_attn_vlm", "failfix_type_film_text_vlm",
             "failfix_type_film_memory_vlm", "failfix_textonly_vlm"]
FAST_ENV = {"FAWAM_META_FAST": "1"}
TYPES_TRAIN = [f"failfix_types_{t}_vlm" for t in ("film_memory_plus_adapterplus", "lora_r16_attn", "adapterplus",
                                                    "ttt_wam")] + ["failfix_types_ft_adapterplus"]
TYPES_METHODS = {**{f"failfix_types_{t}_vlm": [f"icl_types_{t}_VP", f"icl_types_{t}_VP_noctx"]
                    for t in ("film_memory_plus_adapterplus", "lora_r16_attn", "adapterplus", "ttt_wam")},
                 "failfix_types_ft_adapterplus": ["ft_types_adapterplus"]}
P7BC_ROOTS = " ".join(str(ROOT / "results/sweeps" / s / "base" / c) for s, cs in
                      [("p7b_meta_data_dist", ("dist_robot10", "dist_cam05")),
                       ("p7c_meta_data_types", ("dist_objxy", "dist_objyaw", "dist_clutter", "dist_light", "dist_bg"))]
                      for c in cs)
DIST_TRAIN = [f"failfix_dist_{t}_vlm" for t in ("film_memory_plus_adapterplus", "lora_r16_attn", "adapterplus",
                                                  "ttt_wam")] + ["failfix_dist_ft_adapterplus"]
ADAPTERS = ROOT / "artifacts" / "adapters"
# hid minimal path (plan 2026-11-18, docs/PLAN_HID.md): p10_hid_data -> leak test + fail bank -> M3 (all C x aux in
# one job) and C1xM1 (failfix_hid_lora) -> training gate -> p12_hid_eval only for configurations that passed.
HID_ROOT = ROOT / "results/sweeps/p10_hid_data/base/hid_trans"
HID_M3 = ROOT / "artifacts/hid_m3"
HID_LEAK_MAX = 0.15


def hid_leak_ok():
    """Leak test passed: o is (almost) not readable from the first frame. Limit relaxed 0.1 -> 0.15 on user decision
    2026-11-19 (measured x 0.105, z 0.121); the noctx/shuf controls catch a model that reads o from the first frame."""
    p = HID_ROOT / "leak_test.json"
    if not p.exists():
        return False
    first = json.loads(p.read_text()).get("first frame") or {}
    return bool(first) and max(first.values()) < HID_LEAK_MAX


def hid_m3_passed():
    out = []
    for g in sorted(HID_M3.glob("*/gate.json")):
        if json.loads(g.read_text()).get("passed"):
            c = g.parent.name  # e.g. C3_M3_aux0.3
            out.append("hidm3_" + c.split("_M3")[0] + c.split("_M3")[1].replace(".", "").replace("_aux", "_aux"))
    return out


def hid_lora_passed():
    """C1xM1 gate from the meta-training log: last validation loss with memory < 0.9 x without memory."""
    p = ADAPTERS / "failfix_hid_lora" / "train_inside.jsonl"
    if not (ADAPTERS / "failfix_hid_lora" / "adapter.pt").exists() or not p.exists():
        return False
    vals = [json.loads(l) for l in p.read_text().splitlines() if '"val_action"' in l]
    return bool(vals) and vals[-1]["val_action"] < 0.9 * vals[-1]["val_action_noctx"]


def hid_eval():
    methods = ["retry_plain"]
    for m in hid_m3_passed():
        methods += [m, m + "_noctx", m + "_shuf"]
    if hid_lora_passed():
        methods += ["icl_hid_lora", "icl_hid_lora_noctx", "icl_hid_lora_shuf"]
    (GATES / "hid_gate.json").write_text(json.dumps(dict(methods=methods)))
    if len(methods) == 1:  # nothing passed the gate: no simulation (plan, section 6)
        return 0
    return sweep("p12_hid_eval", "--methods", *methods, prio=0)


# P8s (user 2026-11-18): P7d data (stronger/new disturbances) -> detailed VLM analysis of P7b+c+d -> adapters with and
# without the base model trained -> p8s_strong. The first P7d pass (12 episodes per cell) was submitted by hand.
P7BCD_ROOTS = P7BC_ROOTS + " " + " ".join(str(ROOT / "results/sweeps/p7d_meta_data_strong/base" / c) for c in
                                          ("dist_objxy10", "dist_objyaw90", "dist_heavy", "dist_gripweak"))
SC_TRAIN = [f"failfix_sc_{n}" for n in ("diffwin", "diffobswin", "predobswin", "gapkeys", "diffwin_vlmd",
                                          "textonly_vlmd")]
STRONG_TRAIN = [f"failfix_strong_{t}_vlmd{b}" for t in ("lora_r16_attn", "film_memory_plus_adapterplus")
                for b in ("", "_base")] + ["failfix_strong_ft_base"]
STRONG_METHODS = {**{f"failfix_strong_{t}_vlmd{b}": [f"icl_strong_{t}{b}_VPD", f"icl_strong_{t}{b}_VPD_noctx"]
                     for t in ("lora_r16_attn", "film_memory_plus_adapterplus") for b in ("", "_base")},
                  "failfix_strong_ft_base": ["ft_strong_base"]}


def p8_fast():
    ok = [n for n in META_FAST if (ADAPTERS / n / "adapter.pt").exists()]
    methods = ["retry_plain", "icl_fail_predobs_VP"]
    for n in ok:
        t = n[len("failfix_"):-len("_vlm")]
        methods += ["icl_meta_textonly_VP"] if t == "textonly" else [f"icl_{t}_VP", f"icl_{t}_VP_noctx"]
    return sweep("p8_icl_vlm_meta", "--methods", *methods, prio=0, env=VLM_ENV)


GEN_TASKS = ["handover_block", "move_can_pot", "place_shoe", "pick_diverse_bottles"]  # other failure types


def generalize():
    """Methods named in pipeline/gates/generalize (one per line or space-separated) on other tasks and failure types."""
    methods = ["retry_plain"] + [m for m in (GATES / "generalize").read_text().split() if m != "retry_plain"]
    vlm = any("VP" in m or "_V" in m for m in methods)
    cfg = dict(name="p10_generalize", out="results/sweeps/p10_generalize", episodes=30, per_job=10, n_par=2,
               time="16:00:00", tasks=GEN_TASKS, configs=["table_h_m05", "demo_randomized"], methods=methods)
    if vlm:
        cfg.update(vlm=True, extra_env=VLM_ENV)
    import yaml
    (ROOT / "configs/sweeps/p10_generalize.yaml").write_text(
        "# generated by cluster/pipeline.py from pipeline/gates/generalize\n" + yaml.safe_dump(cfg, sort_keys=False))
    return sweep("p10_generalize", prio=1, env=VLM_ENV if vlm else None)
META_ORACLE = ["failfix_type_adapterplus", "failfix_type_film_error", "failfix_type_film_gapcurve"]
DUAL_VARIANTS = [f"paper_P_dual_{k}" for k in ("s10", "s50", "video", "action", "vidtok", "n1", "n6", "filt", "lr3")]
P7_ROOT = "results/sweeps/p7_meta_data/base/table_h_rand"


def trainings_idle(jobs, names):
    return all(not any(j["name"] == f"fawam_train_{n}" for j in jobs) for n in names) and \
        not any(x.get("label", "") in {f"train_{n}" for n in names} for x in read_queue())


def stages(jobs):
    """v1 stages + plan v2 stages (cluster/pipeline_v2.py)."""
    import pipeline_v2
    return _stages_v1(jobs) + pipeline_v2.stages(jobs)


def _stages_v1(jobs):
    """(name, ready, action). Each action runs once; sweeps are re-emitted once for unfinished cells."""
    p7_done = idle(jobs, "p7_meta_data") and gate("p7_submitted")
    p7_oracle = p7_done and sacct_state("fawam_p7_analysis") == "COMPLETED"
    p7_vlm = p7_done and sacct_state("fawam_vlm_analysis_p7") == "COMPLETED"
    vllm_ready = (ROOT.parent / "envs/vllm/bin/python").exists() and "VLLM_ENV_DONE" in _read(ROOT / "../logs/build_vllm.log")
    # the VLM works: live in the loop (smoke) / offline (the P0 comparison job)
    # gate vlm_setup: opened by hand once fawam/vlm_bench.py has picked the VLM and its input/prompt (2026-11-10: the
    # first setup did not give usable corrections, so no VLM sweep or VLM training data before that)
    vlm_live = vllm_ready and smoke_ok("vlm") and gate("vlm_setup")
    vlm_offline = vllm_ready and gate("vlm_setup")
    return [
        # A. Learn-from-Failure through an adapter update: which analysis signal is needed (oracle = upper bound)
        ("lff", lambda: idle(jobs, "p5a_signals") and prep_ok(),
         lambda: sweep("p5a_signals", *T, "--episodes", "30", prio=2)),
        ("lff_vlm", lambda: vlm_live and prep_ok(),
         lambda: sweep("p5b_vlm", *T, "--episodes", "30", "--methods", "lff_sig_VP", prio=2, env=VLM_ENV)),
        # dual-site adapter + the VLM's description of the failure in the prompt
        ("dual_vlm", lambda: vlm_live and prep_ok(),
         lambda: sweep("p5f_paper_vlm", *T, "--episodes", "80", prio=1, env=VLM_ENV)),
        # B. Dual-site adapter, paper protocol: prediction error only (P) vs. + rollout (PR) vs. + oracle correction (OP)
        ("dual", lambda: smoke_ok("paper") and prep_ok(),
         lambda: sweep("p5f_paper", *T, "--episodes", "80", "--methods", "base", "paper_P_dual", "paper_P_inside",
                       "paper_P_outside", "paper_P_dual_150", "paper_PR_dual", "paper_OP_dual", prio=1)),
        # B2. Dual-site variants with the prediction error only (user 2026-11-12: is there more in it?)
        ("dual_variants", lambda: _stage_done("dual") and prep_ok(),
         lambda: sweep("p5f_paper", *T, "--episodes", "80", "--methods", "paper_Pw_dual", *DUAL_VARIANTS, prio=1)),
        # B3. best two variants combined: 1 adaptation episode + adapter on the video tokens only (user 2026-11-13)
        ("dual_best", lambda: prep_ok(),
         lambda: sweep("p5f_paper", *T, "--episodes", "80", "--methods", "paper_P_dual_n1vidtok", prio=0)),
        # C. ICL without meta-training (control: does the frozen WAM use the context at all?)
        ("icl_free", lambda: smoke_ok("icl") and prep_ok(),
         lambda: sweep("p6a_icl_content", *T, "--episodes", "30", "--methods", "retry_plain", "icl_text_only",
                       "icl_fail_predobs", "icl_fail_diff", "icl_vs_adapter", prio=2)),
        ("icl_free_vlm", lambda: vlm_live and prep_ok(),
         lambda: sweep("p6v_icl_free_vlm", "--methods", "retry_plain", "icl_text_only+VP", "icl_txt_negative+VP",
                       "icl_fail_predobs+VP", "icl_manip_noise+VP", prio=1, env=VLM_ENV)),
        # E. Mechanisms of closely related work on the same model and task
        ("related", lambda: smoke_ok("lff") and prep_ok(), lambda: sweep("p9_related", prio=2)),
        # F. A method that works: other tasks / failure types (touch pipeline/gates/generalize with the method names)
        ("generalize", lambda: gate("generalize") and (GATES / "generalize").read_text().strip() != "", generalize),
        # D. Meta-trained ICL (the idea): analyses of the P7 failures, meta-training without the test task, evaluation
        ("p7_analysis", lambda: p7_done,
         lambda: cpu_job("p7_analysis", f"$PY_RT -m fawam.failure_analysis {P7_ROOT}")),
        ("p7_vlm_analysis", lambda: p7_done and vlm_offline,
         lambda: gpu_job("vlm_analysis_p7", dict(ROOTS=" ".join(str(ROOT / "results/sweeps/p7_meta_data/base" / c)
                                                                for c in ("table_h_rand", "demo_randomized"))),
                         "vlm_analysis_job.sh", prio=0, time="01:30:00")),
        ("meta_vlm_fast", lambda: p7_vlm, lambda: train(META_FAST, prio=0, env=FAST_ENV)),
        ("p8_fast", lambda: _stage_done("meta_vlm_fast") and trainings_idle(jobs, META_FAST) and vlm_live and
            any((ADAPTERS / n / "adapter.pt").exists() for n in META_FAST), p8_fast),
        ("meta_vlm", lambda: _stage_done("p8_fast"),
         lambda: train([n for n in META_VLM if n not in META_FAST], prio=2, env=FAST_ENV)),
        # 2026-11-12 (user): no further oracle experiments -- the method has to work without simulator knowledge.
        # The oracle stages meta_oracle / p8_oracle were removed; existing oracle results stay as reference only.
        # more diverse meta-training data: pairs under start-posture and camera disturbances. Starts as soon as the
        # screening has produced episodes with both disturbances (= fawam/sim_disturb.py works in the simulator).
        ("p7b_data", lambda: all(any((ROOT / "results/sweeps/p0s_task_screen/base" / c / "episodes").glob("*/*/*"))
                                 for c in ("dist_robot10", "dist_cam05")),
         lambda: sweep("p7b_meta_data_dist", prio=1)),
        # plain fine-tuning on the same data (no memory, no text) as the baseline of D
        ("ft_train", lambda: p7_done, lambda: train(["failfix_ft_adapterplus"], prio=0, env=FAST_ENV)),
        ("ft_eval", lambda: _stage_done("ft_train") and (ADAPTERS / "failfix_ft_adapterplus" / "adapter.pt").exists(),
         lambda: sweep("p8f_finetune", prio=0)),
        ("p8", lambda: _stage_done("meta_vlm") and trainings_idle(jobs, META_VLM) and vlm_live,
         lambda: sweep("p8_icl_vlm_meta", prio=0, env=VLM_ENV)),
        # held-out FAILURE TYPE (user 2026-11-17): VLM analysis of the P7b failures -> meta-training (DIST_TRAIN) on
        # start-posture + camera failures only -> p8d_heldout_type on the lowered table. Submitted by hand on
        # 2026-11-17 as a SLURM dependency chain (scron ticks stall while VS Code runs on `agent`), not as stages.
        # Only these cells did not fit under the submit limit then; they go through the feeder once trained.
        # many failure types, few episodes each (user 2026-11-17): P7c data -> VLM analysis of P7b+P7c failures ->
        # meta-training on 7 types (12 failures per task and type) -> test on the lowered table (p8t_heldout_types)
        ("p7c_data", lambda: True, lambda: sweep("p7c_meta_data_types", prio=0)),
        ("p7c_vlm_analysis", lambda: _stage_done("p7c_data") and idle(jobs, "p7c_meta_data_types"),
         lambda: gpu_job("vlm_analysis_p7c", dict(ROOTS=P7BC_ROOTS), "vlm_analysis_job.sh", prio=0, time="01:00:00")),
        ("types_train", lambda: sacct_state("fawam_vlm_analysis_p7c") == "COMPLETED",
         lambda: train(TYPES_TRAIN, prio=0, env=FAST_ENV)),
        ("types_eval", lambda: _stage_done("types_train") and trainings_idle(jobs, TYPES_TRAIN) and vlm_live and
            any((ADAPTERS / n / "adapter.pt").exists() for n in TYPES_TRAIN),
         lambda: sweep("p8t_heldout_types", "--methods", *[m for n in TYPES_TRAIN if (ADAPTERS / n / "adapter.pt").exists()
                                                           for m in TYPES_METHODS[n]], prio=0, env=VLM_ENV)),
        ("p7d_topup", lambda: idle(jobs, "p7d_meta_data_strong") and
            any((ROOT / "results/sweeps/p7d_meta_data_strong/base").glob("*/episodes/*/*/*")),
         lambda: sweep("p7d_meta_data_strong", prio=0)),
        ("p7d_vlm_analysis", lambda: _stage_done("p7d_topup") and idle(jobs, "p7d_meta_data_strong") and
            not any(x.get("label", "").startswith("p7d_meta_data_strong") for x in read_queue()),
         lambda: gpu_job("vlm_analysis_p7bcd_detail", dict(ROOTS=P7BCD_ROOTS, VLM_DETAIL="1"), "vlm_analysis_job.sh",
                         prio=0, time="04:00:00")),
        ("strong_train", lambda: sacct_state("fawam_vlm_analysis_p7bcd_detail") == "COMPLETED",
         lambda: train(STRONG_TRAIN, prio=0, env=FAST_ENV)),
        ("strong_eval", lambda: _stage_done("strong_train") and trainings_idle(jobs, STRONG_TRAIN) and vlm_live and
            any((ADAPTERS / n / "adapter.pt").exists() for n in STRONG_TRAIN),
         lambda: sweep("p8s_strong", "--methods", *[m for n in STRONG_TRAIN if (ADAPTERS / n / "adapter.pt").exists()
                                                    for m in STRONG_METHODS[n]], prio=0, env=VLM_ENV)),
        # P8c small contexts (user 2026-11-20): 6 LoRA adapters -> p8c_small_ctx (ctx + noctx each)
        ("sc_train", lambda: True, lambda: train(SC_TRAIN, prio=0, env=FAST_ENV)),
        ("sc_eval", lambda: _stage_done("sc_train") and trainings_idle(jobs, SC_TRAIN) and vlm_live and
            any((ADAPTERS / n / "adapter.pt").exists() for n in SC_TRAIN),
         lambda: sweep("p8c_small_ctx", "--methods", *[m for n in SC_TRAIN if (ADAPTERS / n / "adapter.pt").exists()
                                                      for m in (n.replace("failfix_sc_", "icl_sc_"),
                                                                n.replace("failfix_sc_", "icl_sc_") + "_noctx")],
                       prio=0, env=VLM_ENV)),
        # hid C2 as a gate exception (user 2026-11-20): full eval once the smoke run finished cleanly
        ("hid_c2_eval", lambda: sacct_state("fawam_p12_hid_c2_smoke") == "COMPLETED" and
            any((ROOT / "results/sweeps/p12_hid_c2_smoke").glob("*/*/episodes/*/*/*/episode.h5")),
         lambda: sweep("p12_hid_c2", prio=0)),
        ("hid_leak", lambda: any(HID_ROOT.glob("episodes/*/*/*/episode.h5")) and idle(jobs, "p10_hid_data"),
         lambda: cpu_job("hid_leak", f"$PY_RT -m fawam.hid_leak {HID_ROOT} && $PY_RT -m fawam.demo_bank build "
                                     f"{HID_ROOT} --out {ROOT}/artifacts/fail_bank/hid_trans_eval --success false "
                                     f"--drivers policy")),
        ("hid_train", lambda: _stage_done("hid_leak") and hid_leak_ok(),
         lambda: gpu_job("train_hid_m3", dict(ROOTS=str(HID_ROOT), OUT=str(HID_M3), STEPS="1500",
                                               EXTRA="--max_hours 2.5"), "train_hid_job.sh", prio=0, time="03:00:00")
         + train(["failfix_hid_lora"], prio=0, env=FAST_ENV)),
        ("hid_eval", lambda: _stage_done("hid_train") and sacct_state("fawam_train_hid_m3") in ("COMPLETED", "FAILED",
                                                                                                 "TIMEOUT")
            and trainings_idle(jobs, ["failfix_hid_lora"]), hid_eval),
        ("dist_eval_rest", lambda: all((ADAPTERS / n / "adapter.pt").exists()
                                       for n in ("failfix_dist_ttt_wam_vlm", "failfix_dist_ft_adapterplus")),
         lambda: sweep("p8d_heldout_type", "--methods", "icl_dist_ttt_wam_VP_noctx", "ft_dist_adapterplus", prio=0,
                       env=VLM_ENV) +
         sweep("p8d_heldout_type", "--methods", "icl_dist_ttt_wam_VP", "--configs", "table_h_m10", prio=0,
               env=VLM_ENV)),
    ]


_GRID = None


def _grid_names():
    global _GRID
    if _GRID is None:
        r = sh([PY, "cluster/adapter_grid.py", "list"])
        _GRID = [l.split()[0] for l in r.stdout.splitlines() if l.strip()]
    return _GRID


def _read(p):
    try:
        return Path(p).read_text()
    except OSError:
        return ""


def _touch(name):
    (GATES / name).touch()
    return True


def _done():
    return json.loads(DONE.read_text()) if DONE.exists() else {}


def _stage_done(name):
    return name in _done()


SWEEP_OF = {"lff": ["p5a_signals"], "lff_vlm": ["p5b_vlm"], "dual": ["p5f_paper"], "icl_free": ["p6a_icl_content"],
            "p8": ["p8_icl_vlm_meta"], "p8_oracle": ["p8_icl_oracle_meta"], "icl_free_vlm": ["p6v_icl_free_vlm"],
            "related": ["p9_related"], "ft_eval": ["p8f_finetune"], "generalize": ["p10_generalize"],
            # plan v2: one re-emission per sweep once it is idle (re-queues cells that are short of episodes)
            "v2_data": ["p21_data_sa", "p21_data_sb"], "v2_bank": ["p21_bank"], "v2_d3": ["p20_d3_snap"],
            "v2_d2d4": ["p20_d2_base", "p20_d4_tf"], "v2_ttt": ["p22_ttt"]}


def run_stages(jobs):
    done = _done()
    for name, ready, action in stages(jobs):
        if name in done:
            continue
        try:
            if not ready():
                continue
            n = action()
        except Exception as e:  # keep the pipeline alive; the stage is retried next tick
            log(f"stage {name} error: {e!r}")
            continue
        done[name] = dict(at=f"{datetime.datetime.now():%Y-%m-%d %H:%M}", jobs=int(n))
        DONE.write_text(json.dumps(done, indent=1))
        log(f"stage {name}: queued {int(n)} job(s)")
    # one re-emission per finished sweep stage: re-queues only cells that are still short of episodes
    for name, sweeps in SWEEP_OF.items():
        if name in done and f"{name}_retry" not in done and all(idle(jobs, s) for s in sweeps):
            stage_args = done[name]
            done[f"{name}_retry"] = dict(at=f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
            DONE.write_text(json.dumps(done, indent=1))
            n = _reemit(name)
            log(f"stage {name} finished; re-queued {n} job(s) for unfinished cells")


def _reemit(name):
    """Re-run the stage's emission: submit_sweep skips cells that already have enough episodes."""
    for sname, ready, action in stages(squeue()):
        if sname == name:
            return action()
    return 0


# ------------------------------------------------------------------------------------------------ feeder
def feed(jobs):
    count = {a: sum(j["account"] == a for j in jobs) for a in ACCOUNTS}
    q = sorted(read_queue(), key=lambda x: (x.get("prio", 9), x.get("seq", 0)))
    rest, submitted = [], 0
    for i, x in enumerate(q):
        acct = min(ACCOUNTS, key=lambda a: count[a])
        if count[acct] >= CAP:
            rest += q[i:]
            break
        argv = [x["argv"][0], f"--account={acct}", *x["argv"][1:]]
        r = sh(argv, x.get("env"))
        if r.returncode == 0:
            count[acct] += 1
            submitted += 1
            with open(STATE / "submitted.jsonl", "a") as f:
                f.write(json.dumps(dict(label=x.get("label"), account=acct, out=r.stdout.strip(),
                                        at=f"{datetime.datetime.now():%Y-%m-%d %H:%M}")) + "\n")
        elif "MaxSubmit" in r.stderr or "QOSMax" in r.stderr:
            count[acct] = CAP  # account full after all, try the other one next
            rest.append(x)
        else:
            log(f"sbatch failed for {x.get('label')}: {r.stderr.strip()[:300]}")
            with open(STATE / "failed.jsonl", "a") as f:
                f.write(json.dumps(dict(**x, error=r.stderr.strip())) + "\n")
    write_queue(rest)
    if submitted:
        log(f"submitted {submitted} job(s); {len(rest)} still queued")


# ------------------------------------------------------------------------------------------------ status
def status(jobs):
    by = {}
    for j in jobs:
        if j["name"].startswith("fawam_"):
            by.setdefault(j["name"][6:], {}).setdefault(j["state"], 0)
            by[j["name"][6:]][j["state"]] += 1
    q = read_queue()
    qby = {}
    for x in q:
        k = x.get("label", "?").split()[0]
        qby[k] = qby.get(k, 0) + 1
    lines = [f"# Pipeline status ({datetime.datetime.now():%Y-%m-%d %H:%M})", "",
             "## Stages done", *[f"- {k}: {v}" for k, v in _done().items()], "",
             "## SLURM (fawam jobs)", *[f"- {k}: {v}" for k, v in sorted(by.items())], "",
             f"## Waiting in pipeline queue: {len(q)}", *[f"- {k}: {v}" for k, v in sorted(qby.items())], "",
             "## Gates", *[f"- {g.name}" for g in sorted(GATES.glob('*'))], "",
             "## Accounts", *[f"- {a}: {sum(j['account'] == a for j in jobs)} / 100" for a in ACCOUNTS]]
    (STATE / "STATUS.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tick", "status"])
    a = ap.parse_args()
    for d in (STATE, GATES, ROOT / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    with open(STATE / ".lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another tick is running")
            return
        if a.cmd == "tick":
            jobs = squeue()
            run_stages(jobs)
            feed(squeue())
        print(status(squeue()))


if __name__ == "__main__":
    main()
