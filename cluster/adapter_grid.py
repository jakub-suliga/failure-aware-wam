#!/usr/bin/env python3
"""Adaptation (D) and in-context (E) grid: training jobs, method files and the evaluation sweeps that use them.

  python cluster/adapter_grid.py train [--submit] [--only d1_]   # sbatch cluster/train_job.sh per adapter
  python cluster/adapter_grid.py banks                           # demo banks from the expert data (CPU, minutes)
  python cluster/adapter_grid.py methods                         # configs/methods/*.json + p3/p4 sweep YAMLs
then  python cluster/submit_sweep.py configs/sweeps/p3_adapters.yaml --submit   (and p4_icl.yaml)

Needs P2 (expert replays) and, for the self-supervised variants, P0 (base rollouts) in results/sweeps/.
Adapters are multi-task over TABLE4 (n episodes per task) unless the name says otherwise.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TABLE4 = ["move_playingcard_away", "place_mouse_pad", "pick_diverse_bottles", "stack_blocks_two"]
EXPERT = "results/sweeps/p2_expert_data/base/{cfg}"
ROLL = "results/sweeps/p0_baseline_table/base/{cfg}"
ADAPTERS = "artifacts/adapters"


def expert(*cfgs):
    return " ".join(EXPERT.format(cfg=c) for c in cfgs)


def roll(*cfgs):
    return " ".join(ROLL.format(cfg=c) for c in cfgs)


# name -> (train env, eval configs). LIMIT_PER_TASK = few-shot n per task.
GRID = {}
for n in (3, 10, 30):  # D1: dual-site, sequential (paper), expert demos under the shift
    GRID[f"d1_dual_m05_n{n}"] = (dict(EPISODES=expert("table_h_m05"), LIMIT_PER_TASK=n, SUCCESS="true"),
                                 ["table_h_m05", "table_h_m075", "table_h_m10", "table_h_z00"])
D2 = {  # D2: placement / parametrisation ablations at n=10
    "d2_inside_both": dict(KIND="inside", OUTSIDE="none"),
    "d2_inside_action": dict(KIND="inside", STREAMS="action", OUTSIDE="none"),
    "d2_inside_video": dict(KIND="inside", STREAMS="video", OUTSIDE="none"),
    "d2_outside_only": dict(KIND="outside", OUTSIDE="parallel"),
    "d2_lora": dict(KIND="lora", OUTSIDE="none"),
    "d2_dual_parallel": dict(OUTSIDE="parallel"),
    "d2_dual_actloss": dict(LAMBDA_V="0"),
    "d2_dual_vidloss": dict(LAMBDA_A="0"),
}
for k, v in D2.items():
    GRID[k] = (dict(EPISODES=expert("table_h_m05"), LIMIT_PER_TASK=10, SUCCESS="true", **v),
               ["table_h_m05", "table_h_z00"])
for n in (3, 10, 30):  # D3: paper-faithful self-supervised adaptation on the policy's own rollouts (all outcomes)
    GRID[f"d3_selfsup_m05_n{n}"] = (dict(EPISODES=roll("table_h_m05"), LIMIT_PER_TASK=n, SUCCESS="any"),
                                    ["table_h_m05", "table_h_m10", "table_h_z00"])
GRID["d3_selfsup_actloss_m05_n10"] = (dict(EPISODES=roll("table_h_m05"), LIMIT_PER_TASK=10, SUCCESS="any",
                                           LAMBDA_V="0"), ["table_h_m05", "table_h_z00"])
GRID["d4_filtered_m05"] = (dict(EPISODES=roll("table_h_m05"), SUCCESS="true"), ["table_h_m05", "table_h_z00"])
GRID["d5_multishift_n10"] = (dict(EPISODES=expert("table_h_m025", "table_h_m05", "table_h_m10"), LIMIT_PER_TASK=30,
                                  SUCCESS="true"),
                             ["table_h_m025", "table_h_m05", "table_h_m075", "table_h_m10", "table_h_z00", "table_h_m15"])
GRID["d6_hard_n30"] = (dict(EPISODES=expert("demo_randomized"), LIMIT_PER_TASK=30, SUCCESS="true"), ["demo_randomized"])
# E3: in-context training (one same-task demo in front of every sample), multi-shift; evaluated with prefill (p4)
GRID["e3_icl_multishift"] = (dict(EPISODES=expert("table_h_m025", "table_h_m05", "table_h_m10"), LIMIT_PER_TASK=30,
                                  SUCCESS="true", CONTEXT_K="1", OUTSIDE="none",
                                  EXTRA="--max_frames 20 --context_pool " + expert("table_h_m025", "table_h_m05",
                                                                                   "table_h_m10")),
                             ["table_h_m05", "table_h_m075", "table_h_z00"])
# FailFix meta-training (Learn-from-Failure, ICL mode): context = own failed attempt (imagined + observed), target =
# the expert replay on the SAME seed (P0 and P2 share seeds), prompt = instruction (+ correction text with p 0.5).
# Needs analysis.json next to the failures for the correction text: python -m fawam.failure_analysis <P0 fails>.
GRID["failfix_meta"] = (dict(EPISODES=roll("table_h_m025", "table_h_m05", "table_h_m10", "demo_randomized"),
                             SUCCESS="false", OUTSIDE="none",
                             EXTRA="--include_pred --p_text 0.5 --max_frames 20 --steps 800 --fixes " +
                                   expert("table_h_m025", "table_h_m05", "table_h_m10", "demo_randomized")),
                        ["table_h_m05", "table_h_m10"])
# Meta-training per in-context memory format (docs/ICL.md): the test-time format of icl_meta_<fmt> must match.
# Data: P7 (driver pair: failure + expert fix on the same seed; random table height + hard setting, 8 tasks).
_P7 = " ".join(f"results/sweeps/p7_meta_data/base/{c}" for c in ("table_h_rand", "demo_randomized"))
_FF_EP, _FF_FIX = _P7, _P7
_META = ("--p_text 0.5 --p_cross 0.2 --p_success 0.1 --p_noctx 0.15 --val_frac 0.1 --val_every 1000 "
         "--save_every 2000 --max_frames 20 --steps 12000 --grad_accum 4")
TABLE8 = ["move_playingcard_away", "place_mouse_pad", "pick_diverse_bottles", "stack_blocks_two", "place_shoe",
          "move_can_pot", "lift_pot", "handover_block"]
for _name, _extra in {"failfix_obs": "--context_groups obs",
                      "failfix_predobs": "--context_groups pred obs",
                      "failfix_diff": "--context_groups diff obs",
                      "failfix_window": "--context_groups pred obs --context_window",
                      # ablations of the meta-training itself (format pred+obs)
                      "failfix_predobs_ownonly": "--context_groups pred obs --p_cross 0 --p_success 0 --p_noctx 0",
                      "failfix_predobs_notext": "--context_groups pred obs --p_text 0",
                      "failfix_predobs_small": "--context_groups pred obs --steps 3000",
                      }.items():
    GRID[_name] = (dict(EPISODES=_FF_EP, SUCCESS="false", OUTSIDE="none", TASKS=" ".join(TABLE8),
                        EXTRA=f"{_META} {_extra} --fixes {_FF_FIX}"),
                   ["table_h_m05", "table_h_m10"])
# generalisation to an unseen TASK: meta-trained without stack_blocks_two, tested on it
GRID["failfix_predobs_holdtask"] = (dict(EPISODES=_FF_EP, SUCCESS="false", OUTSIDE="none",
                                         TASKS=" ".join(t for t in TABLE8 if t != "stack_blocks_two"),
                                         EXTRA=f"{_META} --context_groups pred obs --fixes {_FF_FIX}"),
                                    ["table_h_m05", "table_h_m10"])
# demo in context -> success (no failure): the E3 counterpart, for icl_meta_demo
GRID["democtx_meta"] = (dict(EPISODES=_FF_FIX, SUCCESS="true", OUTSIDE="none", CONTEXT_K="1", DRIVERS="expert_replay",
                             TASKS=" ".join(TABLE8),
                             EXTRA=f"--max_frames 20 --steps 12000 --grad_accum 4 --save_every 2000 "
                                   f"--context_pool {_FF_FIX}"),
                        ["table_h_m05", "table_h_m10"])
ICL_CONFIGS = ["table_h_m05", "table_h_m075", "table_h_m10", "table_h_z00", "demo_randomized"]


def train(a):
    for name, (env, _) in GRID.items():
        if a.only and not name.startswith(a.only):
            continue
        out = ROOT / ADAPTERS / name
        if (out / "adapter.pt").exists() and not a.force:
            print(f"{name}: exists, skip")
            continue
        e = dict(FAWAM_ROOT=str(ROOT), OUT=str(out), **{k: str(v) for k, v in env.items()})
        if "hard" not in name:
            e.setdefault("TASKS", " ".join(TABLE4))
        if name.startswith(("failfix_", "democtx_")):
            e.setdefault("TIME", "47:00:00")
        out.mkdir(parents=True, exist_ok=True)
        cmd = ["sbatch", f"--job-name=fawam_train_{name}", f"--time={e.get('TIME', a.time)}", f"--output={out}/slurm_%j.out",
               f"--chdir={ROOT}"]
        if os.environ.get("PARTITION"):
            cmd.append(f"--partition={os.environ['PARTITION']}")
        cmd.append(str(ROOT / "cluster" / "train_job.sh"))
        if a.submit:
            r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **e})
            print(r.stdout.strip() or r.stderr.strip(), "<-", name)
        else:
            print(name, json.dumps(e))
    if not a.submit:
        print("(dry run, add --submit)")


def banks(a):
    import sys
    sys.path.insert(0, str(ROOT))
    from fawam.demo_bank import build
    for cfg in ICL_CONFIGS:
        src = ROOT / EXPERT.format(cfg=cfg)
        if not src.exists():
            print(f"{cfg}: no expert data yet ({src})")
            continue
        build(argparse.Namespace(roots=[str(src)], out=str(ROOT / "artifacts" / "demo_bank" / cfg), success=True,
                                 settings=None, drivers=None))


def methods(a):
    mdir = ROOT / "configs" / "methods"
    mdir.mkdir(parents=True, exist_ok=True)
    p3_methods, overrides = ["base"], {}
    for name, (_, eval_cfgs) in GRID.items():
        if not (ROOT / ADAPTERS / name / "adapter.pt").exists():
            continue
        variants = {name: dict(adapter_path=f"{ADAPTERS}/{name}")}
        if not name.startswith(("d2_inside", "d2_lora", "d2_outside", "e3_")):
            # eval-time site ablation of a dual adapter (no retraining)
            variants[f"{name}@inside"] = dict(adapter_path=f"{ADAPTERS}/{name}", adapter_use="inside")
            variants[f"{name}@outside"] = dict(adapter_path=f"{ADAPTERS}/{name}", adapter_use="outside")
        for vname, spec in variants.items():
            if name.startswith(("e3_", "failfix_", "democtx_")):
                continue  # evaluated with prefill in p4 / through Method.learn.meta_adapter in p5
            f = mdir / f"{vname.replace('@', '__')}.json"
            f.write_text(json.dumps(dict(name=vname, **spec), indent=1))
            p3_methods.append(dict(name=vname, spec=str(f.relative_to(ROOT))))
            overrides[vname] = dict(configs=eval_cfgs)
    sweep = dict(name="p3_adapters", out="results/sweeps/p3_adapters", episodes=30, per_job=15, n_par=3,
                 time="12:00:00", tasks=TABLE4, configs=sorted({c for v in overrides.values() for c in v["configs"]}),
                 methods=p3_methods, method_overrides=overrides)
    (ROOT / "configs" / "sweeps" / "p3_adapters.yaml").write_text(
        "# generated by cluster/adapter_grid.py methods\n" + yaml.safe_dump(sweep, sort_keys=False))
    print(f"p3_adapters: {len(p3_methods)} methods")

    p4, ov4 = ["base"], {}
    e3 = (ROOT / ADAPTERS / "e3_icl_multishift" / "adapter.pt").exists()
    for cfg in ICL_CONFIGS:
        bank = ROOT / "artifacts" / "demo_bank" / cfg
        if not (bank / "index.json").exists():
            continue
        specs = {f"icl1_{cfg}": dict(prefill=dict(bank=f"artifacts/demo_bank/{cfg}", k=1, select="nearest", gap=1)),
                 f"icl2_{cfg}": dict(prefill=dict(bank=f"artifacts/demo_bank/{cfg}", k=2, select="nearest", gap=1)),
                 f"icl1rand_{cfg}": dict(prefill=dict(bank=f"artifacts/demo_bank/{cfg}", k=1, select="random", gap=1)),
                 f"icl1gap8_{cfg}": dict(prefill=dict(bank=f"artifacts/demo_bank/{cfg}", k=1, select="nearest", gap=8)),
                 # wrong-shift demo: does the model use the demo's content or only "some context"?
                 f"icl1wrong_{cfg}": dict(prefill=dict(bank="artifacts/demo_bank/table_h_z00", k=1, select="nearest",
                                                       gap=1))}
        if e3:
            specs[f"e3icl1_{cfg}"] = dict(adapter_path=f"{ADAPTERS}/e3_icl_multishift", prefill=specs[f"icl1_{cfg}"]["prefill"])
            specs[f"e3noctx_{cfg}"] = dict(adapter_path=f"{ADAPTERS}/e3_icl_multishift")
        for vname, spec in specs.items():
            f = mdir / f"{vname}.json"
            f.write_text(json.dumps(dict(name=vname, **spec), indent=1))
            p4.append(dict(name=vname, spec=str(f.relative_to(ROOT))))
            ov4[vname] = dict(configs=[cfg])
    p4.append("retry_memory")
    sweep = dict(name="p4_icl", out="results/sweeps/p4_icl", episodes=30, per_job=15, n_par=3, time="12:00:00",
                 tasks=TABLE4, configs=ICL_CONFIGS, methods=p4, method_overrides=ov4)
    (ROOT / "configs" / "sweeps" / "p4_icl.yaml").write_text(
        "# generated by cluster/adapter_grid.py methods\n" + yaml.safe_dump(sweep, sort_keys=False))
    print(f"p4_icl: {len(p4)} methods")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "banks", "methods", "list"])
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--time", default="06:00:00")
    a = ap.parse_args()
    if a.cmd == "list":
        for k, (env, cfgs) in GRID.items():
            print(f"{k:28s} eval on {cfgs}  train {env}")
        return
    {"train": train, "banks": banks, "methods": methods}[a.cmd](a)


if __name__ == "__main__":
    main()
