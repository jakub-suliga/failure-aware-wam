#!/usr/bin/env python3
"""Expand a sweep YAML (configs/sweeps/*.yaml) into SLURM jobs of cluster/eval_job.sh.

  python cluster/submit_sweep.py configs/sweeps/p1_training_free.yaml            # dry run: list the jobs
  python cluster/submit_sweep.py configs/sweeps/p1_training_free.yaml --submit   # sbatch them
  python cluster/submit_sweep.py configs/sweeps/p1_training_free.yaml --status   # episodes done per cell

Every method of a sweep runs on the same seeds (job j, pair i -> seed index j * n_par + i), so methods are compared
on identical scenes. Cells that already have >= `episodes` finished episodes are skipped (re-submit after failures).

YAML keys: name, out, tasks, configs, methods, episodes (per cell), per_job, n_par, time, partition (optional),
driver (policy | expert_replay), vlm (bool), extra_env (dict), seed_offset (int, default 0),
method_overrides: {method: {configs: [...], tasks: [...]}} to restrict a method to a subset.
"""
import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", s)[:80]


def count_done(cell_dir, task):
    n = 0
    for kind in ("success", "fail"):
        d = cell_dir / "episodes" / kind / task
        if d.exists():
            n += sum(1 for e in d.iterdir() if (e / "episode.h5").exists() and "_attempt" not in e.name)
    return n


def cells(sw):
    for m in sw["methods"]:
        name = m if isinstance(m, str) else m["name"]
        spec = m if isinstance(m, str) else m.get("spec", m["name"])
        ov = (sw.get("method_overrides") or {}).get(name, {})
        for cfg in ov.get("configs", sw["configs"]):
            for task in ov.get("tasks", sw["tasks"]):
                yield name, spec, cfg, task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max_jobs", type=int, default=10**9)
    ap.add_argument("--methods", nargs="*", help="only these methods (tiering)")
    ap.add_argument("--configs", nargs="*", help="only these configs")
    ap.add_argument("--tasks", nargs="*", help="only these tasks")
    ap.add_argument("--episodes", type=int, help="override episodes per cell (e.g. a first, smaller pass)")
    a = ap.parse_args()
    sw = yaml.safe_load(Path(a.sweep).read_text())
    if a.episodes:
        sw["episodes"] = a.episodes
    out_root = (ROOT / sw["out"]).resolve()
    per_job, n_par, total = int(sw.get("per_job", 20)), int(sw.get("n_par", 2)), int(sw["episodes"])
    per_pair = max(1, math.ceil(per_job / n_par))
    jobs_per_cell = math.ceil(total / (per_pair * n_par))
    n_jobs = 0
    for name, spec, cfg, task in cells(sw):
        if (a.methods and name not in a.methods) or (a.configs and cfg not in a.configs) or \
                (a.tasks and task not in a.tasks):
            continue
        cell = out_root / safe(name) / cfg
        done = count_done(cell, task)
        if a.status:
            print(f"{name:28s} {cfg:18s} {task:26s} {done:4d}/{total}")
            continue
        if done >= total:
            continue
        first_job = done // (per_pair * n_par)  # resume at the first unfinished seed block
        for j in range(first_job, jobs_per_cell):
            if n_jobs >= a.max_jobs:
                break
            env = dict(FAWAM_ROOT=str(ROOT), TASK=task, CONFIG=cfg, METHOD=spec, NUM=str(per_pair),
                       SEED0=str(int(sw.get("seed_offset", 0)) + j * n_par), N_PAR=str(n_par), OUT=str(cell),
                       DRIVER=sw.get("driver", "policy"), VLM="1" if sw.get("vlm") else "0",
                       **{k: str(v) for k, v in (sw.get("extra_env") or {}).items()})
            (cell / "logs").mkdir(parents=True, exist_ok=True)
            cmd = ["sbatch", f"--job-name=fawam_{safe(sw['name'])}", f"--time={sw.get('time', '08:00:00')}",
                   f"--output={cell}/logs/slurm_%j.out", f"--chdir={ROOT}",
                   "--export=ALL," + ",".join(f"{k}={v}" for k, v in env.items())]
            part = sw.get("partition") or os.environ.get("PARTITION")
            if part:
                cmd.append(f"--partition={part}")
            cmd.append(str(ROOT / "cluster" / "eval_job.sh"))
            n_jobs += 1
            if a.submit and os.environ.get("FAWAM_QUEUE"):  # cluster/pipeline.py feeds the queue (submit limit)
                with open(os.environ["FAWAM_QUEUE"], "a") as q:
                    q.write(json.dumps(dict(argv=cmd, env={}, label=f"{sw['name']} {name} {cfg} {task} job {j}")) + "\n")
            elif a.submit:
                r = subprocess.run(cmd, capture_output=True, text=True)
                print(r.stdout.strip() or r.stderr.strip(), f"<- {name} {cfg} {task} job {j}")
            else:
                print(" ".join(cmd[:4]), f"... {name} | {cfg} | {task} | seeds {env['SEED0']}+{n_par} x {per_pair}")
    if not a.status:
        print(f"{n_jobs} job(s) {'submitted' if a.submit else '(dry run, add --submit)'}")


if __name__ == "__main__":
    main()
