"""Worker for cluster/dev_runner.sh: runs waiting jobs on one GPU of a multi-GPU dev-partition job.

The dev QoS allows one running job per user, but dev-gpu-b200 has 4 GPUs per node and a 2-day limit, so one job with
several GPUs can work through jobs that are stuck behind the priority queue of the normal partitions.

pipeline/dev_queue.jsonl: one task per line {id, label, script, env, log, est_min, slurm_id, state}. A task that also
waits as a normal SLURM job (slurm_id) is put on hold while it runs here; it is cancelled when the run here succeeds
and released again when it fails or the dev job is stopped. Tasks whose SLURM job already started are skipped.
  python cluster/dev_queue.py work --gpu 0 --end <unix time>
  python cluster/dev_queue.py status
"""
import argparse, fcntl, json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Q, LOCK = ROOT / "pipeline" / "dev_queue.jsonl", ROOT / "pipeline" / ".dev_lock"
ENV_CLEAN = {k: v for k, v in os.environ.items() if k not in ("SLURM_JOB_ID", "SLURM_JOBID")}


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, env=ENV_CLEAN)


def locked(fn):
    with open(LOCK, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        tasks = [json.loads(l) for l in Q.read_text().splitlines() if l.strip()]
        out = fn(tasks)
        tmp = Q.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(t) + "\n" for t in tasks))
        tmp.replace(Q)
        return out


def slurm_state(jid):
    r = sh("squeue", "-h", "-j", str(jid), "-o", "%T")
    return r.stdout.strip() or "GONE"


def take(gpu, minutes_left):
    def fn(tasks):
        for t in tasks:
            if t["state"] != "todo" or t["est_min"] > minutes_left:
                continue
            if t.get("slurm_id"):
                if slurm_state(t["slurm_id"]) != "PENDING":  # started or finished in the normal queue meanwhile
                    t["state"] = "skipped"
                    continue
                sh("scontrol", "hold", str(t["slurm_id"]))
            t.update(state="running", gpu=gpu, start=time.strftime("%Y-%m-%d %H:%M"))
            return dict(t)
        return None
    return locked(fn)


def finish(tid, state, rc=None):
    def fn(tasks):
        for t in tasks:
            if t["id"] == tid:
                t.update(state=state, rc=rc, end=time.strftime("%Y-%m-%d %H:%M"))
                if t.get("slurm_id"):
                    sh(*(["scancel", str(t["slurm_id"])] if state == "done" else ["scontrol", "release", str(t["slurm_id"])]))
    locked(fn)


def work(a):
    job, cur = os.environ.get("SLURM_JOB_ID", "local"), {}

    def stop(*_):  # dev job cancelled or out of time: give the task back to the normal queue
        if cur.get("proc") and cur["proc"].poll() is None:
            os.killpg(cur["proc"].pid, signal.SIGTERM)
        if cur.get("id") is not None:
            finish(cur["id"], "todo")
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    bad = 0
    while True:
        if bad >= 3:  # e.g. the node cannot run the environment: do not burn the whole list
            print(f"[gpu {a.gpu}] 3 failures in a row, stopping", flush=True)
            return
        t = take(a.gpu, (a.end - time.time()) / 60)
        if t is None:
            print(f"[gpu {a.gpu}] nothing left that fits", flush=True)
            return
        env = dict(os.environ, **{k: str(v) for k, v in t["env"].items()}, CUDA_VISIBLE_DEVICES=str(a.gpu),
                   BASE_PORT=str(22000 + 100 * a.gpu), FAWAM_VLM_PORT=str(8010 + a.gpu))
        log = Path(t["log"].replace("%j", f"dev{job}g{a.gpu}"))
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"[gpu {a.gpu}] {time.strftime('%H:%M')} start {t['label']} -> {log}", flush=True)
        with open(log, "w") as f:
            cur.update(id=t["id"], proc=subprocess.Popen(["bash", t["script"]], env=env, stdout=f, stderr=subprocess.STDOUT,
                                                         cwd=ROOT, start_new_session=True))
            rc = cur["proc"].wait()
        cur.clear()
        finish(t["id"], "done" if rc == 0 else "failed", rc)
        bad = 0 if rc == 0 else bad + 1
        print(f"[gpu {a.gpu}] {time.strftime('%H:%M')} rc={rc} {t['label']}", flush=True)


def status(_):
    from collections import Counter
    tasks = [json.loads(l) for l in Q.read_text().splitlines() if l.strip()]
    print(dict(Counter(t["state"] for t in tasks)))
    for t in tasks:
        if t["state"] in ("running", "failed"):
            print(t["state"], t.get("gpu"), t.get("start"), t["label"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("work"); w.add_argument("--gpu", type=int, required=True); w.add_argument("--end", type=float, required=True)
    sub.add_parser("status")
    a = ap.parse_args()
    dict(work=work, status=status)[a.cmd](a)
