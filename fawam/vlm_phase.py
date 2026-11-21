"""p20 D7 (plan v2, 2026-11-21): does a VLM recognise the task phases? Offline only.

Qwen3-VL-32B labels one frame per latent frame (head camera + both wrist cameras side by side, plus the previous key
frame for motion) with one of approach | lower | close | lift | transport | release (fawam/phases.py definitions in the
prompt). Accuracy against the oracle phases of D6 (episode/phases.json); a label that matches the oracle phase of a
neighbouring latent frame (+-1) at a phase transition counts as correct.

  python -m fawam.vlm_phase run --endpoint http://127.0.0.1:PORT/v1 --model Qwen/Qwen3-VL-32B-Instruct <roots>
  python -m fawam.vlm_phase summary
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from fawam.paths import ROOT
from fawam.phases import PHASES, latent_phases, phases_of
from fawam.vlm_bench import chat, hcat, jpeg, load

OUT = ROOT / "results" / "p20_d7"
DEF = ("approach: the gripper moves towards the object, still clearly above it; lower: the open gripper descends onto "
       "the object (within a few cm); close: the gripper is closing on / closed at the object, object not yet "
       "moving; lift: the closed gripper raises the object (or itself); transport: the object is carried or pushed "
       "to its goal; release: the gripper opens and lets go of the object.")


def ask(ep, li, frames, prev, instr):
    parts = [jpeg(hcat([prev[c] for c in ("cam_high", "cam_left_wrist", "cam_right_wrist")])),
             jpeg(hcat([frames[c] for c in ("cam_high", "cam_left_wrist", "cam_right_wrist")])),
             {"type": "text", "text": f"Robot task: {instr}\nThe first image is an earlier moment, the second image is "
                                      f"NOW (each: head camera, left wrist camera, right wrist camera). Phases: {DEF}\n"
                                      f"Which phase is the robot in NOW? Answer with exactly one word from: "
                                      f"{', '.join(PHASES)}."}]
    return parts


def run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "labels.jsonl"
    done = {(r["ep"], r["li"]) for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    eps = []
    for root in a.roots:
        eps += [p.parent for p in sorted(Path(root).glob("**/episodes/*/*/*/episode.h5")) if "_attempt" not in p.parent.name]
    import random
    random.Random(0).shuffle(eps)  # spread over settings
    eps = eps[:a.limit]
    print(f"{len(eps)} episodes", flush=True)
    items = []
    for ep in eps:
        ph = json.loads((ep / "phases.json").read_text()) if (ep / "phases.json").exists() else phases_of(ep / "episode.h5")
        lp = latent_phases(ph)
        meta = json.loads((ep / "meta.json").read_text()) if (ep / "meta.json").exists() else {}
        vids = None
        fli = np.array(ph["latent_index"])
        for li in sorted(lp):
            if li < 1 or (str(ep), li) in done:
                continue
            if vids is None:
                vids = load(ep)
            idx = int(np.where(fli == li)[0][-1])
            pidx = max(0, idx - 4)
            items.append((ep, li, {c: v[min(idx, len(v) - 1)] for c, v in vids.items()},
                          {c: v[min(pidx, len(v) - 1)] for c, v in vids.items()}, meta.get("instruction", ""), lp))

    def one(it):
        ep, li, fr, pv, instr, lp = it
        t0 = time.time()
        rec = dict(ep=str(ep), li=int(li), oracle=lp[li], oracle_prev=lp.get(li - 1), oracle_next=lp.get(li + 1))
        try:
            txt = chat(a.endpoint, a.model, ask(ep, li, fr, pv, instr), 20)
            m = re.search("|".join(PHASES), txt.lower())
            rec.update(text=txt.strip()[:80], vlm=m.group(0) if m else None)
        except Exception as e:
            rec["error"] = repr(e)[:200]
        rec["seconds"] = round(time.time() - t0, 2)
        return rec

    with ThreadPoolExecutor(a.workers) as ex, open(path, "a") as fh:
        for i, rec in enumerate(ex.map(one, items)):
            fh.write(json.dumps(rec) + "\n")
            if i % 50 == 0:
                print(i, rec, flush=True)
    summary(a)


def summary(a=None):
    recs = [json.loads(l) for l in (OUT / "labels.jsonl").read_text().splitlines() if l.strip()]
    ok = [r for r in recs if r.get("vlm")]
    exact = [r["vlm"] == r["oracle"] for r in ok]
    tol = [r["vlm"] in (r["oracle"], r.get("oracle_prev"), r.get("oracle_next")) for r in ok]
    conf = {o: {v: sum(1 for r in ok if r["oracle"] == o and r["vlm"] == v) for v in PHASES} for o in PHASES}
    per = {o: (sum(r["vlm"] == o for r in ok if r["oracle"] == o), sum(r["oracle"] == o for r in ok)) for o in PHASES}
    s = dict(n=len(recs), answered=len(ok), acc_exact=float(np.mean(exact)) if ok else None,
             acc_tol1=float(np.mean(tol)) if ok else None, per_phase=per, confusion=conf)
    (OUT / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: s[k] for k in ("n", "answered", "acc_exact", "acc_tol1", "per_phase")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("roots", nargs="+")
    r.add_argument("--endpoint", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--limit", type=int, default=60)
    r.add_argument("--workers", type=int, default=8)
    sub.add_parser("summary")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else summary(a)


if __name__ == "__main__":
    main()
