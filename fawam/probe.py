"""Statistics, probes and steering vectors from recorded episodes (robotwin env is enough: numpy + h5py only).

  python -m fawam.probe gap-stats results/eval_hard results/table_height            -> artifacts/gap_stats.json
  python -m fawam.probe failure-probe results/eval_hard --labels results/eval_hard/analysis/episodes.csv
                                                                                      -> artifacts/probe/failure_probe.npz
  python -m fawam.probe steer results/eval_hard --labels ...                          -> artifacts/probe/steer.npz

gap-stats: per task and camera, mean/std of the chunk-level prediction gap (max cosine distance predicted vs.
  observed latent over the chunk's new frames) in successful episodes; the server z-scores its online gap with it.
failure-probe: logistic regression on the pooled hidden states of the *predicted* frames of a chunk (the server
  records them at the cache-commit pass), label 1 for chunks of failed episodes within `--horizon` latents before
  the failure. Used by best-of-N (candidate_score=probe) to rank imagined futures before acting.
steer: mean(success) - mean(pre-failure) of the same features per layer, unit-normalised and scaled to the standard
  deviation of the projection; added at inference with Method.steer_alpha (A6).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from fawam.latent_data import find_episodes
from fawam.paths import CAMS, ROOT

ART = ROOT / "artifacts"


def cos_rows(a, b):
    a, b = a.reshape(len(a), -1).astype(np.float32), b.reshape(len(b), -1).astype(np.float32)
    return 1 - (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)


def chunk_gaps(f):
    """{cam: [gap per chunk]} with the server's definition (max over the chunk's new frames)."""
    p_li = f["model/pred/latent_index"][()]
    o_li = {int(k): i for i, k in enumerate(f["model/obs/latent_index"][()])}
    p_chunk = f["model/pred/chunk_id"][()]
    out = {}
    for cam in CAMS:
        P, O = f[f"model/pred/latents/{cam}"], f[f"model/obs/latents/{cam}"]
        per = defaultdict(list)
        for i, k in enumerate(p_li):
            if k < 1 or int(k) not in o_li:
                continue
            per[int(p_chunk[i])].append(float(cos_rows(P[i][None], O[o_li[int(k)]][None])[0]))
        out[cam] = [max(v) for _, v in sorted(per.items())]
    return out


def gap_stats(a):
    vals = defaultdict(lambda: defaultdict(list))
    for e in find_episodes(a.roots, success=True):
        with h5py.File(e / "episode.h5", "r") as f:
            task = str(f.attrs["task_name"])
            for cam, g in chunk_gaps(f).items():
                vals[task][cam] += g
                vals["_all"][cam] += g
    out = {t: {c: dict(mean=float(np.mean(v)), std=float(np.std(v)), n=len(v)) for c, v in cams.items()}
           for t, cams in vals.items()}
    p = Path(a.out or ART / "gap_stats.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps({t: {c: round(v["mean"], 3) for c, v in cams.items()} for t, cams in out.items()}, indent=1))
    print(f"wrote {p}")


def load_labels(path):
    import csv
    lab = {}
    if path is None:
        return lab
    with open(path) as fh:
        for r in csv.DictReader(fh):
            fl = r.get("failure_latent")
            lab[r["path"].rstrip("/").split("/")[-1] + "|" + r["task"]] = float(fl) if fl not in (None, "", "nan") else None
    return lab


def chunk_features(f, layer_index, cams):
    """(n_chunks, len(cams) * D) pooled hidden of the predicted frames per chunk, and the chunk start latent."""
    hid = f["model/pred/hidden"][()].astype(np.float32)  # T, L, cams, D
    chunk = f["model/pred/chunk_id"][()]
    li = f["model/pred/latent_index"][()]
    feats, starts = [], []
    for c in np.unique(chunk):
        rows = np.where(chunk == c)[0]
        feats.append(hid[rows][:, layer_index, cams].mean(0).reshape(-1))
        starts.append(int(li[rows[0]]))
    return np.stack(feats), np.array(starts)


def dataset(a, layer_index, cams):
    labels = load_labels(a.labels)
    X, y, w = [], [], []
    for e in find_episodes(a.roots):
        with h5py.File(e / "episode.h5", "r") as f:
            if "model/pred/hidden" not in f:
                continue
            ok = bool(f.attrs["success"])
            feats, starts = chunk_features(f, layer_index, cams)
            if ok:
                X.append(feats)
                y.append(np.zeros(len(feats)))
                continue
            fl = labels.get(e.name + "|" + str(f.attrs["task_name"]))
            if fl is None:
                continue  # unlabeled failure: skip rather than guess the time
            sel = (starts <= fl) & (starts >= fl - a.horizon)
            if sel.any():
                X.append(feats[sel])
                y.append(np.ones(sel.sum()))
    X, y = np.concatenate(X), np.concatenate(y)
    return X, y


def fit_logreg(X, y, l2=1.0, iters=500, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    pos = y.mean()
    sw = np.where(y == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))  # class-balanced
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = (p - y) * sw
        w -= lr * (Z.T @ g / len(y) + l2 * w / len(y))
        b -= lr * g.mean()
    return w, b, mu, sd


def auc(scores, y):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    npos, nneg = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))


def failure_probe(a):
    cams = [CAMS.index(c) for c in a.cams]
    X, y = dataset(a, a.layer_index, cams)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    w, b, mu, sd = fit_logreg(X[idx[:cut]], y[idx[:cut]], l2=a.l2)
    te = idx[cut:]
    s = ((X[te] - mu) / sd) @ w + b
    print(f"chunks: {len(y)} (pos {int(y.sum())}), held-out AUC {auc(s, y[te]):.3f} (split by chunk, optimistic)")
    w, b, mu, sd = fit_logreg(X, y, l2=a.l2)
    p = Path(a.out or ART / "probe" / "failure_probe.npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, w=w.astype(np.float32), b=np.float32(b), mean=mu.astype(np.float32), std=sd.astype(np.float32),
             layer_index=np.int64(a.layer_index), cams=np.array(cams, dtype=np.int64))
    print(f"wrote {p}")


def steer(a):
    out = {}
    for li, layer in enumerate([9, 19, 29]):
        X, y = dataset(a, li, [0, 1, 2])
        D = X.shape[1] // 3
        X = X.reshape(len(X), 3, D).mean(1)  # camera-agnostic: the vector is added to every token
        d = X[y == 0].mean(0) - X[y == 1].mean(0)
        d /= np.linalg.norm(d) + 1e-8
        out[f"dir_L{layer}"] = (d * np.std(X @ d)).astype(np.float32)
        print(f"L{layer}: separation along direction = "
              f"{(X[y == 0] @ d).mean() - (X[y == 1] @ d).mean():.3f} (std {np.std(X @ d):.3f})")
    p = Path(a.out or ART / "probe" / "steer.npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, layer=np.int64(19), **out)
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gap-stats")
    g.add_argument("roots", nargs="+")
    g.add_argument("--out")
    for name in ("failure-probe", "steer"):
        s = sub.add_parser(name)
        s.add_argument("roots", nargs="+")
        s.add_argument("--labels", help="episodes.csv from analysis/analyze.py (failure_latent column)")
        s.add_argument("--horizon", type=int, default=4, help="latents before the failure labeled positive")
        s.add_argument("--layer_index", type=int, default=1, help="index into recorded layers [9, 19, 29]")
        s.add_argument("--cams", nargs="+", default=["cam_left_wrist", "cam_right_wrist"])
        s.add_argument("--l2", type=float, default=10.0)
        s.add_argument("--out")
    a = ap.parse_args()
    {"gap-stats": gap_stats, "failure-probe": failure_probe, "steer": steer}[a.cmd](a)


if __name__ == "__main__":
    main()
