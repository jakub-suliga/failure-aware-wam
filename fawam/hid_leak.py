"""Leak test for the hid data (plan 2026-11-18, section 3.5). Cross-validated ridge probes predicting the hidden
offset o from
  first  the FIRST frame of an attempt (latents of all 3 cameras, average-pooled, + the first commanded EE pose):
         must be ~0 (R² < 0.1), otherwise o is visible without a failed attempt and the data cannot teach ICL
  window the failure window of the failed attempt (C3 features: real − imagined wrist latents + proprio at the close):
         should be clearly > 0 (goal R² > 0.5), otherwise the information is not in the context
Only policy attempts (driver policy) with o != 0 of one offset type are used; R² per component of o.

  python -m fawam.hid_leak results/sweeps/p10_hid_data/base/hid_trans [more roots]
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from fawam import hid_ctx


def first_frame(ep):
    with h5py.File(Path(ep) / "episode.h5", "r") as f:
        feats = []
        for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            lat = f[f"model/obs/latents/{cam}"][0].astype(np.float32)  # (48, h, w), latent 0 = first frame
            feats.append(lat.reshape(48, 2, lat.shape[1] // 2, 2, lat.shape[2] // 2).mean((2, 4)).ravel())
        feats.append(f["actions/executed_ee_action"][0].astype(np.float32))
    return np.concatenate(feats)


def window(ep):
    d = hid_ctx.extract(ep)
    w = d["win_diff"].astype(np.float32)  # (7, 2, 48, 8, 10)
    pooled = w.reshape(7, 2, 48, 2, 4, 2, 5).mean((4, 6)).ravel()
    return np.concatenate([pooled, d["close"]])


def cv_r2(X, Y, folds=5, alphas=(1e-2, 1e-1, 1, 10, 100), dims=(5, 20, 50)):
    """R² per output from out-of-fold predictions: standardise, PCA (fitted on the training fold), ridge; best
    (dims, alpha) by mean R². PCA keeps the probe usable when there are more features than episodes."""
    n = len(X)
    idx = np.random.default_rng(0).permutation(n)
    best = None
    for d in dims:
        for a in alphas:
            pred = np.zeros_like(Y)
            for k in range(folds):
                te = idx[k::folds]
                tr = np.setdiff1d(idx, te)
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
                Xt, Xe = (X[tr] - mu) / sd, (X[te] - mu) / sd
                _, _, Vt = np.linalg.svd(Xt, full_matrices=False)
                P = Vt[:min(d, len(tr) - 1)].T
                Zt, Ze = Xt @ P, Xe @ P
                ym = Y[tr].mean(0)
                W = np.linalg.solve(Zt.T @ Zt + a * np.eye(Zt.shape[1]), Zt.T @ (Y[tr] - ym))
                pred[te] = Ze @ W + ym
            r2 = 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
            if best is None or r2.mean() > best[1].mean():
                best = ((d, a), r2)
    return best


def main(roots):
    eps, ys = [], []
    for root in roots:
        for h5p in sorted(Path(root).glob("episodes/*/*/*/episode.h5")):
            with h5py.File(h5p, "r") as f:
                if f.attrs.get("driver") != "policy":
                    continue
                h = json.loads(f.attrs.get("hid", "null") or "null")
            lab = hid_ctx.labels(h5p.parent) if h else None
            if lab is None or not np.any(lab[0]):
                continue
            eps.append(h5p.parent)
            ys.append(lab[0])
    Y = np.array(ys)
    keep = Y.std(0) > 0
    print(f"{len(eps)} policy attempts with o != 0")
    if len(eps) < 20:
        print("too few episodes for a probe")
        return
    out = dict(n=len(eps))
    for name, fn, goal in (("first frame", first_frame, "< 0.1"), ("failure window", window, "> 0.5")):
        X = np.stack([fn(e) for e in eps])
        a, r2 = cv_r2(X, Y[:, keep])
        comp = np.array(["x", "y", "z"])[keep]
        print(f"{name}: R² " + ", ".join(f"{c} {v:.2f}" for c, v in zip(comp, r2)) + f"  (pca {a[0]}, alpha {a[1]:g}, goal {goal})")
        out[name] = {c: round(float(v), 3) for c, v in zip(comp, r2)}
    Path(roots[0], "leak_test.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1:])
