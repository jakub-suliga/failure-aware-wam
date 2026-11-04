"""Imagination pose probe: where does the model *imagine* the end effectors? (signal I in docs/METHOD.md)

A ridge regression from the pooled hidden states of a latent frame (recorded for observed AND imagined frames, same
forward pass at t=0) to the end-effector positions of both arms. It is trained on observed frames with proprioceptive
labels (ee pose = robot state, available on a real robot too; no object information). Applied to the imagined frames
of a failed chunk it turns the prediction error into a metric correction:

    delta = probe(imagined frame at the failure) - probe(observed frame at the failure)

(both through the probe, so its bias cancels). If the model imagined the gripper at the object but executed a grasp
8 cm higher, delta_z ~ -8 cm: the correction comes from the model's own imagination, not from a VLM or the simulator.

  python -m fawam.pose_probe fit results/table_height results/eval_hard --out artifacts/probe/pose_probe.npz
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

from fawam.latent_data import find_episodes
from fawam.paths import ROOT

OUT = ROOT / "artifacts" / "probe" / "pose_probe.npz"


def features(hidden, layer_index):
    """hidden (T, L, cams, D) -> (T, cams * D)"""
    h = hidden[:, layer_index].astype(np.float32)
    return h.reshape(len(h), -1)


def labels_at_latents(f, latent_index):
    """Both arms' ee xyz at the end of each latent frame (proprioception)."""
    li, es = f["frames/latent_index"][()], f["frames/env_step"][()]
    end_step = {}
    for k, s in zip(li, es):
        end_step[int(k)] = int(s)
    st = f["sim/env_step"][()]
    out = []
    for k in latent_index:
        j = min(int(np.searchsorted(st, end_step.get(int(k), st[-1]))), len(st) - 1)
        out.append(np.concatenate([f["sim/left_ee_pose"][j][:3], f["sim/right_ee_pose"][j][:3]]))
    return np.array(out, np.float32)


def collect(eps, layer_index):
    X, Y, G = [], [], []
    for gi, e in enumerate(eps):
        with h5py.File(e / "episode.h5", "r") as f:
            if "model/obs/hidden" not in f:
                continue
            li = f["model/obs/latent_index"][()]
            keep = li >= 1
            X.append(features(f["model/obs/hidden"][()], layer_index)[keep])
            Y.append(labels_at_latents(f, li[keep]))
            G.append(np.full(keep.sum(), gi))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)


def ridge(X, Y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d, dtype=X.dtype), X.T @ Y)


def fit(a):
    eps = find_episodes(a.roots)
    X, Y, G = collect(eps, a.layer_index)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    ym = Y.mean(0)
    rng = np.random.default_rng(0)
    groups = np.unique(G)
    test = np.isin(G, rng.choice(groups, size=max(1, len(groups) // 5), replace=False))  # split by episode
    best = None
    for lam in (1e1, 1e2, 1e3, 1e4, 1e5):
        W = ridge(Z[~test], Y[~test] - ym, lam)
        err = np.abs(Z[test] @ W + ym - Y[test])
        r2 = 1 - ((Z[test] @ W + ym - Y[test]) ** 2).sum(0) / ((Y[test] - Y[test].mean(0)) ** 2).sum(0)
        print(f"lambda {lam:.0e}: held-out MAE per axis (cm) {np.round(err.mean(0) * 100, 2)}  R2 {np.round(r2, 3)}")
        if best is None or err[:, [2, 5]].mean() < best[0]:
            best = (err[:, [2, 5]].mean(), lam)
    W = ridge(Z, Y - ym, best[1])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, W=W.astype(np.float32), ym=ym, mean=mu, std=sd, layer_index=np.int64(a.layer_index),
             lam=best[1], mae_z_cm=best[0] * 100)
    print(f"{len(eps)} episodes, {len(X)} frames; lambda {best[1]:.0e}, held-out z MAE {best[0] * 100:.2f} cm -> {a.out}")


class PoseProbe:
    def __init__(self, path=OUT):
        d = np.load(path)
        self.W, self.ym, self.mu, self.sd = d["W"], d["ym"], d["mean"], d["std"]
        self.layer_index = int(d["layer_index"])

    def __call__(self, hidden):
        """hidden (T, L, cams, D) -> (T, 6) ee xyz [left, right]"""
        return ((features(hidden, self.layer_index) - self.mu) / self.sd) @ self.W + self.ym

    def imagination_delta(self, f, latent, arm):
        """probe(imagined) - probe(observed) at `latent` for `arm`: the correction implied by the imagination."""
        pi = {int(k): i for i, k in enumerate(f["model/pred/latent_index"][()])}
        oi = {int(k): i for i, k in enumerate(f["model/obs/latent_index"][()])}
        if latent not in pi or latent not in oi:
            return None
        p = self(f["model/pred/hidden"][pi[latent]][None])[0]
        o = self(f["model/obs/hidden"][oi[latent]][None])[0]
        sl = slice(0, 3) if arm == "left" else slice(3, 6)
        return (p - o)[sl]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fit")
    s.add_argument("roots", nargs="+")
    s.add_argument("--layer_index", type=int, default=1, help="index into recorded layers [9, 19, 29]")
    s.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    fit(a)


if __name__ == "__main__":
    main()
