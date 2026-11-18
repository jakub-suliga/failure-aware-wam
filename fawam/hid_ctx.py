"""Context features of a (failed) attempt for the hid experiments (plan 2026-11-18, section 4). Real signals only:

  win_diff   C2/C3: real − imagined wrist-camera latents, 4 latents before to 2 after the gripper closes
             (both wrist cameras; (7, 2, 48, 8, 10))
  win_real   C5:    the real wrist latents of the same window (ablation: does the prediction add anything?)
  close      C3:    proprioception at the close: commanded (= believed-frame) EE pose of the closing arm, measured
                    gripper width at and 2 latents after the close, close time / step limit  (12,)
  traj       C4:    the attempt's commanded EE positions of both arms + measured gripper widths, 16 time points (16, 8)

The close time comes from the robot's own gripper commands (real). Commanded EE poses are the believed frame
(measured − o), so the hidden offset is not readable from them directly. Simulator ground truth (o, object poses,
success flags except the one failure bit) is never part of these features.

Features are cached as ctx_feats.npz next to episode.h5.
"""
from pathlib import Path

import h5py
import numpy as np

from fawam.failure_analysis import first_close, step_to_latent

WIN = (4, 2)  # latents before / after the close
CAMS = ("cam_left_wrist", "cam_right_wrist")


def _latents(f, group, cam):
    li = f[f"model/{group}/latent_index"][()]
    return {int(k): i for i, k in enumerate(li)}, f[f"model/{group}/latents/{cam}"]


def extract(ep_dir, force=False):
    ep_dir = Path(ep_dir)
    cache = ep_dir / "ctx_feats.npz"
    if cache.exists() and not force:
        return dict(np.load(cache))
    with h5py.File(ep_dir / "episode.h5", "r") as f:
        step, arm = first_close(f)
        steps = f["actions/executed_env_step"][()]
        n_steps = int(f.attrs.get("step_limit", 400)) if "step_limit" in f.attrs else 400
        if step is None:  # never closed: use the middle of the episode
            step, arm = int(steps[len(steps) // 2]), "left"
        L = step_to_latent(f, step)
        win = list(range(L - WIN[0], L + WIN[1] + 1))
        diff = np.zeros((len(win), len(CAMS), 48, 8, 10), np.float16)
        real = np.zeros_like(diff)
        for c, cam in enumerate(CAMS):
            oi, o = _latents(f, "obs", cam)
            pi, p = _latents(f, "pred", cam)
            for w, k in enumerate(win):
                if k in oi:
                    real[w, c] = o[oi[k]]
                    if k in pi:
                        diff[w, c] = o[oi[k]].astype(np.float32) - p[pi[k]].astype(np.float32)
        cmd = f["actions/executed_ee_action"][()]  # commanded absolute EE (believed frame)
        i = int(np.clip(np.searchsorted(steps, step), 0, len(cmd) - 1))
        sl = slice(0, 7) if arm == "left" else slice(8, 15)
        sim_step = f["sim/env_step"][()]
        grip = f[f"sim/{arm}_gripper"][()]
        j0 = min(int(np.searchsorted(sim_step, step)), len(grip) - 1)
        j2 = min(j0 + 8 * 4, len(grip) - 1)  # ~2 latents later (4 frames per latent, ~8 env steps per frame)
        close = np.concatenate([cmd[i, sl], [grip[j0], grip[j2], step / n_steps, 1.0 if arm == "left" else -1.0,
                                             float(len(steps)) / n_steps]]).astype(np.float32)
        idx = np.linspace(0, len(cmd) - 1, 16).astype(int)
        gl, gr = f["sim/left_gripper"][()], f["sim/right_gripper"][()]
        sidx = np.clip(np.searchsorted(sim_step, steps[idx]), 0, len(gl) - 1)
        traj = np.concatenate([cmd[idx, 0:3], cmd[idx, 8:11], gl[sidx, None], gr[sidx, None]], axis=1).astype(np.float32)
    out = dict(win_diff=diff, win_real=real, close=close, traj=traj, close_latent=np.int64(L))
    np.savez(cache, **out)
    return out


def labels(ep_dir):
    """Training-only labels from the simulator: hidden offset as a 3-vector (trans: o; gain: (0, 0, g−1);
    yaw: (0, 0, deg/25)) -- the target of the +aux head, never an input."""
    import json
    with h5py.File(Path(ep_dir) / "episode.h5", "r") as f:
        h = json.loads(f.attrs.get("hid", "null") or "null")
    if not h:
        return None
    o = h["o"]
    v = {"trans": o, "gain": [0.0, 0.0, o[0] - 1.0], "yaw": [0.0, 0.0, o[0] / 25.0]}[h["type"]]
    return np.asarray(v, np.float32), h["type"]
