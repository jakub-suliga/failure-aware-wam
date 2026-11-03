"""Render the action->vision attention recorded by attention_replay.py.

For the chunk being executed, the overlay (JET heatmap blended over the image, white contour at 80 % of the frame's max) shows the attention mass of the action tokens on the video tokens of the
last observed frame plus the chunk's two predicted frames (mean over layers 6-28, heads, action tokens and captured
denoising steps), upsampled from the 12 x 10 token grid onto each camera. Layers 0-5 and 29 are nearly uniform and
are left out. Attention sinks (token positions that are among the top 3 of a chunk in at least 60 % of the episode's chunks,
whatever the image shows -- often empty table) are replaced by their camera's median for display only and marked
with a small gray square; shares use raw mass.
Color scale: min..max per camera and chunk (relative: where the model looks most right now); the share curve
shows the absolute mass.

Writes to <attention dir>:
  videos/<id>.mp4      per camera: real | real + attention | predicted + attention on predicted frames; plus a
                       curve of the attention share per camera with the sim failure time
  strips/<id>.png      keyframes (every few chunks up to and past the failure) with the overlay, for reports
  share_aligned.png    attention share per camera (and on actions) over the episode, failure times marked

Run with the robotwin env:  python analysis/attention_render.py results/eval_hard
"""
import argparse
from pathlib import Path

import cv2
import h5py
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import CAMS, CELL_H, CELL_W, frame_chunks, label, mark_chunks, read_mp4, with_chunk_bar

TOKEN_SLICES = {  # T-shaped token grid (12 x 10): wrists side by side on top, head camera below
    "cam_high": (slice(4, None), slice(None)),
    "cam_left_wrist": (slice(0, 4), slice(0, 5)),
    "cam_right_wrist": (slice(0, 4), slice(5, None)),
}
LAYERS = slice(6, 29)
SINK = np.zeros((12, 10), bool)  # set in main()
SHORT = {"cam_high": "head", "cam_left_wrist": "left wrist", "cam_right_wrist": "right wrist"}


def load(path):
    f = h5py.File(path, "r")
    st = f["chunk_frame_st_id"][:]
    chunks = []
    for c in range(len(st)):
        frames = f[f"frames/{c}"][:]
        maps = f[f"maps/{c}"][LAYERS].astype(np.float32).mean(0)  # nF, 12, 10 (mean over layers)
        key = {(int(k), int(p)): i for i, (k, p) in enumerate(frames)}
        pred = sum(maps[key[(k, 1)]] for k in (st[c], st[c] + 1) if (k, 1) in key)
        last_obs = maps[key[(st[c] - 1, 0)]] if (st[c] - 1, 0) in key else np.zeros_like(pred)
        total = maps.sum(0)
        chunks.append(dict(pred=pred, obs=last_obs, total=total))
    return dict(attrs=dict(f.attrs), st=st, chunks=chunks, action_mass=f["action_mass"][:, LAYERS].mean(1))


def cam_map(grid, cam):
    hs, ws = TOKEN_SLICES[cam]
    return grid[hs, ws]


def disp_map(grid, cam):
    """Camera region with attention-sink tokens replaced by the region's median (display only)."""
    m, sink = cam_map(grid, cam).copy(), cam_map(SINK, cam)
    if sink.any():
        m[sink] = np.median(m[~sink])
    return m


def find_sinks(att, top=3, freq=0.6):
    count = np.zeros(12 * 10)
    for ch in att["chunks"]:
        count[np.argsort(ch["total"].ravel())[-top:]] += 1
    return (count / len(att["chunks"]) >= freq).reshape(12, 10)


def mark_sinks(img, cam):
    """Small gray square at the center of each masked sink token of this camera."""
    sink = cam_map(SINK, cam)
    th, tw = img.shape[0] / sink.shape[0], img.shape[1] / sink.shape[1]
    for r, c in np.argwhere(sink):
        cy, cx = int((r + 0.5) * th), int((c + 0.5) * tw)
        cv2.rectangle(img, (cx - 4, cy - 4), (cx + 4, cy + 4), (150, 150, 150), 1)
    return img


def overlay(img, m):
    """JET heatmap blended 50/50 over the image, normalized per camera and chunk (blue = least, red = most attended
    token of this camera right now); a thin white contour surrounds the top region (>= 80 %)."""
    a = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    a = np.clip((a - m.min()) / max(m.max() - m.min(), 1e-12), 0, 1)
    heat = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    out = cv2.addWeighted(img, 0.5, heat, 0.5, 0)
    cs, _ = cv2.findContours((a >= 0.8).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def shares(att):
    """Per chunk: share of the action tokens' attention on each camera's tokens (all frames) and on actions."""
    rows = []
    for c, ch in enumerate(att["chunks"]):
        r = dict(chunk=c, latent=int(att["st"][c]), actions=float(att["action_mass"][c]))
        for cam in CAMS:
            r[cam] = float(cam_map(ch["total"], cam).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def curve_panel(sh, failure_latent, width, height=240):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    mark_chunks(ax, sh.latent.values)
    for cam in CAMS:
        ax.plot(sh.latent, sh[cam], marker=".", label=SHORT[cam])
    ax.plot(sh.latent, sh.actions, marker=".", color="gray", label="action tokens")
    if failure_latent is not None:
        ax.axvline(failure_latent, color="red", ls="--", label="failure (sim)")
    ax.set_xlabel("latent frame (chunk start)")
    ax.set_ylabel("attention share")
    ax.legend(fontsize=7, loc="upper left", ncol=5)
    ax.set_xlim(0, max(sh.latent.max(), 1) + 2)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    img = img[:img.shape[0] // 2 * 2, :img.shape[1] // 2 * 2].copy()  # libx264 needs even sizes
    to_px = lambda x: int(round(ax.transData.transform((x, 0))[0]))
    plt.close(fig)
    return img, to_px


def chunk_of(st, k):
    return int(np.searchsorted(st, k, side="right") - 1)


def render(ep_dir, att, failure_latent, video_out, strip_out):
    real = {c: read_mp4(ep_dir / f"real_{c}.mp4") for c in CAMS}
    pred = {c: read_mp4(ep_dir / f"pred_{c}.mp4") for c in CAMS}
    frame_latent = h5py.File(ep_dir / "episode.h5", "r")["frames/latent_index"][:]
    n = min(len(real[c]) for c in CAMS)
    sh = shares(att)
    panel, to_px = curve_panel(sh, failure_latent, 3 * CELL_W)
    chunks = frame_chunks(frame_latent, att["st"])

    def tiles(f):
        k = int(frame_latent[min(f, len(frame_latent) - 1)])
        ch = att["chunks"][max(chunk_of(att["st"], k), 0)]
        rows = []
        for cam in CAMS:
            r = cv2.resize(real[cam][f], (CELL_W, CELL_H))
            p = cv2.resize(pred[cam][f], (CELL_W, CELL_H)) if f < len(pred[cam]) else np.zeros_like(r)
            a = mark_sinks(overlay(r.copy(), disp_map(ch["obs"] + ch["pred"], cam)), cam)
            ap = mark_sinks(overlay(p.copy(), disp_map(ch["pred"], cam)), cam)
            rows.append(np.hstack([label(r, f"real {SHORT[cam]}"), label(a, f"action attention (latent {k})"),
                                   label(ap, "predicted + attention")]))
        return k, rows

    with imageio.get_writer(video_out, fps=10, codec="libx264", macro_block_size=1,
                            ffmpeg_params=["-crf", "23", "-pix_fmt", "yuv420p"]) as w:
        for f in range(n):
            k, rows = tiles(f)
            curve = panel.copy()
            x = to_px(k)
            cv2.line(curve, (x, 0), (x, curve.shape[0] - 1), (0, 0, 0), 2)
            w.append_data(with_chunk_bar(np.vstack(rows + [curve]), *chunks[min(f, len(chunks) - 1)], len(att["st"])))

    # Keyframe strip: 6 frames evenly up to the failure (+ a little after), real + attention overlay per camera
    last = len(frame_latent) - 1
    end_k = min(failure_latent + 2, int(frame_latent[last])) if failure_latent is not None else int(frame_latent[last])
    ks = np.linspace(1, end_k, 6).round().astype(int)
    cols = []
    for k in ks:
        f = min(int(np.searchsorted(frame_latent, k, side="right") - 1), n - 1)
        ch = att["chunks"][max(chunk_of(att["st"], k), 0)]
        cells = []
        for cam in CAMS:
            r = cv2.resize(real[cam][f], (CELL_W, CELL_H))
            a = mark_sinks(overlay(r, disp_map(ch["obs"] + ch["pred"], cam)), cam)
            tag = f"t={k} chunk {chunk_of(att['st'], k) + 1}" + (" FAIL" if failure_latent is not None and k >= failure_latent else "")
            cells.append(label(a, f"{SHORT[cam]} {tag}"))
        cols.append(np.vstack(cells))
    imageio.imwrite(strip_out, np.hstack(cols))
    return sh


def plot_share_aligned(all_sh, out):
    """Attention share per camera (and on action tokens) over the episode; fails red with their failure time marked
    (x), successes green. Successes end early (task done), fails run to the step limit."""
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, len(CAMS) + 1, figsize=(16, 3.6))
    styles = dict(zip(sorted(all_sh.task.unique()), ["-", "--", ":", "-."]))
    for ax, col in zip(axes, CAMS + ["actions"]):
        for (task, success, ep), d in all_sh.groupby(["task", "success", "episode"]):
            ls = styles[task]
            ax.plot(d.latent, d[col], color=("tab:green" if success else "tab:red"), alpha=0.6, ls=ls, lw=1.2)
            if not success:
                fl = d.failure_latent.iloc[0]
                at = d.iloc[(d.latent - fl).abs().argmin()]
                ax.plot(at.latent, at[col], "kx", ms=7)
        ax.set_title(SHORT.get(col, "action tokens"))
        ax.set_xlabel("latent frame")
    axes[0].set_ylabel("attention share of action tokens")
    axes[-1].legend([Line2D([], [], color="tab:red"), Line2D([], [], color="tab:green")]
                    + [Line2D([], [], color="k", ls=ls) for ls in styles.values()]
                    + [Line2D([], [], color="k", marker="x", ls="")],
                    ["fail", "success", *styles, "failure (sim)"], fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    args = ap.parse_args()
    att_dir = args.results / "analysis" / "attention"
    (att_dir / "videos").mkdir(exist_ok=True)
    (att_dir / "strips").mkdir(exist_ok=True)
    episodes = pd.read_csv(args.results / "analysis" / "episodes.csv").set_index("episode")
    paths = sorted(att_dir.glob("*.h5"))
    atts = {p.stem: load(p) for p in paths}
    all_sh = []
    for eid, att in atts.items():
        SINK[:] = find_sinks(att)
        print(eid, "attention sinks (token row, col):", [tuple(map(int, x)) for x in np.argwhere(SINK)])
        row = episodes.loc[eid]
        ep_dir = args.results / "episodes" / ("success" if row.success else "fail") / row.task / eid.split("__")[1]
        fl = None if pd.isna(row.failure_latent) else int(row.failure_latent)
        sh = render(ep_dir, att, fl, att_dir / "videos" / f"{eid}.mp4", att_dir / "strips" / f"{eid}.png")
        sh["episode"], sh["task"], sh["success"], sh["failure_latent"] = eid, row.task, bool(row.success), fl
        all_sh.append(sh)
        print("rendered", eid)
    all_sh = pd.concat(all_sh, ignore_index=True)
    all_sh.to_csv(att_dir / "attention_share.csv", index=False)
    plot_share_aligned(all_sh, att_dir / "share_aligned.png")


if __name__ == "__main__":
    main()
