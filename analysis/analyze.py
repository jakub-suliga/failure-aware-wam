"""Build failure-analysis artifacts from recorded LingBot-VA RoboTwin episodes.

Reads <results>/episodes/{success,fail}/<task>/<episode>/episode.h5 (+ real_*.mp4, pred_*.mp4) and writes to
<results>/analysis/ (or --out):

  episodes.parquet / episodes.csv   one row per episode: task, success, failure type + time, mismatch summaries
  events.csv                        all detected grasp events (also recovered ones in successful episodes)
  latent_metrics.parquet            long table: episode, latent_index, camera, metric, value
  mismatch_aligned.png              mismatch curves aligned to the failure time (t = 0), successes as reference
  roc.png / roc_auc.csv             success vs. failure ROC per metric and camera (episode score = max mismatch)
  tsne.png                          t-SNE of per-episode mismatch features, colored by failure type
  videos/<id>.mp4, curves/<id>.png  per episode: real | predicted | mismatch heatmap, with the mismatch curve
  index.html                        episode viewer (serve the folder: python -m http.server)

Run with the robotwin env:  ~/miniconda3/envs/robotwin/bin/python analysis/analyze.py results/eval
"""
import argparse
import json
from pathlib import Path

import cv2
import h5py
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CAMS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
CELL_H, CELL_W = 240, 320

# Failure labeling thresholds (env steps, meters, normalized gripper command: 1 = open, 0 = closed)
GRIPPER_CLOSED = 0.5
CONTACT_EPS = 1e-6
MISS_WINDOW = 8  # a closing gripper must touch an object within this many steps
HOLD_MIN = 5  # minimum grasp duration before a contact loss can count as a drop
DROP_WINDOW = 15
DROP_Z = 0.03


# ----------------------------------------------------------------------------------------------- loading

def load_episode(ep_dir):
    with h5py.File(ep_dir / "episode.h5", "r") as f:
        meta = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        ep = dict(dir=ep_dir, meta=meta,
                  id=f"{meta['task_name']}__{ep_dir.name}",
                  frames={k: f["frames"][k][()] for k in f["frames"]},
                  sim={k: f["sim"][k][()] for k in f["sim"]},
                  object_names=list(f["sim"].attrs["object_names"]))
        m = f["model"]
        for name in ("pred", "obs"):
            g = m[name]
            ep[name] = dict(latent_index=g["latent_index"][()],
                            latents={c: g["latents"][c][()].astype(np.float32) for c in CAMS},
                            hidden=g["hidden"][()].astype(np.float32) if "hidden" in g else None)
        ep["hidden_layers"] = list(m.attrs["hidden_layers"])
        ep["chunk_frame_st_id"] = m["chunks"]["frame_st_id"][()]
        if "denoise" in m:
            d = m["denoise"]
            ep["denoise"] = dict(step=d["step"][()],
                                 x0_hat={c: d["x0_hat"][c][()].astype(np.float32) for c in CAMS})
    ep["success"] = bool(ep["meta"]["success"])
    return ep


def find_episodes(root):
    return sorted(p.parent for p in (root / "episodes").glob("*/*/*/episode.h5") if p.parts[-4] != "_running")


# ----------------------------------------------------------------------------------------------- failure labels

def runs(mask):
    """Start/end (exclusive) index pairs of True runs."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def label_failures(ep):
    """Grasp events from sim ground truth; failure type/time for failed episodes.

    missed_grasp: gripper closes but touches no object within MISS_WINDOW steps
    dropped:      object held >= HOLD_MIN steps, contact ends while the gripper is still closed, object falls DROP_Z
    misplaced:    failed episode without drop/miss after an object was grasped (time = last release)
    timeout:      failed episode where no object was ever grasped
    The failure time of a failed episode is the first drop, else the first missed grasp, else as above.
    """
    sim, steps = ep["sim"], ep["sim"]["env_step"]
    z = sim["object_pose"][:, :, 2]
    events, releases, ever_grasped = [], [], False
    for arm in ("left", "right"):
        closed = sim[f"{arm}_gripper"] < GRIPPER_CLOSED
        contact = sim[f"contact_{arm}_gripper"] > CONTACT_EPS  # T, objects
        touching = contact.any(axis=1)
        for t in np.where(closed[1:] & ~closed[:-1])[0] + 1:
            if not touching[t:t + MISS_WINDOW].any():
                events.append(dict(type="missed_grasp", arm=arm, step=int(steps[t]), object=None))
        for start, end in runs(touching & closed):
            ever_grasped = True
            obj = int(sim[f"contact_{arm}_gripper"][start:end].sum(axis=0).argmax())
            if end >= len(steps):
                continue
            if not closed[end]:
                releases.append(int(steps[end]))
            elif end - start >= HOLD_MIN and z[end:end + DROP_WINDOW, obj].min() < z[end - 1, obj] - DROP_Z:
                events.append(dict(type="dropped", arm=arm, step=int(steps[end]), object=ep["object_names"][obj]))
    events.sort(key=lambda e: e["step"])

    if ep["success"]:
        return events, "none", None
    for kind in ("dropped", "missed_grasp"):
        hit = [e for e in events if e["type"] == kind]
        if hit:
            return events, kind, hit[0]["step"]
    if ever_grasped:
        return events, "misplaced", max(releases) if releases else int(steps[-1])
    return events, "timeout", int(steps[-1])


# ----------------------------------------------------------------------------------------------- mismatch

def latent_env_step(ep):
    """Env step at the end of each latent frame (latent k >= 1 covers raw frames 4k-3..4k)."""
    fr = ep["frames"]
    out = {}
    for k, s in zip(fr["latent_index"], fr["env_step"]):
        out[int(k)] = int(s)
    return out


def cos_dist(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return 1 - (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)


def compute_metrics(ep):
    """Per latent frame (k >= 1, observed) and camera: MSE/cosine of predicted vs. encoded real latents, MSE of the
    intermediate x0 estimates, cosine distance of pooled hidden states. Also per-position MSE maps for heatmaps."""
    pred_idx = {int(k): i for i, k in enumerate(ep["pred"]["latent_index"])}
    obs_idx = {int(k): i for i, k in enumerate(ep["obs"]["latent_index"])}
    common = sorted(k for k in pred_idx if k in obs_idx and k >= 1)
    pi, oi = [pred_idx[k] for k in common], [obs_idx[k] for k in common]
    rows, maps = [], {}
    for ci, cam in enumerate(CAMS):
        p, o = ep["pred"]["latents"][cam][pi], ep["obs"]["latents"][cam][oi]  # n, C, h, w
        sq = (p - o) ** 2
        maps[cam] = dict(zip(common, sq.mean(axis=1)))  # k -> (h, w)
        vals = {"mse": sq.mean(axis=(1, 2, 3)), "cos": cos_dist(p, o)}
        if ep["pred"]["hidden"] is not None and ep["obs"]["hidden"] is not None:
            hp, ho = ep["pred"]["hidden"][pi, :, ci], ep["obs"]["hidden"][oi, :, ci]  # n, L, D
            for li, layer in enumerate(ep["hidden_layers"]):
                vals[f"hid_cos_L{layer}"] = cos_dist(hp[:, li], ho[:, li])
        if "denoise" in ep:
            x0 = ep["denoise"]["x0_hat"][cam]  # chunks, steps, C, F, h, w
            for si, step in enumerate(ep["denoise"]["step"]):
                per_k = {}
                for c, st in enumerate(ep["chunk_frame_st_id"]):
                    for j in range(x0.shape[3]):
                        per_k[int(st) + j] = x0[c, si, :, j]
                vals[f"mse_x0hat_s{step}"] = np.array(
                    [((per_k[k] - ep["obs"]["latents"][cam][obs_idx[k]]) ** 2).mean() if k in per_k else np.nan
                     for k in common])
        for metric, v in vals.items():
            rows += [dict(episode=ep["id"], latent_index=k, camera=cam, metric=metric, value=float(x))
                     for k, x in zip(common, v)]
    return pd.DataFrame(rows), maps


def add_zscores(lat, episodes):
    """z_mse: MSE standardized per task and camera with the latent-level MSE of successful episodes."""
    info = episodes.set_index("episode")[["task", "success"]]
    mse = lat[lat.metric == "mse"].join(info, on="episode")
    out = []
    for (task, cam), g in mse.groupby(["task", "camera"]):
        ref = g[g.success]["value"] if g.success.sum() >= 2 else g["value"]
        z = g.copy()
        z["value"] = (g["value"] - ref.mean()) / (ref.std() + 1e-8)
        z["metric"] = "z_mse"
        out.append(z[lat.columns])
    return pd.concat([lat] + out, ignore_index=True)


# ----------------------------------------------------------------------------------------------- plots

def plot_aligned(lat, episodes, out, metric="z_mse", window=(-12, 6)):
    fig, axes = plt.subplots(1, len(CAMS), figsize=(5 * len(CAMS), 4), sharey=True)
    fails = episodes[~episodes.success & episodes.failure_latent.notna()]
    for ax, cam in zip(axes, CAMS):
        d = lat[(lat.metric == metric) & (lat.camera == cam)]
        succ_vals = d[d.episode.isin(episodes[episodes.success].episode)]["value"]
        if len(succ_vals):
            m, s = succ_vals.mean(), succ_vals.std()
            ax.axhspan(m - s, m + s, color="tab:green", alpha=0.15, label="success mean ± std")
            ax.axhline(m, color="tab:green", lw=1)
        aligned = []
        for _, ep in fails.iterrows():
            e = d[d.episode == ep.episode]
            aligned.append(pd.Series(e["value"].values, index=e["latent_index"].values - int(ep.failure_latent)))
        if aligned:
            a = pd.concat(aligned, axis=1).sort_index().loc[window[0]:window[1]]
            mean, std, n = a.mean(axis=1), a.std(axis=1).fillna(0), a.count(axis=1)
            ax.plot(a.index, mean, color="tab:red", label=f"failures (n={len(aligned)})")
            ax.fill_between(a.index, mean - std, mean + std, color="tab:red", alpha=0.2)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_title(cam)
        ax.set_xlabel("latent frames relative to failure (1 latent = 16 actions)")
    axes[0].set_ylabel(metric)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_roc(lat, episodes, out_png, out_csv):
    from sklearn.metrics import roc_auc_score, roc_curve

    metrics = [m for m in ["mse", "cos", "z_mse"] + sorted(m for m in lat.metric.unique() if m.startswith("hid_"))
               if m in lat.metric.unique()]
    score = lat.groupby(["episode", "camera", "metric"])["value"].max().unstack(["camera", "metric"])
    y = (~episodes.set_index("episode").loc[score.index, "success"]).astype(int)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), squeeze=False)
    aucs = []
    for ax, metric in zip(axes[0], metrics):
        for cam in CAMS:
            s = score.get((cam, metric))
            if s is None or y.nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(y, s.fillna(s.min()))
            auc = roc_auc_score(y, s.fillna(s.min()))
            aucs.append(dict(metric=metric, camera=cam, auc=auc, n=len(y), n_fail=int(y.sum())))
            ax.plot(fpr, tpr, label=f"{cam} AUC={auc:.2f}")
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set_title(f"{metric} (episode max)")
        ax.set_xlabel("FPR")
        ax.legend(fontsize=7)
        if y.nunique() < 2:
            ax.text(0.5, 0.5, "needs successes and failures", ha="center")
    axes[0][0].set_ylabel("TPR")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    pd.DataFrame(aucs).to_csv(out_csv, index=False)


def plot_tsne(lat, episodes, out):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    feats = []
    for _, ep in episodes.iterrows():
        d = lat[(lat.episode == ep.episode) & lat.metric.isin(["mse", "cos", "z_mse"] +
                                                                [m for m in lat.metric.unique() if m.startswith("hid_")])]
        t_ref = ep.failure_latent if pd.notna(ep.failure_latent) else d.latent_index.max()
        row = {}
        for (cam, metric), g in d.groupby(["camera", "metric"]):
            g = g.sort_values("latent_index")
            near = g[(g.latent_index > t_ref - 4) & (g.latent_index <= t_ref)]["value"]
            row.update({f"{cam}/{metric}/max": g.value.max(), f"{cam}/{metric}/mean": g.value.mean(),
                        f"{cam}/{metric}/near_t": near.mean() if len(near) else g.value.iloc[-1]})
        feats.append(row)
    X = pd.DataFrame(feats).fillna(0).values
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(X) < 5:
        ax.text(0.5, 0.5, f"t-SNE needs >= 5 episodes (have {len(X)})", ha="center")
    else:
        emb = TSNE(n_components=2, perplexity=min(30, (len(X) - 1) / 3), init="pca", random_state=0) \
            .fit_transform(StandardScaler().fit_transform(X))
        for label in sorted(episodes.failure_type.unique()):
            m = (episodes.failure_type == label).values
            ax.scatter(emb[m, 0], emb[m, 1], label="success" if label == "none" else label, s=30)
        ax.legend(fontsize=8)
    ax.set_title("t-SNE of mismatch features (per episode)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- per-episode video

def curve_panel(lat_ep, failure_latent, width, height=260, metric="z_mse", chunk_st=None):
    """Mismatch curve per camera as an image, plus a function mapping latent index -> pixel x."""
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    mark_chunks(ax, chunk_st)
    for cam in CAMS:
        d = lat_ep[(lat_ep.metric == metric) & (lat_ep.camera == cam)].sort_values("latent_index")
        ax.plot(d.latent_index, d.value, label=cam, marker=".")
    if failure_latent is not None:
        ax.axvline(failure_latent, color="red", ls="--", label="failure (sim)")
    ax.set_xlabel("latent frame")
    ax.set_ylabel(metric)
    ax.legend(fontsize=7, loc="upper left", ncol=4)
    xmax = max(1, lat_ep.latent_index.max() if len(lat_ep) else 1)
    ax.set_xlim(0, xmax + 0.5)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    to_px = lambda x: int(round(ax.transData.transform((x, 0))[0]))
    plt.close(fig)
    return img, to_px


def mark_chunks(ax, chunk_st):
    """Light vertical lines where the policy generated a new action chunk (chunk start latent)."""
    for i, st in enumerate([] if chunk_st is None else chunk_st):
        ax.axvline(st - 0.5, color="0.8", lw=0.8, zorder=0, label="new action chunk" if i == 0 else None)


def frame_chunks(frame_latent, chunk_st):
    """Per video frame: action chunk executing (the one whose latents contain the frame's latent), frame index
    within that chunk and the chunk's length in frames. Frame 0 is the initial observation (chunk 0)."""
    c = np.clip(np.searchsorted(chunk_st, frame_latent, side="right") - 1, 0, None)
    first = {int(x): int(np.argmax(c == x)) for x in np.unique(c)}
    length = {int(x): int((c == x).sum()) for x in np.unique(c)}
    return [(int(x), i - first[int(x)], length[int(x)]) for i, x in enumerate(c)]


NEW_CHUNK_FRAMES = 3  # frames the "new action chunk" banner stays on


def chunk_bar(width, chunk, pos, length, n_chunks, height=30):
    """Status bar: which action chunk is executing and how far, highlighted right after it was generated."""
    new = pos < NEW_CHUNK_FRAMES and chunk > 0
    bar = np.full((height, width, 3), (255, 140, 0) if new else (40, 40, 40), np.uint8)
    done = int(width * (pos + 1) / max(length, 1))
    cv2.rectangle(bar, (0, height - 6), (done, height - 1), (255, 255, 255) if new else (0, 200, 255), -1)
    text = f"action chunk {chunk + 1}/{n_chunks}   frame {pos + 1}/{length}"
    if new:
        text = "NEW ACTION CHUNK generated  |  " + text
    cv2.putText(bar, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0) if new else (255, 255, 255), 1,
                cv2.LINE_AA)
    return bar, new


def with_chunk_bar(img, chunk, pos, length, n_chunks):
    bar, new = chunk_bar(img.shape[1], chunk, pos, length, n_chunks)
    out = np.vstack([bar, img])
    if new:
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (255, 140, 0), 4)
    return out


def read_mp4(path):
    return [f for f in imageio.get_reader(path)] if path.exists() else []


def label(img, text):
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return img


def render_video(ep, lat_ep, maps, failure_latent, heat_scale, video_out, curve_out):
    real = {c: read_mp4(ep["dir"] / f"real_{c}.mp4") for c in CAMS}
    pred = {c: read_mp4(ep["dir"] / f"pred_{c}.mp4") for c in CAMS}
    n = min(len(real[c]) for c in CAMS)
    width = 3 * CELL_W
    chunk_st = ep["chunk_frame_st_id"]
    panel, to_px = curve_panel(lat_ep, failure_latent, width, chunk_st=chunk_st)
    imageio.imwrite(curve_out, panel)
    frame_latent = ep["frames"]["latent_index"]
    chunks = frame_chunks(frame_latent, chunk_st)
    with imageio.get_writer(video_out, fps=10, codec="libx264", macro_block_size=1,
                            ffmpeg_params=["-crf", "23", "-pix_fmt", "yuv420p"]) as w:
        for f in range(n):
            k = int(frame_latent[f]) if f < len(frame_latent) else int(frame_latent[-1])
            rows = []
            for cam in CAMS:
                r = cv2.resize(real[cam][f], (CELL_W, CELL_H))
                p = cv2.resize(pred[cam][f], (CELL_W, CELL_H)) if f < len(pred[cam]) else np.zeros_like(r)
                m = maps[cam].get(k)
                if m is None:
                    overlay = r.copy()
                else:
                    heat = np.clip(cv2.resize(m, (CELL_W, CELL_H), interpolation=cv2.INTER_CUBIC) / heat_scale[cam],
                                   0, 1)
                    heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
                    overlay = cv2.addWeighted(r, 0.5, heat, 0.5, 0)
                rows.append(np.hstack([label(r, f"real {cam}"), label(p, "predicted"),
                                       label(overlay, f"mismatch (latent {k})")]))
            curve = panel.copy()
            x = to_px(k)
            cv2.line(curve, (x, 0), (x, curve.shape[0] - 1), (0, 0, 0), 2)
            frame = np.vstack(rows + [cv2.resize(curve, (width, curve.shape[0]))])
            w.append_data(with_chunk_bar(frame, *chunks[min(f, len(chunks) - 1)], len(chunk_st)))


# ----------------------------------------------------------------------------------------------- viewer

VIEWER = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Episode viewer</title>
<style>
body{font-family:sans-serif;margin:0;display:flex;height:100vh}
#side{width:360px;overflow:auto;border-right:1px solid #ccc;padding:8px;box-sizing:border-box}
#main{flex:1;overflow:auto;padding:8px}
.ep{padding:4px;cursor:pointer;border-bottom:1px solid #eee;font-size:13px}
.ep:hover,.ep.sel{background:#e8f0fe}.ok{color:#188038}.bad{color:#c5221f}
select{margin:2px 0;width:100%} video{max-width:100%} img{max-width:100%} td{padding:2px 8px;font-size:13px}
</style></head><body>
<div id="side"><select id="task"></select><select id="succ"></select><select id="ftype"></select>
<div id="count"></div><div id="list"></div></div>
<div id="main"><h3 id="title">select an episode</h3><video id="vid" controls></video><br><img id="curve">
<table id="meta"></table></div>
<script>
const EPS = __DATA__;
const opts = (id, vals, all) => { const s = document.getElementById(id);
  s.innerHTML = [`<option value="">${all}</option>`].concat(vals.map(v => `<option>${v}</option>`)).join(''); s.onchange = render; };
const uniq = k => [...new Set(EPS.map(e => String(e[k])))].sort();
opts('task', uniq('task'), 'all tasks'); opts('succ', ['true', 'false'], 'success + fail');
opts('ftype', uniq('failure_type'), 'all failure types');
function render() {
  const t = task.value, s = succ.value, f = ftype.value;
  const sel = EPS.filter(e => (!t || e.task == t) && (!s || String(e.success) == s) && (!f || e.failure_type == f));
  count.textContent = `${sel.length} episodes`;
  list.innerHTML = sel.map(e => `<div class="ep" data-id="${e.episode}"><b class="${e.success ? 'ok' : 'bad'}">
    ${e.success ? '✓' : '✗'}</b> ${e.episode}<br><small>${e.failure_type}${e.failure_step != null ? ' @ step ' + e.failure_step : ''}
    · max z ${Number(e.max_z_mse_cam_high).toFixed(1)}/${Number(e.max_z_mse_cam_left_wrist).toFixed(1)}/${Number(e.max_z_mse_cam_right_wrist).toFixed(1)}</small></div>`).join('');
  list.querySelectorAll('.ep').forEach(d => d.onclick = () => show(d.dataset.id, d));
}
function show(id, el) {
  document.querySelectorAll('.ep').forEach(d => d.classList.remove('sel')); el.classList.add('sel');
  const e = EPS.find(x => x.episode == id);
  title.textContent = id; vid.src = `videos/${id}.mp4`; curve.src = `curves/${id}.png`;
  meta.innerHTML = Object.entries(e).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
}
render();
</script></body></html>
"""


# ----------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="save_root of the eval runs (contains episodes/)")
    ap.add_argument("--out", type=Path, default=None, help="default: <results>/analysis")
    ap.add_argument("--no-videos", action="store_true")
    args = ap.parse_args()
    out = args.out or args.results / "analysis"
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "curves").mkdir(parents=True, exist_ok=True)

    ep_dirs = find_episodes(args.results)
    print(f"{len(ep_dirs)} episodes")
    rows, event_rows, lat_frames, maps_all, eps = [], [], [], {}, {}
    for d in ep_dirs:
        try:
            ep = load_episode(d)
        except KeyError as e:  # e.g. server finalize failed -> no /model group
            print(f"skip {d}: missing {e}")
            continue
        events, ftype, fstep = label_failures(ep)
        lat, maps = compute_metrics(ep)
        step_of = latent_env_step(ep)
        flat = None if fstep is None else min((k for k, s in step_of.items() if s >= fstep), default=max(step_of))
        m = ep["meta"]
        rows.append(dict(episode=ep["id"], task=m["task_name"], setting=m.get("setting"), seed=m["seed"],
                         instruction=m["instruction"], success=ep["success"], failure_type=ftype,
                         failure_step=fstep, failure_latent=flat, n_events=len(events),
                         env_steps=m["env_steps"], num_chunks=m["num_chunks"], path=str(d)))
        event_rows += [dict(episode=ep["id"], **e) for e in events]
        lat_frames.append(lat)
        maps_all[ep["id"]] = maps
        eps[ep["id"]] = ep

    episodes = pd.DataFrame(rows)
    lat = add_zscores(pd.concat(lat_frames, ignore_index=True), episodes)
    summary = lat[lat.metric.isin(["mse", "cos", "z_mse"])].groupby(["episode", "camera", "metric"])["value"] \
        .agg(["max", "mean"]).unstack(["camera", "metric"])
    summary.columns = [f"{agg}_{metric}_{cam}" for agg, cam, metric in summary.columns]
    episodes = episodes.join(summary, on="episode")
    episodes.to_parquet(out / "episodes.parquet", index=False)
    episodes.to_csv(out / "episodes.csv", index=False)
    lat.to_parquet(out / "latent_metrics.parquet", index=False)
    pd.DataFrame(event_rows).to_csv(out / "events.csv", index=False)

    plot_aligned(lat, episodes, out / "mismatch_aligned.png")
    plot_roc(lat, episodes, out / "roc.png", out / "roc_auc.csv")
    plot_tsne(lat, episodes, out / "tsne.png")

    # Heatmap scale per camera: 99th percentile of per-position MSE in successful episodes (all if none), so that
    # colors are comparable across episodes -- a well-aligned success should stay mostly dark.
    heat_scale = {}
    ref_ids = list(episodes[episodes.success].episode) or list(episodes.episode)
    for cam in CAMS:
        vals = np.concatenate([np.concatenate([v.ravel() for v in maps_all[i][cam].values()])
                               for i in ref_ids if maps_all[i][cam]] or [np.ones(1)])
        heat_scale[cam] = float(np.percentile(vals, 99)) or 1.0
    if not args.no_videos:
        for _, row in episodes.iterrows():
            ep = eps[row.episode]
            render_video(ep, lat[lat.episode == row.episode], maps_all[row.episode],
                         None if pd.isna(row.failure_latent) else int(row.failure_latent), heat_scale,
                         out / "videos" / f"{row.episode}.mp4", out / "curves" / f"{row.episode}.png")
            print("video", row.episode)

    data = json.loads(episodes.drop(columns=["path"]).to_json(orient="records"))
    (out / "index.html").write_text(VIEWER.replace("__DATA__", json.dumps(data)))
    print(episodes[["episode", "success", "failure_type", "failure_step"]].to_string(index=False))
    print(f"artifacts in {out}")


if __name__ == "__main__":
    main()
