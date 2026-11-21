"""Learn-from-Failure update step (docs/METHOD.md): failed episode(s) with analysis.json -> a new adapter version.

Called by the RoboTwin client after a failed attempt (Method.learn, mode adapter|both), in the lingbot env:

  python -m fawam.learn --episodes <failed episode dirs ...> --out <adapter dir> [--init <previous adapter>]
         [--steps 150] [--gap_beta 2] [--no_relabel] [--video_in_window skip] [--lambda_video 1] [--lambda_action 1]
         [--success_episodes <dirs>]   (successful own episodes from the memory, trained without relabel)

Loss = LingBot-VA's flow-matching loss on the failed episode, with (i) the failure analysis' correction applied to
the actions around the failure (relabel), (ii) per-frame weights from the prediction error (gap_beta), (iii) video loss
on the observed frames (the world model learns what really happens) except, optionally, inside the corrected window.
"""
import argparse
import json
import time
from pathlib import Path

from fawam.paths import LINGBOT, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--success_episodes", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--init")
    ap.add_argument("--kind", default="dual", choices=["dual", "inside", "lora"])
    ap.add_argument("--inside_streams", nargs="+", default=["video", "action"])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--gap_beta", type=float, default=2.0)
    ap.add_argument("--no_relabel", action="store_true")
    ap.add_argument("--video_in_window", default="skip", choices=["real", "skip"])
    ap.add_argument("--lambda_video", type=float, default=1.0)
    ap.add_argument("--lambda_action", type=float, default=1.0)
    ap.add_argument("--action_clip", type=float, default=3.0,
                    help="normalized clip for relabeled targets (corrections can leave the training range)")
    ap.add_argument("--text_mode", default="instruction", choices=["instruction", "correction"])
    ap.add_argument("--gap_filter", type=float, default=0.0)
    ap.add_argument("--outside", default="none", choices=["none", "parallel", "sequential"])
    ap.add_argument("--outside_steps", type=int, default=100)
    ap.add_argument("--pref_beta", type=float, default=0.0)
    ap.add_argument("--lr", type=float, help="inside learning rate (default: train_adapter's)")
    ap.add_argument("--hinge_episode")
    ap.add_argument("--hinge_beta", type=float, default=0.0)
    ap.add_argument("--guard_episodes", nargs="*", default=[])
    ap.add_argument("--guard_tau", type=float, default=1.1)
    ap.add_argument("--model_path", default=str(LINGBOT / "checkpoints" / "lingbot-va-posttrain-robotwin"))
    a = ap.parse_args()
    t0 = time.time()
    eps = [str(Path(e).resolve()) for e in a.episodes + a.success_episodes]
    if a.gap_beta == 0 and a.no_relabel and a.text_mode == "instruction":
        pass  # pure self-supervised update on the failure (prediction-loss-only baseline)
    missing = [e for e in a.episodes if not (Path(e) / "analysis.json").exists() and not a.no_relabel]
    if missing:
        raise SystemExit(f"no analysis.json in {missing} (run fawam.failure_analysis first)")

    from fawam import text_cache
    import sys
    sys.argv = ["text_cache", "build", *eps, "--model_path", a.model_path]
    text_cache.main()

    from fawam import train_adapter
    sys.argv = ["train_adapter", "inside", "--episodes", *eps, "--out", a.out, "--kind",
                "inside" if a.kind == "dual" else a.kind, "--inside_streams", *a.inside_streams,
                "--steps", str(a.steps), "--gap_beta", str(a.gap_beta), "--video_in_window", a.video_in_window,
                "--lambda_video", str(a.lambda_video), "--lambda_action", str(a.lambda_action),
                "--text_mode", a.text_mode, "--max_frames", "24", "--model_path", a.model_path, "--log_every", "25"]
    if not a.no_relabel:
        sys.argv += ["--relabel", "--action_clip", str(a.action_clip)]
    if a.init:
        sys.argv += ["--init", a.init]
    sys.argv += ["--gap_filter", str(a.gap_filter), "--pref_beta", str(a.pref_beta)]
    if a.lr:
        sys.argv += ["--lr", str(a.lr)]
    if a.hinge_episode and a.hinge_beta:
        sys.argv += ["--hinge_episode", a.hinge_episode, "--hinge_beta", str(a.hinge_beta)]
    if a.guard_episodes:
        sys.argv += ["--guard_episodes", *a.guard_episodes, "--guard_tau", str(a.guard_tau)]
    if a.steps > 0:
        train_adapter.main()
    elif a.init:  # outside-only update: start from the previous adapter
        import shutil
        Path(a.out).mkdir(parents=True, exist_ok=True)
        for fn in ("adapter.pt", "adapter_config.json"):
            shutil.copy(Path(a.init) / fn, Path(a.out) / fn)
    if a.outside != "none":
        # second site (paper Sec. 3.3): residual MLP on the imagined latent, trained with L1 on (pred, obs) pairs
        have_inside = (Path(a.out) / "adapter.pt").exists()
        if a.outside == "sequential" and have_inside:
            from fawam import replay
            sys.argv = ["replay", "--episodes", *eps, "--adapter", a.out, "--model_path", a.model_path,
                        "--out", str(Path(a.out) / "pairs.npz")]
            replay.main()
            src = ["--pairs", str(Path(a.out) / "pairs.npz")]
        else:  # parallel: predictions recorded during the episodes (made by the adapter active at that time)
            src = ["--episodes", *eps]
        sys.argv = ["train_adapter", "outside", *src, "--steps_outside", str(a.outside_steps), "--out", a.out,
                    "--model_path", a.model_path]
        if have_inside:
            sys.argv += ["--init", a.out]
        elif a.init:
            sys.argv += ["--init", a.init]
        train_adapter.main()
    guard = Path(a.out) / "guard.json"
    accepted = json.loads(guard.read_text())["accepted"] if guard.exists() else True
    method = dict(name=f"lff_{Path(a.out).name}", adapter_path=str(Path(a.out).resolve()), accepted=accepted)
    (Path(a.out) / "method.json").write_text(json.dumps(method, indent=1))
    print(json.dumps(dict(adapter=str(Path(a.out).resolve()), accepted=accepted, seconds=round(time.time() - t0, 1))))


if __name__ == "__main__":
    main()
