"""
Stage 6b: Rolling per-frame forecasts (animatable)
---------------------------------------------------
The original stage6_forecast.py emits ONE forecast per track, anchored at
whatever frame that track happened to end on. That is fine as a data
artifact but impossible to animate - there is no per-frame state.

This produces a forecast for EVERY track on EVERY frame, which is what a
demo video needs, and what Stage 7 wants if you overlay predicted density.

Two methods:
  --method lstm   uses models/pretrained_traj_lstm.pt (train it first with
                  src/train_traj_lstm.py)
  --method cv     constant-velocity + social-force blend. No checkpoint, no
                  torch. Physically reasonable and completely honest - just
                  do NOT label it "LSTM" on your slide. Label it
                  "kinematic baseline".

Output schema (outputs/stage6_rolling.json):
[
  {"frame_idx": int, "timestamp_sec": float,
   "people": [
     {"track_id": int, "x": float, "y": float,
      "method": "lstm"|"cv",
      "path": [{"t": float, "x": float, "y": float}, ...]}
   ]}
]

Usage:
    python src/stage6_rolling_forecast.py --method cv
    python src/stage6_rolling_forecast.py --method lstm
"""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

VIDEO_FPS = 25.0


def load_frames(path):
    return json.loads(Path(path).read_text())


def resample_tail(hist, step_dt, n_steps, fps=VIDEO_FPS):
    """Last n_steps positions spaced step_dt apart. hist = deque[(frame,x,y)]."""
    if len(hist) < 2:
        return None
    t = np.array([h[0] / fps for h in hist])
    x = np.array([h[1] for h in hist])
    y = np.array([h[2] for h in hist])
    if t[-1] - t[0] < step_dt * (n_steps - 1):
        return None
    q = t[-1] - step_dt * np.arange(n_steps - 1, -1, -1)
    if q[0] < t[0]:
        return None
    return np.stack([np.interp(q, t, x), np.interp(q, t, y)], axis=1)


def fit_velocity(hist, window_sec, fps=VIDEO_FPS, min_pts=4):
    """
    Least-squares velocity over the last `window_sec` of history.

    The per-frame vx/vy in stage4_forces.json are single-frame finite
    differences on wobbly bounding boxes, so their DIRECTION is nearly
    random even when the speed looks sane. Clamping speed does not fix
    that. Fitting a line through the recent positions does: it averages
    out box jitter and returns a heading that actually reflects where the
    person is walking.

    Returns (velocity, quality) where quality is R^2-ish in [0,1]:
    how straight the recent motion was. Low quality = don't trust it.
    """
    if len(hist) < min_pts:
        return None, 0.0
    t = np.array([h[0] / fps for h in hist], float)
    keep = t >= t[-1] - window_sec
    if keep.sum() < min_pts:
        return None, 0.0
    t = t[keep]
    x = np.array([h[1] for h in hist], float)[keep]
    y = np.array([h[2] for h in hist], float)[keep]
    tc = t - t.mean()
    denom = float((tc ** 2).sum())
    if denom < 1e-9:
        return None, 0.0
    vx = float((tc * (x - x.mean())).sum() / denom)
    vy = float((tc * (y - y.mean())).sum() / denom)

    # residual of the straight-line fit, normalised by how far they moved
    xr = x - (x.mean() + vx * tc)
    yr = y - (y.mean() + vy * tc)
    resid = float(np.sqrt((xr ** 2 + yr ** 2).mean()))
    travelled = float(np.hypot(vx, vy) * (t[-1] - t[0]))
    quality = float(np.clip(1.0 - resid / max(travelled, 0.25), 0.0, 1.0))
    return np.array([vx, vy]), quality


def forecast_cv(pos, vel, force, steps, step_dt, force_weight, speed_clamp,
                force_clamp=3.0):
    """Constant velocity, gently bent by the Stage 4 social force."""
    sp = float(np.linalg.norm(vel))
    if sp > speed_clamp:                      # ignore ID-switch teleports
        vel = vel / max(sp, 1e-6) * speed_clamp
    fm = float(np.linalg.norm(force))
    if fm > force_clamp:                      # one close neighbour can spike this
        force = force / max(fm, 1e-6) * force_clamp
    out = []
    p = pos.astype(float).copy()
    v = vel.astype(float).copy()
    for _ in range(steps):
        v = v + force * step_dt * force_weight
        s = float(np.linalg.norm(v))
        if s > speed_clamp:
            v = v / s * speed_clamp
        p = p + v * step_dt
        out.append(p.copy())
    return np.array(out)


def main(a):
    frames = load_frames(a.stage4)
    print(f"[S6r] {len(frames)} frames from {a.stage4}")

    model = None
    obs_len, pred_len, step_dt = a.obs_len, a.steps, a.step_dt
    if a.method == "lstm":
        import torch
        from model import TrajLSTM
        ck = Path(a.checkpoint)
        if not ck.exists():
            raise SystemExit(
                f"[S6r] Checkpoint not found: {ck}\n"
                "       Train one first:  python src/train_traj_lstm.py\n"
                "       Or run with --method cv for the kinematic baseline.")
        ckpt = torch.load(str(ck), map_location="cpu")
        model = TrajLSTM(hidden=ckpt["hidden"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        obs_len, pred_len, step_dt = ckpt["obs_len"], ckpt["pred_len"], ckpt["step_dt_sec"]
        print(f"[S6r] LSTM loaded: obs {obs_len} pred {pred_len} @ {step_dt}s")
        if "val_ade_m" in ckpt:
            print(f"[S6r] checkpoint val ADE {ckpt['val_ade_m']}m "
                  f"FDE {ckpt['val_fde_m']}m (trained on {ckpt.get('trained_on','?')})")
        torch.set_grad_enabled(False)

    hist = defaultdict(lambda: deque(maxlen=200))
    out_frames = []
    n_lstm = n_cv = n_skip = 0

    for fr in frames:
        fidx = fr["frame_idx"]
        people_out = []
        for p in fr["people"]:
            tid = p["track_id"]
            hist[tid].append((fidx, p["x"], p["y"]))
            pos = np.array([p["x"], p["y"]], float)
            force = np.array([p.get("force_x", 0.0), p.get("force_y", 0.0)], float)

            # Fitted heading beats the raw single-frame vx/vy by a wide margin.
            vfit, quality = fit_velocity(hist[tid], a.vel_window)
            if vfit is None:
                vel = np.array([p.get("vx", 0.0), p.get("vy", 0.0)], float)
                quality = 0.0
            else:
                vel = vfit

            # Too little history, or motion too erratic to extrapolate: emit an
            # empty path rather than a confident-looking wrong one. This is why
            # the video stops flailing.
            if quality < a.min_quality or len(hist[tid]) < a.min_history:
                people_out.append({
                    "track_id": tid,
                    "x": round(float(pos[0]), 3),
                    "y": round(float(pos[1]), 3),
                    "method": "none",
                    "quality": round(float(quality), 3),
                    "path": [],
                })
                n_skip += 1
                continue

            path, used = None, "cv"
            if model is not None:
                seq = resample_tail(hist[tid], step_dt, obs_len + 1)
                if seq is not None:
                    import torch
                    disp = np.diff(seq, axis=0).astype(np.float32)
                    pred = model(torch.from_numpy(disp[None]), pred_len=pred_len)[0].numpy()
                    path = seq[-1] + np.cumsum(pred, axis=0)
                    used = "lstm"
                    n_lstm += 1
            if path is None:
                path = forecast_cv(pos, vel, force, pred_len, step_dt,
                                   a.force_weight, a.speed_clamp, a.force_clamp)
                n_cv += 1

            people_out.append({
                "track_id": tid,
                "x": round(float(pos[0]), 3),
                "y": round(float(pos[1]), 3),
                "method": used,
                "quality": round(float(quality), 3),
                "path": [{"t": round(step_dt * (k + 1), 2),
                          "x": round(float(q[0]), 3),
                          "y": round(float(q[1]), 3)}
                         for k, q in enumerate(path)],
            })

        out_frames.append({
            "frame_idx": fidx,
            "timestamp_sec": fr.get("timestamp_sec", round(fidx / VIDEO_FPS, 3)),
            "people": people_out,
        })

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out_frames, indent=1))
    total = n_lstm + n_cv + n_skip
    print(f"[S6r] Wrote {len(out_frames)} frames -> {a.out}")
    if total:
        print(f"[S6r] {n_lstm} LSTM, {n_cv} kinematic, {n_skip} suppressed "
              f"({100*n_skip/total:.0f}% too short or too erratic to forecast)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage4", default="outputs/stage4_forces.json")
    ap.add_argument("--checkpoint", default="models/pretrained_traj_lstm.pt")
    ap.add_argument("--out", default="outputs/stage6_rolling.json")
    ap.add_argument("--method", choices=["lstm", "cv"], default="cv")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--step-dt", type=float, default=0.4)
    ap.add_argument("--obs-len", type=int, default=5)
    ap.add_argument("--force-weight", type=float, default=0.25,
                    help="how hard the social force bends the forecast")
    ap.add_argument("--speed-clamp", type=float, default=2.0)
    ap.add_argument("--force-clamp", type=float, default=3.0)
    ap.add_argument("--vel-window", type=float, default=0.6,
                    help="seconds of history used to fit heading; raise to smooth more")
    ap.add_argument("--min-quality", type=float, default=0.35,
                    help="0-1 straightness of recent motion; below this, no forecast")
    ap.add_argument("--min-history", type=int, default=6,
                    help="frames of history required before forecasting")
    main(ap.parse_args())