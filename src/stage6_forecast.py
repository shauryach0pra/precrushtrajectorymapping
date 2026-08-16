"""
Stage 6: PyTorch LSTM trajectory forecasting.

Loads the OFFLINE-PRETRAINED checkpoint (pretrained_traj_lstm.pt, trained on
real ETH/UCY pedestrian data in train.py — not trained here). This script
only does inference, so it's fast enough to run as part of your live/pre-
rendered demo pass.

------------------------------------------------------------------------
CONFIG BLOCK — verified against the real stage4_forces.json
------------------------------------------------------------------------
stage4_forces.json is a top-level LIST of per-frame entries (not wrapped in
a "frames" key), and the per-frame index field is "frame_idx", not "frame".
track_id / x / y / force_x / force_y already matched. FIELDS below is
updated to match.
"""
import json
import argparse
from pathlib import Path

import torch
import numpy as np

from model import TrajLSTM

# ----------------------- CONFIG: matches stage4_forces.json --------------------
FIELDS = {
    "frames_key": "frames",       # unused when the JSON top level is a list (it is here)
    "frame_index_key": "frame_idx",  # per-frame: integer frame number
    "people_key": "people",       # per-frame: list of per-person entries
    "track_id_key": "track_id",   # per-person: persistent ID from Stage 2
    "x_key": "x",                 # per-person: world x (meters, from Stage 3)
    "y_key": "y",                 # per-person: world y (meters, from Stage 3)
    "force_x_key": "force_x",     # optional, only used if BLEND_WITH_FORCE=True
    "force_y_key": "force_y",
}
VIDEO_FPS = 25.0          # your cctv_clip.avi fps
FORECAST_HORIZON_SEC = 0.5
BLEND_WITH_FORCE = False  # keep off until force units are confirmed (see note below)
FORCE_BLEND_WEIGHT = 0.3  # only used if BLEND_WITH_FORCE=True
# ---------------------------------------------------------------------------


def load_tracks(stage4_path):
    """Returns dict: track_id -> sorted list of (frame_idx, x, y, fx, fy)."""
    raw = json.loads(Path(stage4_path).read_text())
    frames = raw[FIELDS["frames_key"]] if isinstance(raw, dict) else raw
    tracks = {}
    for fr in frames:
        frame_idx = fr[FIELDS["frame_index_key"]]
        people = fr[FIELDS["people_key"]]
        for p in people:
            tid = p[FIELDS["track_id_key"]]
            x, y = p[FIELDS["x_key"]], p[FIELDS["y_key"]]
            fx = p.get(FIELDS["force_x_key"], 0.0)
            fy = p.get(FIELDS["force_y_key"], 0.0)
            tracks.setdefault(tid, []).append((frame_idx, x, y, fx, fy))
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])
    return tracks


def resample_to_step_dt(history, step_dt_sec, fps, n_steps):
    """
    history: list of (frame_idx, x, y) for one track, ascending frame order.
    Resamples the last `n_steps` positions spaced `step_dt_sec` apart
    (matching the training data's native cadence) via linear interpolation
    over the observed (frame_idx / fps) timestamps. Returns None if there's
    not enough history to cover n_steps * step_dt_sec of real time.
    """
    if len(history) < 2:
        return None
    times = np.array([h[0] / fps for h in history])
    xs = np.array([h[1] for h in history])
    ys = np.array([h[2] for h in history])
    t_last = times[-1]
    needed_span = n_steps * step_dt_sec
    if t_last - times[0] < needed_span:
        return None  # not enough history yet for this track
    query_times = t_last - step_dt_sec * np.arange(n_steps - 1, -1, -1)
    if query_times[0] < times[0]:
        return None
    x_r = np.interp(query_times, times, xs)
    y_r = np.interp(query_times, times, ys)
    return np.stack([x_r, y_r], axis=1)  # (n_steps, 2) positions


def forecast_all_tracks(stage4_path, checkpoint_path, out_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = TrajLSTM(hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    obs_len, pred_len, step_dt = ckpt["obs_len"], ckpt["pred_len"], ckpt["step_dt_sec"]

    tracks = load_tracks(stage4_path)
    results = []

    for tid, records in tracks.items():
        pos_history = [(r[0], r[1], r[2]) for r in records]
        resampled = resample_to_step_dt(pos_history, step_dt, VIDEO_FPS, obs_len + 1)
        if resampled is None:
            continue  # not enough history yet — skip until this track has more frames
        obs_disp = np.diff(resampled, axis=0)  # (obs_len, 2)
        obs_t = torch.from_numpy(obs_disp[None].astype(np.float32))

        with torch.no_grad():
            pred_disp = model(obs_t, pred_len=pred_len, target=None)[0].numpy()

        last_pos = resampled[-1]
        cum = np.cumsum(pred_disp, axis=0)
        future_pos = last_pos + cum  # (pred_len, 2), one point every step_dt seconds

        if BLEND_WITH_FORCE:
            fx, fy = records[-1][3], records[-1][4]
            # simple physics nudge: treat (fx, fy) as an acceleration-like term
            # over the same step_dt; scale/units should be validated against
            # your Stage 4 output before trusting this in the demo.
            force_step = np.array([fx, fy]) * step_dt
            for i in range(len(future_pos)):
                physics_pos = last_pos + force_step * (i + 1)
                future_pos[i] = (1 - FORCE_BLEND_WEIGHT) * future_pos[i] + FORCE_BLEND_WEIGHT * physics_pos

        # interpolate to the exact requested horizon (e.g. 0.5s) between the
        # bracketing predicted steps
        step_times = step_dt * np.arange(1, pred_len + 1)
        if FORECAST_HORIZON_SEC <= step_times[0]:
            frac = FORECAST_HORIZON_SEC / step_times[0]
            pos_at_horizon = last_pos + frac * (future_pos[0] - last_pos)
        else:
            idx = np.searchsorted(step_times, FORECAST_HORIZON_SEC) - 1
            idx = min(max(idx, 0), pred_len - 2)
            t0, t1 = step_times[idx], step_times[idx + 1]
            frac = (FORECAST_HORIZON_SEC - t0) / (t1 - t0)
            pos_at_horizon = future_pos[idx] + frac * (future_pos[idx + 1] - future_pos[idx])

        results.append({
            "track_id": tid,
            "last_observed_frame": records[-1][0],
            "last_observed_pos": [float(last_pos[0]), float(last_pos[1])],
            "forecast_horizon_sec": FORECAST_HORIZON_SEC,
            "forecast_pos": [float(pos_at_horizon[0]), float(pos_at_horizon[1])],
            "predicted_path": [
                {"t_ahead_sec": float(t), "x": float(p[0]), "y": float(p[1])}
                for t, p in zip(step_times, future_pos)
            ],
        })

    Path(out_path).write_text(json.dumps({"forecasts": results}, indent=2))
    print(f"Wrote {len(results)} track forecasts to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage4", default="../outputs/stage4_forces.json")
    ap.add_argument("--checkpoint", default="../models/pretrained_traj_lstm.pt")
    ap.add_argument("--out", default="../outputs/stage6_forecasts.json")
    args = ap.parse_args()
    forecast_all_tracks(args.stage4, args.checkpoint, args.out)