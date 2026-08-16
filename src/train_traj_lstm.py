"""
Train TrajLSTM on your own world trajectories.
----------------------------------------------
Produces models/pretrained_traj_lstm.pt in exactly the format
src/model.py and src/stage6_forecast.py expect:

    {"hidden": int, "state_dict": dict,
     "obs_len": int, "pred_len": int, "step_dt_sec": float}

HONESTY NOTE - read before you present this.
Training on the same clip you demo on is circular. The model will look
better than it deserves. This script therefore holds out 20% of tracks as
a validation set and prints validation ADE/FDE in metres, so you have a
number you can defend rather than a vibe. If a judge asks "did you train
on your test data?", the correct answer is "yes, with a held-out track
split, and here is the validation error" - not silence.

For a stronger result, train on ETH/UCY instead and pass --data with a
file in the same schema. That is the proper pretraining corpus.

Usage:
    python src/train_traj_lstm.py
    python src/train_traj_lstm.py --epochs 400 --hidden 128
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import TrajLSTM

VIDEO_FPS = 25.0


def load_tracks(path):
    """world_trajectories.json / stage4_forces.json -> {tid: [(frame, x, y), ...]}"""
    data = json.loads(Path(path).read_text())
    tracks = defaultdict(list)
    for fr in data:
        for p in fr["people"]:
            tracks[p["track_id"]].append((fr["frame_idx"], p["x"], p["y"]))
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])
    return tracks


def resample(records, step_dt, n_steps, fps=VIDEO_FPS):
    """Uniformly resample a track to n_steps positions spaced step_dt apart."""
    if len(records) < 2:
        return None
    t = np.array([r[0] / fps for r in records])
    x = np.array([r[1] for r in records])
    y = np.array([r[2] for r in records])
    if t[-1] - t[0] < step_dt * n_steps:
        return None
    q = t[-1] - step_dt * np.arange(n_steps - 1, -1, -1)
    if q[0] < t[0]:
        return None
    return np.stack([np.interp(q, t, x), np.interp(q, t, y)], axis=1)


def build_windows(tracks, obs_len, pred_len, step_dt, fps=VIDEO_FPS):
    """Sliding windows of (obs displacements, future displacements)."""
    need = obs_len + pred_len + 1
    X, Y, owners = [], [], []
    for tid, recs in tracks.items():
        t0, t1 = recs[0][0] / fps, recs[-1][0] / fps
        span = step_dt * need
        if t1 - t0 < span:
            continue
        # slide the window end backwards across the track's lifetime
        end = t1
        while end - span >= t0:
            sub = [r for r in recs if r[0] / fps <= end]
            seq = resample(sub, step_dt, need, fps)
            if seq is not None:
                disp = np.diff(seq, axis=0)          # (obs_len+pred_len, 2)
                X.append(disp[:obs_len])
                Y.append(disp[obs_len:])
                owners.append(tid)
            end -= step_dt
    if not X:
        return None
    return (np.array(X, np.float32), np.array(Y, np.float32), np.array(owners))


def main(a):
    tracks = load_tracks(a.data)
    print(f"[Train] {len(tracks)} tracks from {a.data}")

    built = build_windows(tracks, a.obs_len, a.pred_len, a.step_dt)
    if built is None:
        raise SystemExit(
            "[Train] No usable windows. Your tracks are too short for "
            f"obs_len={a.obs_len} + pred_len={a.pred_len} at {a.step_dt}s "
            f"(needs {(a.obs_len + a.pred_len + 1) * a.step_dt:.1f}s of continuous "
            "history per person). Lower --obs-len/--pred-len or --step-dt.")
    X, Y, owners = built
    print(f"[Train] {len(X)} windows, each {a.obs_len} obs -> {a.pred_len} pred "
          f"at {a.step_dt}s/step")

    # split by TRACK, never by window - windows from one person overlap heavily
    uniq = np.unique(owners)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    n_val = max(1, int(0.2 * len(uniq)))
    val_ids = set(uniq[:n_val].tolist())
    is_val = np.array([o in val_ids for o in owners])
    print(f"[Train] {len(uniq) - n_val} train tracks / {n_val} val tracks")

    Xtr = torch.from_numpy(X[~is_val]);  Ytr = torch.from_numpy(Y[~is_val])
    Xva = torch.from_numpy(X[is_val]);   Yva = torch.from_numpy(Y[is_val])
    if len(Xtr) == 0 or len(Xva) == 0:
        raise SystemExit("[Train] Split left one side empty; need more tracks.")

    model = TrajLSTM(hidden=a.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lossf = nn.MSELoss()

    best = float("inf")
    best_state = None
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), a.batch):
            idx = perm[i:i + a.batch]
            opt.zero_grad()
            pred = model(Xtr[idx], pred_len=a.pred_len)
            loss = lossf(pred, Ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        tr = tot / len(Xtr)

        model.eval()
        with torch.no_grad():
            pv = model(Xva, pred_len=a.pred_len)
            va = float(lossf(pv, Yva))
            # ADE/FDE in metres, on cumulative positions not displacements
            pp = torch.cumsum(pv, dim=1)
            gg = torch.cumsum(Yva, dim=1)
            err = torch.linalg.norm(pp - gg, dim=-1)
            ade, fde = float(err.mean()), float(err[:, -1].mean())

        if va < best:
            best, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
            tag = " *"
        else:
            tag = ""
        if ep % a.log_every == 0 or ep == 1:
            print(f"[Train] ep {ep:4d}  train {tr:.5f}  val {va:.5f}  "
                  f"ADE {ade:.3f}m  FDE {fde:.3f}m{tag}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pp = torch.cumsum(model(Xva, pred_len=a.pred_len), dim=1)
        gg = torch.cumsum(Yva, dim=1)
        err = torch.linalg.norm(pp - gg, dim=-1)
        ade, fde = float(err.mean()), float(err[:, -1].mean())

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "hidden": a.hidden,
        "state_dict": best_state,
        "obs_len": a.obs_len,
        "pred_len": a.pred_len,
        "step_dt_sec": a.step_dt,
        # provenance - so nobody later mistakes this for an ETH/UCY model
        "trained_on": str(a.data),
        "val_ade_m": round(ade, 4),
        "val_fde_m": round(fde, 4),
        "n_train_windows": int(len(Xtr)),
    }, a.out)

    print(f"\n[Train] Saved -> {a.out}")
    print(f"[Train] Held-out validation ADE {ade:.3f} m over "
          f"{a.pred_len * a.step_dt:.1f}s, FDE {fde:.3f} m")
    print("[Train] Quote these numbers if asked. They are the honest measure.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/world_trajectories.json")
    ap.add_argument("--out", default="models/pretrained_traj_lstm.pt")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--obs-len", type=int, default=5)
    ap.add_argument("--pred-len", type=int, default=5)
    ap.add_argument("--step-dt", type=float, default=0.4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=25)
    main(ap.parse_args())