"""
Stage 7: Density Convergence Engine.

Turns the per-frame 2D world coordinates (Stage 3 output, outputs/world_trajectories.json)
into a persons/m^2 density heatmap, warps it into the original camera view using the
Stage-3 homography, and burns it onto cctv_clip.avi frame-by-frame -- same pattern as
stage1_annotated.mp4 / stage2_tracked.mp4.

It also folds in Stage 6's forecasts (outputs/stage6_forecasts.json) as a short-lived
"ghost" overlay: for the handful of frames right after a track was forecasted, we draw
a faded predicted-position marker + trail so you can visually see people converging on
a hot zone *before* they arrive there. This is the "convergence" part -- it's a visual
aid, not a second rigorous density estimate.

------------------------------------------------------------------------
CONFIG BLOCK -- verified against the real files in this upload
------------------------------------------------------------------------
world_trajectories.json: top-level LIST, one entry per video frame, in order.
  entry = {"frame_idx": int, "timestamp_sec": float,
           "people": [{"track_id": int, "x": float, "y": float}, ...]}
  Confirmed 250 entries, frame_idx 0..249, x in [0.74, 11.38] m, y in [-0.07, 8.09] m.

stage6_forecasts.json: top-level DICT with key "forecasts" -> list.
  entry = {"track_id", "last_observed_frame", "last_observed_pos": [x,y],
           "forecast_horizon_sec", "forecast_pos": [x,y],
           "predicted_path": [{"t_ahead_sec","x","y"}, ...]}
  NOTE: this is a one-shot forecast per track (computed once, at whatever frame that
  track had enough history), not a per-frame stream. We only show it for a short
  window after last_observed_frame -- see FORECAST_GHOST_SEC below.

homography.json: {"homography_matrix": 3x3, ...}. Verified direction: applying the
  matrix to a homogeneous PIXEL point yields the WORLD point (checked all 4 calibration
  points, round-trips to <0.01 m). So homography_matrix is pixel->world, and its
  inverse is world->pixel, which is what we need to warp the density grid back onto
  the video.
------------------------------------------------------------------------
"""
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# ----------------------------- CONFIG ---------------------------------------
GRID_RES_M = 0.15          # world-space cell size (meters) for the density grid
PERSON_SIGMA_M = 0.45      # Gaussian "footprint" radius per person, meters
                            # (~shoulder-width personal space; tune against your scene)

# Fruin/Still-style crowd Level-of-Service bands, in persons/m^2. These are the
# standard industry heuristics for crowd risk -- validate against your own footage
# before trusting the exact cutoffs in a real safety decision.
LOS_BANDS = [
    (0.0, 1.0, (60, 180, 60)),     # comfortable   - green
    (1.0, 2.0, (60, 200, 220)),    # busy          - yellow
    (2.0, 4.0, (0, 140, 255)),     # constrained   - orange
    (4.0, 6.0, (0, 60, 255)),      # high risk     - red
    (6.0, 1e9, (0, 0, 180)),       # crush risk    - dark red
]
CRUSH_DENSITY = 4.0        # persons/m^2 -- cells at/above this get flagged + labeled

HEATMAP_ALPHA = 0.55        # opacity of the density layer over the video frame
FORECAST_GHOST_SEC = 1.0    # how long (seconds of video) a Stage-6 forecast stays
                             # visible as a ghost marker after its last_observed_frame
FORECAST_COLOR = (255, 210, 0)  # cyan-ish BGR for the predicted-position ghost

WORLD_BOUNDS_MARGIN_M = 1.0  # pad the auto-detected world bounding box by this much
# ------------------------------------------------------------------------


def load_trajectories(path):
    data = json.loads(Path(path).read_text())
    by_frame = {fr["frame_idx"]: fr for fr in data}
    return data, by_frame


def load_forecasts(path):
    if path is None or not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text())
    return data.get("forecasts", [])


def world_bounds(trajectories, margin=WORLD_BOUNDS_MARGIN_M):
    xs, ys = [], []
    for fr in trajectories:
        for p in fr["people"]:
            xs.append(p["x"])
            ys.append(p["y"])
    x_min, x_max = min(xs) - margin, max(xs) + margin
    y_min, y_max = min(ys) - margin, max(ys) + margin
    return x_min, x_max, y_min, y_max


def density_to_bgr(density):
    """Vectorized LOS-band colorize. density: (H,W) float32 persons/m^2 -> (H,W,3) uint8."""
    h, w = density.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for lo, hi, color in LOS_BANDS:
        mask = (density >= lo) & (density < hi)
        out[mask] = color
    return out


def build_density_grid(people_xy, x_min, y_min, grid_w, grid_h):
    """people_xy: (N,2) world coords -> (grid_h, grid_w) persons/m^2 field."""
    field = np.zeros((grid_h, grid_w), dtype=np.float32)
    if len(people_xy) == 0:
        return field
    cols = ((people_xy[:, 0] - x_min) / GRID_RES_M).astype(int)
    rows = ((people_xy[:, 1] - y_min) / GRID_RES_M).astype(int)
    valid = (cols >= 0) & (cols < grid_w) & (rows >= 0) & (rows < grid_h)
    for c, r in zip(cols[valid], rows[valid]):
        field[r, c] += 1.0  # one "person mass" per track, splatted to nearest cell

    sigma_cells = PERSON_SIGMA_M / GRID_RES_M
    field = gaussian_filter(field, sigma=sigma_cells, mode="constant")
    # gaussian_filter preserves total mass, so dividing by cell area converts
    # "mass per cell" into an approximate persons/m^2 density field.
    field = field / (GRID_RES_M ** 2)
    return field


def build_grid_to_pixel_matrix(H_pix_to_world, x_min, y_min):
    """
    Composes: grid-image pixel (col,row,1) -> world (x,y,1) -> video pixel (u,v,1).
    A maps grid indices to world coords (regular affine grid).
    H_pix_to_world is pixel->world, so its inverse is world->pixel.
    """
    A = np.array([
        [GRID_RES_M, 0, x_min],
        [0, GRID_RES_M, y_min],
        [0, 0, 1],
    ])
    H_world_to_pix = np.linalg.inv(H_pix_to_world)
    return H_world_to_pix @ A


def active_forecast_ghosts(forecasts, frame_idx, fps):
    """Yield (world_x, world_y, alpha, track_id) for forecasts still within their
    ghost window at this video frame."""
    ghosts = []
    ghost_frames = FORECAST_GHOST_SEC * fps
    for fc in forecasts:
        start = fc["last_observed_frame"]
        elapsed = frame_idx - start
        if 0 <= elapsed <= ghost_frames:
            t_ahead = elapsed / fps
            path = fc["predicted_path"]
            times = [p["t_ahead_sec"] for p in path]
            xs = [p["x"] for p in path]
            ys = [p["y"] for p in path]
            if t_ahead <= times[0]:
                x0, y0 = fc["last_observed_pos"]
                frac = t_ahead / times[0] if times[0] > 0 else 1.0
                gx = x0 + frac * (xs[0] - x0)
                gy = y0 + frac * (ys[0] - y0)
            else:
                gx = float(np.interp(t_ahead, times, xs))
                gy = float(np.interp(t_ahead, times, ys))
            alpha = 1.0 - (elapsed / ghost_frames if ghost_frames > 0 else 1.0)
            ghosts.append((gx, gy, max(alpha, 0.0), fc["track_id"]))
    return ghosts


def world_to_pixel(H_world_to_pix, x, y):
    v = H_world_to_pix @ np.array([x, y, 1.0])
    return v[0] / v[2], v[1] / v[2]


def run(traj_path, forecast_path, homography_path, video_path, out_video, out_json):
    trajectories, traj_by_frame = load_trajectories(traj_path)
    forecasts = load_forecasts(forecast_path)
    homog = json.loads(Path(homography_path).read_text())
    H_pix_to_world = np.array(homog["homography_matrix"])
    H_world_to_pix = np.linalg.inv(H_pix_to_world)

    x_min, x_max, y_min, y_max = world_bounds(trajectories)
    grid_w = max(int((x_max - x_min) / GRID_RES_M), 1)
    grid_h = max(int((y_max - y_min) / GRID_RES_M), 1)
    grid_to_pixel = build_grid_to_pixel_matrix(H_pix_to_world, x_min, y_min)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video at {video_path}. Run this script from your "
            f"crowd-demo folder (or pass --video) so cctv_clip.avi resolves."
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(out_video).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h)
    )

    frame_summaries = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        traj_entry = traj_by_frame.get(frame_idx)
        people = traj_entry["people"] if traj_entry else []
        people_xy = np.array([[p["x"], p["y"]] for p in people], dtype=np.float32)

        density = build_density_grid(people_xy, x_min, y_min, grid_w, grid_h)
        heat_bgr = density_to_bgr(density)
        heat_on_frame = cv2.warpPerspective(
            heat_bgr, grid_to_pixel, (frame_w, frame_h), flags=cv2.INTER_LINEAR
        )
        # only blend where the warped heatmap actually covers (avoid darkening
        # the parts of the frame outside the calibrated ground plane)
        coverage = cv2.warpPerspective(
            np.ones((grid_h, grid_w), dtype=np.uint8) * 255,
            grid_to_pixel, (frame_w, frame_h), flags=cv2.INTER_LINEAR,
        )
        mask = (coverage > 0).astype(np.float32)[..., None] * HEATMAP_ALPHA
        blended = (frame.astype(np.float32) * (1 - mask)
                   + heat_on_frame.astype(np.float32) * mask).astype(np.uint8)

        # forecast "convergence" ghosts
        for gx, gy, alpha, tid in active_forecast_ghosts(forecasts, frame_idx, fps):
            u, v = world_to_pixel(H_world_to_pix, gx, gy)
            u, v = int(round(u)), int(round(v))
            if 0 <= u < frame_w and 0 <= v < frame_h:
                overlay = blended.copy()
                cv2.circle(overlay, (u, v), 8, FORECAST_COLOR, 2)
                cv2.putText(overlay, f"#{tid}", (u + 10, v - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, FORECAST_COLOR, 1)
                blended = cv2.addWeighted(overlay, alpha, blended, 1 - alpha, 0)

        max_density = float(density.max()) if density.size else 0.0
        crush_cells = int((density >= CRUSH_DENSITY).sum())
        if crush_cells > 0:
            cv2.putText(blended, "CRUSH RISK", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        writer.write(blended)
        frame_summaries.append({
            "frame_idx": frame_idx,
            "num_people": len(people),
            "max_density_per_m2": round(max_density, 3),
            "crush_cells": crush_cells,
        })
        frame_idx += 1

    cap.release()
    writer.release()

    Path(out_json).write_text(json.dumps({
        "grid_res_m": GRID_RES_M,
        "person_sigma_m": PERSON_SIGMA_M,
        "crush_density_threshold": CRUSH_DENSITY,
        "world_bounds": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "frames": frame_summaries,
    }, indent=2))
    print(f"Wrote {frame_idx} frames to {out_video}")
    print(f"Wrote per-frame density summary to {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", default="../outputs/world_trajectories.json")
    ap.add_argument("--forecasts", default="../outputs/stage6_forecasts.json")
    ap.add_argument("--homography", default="../config/homography.json")
    ap.add_argument("--video", default="../data/cctv_clip.avi")
    ap.add_argument("--out", default="../outputs/stage7_density.mp4")
    ap.add_argument("--out-json", default="../outputs/stage7_density.json")
    ap.add_argument("--no-forecast", action="store_true",
                     help="disable the Stage-6 forecast ghost overlay")
    args = ap.parse_args()
    run(
        args.trajectories,
        None if args.no_forecast else args.forecasts,
        args.homography,
        args.video,
        args.out,
        args.out_json,
    )