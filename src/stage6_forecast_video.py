"""
Stage 6 Demo: Trajectory Forecasting
-------------------------------------
Side-by-side camera + BEV showing, for every person, where they are NOW and
where the model says they will be over the next ~2 seconds.

  solid dot + short tail  = observed position and recent history
  dashed forward path     = forecast, fading with time-ahead
  hollow ring at the end  = predicted position at the horizon

Optionally overlays a HINDCAST accuracy check (--verify): re-draws the
forecast that was made N frames ago next to where the person actually
ended up, with the error in metres. This is the single most convincing
thing you can put on a forecasting slide, because it shows the model being
checked rather than just asserted.

Inputs:
    outputs/stage6_rolling.json   (from src/stage6_rolling_forecast.py)
    config/homography.json        (to project onto the camera frame)
    data/cctv_clip.avi            (optional; BEV-only without it)

Usage:
    python src/stage6_forecast_video.py --video data/cctv_clip.avi --verify
"""

import argparse
import colorsys
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

BG = (26, 24, 22)
PANEL = (34, 31, 29)
GRID_MINOR = (52, 48, 45)
GRID_MAJOR = (80, 74, 69)
TEXT = (238, 238, 238)
MUTED = (150, 146, 142)
PRED = (255, 190, 90)     # forecast path
TRUTH = (120, 235, 140)   # actual outcome, in --verify mode
ERRC = (90, 120, 255)     # error segment
FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(tid):
    h = (int(tid) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.6, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


class BEV:
    def __init__(self, xmin, xmax, ymin, ymax, ppm):
        self.xmin, self.xmax, self.ymin, self.ymax, self.ppm = xmin, xmax, ymin, ymax, ppm
        self.w = int(round((xmax - xmin) * ppm))
        self.h = int(round((ymax - ymin) * ppm))

    def pt(self, x, y):
        return (int(round((x - self.xmin) * self.ppm)),
                int(round((self.ymax - y) * self.ppm)))


def world_to_px(Hinv, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2)


def dashed(img, p0, p1, color, thick=2, dash=8):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    L = np.linalg.norm(p1 - p0)
    if L < 1:
        return
    n = max(2, int(L // dash))
    for i in range(0, n, 2):
        a = p0 + (p1 - p0) * (i / n)
        b = p0 + (p1 - p0) * (min(i + 1, n) / n)
        cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), color, thick, cv2.LINE_AA)


def label(img, text, org, color=TEXT, scale=0.42, thick=1, pad=3):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x = int(min(max(pad, org[0]), img.shape[1] - tw - pad))
    y = int(min(max(th + pad, org[1]), img.shape[0] - base - pad))
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base), (16, 15, 14), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def header(img, title, sub):
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (img.shape[1], 46), (14, 13, 12), -1)
    cv2.addWeighted(ov, 0.8, img, 0.2, 0, img)
    cv2.putText(img, title, (14, 22), FONT, 0.55, TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, sub, (14, 39), FONT, 0.42, MUTED, 1, cv2.LINE_AA)


def draw_grid(img, bev):
    for xm in range(int(np.floor(bev.xmin)), int(np.ceil(bev.xmax)) + 1):
        maj = xm % 5 == 0
        cv2.line(img, bev.pt(xm, bev.ymax), bev.pt(xm, bev.ymin),
                 GRID_MAJOR if maj else GRID_MINOR, 1, cv2.LINE_AA)
        if maj:
            cv2.putText(img, f"{xm}m", (bev.pt(xm, bev.ymin)[0] + 4, bev.h - 8),
                        FONT, 0.4, MUTED, 1, cv2.LINE_AA)
    for ym in range(int(np.floor(bev.ymin)), int(np.ceil(bev.ymax)) + 1):
        maj = ym % 5 == 0
        cv2.line(img, bev.pt(bev.xmin, ym), bev.pt(bev.xmax, ym),
                 GRID_MAJOR if maj else GRID_MINOR, 1, cv2.LINE_AA)
        if maj:
            cv2.putText(img, f"{ym}m", (6, bev.pt(bev.xmin, ym)[1] - 5),
                        FONT, 0.4, MUTED, 1, cv2.LINE_AA)


def draw_forecast(img, project, people, trails, args, errors=None):
    """project: (x,y) -> pixel tuple. Works for both BEV and camera panels."""
    for tid, pts in trails.items():
        if len(pts) < 2:
            continue
        c = color_for(tid)
        arr = [project(x, y) for x, y in pts]
        for i in range(1, len(arr)):
            f = i / len(arr)
            cv2.line(img, arr[i - 1], arr[i],
                     tuple(int(v * (0.2 + 0.6 * f)) for v in c), 2, cv2.LINE_AA)

    for p in people:
        tid = p["track_id"]
        c = color_for(tid)
        here = project(p["x"], p["y"])
        path = p["path"]

        if path:
            prev = here
            for k, q in enumerate(path):
                nxt = project(q["x"], q["y"])
                fade = 1.0 - 0.55 * (k / max(1, len(path) - 1))
                dashed(img, prev, nxt, tuple(int(v * fade) for v in PRED), 2)
                prev = nxt
            cv2.circle(img, prev, 6, PRED, 2, cv2.LINE_AA)
        else:
            # no reliable forecast for this person - say so instead of guessing
            cv2.circle(img, here, 9, (70, 68, 66), 1, cv2.LINE_AA)

        cv2.circle(img, here, 6, (18, 16, 15), -1, cv2.LINE_AA)
        cv2.circle(img, here, 4, c, -1, cv2.LINE_AA)

        if errors and tid in errors:
            pred_pt, true_pt, err = errors[tid]
            pp, tp = project(*pred_pt), project(*true_pt)
            cv2.circle(img, pp, 5, PRED, 1, cv2.LINE_AA)
            cv2.circle(img, tp, 5, TRUTH, -1, cv2.LINE_AA)
            cv2.line(img, pp, tp, ERRC, 2, cv2.LINE_AA)
            if err > args.err_label_min:
                label(img, f"{err:.2f}m", (tp[0] + 8, tp[1] - 6), ERRC, 0.38, 1)


def main(a):
    frames = json.loads(Path(a.forecasts).read_text())
    by_idx = {f["frame_idx"]: f for f in frames}
    xs = [p["x"] for f in frames for p in f["people"]]
    ys = [p["y"] for f in frames for p in f["people"]]
    xlo, xhi = np.percentile(xs, [1, 99])
    ylo, yhi = np.percentile(ys, [1, 99])

    horizon = next((p["path"][-1]["t"] for f in frames for p in f["people"] if p["path"]), 2.0)
    methods = {p["method"] for f in frames for p in f["people"] if p["path"]}
    mlabel = "LSTM" if methods == {"lstm"} else (
        "kinematic baseline" if methods == {"cv"} else "LSTM + kinematic fallback")
    print(f"[S6v] {len(frames)} frames | horizon {horizon}s | method: {mlabel}")

    cap, Hinv, cam_h = None, None, 0
    if a.video:
        H = np.array(json.loads(Path(a.homography).read_text())["homography_matrix"])
        Hinv = np.linalg.inv(H)
        cap = cv2.VideoCapture(a.video)
        if not cap.isOpened():
            raise SystemExit(f"[S6v] Could not open video: {a.video}")
        cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ppm = (cam_h / (yhi - ylo + 2 * a.pad)) if cap else a.ppm
    bev = BEV(xlo - a.pad, xhi + a.pad, ylo - a.pad, yhi + a.pad, ppm)

    fps = (cap.get(cv2.CAP_PROP_FPS) or 25.0) if cap else 25.0
    lag = int(round(horizon * fps))          # frames between forecast and outcome
    trails = defaultdict(lambda: deque(maxlen=max(2, int(a.trail_sec * fps))))

    Path(a.out_png).parent.mkdir(parents=True, exist_ok=True)
    gap = 10
    vw = None
    idx = 0
    err_hist = []

    while True:
        fr = by_idx.get(idx)
        cam = None
        if cap:
            ok, cam = cap.read()
            if not ok:
                break
        elif fr is None:
            break
        if a.max_frames and idx >= a.max_frames:
            break
        if fr is None:
            idx += 1
            continue

        people = fr["people"]
        for p in people:
            trails[p["track_id"]].append((p["x"], p["y"]))

        # hindcast check: what did we predict `lag` frames ago, and what happened?
        errors = {}
        if a.verify and idx - lag in by_idx:
            past = by_idx[idx - lag]
            now_pos = {p["track_id"]: (p["x"], p["y"]) for p in people}
            for p in past["people"]:
                tid = p["track_id"]
                if tid not in now_pos or not p["path"]:
                    continue
                pred = (p["path"][-1]["x"], p["path"][-1]["y"])
                true = now_pos[tid]
                e = float(np.hypot(pred[0] - true[0], pred[1] - true[1]))
                errors[tid] = (pred, true, e)
                err_hist.append(e)

        right = np.full((bev.h, bev.w, 3), PANEL, np.uint8)
        draw_grid(right, bev)
        draw_forecast(right, bev.pt, people, trails, a, errors)
        mean_e = f"  |  mean {horizon}s error {np.mean(err_hist):.2f}m" if err_hist else ""
        header(right, f"Stage 6  -  {horizon}s trajectory forecast",
               f"{len(people)} people  |  {mlabel}{mean_e}")

        ly = bev.h - 56
        cv2.rectangle(right, (10, ly - 14), (232, bev.h - 12), (14, 13, 12), -1)
        dashed(right, (22, ly + 4), (56, ly + 4), PRED, 2)
        cv2.putText(right, "forecast path", (64, ly + 8), FONT, 0.4, TEXT, 1, cv2.LINE_AA)
        if a.verify:
            cv2.circle(right, (39, ly + 26), 5, TRUTH, -1, cv2.LINE_AA)
            cv2.putText(right, "what actually happened", (64, ly + 30),
                        FONT, 0.4, TEXT, 1, cv2.LINE_AA)

        if cam is not None:
            left = cam.copy()
            proj = lambda x, y: tuple(np.int32(world_to_px(Hinv, [(x, y)])[0]))
            draw_forecast(left, proj, people, trails, a, errors)
            header(left, "Camera view", "forecast projected onto the ground plane")
            out = np.hstack([left, np.full((left.shape[0], gap, 3), BG, np.uint8), right])
        else:
            out = right

        cv2.putText(out, f"frame {idx}   t={idx/fps:5.2f}s",
                    (out.shape[1] - 190, out.shape[0] - 14),
                    FONT, 0.44, MUTED, 1, cv2.LINE_AA)

        if idx == a.still_frame:
            cv2.imwrite(a.out_png, out)
            print(f"[S6v] Wrote still -> {a.out_png}")
            if a.still_only:
                break
        if not a.still_only:
            if vw is None:
                vw = cv2.VideoWriter(a.out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (out.shape[1], out.shape[0]))
                if not vw.isOpened():
                    raise SystemExit(f"[S6v] Could not open writer for {a.out_video}")
            vw.write(out)
            if (idx + 1) % 50 == 0:
                print(f"[S6v]  {idx+1} frames...")
        idx += 1

    if cap:
        cap.release()
    if vw:
        vw.release()
        print(f"[S6v] Wrote -> {a.out_video}")
    if err_hist:
        print(f"[S6v] Hindcast over {len(err_hist)} samples: "
              f"mean {np.mean(err_hist):.3f}m, median {np.median(err_hist):.3f}m, "
              f"p90 {np.percentile(err_hist,90):.3f}m at {horizon}s ahead")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecasts", default="outputs/stage6_rolling.json")
    ap.add_argument("--video", default=None)
    ap.add_argument("--homography", default="config/homography.json")
    ap.add_argument("--out-video", default="outputs/stage6_forecast.mp4")
    ap.add_argument("--out-png", default="outputs/stage6_forecast.png")
    ap.add_argument("--verify", action="store_true",
                    help="overlay hindcast: prediction vs what actually happened")
    ap.add_argument("--still-frame", type=int, default=60)
    ap.add_argument("--still-only", action="store_true")
    ap.add_argument("--ppm", type=float, default=85.0)
    ap.add_argument("--pad", type=float, default=1.0)
    ap.add_argument("--trail-sec", type=float, default=1.5)
    ap.add_argument("--err-label-min", type=float, default=0.4)
    ap.add_argument("--max-frames", type=int, default=0)
    main(ap.parse_args())