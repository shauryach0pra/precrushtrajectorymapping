"""
Stage 3c: Homography Demo Video - Camera Pixels -> BEV Metres
------------------------------------------------------------
Renders a side-by-side demo video showing what the homography actually does:

    LEFT  : original camera frame, tracked people, foot-points, and the
            4-point calibration quad you clicked in calibrate_homography.py
    RIGHT : the SAME ground plane rectified to a metric bird's-eye view,
            with a 1 m grid, the calibration quad in world coordinates,
            person dots + trails, and a live nearest-pair distance in metres

The visual payoff is that the cyan quad on the right lands exactly on its
world coordinates on the metric grid - that is the proof the calibration is
sane. The nearest-pair readout shows the thing pixels can never give you:
a real distance in metres.

Inputs (all already produced by earlier stages):
    data/cctv_clip.avi            - source video
    config/homography.json        - from calibrate_homography.py
    outputs/stage2_tracks.json    - from stage2 (YOLOv11 + BoTSORT/ByteTrack)

Usage (from project root):
    python src/stage3_bev_video.py \
        --video data/cctv_clip.avi \
        --homography config/homography.json \
        --tracks outputs/stage2_tracks.json \
        --out outputs/stage3_bev.mp4

Useful flags:
    --extent  xmin,xmax,ymin,ymax   force the BEV window (metres)
    --intro-sec 3.0                 opening sequence that builds the quad
    --trail-sec 2.0                 length of BEV motion trails
    --no-warp                       schematic BEV only (no rectified image)
"""

import argparse
import colorsys
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# palette (BGR)
# ----------------------------------------------------------------------------
BG = (24, 22, 20)
PANEL = (34, 31, 29)
GRID_MINOR = (58, 54, 50)
GRID_MAJOR = (86, 80, 74)
QUAD = (222, 214, 90)      # cyan-ish, used for the calibration quad in BOTH panels
TEXT = (238, 238, 238)
MUTED = (150, 146, 142)
ACCENT = (90, 190, 255)    # amber, used for the distance readout
FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(track_id: int):
    """Deterministic, well-spaced colour per track id."""
    h = (int(track_id) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.68, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def to_world(H, pts):
    """Pixel points (N,2) -> world metres (N,2) via the homography."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


class BEVCanvas:
    """Maps world metres to a bird's-eye raster. +y is drawn upward (away from camera)."""

    def __init__(self, xmin, xmax, ymin, ymax, height_px):
        self.xmin, self.xmax = float(xmin), float(xmax)
        self.ymin, self.ymax = float(ymin), float(ymax)
        self.h = int(height_px)
        self.ppm = self.h / (self.ymax - self.ymin)          # pixels per metre
        self.w = max(120, int(round((self.xmax - self.xmin) * self.ppm)))
        # world metres -> BEV pixels
        self.S = np.array([
            [self.ppm, 0.0, -self.xmin * self.ppm],
            [0.0, -self.ppm, self.ymax * self.ppm],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def pt(self, x, y):
        return (int(round((x - self.xmin) * self.ppm)),
                int(round((self.ymax - y) * self.ppm)))

    def warp_matrix(self, H):
        """Camera pixels -> BEV raster pixels."""
        return self.S @ H


# ----------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------
def dashed_line(img, p0, p1, color, thickness=1, dash=9):
    p0 = np.array(p0, float)
    p1 = np.array(p1, float)
    length = np.linalg.norm(p1 - p0)
    if length < 1:
        return
    n = max(2, int(length // dash))
    for i in range(n):
        if i % 2:
            continue
        a = p0 + (p1 - p0) * (i / n)
        b = p0 + (p1 - p0) * (min(i + 1, n) / n)
        cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), color, thickness, cv2.LINE_AA)


def label(img, text, org, color=TEXT, scale=0.46, thick=1, box=True, pad=4):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    # keep the label fully inside the panel
    x = int(min(max(pad, x), img.shape[1] - tw - pad))
    y = int(min(max(th + pad, y), img.shape[0] - base - pad))
    if box:
        overlay = img.copy()
        cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + base), (18, 16, 15), -1)
        cv2.addWeighted(overlay, 0.62, img, 0.38, 0, img)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_grid(canvas, bev):
    """1 m minor grid, 5 m major grid, with metre labels along the axes."""
    x0, x1 = int(np.floor(bev.xmin)), int(np.ceil(bev.xmax))
    y0, y1 = int(np.floor(bev.ymin)), int(np.ceil(bev.ymax))

    for xm in range(x0, x1 + 1):
        major = (xm % 5 == 0)
        p_top = bev.pt(xm, bev.ymax)
        p_bot = bev.pt(xm, bev.ymin)
        cv2.line(canvas, p_top, p_bot, GRID_MAJOR if major else GRID_MINOR, 1, cv2.LINE_AA)
        if major:
            cv2.putText(canvas, f"{xm}m", (p_bot[0] + 4, bev.h - 8), FONT, 0.4, MUTED, 1, cv2.LINE_AA)

    for ym in range(y0, y1 + 1):
        major = (ym % 5 == 0)
        p_l = bev.pt(bev.xmin, ym)
        p_r = bev.pt(bev.xmax, ym)
        cv2.line(canvas, p_l, p_r, GRID_MAJOR if major else GRID_MINOR, 1, cv2.LINE_AA)
        if major:
            cv2.putText(canvas, f"{ym}m", (6, p_l[1] - 5), FONT, 0.4, MUTED, 1, cv2.LINE_AA)


def draw_scale_bar(canvas, bev):
    """A 1 m reference bar so the viewer can eyeball distances."""
    x = bev.w - int(bev.ppm) - 22
    y = 26
    cv2.line(canvas, (x, y), (x + int(bev.ppm), y), TEXT, 2, cv2.LINE_AA)
    cv2.line(canvas, (x, y - 5), (x, y + 5), TEXT, 2, cv2.LINE_AA)
    cv2.line(canvas, (x + int(bev.ppm), y - 5), (x + int(bev.ppm), y + 5), TEXT, 2, cv2.LINE_AA)
    cv2.putText(canvas, "1 m", (x + int(bev.ppm) // 2 - 12, y - 10), FONT, 0.44, TEXT, 1, cv2.LINE_AA)


def draw_quad(img, pts, color=QUAD, thickness=2, labels=None, dot=5):
    pts_i = np.int32(pts).reshape(-1, 2)
    cv2.polylines(img, [pts_i], True, color, thickness, cv2.LINE_AA)
    for i, p in enumerate(pts_i):
        cv2.circle(img, tuple(p), dot + 2, (18, 16, 15), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(p), dot, color, -1, cv2.LINE_AA)
        if labels is not None:
            label(img, labels[i], (int(p[0]) + 10, int(p[1]) - 8), color, 0.42, 1)


def panel_header(img, title, subtitle=None):
    strip = img.copy()
    cv2.rectangle(strip, (0, 0), (img.shape[1], 44), (16, 15, 14), -1)
    cv2.addWeighted(strip, 0.78, img, 0.22, 0, img)
    cv2.putText(img, title, (14, 28), FONT, 0.6, TEXT, 1, cv2.LINE_AA)
    if subtitle:
        (tw, _), _ = cv2.getTextSize(title, FONT, 0.6, 1)
        cv2.putText(img, subtitle, (14 + tw + 14, 28), FONT, 0.46, MUTED, 1, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# main render
# ----------------------------------------------------------------------------
def build_extent(world_pts, traj_path, pad=1.5, override=None):
    if override:
        return [float(v) for v in override.split(",")]
    xs = list(world_pts[:, 0])
    ys = list(world_pts[:, 1])
    if traj_path and Path(traj_path).exists():
        data = json.loads(Path(traj_path).read_text())
        for fr in data:
            for p in fr.get("people", []):
                xs.append(p["x"])
                ys.append(p["y"])
    # trim wild outliers from bad tracks near the horizon
    xs = np.percentile(xs, [1, 99])
    ys = np.percentile(ys, [1, 99])
    return [xs[0] - pad, xs[1] + pad, ys[0] - pad, ys[1] + pad]


def render(args):
    calib = json.loads(Path(args.homography).read_text())
    H = np.array(calib["homography_matrix"], dtype=np.float64)
    px_pts = np.array(calib["pixel_points"], dtype=np.float64)
    world_pts = np.array(calib["world_points"], dtype=np.float64)

    tracks_by_frame = {}
    if args.tracks and Path(args.tracks).exists():
        for fr in json.loads(Path(args.tracks).read_text()):
            tracks_by_frame[fr["frame_idx"]] = fr.get("people", [])
        print(f"[Stage3v] Loaded tracks for {len(tracks_by_frame)} frames")
    else:
        print("[Stage3v] No tracks file - rendering geometry only")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"[Stage3v] Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Stage3v] Video {fw}x{fh} @ {fps:.2f} fps, {n_frames} frames")

    extent = build_extent(world_pts, args.trajectories, args.pad, args.extent)
    bev = BEVCanvas(*extent, height_px=fh)
    Hb = bev.warp_matrix(H)
    print(f"[Stage3v] BEV extent x=[{extent[0]:.1f}, {extent[1]:.1f}]m "
          f"y=[{extent[2]:.1f}, {extent[3]:.1f}]m  ({bev.ppm:.1f} px/m)")

    gap = 10
    out_w, out_h = fw + gap + bev.w, fh
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*args.fourcc), fps, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit(f"[Stage3v] Could not open writer for {args.out}")

    quad_world_labels = [f"({x:g}, {y:g})m" for x, y in world_pts]
    trail_len = max(1, int(args.trail_sec * fps))
    trails = defaultdict(lambda: deque(maxlen=trail_len))

    # ---- optional intro: build the calibration quad on a frozen frame -------
    if args.intro_sec > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(calib.get("calibration_frame", 0)))
        ok, intro_frame = cap.read()
        if ok:
            n_intro = int(args.intro_sec * fps)
            for i in range(n_intro):
                p = i / max(1, n_intro - 1)
                left = intro_frame.copy()
                n_show = min(4, int(p * 5.6))
                for k in range(n_show):
                    cv2.circle(left, tuple(np.int32(px_pts[k])), 7, (18, 16, 15), -1, cv2.LINE_AA)
                    cv2.circle(left, tuple(np.int32(px_pts[k])), 5, QUAD, -1, cv2.LINE_AA)
                    label(left, f"{k+1}  {quad_world_labels[k]}",
                          (int(px_pts[k][0]) + 11, int(px_pts[k][1]) - 9), QUAD, 0.42, 1)
                if n_show >= 4:
                    cv2.polylines(left, [np.int32(px_pts)], True, QUAD, 2, cv2.LINE_AA)
                panel_header(left, "Camera view", "click 4 known ground points")

                right = np.full((bev.h, bev.w, 3), PANEL, np.uint8)
                draw_grid(right, bev)
                if p > 0.62 and not args.no_warp:
                    warped = cv2.warpPerspective(intro_frame, Hb, (bev.w, bev.h),
                                                 borderValue=PANEL)
                    a = min(1.0, (p - 0.62) / 0.28) * 0.75
                    cv2.addWeighted(warped, a, right, 1 - a, 0, right)
                    draw_grid(right, bev)
                if p > 0.62:
                    draw_quad(right, [bev.pt(x, y) for x, y in world_pts],
                              labels=quad_world_labels)
                draw_scale_bar(right, bev)
                panel_header(right, "Bird's-eye view", "same plane, in metres")

                writer.write(np.hstack([left,
                                        np.full((fh, gap, 3), BG, np.uint8),
                                        right]))
            print(f"[Stage3v] Wrote {n_intro} intro frames")

    # ---- main pass ---------------------------------------------------------
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and written >= args.max_frames:
            break

        people = tracks_by_frame.get(idx, [])

        # ---------------- left panel: camera space -------------------------
        left = frame.copy()
        cv2.polylines(left, [np.int32(px_pts)], True, QUAD, 2, cv2.LINE_AA)
        for k, p in enumerate(px_pts):
            cv2.circle(left, tuple(np.int32(p)), 5, QUAD, -1, cv2.LINE_AA)

        world_now = {}
        for pr in people:
            tid = pr["track_id"]
            c = color_for(tid)
            fx, fy = pr["foot_x"], pr["foot_y"]
            if "bbox" in pr:
                x1, y1, x2, y2 = [int(round(v)) for v in pr["bbox"]]
                cv2.rectangle(left, (x1, y1), (x2, y2), c, 1, cv2.LINE_AA)
                label(left, f"#{tid}", (x1, max(12, y1 - 6)), c, 0.4, 1)
            cv2.circle(left, (int(round(fx)), int(round(fy))), 4, c, -1, cv2.LINE_AA)
            cv2.circle(left, (int(round(fx)), int(round(fy))), 6, (18, 16, 15), 1, cv2.LINE_AA)
            wx, wy = to_world(H, [(fx, fy)])[0]
            world_now[tid] = (wx, wy)
            trails[tid].append((wx, wy))

        panel_header(left, "Camera pixels",
                     f"frame {idx}  |  t={idx/fps:5.2f}s  |  {len(people)} tracked")

        # ---------------- right panel: BEV metres --------------------------
        right = np.full((bev.h, bev.w, 3), PANEL, np.uint8)
        if not args.no_warp:
            warped = cv2.warpPerspective(frame, Hb, (bev.w, bev.h), borderValue=PANEL)
            cv2.addWeighted(warped, 0.55, right, 0.45, 0, right)
        draw_grid(right, bev)
        draw_quad(right, [bev.pt(x, y) for x, y in world_pts], thickness=2)

        for tid, pts in trails.items():
            if tid not in world_now or len(pts) < 2:
                continue
            c = color_for(tid)
            arr = [bev.pt(x, y) for x, y in pts]
            for i in range(1, len(arr)):
                a = i / len(arr)
                cv2.line(right, arr[i - 1], arr[i],
                         tuple(int(v * (0.25 + 0.75 * a)) for v in c), 2, cv2.LINE_AA)

        for tid, (wx, wy) in world_now.items():
            c = color_for(tid)
            p = bev.pt(wx, wy)
            cv2.circle(right, p, 7, (18, 16, 15), -1, cv2.LINE_AA)
            cv2.circle(right, p, 5, c, -1, cv2.LINE_AA)
            cv2.putText(right, f"#{tid}", (p[0] + 9, p[1] - 8), FONT, 0.38, c, 1, cv2.LINE_AA)

        # nearest pair - the measurement you cannot make in pixel space
        if len(world_now) >= 2:
            ids = list(world_now)
            best, bd = None, 1e9
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = world_now[ids[i]], world_now[ids[j]]
                    d = float(np.hypot(a[0] - b[0], a[1] - b[1]))
                    if d < bd:
                        bd, best = d, (ids[i], ids[j])
            if best and bd < args.max_pair_dist:
                pa, pb = bev.pt(*world_now[best[0]]), bev.pt(*world_now[best[1]])
                dashed_line(right, pa, pb, ACCENT, 2)
                mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
                label(right, f"{bd:.2f} m", (mid[0] - 22, mid[1] - 10), ACCENT, 0.5, 1)

        draw_scale_bar(right, bev)
        panel_header(right, "Bird's-eye metres", "ground plane rectified via H")

        writer.write(np.hstack([left, np.full((fh, gap, 3), BG, np.uint8), right]))
        idx += 1
        written += 1
        if written % 50 == 0:
            print(f"[Stage3v]  {written} frames...")

    cap.release()
    writer.release()
    print(f"[Stage3v] Wrote {written} frames -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/cctv_clip.avi")
    ap.add_argument("--homography", default="config/homography.json")
    ap.add_argument("--tracks", default="outputs/stage2_tracks.json")
    ap.add_argument("--trajectories", default="outputs/world_trajectories.json",
                    help="only used to auto-size the BEV window")
    ap.add_argument("--out", default="outputs/stage3_bev.mp4")
    ap.add_argument("--extent", default=None, help="xmin,xmax,ymin,ymax in metres")
    ap.add_argument("--pad", type=float, default=1.5)
    ap.add_argument("--intro-sec", type=float, default=3.0)
    ap.add_argument("--trail-sec", type=float, default=2.0)
    ap.add_argument("--max-pair-dist", type=float, default=6.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-warp", action="store_true")
    ap.add_argument("--fourcc", default="mp4v")
    render(ap.parse_args())