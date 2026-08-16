"""
Stage 4 + 5 Demo: Social Force Model + KD-Tree Neighbour Search
---------------------------------------------------------------
Renders a bird's-eye visualisation of what stage4_social_force.py computes.
Needs NO video and NO model weights - it reads outputs/stage4_forces.json only.

What you see:
  - every tracked person as a dot in world metres, on a 1 m grid
  - a muted arrow  = observed velocity (where they ARE going)
  - a bright arrow = net social force (where the model PUSHES them)
  - one highlighted "focus" person with the 3 m NEIGHBOR_RADIUS ring drawn,
    and thin spokes to exactly the neighbours cKDTree.query_ball_point()
    returns - that is Stage 5 made visible
  - a live counter comparing KD-tree pair checks vs brute-force O(n^2)

Usage:
    python src/stage45_force_video.py                       # video + hero PNG
    python src/stage45_force_video.py --still-frame 9       # just the PNG
"""

import argparse
import colorsys
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

NEIGHBOR_RADIUS = 3.0     # must match stage4_social_force.py
PERSON_RADIUS = 0.3

BG = (26, 24, 22)
GRID_MINOR = (52, 48, 45)
GRID_MAJOR = (80, 74, 69)
TEXT = (238, 238, 238)
MUTED = (150, 146, 142)
VEL = (150, 200, 130)      # green - observed velocity
FORCE = (90, 170, 255)     # amber - net social force
RING = (222, 214, 90)      # cyan - KD-tree query radius
SPOKE = (170, 165, 80)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(tid):
    h = (int(tid) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


class BEV:
    def __init__(self, xmin, xmax, ymin, ymax, ppm):
        self.xmin, self.xmax, self.ymin, self.ymax, self.ppm = xmin, xmax, ymin, ymax, ppm
        self.w = int(round((xmax - xmin) * ppm))
        self.h = int(round((ymax - ymin) * ppm))

    def pt(self, x, y):
        return (int(round((x - self.xmin) * self.ppm)),
                int(round((self.ymax - y) * self.ppm)))


def label(img, text, org, color=TEXT, scale=0.45, thick=1, pad=4):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x = int(min(max(pad, org[0]), img.shape[1] - tw - pad))
    y = int(min(max(th + pad, org[1]), img.shape[0] - base - pad))
    ov = img.copy()
    cv2.rectangle(ov, (x - pad, y - th - pad), (x + tw + pad, y + base), (16, 15, 14), -1)
    cv2.addWeighted(ov, 0.66, img, 0.34, 0, img)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def arrow(img, p0, p1, color, thick=2, tip=0.3):
    if abs(p1[0] - p0[0]) + abs(p1[1] - p0[1]) < 3:
        return
    cv2.arrowedLine(img, p0, p1, color, thick, cv2.LINE_AA, tipLength=tip)


def draw_grid(img, bev):
    for xm in range(int(np.floor(bev.xmin)), int(np.ceil(bev.xmax)) + 1):
        major = xm % 5 == 0
        cv2.line(img, bev.pt(xm, bev.ymax), bev.pt(xm, bev.ymin),
                 GRID_MAJOR if major else GRID_MINOR, 1, cv2.LINE_AA)
        if major:
            cv2.putText(img, f"{xm}m", (bev.pt(xm, bev.ymin)[0] + 4, bev.h - 8),
                        FONT, 0.4, MUTED, 1, cv2.LINE_AA)
    for ym in range(int(np.floor(bev.ymin)), int(np.ceil(bev.ymax)) + 1):
        major = ym % 5 == 0
        cv2.line(img, bev.pt(bev.xmin, ym), bev.pt(bev.xmax, ym),
                 GRID_MAJOR if major else GRID_MINOR, 1, cv2.LINE_AA)
        if major:
            cv2.putText(img, f"{ym}m", (6, bev.pt(bev.xmin, ym)[1] - 5),
                        FONT, 0.4, MUTED, 1, cv2.LINE_AA)


def world_to_px(Hinv, pts):
    """World metres (N,2) -> camera pixels (N,2) using the inverse homography."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2)


def render_camera_panel(frame_img, people, positions, fi, neigh, Hinv, args):
    """Draws the same forces back onto the real camera frame via H^-1."""
    img = frame_img.copy()
    h, w = img.shape[:2]

    def ok(p):
        return -4 * w < p[0] < 4 * w and -4 * h < p[1] < 4 * h

    # KD-tree search radius, projected -> becomes a perspective ellipse on the ground
    ring_world = [(positions[fi][0] + NEIGHBOR_RADIUS * np.cos(t),
                   positions[fi][1] + NEIGHBOR_RADIUS * np.sin(t))
                  for t in np.linspace(0, 2 * np.pi, 72)]
    ring_px = world_to_px(Hinv, ring_world)
    if all(ok(p) for p in ring_px):
        ov = img.copy()
        cv2.fillPoly(ov, [np.int32(ring_px)], (55, 55, 25))
        cv2.addWeighted(ov, 0.28, img, 0.72, 0, img)
        cv2.polylines(img, [np.int32(ring_px)], True, RING, 2, cv2.LINE_AA)

    fc_px = world_to_px(Hinv, [positions[fi]])[0]

    for j in neigh:
        pj = world_to_px(Hinv, [positions[j]])[0]
        if ok(pj) and ok(fc_px):
            cv2.line(img, tuple(np.int32(fc_px)), tuple(np.int32(pj)), SPOKE, 1, cv2.LINE_AA)

    for i, p in enumerate(people):
        c = color_for(p["track_id"])
        base = world_to_px(Hinv, [(p["x"], p["y"])])[0]
        if not ok(base):
            continue
        b = tuple(np.int32(base))

        fm = float(np.hypot(p["force_x"], p["force_y"]))
        if fm > 1e-2:
            s = min(fm, args.force_clamp) / max(fm, 1e-6)
            tipw = (p["x"] + p["force_x"] * s * args.force_scale,
                    p["y"] + p["force_y"] * s * args.force_scale)
            tip = world_to_px(Hinv, [tipw])[0]
            if ok(tip):
                arrow(img, b, tuple(np.int32(tip)), FORCE, 2, 0.3)

        cv2.circle(img, b, 5, (18, 16, 15), -1, cv2.LINE_AA)
        cv2.circle(img, b, 3, c, -1, cv2.LINE_AA)

    if ok(fc_px):
        cv2.circle(img, tuple(np.int32(fc_px)), 8, (18, 16, 15), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(np.int32(fc_px)), 6, RING, -1, cv2.LINE_AA)

    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (w, 44), (14, 13, 12), -1)
    cv2.addWeighted(ov, 0.78, img, 0.22, 0, img)
    cv2.putText(img, "Camera view", (14, 28), FONT, 0.58, TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, "forces projected back via H-inverse", (150, 28), FONT, 0.44, MUTED, 1, cv2.LINE_AA)
    return img


def pick_focus(people, positions, tree):
    """Choose the person with the most neighbours - the most interesting one to explain."""
    best, best_n = 0, -1
    for i in range(len(people)):
        n = len(tree.query_ball_point(positions[i], r=NEIGHBOR_RADIUS)) - 1
        if n > best_n:
            best, best_n = i, n
    return best, best_n


def render_frame(frame, bev, args, focus_lock=None):
    people = frame["people"]
    img = np.full((bev.h, bev.w, 3), BG, np.uint8)
    draw_grid(img, bev)

    if not people:
        return img, None, None

    positions = np.array([[p["x"], p["y"]] for p in people])
    tree = cKDTree(positions)

    # pick focus person (sticky by track_id across frames if possible)
    fi = None
    if focus_lock is not None:
        for i, p in enumerate(people):
            if p["track_id"] == focus_lock:
                fi = i
                break
    if fi is None:
        fi, _ = pick_focus(people, positions, tree)

    neigh = [j for j in tree.query_ball_point(positions[fi], r=NEIGHBOR_RADIUS) if j != fi]

    # --- KD-tree ring + spokes (Stage 5) ---------------------------------
    fc = bev.pt(*positions[fi])
    ov = img.copy()
    cv2.circle(ov, fc, int(NEIGHBOR_RADIUS * bev.ppm), (60, 60, 30), -1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.30, img, 0.70, 0, img)
    cv2.circle(img, fc, int(NEIGHBOR_RADIUS * bev.ppm), RING, 2, cv2.LINE_AA)
    for j in neigh:
        pj = bev.pt(*positions[j])
        cv2.line(img, fc, pj, SPOKE, 1, cv2.LINE_AA)
        d = float(np.linalg.norm(positions[fi] - positions[j]))
        mid = ((fc[0] + pj[0]) // 2, (fc[1] + pj[1]) // 2)
        if args.show_pair_dist:
            cv2.putText(img, f"{d:.1f}", (mid[0] + 3, mid[1] - 3), FONT, 0.34, SPOKE, 1, cv2.LINE_AA)

    # --- people, velocity and force vectors (Stage 4) --------------------
    for i, p in enumerate(people):
        c = color_for(p["track_id"])
        pc = bev.pt(p["x"], p["y"])

        # personal-space disc
        cv2.circle(img, pc, max(2, int(PERSON_RADIUS * bev.ppm)), tuple(int(v * 0.35) for v in c), -1, cv2.LINE_AA)

        # velocity arrow (clamped - raw tracks contain ID-switch spikes)
        sp = float(np.hypot(p["vx"], p["vy"]))
        if sp > 1e-2:
            s = min(sp, args.speed_clamp) / max(sp, 1e-6)
            vend = bev.pt(p["x"] + p["vx"] * s * args.vel_scale,
                          p["y"] + p["vy"] * s * args.vel_scale)
            arrow(img, pc, vend, VEL, 1, 0.35)

        # net social force arrow
        fm = float(np.hypot(p["force_x"], p["force_y"]))
        if fm > 1e-2:
            s = min(fm, args.force_clamp) / max(fm, 1e-6)
            fend = bev.pt(p["x"] + p["force_x"] * s * args.force_scale,
                          p["y"] + p["force_y"] * s * args.force_scale)
            arrow(img, pc, fend, FORCE, 2, 0.3)

        cv2.circle(img, pc, 5, (18, 16, 15), -1, cv2.LINE_AA)
        cv2.circle(img, pc, 3, c, -1, cv2.LINE_AA)

    # focus person on top
    cv2.circle(img, fc, 8, (18, 16, 15), -1, cv2.LINE_AA)
    cv2.circle(img, fc, 6, RING, -1, cv2.LINE_AA)

    # --- HUD --------------------------------------------------------------
    n = len(people)
    kd_pairs = sum(len(tree.query_ball_point(positions[i], r=NEIGHBOR_RADIUS)) - 1
                   for i in range(n)) // 2
    brute = n * (n - 1) // 2

    hud = img.copy()
    cv2.rectangle(hud, (0, 0), (bev.w, 96), (14, 13, 12), -1)
    cv2.addWeighted(hud, 0.80, img, 0.20, 0, img)
    cv2.putText(img, "Social Force Model + KD-Tree neighbour search",
                (14, 26), FONT, 0.58, TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, f"frame {frame['frame_idx']}  t={frame['timestamp_sec']:.2f}s  |  {n} people",
                (14, 48), FONT, 0.44, MUTED, 1, cv2.LINE_AA)
    cv2.putText(img, f"cKDTree r={NEIGHBOR_RADIUS:g}m  ->  {kd_pairs} interacting pairs "
                     f"(brute force would test {brute})",
                (14, 68), FONT, 0.44, RING, 1, cv2.LINE_AA)
    cv2.putText(img, f"focus #{people[fi]['track_id']}: {len(neigh)} neighbours inside radius",
                (14, 88), FONT, 0.44, MUTED, 1, cv2.LINE_AA)

    # legend
    ly = bev.h - 74
    cv2.rectangle(img, (10, ly - 16), (250, bev.h - 12), (14, 13, 12), -1)
    cv2.arrowedLine(img, (22, ly + 4), (58, ly + 4), VEL, 1, cv2.LINE_AA, tipLength=0.35)
    cv2.putText(img, "observed velocity", (66, ly + 8), FONT, 0.4, TEXT, 1, cv2.LINE_AA)
    cv2.arrowedLine(img, (22, ly + 26), (58, ly + 26), FORCE, 2, cv2.LINE_AA, tipLength=0.3)
    cv2.putText(img, "net social force", (66, ly + 30), FONT, 0.4, TEXT, 1, cv2.LINE_AA)
    cv2.circle(img, (40, ly + 48), 9, RING, 2, cv2.LINE_AA)
    cv2.putText(img, "KD-tree search radius", (66, ly + 52), FONT, 0.4, TEXT, 1, cv2.LINE_AA)

    return img, people[fi]["track_id"], (people, positions, fi, neigh)


def main(args):
    frames = json.loads(Path(args.forces).read_text())
    xs = [p["x"] for f in frames for p in f["people"]]
    ys = [p["y"] for f in frames for p in f["people"]]
    xlo, xhi = np.percentile(xs, [1, 99])
    ylo, yhi = np.percentile(ys, [1, 99])

    # ---- optional camera panel -------------------------------------------
    cap, Hinv, cam_w, cam_h = None, None, 0, 0
    if args.video:
        if not Path(args.homography).exists():
            raise SystemExit(f"[S45] --video needs --homography; not found: {args.homography}")
        H = np.array(json.loads(Path(args.homography).read_text())["homography_matrix"])
        Hinv = np.linalg.inv(H)
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"[S45] Could not open video: {args.video}")
        cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[S45] Camera panel ON - {cam_w}x{cam_h}")

    # match BEV height to the video so the panels line up
    ppm = (cam_h / (yhi - ylo + 2 * args.pad)) if cap else args.ppm
    bev = BEV(xlo - args.pad, xhi + args.pad, ylo - args.pad, yhi + args.pad, ppm)
    print(f"[S45] BEV {bev.w}x{bev.h}px  x=[{bev.xmin:.1f},{bev.xmax:.1f}]m "
          f"y=[{bev.ymin:.1f},{bev.ymax:.1f}]m")

    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    gap = 10

    def compose(bev_img, ctx, cam_img):
        if cam_img is None or ctx is None:
            return bev_img
        left = render_camera_panel(cam_img, ctx[0], ctx[1], ctx[2], ctx[3], Hinv, args)
        if left.shape[0] != bev_img.shape[0]:
            bev_img = cv2.resize(bev_img, (bev_img.shape[1], left.shape[0]))
        return np.hstack([left, np.full((left.shape[0], gap, 3), BG, np.uint8), bev_img])

    # ---- hero still ------------------------------------------------------
    fr = next((f for f in frames if f["frame_idx"] == args.still_frame), frames[0])
    still, _, ctx = render_frame(fr, bev, args)
    cam_img = None
    if cap:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr["frame_idx"])
        ok, cam_img = cap.read()
        if not ok:
            cam_img = None
    cv2.imwrite(args.out_png, compose(still, ctx, cam_img))
    print(f"[S45] Wrote still -> {args.out_png}")

    if args.still_only:
        if cap:
            cap.release()
        return

    # ---- video -----------------------------------------------------------
    probe = compose(still, ctx, cam_img)
    vw = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (probe.shape[1], probe.shape[0]))
    if not vw.isOpened():
        raise SystemExit(f"[S45] Could not open writer for {args.out_video}")

    if cap:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    lock = None
    for k, f in enumerate(frames):
        cam_img = None
        if cap:
            ok, cam_img = cap.read()
            if not ok:
                print(f"[S45] Video ended at frame {k}; stopping")
                break
        img, lock, ctx = render_frame(f, bev, args, focus_lock=lock)
        out = compose(img, ctx, cam_img)
        if out.shape[:2] != probe.shape[:2]:
            out = cv2.resize(out, (probe.shape[1], probe.shape[0]))
        vw.write(out)
        if (k + 1) % 50 == 0:
            print(f"[S45]  {k+1}/{len(frames)} frames...")
    vw.release()
    if cap:
        cap.release()
    print(f"[S45] Wrote video -> {args.out_video}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--forces", default="outputs/stage4_forces.json")
    ap.add_argument("--video", default=None,
                    help="optional: camera clip, adds a side-by-side camera panel")
    ap.add_argument("--homography", default="config/homography.json")
    ap.add_argument("--out-video", default="outputs/stage45_forces.mp4")
    ap.add_argument("--out-png", default="outputs/stage45_forces.png")
    ap.add_argument("--still-frame", type=int, default=9)
    ap.add_argument("--still-only", action="store_true")
    ap.add_argument("--ppm", type=float, default=90.0, help="pixels per metre")
    ap.add_argument("--pad", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--vel-scale", type=float, default=0.55)
    ap.add_argument("--force-scale", type=float, default=0.30)
    ap.add_argument("--speed-clamp", type=float, default=2.5, help="m/s, hides ID-switch spikes")
    ap.add_argument("--force-clamp", type=float, default=5.0)
    ap.add_argument("--show-pair-dist", action="store_true")
    main(ap.parse_args())