"""
Stage 1 + 2 Demo: YOLOv11 Detection and BoTSORT Tracking
--------------------------------------------------------
Draws saved detections and tracks back onto the ORIGINAL video clip.
Re-runs nothing - no YOLO weights, no GPU, no Ultralytics needed. It reads
outputs/stage1_detections.json and outputs/stage2_tracks.json and overlays
them on data/cctv_clip.avi.

Modes:
  --mode detect   Stage 1 only. Uniform boxes + confidence. No identity:
                  box colour is confidence, and nothing persists between
                  frames. This is the "before" half of the story.
  --mode track    Stage 2 only. One stable colour and #ID per person, plus
                  a motion trail through the foot-points.
  --mode both     (default) side-by-side, detect | track. The strongest
                  demo, because the trails on the right are visibly absent
                  on the left - that is exactly what the tracker adds.

Usage (from project root):
    python src/stage12_detect_track_video.py
    python src/stage12_detect_track_video.py --mode track
    python src/stage12_detect_track_video.py --still-frame 40 --still-only
"""

import argparse
import colorsys
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

TEXT = (238, 238, 238)
MUTED = (150, 146, 142)
GAPC = (26, 24, 22)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(tid):
    """Stable, well-spaced colour per track id (golden-ratio hue hopping)."""
    h = (int(tid) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.72, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def conf_color(c):
    """Low confidence = amber, high = green."""
    c = float(np.clip((c - 0.25) / 0.55, 0, 1))
    return (int(60 + 40 * c), int(120 + 110 * c), int(250 - 160 * c))


def label(img, text, org, color=TEXT, scale=0.42, thick=1, pad=3):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x = int(min(max(pad, org[0]), img.shape[1] - tw - pad))
    y = int(min(max(th + pad, org[1]), img.shape[0] - base - pad))
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base), (16, 15, 14), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def header(img, title, subtitle):
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (img.shape[1], 46), (14, 13, 12), -1)
    cv2.addWeighted(ov, 0.78, img, 0.22, 0, img)
    cv2.putText(img, title, (14, 22), FONT, 0.55, TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, subtitle, (14, 39), FONT, 0.42, MUTED, 1, cv2.LINE_AA)


def draw_detect(frame, dets, args):
    """Stage 1 panel: boxes only, no identity."""
    img = frame.copy()
    kept = 0
    for d in dets:
        if d["conf"] < args.min_conf:
            continue
        kept += 1
        x1, y1, x2, y2 = [int(round(v)) for v in d["bbox"]]
        c = conf_color(d["conf"])
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2, cv2.LINE_AA)
        if args.show_conf:
            label(img, f"{d['conf']:.2f}", (x1, max(14, y1 - 5)), c, 0.36, 1)
    header(img, "Stage 1  -  YOLOv11 person detection",
           f"{kept} boxes this frame  |  no identity, nothing persists")
    return img


def draw_track(frame, people, trails, args):
    """Stage 2 panel: stable IDs plus motion trails."""
    img = frame.copy()

    for tid, pts in trails.items():
        if len(pts) < 2:
            continue
        c = color_for(tid)
        for i in range(1, len(pts)):
            a = i / len(pts)
            cv2.line(img, pts[i - 1], pts[i],
                     tuple(int(v * (0.2 + 0.8 * a)) for v in c), 2, cv2.LINE_AA)

    for p in people:
        tid = p["track_id"]
        c = color_for(tid)
        x1, y1, x2, y2 = [int(round(v)) for v in p["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2, cv2.LINE_AA)
        label(img, f"#{tid}", (x1, max(14, y1 - 5)), c, 0.42, 1)
        fx, fy = int(round(p["foot_x"])), int(round(p["foot_y"]))
        cv2.circle(img, (fx, fy), 4, c, -1, cv2.LINE_AA)
        cv2.circle(img, (fx, fy), 6, (18, 16, 15), 1, cv2.LINE_AA)

    header(img, "Stage 2  -  BoTSORT tracking",
           f"{len(people)} persistent IDs  |  trails = per-person trajectory")
    return img


def main(args):
    det_by_frame, trk_by_frame = {}, {}
    if args.mode in ("detect", "both"):
        for f in json.loads(Path(args.detections).read_text()):
            det_by_frame[f["frame_idx"]] = f.get("detections", [])
        print(f"[S12] Detections for {len(det_by_frame)} frames")
    if args.mode in ("track", "both"):
        for f in json.loads(Path(args.tracks).read_text()):
            trk_by_frame[f["frame_idx"]] = f.get("people", [])
        print(f"[S12] Tracks for {len(trk_by_frame)} frames")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"[S12] Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[S12] Video {w}x{h} @ {fps:.2f} fps  |  mode={args.mode}")

    gap = 10
    modes = ["detect", "track", "both"] if args.mode == "all" else [args.mode]
    out_paths = {
        "detect": (args.out_detect, args.out_detect.replace(".mp4", ".png")),
        "track": (args.out_track, args.out_track.replace(".mp4", ".png")),
        "both": (args.out_video, args.out_png),
    }
    Path(args.out_video).parent.mkdir(parents=True, exist_ok=True)

    trail_len = max(2, int(args.trail_sec * fps))
    trails = defaultdict(lambda: deque(maxlen=trail_len))

    writers = {}
    if not args.still_only:
        for m in modes:
            ww = w * 2 + gap if m == "both" else w
            vw = cv2.VideoWriter(out_paths[m][0], cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, h))
            if not vw.isOpened():
                raise SystemExit(f"[S12] Could not open writer for {out_paths[m][0]}")
            writers[m] = vw

    idx, written = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and written >= args.max_frames:
            break

        people = trk_by_frame.get(idx, [])
        for p in people:
            trails[p["track_id"]].append((int(round(p["foot_x"])), int(round(p["foot_y"]))))

        panels = {}
        if "detect" in modes or "both" in modes:
            panels["detect"] = draw_detect(frame, det_by_frame.get(idx, []), args)
        if "track" in modes or "both" in modes:
            panels["track"] = draw_track(frame, people, trails, args)
        if "both" in modes:
            panels["both"] = np.hstack([panels["detect"],
                                        np.full((h, gap, 3), GAPC, np.uint8),
                                        panels["track"]])

        for m in modes:
            out = panels[m]
            cv2.putText(out, f"frame {idx}   t={idx/fps:5.2f}s",
                        (out.shape[1] - 190, out.shape[0] - 14),
                        FONT, 0.44, MUTED, 1, cv2.LINE_AA)
            if idx == args.still_frame:
                cv2.imwrite(out_paths[m][1], out)
                print(f"[S12] Wrote still -> {out_paths[m][1]}")
            if m in writers:
                writers[m].write(out)

        if idx == args.still_frame and args.still_only:
            break

        if writers:
            written += 1
            if written % 50 == 0:
                print(f"[S12]  {written} frames...")
        idx += 1

    cap.release()
    for m, vw in writers.items():
        vw.release()
        print(f"[S12] Wrote {written} frames -> {out_paths[m][0]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/cctv_clip.avi")
    ap.add_argument("--detections", default="outputs/stage1_detections.json")
    ap.add_argument("--tracks", default="outputs/stage2_tracks.json")
    ap.add_argument("--mode", choices=["detect", "track", "both", "all"], default="all")
    ap.add_argument("--out-video", default="outputs/stage12_detect_track.mp4")
    ap.add_argument("--out-png", default="outputs/stage12_detect_track.png")
    ap.add_argument("--out-detect", default="outputs/stage1_detection.mp4")
    ap.add_argument("--out-track", default="outputs/stage2_tracking.mp4")
    ap.add_argument("--still-frame", type=int, default=40)
    ap.add_argument("--still-only", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.25)
    ap.add_argument("--show-conf", action="store_true", default=True)
    ap.add_argument("--trail-sec", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=0)
    main(ap.parse_args())