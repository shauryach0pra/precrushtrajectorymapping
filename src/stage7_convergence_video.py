"""
Stage 7 Demo: Density Convergence Engine
------------------------------------------
Turns 2D world coordinates into a persons/m^2 field, then does the thing the
pipeline is actually for: builds the SAME field from Stage 6's forecast
positions, so you see density that is about to form, not density that
already formed.

Three panels:
  LEFT   camera view with the current density warped onto the ground plane
  RIGHT  bird's-eye density with Fruin Level-of-Service bands, people dots,
         and forecast arrows showing who is converging on the hot zone
  BOTTOM timeline of current (solid) and predicted (dashed) peak density
         against the LOS thresholds, so the whole clip reads at a glance

And a verdict banner that is ALWAYS shown, green included. A safety system
that only speaks when it is alarmed is indistinguishable from a broken one.
On calm footage the correct output is a steady green "NO CRUSH RISK" - that
is a true negative, and it is a real result worth showing.

Fruin/Still LOS bands (persons/m^2) are industry heuristics, not law.
Validate against your own scene before anyone relies on them.

Usage:
    python src/stage7_convergence_video.py \
        --trajectories outputs/world_trajectories.json \
        --forecasts outputs/stage6_rolling.json \
        --homography config/homography.json \
        --video data/cctv_clip.avi
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

GRID_RES_M = 0.15
PERSON_SIGMA_M = 0.45

# (upper bound, label, BGR)
LOS = [
    (1.0, "COMFORTABLE", (90, 190, 90)),
    (2.0, "BUSY", (90, 210, 225)),
    (4.0, "CONSTRAINED", (60, 150, 255)),
    (6.0, "HIGH RISK", (60, 80, 255)),
    (1e9, "CRUSH RISK", (40, 40, 190)),
]

TEXT = (238, 238, 238)
MUTED = (150, 146, 142)
BG = (26, 24, 22)
CUR = (235, 235, 235)
PRD = (90, 190, 255)
PRESS = (200, 120, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def band(d):
    for hi, name, col in LOS:
        if d < hi:
            return name, col
    return LOS[-1][1], LOS[-1][2]


def density_field(pts, x0, y0, gw, gh):
    f = np.zeros((gh, gw), np.float32)
    if len(pts) == 0:
        return f
    pts = np.asarray(pts, float)
    c = ((pts[:, 0] - x0) / GRID_RES_M).astype(int)
    r = ((pts[:, 1] - y0) / GRID_RES_M).astype(int)
    ok = (c >= 0) & (c < gw) & (r >= 0) & (r < gh)
    for cc, rr in zip(c[ok], r[ok]):
        f[rr, cc] += 1.0
    f = gaussian_filter(f, sigma=PERSON_SIGMA_M / GRID_RES_M, mode="constant")
    return f / (GRID_RES_M ** 2)      # mass per cell -> persons/m^2


def colorize(field):
    out = np.zeros((*field.shape, 3), np.uint8)
    lo = 0.0
    for hi, _, col in LOS:
        m = (field >= lo) & (field < hi)
        out[m] = col
        lo = hi
    out[field < 0.15] = 0              # leave near-empty ground uncoloured
    return out


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


def timeline(width, height, cur, prd, idx, total, crush_thr, warn_thr, horizon,
             press=None, press_thr=None):
    """
    Slim strip: current vs predicted peak density over the clip.

    The amber fill between the two lines is the whole product. When it opens
    up, predicted density has pulled ahead of current - people are converging
    and the crowd is about to compress. That gap is the lead time. Detection
    alone cannot draw it, because detection only knows about now.
    """
    img = np.full((height, width, 3), (21, 20, 19), np.uint8)
    pad_l, pad_r, pad_t, pad_b = 52, 250, 20, 16
    plot_w = width - pad_l - pad_r
    top_d = max(crush_thr * 1.15, (max(max(cur), max(prd)) if cur else 1.0) * 1.25)

    def X(i):
        return int(pad_l + plot_w * i / max(1, total - 1))

    def Y(d):
        return int(height - pad_b - (height - pad_b - pad_t) * min(d, top_d) / top_d)

    # thresholds as thin rules, not heavy colour blocks
    for thr, col, txt in ((warn_thr, (110, 150, 170), f"{warn_thr:g}"),
                          (crush_thr, (70, 70, 190), f"crush {crush_thr:g}")):
        if thr < top_d:
            y = Y(thr)
            for x in range(pad_l, width - pad_r, 9):
                cv2.line(img, (x, y), (x + 4, y), col, 1)
            cv2.putText(img, txt, (6, y + 4), FONT, 0.34, col, 1, cv2.LINE_AA)

    n = len(cur)
    if n > 1:
        # amber fill wherever the forecast exceeds the present
        poly_top, poly_bot = [], []
        for i in range(n):
            if prd[i] > cur[i] + 0.02:
                poly_top.append((X(i), Y(prd[i])))
                poly_bot.append((X(i), Y(cur[i])))
            elif poly_top:
                cv2.fillPoly(img, [np.int32(poly_top + poly_bot[::-1])], (40, 95, 140))
                poly_top, poly_bot = [], []
        if poly_top:
            cv2.fillPoly(img, [np.int32(poly_top + poly_bot[::-1])], (40, 95, 140))

        for series, col, th in ((prd, PRD, 1), (cur, CUR, 2)):
            pts = np.int32([(X(i), Y(series[i])) for i in range(n)])
            cv2.polylines(img, [pts], False, col, th, cv2.LINE_AA)
        cv2.circle(img, (X(n - 1), Y(cur[-1])), 4, CUR, -1, cv2.LINE_AA)

    # crowd pressure on its own normalised axis - different units (1/s^2),
    # so it shares the strip but not the scale.
    if press and press_thr:
        pmax = max(max(press), press_thr) * 1.15
        ppts = np.int32([(X(i), Y(top_d * press[i] / pmax)) for i in range(len(press))])
        cv2.polylines(img, [ppts], False, PRESS, 1, cv2.LINE_AA)
        yt = Y(top_d * press_thr / pmax)
        for x in range(pad_l, width - pad_r, 14):
            cv2.line(img, (x, yt), (x + 5, yt), (90, 70, 150), 1)

    cv2.putText(img, "peak persons/m2", (6, 13), FONT, 0.36, MUTED, 1, cv2.LINE_AA)
    cv2.line(img, (pad_l, height - 7), (pad_l + 18, height - 7), CUR, 2, cv2.LINE_AA)
    cv2.putText(img, "now", (pad_l + 24, height - 3), FONT, 0.35, MUTED, 1, cv2.LINE_AA)
    cv2.line(img, (pad_l + 62, height - 7), (pad_l + 80, height - 7), PRD, 1, cv2.LINE_AA)
    cv2.putText(img, f"predicted +{horizon:.1f}s", (pad_l + 86, height - 3),
                FONT, 0.35, MUTED, 1, cv2.LINE_AA)
    if press and press_thr:
        cv2.line(img, (pad_l + 214, height - 7), (pad_l + 232, height - 7), PRESS, 1, cv2.LINE_AA)
        cv2.putText(img, "crowd pressure", (pad_l + 238, height - 3),
                    FONT, 0.35, MUTED, 1, cv2.LINE_AA)

    # compact readout on the right - replaces the old banner
    c_d = cur[-1] if cur else 0.0
    p_d = prd[-1] if prd else 0.0
    if p_d >= crush_thr or c_d >= crush_thr:
        dot, word = (50, 50, 210), "CRUSH RISK"
    elif p_d >= warn_thr:
        dot, word = (60, 150, 245), "RISING"
    else:
        dot, word = (90, 180, 90), "CLEAR"
    rx = width - pad_r + 16
    cv2.circle(img, (rx, 26), 6, dot, -1, cv2.LINE_AA)
    cv2.putText(img, word, (rx + 16, 31), FONT, 0.5, dot, 1, cv2.LINE_AA)
    cv2.putText(img, f"now {c_d:.2f}", (rx, 56), FONT, 0.42, CUR, 1, cv2.LINE_AA)
    cv2.putText(img, f"+{horizon:.1f}s {p_d:.2f}", (rx + 92, 56), FONT, 0.42, PRD, 1, cv2.LINE_AA)
    arrow = "^" if p_d > c_d + 0.02 else ("v" if p_d < c_d - 0.02 else "=")
    cv2.putText(img, arrow, (rx + 208, 56), FONT, 0.46,
                (60, 150, 245) if arrow == "^" else MUTED, 1, cv2.LINE_AA)
    return img


def verdict_bar(width, height, cur_d, prd_d, horizon, crush_thr, warn_thr, lead):
    img = np.full((height, width, 3), (18, 17, 16), np.uint8)
    cname, ccol = band(cur_d)
    pname, pcol = band(prd_d)

    if prd_d >= crush_thr:
        msg, col = f"CRUSH RISK PREDICTED in {horizon:.1f}s", (40, 40, 200)
    elif cur_d >= crush_thr:
        msg, col = "CRUSH DENSITY NOW", (40, 40, 200)
    elif prd_d >= warn_thr:
        msg, col = f"DENSITY RISING - watch {horizon:.1f}s ahead", (60, 150, 255)
    else:
        msg, col = "NO CRUSH RISK", (80, 180, 80)

    cv2.rectangle(img, (0, 0), (14, height), col, -1)
    cv2.putText(img, msg, (28, 30), FONT, 0.72, col, 2, cv2.LINE_AA)

    right = (f"now {cur_d:.2f}/m2 [{cname}]   ->   "
             f"+{horizon:.1f}s {prd_d:.2f}/m2 [{pname}]   "
             f"trend {lead:+.2f}/m2/s")
    (tw, _), _ = cv2.getTextSize(right, FONT, 0.46, 1)
    cv2.putText(img, right, (width - tw - 16, 30), FONT, 0.46, TEXT, 1, cv2.LINE_AA)
    return img


def main(a):
    traj = json.loads(Path(a.trajectories).read_text())
    by_idx = {f["frame_idx"]: f for f in traj}

    fc = {}
    horizon = 0.0
    if a.forecasts and Path(a.forecasts).exists():
        raw = json.loads(Path(a.forecasts).read_text())
        if isinstance(raw, dict):
            print("[S7] WARNING: forecasts file is the one-shot stage6_forecasts.json.\n"
                  "     Predicted density needs the per-frame stream. Run:\n"
                  "       python src/stage6_rolling_forecast.py --method cv\n"
                  "     Continuing with current density only.")
        else:
            for f in raw:
                fc[f["frame_idx"]] = f["people"]
            hs = [p["path"][-1]["t"] for f in raw for p in f["people"] if p["path"]]
            horizon = float(np.median(hs)) if hs else 0.0
            print(f"[S7] Forecast horizon {horizon:.2f}s")

    press_by_idx, press_thr = {}, None
    if a.pressure and Path(a.pressure).exists():
        pj = json.loads(Path(a.pressure).read_text())
        press_thr = pj.get("relative_threshold_1_per_s2")
        for f in pj["frames"]:
            press_by_idx[f["frame_idx"]] = f["max_pressure_1_per_s2"]
        print(f"[S7] Crowd pressure loaded (alarm above {press_thr} 1/s2)")

    xs = [p["x"] for f in traj for p in f["people"]]
    ys = [p["y"] for f in traj for p in f["people"]]
    x0, x1 = min(xs) - 1.0, max(xs) + 1.0
    y0, y1 = min(ys) - 1.0, max(ys) + 1.0
    gw = int(np.ceil((x1 - x0) / GRID_RES_M))
    gh = int(np.ceil((y1 - y0) / GRID_RES_M))
    print(f"[S7] World {x1-x0:.1f} x {y1-y0:.1f} m  grid {gw}x{gh} @ {GRID_RES_M}m")

    H = np.array(json.loads(Path(a.homography).read_text())["homography_matrix"])
    Hinv = np.linalg.inv(H)
    # grid cell -> world -> pixel
    g2w = np.array([[GRID_RES_M, 0, x0], [0, GRID_RES_M, y0], [0, 0, 1]], float)
    g2p = Hinv @ g2w

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"[S7] Could not open video: {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ppm = fh / (y1 - y0)
    bw = int((x1 - x0) * ppm)
    gap, bar_h, tl_h = 10, 46, 74
    out_w = fw + gap + bw
    out_h = fh + (bar_h if a.banner else 0) + (tl_h if a.timeline else 0)

    Path(a.out_video).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(a.out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not vw.isOpened():
        raise SystemExit(f"[S7] Could not open writer for {a.out_video}")

    cur_series, prd_series, press_series, summaries = [], [], [], []
    peak_cur = peak_prd = 0.0
    alarm_frames = 0
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if a.max_frames and idx >= a.max_frames:
            break

        people = by_idx.get(idx, {}).get("people", [])
        pts = [(p["x"], p["y"]) for p in people]
        field = density_field(pts, x0, y0, gw, gh)
        cur_d = float(field.max()) if field.size else 0.0

        fpts, movers = [], []
        for p in fc.get(idx, []):
            if p["path"]:
                q = p["path"][-1]
                fpts.append((q["x"], q["y"]))
                movers.append(((p["x"], p["y"]), (q["x"], q["y"])))
            else:
                fpts.append((p["x"], p["y"]))
        ffield = density_field(fpts, x0, y0, gw, gh) if fpts else field
        prd_d = float(ffield.max()) if ffield.size else 0.0

        press_series.append(press_by_idx.get(idx, 0.0))
        cur_series.append(cur_d)
        prd_series.append(prd_d)
        peak_cur = max(peak_cur, cur_d)
        peak_prd = max(peak_prd, prd_d)
        if prd_d >= a.crush_threshold or cur_d >= a.crush_threshold:
            alarm_frames += 1

        # ---- left: camera with current density on the ground -------------
        heat = colorize(field)
        warped = cv2.warpPerspective(heat, g2p, (fw, fh), flags=cv2.INTER_LINEAR)
        cov = cv2.warpPerspective(np.full((gh, gw), 255, np.uint8), g2p, (fw, fh))
        m = ((cov > 0) & (warped.sum(2) > 0)).astype(np.float32)[..., None] * a.alpha
        left = (frame * (1 - m) + warped * m).astype(np.uint8)
        header(left, "Camera view", f"{len(people)} people  |  density on ground plane")

        # ---- right: BEV density ------------------------------------------
        right = cv2.resize(cv2.flip(colorize(field), 0), (bw, fh), interpolation=cv2.INTER_LINEAR)
        right = (right * 0.85 + np.full_like(right, 26) * 0.15).astype(np.uint8)

        def bp(x, y):
            return (int((x - x0) * ppm), int((y1 - y) * ppm))

        for xm in range(int(np.floor(x0)), int(np.ceil(x1)) + 1):
            if xm % 5 == 0:
                cv2.line(right, bp(xm, y1), bp(xm, y0), (70, 66, 62), 1)
                cv2.putText(right, f"{xm}m", (bp(xm, y0)[0] + 4, fh - 8), FONT, 0.36, MUTED, 1)
        for ym in range(int(np.floor(y0)), int(np.ceil(y1)) + 1):
            if ym % 5 == 0:
                cv2.line(right, bp(x0, ym), bp(x1, ym), (70, 66, 62), 1)
                cv2.putText(right, f"{ym}m", (6, bp(x0, ym)[1] - 5), FONT, 0.36, MUTED, 1)

        for (px, py), (qx, qy) in movers:
            cv2.arrowedLine(right, bp(px, py), bp(qx, qy), (255, 220, 140), 1,
                            cv2.LINE_AA, tipLength=0.35)
        for p in people:
            cv2.circle(right, bp(p["x"], p["y"]), 4, (20, 18, 17), -1, cv2.LINE_AA)
            cv2.circle(right, bp(p["x"], p["y"]), 2, (245, 245, 245), -1, cv2.LINE_AA)

        # predicted hot zone: dashed outline of where density is ABOUT to be.
        # This is the forecast made visible without a banner or a chart - the
        # heatmap shows now, the outline shows +horizon seconds.
        if fpts and horizon > 0:
            for thr, col, th in ((a.warn_threshold, (255, 210, 130), 1),
                                 (a.crush_threshold, (80, 80, 255), 2)):
                mask = (ffield >= thr).astype(np.uint8) * 255
                if mask.max() == 0:
                    continue
                mask = cv2.resize(cv2.flip(mask, 0), (bw, fh), interpolation=cv2.INTER_NEAREST)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cn in cnts:
                    if cv2.contourArea(cn) < 40:
                        continue
                    pts = cn.reshape(-1, 2)
                    for i in range(0, len(pts) - 1, 4):     # dashed
                        cv2.line(right, tuple(pts[i]), tuple(pts[min(i + 2, len(pts) - 1)]),
                                 col, th, cv2.LINE_AA)
                    if thr >= a.crush_threshold:
                        label(right, f"predicted crush +{horizon:.1f}s",
                              (pts[0][0] + 6, pts[0][1] - 6), col, 0.4, 1)

        # hottest cell marker
        if cur_d > 0.5:
            r, c = np.unravel_index(np.argmax(field), field.shape)
            hp = bp(x0 + c * GRID_RES_M, y0 + r * GRID_RES_M)
            cv2.circle(right, hp, 14, (255, 255, 255), 1, cv2.LINE_AA)
            label(right, f"{cur_d:.2f}/m2", (hp[0] + 18, hp[1]), TEXT, 0.4, 1)

        header(right, "Density convergence",
               f"peak now {cur_d:.2f}/m2   predicted {prd_d:.2f}/m2")

        ly = fh - 20 - len(LOS) * 16
        for hi, name, col in LOS:
            cv2.rectangle(right, (12, ly), (28, ly + 11), col, -1)
            cv2.putText(right, f"<{hi:g}  {name}" if hi < 1e8 else f">6  {name}",
                        (34, ly + 10), FONT, 0.35, TEXT, 1, cv2.LINE_AA)
            ly += 16
        if fpts and horizon > 0:
            for i in range(0, 16, 5):
                cv2.line(right, (12 + i, ly + 5), (12 + i + 2, ly + 5), (255, 210, 130), 1)
            cv2.putText(right, f"dashed = predicted +{horizon:.1f}s",
                        (34, ly + 10), FONT, 0.35, TEXT, 1, cv2.LINE_AA)

        parts = []
        if a.banner:
            lead = (prd_d - cur_d) / horizon if horizon > 0 else 0.0
            parts.append(verdict_bar(out_w, bar_h, cur_d, prd_d, horizon,
                                     a.crush_threshold, a.warn_threshold, lead))
        parts.append(np.hstack([left, np.full((fh, gap, 3), BG, np.uint8), right]))
        if a.timeline:
            parts.append(timeline(out_w, tl_h, cur_series, prd_series,
                                  idx, len(traj), a.crush_threshold,
                                  a.warn_threshold, horizon,
                                  press_series if press_by_idx else None, press_thr))
        out = np.vstack(parts)

        if idx == a.still_frame:
            cv2.imwrite(a.out_png, out)
            print(f"[S7] Wrote still -> {a.out_png}")
        vw.write(out)

        summaries.append({"frame_idx": idx, "num_people": len(people),
                          "max_density_per_m2": round(cur_d, 3),
                          "predicted_max_density_per_m2": round(prd_d, 3),
                          "crush_cells": int((field >= a.crush_threshold).sum()),
                          "predicted_crush_cells": int((ffield >= a.crush_threshold).sum()),
                          "crowd_pressure_1_per_s2": press_by_idx.get(idx, 0.0)})
        idx += 1
        if idx % 50 == 0:
            print(f"[S7]  {idx} frames...")

    cap.release()
    vw.release()
    Path(a.out_json).write_text(json.dumps({
        "forecast_horizon_sec": horizon,
        "crush_threshold_per_m2": a.crush_threshold,
        "peak_observed_density_per_m2": round(peak_cur, 3),
        "peak_predicted_density_per_m2": round(peak_prd, 3),
        "frames_in_alarm": alarm_frames,
        "verdict": ("CRUSH RISK DETECTED" if alarm_frames else "NO CRUSH RISK DETECTED"),
        "frames": summaries,
    }, indent=1))

    print(f"\n[S7] Wrote -> {a.out_video}")
    print(f"[S7] Peak observed density   {peak_cur:.2f} persons/m2  [{band(peak_cur)[0]}]")
    print(f"[S7] Peak predicted density  {peak_prd:.2f} persons/m2  [{band(peak_prd)[0]}]")
    print(f"[S7] Crush threshold {a.crush_threshold:.1f}/m2 -> "
          f"{alarm_frames}/{idx} frames in alarm")
    print(f"[S7] VERDICT: {'CRUSH RISK DETECTED' if alarm_frames else 'NO CRUSH RISK DETECTED'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", default="outputs/world_trajectories.json")
    ap.add_argument("--forecasts", default="outputs/stage6_rolling.json")
    ap.add_argument("--pressure", default="outputs/stage5_pressure.json",
                    help="Stage 5 crowd pressure; adds a third trace if present")
    ap.add_argument("--homography", default="config/homography.json")
    ap.add_argument("--video", default="data/cctv_clip.avi")
    ap.add_argument("--out-video", default="outputs/stage7_convergence.mp4")
    ap.add_argument("--out-png", default="outputs/stage7_convergence.png")
    ap.add_argument("--out-json", default="outputs/stage7_convergence.json")
    ap.add_argument("--crush-threshold", type=float, default=4.0,
                    help="persons/m2 that triggers the alarm (Fruin high-risk band)")
    ap.add_argument("--warn-threshold", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--banner", action="store_true",
                    help="add the verdict bar across the top (off by default)")
    ap.add_argument("--no-timeline", dest="timeline", action="store_false",
                    help="drop the density-over-time strip chart")
    ap.set_defaults(timeline=True)
    ap.add_argument("--still-frame", type=int, default=110)
    ap.add_argument("--max-frames", type=int, default=0)
    main(ap.parse_args())