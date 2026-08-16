"""
Stage 3a: Homography Calibration Tool
-----------------------------------------
Run this ONCE per camera angle. It grabs a frame from your video, lets you
click 4 points that you know the real-world (ground-plane) coordinates of,
computes the homography matrix, and saves it to disk for Stage 3b to use.

Picking good points:
  - Choose 4 points that are actually on the GROUND (not on walls/people)
  - Prefer points that form a large quadrilateral spanning most of the frame
    (more accurate than 4 points clustered in a small area)
  - Good choices: corners of paving slabs/tiles, gate posts, floor markings,
    known-width paths - anything you can measure or confidently estimate
  - Click them in the SAME order you'll enter world coordinates for

Controls:
  - Left click: place a point (up to 4)
  - 'r': reset/clear points
  - 'q' or ESC: quit without saving (before 4 points placed)
  - After the 4th click, you'll be prompted in the terminal for real-world
    (x, y) meters for each point, in click order.

Usage:
    python src/calibrate_homography.py --video data/cctv_clip.avi --frame 100
"""

import argparse
import json

import cv2
import numpy as np

points_px = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points_px) < 4:
        points_px.append((x, y))
        print(f"[Calib] Point {len(points_px)} clicked at pixel ({x}, {y})")


def main(video_path: str, frame_idx: int, out_path: str):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    display = frame.copy()
    window = "Calibration - click 4 ground points, 'r' to reset, 'q' to quit"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse_callback)

    print("[Calib] Click 4 ground-plane points on the image, in an order you'll remember")
    print("[Calib] (e.g. corners of a paved square, gate posts, tile grid)")

    while True:
        display = frame.copy()
        for i, (px, py) in enumerate(points_px):
            cv2.circle(display, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(display, str(i + 1), (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(window, display)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            points_px.clear()
            print("[Calib] Points reset")
        elif key == ord('q') or key == 27:
            print("[Calib] Quit without saving")
            cv2.destroyAllWindows()
            return
        elif len(points_px) == 4:
            break

    cv2.destroyAllWindows()

    print("\n[Calib] Now enter the REAL-WORLD ground coordinates (in meters) for each point,")
    print("[Calib] in the same order you clicked them. Use any consistent origin/axes")
    print("[Calib] (e.g. one corner of the scene = (0,0), x = rightward, y = away from camera).\n")

    points_world = []
    for i in range(4):
        while True:
            try:
                raw = input(f"  World (x, y) for point {i+1} (e.g. '0, 0'): ")
                wx, wy = [float(v.strip()) for v in raw.split(",")]
                points_world.append((wx, wy))
                break
            except Exception:
                print("  Invalid input, format as: x, y  (e.g. 3.5, 0)")

    src = np.array(points_px, dtype=np.float32)
    dst = np.array(points_world, dtype=np.float32)

    H, status = cv2.findHomography(src, dst)
    if H is None:
        raise RuntimeError("Homography computation failed - points may be collinear or degenerate. Re-run and pick 4 points that form a proper quadrilateral.")

    print("\n[Calib] Homography matrix computed:")
    print(H)

    with open(out_path, "w") as f:
        json.dump({
            "homography_matrix": H.tolist(),
            "pixel_points": points_px,
            "world_points": points_world,
            "calibration_frame": frame_idx,
            "video": video_path,
        }, f, indent=2)

    print(f"\n[Calib] Saved -> {out_path}")
    print("[Calib] Use this file with stage3_apply_homography.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=0, help="Which frame index to calibrate on")
    parser.add_argument("--out", default="../config/homography.json")
    args = parser.parse_args()

    main(args.video, args.frame, args.out)
