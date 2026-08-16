"""
Stage 3b: Apply Homography - Pixels to World Meters
--------------------------------------------------------
Reads stage2_tracks.json (pixel-space tracks) and homography.json (from
calibrate_homography.py), transforms every person's foot-point into
real-world ground coordinates (meters), and writes world_trajectories.json -
the same schema Stage 4 (Social Force Model) expects, whether it came from
this live pipeline or the ETH/UCY adapter.

Output schema -> outputs/world_trajectories.json:
[
  {
    "frame_idx": int,
    "timestamp_sec": float,
    "people": [
        {"track_id": int, "x": float, "y": float}
    ]
  },
  ...
]

Usage:
    python src/stage3_apply_homography.py --tracks ../outputs/stage2_tracks.json --homography ../config/homography.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def apply_homography_point(H: np.ndarray, px: float, py: float):
    """Transforms a single pixel point through the homography matrix into world coords."""
    vec = np.array([px, py, 1.0])
    world = H @ vec
    world /= world[2]  # normalize homogeneous coordinate
    return float(world[0]), float(world[1])


def main(tracks_path: str, homography_path: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(homography_path, "r") as f:
        calib = json.load(f)
    H = np.array(calib["homography_matrix"])
    print(f"[Stage3] Loaded homography (calibrated on frame {calib['calibration_frame']} of {calib['video']})")

    with open(tracks_path, "r") as f:
        tracks = json.load(f)
    print(f"[Stage3] Loaded {len(tracks)} frames of pixel-space tracks")

    output = []
    all_x, all_y = [], []

    for frame in tracks:
        people_world = []
        for person in frame["people"]:
            wx, wy = apply_homography_point(H, person["foot_x"], person["foot_y"])
            people_world.append({
                "track_id": person["track_id"],
                "x": round(wx, 3),
                "y": round(wy, 3),
            })
            all_x.append(wx)
            all_y.append(wy)

        output.append({
            "frame_idx": frame["frame_idx"],
            "timestamp_sec": frame["timestamp_sec"],
            "people": people_world,
        })

    json_path = out_dir / "world_trajectories.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    if all_x:
        print(f"[Stage3] World coordinate range: x=[{min(all_x):.2f}, {max(all_x):.2f}]m, "
              f"y=[{min(all_y):.2f}, {max(all_y):.2f}]m")
        print("[Stage3] Sanity check: does this span roughly match the real scene size?")
        print("[Stage3] If numbers look wildly off (e.g. thousands of meters, or all near 0),")
        print("[Stage3] your calibration points/world coordinates likely need rechecking.")

    print(f"[Stage3] Wrote -> {json_path}")
    print("[Stage3] This file is the drop-in input for Stage 4 (Social Force Model).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", default="../outputs/stage2_tracks.json")
    parser.add_argument("--homography", default="../config/homography.json")
    parser.add_argument("--out", default="../outputs")
    args = parser.parse_args()

    main(args.tracks, args.homography, args.out)
