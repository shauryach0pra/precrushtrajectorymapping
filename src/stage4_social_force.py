"""
Stage 4: Social Force Model (+ KD-Tree neighbor search)
------------------------------------------------------------
For every person, every frame, computes a net social force vector from:

  1. DRIVE FORCE   - pulls the person toward their own recent heading/speed
                      (keeps simulated behavior grounded in real observed motion)
  2. REPULSION     - pushes them away from nearby people, stronger the closer
                      they are (classic pedestrian personal-space physics).
                      Neighbors are found efficiently with scipy's cKDTree
                      instead of an O(n^2) distance check every frame.
  3. ATTRACTOR (optional, "demo convergence mode")
                   - an artificial goal point (e.g. simulating a gate/exit)
                      that pulls people toward it starting at a chosen frame.
                      This is what lets you DEMO a forming bottleneck on
                      footage that has no real one - clearly logged as a
                      synthetic scenario layered on real trajectories, not
                      fabricated sensor data.

Output -> outputs/stage4_forces.json, one entry per frame:
[
  {
    "frame_idx": int,
    "timestamp_sec": float,
    "people": [
        {
          "track_id": int,
          "x": float, "y": float,             # current position (m)
          "vx": float, "vy": float,            # observed velocity (m/s)
          "force_x": float, "force_y": float,  # net social force
          "next_x": float, "next_y": float     # simple one-step Euler forecast
        }
    ]
  }
]

`next_x/next_y` is a cheap physics-based forecast (not the LSTM) - useful as
a sanity check now, and as an extra input feature for Stage 6 later.

Usage (normal, no synthetic scenario):
    python src/stage4_social_force.py --input ../outputs/world_trajectories.json

Usage (demo convergence mode - people pulled toward a gate point from frame 100):
    python src/stage4_social_force.py --input ../outputs/world_trajectories.json \\
        --attractor 6.0,4.0 --attractor_start_frame 100 --attractor_strength 1.5
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# --- Tunable social force parameters (standard pedestrian-dynamics ranges) ---
TAU = 0.5                 # relaxation time (s) - how fast people accelerate toward desired velocity
REPULSION_A = 2.0         # repulsion strength
REPULSION_B = 0.5         # repulsion decay distance (m)
PERSON_RADIUS = 0.3       # approx. half-body-width (m)
NEIGHBOR_RADIUS = 3.0     # only consider neighbors within this range (m) - keeps KD-Tree queries cheap
DT = 1.0 / 25.0           # frame timestep, matches video fps - overridden by --fps if passed


def compute_velocities(frames):
    """Finite-difference velocity from consecutive real positions per track_id."""
    last_pos = {}
    for frame in frames:
        for p in frame["people"]:
            tid = p["track_id"]
            if tid in last_pos:
                px, py = last_pos[tid]
                p["vx"] = (p["x"] - px) / DT
                p["vy"] = (p["y"] - py) / DT
            else:
                p["vx"] = 0.0
                p["vy"] = 0.0
            last_pos[tid] = (p["x"], p["y"])
    return frames


def compute_frame_forces(people, attractor=None, attractor_strength=0.0):
    """Vectorized-ish per-frame force computation using a KD-Tree for neighbor lookups."""
    n = len(people)
    if n == 0:
        return

    positions = np.array([[p["x"], p["y"]] for p in people])
    velocities = np.array([[p["vx"], p["vy"]] for p in people])

    tree = cKDTree(positions)

    for i, p in enumerate(people):
        pos_i = positions[i]
        vel_i = velocities[i]

        # --- 1. Drive force: pull toward own current heading (keeps them moving naturally) ---
        speed = np.linalg.norm(vel_i)
        desired_dir = vel_i / speed if speed > 1e-3 else np.array([0.0, 0.0])
        desired_vel = desired_dir * speed  # for now, "desired" = "keep doing what you're doing"
        drive_force = (desired_vel - vel_i) / TAU

        # --- 2. Repulsion force: push away from nearby people ---
        neighbor_idxs = tree.query_ball_point(pos_i, r=NEIGHBOR_RADIUS)
        repulsion_force = np.array([0.0, 0.0])
        for j in neighbor_idxs:
            if j == i:
                continue
            diff = pos_i - positions[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-3:
                dist = 1e-3
            direction = diff / dist
            magnitude = REPULSION_A * np.exp((2 * PERSON_RADIUS - dist) / REPULSION_B)
            repulsion_force += direction * magnitude

        # --- 3. Optional attractor force (demo convergence mode) ---
        attractor_force = np.array([0.0, 0.0])
        if attractor is not None:
            to_goal = np.array(attractor) - pos_i
            dist_to_goal = np.linalg.norm(to_goal)
            if dist_to_goal > 1e-3:
                attractor_force = (to_goal / dist_to_goal) * attractor_strength

        net_force = drive_force + repulsion_force + attractor_force

        # Simple one-step Euler forecast (position after one more frame under this force)
        next_vel = vel_i + net_force * DT
        next_pos = pos_i + next_vel * DT

        p["force_x"] = round(float(net_force[0]), 4)
        p["force_y"] = round(float(net_force[1]), 4)
        p["next_x"] = round(float(next_pos[0]), 3)
        p["next_y"] = round(float(next_pos[1]), 3)


def main(input_path, out_dir, attractor_str, attractor_start_frame, attractor_strength, fps):
    global DT
    if fps:
        DT = 1.0 / fps

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r") as f:
        frames = json.load(f)
    print(f"[Stage4] Loaded {len(frames)} frames from {input_path}")

    attractor = None
    if attractor_str:
        ax, ay = [float(v) for v in attractor_str.split(",")]
        attractor = (ax, ay)
        print(f"[Stage4] DEMO CONVERGENCE MODE: attractor at ({ax}, {ay}), "
              f"active from frame {attractor_start_frame}, strength {attractor_strength}")
        print("[Stage4] NOTE: this is a synthetic scenario layered on real trajectories, "
              "for demoing bottleneck detection - not real sensor data.")

    frames = compute_velocities(frames)

    for frame in frames:
        active_attractor = attractor if (attractor and frame["frame_idx"] >= attractor_start_frame) else None
        compute_frame_forces(frame["people"], active_attractor, attractor_strength)

    out_path = out_dir / "stage4_forces.json"
    with open(out_path, "w") as f:
        json.dump(frames, f, indent=2)

    print(f"[Stage4] Wrote -> {out_path}")
    print("[Stage4] This file feeds Stage 6 (LSTM forecasting) and Stage 7 (density heatmap).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../outputs/world_trajectories.json")
    parser.add_argument("--out", default="../outputs")
    parser.add_argument("--fps", type=float, default=None, help="Override frame rate for velocity calc (defaults to 25)")
    parser.add_argument("--attractor", default=None, help="Optional 'x,y' world coords for a demo convergence point (e.g. simulated gate)")
    parser.add_argument("--attractor_start_frame", type=int, default=0, help="Frame index the attractor activates at")
    parser.add_argument("--attractor_strength", type=float, default=1.0, help="Pull strength toward the attractor")
    args = parser.parse_args()

    main(args.input, args.out, args.attractor, args.attractor_start_frame, args.attractor_strength, args.fps)