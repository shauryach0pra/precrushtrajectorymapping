"""
Stage 2: ByteTrack Trajectory Tracking
-----------------------------------------
Runs YOLOv11 + ByteTrack together (ultralytics' model.track() handles both)
to assign a PERSISTENT ID to each person across frames, not just per-frame boxes.

Outputs:
  1. Annotated video with track IDs drawn -> outputs/stage2_tracked.mp4
  2. Per-frame track JSON -> outputs/stage2_tracks.json

Output schema (pixel coordinates - Stage 3 will convert to world/meters):
[
  {
    "frame_idx": int,
    "timestamp_sec": float,
    "people": [
        {"track_id": int, "bbox": [x1,y1,x2,y2], "conf": float,
         "foot_x": float, "foot_y": float}
    ]
  },
  ...
]

Note: we compute "foot point" (bottom-center of the bbox) as the person's
ground-contact pixel - this is what Stage 3's homography should transform,
NOT the box center, since the center is roughly torso height, not ground level.

Usage:
    python src/stage2_bytetrack.py --video data/cctv_clip.avi --model models/yolo11m.pt
"""

import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

PERSON_CLASS_ID = 0


def run_stage2(video_path: str, model_path: str, conf_thresh: float, imgsz: int, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage2] Loading model: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Stage2] Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_dir / "stage2_tracked.mp4"), fourcc, fps, (width, height))

    all_tracks = []
    frame_idx = 0
    t_start = time.time()
    seen_ids = set()

    # persist=True keeps the tracker's internal state alive across .track() calls
    # so IDs stay consistent frame-to-frame instead of resetting each call.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=conf_thresh,
            imgsz=imgsz,
            tracker="botsort.yaml",
            persist=True,
            verbose=False,
        )[0]

        frame_people = []
        if results.boxes is not None and results.boxes.id is not None:
            for box, track_id in zip(results.boxes, results.boxes.id):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                tid = int(track_id)
                seen_ids.add(tid)

                foot_x = (x1 + x2) / 2.0
                foot_y = y2  # bottom of the box = ground contact point

                frame_people.append({
                    "track_id": tid,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "conf": round(conf, 3),
                    "foot_x": round(foot_x, 1),
                    "foot_y": round(foot_y, 1),
                })

                # Draw box + persistent ID (color derived from ID so each person is distinct)
                color = ((tid * 47) % 255, (tid * 91) % 255, (tid * 137) % 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"ID {tid}", (int(x1), max(int(y1) - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(frame, (int(foot_x), int(foot_y)), 3, color, -1)

        cv2.putText(frame, f"Frame {frame_idx} | Active: {len(frame_people)} | Total seen: {len(seen_ids)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        writer.write(frame)

        all_tracks.append({
            "frame_idx": frame_idx,
            "timestamp_sec": round(frame_idx / fps, 3),
            "people": frame_people,
        })

        if frame_idx % 30 == 0:
            print(f"[Stage2] frame {frame_idx}/{total_frames} | active tracks: {len(frame_people)} | total unique IDs so far: {len(seen_ids)}")

        frame_idx += 1

    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    print(f"[Stage2] Done. {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps effective)")
    print(f"[Stage2] Total unique people tracked: {len(seen_ids)}")

    json_path = out_dir / "stage2_tracks.json"
    with open(json_path, "w") as f:
        json.dump(all_tracks, f, indent=2)

    print(f"[Stage2] Annotated video -> {out_dir / 'stage2_tracked.mp4'}")
    print(f"[Stage2] Tracks JSON -> {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="../data/cctv_clip.avi")
    parser.add_argument("--model", default="../models/yolo11m.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--out", default="../outputs")
    args = parser.parse_args()

    run_stage2(args.video, args.model, args.conf, args.imgsz, args.out)