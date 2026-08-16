# Crowd Analysis Demo

A multi-stage pipeline for crowd detection, tracking, trajectory analysis, and density estimation using computer vision and deep learning.

## Project Structure

```
crowd-demo/
├── src/                    # Python scripts for each processing stage
│   ├── calibrate_homography.py    # Stage 3a: Camera calibration
│   ├── stage2_bytetrack.py        # Stage 2: Object tracking
│   ├── stage3_apply_homography.py # Stage 3b: World coordinate transformation
│   ├── stage4_social_force.py     # Stage 4: Social force modeling
│   ├── stage6_forecast.py         # Stage 6: LSTM trajectory forecasting
│   ├── stage7_density.py          # Stage 7: Density heatmap generation
│   └── model.py                    # LSTM model definition
├── models/                 # Model files
│   ├── yolo11m.pt               # YOLOv11 medium model for detection
│   └── pretrained_traj_lstm.pt  # Pre-trained LSTM (missing - needs to be added)
├── data/                   # Input data
│   └── cctv_clip.avi            # Input video file
├── config/                 # Configuration files
│   └── homography.json          # Camera calibration matrix
├── outputs/                # Generated outputs
│   ├── stage1_annotated.mp4     # Detection results (if stage 1 exists)
│   ├── stage1_detections.json   # Detection data
│   ├── stage2_tracked.mp4       # Tracking results with IDs
│   ├── stage2_tracks.json       # Tracking data
│   ├── world_trajectories.json  # World coordinate trajectories
│   ├── stage4_forces.json        # Social force data
│   ├── stage6_forecasts.json     # LSTM forecasts
│   ├── stage7_density.mp4       # Final density heatmap video
│   └── stage7_density.json      # Density analysis data
└── README.md               # This file
```

## Pipeline Stages

### Stage 2: ByteTrack Trajectory Tracking
Runs YOLOv11 + ByteTrack to assign persistent IDs to people across frames.

```bash
python src/stage2_bytetrack.py --video data/cctv_clip.avi --model models/yolo11m.pt
```

**Outputs:**
- `outputs/stage2_tracked.mp4` - Video with track IDs
- `outputs/stage2_tracks.json` - Per-frame tracking data

### Stage 3a: Homography Calibration
Calibrate camera to transform pixel coordinates to real-world coordinates.

```bash
python src/calibrate_homography.py --video data/cctv_clip.avi --frame 100
```

**Outputs:**
- `config/homography.json` - Camera calibration matrix

### Stage 3b: Apply Homography
Transform pixel coordinates to world coordinates using calibration.

```bash
python src/stage3_apply_homography.py --tracks outputs/stage2_tracks.json --homography config/homography.json
```

**Outputs:**
- `outputs/world_trajectories.json` - World coordinate trajectories

### Stage 4: Social Force Model
Compute social forces between pedestrians for crowd dynamics analysis.

```bash
python src/stage4_social_force.py --input outputs/world_trajectories.json
```

**Outputs:**
- `outputs/stage4_forces.json` - Social force data per frame

### Stage 6: LSTM Trajectory Forecasting
Predict future trajectories using LSTM (requires pretrained model).

```bash
python src/stage6_forecast.py --stage4 outputs/stage4_forces.json --checkpoint models/pretrained_traj_lstm.pt
```

**Outputs:**
- `outputs/stage6_forecasts.json` - Trajectory predictions

### Stage 7: Density Heatmap
Generate density heatmap and overlay on original video.

```bash
python src/stage7_density.py --trajectories outputs/world_trajectories.json --forecasts outputs/stage6_forecasts.json --homography config/homography.json --video data/cctv_clip.avi
```

**Outputs:**
- `outputs/stage7_density.mp4` - Final density heatmap video
- `outputs/stage7_density.json` - Density analysis data

## Dependencies

- Python 3.8+
- OpenCV
- NumPy
- PyTorch
- Ultralytics YOLO
- SciPy

## Missing Files

The following file is missing and needs to be obtained:
- `models/pretrained_traj_lstm.pt` - Pre-trained LSTM model for trajectory forecasting

## Notes

- The pipeline assumes video is at 25 FPS by default
- Homography calibration should be done once per camera setup
- Stage 6 (forecasting) is optional and requires the missing pretrained model
- All scripts can be run from the project root directory using the relative paths shown above
