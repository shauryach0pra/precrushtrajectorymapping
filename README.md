<div align="center">

### Pre-Crush Trajectory Mapping

</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=fff)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-111F68?logo=yolo&logoColor=fff)](#)

</div>

---

### Overview

This project implements a computer vision pipeline for early detection of crowd crush risks through predictive density analysis. It combines YOLOv11 person detection, BoT-SORT multi-object tracking, social force modeling, and LSTM trajectory forecasting to analyze crowd dynamics in real-time CCTV footage. The system transforms camera coordinates to world coordinates, predicts future crowd movements, and generates density heatmaps with convergence detection for safety monitoring.

---

### Demo

Demo Video: https://www.youtube.com/watch?v=k9_PfLIWqPE

Live Dashboard: https://shauryachopra.dev/sentinelgrid

---

### Project Structure

```
precrushtrajectorymapping/
├── src/                    # Python scripts for each processing stage
│   ├── calibrate_homography.py        # Stage 3a: Camera calibration
│   ├── stage2_bytetrack.py            # Stage 2: Object tracking (BoT-SORT)
│   ├── stage3_apply_homography.py     # Stage 3b: World coordinate transformation
│   ├── stage4_social_force.py         # Stage 4: Social force modeling
│   ├── stage6_forecast.py             # Stage 6a: LSTM trajectory forecasting (one-shot)
│   ├── stage6_rolling_forecast.py     # Stage 6b: Rolling per-frame forecasts (animatable)
│   ├── stage7_density.py              # Stage 7a: Density heatmap generation
│   ├── model.py                       # LSTM model definition
│   ├── train_traj_lstm.py             # Training script for LSTM model
│   └── Visualization/Demo scripts:
│       ├── stage12_detect_track_video.py    # Detection vs tracking demo
│       ├── stage3_bev_video.py               # Bird's-eye view homography demo
│       ├── stage45_force_video.py            # Social force visualization
│       ├── stage6_forecast_video.py         # Trajectory forecasting demo
│       └── stage7_convergence_video.py      # Density convergence engine demo
├── models/                 # Model files
│   ├── yolo11m.pt                   # YOLOv11 medium model for detection
│   └── pretrained_traj_lstm.pt      # Pre-trained LSTM model (train with train_traj_lstm.py)
├── data/                   # Input data
│   └── cctv_clip.avi                # Input video file
├── config/                 # Configuration files
│   ├── homography.json              # Camera calibration matrix
│   └── botsort_tuned.yaml          # Tuned BoT-SORT tracker configuration
├── outputs/                # Generated outputs
│   ├── stage1_detect.mp4            # Detection results
│   ├── stage1_detections.json      # Detection data
│   ├── stage2_track.mp4             # Tracking results with IDs
│   ├── stage2_tracks.json           # Tracking data
│   ├── world_trajectories.json      # World coordinate trajectories
│   ├── stage4_forces.json           # Social force data
│   ├── stage5_pressure.json         # Crowd pressure analysis
│   ├── stage6_forecasts.json        # One-shot LSTM forecasts
│   ├── stage6_rolling.json          # Rolling per-frame forecasts
│   ├── stage7_density.mp4           # Density heatmap video
│   ├── stage7_density.json          # Density analysis data
│   ├── stage7_convergence.mp4       # Convergence demo video
│   └── stage7_convergence.json      # Convergence analysis data
└── dashboard/               # Web-based demo dashboard
    └── index.html                    # Operations console interface
```

---

### Tech Stack

Core Technologies:
- Python 3.8+ - Primary programming language
- PyTorch - Deep learning framework for LSTM trajectory forecasting
- OpenCV - Computer vision library for video processing and transformations
- NumPy - Numerical computing and array operations
- SciPy - Scientific computing for social force calculations

Key Models:
- YOLOv11 - Person detection (Ultralytics)
- Custom LSTM - Trajectory prediction and forecasting
- BoT-SORT - Multi-object tracking with ID stability

---

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python
- numpy
- torch
- ultralytics
- scipy

---

### Configuration

Before running the pipeline, you need to configure:

Camera Calibration - Run homography calibration once per camera setup:
```bash
python src/calibrate_homography.py --video data/cctv_clip.avi --frame 100
```

Tracker Configuration - The project includes a tuned BoT-SORT configuration in `config/botsort_tuned.yaml` that reduces ID churn compared to default settings.

Model Files - Download or train the required models:
- `models/yolo11m.pt` - YOLOv11 medium model
- `models/pretrained_traj_lstm.pt` - Pre-trained LSTM model (or train your own)

---

### Usage

Complete Pipeline

Run the full pipeline from detection to density analysis:

```bash
# Stage 2: Object tracking
python src/stage2_bytetrack.py --video data/cctv_clip.avi --model models/yolo11m.pt --tracker config/botsort_tuned.yaml

# Stage 3b: Apply homography transformation
python src/stage3_apply_homography.py --tracks outputs/stage2_tracks.json --homography config/homography.json

# Stage 4: Social force modeling
python src/stage4_social_force.py --input outputs/world_trajectories.json

# Stage 6a: LSTM trajectory forecasting
python src/stage6_forecast.py --stage4 outputs/stage4_forces.json --checkpoint models/pretrained_traj_lstm.pt

# Stage 7a: Density heatmap generation
python src/stage7_density.py --trajectories outputs/world_trajectories.json --forecasts outputs/stage6_forecasts.json --homography config/homography.json --video data/cctv_clip.avi
```

Individual Stage Examples

Stage 2: BoT-SORT Trajectory Tracking
```bash
python src/stage2_bytetrack.py --video data/cctv_clip.avi --model models/yolo11m.pt --tracker config/botsort_tuned.yaml
```

Stage 6b: Rolling Per-Frame Forecasts
```bash
# Kinematic baseline (no training required)
python src/stage6_rolling_forecast.py --method cv

# LSTM-based (requires trained model)
python src/stage6_rolling_forecast.py --method lstm --checkpoint models/pretrained_traj_lstm.pt
```

Stage 7b: Density Convergence Engine
```bash
python src/stage7_convergence_video.py --trajectories outputs/world_trajectories.json --forecasts outputs/stage6_rolling.json --homography config/homography.json --video data/cctv_clip.avi
```

Model Training

Train the LSTM trajectory forecasting model on your own data:

```bash
python src/train_traj_lstm.py --data outputs/world_trajectories.json --out models/pretrained_traj_lstm.pt --epochs 400 --hidden 128 --obs-len 5 --pred-len 5
```

Visualization Scripts

Detection vs Tracking Demo
```bash
python src/stage12_detect_track_video.py --mode both
```

Bird's-Eye View Homography Demo
```bash
python src/stage3_bev_video.py --video data/cctv_clip.avi --homography config/homography.json --tracks outputs/stage2_tracks.json
```

Social Force Visualization
```bash
python src/stage45_force_video.py
```

Trajectory Forecasting Demo
```bash
python src/stage6_forecast_video.py --video data/cctv_clip.avi --forecasts outputs/stage6_rolling.json --homography config/homography.json --verify
```

---

<div align="center">

Built by [Shaurya Chopra](https://shauryachopra.dev/)

</div>
