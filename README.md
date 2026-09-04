# Pre-Crush Trajectory Mapping
Demo Video : https://www.youtube.com/watch?v=k9_PfLIWqPE
Demo Dashboard : https://shauryachopra.dev/sentinelgrid

A multi-stage pipeline for crowd detection, tracking, trajectory analysis, and density estimation using computer vision and deep learning. Designed for early detection of crowd crush risks through predictive density analysis.

## Project Structure

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
└── README.md               # This file
```

## Pipeline Stages

### Stage 2: BoT-SORT Trajectory Tracking
Runs YOLOv11 + BoT-SORT to assign persistent IDs to people across frames.

```bash
python src/stage2_bytetrack.py --video data/cctv_clip.avi --model models/yolo11m.pt --tracker config/botsort_tuned.yaml
```

**Outputs:**
- `outputs/stage2_track.mp4` - Video with track IDs
- `outputs/stage2_tracks.json` - Per-frame tracking data

**Note:** The tuned BoT-SORT configuration (`config/botsort_tuned.yaml`) fixes ID-churn issues with default settings by holding lost tracks longer and using stricter detection thresholds.

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
Compute social forces between pedestrians for crowd dynamics analysis using KD-tree neighbor search.

```bash
python src/stage4_social_force.py --input outputs/world_trajectories.json
```

**Optional demo mode:** Simulate bottleneck convergence
```bash
python src/stage4_social_force.py --input outputs/world_trajectories.json --attractor 6.0,4.0 --attractor_start_frame 100 --attractor_strength 1.5
```

**Outputs:**
- `outputs/stage4_forces.json` - Social force data per frame

### Stage 5: Crowd Pressure Analysis
(Computed implicitly in Stage 4) Analyzes crowd pressure dynamics for early crush detection.

**Outputs:**
- `outputs/stage5_pressure.json` - Crowd pressure metrics

### Stage 6a: LSTM Trajectory Forecasting (One-shot)
Predict future trajectories using LSTM (requires pretrained model).

```bash
python src/stage6_forecast.py --stage4 outputs/stage4_forces.json --checkpoint models/pretrained_traj_lstm.pt
```

**Outputs:**
- `outputs/stage6_forecasts.json` - One-shot trajectory predictions per track

### Stage 6b: Rolling Per-Frame Forecasts
Generate forecasts for every track on every frame for animation and real-time analysis.

**Kinematic baseline (no training required):**
```bash
python src/stage6_rolling_forecast.py --method cv
```

**LSTM-based (requires trained model):**
```bash
python src/stage6_rolling_forecast.py --method lstm --checkpoint models/pretrained_traj_lstm.pt
```

**Outputs:**
- `outputs/stage6_rolling.json` - Per-frame forecast data suitable for animation

### Train LSTM Model
Train the trajectory forecasting model on your own data.

```bash
python src/train_traj_lstm.py --data outputs/world_trajectories.json --out models/pretrained_traj_lstm.pt
```

**Training options:**
- `--epochs 400` - Number of training epochs
- `--hidden 128` - LSTM hidden layer size
- `--obs-len 5` - Observation length
- `--pred-len 5` - Prediction length

**Note:** The script uses a 20% track holdout for validation and reports honest ADE/FDE metrics to prevent circular validation.

### Stage 7a: Density Heatmap
Generate density heatmap and overlay on original video.

```bash
python src/stage7_density.py --trajectories outputs/world_trajectories.json --forecasts outputs/stage6_forecasts.json --homography config/homography.json --video data/cctv_clip.avi
```

**Outputs:**
- `outputs/stage7_density.mp4` - Final density heatmap video
- `outputs/stage7_density.json` - Density analysis data

### Stage 7b: Density Convergence Engine
Advanced density analysis showing current vs predicted density with convergence detection.

```bash
python src/stage7_convergence_video.py --trajectories outputs/world_trajectories.json --forecasts outputs/stage6_rolling.json --homography config/homography.json --video data/cctv_clip.avi
```

**Outputs:**
- `outputs/stage7_convergence.mp4` - Convergence demo video with 3-panel layout
- `outputs/stage7_convergence.json` - Convergence analysis data

## Visualization & Demo Scripts

### Stage 1+2 Demo: Detection vs Tracking
Side-by-side comparison showing detection-only vs tracking with persistent IDs.

```bash
python src/stage12_detect_track_video.py --mode both
```

**Options:**
- `--mode detect` - Detection only
- `--mode track` - Tracking only
- `--mode both` - Side-by-side comparison (default)

### Stage 3 Demo: Bird's-Eye View Homography
Demonstrates pixel-to-world transformation with side-by-side camera and BEV views.

```bash
python src/stage3_bev_video.py --video data/cctv_clip.avi --homography config/homography.json --tracks outputs/stage2_tracks.json
```

### Stage 4+5 Demo: Social Force Visualization
Visualizes social forces and KD-tree neighbor search in bird's-eye view.

```bash
python src/stage45_force_video.py
```

### Stage 6 Demo: Trajectory Forecasting
Shows current positions vs predicted trajectories with optional hindcast verification.

```bash
python src/stage6_forecast_video.py --video data/cctv_clip.avi --forecasts outputs/stage6_rolling.json --homography config/homography.json
```

**Options:**
- `--verify` - Enable hindcast accuracy check
- `--video` - Source video (optional for BEV-only)

### Stage 7 Demo: Density Convergence
Three-panel demo showing camera view, BEV density, and convergence timeline.

```bash
python src/stage7_convergence_video.py --trajectories outputs/world_trajectories.json --forecasts outputs/stage6_rolling.json --homography config/homography.json --video data/cctv_clip.avi
```

## Dependencies

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- PyTorch (`torch`)
- Ultralytics YOLO (`ultralytics`)
- SciPy

Install with:
```bash
pip install -r requirements.txt
```

## Model Training

The LSTM model for trajectory forecasting can be trained on your own data:

```bash
python src/train_traj_lstm.py --data outputs/world_trajectories.json --out models/pretrained_traj_lstm.pt
```

**Training features:**
- 20% track holdout for honest validation
- Reports ADE (Average Displacement Error) and FDE (Final Displacement Error) in meters
- Supports custom hyperparameters for observation/prediction length
- Prevents circular validation on test data

For production use, consider training on standard pedestrian datasets (ETH/UCY) for better generalization.

## Key Features

### ID Stability
The tuned BoT-SORT configuration (`config/botsort_tuned.yaml`) significantly reduces ID churn compared to default settings by:
- Holding lost tracks for 75 frames (3 seconds at 25fps)
- Using higher detection thresholds to prevent track spawning from noise
- Optimized association parameters for stable tracking

### Predictive Analysis
- **Rolling forecasts**: Per-frame predictions enable real-time animation
- **Kinematic baseline**: Constant-velocity + social-force blend works without training
- **Convergence detection**: Identifies areas where density will increase, not just where it already is

### Density Analysis
- **Fruin/Still LOS bands**: Industry-standard crowd density classifications
- **Multi-panel visualization**: Camera view + BEV + timeline for comprehensive analysis
- **Verdict system**: Always-visible status (CLEAR/RISING/CRUSH RISK) for safety monitoring

## Notes

- The pipeline assumes video is at 25 FPS by default (configurable per script)
- Homography calibration should be done once per camera setup
- Stage 6 offers two methods: LSTM (requires training) and kinematic baseline (no training)
- The kinematic baseline is honest and physically reasonable - label it as such, not as "LSTM"
- All scripts can be run from the project root directory using the relative paths shown above
- Forecast accuracy metrics (ADE/FDE) should be quoted when presenting results to ensure honesty
