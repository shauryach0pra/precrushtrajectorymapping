# Model Files

This directory should contain:

- `yolo11m.pt` - YOLOv11 medium model for person detection (download from Ultralytics)
- `pretrained_traj_lstm.pt` - Pre-trained LSTM model for trajectory forecasting (needs to be trained or obtained)

## How to obtain YOLO model:
```bash
# Install ultralytics
pip install ultralytics

# Download model
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
```

## LSTM Model:
The LSTM model structure is defined in `src/model.py`. You need to train it on pedestrian trajectory data (like ETH/UCY datasets) or obtain a pre-trained checkpoint.
