# Model Files

This directory should contain:

- `yolo11m.pt` - YOLOv11 medium model for person detection (download from Ultralytics)
- `pretrained_traj_lstm.pt` - Pre-trained LSTM model for trajectory forecasting (train using provided script)

## How to obtain YOLO model:
```bash
# Install ultralytics
pip install ultralytics

# Download model
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
```

## LSTM Model Training:
The LSTM model structure is defined in `src/model.py`. Train it on your own trajectory data:

```bash
# Train on your world trajectories
python src/train_traj_lstm.py --data outputs/world_trajectories.json --out models/pretrained_traj_lstm.pt

# With custom hyperparameters
python src/train_traj_lstm.py --data outputs/world_trajectories.json --out models/pretrained_traj_lstm.pt --epochs 400 --hidden 128 --obs-len 5 --pred-len 5
```

**Training Features:**
- 20% track holdout for honest validation
- Reports ADE (Average Displacement Error) and FDE (Final Displacement Error) in meters
- Prevents circular validation on test data
- Supports custom observation/prediction lengths and time steps

**Alternative:** Train on standard pedestrian datasets (ETH/UCY) for better generalization. The training script accepts data in the same schema as `world_trajectories.json`.
