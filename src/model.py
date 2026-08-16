"""
TrajLSTM: Simple LSTM model for trajectory forecasting.

This is a basic LSTM model for pedestrian trajectory prediction.
It takes observed displacements and predicts future displacements.

Expected checkpoint format:
{
    "hidden": int,          # LSTM hidden size
    "state_dict": dict,     # PyTorch state dict
    "obs_len": int,         # Observation length
    "pred_len": int,        # Prediction length  
    "step_dt_sec": float    # Time step in seconds
}
"""
import torch
import torch.nn as nn


class TrajLSTM(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.hidden = hidden
        # Input: (batch, obs_len, 2) for x,y displacements
        self.lstm = nn.LSTM(2, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)
    
    def forward(self, obs, pred_len=12, target=None):
        """
        Args:
            obs: (batch, obs_len, 2) observed displacements
            pred_len: number of future steps to predict
            target: optional target for training (not used in inference)
        
        Returns:
            pred: (batch, pred_len, 2) predicted displacements
        """
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(obs)
        
        # Use last hidden state to predict future steps
        last_hidden = h_n  # (1, batch, hidden)
        
        # Auto-regressive prediction
        pred = []
        current_input = obs[:, -1:, :]  # (batch, 1, 2)
        h, c = last_hidden, c_n
        
        for _ in range(pred_len):
            lstm_out, (h, c) = self.lstm(current_input, (h, c))
            next_disp = self.fc(lstm_out)  # (batch, 1, 2)
            pred.append(next_disp)
            current_input = next_disp
        
        pred = torch.cat(pred, dim=1)  # (batch, pred_len, 2)
        return pred


# NOTE: The pretrained_traj_lstm.pt checkpoint file is missing.
# You'll need to either:
# 1. Train the model on ETH/UCY pedestrian dataset
# 2. Download a pre-trained checkpoint if available
# 3. Contact the original source for the checkpoint file

# For now, this model structure is provided to make the import work,
# but stage6_forecast.py will fail without the actual checkpoint file.
