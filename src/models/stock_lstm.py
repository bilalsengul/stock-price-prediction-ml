import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_size=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, 2, batch_first=True)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def create_model():
    """Create a new model for testing."""
    return StockLSTM(input_size=4) 