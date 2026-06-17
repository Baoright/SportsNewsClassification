import torch
import torch.nn as nn

class LSTMPhoBERT(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=9, num_layers=2, dropout=0.5):
        super(LSTMPhoBERT, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)
