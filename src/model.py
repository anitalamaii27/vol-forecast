import torch 
import torch.nn as nn

class VolLSTM(nn.Module):
    def __init__(self, n_features=1, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

if __name__ == "__main__":
    model = VolLSTM()
    fake = torch.randn(32, 20, 1)
    print(model(fake).shape)