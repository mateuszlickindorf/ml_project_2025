import torch
import torch.nn as nn
import torch.nn.functional as F


class ECG_CNN_LSTM(nn.Module):
    def __init__(self, in_ch=12):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
    def forward(self, x):
        # x: (batch, ch, seq)
        x = F.relu(self.conv(x))
        x = self.pool(x)            # -> (batch, 32, seq/2)
        x = x.permute(0,2,1)        # -> (batch, seq/2, 32)
        _, (h,_) = self.lstm(x)
        return h[-1]                # (batch, 64)

class ECG_CNN_GRU(nn.Module):
    def __init__(self, in_ch: int = 12):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.gru  = nn.GRU(32, 64, batch_first=True)   # ← GRU zamiast LSTM

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = F.relu(self.conv(x))
        x = self.pool(x)            # (batch, 32, seq_len/2)
        x = x.permute(0, 2, 1)      # (batch, seq_len/2, 32)  —> time-major dla GRU
        _, h = self.gru(x)          # h: (num_layers, batch, 64)
        return h[-1]                # (batch, 64) — ostatnia warstwa

class TabularMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class EarlyFusion(nn.Module):
    def __init__(self, ecg_m, tab_m):
        super().__init__()
        self.ecg = ecg_m
        self.tab = tab_m
        self.clf = nn.Sequential(
            nn.Linear(64+32, 64), nn.ReLU(),
            nn.Linear(64, 5)  # 5 klas
        )
    def forward(self, x_ecg, x_tab):
        e = self.ecg(x_ecg)
        t = self.tab(x_tab)
        return self.clf(torch.cat([e,t], dim=1)).squeeze()

class LateFusion(nn.Module):
    def __init__(self, ecg_m, tab_m):
        super().__init__()
        self.ecg = ecg_m
        self.tab = tab_m

        # Osobne klasyfikatory dla każdego wejścia
        self.ecg_clf = nn.Linear(64, 5)  # 5 klas
        self.tab_clf = nn.Linear(32, 5)  # 5 klas

    def forward(self, x_ecg, x_tab):
        e_feat = self.ecg(x_ecg)           # (batch, 64)
        t_feat = self.tab(x_tab)           # (batch, 32)

        e_logits = self.ecg_clf(e_feat)    # (batch, 5)
        t_logits = self.tab_clf(t_feat)    # (batch, 5)

        # Late fusion przez sumowanie logitów
        return e_logits + t_logits         # (batch, 5)
        # alternatywy:
        # return 0.7 * e_logits + 0.3 * t_logits - wspołczynniki wag
        # return (e_logits + t_logits) / 2