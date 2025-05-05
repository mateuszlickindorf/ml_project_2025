import os
import numpy  as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import accuracy_score, roc_auc_score

import wfdb #pip install wfdb

# 1) wczytanie danych
df = pd.read_csv('merged_filtered.csv', parse_dates=['deathdate'])
num_cols = ['age_at_admit', 'bmi', 'blood_pressure']
cat_cols = ['gender', 'race']

# obróbka tabelaryczna
df_cat = pd.get_dummies(df[cat_cols], drop_first=False)
scaler = StandardScaler()
df_num = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)
X_tab = np.hstack([df_num.values, df_cat.values])
y     = df['deathdate'].notna().astype(int).values

# 2) wczytanie listy plików ecg i filtracja
# record_list.csv zawiera: subject_id, study_id, filepath (ścieżka bez .hea/.dat), ecg_time
records = pd.read_csv('record_list.csv', parse_dates=['ecg_time'])
# wybieramy tylko te ECG, gdzie ecg_time jest między admit i death (jeśli death brakuje, przyjmujemy do discharge)
df = df.merge(records, on='subject_id', how='left')
df = df[df['ecg_time'].notna()]  # usuwamy bez ECG
# synchronizacja: tylko ecg_time >= admittime (jeśli admittime mamy) i <= deathdate
# TODO: dodać
#  admittime

# 3) wczytanie i przygotowanie sygnałów ECG
ecg_signals = []
for idx, row in df.iterrows():
    rec_path = row['filepath']      # np. 'files/p1000/p10001725/s41420867/41420867'
    try:
        rec = wfdb.rdrecord(rec_path)
        sig = rec.p_signal         # shape (n_samples, 12)
        # jeśli trzeba: przytnij / ujednolić długość do stałej (np. 5000 próbek)
        sig = sig[:5000, :]        # 10s @500Hz = 5000 próbek
        sig = sig.T                 # -> (12, 5000)
    except Exception as e:
        # jeśli błąd wczytania, zastąp zerami
        sig = np.zeros((12, 5000))
    ecg_signals.append(sig)
X_ecg = np.stack(ecg_signals)       # (N, 12, 5000)

# 4) tensory pytorch
X_tab_tensor = torch.tensor(X_tab, dtype=torch.float32)
X_ecg_tensor = torch.tensor(X_ecg, dtype=torch.float32)
y_tensor     = torch.tensor(y, dtype=torch.long)

# 5) definicja modeli

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
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x_ecg, x_tab):
        e = self.ecg(x_ecg)
        t = self.tab(x_tab)
        return self.clf(torch.cat([e,t], dim=1)).squeeze()

# TODO: Dodać LateFusion analogicznie

# 6) dataloadery
idx = np.arange(len(y))
tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
ds_tr = TensorDataset(X_ecg_tensor[tr], X_tab_tensor[tr], y_tensor[tr])
ds_te = TensorDataset(X_ecg_tensor[te], X_tab_tensor[te], y_tensor[te])
loader_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)
loader_te = DataLoader(ds_te, batch_size=32, shuffle=False)

# 7) trening i ewaluacja
model = EarlyFusion(ECG_CNN_LSTM(), TabularMLP(X_tab.shape[1]))
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
crit  = nn.BCELoss()

for epoch in range(1,11):
    model.train()
    for x_e, x_t, yb in loader_tr:
        opt.zero_grad()
        loss = crit(model(x_e, x_t), yb.float())
        loss.backward(); opt.step()
    # TODO: walidacja co epokę, logi

# 8) test
model.eval()
preds, trues = [], []
with torch.no_grad():
    for x_e, x_t, yb in loader_te:
        p = model(x_e, x_t).cpu().numpy()
        preds.append(p); trues.append(yb.numpy())
preds = np.concatenate(preds)
trues = np.concatenate(trues)
label = (preds>0.5).astype(int)
print("Acc:", accuracy_score(trues, label),
      "AUC:", roc_auc_score(trues, preds))

# TODO:
# - Wczytać machine_measurements.csv i dodać globalne cechy maszynowe
# - Late fusion model
# - Grid search / StratifiedKFold
# - Zapis modelu torch.save(...)
