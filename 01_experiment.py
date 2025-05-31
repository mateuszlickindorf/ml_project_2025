import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, ParameterGrid, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm, trange
from models import (ECG_CNN_LSTM, ECG_CNN_GRU,
                    TabularMLP, EarlyFusion, LateFusion)

def load_signal(path):
    try:
        return np.load(path)
    except:
        return np.zeros((12, 5000))  #jeśli pliku nie ma


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) wczytanie danych
df_tab = pd.read_csv('merged_filtered.csv',
                     parse_dates=['deathdate', 'admittime'])
df_ecg = pd.read_csv('ecg_signals_list.csv',
                     parse_dates=['ecg_time'])

assert df_ecg['subject_id'].value_counts().max() == 1, "Pacjenci mają więcej niż jedno ECG – scalanie po subject_id błędne!"

df = pd.merge(df_tab,
              df_ecg[['subject_id', 'ecg_time', 'ecg_file']],
              on=['subject_id'],
              how='inner')

# czas do zgonu -> klasa 0-4
df['time_to_death_days'] = (
    (df['deathdate'] - df['ecg_time']).dt.total_seconds() / 86400
)

def assign_class(days):
    if np.isnan(days) or days > 365: return 0
    if days <= 1:   return 1
    if days <= 7:  return 2
    if days <= 365: return 3
    return 4
df['death_class'] = df['time_to_death_days'].apply(assign_class)

print("Liczność klas przed balansowaniem:")
print(df["death_class"].value_counts().sort_index())

# zbalansowanie klas
min_count = df["death_class"].value_counts().min()
df_bal = (
    df.groupby("death_class", group_keys=False)
      .apply(lambda g: g.sample(min_count, random_state=42))
      .sample(frac=1, random_state=42)        # dodatkowo mieszamy cały zbiór
      .reset_index(drop=True)
)

print("Liczba przykładów w każdej klasie po balansowaniu:")
print(df_bal["death_class"].value_counts())

# tabular
num_cols = ["age_at_admit", "bmi", "systolic_bp", "diastolic_bp"]
cat_cols = ["gender", "race"]

scaler   = StandardScaler()
X_num    = scaler.fit_transform(df_bal[num_cols])
X_cat    = pd.get_dummies(df_bal[cat_cols], drop_first=False).values
X_tab    = np.hstack([X_num, X_cat])

df_bal["ecg_signal"] = df_bal["ecg_file"].apply(load_signal) #ładowanie sygału z plików
X_ecg = np.stack(df_bal["ecg_signal"].values)
y = df_bal["death_class"].values

X_tab_t = torch.from_numpy(X_tab).float().to(device)
X_ecg_t = torch.from_numpy(X_ecg).float().to(device)
y_t     = torch.from_numpy(y).long().to(device)
tab_dim = X_tab_t.shape[1]


# 4) modele
models = {
    "early_lstm": lambda: EarlyFusion(ECG_CNN_LSTM(), TabularMLP(tab_dim)),
    "late_lstm" : lambda: LateFusion(ECG_CNN_LSTM(), TabularMLP(tab_dim)),
    "early_gru" : lambda: EarlyFusion(ECG_CNN_GRU(),  TabularMLP(tab_dim)),
    "late_gru"  : lambda: LateFusion(ECG_CNN_GRU(),  TabularMLP(tab_dim)),
}

# 5) grid search
param_grid = {
    "lr":   [1e-3, 3e-4],
    "bs":   [32, 64],
}
grid = list(ParameterGrid(param_grid))

# 6) cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# score tensor: [model, grid_combo, fold, metrics(0=acc,1=auc)]
scores = np.zeros((len(models), len(grid), skf.get_n_splits(), 2)) # ostatni parametr - liczba metryk

# 7) trening
def train_one(model, train_loader, val_loader, lr, bs, m_name, fold_idx, epochs=50, patience=4):
    import csv

    # Save directory setup
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    model_name_tag = f"{m_name}_lr{lr}_bs{bs}_fold{fold_idx}"
    save_path = f"saved_models/{model_name_tag}.pt"
    log_path  = f"logs/{model_name_tag}.csv"

    # Initialize
    model.to(device)
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_acc, no_imp, best_state = 0, 0, None
    logs = []

    for ep in trange(epochs, desc=model_name_tag, leave=False):
        # Train
        model.train()
        running_loss = 0.0
        for ecg_b, tab_b, y_b in train_loader:
            ecg_b, tab_b, y_b = ecg_b.to(device), tab_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = crit(model(ecg_b, tab_b), y_b)
            loss.backward()
            opt.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Validate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for ecg_b, tab_b, y_b in val_loader:
                ecg_b, tab_b = ecg_b.to(device), tab_b.to(device)
                out = model(ecg_b, tab_b)
                preds.append(out.cpu()); trues.append(y_b)
        preds = torch.cat(preds); trues = torch.cat(trues)
        acc = accuracy_score(trues, preds.argmax(1))

        # Logging
        logs.append({
            'epoch': ep + 1,
            'train_loss': avg_loss,
            'val_acc': acc,
            'early_stop_counter': no_imp
        })

        if acc > best_acc:
            best_acc, no_imp = acc, 0
            best_state = model.state_dict()
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    # Restore and save best model
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), save_path)
        print(f"[✓] Saved best model: {save_path} (acc={best_acc:.3f})")
    else:
        print("[!] No improvement found during training.")

    # Save logs to CSV
    pd.DataFrame(logs).to_csv(log_path, index=False)
    print(f"[i] Log saved to: {log_path}")

    return model


for m_idx, (m_name, m_fn) in enumerate(models.items()):
    for g_idx, g in enumerate(grid):
        lr, bs = g["lr"], g["bs"]
        print(f"\n=== Model {m_name} | lr={lr}, bs={bs} ===")

        fold_iterator = tqdm(
            enumerate(skf.split(X_tab_t, y_t)),
            total=skf.get_n_splits(),
            desc=f"{m_name} | lr={lr} | bs={bs}",
            leave=False
        )

        for f_idx, (train_id, val_id) in fold_iterator:
            tr_ds = TensorDataset(X_ecg_t[train_id], X_tab_t[train_id], y_t[train_id])
            va_ds = TensorDataset(X_ecg_t[val_id], X_tab_t[val_id], y_t[val_id])

            loader_tr = DataLoader(tr_ds, batch_size=bs, shuffle=True)
            loader_va = DataLoader(va_ds, batch_size=bs, shuffle=False)

            model = m_fn()
            model = train_one(model, loader_tr, loader_va, lr, bs, m_name, f_idx)

            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for ecg_b, tab_b, y_b in loader_va:
                    ecg_b, tab_b = ecg_b.to(device), tab_b.to(device)
                    out = model(ecg_b, tab_b)
                    preds.append(out.cpu()); trues.append(y_b)
            preds = torch.cat(preds); trues = torch.cat(trues)
            acc = accuracy_score(trues, preds.argmax(1))
            try:
                auc = roc_auc_score(trues.numpy(),
                                    preds.softmax(1).numpy(),
                                    multi_class='ovr')
            except:
                auc = np.nan

            scores[m_idx, g_idx, f_idx, 0] = acc
            scores[m_idx, g_idx, f_idx, 1] = auc
            print(f"  Fold {f_idx+1}: acc={acc:.3f}, auc={auc:.3f}")
# ---------- 6) ZAPIS WYNIKÓW ----------
np.save("deep_scores.npy", scores)

# TODO:
# - Wczytać machine_measurements.csv i dodać globalne cechy maszynowe
# - Late fusion model
# - Grid search / StratifiedKFold
# - Zapis modelu torch.save(...)
