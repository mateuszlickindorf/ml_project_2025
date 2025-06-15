# === IMPORTY ===
import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import (
    StratifiedKFold, ParameterGrid, train_test_split, cross_val_predict, cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
from models import (
    ECG_CNN_LSTM, ECG_CNN_GRU, TabularMLP,
    EarlyFusion, LateFusion, EnhancedTabularMLP
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn as nn

# === USTAWIENIE URZĄDZENIA ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === FUNKCJE POMOCNICZE ===
def load_signal(path):
    try:
        return np.load(path)
    except:
        return np.zeros((12, 5000))

def assign_class(days):
    if np.isnan(days) or days > 365:
        return 0
    if days <= 1:
        return 1
    if days <= 7:
        return 2
    if days <= 365:
        return 3
    return 4

# === WCZYTANIE I ŁĄCZENIE DANYCH ===
df_tab = pd.read_csv('merged_filtered.csv', parse_dates=['deathdate', 'admittime'])
df_ecg = pd.read_csv('ecg_signals_list.csv', parse_dates=['ecg_time'])

assert df_ecg['subject_id'].value_counts().max() == 1, (
    "Pacjenci mają więcej niż jedno ECG – scalanie po subject_id błędne!"
)

df = pd.merge(
    df_tab,
    df_ecg[['subject_id', 'ecg_time', 'ecg_file']],
    on='subject_id',
    how='inner'
)

# === PRZYPISANIE KLASY ZGONU ===
df['time_to_death_days'] = (
    (df['deathdate'] - df['ecg_time']).dt.total_seconds() / 86400
)
df['death_class'] = df['time_to_death_days'].apply(assign_class)

# === TWORZENIE FOLDERÓW ===
os.makedirs("results/metadata", exist_ok=True)
os.makedirs("results/tabular_randomforest", exist_ok=True)
os.makedirs("results/pca_tsne", exist_ok=True)
os.makedirs("results/icd_tabular_randomforest", exist_ok=True)  # przyszłość
os.makedirs("results/ekg_tabular_randomforest", exist_ok=True)  # przyszłość
os.makedirs("results/fusion", exist_ok=True)  # przyszłość

# === ZAPIS METADANYCH PRZED BALANSOWANIEM ===
with open("results/metadata/dataset_overview.txt", "w") as f:
    f.write("Rozmiar danych wejściowych:\n")
    f.write(f"merged_filtered.csv: {len(df_tab)} rekordów\n")
    f.write(f"ecg_signals_list.csv: {len(df_ecg)} rekordów\n")
    f.write(f"Po scaleniu (df): {len(df)} rekordów\n\n")
    f.write("Liczność klas PRZED balansowaniem:\n")
    f.write(df["death_class"].value_counts().sort_index().to_string())
    f.write("\n")

# === BALANSOWANIE KLAS ===
min_count = df["death_class"].value_counts().min()
df_bal = (
    df.groupby("death_class", group_keys=False)
      .apply(lambda g: g.sample(min_count, random_state=42))
      .sample(frac=1, random_state=42)
      .reset_index(drop=True)
)

with open("results/metadata/class_distribution_balanced.txt", "w") as f:
    f.write("Liczba klas PO balansowaniu:\n")
    f.write(df_bal["death_class"].value_counts().sort_index().to_string())

# === PRZETWARZANIE DANYCH TABELARYCZNYCH I SYGNAŁOWYCH ===
num_cols = ["age_at_admit", "bmi", "systolic_bp", "diastolic_bp"]
cat_cols = ["gender", "race"]

scaler = StandardScaler()
X_num = scaler.fit_transform(df_bal[num_cols])
X_cat = pd.get_dummies(df_bal[cat_cols], drop_first=False).values
X_tab = np.hstack([X_num, X_cat])

df_bal["ecg_signal"] = df_bal["ecg_file"].apply(load_signal)
X_ecg = np.stack(df_bal["ecg_signal"].values)
y = df_bal["death_class"].values

X_tab_t = torch.from_numpy(X_tab).float().to(device)
X_ecg_t = torch.from_numpy(X_ecg).float().to(device)
y_t = torch.from_numpy(y).long().to(device)
tab_dim = X_tab_t.shape[1]

# === RANDOM FOREST NA DANYCH TABELARYCZNYCH ===
X = X_tab
clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_pred = cross_val_predict(clf, X, y, cv=5)

acc = accuracy_score(y, y_pred)
clf_report = classification_report(y, y_pred, digits=3)
cm = confusion_matrix(y, y_pred)

with open("results/tabular_randomforest/report.txt", "w") as f:
    f.write(f"Średnie accuracy (RandomForest): {acc:.4f}\n\n")
    f.write("Classification report:\n")
    f.write(clf_report)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.title("Confusion Matrix – RandomForest (tabular)")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig("results/tabular_randomforest/confusion_matrix.png")
plt.close()

# === PCA I t-SNE NA DANYCH EKG ===
X_flat = X_ecg.reshape(X_ecg.shape[0], -1)
nan_mask = ~np.isnan(X_flat).any(axis=1)
X_clean = X_flat[nan_mask]
y_clean = y[nan_mask]

X_pca = PCA(n_components=2).fit_transform(X_clean)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_clean, cmap='tab10', alpha=0.6)
plt.title("PCA na sygnałach EKG")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Klasa")
plt.tight_layout()
plt.savefig("results/pca_tsne/pca.png")
plt.close()

X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_clean)
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_clean, cmap='tab10', alpha=0.6)
plt.title("t-SNE na sygnałach EKG")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(label="Klasa")
plt.tight_layout()
plt.savefig("results/pca_tsne/tsne.png")
plt.close()

# === MODELE FUZJI EKG+TABULAR ===

from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score
)

models = {
    "early_lstm": lambda: EarlyFusion(ECG_CNN_LSTM(), TabularMLP(tab_dim)),
    "late_lstm" : lambda: LateFusion(ECG_CNN_LSTM(), TabularMLP(tab_dim)),
    "early_gru" : lambda: EarlyFusion(ECG_CNN_GRU(),  TabularMLP(tab_dim)),
    "late_gru"  : lambda: LateFusion(ECG_CNN_GRU(),  TabularMLP(tab_dim)),
}

param_grid = {
    "lr": [1e-3, 3e-4],
    "bs": [32, 64],
}
grid = list(ParameterGrid(param_grid))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# scores[model, grid, fold, metric]: metrics = acc, auc, f1, prec, rec
scores = np.zeros((len(models), len(grid), skf.get_n_splits(), 5))

def train_one(model, train_loader, val_loader, lr, bs, m_name, fold_idx, epochs=50, patience=4):
    tag = f"{m_name}_lr{lr}_bs{bs}_fold{fold_idx}"
    save_dir = f"results/fusion/{m_name}/lr{lr}_bs{bs}/fold{fold_idx}"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "model.pt")
    log_path = os.path.join(save_dir, "log.csv")
    report_path = os.path.join(save_dir, "report.txt")

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_acc, no_imp, best_state = 0, 0, None
    logs = []

    for ep in trange(epochs, desc=tag, leave=False):
        model.train()
        total_loss = 0.0
        for ecg_b, tab_b, y_b in train_loader:
            ecg_b, tab_b, y_b = ecg_b.to(device), tab_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = crit(model(ecg_b, tab_b), y_b)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for ecg_b, tab_b, y_b in val_loader:
                ecg_b, tab_b = ecg_b.to(device), tab_b.to(device)
                out = model(ecg_b, tab_b)
                preds.append(out.cpu())
                trues.append(y_b)
        preds = torch.cat(preds)
        trues = torch.cat(trues)
        acc = accuracy_score(trues, preds.argmax(1))

        logs.append({'epoch': ep+1, 'train_loss': avg_loss, 'val_acc': acc, 'early_stop_counter': no_imp})

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), save_path)

    pd.DataFrame(logs).to_csv(log_path, index=False)
    return model

# === TRENING MODELI FUZJI ===
for m_idx, (m_name, m_fn) in enumerate(models.items()):
    for g_idx, g in enumerate(grid):
        lr, bs = g["lr"], g["bs"]
        print(f"\n=== Model {m_name} | lr={lr}, bs={bs} ===")

        for f_idx, (train_id, val_id) in tqdm(
            enumerate(skf.split(X_tab_t, y_t)),
            total=skf.get_n_splits(),
            desc=f"{m_name} | lr={lr} | bs={bs}",
            leave=False
        ):
            tr_ds = TensorDataset(X_ecg_t[train_id], X_tab_t[train_id], y_t[train_id])
            va_ds = TensorDataset(X_ecg_t[val_id], X_tab_t[val_id], y_t[val_id])
            tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True)
            va_dl = DataLoader(va_ds, batch_size=bs, shuffle=False)

            model = m_fn()
            model = train_one(model, tr_dl, va_dl, lr, bs, m_name, f_idx)

            # Ewaluacja
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for ecg_b, tab_b, y_b in va_dl:
                    ecg_b, tab_b = ecg_b.to(device), tab_b.to(device)
                    out = model(ecg_b, tab_b)
                    preds.append(out.cpu())
                    trues.append(y_b)
            preds = torch.cat(preds)
            trues = torch.cat(trues)
            y_pred = preds.argmax(1).numpy()
            y_true = trues.numpy()

            acc  = accuracy_score(y_true, y_pred)
            f1   = f1_score(y_true, y_pred, average='macro')
            prec = precision_score(y_true, y_pred, average='macro')
            rec  = recall_score(y_true, y_pred, average='macro')
            try:
                auc = roc_auc_score(y_true, preds.softmax(1).numpy(), multi_class='ovr')
            except:
                auc = np.nan

            scores[m_idx, g_idx, f_idx] = [acc, auc, f1, prec, rec]

            report_txt = os.path.join(f"results/fusion/{m_name}/lr{lr}_bs{bs}/fold{f_idx}/report.txt")
            with open(report_txt, "w") as f:
                f.write(f"ACC: {acc:.4f}\nAUC: {auc:.4f}\nF1: {f1:.4f}\nPRE: {prec:.4f}\nREC: {rec:.4f}\n")

            print(f"  Fold {f_idx+1}: acc={acc:.3f}, auc={auc:.3f}, f1={f1:.3f}")

# === ZAPIS WYNIKÓW ZBIORCZYCH ===
np.save("results/fusion/deep_scores.npy", scores)

# === PODSUMOWANIE: NAJLEPSZE HIPERPARAMETRY I ŚREDNIE METRYKI ===
summary_path = "results/fusion/summary.txt"
with open(summary_path, "w") as f:
    for m_idx, m_name in enumerate(models):
        acc_avg = scores[m_idx, :, :, 0].mean(axis=1)
        best_g = np.argmax(acc_avg)
        f.write(f"{m_name:<12} | best: lr={grid[best_g]['lr']}, bs={grid[best_g]['bs']} | acc={acc_avg[best_g]:.4f}\n")
    f.write("\nŚREDNIE METRYKI (acc, auc, f1, prec, rec):\n")
    for metric_idx, metric_name in enumerate(['ACC', 'AUC', 'F1', 'PREC', 'REC']):
        f.write(f"\n=== {metric_name} ===\n")
        for m_idx, m_name in enumerate(models):
            means = scores[m_idx, :, :, metric_idx].mean(axis=1)
            f.write(f"{m_name:<12} " + " ".join(f"{val:.4f}" for val in means) + "\n")

# === ICD + TABULAR: MLP / RF / LightGBM ===

import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import lightgbm as lgb

os.makedirs("results/icd_tabular_randomforest", exist_ok=True)
os.makedirs("results/icd_tabular_mlp", exist_ok=True)
os.makedirs("results/icd_tabular_lgbm", exist_ok=True)

# 1. Wczytanie kodów ICD i przygotowanie
df_diag = pd.read_csv("records_w_diag_icd10.csv")
df_diag_sub = df_diag[df_diag["subject_id"].isin(df_bal["subject_id"])].copy()
df_diag_sub = df_diag_sub[["subject_id", "all_diag_all"]].dropna()
df_diag_sub["all_diag_all"] = df_diag_sub["all_diag_all"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)

df_diag_grouped = (
    df_diag_sub
    .groupby("subject_id")["all_diag_all"]
    .sum()
    .apply(set)
    .reset_index()
)

df_merged = df_bal.merge(df_diag_grouped, on="subject_id", how="left")
df_merged["all_diag_all"] = df_merged["all_diag_all"].apply(lambda x: [] if pd.isna(x) else list(x))
df_merged["icd_grouped"] = df_merged["all_diag_all"].apply(lambda codes: list(set(code[:3] for code in codes)))

mlb = MultiLabelBinarizer()
X_icd = mlb.fit_transform(df_merged["icd_grouped"])
X_tab_enhanced = np.hstack([X_tab, X_icd])
X_scaled = StandardScaler().fit_transform(X_tab_enhanced)
y = df_bal["death_class"].values

# === RANDOM FOREST ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_pred_rf = cross_val_predict(clf, X_tab_enhanced, y, cv=5)
acc_rf = accuracy_score(y, y_pred_rf)
report_rf = classification_report(y, y_pred_rf, digits=3)
cm_rf = confusion_matrix(y, y_pred_rf)

with open("results/icd_tabular_randomforest/report.txt", "w") as f:
    f.write(f"Accuracy: {acc_rf:.4f}\n\n")
    f.write(report_rf)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix – RandomForest (ICD+Tabular)")
plt.tight_layout()
plt.savefig("results/icd_tabular_randomforest/confusion_matrix.png")
plt.close()

# === ENHANCED MLP ===
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

class EnhancedTabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4)
        )
    def forward(self, x):
        return self.model(x)

def train_mlp_cv(X, y, device, epochs=50, batch_size=64):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = np.zeros_like(y)

    for train_idx, val_idx in skf.split(X, y):
        model = EnhancedTabularMLP(input_dim=X.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        crit = nn.CrossEntropyLoss(weight=class_weights_tensor)

        X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
        y_train = torch.tensor(y[train_idx], dtype=torch.long).to(device)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(device)

        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_val).cpu().numpy()
            all_preds[val_idx] = logits.argmax(1)

    return all_preds

y_pred_mlp = train_mlp_cv(X_scaled, y, device)
acc_mlp = accuracy_score(y, y_pred_mlp)
report_mlp = classification_report(y, y_pred_mlp, digits=3)
cm_mlp = confusion_matrix(y, y_pred_mlp)

with open("results/icd_tabular_mlp/report.txt", "w") as f:
    f.write(f"Accuracy: {acc_mlp:.4f}\n\n")
    f.write(report_mlp)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Greens', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix – MLP (ICD+Tabular)")
plt.tight_layout()
plt.savefig("results/icd_tabular_mlp/confusion_matrix.png")
plt.close()

# === LIGHTGBM ===
clf = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=42
)
y_pred_lgbm = cross_val_predict(clf, X_tab_enhanced, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
acc_lgbm = accuracy_score(y, y_pred_lgbm)
report_lgbm = classification_report(y, y_pred_lgbm, digits=3)
cm_lgbm = confusion_matrix(y, y_pred_lgbm)

with open("results/icd_tabular_lgbm/report.txt", "w") as f:
    f.write(f"Accuracy: {acc_lgbm:.4f}\n\n")
    f.write(report_lgbm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_lgbm, annot=True, fmt='d', cmap='Purples', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix – LightGBM (ICD+Tabular)")
plt.tight_layout()
plt.savefig("results/icd_tabular_lgbm/confusion_matrix.png")
plt.close()

# === WAŻNOŚĆ CECH W LIGHTGBM ===
clf.fit(X_tab_enhanced, y)

tabular_feature_names = list(df_bal[["age_at_admit", "bmi", "systolic_bp", "diastolic_bp"]].columns)
cat_feature_names = list(pd.get_dummies(df_bal[["gender", "race"]], drop_first=False).columns)
icd_feature_names = list(mlb.classes_)
feature_names = tabular_feature_names + cat_feature_names + icd_feature_names

importances = clf.feature_importances_
feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feat_imp.head(25), x="importance", y="feature", palette="viridis")
plt.title("Top 25 najważniejszych cech – LightGBM")
plt.tight_layout()
plt.savefig("results/icd_tabular_lgbm/feature_importance.png")
plt.close()
