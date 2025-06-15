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
                    TabularMLP, EarlyFusion, LateFusion, EnhancedTabularMLP)
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer

########################ŁADOWANIE SYGNAŁU I BALANSOWANIE KLAS ###########################################
def load_signal(path):
    try:
        return np.load(path)
    except:
        return np.zeros((12, 5000))  #jeśli pliku nie ma

def assign_class(days):
    if np.isnan(days) or days > 365: return 0
    if days <= 1:   return 1
    if days <= 7:  return 2
    if days <= 365: return 3
    return 4

# Ustawienie urządzenia
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

# # OPCJONALNE: Zapisywanie sygnałów ECG do folderów związanych z klasami

# # Gdzie mają trafić pliki?
# target_root = "ecg_by_class"
# os.makedirs(target_root, exist_ok=True)

# # Utwórz podfoldery class_0, class_1, ...
# for i in sorted(df_bal["death_class"].unique()):
#     os.makedirs(os.path.join(target_root, f"class_{i}"), exist_ok=True)

# # Dla każdego rekordu skopiuj plik do odpowiedniego folderu
# for _, row in tqdm(df_bal.iterrows(), total=len(df_bal), desc="Copying ECGs to class folders"):
#     src = row["ecg_file"]  # ścieżka do .npy
#     cls = row["death_class"]
#     fname = os.path.basename(src)
#     dst = os.path.join(target_root, f"class_{cls}", fname)
#     if os.path.exists(src):  # tylko jeśli plik istnieje
#         shutil.copy2(src, dst)
#     else:
#         print(f"[!] File not found: {src}")


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



######################### RANDOM FOREST ###########################################
# # Dane tabelaryczne
# X = X_tab
# y = df_bal["death_class"].values

# clf = RandomForestClassifier(n_estimators=100, random_state=42)

# # cross_val_predict zwraca przewidziane klasy (nie accuracy!)
# y_pred = cross_val_predict(clf, X, y, cv=5)

# # --- ogólna skuteczność
# acc = accuracy_score(y, y_pred)
# print(f"Średnie accuracy (RandomForest): {acc:.4f}")

# # --- raport metryk per klasa
# print("\nClassification report:")
# print(classification_report(y, y_pred, digits=3))

# # --- macierz pomyłek
# cm = confusion_matrix(y, y_pred)
# plt.figure(figsize=(6,5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=np.unique(y),
#             yticklabels=np.unique(y))
# plt.title("Confusion Matrix – RandomForest (dane tabelaryczne)")
# plt.xlabel("Predicted label")
# plt.ylabel("True label")
# plt.show()

# ######################### PCA I TSNE ###########################################
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# # Załaduj dane
# X = X_ecg.reshape(len(X_ecg), -1)   # (n_samples, 60000)
# y = y  # etykiety klas

# print("Shape X_ecg:", X_ecg.shape)
# print("NaNs in X_ecg:", np.isnan(X_ecg).sum())
# nan_mask = np.isnan(X_ecg).any(axis=(1,2))  # które próbki mają jakiekolwiek NaN
# print(f"Liczba próbek z NaN: {nan_mask.sum()} / {len(X_ecg)}")
# # 1. Spłaszczamy EKG (np. globalnie po wszystkich kanałach)
# X = X_ecg.reshape(X_ecg.shape[0], -1)  # (2348, 60000)
# print("Shape X_ecg:", X_ecg.shape)
# print("NaNs in X_ecg:", np.isnan(X_ecg).sum())
# # 2. Tworzymy maskę na brak NaN
# nan_mask = ~np.isnan(X).any(axis=1)   # True → brak NaN

# # 3. Filtrowanie danych
# X_clean = X[nan_mask]
# y_clean = y[nan_mask]
# print("Shape X_clean:", X_clean.shape)
# print("NaNs in X_clean:", np.isnan(X_clean).sum())

# # 4. PCA (teraz już bez NaN)
# X_pca = PCA(n_components=2).fit_transform(X_clean)

# plt.figure(figsize=(8,6))
# plt.scatter(X_pca[:,0], X_pca[:,1], c=y_clean, cmap='tab10', alpha=0.6)
# plt.title("PCA na sygnałach EKG")
# plt.xlabel("PC1"); plt.ylabel("PC2")
# plt.colorbar(label="Klasa")
# plt.show()

# # t-SNE (lepsza jakość, wolniejsze)
# X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_clean)

# plt.figure(figsize=(8,6))
# plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_clean, cmap='tab10', alpha=0.6)
# plt.title("t-SNE na sygnałach EKG")
# plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
# plt.colorbar(label="Klasa")
# plt.show()

# ######################### MODELE I TRENING ###########################################

######## MLP ########
import ast 
# 1. Wczytaj plik z ICD
df_diag = pd.read_csv("records_w_diag_icd10.csv")

# 2. Odfiltruj tylko pacjentów obecnych w df_bal
df_diag_sub = df_diag[df_diag["subject_id"].isin(df_bal["subject_id"])].copy()

# 3. Zostaw tylko potrzebne kolumny i przekształć string → lista
df_diag_sub = df_diag_sub[["subject_id", "all_diag_all"]].dropna()
df_diag_sub["all_diag_all"] = df_diag_sub["all_diag_all"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)

# 4. Grupowanie po subject_id – scal wszystkie kody
df_diag_grouped = (
    df_diag_sub
    .groupby("subject_id")["all_diag_all"]
    .sum()  # scala wszystkie listy
    .apply(set)  # usuwa duplikaty
    .reset_index()
)

# 5. Merge z df_bal
df_merged = df_bal.merge(df_diag_grouped, on="subject_id", how="left")
df_merged["all_diag_all"] = df_merged["all_diag_all"].apply(lambda x: [] if pd.isna(x) else list(x))

# 6. Grupuj kody ICD do 3-literowych bloków (np. "K7469" → "K74")
df_merged["icd_grouped"] = df_merged["all_diag_all"].apply(
    lambda codes: list(set(code[:3] for code in codes))
)

# 7. One-hot encoding (MultiLabelBinarizer)
mlb = MultiLabelBinarizer()
X_icd = mlb.fit_transform(df_merged["icd_grouped"])

print(f"Rozmiar macierzy kodów ICD: {X_icd.shape} (n próbek x {len(mlb.classes_)} grup ICD)")

# 8. Połącz z danymi tabelarycznymi
X_tab_enhanced = np.hstack([X_tab, X_icd])
X_tab_enhanced_t = torch.from_numpy(X_tab_enhanced).float().to(device)

X = X_tab_enhanced
y = df_bal["death_class"].values


####nowe
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_tab_enhanced)


# ----------- 3. RANDOM FOREST -----------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_pred_rf = cross_val_predict(clf, X, y, cv=5)

print("=== RandomForest ===")
print(f"Accuracy: {accuracy_score(y, y_pred_rf):.4f}")
print(classification_report(y, y_pred_rf))

cm_rf = confusion_matrix(y, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix – RandomForest")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# ----------- 4. ENHANCED MLP -----------
from sklearn.utils.class_weight import compute_class_weight

# 2. Wagi klas
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# 3. MLP z dropoutem i większą pojemnością
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
            nn.Linear(hidden_dim // 2, 4)  # 4 klasy
        )

    def forward(self, x):
        return self.model(x)

# 4. Trening z CV
def train_mlp_cv(X, y, device, epochs=50, batch_size=64):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = np.zeros_like(y)

    for train_idx, val_idx in skf.split(X, y):
        model = EnhancedTabularMLP(input_dim=X.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # L2 regularization
        crit = nn.CrossEntropyLoss(weight=class_weights_tensor)

        X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
        y_train = torch.tensor(y[train_idx], dtype=torch.long).to(device)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(device)

        ds = TensorDataset(X_train, y_train)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                out = model(xb)
                loss = crit(out, yb)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_val).cpu().numpy()
            preds = logits.argmax(1)
            all_preds[val_idx] = preds

    return all_preds

# 5. Predykcje i ewaluacja
y_pred_mlp = train_mlp_cv(X_scaled, y, device)

print("=== MLP z poprawkami ===")
print(f"Accuracy: {accuracy_score(y, y_pred_mlp):.4f}")
print(classification_report(y, y_pred_mlp, digits=3))

cm = confusion_matrix(y, y_pred_mlp)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix – MLP (scaled + weights)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

#########Light_GBM#############
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict


# 1. Dane wejściowe
X = X_tab_enhanced  # zakładam, że masz to już połączone (tab + ICD)
y = df_bal["death_class"].values

# 2. Model LightGBM
clf = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=42
)

# 3. Cross-val predict
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(clf, X, y, cv=skf)

# 4. Wyniki
acc = accuracy_score(y, y_pred)
print("=== LightGBM ===")
print(f"Accuracy: {acc:.4f}")
print(classification_report(y, y_pred, digits=3))

# 5. Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix – LightGBM")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Lista nazw cech: jeśli masz połączone X_tab + ICD
tabular_feature_names = list(df_bal[["age_at_admit", "bmi", "systolic_bp", "diastolic_bp"]].columns)
cat_feature_names = list(pd.get_dummies(df_bal[["gender", "race"]], drop_first=False).columns)
icd_feature_names = list(mlb.classes_)  # z MultiLabelBinarizer

# Zakładamy kolejność: tabular + ICD
feature_names = tabular_feature_names + cat_feature_names + icd_feature_names

# Stwórz DataFrame z ważnościami
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

clf.fit(X, y)

importances = clf.feature_importances_
feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# Wyświetl top 25
plt.figure(figsize=(10, 8))
sns.barplot(data=feat_imp.head(25), x="importance", y="feature", palette="viridis")
plt.title("Top 25 najważniejszych cech – LightGBM")
plt.xlabel("Wartość ważności (gain)")
plt.ylabel("Cecha")
plt.tight_layout()
plt.show()

# # 4) modele
# models = {
#     "early_lstm": lambda: EarlyFusion(ECG_CNN_LSTM(), TabularMLP(tab_dim)),
#     "late_lstm" : lambda: LateFusion(ECG_CNN_LSTM(), TabularMLP(tab_dim)),
#     "early_gru" : lambda: EarlyFusion(ECG_CNN_GRU(),  TabularMLP(tab_dim)),
#     "late_gru"  : lambda: LateFusion(ECG_CNN_GRU(),  TabularMLP(tab_dim)),
# }

# # 5) grid search
# param_grid = {
#     "lr":   [1e-3, 3e-4],
#     "bs":   [32, 64],
# }
# grid = list(ParameterGrid(param_grid))

# # 6) cross-validation
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# # score tensor: [model, grid_combo, fold, metrics(0=acc,1=auc)]
# scores = np.zeros((len(models), len(grid), skf.get_n_splits(), 2)) # ostatni parametr - liczba metryk

# # 7) trening
# def train_one(model, train_loader, val_loader, lr, bs, m_name, fold_idx, epochs=50, patience=4):
#     import csv

#     # Save directory setup
#     os.makedirs("saved_models", exist_ok=True)
#     os.makedirs("logs", exist_ok=True)
    
#     model_name_tag = f"{m_name}_lr{lr}_bs{bs}_fold{fold_idx}"
#     save_path = f"saved_models/{model_name_tag}.pt"
#     log_path  = f"logs/{model_name_tag}.csv"

#     # Initialize
#     model.to(device)
#     opt  = torch.optim.Adam(model.parameters(), lr=lr)
#     crit = nn.CrossEntropyLoss()
#     best_acc, no_imp, best_state = 0, 0, None
#     logs = []

#     for ep in trange(epochs, desc=model_name_tag, leave=False):
#         # Train
#         model.train()
#         running_loss = 0.0
#         for ecg_b, tab_b, y_b in train_loader:
#             ecg_b, tab_b, y_b = ecg_b.to(device), tab_b.to(device), y_b.to(device)
#             opt.zero_grad()
#             loss = crit(model(ecg_b, tab_b), y_b)
#             loss.backward()
#             opt.step()
#             running_loss += loss.item()

#         avg_loss = running_loss / len(train_loader)

#         # Validate
#         model.eval()
#         preds, trues = [], []
#         with torch.no_grad():
#             for ecg_b, tab_b, y_b in val_loader:
#                 ecg_b, tab_b = ecg_b.to(device), tab_b.to(device)
#                 out = model(ecg_b, tab_b)
#                 preds.append(out.cpu()); trues.append(y_b)
#         preds = torch.cat(preds); trues = torch.cat(trues)
#         acc = accuracy_score(trues, preds.argmax(1))

#         # Logging
#         logs.append({
#             'epoch': ep + 1,
#             'train_loss': avg_loss,
#             'val_acc': acc,
#             'early_stop_counter': no_imp
#         })

#         if acc > best_acc:
#             best_acc, no_imp = acc, 0
#             best_state = model.state_dict()
#         else:
#             no_imp += 1
#         if no_imp >= patience:
#             break

#     # Restore and save best model
#     if best_state is not None:
#         model.load_state_dict(best_state)
#         torch.save(model.state_dict(), save_path)
#         print(f"[✓] Saved best model: {save_path} (acc={best_acc:.3f})")
#     else:
#         print("[!] No improvement found during training.")

#     # Save logs to CSV
#     pd.DataFrame(logs).to_csv(log_path, index=False)
#     print(f"[i] Log saved to: {log_path}")

#     return model


# for m_idx, (m_name, m_fn) in enumerate(models.items()):
#     for g_idx, g in enumerate(grid):
#         lr, bs = g["lr"], g["bs"]
#         print(f"\n=== Model {m_name} | lr={lr}, bs={bs} ===")

#         fold_iterator = tqdm(
#             enumerate(skf.split(X_tab_t, y_t)),
#             total=skf.get_n_splits(),
#             desc=f"{m_name} | lr={lr} | bs={bs}",
#             leave=False
#         )

#         for f_idx, (train_id, val_id) in fold_iterator:
#             tr_ds = TensorDataset(X_ecg_t[train_id], X_tab_t[train_id], y_t[train_id])
#             va_ds = TensorDataset(X_ecg_t[val_id], X_tab_t[val_id], y_t[val_id])

#             loader_tr = DataLoader(tr_ds, batch_size=bs, shuffle=True)
#             loader_va = DataLoader(va_ds, batch_size=bs, shuffle=False)

#             model = m_fn()
#             model = train_one(model, loader_tr, loader_va, lr, bs, m_name, f_idx)

#             model.eval()
#             preds, trues = [], []
#             with torch.no_grad():
#                 for ecg_b, tab_b, y_b in loader_va:
#                     ecg_b, tab_b = ecg_b.to(device), tab_b.to(device)
#                     out = model(ecg_b, tab_b)
#                     preds.append(out.cpu()); trues.append(y_b)
#             preds = torch.cat(preds); trues = torch.cat(trues)
#             acc = accuracy_score(trues, preds.argmax(1))
#             try:
#                 auc = roc_auc_score(trues.numpy(),
#                                     preds.softmax(1).numpy(),
#                                     multi_class='ovr')
#             except:
#                 auc = np.nan

#             scores[m_idx, g_idx, f_idx, 0] = acc
#             scores[m_idx, g_idx, f_idx, 1] = auc
#             print(f"  Fold {f_idx+1}: acc={acc:.3f}, auc={auc:.3f}")
# # ---------- 6) ZAPIS WYNIKÓW ----------
# np.save("deep_scores.npy", scores)

# # TODO:
# # - Wczytać machine_measurements.csv i dodać globalne cechy maszynowe
# # - Late fusion model
# # - Grid search / StratifiedKFold
# # - Zapis modelu torch.save(...)
