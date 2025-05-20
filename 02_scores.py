import numpy as np
import sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tabulate import tabulate
from scipy.stats import ttest_rel
import torch

# dane testowe - jeszcze nie użyte, zastanawiam się czy zostawić
data = np.load("test_set.npz")
X_tab_te = torch.from_numpy(data["X_tab"]).float()
X_ecg_te = torch.from_numpy(data["X_ecg"]).float()
y_te     = torch.from_numpy(data["y"]).long()

# wyniki
scores = np.load("deep_scores.npy")  # [model, grid, fold, metric]
model_names = ["early_lstm", "late_lstm", "early_gru", "late_gru"]
grid_str    = ["lr1e-3_bs32","lr1e-3_bs64","lr3e-4_bs32","lr3e-4_bs64"]


# średnie accuracy po foldach
acc_mean = scores[...,0].mean(axis=-1)      # shape (model, grid)
acc_mean_std = scores[...,1].std(axis=-1)

# najlepsze hiperparametry dla każdego modelu
print("\nNAJLEPSZE PARAMETRY:")
for i, m in enumerate(model_names):
    best = np.argmax(acc_mean[i])
    print(f"{m:<10} → {grid_str[best]} (acc={acc_mean[i,best]:.4f})")
table_mean = tabulate(acc_mean, headers=grid_str,
                 showindex=model_names, tablefmt="grid")

# acc po foldach
print("ŚREDNIE ACC:\n", table_mean)

table_std = tabulate(acc_mean_std, headers=grid_str,
                 showindex=model_names, tablefmt="grid")
print("STD:\n", table_mean)


# metryki na testowym
logits = np.load("test_logits.npy")
y_true = np.load("test_labels.npy")
y_pred = logits.argmax(1)

acc  = accuracy_score(y_true, y_pred) # tak z ciekawosci jak to sie ma do tego wyzej
prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0,1,2,3,4], zero_division=0)
print(f"\nTEST ACC: {acc:.4f}")
print("\nClassification report (test):\n",
      classification_report(y_true, y_pred, digits=3))

# confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion matrix (test)")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.show()

# porównanie modeli — najlepsze konfiguracje ===
best_cfg_idx = np.argmax(acc_mean[:, :, 0], axis=1)  # max Accuracy
best_scores = np.array([
    scores[m_idx, best_cfg_idx[m_idx], :, 0]  # accuracy
    for m_idx in range(scores.shape[0])
])  # shape: [n_models, n_folds]

# test t-Studenta dla par modeli
print("\n=== Testy t-Studenta między modelami (Accuracy, najlepsze konfiguracje) ===")
stat_mat = np.zeros((len(model_names), len(model_names)))

for i in range(len(model_names)):
    for j in range(len(model_names)):
        if i == j:
            stat_mat[i, j] = 0
        else:
            t, p = ttest_rel(best_scores[i], best_scores[j])
            stat_mat[i, j] = p < 0.05  # 1 jeśli istotna różnica

print(tabulate(stat_mat, headers=model_names, showindex=model_names, tablefmt="grid"))
