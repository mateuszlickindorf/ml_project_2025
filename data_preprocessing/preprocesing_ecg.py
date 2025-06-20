import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import accuracy_score, roc_auc_score
import wfdb #pip install wfdb
import numpy as np
import os



# 1) wczytanie danych
df = pd.read_csv('merged_filtered.csv', parse_dates=['deathdate', 'admittime'])
num_cols = ['age_at_admit', 'bmi', 'systolic_bp', 'diastolic_bp']
cat_cols = ['gender', 'race']

# obróbka tabelaryczna
df_cat = pd.get_dummies(df[cat_cols], drop_first=False)
scaler = StandardScaler()
df_num = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

# 2) wczytanie listy plików ecg i filtracja
# record_list.csv zawiera: subject_id, study_id, filepath (ścieżka bez .hea/.dat), ecg_time
records = pd.read_csv('record_list.csv', parse_dates=['ecg_time'])
df = df.merge(records, on='subject_id', how='left')
df = df[df['ecg_time'].notna()]  # usuwamy bez ECG

# synchronizacja: tylko ecg_time >= admittime (jeśli admittime mamy) i <= deathdate
# filtrujemy, żeby ECG było po admittime
df = df[df['ecg_time'] >= df['admittime']]
# filtrujemy, aby dla zmarłych ECG było przed śmiercią
df = df[(df['deathdate'].isna()) | (df['ecg_time'] <= df['deathdate'])]
# obliczamy różnicę do deathdate tylko dla zmarłych
df['delta_to_death'] = (df['deathdate'] - df['ecg_time']).dt.total_seconds()
# wybór ECG
dead = df[df['deathdate'].notna()].sort_values(['subject_id', 'delta_to_death'])
dead_closest = dead.groupby('subject_id').first().reset_index()

alive = df[df['deathdate'].isna()].sort_values(['subject_id', 'ecg_time'])
alive_last = alive.groupby('subject_id').last().reset_index()
ecg_selected = pd.concat([dead_closest, alive_last], ignore_index=True)

# 3) Przetwarzanie EKG pojedynczo (oszczędzanie RAM-u) i zapis do .npy
output_dir = 'ecg_signals'
BASE_PATH = "/Users/mateuszlickindorf/Downloads/mimic-iv-ecg"
os.makedirs(output_dir, exist_ok=True)

records_to_save = []

for idx, row in ecg_selected.iterrows():
    subject_id = row['subject_id']
    ecg_time = row['ecg_time']
    relative_path = row['path']  # np. 'files/p1000/p10001725/s41420867/41420867'
    full_path = os.path.join(BASE_PATH, relative_path)

    try:
        rec = wfdb.rdrecord(full_path)
        sig = rec.p_signal[:5000, :].T
        print(f"[✓] Loaded {full_path}")
    except Exception as e:
        print(f"[!] Failed to load {full_path}: {e}")
        sig = np.zeros((12, 5000))

    # Zapisz sygnał do osobnego pliku .npy
    filename = f"{subject_id}_{idx}.npy"
    filepath = os.path.join(output_dir, filename)
    np.save(filepath, sig)

    # Dodaj metadane do listy
    records_to_save.append({
        'subject_id': subject_id,
        'ecg_time': ecg_time,
        'ecg_file': filepath
    })

# Zapisz listę ścieżek plików z sygnałami
df_ecg_files = pd.DataFrame(records_to_save)
df_ecg_files.to_csv('ecg_signals_list.csv', index=False)

# Zapisz do csv z odnośnikami do plików z sygnałami
df_ecg_files = pd.DataFrame(records_to_save)
df_ecg_files.to_csv('ecg_signals_list.csv', index=False)