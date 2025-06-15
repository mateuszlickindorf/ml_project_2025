import os
print(os.path.exists("/Users/mateuszlickindorf/Downloads/mimic-iv-ecg/files/p1000/p10000032/s40689238/40689238.dat"))
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # dane
# df_tab = pd.read_csv('merged_filtered.csv', parse_dates=['deathdate', 'admittime'])
# df_ecg = pd.read_csv('ecg_signals_list.csv', parse_dates=['ecg_time'])

# # merge tabular + ecg
# df = pd.merge(df_tab, df_ecg[['subject_id', 'ecg_time', 'ecg_file']], on='subject_id', how='inner')

# # dni od badania do zgonu
# df['time_to_death_days'] = (df['deathdate'] - df['ecg_time']).dt.total_seconds() / 86400

# # tylko zmarli w ciągu 365 dni
# df_filtered = df[(~df['time_to_death_days'].isna()) & (df['time_to_death_days'] <= 365)]

# # histogram
# plt.figure(figsize=(10, 6))
# plt.hist(df_filtered['time_to_death_days'], bins=30, edgecolor='black')
# plt.title('Histogram dni do zgonu (maksymalnie 365 dni)')
# plt.xlabel('Dni do zgonu')
# plt.ylabel('Liczba pacjentów')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # na potrzeby analizy usuwamy zmarłych lub żyjących dłużej niż 365 dni
# df_filtered = df[(~df['time_to_death_days'].isna()) & (df['time_to_death_days'] <= 365)]

# # ilość klas
# n_death_classes = 3

# # kwantyle dla dni do zgonu
# quantiles = df_filtered['time_to_death_days'].quantile(q=[i / n_death_classes for i in range(1, n_death_classes)]).values

# # przypisanie klas
# def quantile_class(days):
#     if np.isnan(days) or days > 365:
#         return 0  # alive or late
#     for i, q in enumerate(quantiles):
#         if days <= q:
#             return i + 1
#     return n_death_classes

# df['death_class'] = df['time_to_death_days'].apply(quantile_class)

# # statystyki klas
# print(df['death_class'].value_counts().sort_index())

# # progi klas na podstawie kwantyli
# quantile_cutoffs = df_filtered['time_to_death_days'].quantile(
#     q=[i / n_death_classes for i in range(1, n_death_classes)]
# ).values
# print(f"Quantile-based death class thresholds for {n_death_classes} classes:")
# for i, q in enumerate(quantile_cutoffs):
#     if i == 0:
#         print(f"Class 1: 0 < days ≤ {q:.1f}")
#     else:
#         print(f"Class {i+1}: > {quantile_cutoffs[i-1]:.1f} days ≤ {q:.1f}")
# print(f"Class {n_death_classes}: > {quantile_cutoffs[-1]:.1f} days ≤ 365")