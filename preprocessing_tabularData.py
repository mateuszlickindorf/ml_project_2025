# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
#admissions - admittime, dischtime?, deathtime
#diagnoses - list of icd_code in order
#omr - BMI, blood presure (oldest)
#patients - gender, anchor age, anchor year

#real_age = anchor_age + (admittime.year - anchor_year)

patients_df = pd.read_csv('patients.csv')
admissions_df = pd.read_csv('admissions.csv', parse_dates=['admittime'])
omr_df = pd.read_csv('omr.csv')

# Znajdź tylko ostatnie przyjęcie każdego pacjenta
last_admissions = admissions_df.sort_values(['subject_id', 'admittime']).groupby('subject_id').tail(1)

# Znajdź najstarsze BMI i blood presure dla każdego pacjenta
relevant_features = ['BMI (kg/m2)', 'Blood Pressure', 'Weight (Lbs)', 'Height (Inches)']
omr_filtered = omr_df[omr_df['result_name'].isin(relevant_features)]
omr_filtered = omr_filtered.sort_values(by=['subject_id', 'result_name', 'chartdate'])

# Najstarsze pomiary każdego typu dla każdego pacjenta
omr_first = omr_filtered.groupby(['subject_id', 'result_name']).first().reset_index()

# Pivot: kolumny: bmi, bp, weight, height
omr_wide = omr_first.pivot(index='subject_id', columns='result_name', values='result_value').reset_index()
omr_wide.columns.name = None  # usuń multiindex

# Obliczanie BMI jeśli brakuje
missing_bmi_mask = (omr_wide['BMI (kg/m2)'].isna() & omr_wide['Weight (Lbs)'].notna() & omr_wide['Height (Inches)'].notna()) | omr_wide['BMI (kg/m2)'] > 50.0
omr_wide.loc[missing_bmi_mask, 'BMI (kg/m2)'] = (
    omr_wide.loc[missing_bmi_mask, 'Weight (Lbs)'] * 703 / (omr_wide.loc[missing_bmi_mask, 'Height (Inches)'] ** 2)
)

omr_wide = omr_wide.rename(columns={
    'BMI (kg/m2)': 'bmi',
    'Blood Pressure': 'blood_pressure'
})


merged = last_admissions.merge(patients_df, on='subject_id').head(100) #liczba rekordów na jaką się zdecydujemy
merged = merged.merge(omr_wide, on='subject_id')

# Oblicz wiek w momencie przyjęcia
merged['admit_year'] = merged['admittime'].dt.year
merged['age_at_admit'] = merged['anchor_age'] + (merged['admit_year'] - merged['anchor_year'])

merged['deathtime'] = pd.to_datetime(merged['deathtime']).dt.date
merged = merged.rename(columns={
    'deathtime': 'deathdate'
})

selected_columns = [
    'subject_id',
    'admittime',
    'deathdate',
    'race',
    'gender',
    'bmi',
    'blood_pressure',
    'age_at_admit'
]

merged_filtered = merged[selected_columns]
merged_filtered['admittime'] = merged_filtered['admittime'].dt.date
merged_filtered.to_csv("merged_filtered.csv", index=False)

print(merged_filtered[merged_filtered['deathdate'].notna()][['admittime', 'deathdate']])