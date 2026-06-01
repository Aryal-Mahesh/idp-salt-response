"""Build jobs.csv: one row per (sequence, salt) pair."""
import pandas as pd

SALTS = [0.05, 0.10, 0.15, 0.30, 0.50]

df = pd.read_csv('library.csv')
records = []
for _, row in df.iterrows():
    for salt in SALTS:
        records.append({'seq_id': row['seq_id'], 'salt': salt})

jobs = pd.DataFrame(records)
jobs.to_csv('jobs.csv', index=False)
print(f"Wrote {len(jobs)} jobs to jobs.csv")
print(f"Sequences: {jobs['seq_id'].nunique()}, salts per sequence: {jobs.groupby('seq_id').size().iloc[0]}")
