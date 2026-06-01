"""Audit each wave separately to track progress unambiguously."""
import pandas as pd
import os

jobs = pd.read_csv('jobs.csv')

def is_done(row):
    sysname = f"{row['seq_id']}_salt{int(row['salt']*1000):03d}mM"
    return os.path.exists(f'data/{sysname}/conf_prop.csv')

WAVES = [
    ('Wave 1', 0, 900),
    ('Wave 2', 900, 1800),
    ('Wave 3', 1800, 2555),
]

print(f"{'Wave':<10} {'Range':<14} {'Done':<10} {'Missing':<10} {'%':<6}")
print('-' * 50)

total_done = 0
for name, start, end in WAVES:
    chunk = jobs.iloc[start:end].copy()
    chunk['done'] = chunk.apply(is_done, axis=1)
    n_done = chunk['done'].sum()
    n_total = len(chunk)
    n_missing = n_total - n_done
    pct = 100 * n_done / n_total
    total_done += n_done
    print(f"{name:<10} tasks {start+1:<4}-{end:<4} {n_done:>4}/{n_total:<4} {n_missing:<10} {pct:>5.1f}")

print('-' * 50)
total = len(jobs)
print(f"{'Total':<10} {'':<14} {total_done:>4}/{total:<4} {total-total_done:<10} {100*total_done/total:>5.1f}")
