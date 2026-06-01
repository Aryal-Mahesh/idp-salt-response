"""
Walk all data/<sysname>/ directories, load conf_prop.csv, build the
per-sequence salt-response table.

Output: observables_long.csv (one row per (seq_id, salt) pair)
         observables_wide.csv (one row per seq_id, columns for each salt)
"""
import os
import re
import pandas as pd
import numpy as np

DATA_DIR = 'data'
EXPECTED_SALTS = [50, 100, 150, 300, 500]   # in mM

# Sysname pattern: <seq_id>_salt<NNN>mM where seq_id may contain underscores
SYSNAME_RE = re.compile(r'^(.+)_salt(\d+)mM$')

records = []
n_loaded = 0
n_missing_csv = 0
n_parse_fail = 0

print("Walking data directories...")
for sysname in sorted(os.listdir(DATA_DIR)):
    sysname_dir = os.path.join(DATA_DIR, sysname)
    if not os.path.isdir(sysname_dir):
        continue
    
    m = SYSNAME_RE.match(sysname)
    if not m:
        print(f"  Skipping (cannot parse): {sysname}")
        n_parse_fail += 1
        continue
    seq_id = m.group(1)
    salt_mM = int(m.group(2))
    salt_M = salt_mM / 1000.0
    
    csv_path = os.path.join(sysname_dir, 'conf_prop.csv')
    if not os.path.exists(csv_path):
        n_missing_csv += 1
        continue
    
    try:
        df = pd.read_csv(csv_path, index_col=0)
        # df has rows Rg, Ree, nu; columns: value, error
        rec = {
            'seq_id': seq_id,
            'salt_mM': salt_mM,
            'salt_M': salt_M,
            'Rg': df.loc['Rg', 'value'],
            'Rg_err': df.loc['Rg', 'error'],
            'Ree': df.loc['Ree', 'value'],
            'Ree_err': df.loc['Ree', 'error'],
            'nu': df.loc['nu', 'value'],
            'nu_err': df.loc['nu', 'error'],
        }
        records.append(rec)
        n_loaded += 1
    except Exception as e:
        print(f"  Failed to load {csv_path}: {e}")

print(f"\nLoaded {n_loaded} observable records")
print(f"Missing CSV: {n_missing_csv}")
print(f"Parse failures: {n_parse_fail}")

long_df = pd.DataFrame(records)
long_df = long_df.sort_values(['seq_id', 'salt_mM']).reset_index(drop=True)
long_df.to_csv('observables_long.csv', index=False)
print(f"\nWrote observables_long.csv ({len(long_df)} rows)")

# Pivot to wide format: one row per seq_id, columns for each salt point
print("\nPivoting to wide format...")
wide_pieces = []
for obs in ['Rg', 'Ree', 'nu']:
    pivot = long_df.pivot(index='seq_id', columns='salt_mM', values=obs)
    pivot.columns = [f'{obs}_{int(c)}mM' for c in pivot.columns]
    wide_pieces.append(pivot)
    # Also pivot errors
    pivot_err = long_df.pivot(index='seq_id', columns='salt_mM', values=f'{obs}_err')
    pivot_err.columns = [f'{obs}_{int(c)}mM_err' for c in pivot_err.columns]
    wide_pieces.append(pivot_err)

wide_df = pd.concat(wide_pieces, axis=1).reset_index()
wide_df.to_csv('observables_wide.csv', index=False)
print(f"Wrote observables_wide.csv ({len(wide_df)} sequences)")

# Quality check: which sequences are missing any salt points?
incomplete = []
for seq_id in wide_df['seq_id']:
    row = wide_df[wide_df['seq_id'] == seq_id].iloc[0]
    missing_salts = [s for s in EXPECTED_SALTS if pd.isna(row[f'Rg_{s}mM'])]
    if missing_salts:
        incomplete.append((seq_id, missing_salts))

if incomplete:
    print(f"\n{len(incomplete)} sequences with incomplete salt coverage:")
    for seq_id, missing_salts in incomplete[:10]:
        print(f"  {seq_id}: missing {missing_salts}")
    if len(incomplete) > 10:
        print(f"  ... and {len(incomplete) - 10} more")
else:
    print(f"\nAll sequences have complete 5-salt coverage ✓")

# Quick sanity stats
print("\n--- Quick stats ---")
print(f"Sequences in wide table: {len(wide_df)}")
print(f"Rg range across all salts: {long_df['Rg'].min():.2f} to {long_df['Rg'].max():.2f} nm")
print(f"nu range: {long_df['nu'].min():.3f} to {long_df['nu'].max():.3f}")
