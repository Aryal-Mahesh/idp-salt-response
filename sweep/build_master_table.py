"""
Build the master analysis table: one row per sequence with all features + observables + regime.
This is the table ML will train on.
"""
import pandas as pd
import numpy as np

# Load all three tables
lib = pd.read_csv('../library/library.csv')
fits = pd.read_csv('salt_response_fits.csv')
obs_wide = pd.read_csv('observables_wide.csv')

print(f"Library:       {len(lib)} sequences with {len(lib.columns)} columns")
print(f"Fits:          {len(fits)} sequences with {len(fits.columns)} columns")
print(f"Observables:   {len(obs_wide)} sequences with {len(obs_wide.columns)} columns")

# Merge: features (from library) + raw observables at each salt (from obs_wide) + fits/regime (from fits)
master = lib.merge(obs_wide, on='seq_id', how='left')
master = master.merge(fits, on='seq_id', how='left', suffixes=('', '_fits_dup'))

# Drop any duplicate columns from the merge
dup_cols = [c for c in master.columns if c.endswith('_fits_dup')]
master = master.drop(columns=dup_cols)

# Sanity check
n_dropped = len(lib) - len(master.dropna(subset=['regime']))
if n_dropped > 0:
    print(f"\nWarning: {n_dropped} library sequences lack regime/fit data")

print(f"\nMaster table: {len(master)} sequences, {len(master.columns)} columns")
print()
print("--- Regime composition ---")
print(master['regime'].value_counts(dropna=False).to_string())
print()
print("--- Master columns (categorized) ---")
sequence_cols = [c for c in master.columns if c in ['seq_id', 'subset', 'sequence', 'length', 'source_name']]
feature_cols = [c for c in master.columns if c in ['fcr', 'ncpr', 'kappa', 'scd', 'shd', 'mean_hydropathy', 'mean_lambda_calvados', 'f_aro', 'f_pos', 'f_neg']]
obs_cols = [c for c in master.columns if any(c.startswith(p) for p in ['Rg_', 'Ree_', 'nu_'])]
fit_cols = [c for c in master.columns if c.endswith(('_slope', '_intercept', '_r2_lin', '_rmse_lin', '_curvature', '_r2_quad', '_at_50mM', '_at_500mM', '_rel_change', '_n_valid'))]
classification_cols = [c for c in master.columns if c in ['regime', 'monotonicity']]
other_cols = [c for c in master.columns if c not in sequence_cols + feature_cols + obs_cols + fit_cols + classification_cols]

print(f"  Sequence/metadata ({len(sequence_cols)}): {sequence_cols}")
print(f"  Features ({len(feature_cols)}): {feature_cols}")
print(f"  Observables at each salt ({len(obs_cols)}): {len(obs_cols)} cols (5 salts × multiple obs)")
print(f"  Salt-response fits ({len(fit_cols)}): {len(fit_cols)} cols")
print(f"  Classification ({len(classification_cols)}): {classification_cols}")
if other_cols:
    print(f"  Other ({len(other_cols)}): {other_cols}")

# Write
master.to_csv('master_table.csv', index=False)
print(f"\nWrote master_table.csv: {len(master)} rows × {len(master.columns)} columns")
