"""
Classify each sequence into a polymer-physics regime based on its
Rg-vs-salt curve.

4 regimes:
  - PE_contraction: monotonic decrease, large negative slope (polyelectrolyte)
  - PA_swelling:    monotonic increase, large positive slope (polyampholyte release)
  - non_monotonic:  significant curvature with sign change
  - salt_insensitive: small slope, no meaningful response

Output: salt_response_fits.csv (updated with regime + diagnostic columns)
"""
import pandas as pd
import numpy as np

fits = pd.read_csv('salt_response_fits.csv')
long_df = pd.read_csv('observables_long.csv')

# Classification thresholds (relative change from 50 to 500 mM)
# These can be tuned later; start with sensible defaults
THRESH_STRONG = 0.10   # |rel_change| > 10% → "strong" response
THRESH_WEAK = 0.03     # |rel_change| < 3% → "salt-insensitive"

def is_monotonic(rg_values):
    """Returns 'increasing', 'decreasing', or 'non-monotonic'."""
    diffs = np.diff(rg_values)
    if np.all(diffs >= -1e-3): return 'increasing'
    if np.all(diffs <= 1e-3): return 'decreasing'
    return 'non-monotonic'

# Build Rg curves per sequence to detect monotonicity
rg_curves = {}
for seq_id, group in long_df.groupby('seq_id'):
    sorted_g = group.sort_values('salt_mM')
    rg_curves[seq_id] = sorted_g['Rg'].values

def classify(row):
    seq_id = row['seq_id']
    rel = row['Rg_rel_change']
    
    if pd.isna(rel):
        return 'unclassified'
    
    rg_vals = rg_curves.get(seq_id)
    if rg_vals is None:
        return 'unclassified'
    
    monotonicity = is_monotonic(rg_vals)
    abs_rel = abs(rel)
    
    if abs_rel < THRESH_WEAK:
        return 'salt_insensitive'
    
    if monotonicity == 'non-monotonic' and abs_rel > THRESH_WEAK:
        return 'non_monotonic'
    
    if rel < -THRESH_WEAK:
        return 'PE_contraction'
    elif rel > THRESH_WEAK:
        return 'PA_swelling'
    
    return 'unclassified'

fits['regime'] = fits.apply(classify, axis=1)

# Also add monotonicity as its own column for diagnostics
fits['monotonicity'] = fits['seq_id'].map(
    lambda s: is_monotonic(rg_curves[s]) if s in rg_curves else 'unknown'
)

fits.to_csv('salt_response_fits.csv', index=False)

print(f"Updated salt_response_fits.csv with regime + monotonicity columns")
print()
print("--- Regime distribution ---")
print(fits['regime'].value_counts().to_string())
print()
print("--- Monotonicity distribution ---")
print(fits['monotonicity'].value_counts().to_string())
print()
print("--- Regime by subset ---")
fits['subset_prefix'] = fits['seq_id'].str.split('_').str[0]
print(fits.groupby(['subset_prefix', 'regime']).size().unstack(fill_value=0))
print()
print("--- Sample sequences by regime ---")
for regime in ['PE_contraction', 'PA_swelling', 'non_monotonic', 'salt_insensitive']:
    n = (fits['regime'] == regime).sum()
    print(f"\n[{regime}] n={n}")
    sample = fits[fits['regime'] == regime].head(3)[['seq_id', 'Rg_slope', 'Rg_rel_change']]
    print(sample.to_string(index=False))
