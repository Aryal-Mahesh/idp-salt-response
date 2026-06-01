"""
For each sequence, fit Rg, Ree, and (Ree/Rg)^2 vs [salt].
- Linear fit: y = a + b * salt_M           (slope b is primary ML target)
- Quadratic fit: y = a + b * salt_M + c * salt_M^2  (curvature c flags non-monotonic)

Output: salt_response_fits.csv (one row per seq_id)
"""
import pandas as pd
import numpy as np
from scipy import stats

# Load long-form data
df = pd.read_csv('observables_long.csv')

# Drop seq_id with fewer than 4 valid salt points (defensive)
counts = df.groupby('seq_id').size()
bad_count = counts[counts < 4].index.tolist()
if bad_count:
    print(f"Excluding {len(bad_count)} seq_ids with <4 salt points")
    df = df[~df['seq_id'].isin(bad_count)]

# Observables to fit
OBSERVABLES = ['Rg', 'Ree', 'Ree_Rg_sq']

# Add nu only if reliable — we'll fit it per-sequence ignoring NaNs
OBSERVABLES_WITH_NU = OBSERVABLES + ['nu_masked']

results = []

for seq_id, group in df.groupby('seq_id'):
    group = group.sort_values('salt_mM')
    salts_M = group['salt_M'].values  # in M
    
    row = {'seq_id': seq_id, 'n_points': len(group)}
    
    for obs in OBSERVABLES_WITH_NU:
        y = group[obs].values
        mask = ~np.isnan(y)
        x = salts_M[mask]
        y = y[mask]
        n_valid = len(y)
        
        # Linear fit
        if n_valid >= 2:
            lr = stats.linregress(x, y)
            slope, intercept, r2 = lr.slope, lr.intercept, lr.rvalue**2
            
            # Mean predicted error from linear fit
            y_pred = intercept + slope * x
            residuals = y - y_pred
            rmse = float(np.sqrt(np.mean(residuals**2)))
        else:
            slope, intercept, r2, rmse = np.nan, np.nan, np.nan, np.nan
        
        # Quadratic fit (need >= 3 points)
        if n_valid >= 3:
            coeffs = np.polyfit(x, y, 2)  # [c, b, a] in y = a + b*x + c*x^2
            curvature = coeffs[0]  # c
            # R² of quadratic
            y_pred_q = np.polyval(coeffs, x)
            ss_res = float(np.sum((y - y_pred_q)**2))
            ss_tot = float(np.sum((y - np.mean(y))**2))
            r2_q = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            
            # Value at extrapolated 0 mM (intercept of quadratic = a)
            extrap_0 = coeffs[2]
        else:
            curvature, r2_q, extrap_0 = np.nan, np.nan, np.nan
        
        # Relative change from 50 to 500 mM (simple signal magnitude)
        try:
            y_50 = group[group['salt_mM'] == 50][obs].values[0]
            y_500 = group[group['salt_mM'] == 500][obs].values[0]
            rel_change = (y_500 - y_50) / y_50 if y_50 != 0 else np.nan
        except Exception:
            y_50, y_500, rel_change = np.nan, np.nan, np.nan
        
        prefix = obs
        row[f'{prefix}_slope']     = slope        # in nm/M for Rg/Ree
        row[f'{prefix}_intercept'] = intercept
        row[f'{prefix}_r2_lin']    = r2
        row[f'{prefix}_rmse_lin']  = rmse
        row[f'{prefix}_curvature'] = curvature
        row[f'{prefix}_r2_quad']   = r2_q
        row[f'{prefix}_at_50mM']   = y_50
        row[f'{prefix}_at_500mM']  = y_500
        row[f'{prefix}_rel_change']= rel_change
        row[f'{prefix}_n_valid']   = n_valid
    
    results.append(row)

out = pd.DataFrame(results)
out.to_csv('salt_response_fits.csv', index=False)

print(f"Wrote salt_response_fits.csv: {len(out)} sequences")
print()
print("--- Slope distribution: Rg vs [salt] (nm/M) ---")
print(out['Rg_slope'].describe().round(4))
print()
print("--- Slope distribution: Ree_Rg_sq vs [salt] ---")
print(out['Ree_Rg_sq_slope'].describe().round(4))
print()
print("--- Number of sequences with curvature suggesting non-monotonic Rg ---")
# Heuristic: curvature large relative to slope^2 / dynamic range
strong_curv = out[out['Rg_r2_quad'] - out['Rg_r2_lin'] > 0.05]
print(f"Sequences where quadratic fit improves R^2 by >0.05 over linear: {len(strong_curv)}")
print()
print("--- Most strongly contracting (most negative Rg slope) ---")
print(out.nsmallest(10, 'Rg_slope')[['seq_id', 'Rg_slope', 'Rg_rel_change', 'Rg_at_50mM', 'Rg_at_500mM']].to_string(index=False))
print()
print("--- Most strongly expanding ---")
print(out.nlargest(10, 'Rg_slope')[['seq_id', 'Rg_slope', 'Rg_rel_change', 'Rg_at_50mM', 'Rg_at_500mM']].to_string(index=False))
