"""
Coverage plots for the sequence library.
Three figures:
  1. (NCPR, FCR) - Pappu sequence space; shows polyampholyte/polyelectrolyte regimes
  2. (SCD, length) - charge patterning by chain size
  3. (kappa, FCR) - patterning axis at fixed composition (synthetic-only insight)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

os.makedirs('plots', exist_ok=True)
df = pd.read_csv('library.csv')
print(f"Loaded {len(df)} sequences")

# Color/marker by subset
SUBSET_STYLE = {
    'kappa_variant':     dict(color='#1f77b4', marker='o', label='κ-variants (synthetic)'),
    'ncpr_series':       dict(color='#ff7f0e', marker='s', label='NCPR series (synthetic)'),
    'idrome_stratified': dict(color='#2ca02c', marker='^', label='IDRome stratified'),
    'idrome_lowfcr':     dict(color='#d62728', marker='D', label='IDRome low-FCR'),
}

def scatter_by_subset(ax, df, x_col, y_col, alpha=0.7, size=30):
    for subset, style in SUBSET_STYLE.items():
        sub = df[df['subset'] == subset]
        ax.scatter(sub[x_col], sub[y_col],
                   c=style['color'], marker=style['marker'],
                   s=size, alpha=alpha, edgecolors='none',
                   label=f"{style['label']} (n={len(sub)})")

# ----- Figure 1: NCPR vs FCR -----
fig, ax = plt.subplots(figsize=(7, 6))
scatter_by_subset(ax, df, 'ncpr', 'fcr')

# Add Pappu regime boundaries
# IDPs sit roughly: weak polyampholytes near origin; strong polyelectrolytes |NCPR| > 0.25
ax.axvspan(-0.25, 0.25, alpha=0.1, color='gray', zorder=0)
ax.text(0, 0.62, 'polyampholyte\nregion', ha='center', va='bottom', fontsize=9, color='gray')
ax.text(0.35, 0.62, 'polyelectrolyte\n(positive)', ha='center', va='bottom', fontsize=9, color='gray')
ax.text(-0.35, 0.62, 'polyelectrolyte\n(negative)', ha='center', va='bottom', fontsize=9, color='gray')

ax.set_xlabel('NCPR (net charge per residue)', fontsize=11)
ax.set_ylabel('FCR (fraction charged)', fontsize=11)
ax.set_title('Library coverage in Pappu sequence space', fontsize=12)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(0, 0.7)
plt.tight_layout()
plt.savefig('plots/01_ncpr_fcr_coverage.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved plots/01_ncpr_fcr_coverage.png")

# ----- Figure 2: SCD vs length -----
fig, ax = plt.subplots(figsize=(7, 6))
scatter_by_subset(ax, df, 'scd', 'length')
ax.set_xlabel('SCD (sequence charge decoration)', fontsize=11)
ax.set_ylabel('Length (residues)', fontsize=11)
ax.set_title('Library coverage: charge patterning vs chain length', fontsize=12)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xscale('symlog', linthresh=1)
plt.tight_layout()
plt.savefig('plots/02_scd_length_coverage.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved plots/02_scd_length_coverage.png")

# ----- Figure 3: kappa vs FCR (synthetic-only focus on patterning axis) -----
fig, ax = plt.subplots(figsize=(7, 6))
df_with_kappa = df.dropna(subset=['kappa'])
scatter_by_subset(ax, df_with_kappa, 'kappa', 'fcr')
ax.set_xlabel('κ (Das-Pappu charge patterning)', fontsize=11)
ax.set_ylabel('FCR (fraction charged)', fontsize=11)
ax.set_title('Charge patterning coverage (sequences with defined κ)', fontsize=12)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/03_kappa_fcr_coverage.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved plots/03_kappa_fcr_coverage.png")

# ----- Summary -----
print("\nLibrary coverage summary:")
print(f"  NCPR: {df['ncpr'].min():.2f} to {df['ncpr'].max():.2f}")
print(f"  FCR:  {df['fcr'].min():.2f} to {df['fcr'].max():.2f}")
print(f"  SCD:  {df['scd'].min():.1f} to {df['scd'].max():.1f}")
print(f"  Length: {df['length'].min()} to {df['length'].max()}")
print(f"  Sequences with defined kappa: {df['kappa'].notna().sum()}/{len(df)}")
