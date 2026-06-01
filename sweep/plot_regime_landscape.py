import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
os.makedirs('plots', exist_ok=True)
m = pd.read_csv('master_table.csv')
REGIME_COLORS = {
    'PE_contraction':   '#d62728',  # red - contracts with salt
    'PA_swelling':      '#1f77b4',  # blue - expands with salt
    'non_monotonic':    '#9467bd',  # purple - complex response
    'salt_insensitive': '#7f7f7f',  # gray - no response
}
REGIME_ORDER = ['PE_contraction', 'PA_swelling', 'non_monotonic', 'salt_insensitive']

def pretty(regime):
    return regime.replace('_', ' ')

# ----- Plot 1: NCPR vs FCR colored by regime -----
fig, ax = plt.subplots(figsize=(8, 6.5))
for regime in REGIME_ORDER:
    sub = m[m['regime'] == regime]
    ax.scatter(sub['ncpr'], sub['fcr'],
               c=REGIME_COLORS[regime], s=30, alpha=0.7,
               edgecolors='none',
               label=f'{pretty(regime)} (n={len(sub)})')
ax.set_xlabel('NCPR (net charge per residue)', fontsize=12)
ax.set_ylabel('FCR (fraction charged)', fontsize=12)
ax.set_title('Salt-response regime in Pappu sequence space', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 0.5)
plt.tight_layout()
plt.savefig('plots/regime_landscape_ncpr_fcr.png', dpi=150, bbox_inches='tight')
plt.close()
print("Wrote plots/regime_landscape_ncpr_fcr.png")

# ----- Plot 2: SCD vs Length colored by regime -----
fig, ax = plt.subplots(figsize=(8, 6.5))
for regime in REGIME_ORDER:
    sub = m[m['regime'] == regime]
    ax.scatter(sub['scd'], sub['length'],
               c=REGIME_COLORS[regime], s=30, alpha=0.7,
               edgecolors='none',
               label=f'{pretty(regime)} (n={len(sub)})')
ax.set_xlabel('SCD (sequence charge decoration)', fontsize=12)
ax.set_ylabel('Length (residues)', fontsize=12)
ax.set_title('Salt-response regime: charge patterning vs chain length', fontsize=13)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('symlog', linthresh=1)
plt.tight_layout()
plt.savefig('plots/regime_landscape_scd_length.png', dpi=150, bbox_inches='tight')
plt.close()
print("Wrote plots/regime_landscape_scd_length.png")

# ----- Plot 3: Rg slope distribution by regime -----
fig, ax = plt.subplots(figsize=(8, 5))
for regime in REGIME_ORDER:
    sub = m[m['regime'] == regime]
    ax.hist(sub['Rg_slope'], bins=40, alpha=0.6,
            color=REGIME_COLORS[regime],
            label=f'{pretty(regime)} (n={len(sub)})')
ax.set_xlabel('Rg slope dR_g/d[salt] (nm/M)', fontsize=12)
ax.set_ylabel('Number of sequences', fontsize=12)
ax.set_title('Salt-response slope distribution by regime', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/slope_distribution_by_regime.png', dpi=150, bbox_inches='tight')
plt.close()
print("Wrote plots/slope_distribution_by_regime.png")

# ----- Plot 4: Representative Rg vs salt curves -----
SALTS_MM = [50, 100, 150, 300, 500]
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
for ax, regime in zip(axes, REGIME_ORDER):
    sub = m[m['regime'] == regime].sample(min(20, len(m[m['regime'] == regime])), random_state=42)
    for _, row in sub.iterrows():
        rgs = [row[f'Rg_{s}mM'] for s in SALTS_MM]
        ax.plot(SALTS_MM, rgs, color=REGIME_COLORS[regime], alpha=0.4, linewidth=1)

    # Plot mean curve
    mean_rgs = [m[m['regime'] == regime][f'Rg_{s}mM'].mean() for s in SALTS_MM]
    ax.plot(SALTS_MM, mean_rgs, color=REGIME_COLORS[regime], linewidth=3, marker='o',
            label='mean')

    ax.set_xscale('log')
    ax.set_xlabel('[salt] (mM)', fontsize=11)
    ax.set_ylabel('R_g (nm)', fontsize=11)
    ax.set_title(f'{pretty(regime)}\n(n={(m["regime"]==regime).sum()})', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(SALTS_MM)
    ax.set_xticklabels(SALTS_MM)
plt.tight_layout()
plt.savefig('plots/representative_curves_by_regime.png', dpi=150, bbox_inches='tight')
plt.close()
print("Wrote plots/representative_curves_by_regime.png")

print("\nAll plots done.")
