"""
Hero figure: predicted vs actual Rg slope, comparing ridge baseline to XGBoost.
Both panels show in-distribution (random CV) predictions colored by regime.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import os

os.makedirs('plots', exist_ok=True)

# Load both prediction sets
ml = pd.read_csv('ml_predictions.csv')
print(f"Loaded {len(ml)} predictions")

REGIME_COLORS = {
    'PE_contraction':   '#d62728',
    'PA_swelling':      '#1f77b4',
    'non_monotonic':    '#9467bd',
    'salt_insensitive': '#7f7f7f',
}
REGIME_ORDER = ['PE_contraction', 'PA_swelling', 'non_monotonic', 'salt_insensitive']

def pretty(regime):
    return regime.replace('_', ' ')

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, (label, pred_col) in zip(axes, [
    ('Physics-motivated Ridge', 'ridge_pred_slope'),
    ('XGBoost (nonlinear)', 'xgb_pred_slope'),
]):
    y_true = ml['Rg_slope']
    y_pred = ml[pred_col]
    
    # Per-regime scatter
    for regime in REGIME_ORDER:
        m = ml['regime'] == regime
        ax.scatter(y_true[m], y_pred[m],
                   c=REGIME_COLORS[regime], s=30, alpha=0.7, edgecolors='none',
                   label=f'{pretty(regime)} (n={m.sum()})')
    
    # y=x reference
    lo, hi = -5, 6
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label='y = x')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    
    # Metrics box
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ax.text(0.04, 0.96,
            f'R² = {r2:.3f}\nMAE = {mae:.2f} nm/M',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('CALVADOS Rg slope (nm/M)', fontsize=12)
    ax.set_ylabel('Predicted Rg slope (nm/M)', fontsize=12)
    ax.set_title(label, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

# Single legend at the bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles[:4], labels[:4], loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, -0.02), fontsize=10)

fig.suptitle('Salt-response slope prediction: physics baseline vs nonlinear ML', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(bottom=0.13)
plt.savefig('plots/pred_vs_actual_hero.png', dpi=150, bbox_inches='tight')
plt.close()
print("Wrote plots/pred_vs_actual_hero.png")

# Also a per-regime residual plot
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, regime in zip(axes, REGIME_ORDER):
    m = ml['regime'] == regime
    resid_ridge = ml.loc[m, 'Rg_slope'] - ml.loc[m, 'ridge_pred_slope']
    resid_xgb = ml.loc[m, 'Rg_slope'] - ml.loc[m, 'xgb_pred_slope']
    ax.hist(resid_ridge, bins=20, alpha=0.5, color='gray', label='Ridge', edgecolor='none')
    ax.hist(resid_xgb, bins=20, alpha=0.7, color=REGIME_COLORS[regime], label='XGBoost', edgecolor='none')
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Residual (true − predicted, nm/M)', fontsize=10)
    ax.set_title(f'{pretty(regime)}\n(n={m.sum()})', fontsize=11)
    if ax is axes[0]:
        ax.set_ylabel('Count', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle('Per-regime residual distributions: ridge vs XGBoost', fontsize=13)
plt.tight_layout()
plt.savefig('plots/residuals_per_regime.png', dpi=150, bbox_inches='tight')
plt.close()
print("Wrote plots/residuals_per_regime.png")