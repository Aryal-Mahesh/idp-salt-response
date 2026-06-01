"""
make_fig4_shap.py
Combine the SHAP dependence (top-3 features) and per-regime importance into a
single two-panel publication figure: plots/fig4_shap.png

Reproduces the exact pipeline from shap_analysis.py (same features, same model,
same seed) so the SHAP values match the existing standalone plots.

Run from the sweep/ directory:
    python make_fig4_shap.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import shap
from xgboost import XGBRegressor

# Import the EXACT feature builder and seed used everywhere else in the project,
# so this figure cannot diverge from the trained model / reported numbers.
from physics_baseline import build_features, RNG_SEED

# ---- nicer display names for features (keys must match build_features columns) ----
PRETTY = {
    'scd_length':      r'$\mathrm{SCD}\times N$',
    'scd':             r'$\mathrm{SCD}$',
    'ncpr_sq_length':  r'$\mathrm{NCPR}^2\times N$',
    'fcr_length':      r'$\mathrm{FCR}\times N$',
    'kappa_fcr_sq':    r'$\kappa\,\mathrm{FCR}^2$',
    'inv_length':      r'$1/N$',
    'mean_hydropathy': 'mean hydropathy',
    'kappa':           r'$\kappa$',
}
REGIME_PRETTY = {
    'PA_swelling':      'PA\nswelling',
    'PE_contraction':   'PE\ncontraction',
    'non_monotonic':    'non-\nmonotonic',
    'salt_insensitive': 'salt-\ninsensitive',
}
REGIME_ORDER = ['PE_contraction', 'PA_swelling', 'non_monotonic', 'salt_insensitive']

# ---------- reproduce the pipeline ----------
df = pd.read_csv('master_table.csv')
X = build_features(df)
y = df['Rg_slope'].values
regimes = df['regime'].values

model = XGBRegressor(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85,
    random_state=RNG_SEED, verbosity=0, n_jobs=-1,
)
model.fit(X.values, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X.values)

mean_abs = np.abs(shap_values).mean(axis=0)
importance = (pd.DataFrame({'feature': X.columns, 'mean_abs_shap': mean_abs})
              .sort_values('mean_abs_shap', ascending=False))
top3 = importance.head(3)['feature'].tolist()

# per-regime mean |SHAP|, ordered features (desc importance) x ordered regimes
feat_order = importance['feature'].tolist()
present_regimes = [r for r in REGIME_ORDER if r in set(regimes)]
regime_mat = np.zeros((len(feat_order), len(present_regimes)))
for j, reg in enumerate(present_regimes):
    mask = regimes == reg
    col = np.abs(shap_values[mask]).mean(axis=0)
    col = pd.Series(col, index=X.columns)[feat_order].values
    regime_mat[:, j] = col

# ---------- figure ----------
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'font.family': 'sans-serif',
})

fig = plt.figure(figsize=(11, 8))
gs = GridSpec(2, 3, height_ratios=[1, 1.25], hspace=0.42, wspace=0.32,
              left=0.08, right=0.97, top=0.93, bottom=0.10)

# ----- Panel (a): dependence plots for top-3 -----
axes_a = [fig.add_subplot(gs[0, k]) for k in range(3)]
for k, (ax, feat) in enumerate(zip(axes_a, top3)):
    fidx = list(X.columns).index(feat)
    sc = ax.scatter(X[feat], shap_values[:, fidx], c=X[feat],
                    cmap='viridis', s=16, alpha=0.75, edgecolors='none')
    ax.axhline(0, color='0.3', ls='--', lw=0.8, alpha=0.6)
    ax.set_xlabel(PRETTY.get(feat, feat))
    if k == 0:
        ax.set_ylabel('SHAP value\n(contribution to $\\mathrm{d}R_g/\\mathrm{d}[\\mathrm{salt}]$)')
    ax.set_title(PRETTY.get(feat, feat))
    ax.grid(True, alpha=0.25)
axes_a[0].text(-0.30, 1.08, '(a)', transform=axes_a[0].transAxes,
               fontsize=14, fontweight='bold', va='bottom', ha='right')

# ----- Panel (b): per-regime heatmap -----
ax_b = fig.add_subplot(gs[1, :])
im = ax_b.imshow(regime_mat, aspect='auto', cmap='viridis')
ax_b.set_xticks(range(len(present_regimes)))
ax_b.set_xticklabels([REGIME_PRETTY.get(r, r) for r in present_regimes])
ax_b.set_yticks(range(len(feat_order)))
ax_b.set_yticklabels([PRETTY.get(f, f) for f in feat_order])
# annotate each cell
vmax = regime_mat.max()
for i in range(regime_mat.shape[0]):
    for j in range(regime_mat.shape[1]):
        val = regime_mat[i, j]
        ax_b.text(j, i, f'{val:.2f}', ha='center', va='center',
                  color='white' if val < 0.55 * vmax else 'black', fontsize=8)
cbar = fig.colorbar(im, ax=ax_b, fraction=0.025, pad=0.02)
cbar.set_label('Mean |SHAP value|  (nm/M)')
ax_b.set_title('Per-regime feature importance', pad=8)
ax_b.text(-0.085, 1.04, '(b)', transform=ax_b.transAxes,
          fontsize=14, fontweight='bold', va='bottom', ha='right')

fig.savefig('plots/fig4_shap.png', dpi=300, bbox_inches='tight')
print('Wrote plots/fig4_shap.png')
print('Top-3 features:', top3)
