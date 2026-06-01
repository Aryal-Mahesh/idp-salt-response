"""
SHAP analysis on the XGBoost model.

Produces:
  - Global feature importance (mean |SHAP|)
  - Feature importance bar plot
  - SHAP summary "beeswarm" plot (per-sample contributions)
  - SHAP dependence plots for the top 3 features (shows how feature value -> SHAP value)
  - Per-regime SHAP importance (which features matter most in each regime)

Output: shap_feature_importance.csv + 4 PNG plots in plots/
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
import os

from physics_baseline import build_features

os.makedirs('plots', exist_ok=True)
RNG_SEED = 42

# ----- Load and prep -----
df = pd.read_csv('master_table.csv')
X = build_features(df)
y = df['Rg_slope'].values
regimes = df['regime'].values

# Train XGBoost on full data (for interpretation, not for evaluation)
model = XGBRegressor(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85,
    random_state=RNG_SEED, verbosity=0, n_jobs=-1,
)
model.fit(X.values, y)
print(f"Trained XGBoost on {len(X)} sequences x {X.shape[1]} features")

# ----- SHAP values -----
print("Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X.values)
print(f"SHAP matrix shape: {shap_values.shape}")

# ----- Global feature importance (mean |SHAP|) -----
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance = pd.DataFrame({
    'feature': X.columns,
    'mean_abs_shap': mean_abs_shap,
}).sort_values('mean_abs_shap', ascending=False)
importance.to_csv('shap_feature_importance.csv', index=False)
print("\n=== Global feature importance ===")
print(importance.round(4).to_string(index=False))

# ----- Bar plot of importance -----
fig, ax = plt.subplots(figsize=(8, 5))
imp_sorted = importance.iloc[::-1]
ax.barh(imp_sorted['feature'], imp_sorted['mean_abs_shap'], color='#1f77b4')
ax.set_xlabel('Mean |SHAP value| (nm/M)', fontsize=11)
ax.set_title('XGBoost feature importance (SHAP)', fontsize=12)
plt.tight_layout()
plt.savefig('plots/shap_importance_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Wrote plots/shap_importance_bar.png")

# ----- Beeswarm plot (SHAP's signature plot) -----
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X, show=False, max_display=8)
plt.tight_layout()
plt.savefig('plots/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Wrote plots/shap_beeswarm.png")

# ----- Dependence plots for the top 3 features -----
top3 = importance.head(3)['feature'].tolist()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, feat in zip(axes, top3):
    feat_idx = list(X.columns).index(feat)
    ax.scatter(X[feat], shap_values[:, feat_idx],
               c=X[feat], cmap='viridis', s=20, alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', alpha=0.4)
    ax.set_xlabel(feat, fontsize=11)
    ax.set_ylabel(f'SHAP value (contribution to slope)', fontsize=10)
    ax.set_title(f'{feat}', fontsize=11)
    ax.grid(True, alpha=0.3)
fig.suptitle('SHAP dependence: how feature value drives prediction', fontsize=13)
plt.tight_layout()
plt.savefig('plots/shap_dependence_top3.png', dpi=150, bbox_inches='tight')
plt.close()
print("Wrote plots/shap_dependence_top3.png")

# ----- Per-regime importance -----
print("\n=== Per-regime mean |SHAP| ===")
regime_importance = {}
for regime in np.unique(regimes):
    mask = regimes == regime
    regime_importance[regime] = np.abs(shap_values[mask]).mean(axis=0)

regime_df = pd.DataFrame(regime_importance, index=X.columns)
print(regime_df.round(3))

# Heatmap of per-regime importance
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(regime_df.values, aspect='auto', cmap='viridis')
ax.set_xticks(range(len(regime_df.columns)))
ax.set_xticklabels(regime_df.columns, rotation=20, ha='right')
ax.set_yticks(range(len(regime_df.index)))
ax.set_yticklabels(regime_df.index)
plt.colorbar(im, ax=ax, label='Mean |SHAP value|')
ax.set_title('Feature importance per regime', fontsize=12)
plt.tight_layout()
plt.savefig('plots/shap_per_regime.png', dpi=150, bbox_inches='tight')
plt.close()
print("Wrote plots/shap_per_regime.png")

print("\nDone.")
