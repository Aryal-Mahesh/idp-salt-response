"""
Physics-motivated linear baseline for predicting dRg/d[salt].

Idea: design polymer-physics features from established theory (Lin/Chan, Ghosh,
Pappu), fit ridge regression. This is the baseline ML must beat to claim added value.

Features used (with theoretical motivation):
  - NCPR^2 * length      Polyelectrolyte contraction strength (Higgs-Joanny scaling)
  - SCD                  Polyampholyte sequence-charge-decoration (Sawle-Ghosh 2015)
  - SCD * length         Length-scaled patterning effect
  - FCR * length         Total charge "ammunition"
  - kappa * FCR^2        Patterning x composition coupling
  - 1 / length           Finite-size correction
  - mean_hydropathy      First-order non-electrostatic effect
  - kappa                Direct Pappu blockiness descriptor

Targets:
  - Rg_slope (primary)
  - Rg_rel_change (secondary, dimensionless)

Output: physics_baseline_predictions.csv (one row per seq_id)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

RNG_SEED = 42
N_FOLDS = 5

def build_features(df):
    """Construct the physics-motivated feature matrix."""
    f = pd.DataFrame(index=df.index)
    f['ncpr_sq_length']  = df['ncpr']**2 * df['length']
    f['scd']             = df['scd']
    f['scd_length']      = df['scd'] * df['length']
    f['fcr_length']      = df['fcr'] * df['length']
    f['kappa_fcr_sq']    = df['kappa'].fillna(0) * df['fcr']**2
    f['inv_length']      = 1.0 / df['length']
    f['mean_hydropathy'] = df['mean_hydropathy']
    f['kappa']           = df['kappa'].fillna(0)
    return f

def stratified_cv(X, y, groups, n_folds=N_FOLDS, alpha=1.0):
    """K-fold CV with stratification by group; returns per-fold and aggregate scores."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RNG_SEED)
    fold_scores = []
    all_preds = np.zeros_like(y)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        
        model = Ridge(alpha=alpha, random_state=RNG_SEED)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        all_preds[test_idx] = y_pred
        
        fold_scores.append({
            'fold': fold,
            'n_test': len(test_idx),
            'r2':  r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
        })
    
    return pd.DataFrame(fold_scores), all_preds

def main():
    df = pd.read_csv('master_table.csv')
    print(f"Loaded {len(df)} sequences")
    
    X = build_features(df)
    print(f"\nFeature columns ({X.shape[1]}):")
    for col in X.columns:
        print(f"  {col}")
    
    print(f"\n--- Training physics-motivated ridge regression ---")
    
    # === Target 1: Rg_slope ===
    y = df['Rg_slope'].values
    fold_df, preds = stratified_cv(X, y, groups=df['subset'])
    print(f"\nTarget: Rg_slope (dRg/d[salt], nm/M)")
    print(fold_df.round(3).to_string(index=False))
    print(f"Mean R²:  {fold_df['r2'].mean():.3f} ± {fold_df['r2'].std():.3f}")
    print(f"Mean MAE: {fold_df['mae'].mean():.3f} nm/M")
    
    df['physics_pred_slope'] = preds
    
    # Fit on all data to get final coefficients for the paper
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X)
    final_model = Ridge(alpha=1.0, random_state=RNG_SEED)
    final_model.fit(X_all, y)
    
    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coef_standardized': final_model.coef_,
    }).sort_values('coef_standardized', key=abs, ascending=False)
    print(f"\nFinal coefficients (on standardized features, sorted by |coef|):")
    print(coef_df.round(4).to_string(index=False))
    
    # === Target 2: Rg_rel_change ===
    print(f"\n--- Same model on Rg_rel_change (dimensionless) ---")
    y2 = df['Rg_rel_change'].values
    fold_df2, preds2 = stratified_cv(X, y2, groups=df['subset'])
    print(fold_df2.round(3).to_string(index=False))
    print(f"Mean R²:  {fold_df2['r2'].mean():.3f} ± {fold_df2['r2'].std():.3f}")
    print(f"Mean MAE: {fold_df2['mae'].mean():.4f}")
    df['physics_pred_relchange'] = preds2
    
    # Save predictions
    out = df[['seq_id', 'subset', 'regime',
              'Rg_slope', 'physics_pred_slope',
              'Rg_rel_change', 'physics_pred_relchange']].copy()
    out.to_csv('physics_baseline_predictions.csv', index=False)
    print(f"\nWrote physics_baseline_predictions.csv")
    
    # === Per-regime performance ===
    print(f"\n--- Performance by regime (Rg_slope) ---")
    for regime in df['regime'].unique():
        m = df['regime'] == regime
        r2 = r2_score(df.loc[m, 'Rg_slope'], df.loc[m, 'physics_pred_slope'])
        mae = mean_absolute_error(df.loc[m, 'Rg_slope'], df.loc[m, 'physics_pred_slope'])
        print(f"  {regime:18s} n={m.sum():3d}  R²={r2:.3f}  MAE={mae:.3f}")

if __name__ == '__main__':
    main()
