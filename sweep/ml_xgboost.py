"""
XGBoost regressor on Rg_slope using the same physics-motivated features.
Evaluated head-to-head with the ridge baseline on:
  1. Random 5-fold CV (charitable / interpolation regime)
  2. Leave-one-subset-out CV (stress test / generalization regime)
  3. Per-regime breakdown

Success criterion: improve worst-regime R² without crashing best ones.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from physics_baseline import build_features

RNG_SEED = 42
N_FOLDS = 5

def make_xgb():
    return XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RNG_SEED,
        verbosity=0,
        n_jobs=-1,
    )

def make_ridge():
    return Ridge(alpha=1.0, random_state=RNG_SEED)

def random_kfold(X, y, model_fn, n_folds=N_FOLDS, scale=False):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RNG_SEED)
    preds = np.zeros_like(y, dtype=float)
    fold_r2 = []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx].values, X.iloc[test_idx].values
        y_tr, y_te = y[train_idx], y[test_idx]
        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
        m = model_fn()
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        preds[test_idx] = p
        fold_r2.append(r2_score(y_te, p))
    return np.array(fold_r2), preds

def leave_one_subset_out(X, y, subsets, model_fn, scale=False):
    out = []
    preds_all = np.full_like(y, np.nan, dtype=float)
    for held in np.unique(subsets):
        train_mask = subsets != held
        test_mask = ~train_mask
        X_tr, X_te = X[train_mask].values, X[test_mask].values
        y_tr, y_te = y[train_mask], y[test_mask]
        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
        m = model_fn()
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        preds_all[test_mask] = p
        out.append({
            'held_out': held,
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
            'r2': r2_score(y_te, p),
            'mae': mean_absolute_error(y_te, p),
        })
    return pd.DataFrame(out), preds_all

def per_regime_breakdown(y_true, y_pred, regimes):
    out = []
    for r in np.unique(regimes):
        m = regimes == r
        out.append({
            'regime': r,
            'n': int(m.sum()),
            'r2': r2_score(y_true[m], y_pred[m]),
            'mae': mean_absolute_error(y_true[m], y_pred[m]),
        })
    return pd.DataFrame(out)

def main():
    df = pd.read_csv('master_table.csv')
    X = build_features(df)
    y = df['Rg_slope'].values
    subsets = df['subset'].values
    regimes = df['regime'].values
    
    print(f"Loaded {len(df)} sequences, {X.shape[1]} features")
    
    # === Random 5-fold CV ===
    print("\n=== RANDOM 5-FOLD CV ===")
    print("(interpolation regime — charitable)")
    
    r2_ridge, preds_ridge = random_kfold(X, y, make_ridge, scale=True)
    r2_xgb, preds_xgb = random_kfold(X, y, make_xgb, scale=False)
    
    print(f"  Ridge:   R² = {r2_ridge.mean():.3f} ± {r2_ridge.std():.3f}")
    print(f"  XGBoost: R² = {r2_xgb.mean():.3f} ± {r2_xgb.std():.3f}")
    
    print("\n  Per-regime (random CV):")
    pr_ridge = per_regime_breakdown(y, preds_ridge, regimes)
    pr_xgb = per_regime_breakdown(y, preds_xgb, regimes)
    cmp = pr_ridge.merge(pr_xgb, on='regime', suffixes=('_ridge', '_xgb'))
    cmp = cmp[['regime', 'n_ridge', 'r2_ridge', 'mae_ridge', 'r2_xgb', 'mae_xgb']]
    cmp.columns = ['regime', 'n', 'R²_ridge', 'MAE_ridge', 'R²_xgb', 'MAE_xgb']
    print(cmp.round(3).to_string(index=False))
    
    # === Leave-one-subset-out CV ===
    print("\n=== LEAVE-ONE-SUBSET-OUT CV ===")
    print("(generalization stress test)")
    
    loso_ridge, preds_ridge_loso = leave_one_subset_out(X, y, subsets, make_ridge, scale=True)
    loso_xgb, preds_xgb_loso = leave_one_subset_out(X, y, subsets, make_xgb, scale=False)
    
    cmp2 = loso_ridge.merge(loso_xgb, on=['held_out', 'n_train', 'n_test'], suffixes=('_ridge', '_xgb'))
    print(cmp2.round(3).to_string(index=False))
    
    print(f"\n  Mean R² (LOSO):")
    print(f"    Ridge:   {loso_ridge['r2'].mean():.3f}")
    print(f"    XGBoost: {loso_xgb['r2'].mean():.3f}")
    
    # === Verdict ===
    print("\n=== VERDICT ===")
    
    # Did worst regime improve?
    worst_ridge_random = pr_ridge['r2'].min()
    worst_xgb_random   = pr_xgb['r2'].min()
    print(f"  Worst-regime R² (random CV): Ridge={worst_ridge_random:.3f}, XGBoost={worst_xgb_random:.3f}")
    
    # Did best regime drop?
    best_ridge_random = pr_ridge['r2'].max()
    best_xgb_random   = pr_xgb['r2'].max()
    print(f"  Best-regime R²  (random CV): Ridge={best_ridge_random:.3f}, XGBoost={best_xgb_random:.3f}")
    
    if worst_xgb_random > worst_ridge_random + 0.05 and best_xgb_random > best_ridge_random - 0.05:
        print("  → XGBoost improves uniform per-regime performance. Real win.")
    elif r2_xgb.mean() > r2_ridge.mean() + 0.03:
        print("  → XGBoost lifts global R² but not uniformly. Possible overfit or just nonlinear gains.")
    else:
        print("  → XGBoost doesn't meaningfully beat ridge. Physics features carry the signal.")
    
    # Save predictions for later analysis
    df['ridge_pred_slope'] = preds_ridge
    df['xgb_pred_slope'] = preds_xgb
    df['ridge_pred_slope_loso'] = preds_ridge_loso
    df['xgb_pred_slope_loso'] = preds_xgb_loso
    df.to_csv('ml_predictions.csv', index=False)
    print("\nWrote ml_predictions.csv")

if __name__ == '__main__':
    main()
