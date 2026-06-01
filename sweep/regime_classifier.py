"""Regime classifier — verbose version with thread limits."""
import os
# Limit threads BEFORE importing anything else to avoid login-node contention
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import sys
print("Starting imports...", flush=True)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from xgboost import XGBClassifier
print("Imports done.", flush=True)

from physics_baseline import build_features

RNG_SEED = 42

def pretty(regime):
    return regime.replace('_', ' ')

print("Loading master_table...", flush=True)
df = pd.read_csv('master_table.csv')
X = build_features(df)
le = LabelEncoder()
y = le.fit_transform(df['regime'].values)
class_names = list(le.classes_)
class_names_pretty = [pretty(c) for c in class_names]
print(f"  Classes: {class_names}", flush=True)
for i, c in enumerate(class_names):
    print(f"  {c}: {(y == i).sum()}", flush=True)

# === Logistic regression (random CV) ===
print("\nLogistic regression — random CV...", flush=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
preds_log = np.zeros_like(y)
fold_acc_log = []
for fold, (tr, te) in enumerate(skf.split(X, y)):
    print(f"  fold {fold+1}/5...", flush=True)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X.iloc[tr].values)
    X_te = sc.transform(X.iloc[te].values)
    m = LogisticRegression(max_iter=2000, random_state=RNG_SEED)
    m.fit(X_tr, y[tr])
    preds_log[te] = m.predict(X_te)
    fold_acc_log.append(accuracy_score(y[te], preds_log[te]))
print(f"  Logistic mean accuracy: {np.mean(fold_acc_log):.3f}", flush=True)

# === XGBoost classifier (random CV) ===
print("\nXGBoost — random CV...", flush=True)
preds_xgb = np.zeros_like(y)
fold_acc_xgb = []
for fold, (tr, te) in enumerate(skf.split(X, y)):
    print(f"  fold {fold+1}/5 — fitting...", flush=True)
    m = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.85, colsample_bytree=0.85,
        random_state=RNG_SEED, verbosity=0,
        n_jobs=2,  # explicit limit
        objective='multi:softprob',
        num_class=len(class_names),
    )
    m.fit(X.iloc[tr].values, y[tr])
    preds_xgb[te] = m.predict(X.iloc[te].values)
    fold_acc_xgb.append(accuracy_score(y[te], preds_xgb[te]))
    print(f"  fold {fold+1}/5 — done (acc={fold_acc_xgb[-1]:.3f})", flush=True)
print(f"  XGBoost mean accuracy: {np.mean(fold_acc_xgb):.3f}", flush=True)

# === Reports ===
print("\n--- XGBoost classification report ---", flush=True)
print(classification_report(y, preds_xgb, target_names=class_names, zero_division=0))

cm = confusion_matrix(y, preds_xgb)
print("Confusion matrix:")
print(pd.DataFrame(cm, index=class_names, columns=class_names))

# === Save ===
df_out = df[['seq_id', 'subset', 'regime']].copy()
df_out['regime_pred_logistic'] = le.inverse_transform(preds_log)
df_out['regime_pred_xgb'] = le.inverse_transform(preds_xgb)
df_out.to_csv('regime_classifier_predictions.csv', index=False)
print("\nWrote regime_classifier_predictions.csv", flush=True)

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names_pretty, rotation=30, ha='right')
ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names_pretty)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'XGBoost confusion matrix (acc={np.mean(fold_acc_xgb):.3f})')
for i in range(len(class_names)):
    for j in range(len(class_names)):
        col = 'white' if cm[i,j] > cm.max()/2 else 'black'
        ax.text(j, i, str(cm[i,j]), ha='center', va='center', color=col, fontsize=11)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('plots/regime_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Wrote plots/regime_confusion_matrix.png", flush=True)

print("\nDONE.", flush=True)
