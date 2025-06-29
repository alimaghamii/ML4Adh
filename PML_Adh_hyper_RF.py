"""
rf_only_hyperparam_and_learning_curve.py
"""

import os, time, warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection   import train_test_split, RandomizedSearchCV
from sklearn.ensemble          import RandomForestRegressor
from sklearn.metrics           import mean_squared_error

# ────────────────── project-specific helpers (keep as-is) ──────────────────
from utilities import (
    process_columns, file_path_PARPML_6inputs,  # CSV path
    save_dir_PARPML_6inputs                     # output directory
)

rand_st = 40
os.makedirs(save_dir_PARPML_6inputs, exist_ok=True)

# ──────────────────────────────── data ──────────────────────────────────────
df  = process_columns(pd.read_csv(file_path_PARPML_6inputs))
X   = df[['runload', 'alphaload', 'mu', 'k', 'n', 'PB']]
y   = df['outputAmplification']

# 20 % test, then 15 % of remaining for validation
X_tr_full, X_test, y_tr_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=rand_st)
X_train, X_val, y_train, y_val = train_test_split(
    X_tr_full, y_tr_full, test_size=0.15, random_state=rand_st)

# ───────────────── Hyper-parameter search (Randomized) ─────────────────────
param_dist = {
    'n_estimators'     : np.arange(100, 401, 50),   # 100-400
    'max_depth'        : [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf' : [1, 2, 4, 8],
    'max_features'     : ['auto', 'sqrt', 0.5]
}

search = RandomizedSearchCV(
    estimator  = RandomForestRegressor(random_state=rand_st),
    param_distributions = param_dist,
    n_iter     = 30,                # 30 random combos
    cv         = 5,
    scoring    = 'neg_mean_squared_error',
    n_jobs     = -1,
    random_state = rand_st,
    verbose    = 1
)

print("⏳  Running hyper-parameter search …")
tic = time.perf_counter()
search.fit(X_train, y_train)
toc = time.perf_counter()
print(f"✅  Done in {toc - tic:.1f} s Best params → {search.best_params_}")

best_params = search.best_params_

# ─────────────────── incremental learning curve build ──────────────────────
warnings.filterwarnings(
    "ignore",
    message="Warm-start fitting without increasing n_estimators"
)

# copy all best params *except* n_estimators; we’ll grow it manually
base_params = {k: v for k, v in best_params.items() if k != 'n_estimators'}
base_params.update({'warm_start': True, 'random_state': rand_st})

step         = 10
max_trees    = best_params['n_estimators']
rf           = RandomForestRegressor(n_estimators=step, **base_params)

train_mse, val_mse, n_seq = [], [], []

while rf.n_estimators <= max_trees:
    rf.fit(X_train, y_train)

    train_mse.append(mean_squared_error(y_train, rf.predict(X_train)))
    val_mse.append(mean_squared_error(y_val,   rf.predict(X_val)))
    n_seq.append(rf.n_estimators)

    rf.set_params(n_estimators=rf.n_estimators + step)

# ─────────────────────────── plot & save curve ─────────────────────────────
plt.figure()
plt.plot(n_seq, train_mse, label="Train MSE")
plt.plot(n_seq, val_mse,   label="Validation MSE")
plt.xlabel("Number of Trees");  plt.ylabel("MSE")
plt.title("Random Forest Learning Curve");  plt.legend();  plt.tight_layout()

curve_path = os.path.join(
    save_dir_PARPML_6inputs, "random_forest_learning_curve.png")
plt.savefig(curve_path, dpi=300);  plt.close()
print(f"📈  Learning curve saved to {curve_path}")

# ─────────────────────────── final test score ──────────────────────────────
best_rf = rf  # after loop, rf has > max_trees; we want previous model
best_rf.set_params(n_estimators=max_trees, warm_start=False)
best_rf.fit(X_tr_full, y_tr_full)                # train on full train+val

test_pred = best_rf.predict(X_test)
print(f"\nFinal Test MSE: {mean_squared_error(y_test, test_pred):.6f}")
