import os
import itertools
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics        import mean_squared_error

# ─────────── your utilities ───────────
from utilities import process_columns, file_path_PARPML_6inputs, save_dir_PARPML_6inputs

os.makedirs(save_dir_PARPML_6inputs, exist_ok=True)

# ─────────── load & split ───────────
df = process_columns(pd.read_csv(file_path_PARPML_6inputs))
X  = df[['runload','alphaload','mu','k','n','PB']]
y  = df['outputAmplification']

# 80/20 train/test, then 85/15 train/val
X_tr_full, X_test, y_tr_full, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train,    X_val,  y_train,    y_val  = train_test_split(X_tr_full, y_tr_full, test_size=0.15, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)

# ─────────── manual grid search ───────────
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth'    : [3, 6, 9]
}

best_rmse = float('inf')
best_params = None
best_bst    = None

for lr, md in itertools.product(param_grid['learning_rate'], param_grid['max_depth']):
    params = {
        'objective'   : 'reg:squarederror',
        'eval_metric' : 'rmse',
        'eta'          : lr,
        'max_depth'    : md,
        'seed'         : 42
    }
    # train with early stopping on the validation split
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=10
    )
    # the best validation RMSE is at bst.best_score
    val_rmse = bst.best_score
    print(f"lr={lr:0.3f}, depth={md:2d} → best_val_RMSE={val_rmse:.4f} at round {bst.best_iteration}")
    if val_rmse < best_rmse:
        best_rmse   = val_rmse
        best_params = params.copy()
        best_bst    = bst

print("\n▶︎ Best combo:", best_params, "→ val_RMSE =", best_rmse)

# … (everything up through your manual grid search loop) …

best_rmse    = float('inf')
best_params  = None
best_bst     = None
best_history = None

for lr, md in itertools.product(param_grid['learning_rate'],
                                param_grid['max_depth']):
    params = {
        'objective'   : 'reg:squarederror',
        'eval_metric' : 'rmse',
        'eta'         : lr,
        'max_depth'   : md,
        'seed'        : 42
    }

    # prepare a container for the history
    history = {}

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=10,
        evals_result=history            # <— pass in your dict
    )

    val_rmse = bst.best_score
    print(f"lr={lr:.3f}, depth={md:2d} → "
          f"best_val_RMSE={val_rmse:.4f} "
          f"@ round {bst.best_iteration}")

    if val_rmse < best_rmse:
        best_rmse    = val_rmse
        best_params  = params.copy()
        best_bst     = bst
        best_history = history         # <— keep its history

print("\n▶︎ Best combo:", best_params, "→ val_RMSE =", best_rmse)

# ─── Plot learning curve from best_history ───────────────────────────────
train_rmse = best_history['train']['rmse']
val_rmse   = best_history['validation']['rmse']
rounds     = range(1, len(train_rmse) + 1)

plt.figure(figsize=(6,4))
plt.plot(rounds, train_rmse,      label='Train RMSE')
plt.plot(rounds, val_rmse,        label='Val RMSE')
plt.axvline(best_bst.best_iteration,
            color='k', linestyle='--',
            label=f"Stopping at {best_bst.best_iteration}")
plt.xlabel('Boosting Round')
plt.ylabel('RMSE')
plt.title('XGBoost Learning Curve (best params)')
plt.legend()
plt.tight_layout()

curve_path = os.path.join(save_dir_PARPML_6inputs,
                          'xgb_learning_curve.png')
plt.savefig(curve_path, dpi=300)
plt.close()
print(f"Learning curve saved to {curve_path}")

# ─── Final test evaluation ───────────────────────────────────────────────
dtest    = xgb.DMatrix(X_test)
y_pred   = best_bst.predict(dtest)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Test RMSE with best model: {test_rmse:.4f}")

