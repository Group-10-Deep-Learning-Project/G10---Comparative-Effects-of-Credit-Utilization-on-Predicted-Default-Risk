from LR import run_Model as run_lr
from SVM import run_Model as run_svm
from RF import run_Model as run_rf
from XGBoost import run_Model as run_xgb
from MLP import run_Model as run_mlp
import Preprocessing
import counterfactual

#Set seeds for testing
seeds = [42,123,456]

# Collect trained models/data for CF
models_results = []

### Preprocessing & Model Running
for seed in seeds:

    #Preprocessing
    x_v, y_v, x_train, y_train, x_test, y_test = Preprocessing.valuesWithSeed(seed)

    print(f"Seed Number:{seed}")

    # Linear Model (LR)
    try:
        run_lr(seed, x_v, y_v, x_train, y_train, x_test, y_test)
    except Exception as e:
        print(f'LR training failed or skipped: {e}')

    # Support Vector Machine (SVM)
    try:
        run_svm(seed, x_v, y_v, x_train, y_train, x_test, y_test)
    except Exception as e:
        print(f'SVM training failed or skipped: {e}')

    # XGBoost — capture return for counterfactuals
    try:
        xgb_model, X_tr_xgb, X_te_xgb, y_tr_xgb, y_te_xgb = run_xgb(
            seed, x_v, y_v, x_train, y_train, x_test, y_test
        )
        models_results.append({
            'name'   : 'XGBoost',
            'model'  : xgb_model,
            'X_train': X_tr_xgb,
            'X_test' : X_te_xgb,
            'y_train': y_tr_xgb,
            'y_test' : y_te_xgb,
        })
    except Exception as e:
        print(f'XGBoost training failed or skipped: {e}')

    # Random Forest — capture return for counterfactuals
    print(f"Random Forest Model - Seed {seed}")
    try:
        rf_model, X_tr_rf, X_te_rf, y_tr_rf, y_te_rf = run_rf(
            seed, x_v, y_v, x_train, y_train, x_test, y_test
        )
        models_results.append({
            'name'   : 'RandomForest',
            'model'  : rf_model,
            'X_train': X_tr_rf,
            'X_test' : X_te_rf,
            'y_train': y_tr_rf,
            'y_test' : y_te_rf,
        })
    except Exception as e:
        print(f'RF training failed or skipped: {e}')

    print("\n")

### Counterfactual Script Running for all collected models
import pandas as pd
import numpy as _np
import os
from sklearn.metrics import f1_score, roc_auc_score

results_rows = []

def _derive_feature_mapping(X_train_df):
    cols = X_train_df.columns.tolist()
    n_features = len(cols)
    detected_bill  = [c for c in cols if any(k in c.upper() for k in ['BILL', 'BILL_AMT', 'BILLAMT', 'AMT'])]
    detected_limit = next((c for c in cols if any(k in c.upper() for k in ['LIMIT', 'LIMIT_BAL', 'CREDIT_LIMIT'])), None)
    detected_util  = [c for c in cols if 'UTIL' in c.upper() or 'UTILI' in c.upper()]

    if detected_bill and detected_limit:
        bill_cols      = detected_bill[:6]
        limit_col      = detected_limit
        util_col_names = detected_util if detected_util else []
    else:
        if n_features >= 18:
            limit_col      = cols[0]
            util_col_names = cols[6:12]
            bill_cols      = cols[12:18]
        elif n_features >= 7:
            limit_col      = cols[0]
            bill_cols      = cols[-6:]
            util_col_names = cols[-12:-6] if n_features >= 12 else cols[1:1+min(6, n_features-1)]
        else:
            limit_col      = cols[0] if cols else None
            bill_cols      = cols.copy()
            util_col_names = []

    bill_cols      = [c for c in bill_cols      if c in cols]
    util_col_names = [c for c in util_col_names if c in cols]
    if limit_col not in cols:
        limit_col = cols[0] if cols else None

    return bill_cols, util_col_names, limit_col


for info in models_results:
    name  = info['name']
    model = info['model']
    X_tr  = info['X_train']
    X_te  = info['X_test']
    y_tr  = info.get('y_train', None)
    y_te  = info.get('y_test',  None)

    # Ensure DataFrames
    try:
        if not hasattr(X_tr, 'columns'):
            X_tr = pd.DataFrame(_np.asarray(X_tr), columns=[f"Feature_{i}" for i in range(_np.asarray(X_tr).shape[1])])
        if not hasattr(X_te, 'columns'):
            X_te = pd.DataFrame(_np.asarray(X_te), columns=[f"Feature_{i}" for i in range(_np.asarray(X_te).shape[1])])
    except Exception:
        print(f"Skipping {name} due to invalid feature matrices")
        continue

    bill_cols, util_col_names, limit_col = _derive_feature_mapping(X_tr)

    print(f"Running counterfactual for {name}...")
    try:
        cf_results = counterfactual.run_counterfactual_and_tests(
            model, X_te, X_tr, bill_cols, util_col_names, limit_col
        )
    except Exception as e:
        print(f"Counterfactual failed for {name}: {e}")
        continue

    # Compute F1 / AUC if possible
    try:
        probs = model.predict_proba(X_te)[:, 1]
        y_te_arr = _np.asarray(y_te).ravel()
        f1  = f1_score(y_te_arr, (probs >= 0.5).astype(int))
        auc = roc_auc_score(y_te_arr, probs)
    except Exception:
        f1 = None
        auc = None

    for lvl, r in cf_results.items():
        results_rows.append({
            'model'             : name,
            'seed'              : seed,
            'intervention_level': int(lvl * 100),
            'mean_A'            : float(r['mean_A']),
            'mean_B'            : float(r['mean_B']),
            'mean_abs_A'        : float(r['mean_abs_A']),
            'mean_abs_B'        : float(r['mean_abs_B']),
            'wilcoxon_p'        : float(r['wilcoxon_p']) if r['wilcoxon_p'] is not None else None,
            'f1'                : f1,
            'auc'               : auc,
            'notes'             : None,
        })

os.makedirs('results', exist_ok=True)
df_results = pd.DataFrame(results_rows)
df_results.to_csv('results/counterfactual_summary.csv', index=False)

print(f"\nSaved {len(df_results)} rows to results/counterfactual_summary.csv")
print(df_results.to_string(index=False))
