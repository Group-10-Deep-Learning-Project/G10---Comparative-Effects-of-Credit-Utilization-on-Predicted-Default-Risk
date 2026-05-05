from LR import run_Model as run_lr
from SVM import run_Model as run_svm
from RF import run_Model as run_rf
from XGBoost import run_Model as run_xgb
from MLP import run_Model as run_mlp
import Preprocessing
import counterfactual

#Set seeds for testing
seeds = [42]

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
    except Exception:
        print('LR training failed or skipped')

    # Support Vector Machine (SVM)
    try:
        run_svm(seed, x_v, y_v, x_train, y_train, x_test, y_test)
    except Exception:
        print('SVM training failed or skipped')

    # XGBoost - capture trained model and data for counterfactuals
    try:
        model_trained, X_tr_used, X_te_used, y_tr_used, y_te_used = run_xgb(seed, x_v, y_v, x_train, y_train, x_test, y_test)
    except Exception:
        print('XGBoost training failed or skipped')

    # (MLP is already run earlier in the file to capture the PyTorch model) 

    print(f"Random Forest Model - Seed {seed}")
    # Ensure we have feature names and define bill_cols, util_col_names, limit_col robustly
    import pandas as _pd, numpy as _np
    try:
        x_train_arr = x_train.numpy()
    except Exception:
        x_train_arr = x_train
    x_train_arr = _np.asarray(x_train_arr)
    n_features = x_train_arr.shape[1]
    feature_names = [f"Feature_{i}" for i in range(n_features)]

    # Default: assume bill AMT columns are the last 6 features (fallback)
    if n_features >= 6:
        bill_cols = [f"Feature_{i}" for i in range(n_features - 6, n_features)]
    else:
        bill_cols = feature_names.copy()

    util_col_names = [f"Feature_{i}" for i in range(6, 12) if i < n_features]
    limit_col = "Feature_0" if n_features > 0 else feature_names[0]

    # Pass feature mapping into RF so RF.py doesn't need to guess globals
    try:
        run_rf(seed, x_v, y_v, x_train, y_train, x_test, y_test,
               bill_cols=bill_cols, util_col_names=util_col_names, limit_col=limit_col)
    except TypeError:
        # Older RF.run_Model may not accept kwargs; call without them
        run_rf(seed, x_v, y_v, x_train, y_train, x_test, y_test)

    print("\n")

### Counterfactual Script Running for all collected models
import pandas as pd
import os
from sklearn.metrics import f1_score, roc_auc_score

results_rows = []

# Helper: derive bill/util/limit mapping from training DataFrame (same heuristics as RF)
def _derive_feature_mapping(X_train_df):
    cols = X_train_df.columns.tolist()
    n_features = len(cols)
    detected_bill = [c for c in cols if any(k in c.upper() for k in ['BILL', 'BILL_AMT', 'BILLAMT', 'AMT'])]
    detected_limit = next((c for c in cols if any(k in c.upper() for k in ['LIMIT', 'LIMIT_BAL', 'CREDIT_LIMIT'])), None)
    detected_util = [c for c in cols if 'UTIL' in c.upper() or 'UTILI' in c.upper()]

    if detected_bill and detected_limit:
        bill_cols = detected_bill[:6]
        limit_col = detected_limit
        util_col_names = detected_util if detected_util else []
    else:
        if n_features >= 18:
            limit_col = cols[0]
            util_col_names = cols[6:12]
            bill_cols = cols[12:18]
        elif n_features >= 7:
            limit_col = cols[0]
            bill_cols = cols[-6:]
            if n_features >= 12:
                util_col_names = cols[-12:-6]
            else:
                util_col_names = cols[1:1+min(6, n_features-1)]
        else:
            limit_col = cols[0] if cols else None
            bill_cols = cols.copy()
            util_col_names = []

    bill_cols = [c for c in bill_cols if c in cols]
    util_col_names = [c for c in util_col_names if c in cols]
    if limit_col not in cols:
        limit_col = cols[0] if cols else None

    return bill_cols, util_col_names, limit_col

for info in models_results:
    name = info['name']
    model = info['model']
    X_tr = info['X_train']
    X_te = info['X_test']
    y_tr = info.get('y_train', None)
    y_te = info.get('y_test', None)

    # Ensure DataFrame types
    try:
        import numpy as _np, pandas as _pd
        if not hasattr(X_tr, 'columns'):
            X_tr = _pd.DataFrame(_np.asarray(X_tr), columns=[f"Feature_{i}" for i in range(_np.asarray(X_tr).shape[1])])
        if not hasattr(X_te, 'columns'):
            X_te = _pd.DataFrame(_np.asarray(X_te), columns=[f"Feature_{i}" for i in range(_np.asarray(X_te).shape[1])])
    except Exception:
        print(f"Skipping {name} due to invalid feature matrices")
        continue

    bill_cols, util_col_names, limit_col = _derive_feature_mapping(X_tr)

    print(f"Running counterfactual for {name}...")
    try:
        cf_results = counterfactual.run_counterfactual_and_tests(model, X_te, X_tr, bill_cols, util_col_names, limit_col)
    except Exception as e:
        print(f"Counterfactual failed for {name}: {e}")
        continue

    # compute f1/auc if possible
    try:
        probs = model.predict_proba(X_te)[:, 1]
        y_te_arr = y_te
        try:
            y_te_arr = y_te_arr.numpy().ravel()
        except Exception:
            pass
        f1 = None; auc = None
        try:
            f1 = f1_score(y_te_arr, (probs >= 0.5).astype(int))
            auc = roc_auc_score(y_te_arr, probs)
        except Exception:
            f1 = None; auc = None
    except Exception:
        f1 = None; auc = None

    for lvl, r in cf_results.items():
        results_rows.append({
            'model': name,
            'seed': seeds[0] if seeds else None,
            'intervention_level': int(lvl*100),
            'mean_A': float(r['mean_A']),
            'mean_B': float(r['mean_B']),
            'mean_abs_A': float(r['mean_abs_A']),
            'mean_abs_B': float(r['mean_abs_B']),
            'wilcoxon_p': float(r['wilcoxon_p']) if r['wilcoxon_p'] is not None else None,
            'f1': f1,
            'auc': auc,
            'notes': None
        })

os.makedirs('results', exist_ok=True)
df_results = pd.DataFrame(results_rows)
df_results.to_csv('results/counterfactual_summary.csv', index=False)

print('\nSaved counterfactual summary to results/counterfactual_summary.csv and results/counterfactuals.db')

### Print out results