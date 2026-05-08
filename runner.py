from LR import run_Model as run_lr
from SVM import run_Model as run_svm
from RF import run_Model as run_rf
from XGBoost import run_Model as run_xgb
from MLP import run_Model as run_mlp
import Preprocessing
import counterfactual
import torch

# Set seeds for testing
seeds = [42, 123, 456]

# Collect trained models/data for CF
models_results = []

### Preprocessing & Model Running
for seed in seeds:

    # Preprocessing
    x_v, y_v, x_train, y_train, x_test, y_test = Preprocessing.valuesWithSeed(seed)

    print(f"Seed Number:{seed}")

    # Linear Model (LR)
    try:
        run_lr(seed, x_v, y_v, x_train, y_train, x_test, y_test)
    except Exception as e:
        print(f'LR training failed or skipped: {e}')

    # Support Vector Machine (SVM)
    try:
        svm_model, X_tr_svm, X_te_svm, y_tr_svm, y_te_svm = run_svm(
            seed, x_v, y_v, x_train, y_train, x_test, y_test
        )
        models_results.append({
            'name'    : 'SVM',
            'seed'    : seed,
            'model'   : svm_model,
            'X_train' : X_tr_svm,
            'X_test'  : X_te_svm,
            'y_train' : y_tr_svm,
            'y_test'  : y_te_svm,
            'is_torch': False,
        })
    except Exception as e:
        print(f'SVM training failed or skipped: {e}')

    # XGBoost
    try:
        xgb_model, X_tr_xgb, X_te_xgb, y_tr_xgb, y_te_xgb = run_xgb(
            seed, x_v, y_v, x_train, y_train, x_test, y_test
        )
        models_results.append({
            'name'    : 'XGBoost',
            'seed'    : seed,
            'model'   : xgb_model,
            'X_train' : X_tr_xgb,
            'X_test'  : X_te_xgb,
            'y_train' : y_tr_xgb,
            'y_test'  : y_te_xgb,
            'is_torch': False,
        })
    except Exception as e:
        print(f'XGBoost training failed or skipped: {e}')

    # Random Forest
    print(f"Random Forest Model - Seed {seed}")
    try:
        rf_model, X_tr_rf, X_te_rf, y_tr_rf, y_te_rf = run_rf(
            seed, x_v, y_v, x_train, y_train, x_test, y_test
        )
        models_results.append({
            'name'    : 'RandomForest',
            'seed'    : seed,
            'model'   : rf_model,
            'X_train' : X_tr_rf,
            'X_test'  : X_te_rf,
            'y_train' : y_tr_rf,
            'y_test'  : y_te_rf,
            'is_torch': False,
        })
    except Exception as e:
        print(f'RF training failed or skipped: {e}')

    # MLP
    try:
        mlp_model, x_tr_mlp, x_te_mlp, y_tr_mlp, y_te_mlp = run_mlp(
            seed, x_v, y_v, x_train, y_train, x_test, y_test
        )
        models_results.append({
            'name'    : 'MLP',
            'seed'    : seed,
            'model'   : mlp_model,
            'X_train' : x_tr_mlp,
            'X_test'  : x_te_mlp,
            'y_train' : y_tr_mlp,
            'y_test'  : y_te_mlp,
            'is_torch': True,
        })
    except Exception as e:
        print(f'MLP training failed or skipped: {e}')

    print("\n")

### Counterfactual Script Running for all collected models
import pandas as pd
import numpy as _np
import os
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss

cf_rows  = []   # one row per model x seed x intervention level
seg_rows = []   # one row per model x seed x risk group


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


def _get_probs(model, X, is_torch=False):
    if is_torch:
        model.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(_np.asarray(X), dtype=torch.float32)
            probs = torch.sigmoid(model(X)).cpu().numpy().flatten()
        return probs
    else:
        return model.predict_proba(X)[:, 1]


for info in models_results:
    name       = info['name']
    model      = info['model']
    X_tr       = info['X_train']
    X_te       = info['X_test']
    y_te       = info.get('y_test', None)
    entry_seed = info['seed']
    is_torch   = info.get('is_torch', False)

    # Convert to DataFrames
    try:
        if is_torch:
            X_tr_cf = pd.DataFrame(_np.asarray(X_tr), columns=[f"Feature_{i}" for i in range(_np.asarray(X_tr).shape[1])])
            X_te_cf = pd.DataFrame(_np.asarray(X_te), columns=[f"Feature_{i}" for i in range(_np.asarray(X_te).shape[1])])
        else:
            X_tr_cf = X_tr if hasattr(X_tr, 'columns') else pd.DataFrame(_np.asarray(X_tr), columns=[f"Feature_{i}" for i in range(_np.asarray(X_tr).shape[1])])
            X_te_cf = X_te if hasattr(X_te, 'columns') else pd.DataFrame(_np.asarray(X_te), columns=[f"Feature_{i}" for i in range(_np.asarray(X_te).shape[1])])
    except Exception:
        print(f"Skipping {name} (seed {entry_seed}) due to invalid feature matrices")
        continue

    bill_cols, util_col_names, limit_col = _derive_feature_mapping(X_tr_cf)

    # Wrap MLP for sklearn-compatible interface
    if is_torch:
        class TorchWrapper:
            def __init__(self, torch_model):
                self._model = torch_model
            def predict_proba(self, X):
                self._model.eval()
                with torch.no_grad():
                    X_t = torch.tensor(_np.asarray(X), dtype=torch.float32)
                    probs = torch.sigmoid(self._model(X_t)).cpu().numpy().flatten()
                return _np.column_stack([1 - probs, probs])
        model_cf = TorchWrapper(model)
    else:
        model_cf = model

    print(f"\nRunning counterfactual for {name} (seed {entry_seed})...")
    try:
        cf_results, seg_results = counterfactual.run_counterfactual_and_tests(
            model_cf, X_te_cf, X_tr_cf, bill_cols, util_col_names, limit_col
        )
    except Exception as e:
        print(f"Counterfactual failed for {name} (seed {entry_seed}): {e}")
        continue

    # Compute F1 / AUC / Brier at default 0.5 threshold
    try:
        probs     = _get_probs(model, X_te, is_torch=is_torch)
        y_te_arr  = _np.asarray(y_te).ravel()
        f1_val    = f1_score(y_te_arr, (probs >= 0.5).astype(int))
        auc_val   = roc_auc_score(y_te_arr, probs)
        brier_val = brier_score_loss(y_te_arr, probs)
    except Exception:
        f1_val    = None
        auc_val   = None
        brier_val = None

    # Save counterfactual rows (one per level)
    for lvl, r in cf_results.items():
        cf_rows.append({
            'model'             : name,
            'seed'              : entry_seed,
            'intervention_level': int(lvl * 100),
            'mean_A'            : float(r['mean_A']),
            'mean_B'            : float(r['mean_B']),
            'mean_C'            : float(r['mean_C']),
            'mean_abs_A'        : float(r['mean_abs_A']),
            'mean_abs_B'        : float(r['mean_abs_B']),
            'mean_abs_C'        : float(r['mean_abs_C']),
            'n_reduced_A'       : r['n_reduced_A'],
            'n_reduced_B'       : r['n_reduced_B'],
            'n_reduced_C'       : r['n_reduced_C'],
            'n_total'           : r['n_total'],
            'wilcoxon_p_AB'     : r['wilcoxon_p_AB'],
            'wilcoxon_p_AC'     : r['wilcoxon_p_AC'],
            'wilcoxon_p_BC'     : r['wilcoxon_p_BC'],
            'f1'                : f1_val,
            'auc'               : auc_val,
            'brier'             : brier_val,
        })

    # Save segmentation rows (one per risk group)
    for group, sr in seg_results.items():
        seg_rows.append({
            'model'         : name,
            'seed'          : entry_seed,
            'risk_group'    : group,
            'n'             : sr['n'],
            'mean_A'        : sr['mean_A'],
            'mean_B'        : sr['mean_B'],
            'mean_abs_A'    : sr['mean_abs_A'],
            'mean_abs_B'    : sr['mean_abs_B'],
            'wilcoxon_p_AB' : sr['wilcoxon_p_AB'],
            'f1'            : f1_val,
            'auc'           : auc_val,
            'brier'         : brier_val,
        })

# Save both CSVs
os.makedirs('results', exist_ok=True)

df_cf  = pd.DataFrame(cf_rows)
df_seg = pd.DataFrame(seg_rows)

df_cf.to_csv('results/counterfactual_summary.csv',  index=False)
df_seg.to_csv('results/segmentation_summary.csv',   index=False)

print(f"\nSaved {len(df_cf)} rows  -> results/counterfactual_summary.csv")
print(f"Saved {len(df_seg)} rows  -> results/segmentation_summary.csv")
print("\n=== COUNTERFACTUAL SUMMARY ===")
print(df_cf.to_string(index=False))
print("\n=== SEGMENTATION SUMMARY ===")
print(df_seg.to_string(index=False))
