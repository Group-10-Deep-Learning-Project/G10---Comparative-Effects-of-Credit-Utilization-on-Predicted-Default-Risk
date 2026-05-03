from LR import run_Model as run_lr
from SVM import run_Model as run_svm
from RF import run_Model as run_rf
from XGBoost import run_Model as run_xgb
from MLP import run_Model as run_mlp
import Preprocessing
import counterfactual

#Set seeds for testing
seeds = [42]


### Preprocessing & Model Running
for seed in seeds:
    
    #Preprocessing
    x_v, y_v, x_train, y_train, x_test, y_test = Preprocessing.valuesWithSeed(seed)

    #Run Models based on seed

    print(f"Seed Number:{seed}")
    
    #print(f"Linear Model - Seed {seed}")
    #run_lr(seed,x_v, y_v, x_train, y_train, x_test, y_test)
    #print("\n")

    print(f"MLP Model - Seed {seed}")
    # Capture trained PyTorch model and tensors returned by run_mlp
    mlp_result = run_mlp(seed, x_v, y_v, x_train, y_train, x_test, y_test)
    try:
        model_mlp, x_train_mlp, x_test_mlp, y_train_mlp, y_test_mlp = mlp_result
    except Exception:
        # If run_mlp didn't return model, keep previous behavior
        model_mlp = None
        print("Warning: MLP.run_Model did not return a model object.")

    print("\n")

    # If we have a trained MLP model, run counterfactuals for it (wrap to sklearn-like API)
    if model_mlp is not None:
        import numpy as _np
        import pandas as _pd
        import torch as _torch
        # Convert tensors to numpy arrays if needed
        try:
            x_train_np_mlp = x_train_mlp.numpy()
            x_test_np_mlp  = x_test_mlp.numpy()
        except Exception:
            x_train_np_mlp = x_train_mlp
            x_test_np_mlp  = x_test_mlp

        n_features_mlp = x_train_np_mlp.shape[1]
        feature_names_mlp = [f"Feature_{i}" for i in range(n_features_mlp)]
        X_train_mlp_df = _pd.DataFrame(x_train_np_mlp, columns=feature_names_mlp)
        X_test_mlp_df  = _pd.DataFrame(x_test_np_mlp,  columns=feature_names_mlp)

        # Small wrapper so counterfactual can call predict_proba(model)
        class MLPWrapper:
            def __init__(self, model):
                self.model = model
            def predict_proba(self, X_df):
                X_arr = X_df.values.astype(_np.float32)
                X_tensor = _torch.tensor(X_arr)
                self.model.eval()
                with _torch.no_grad():
                    out = self.model(X_tensor)
                    probs = _torch.sigmoid(out).cpu().numpy().flatten()
                return _np.vstack([1 - probs, probs]).T

        mlp_wrapper = MLPWrapper(model_mlp)

        # Feature mapping (same convention as XGBoost)
        bill_cols = [f"Feature_{i}" for i in range(12, 18) if i < n_features_mlp]
        util_col_names = [f"Feature_{i}" for i in range(6, 12) if i < n_features_mlp]
        limit_col = "Feature_0"

        print("Running counterfactual for MLP (A vs B)...")
        cf_results_mlp = counterfactual.run_counterfactual_and_tests(mlp_wrapper, X_test_mlp_df, X_train_mlp_df, bill_cols, util_col_names, limit_col)

        print("\nMLP Wilcoxon p-values by level:")
        for lvl, res in cf_results_mlp.items():
            print(f"{int(lvl*100)}% -> p = {res['wilcoxon_p']}")

    print(f"Random Forest Model - Seed {seed}")
    run_rf(seed,x_v, y_v, x_train, y_train, x_test, y_test)
    print("\n")

    print(f"Support Vector Machine Model - Seed {seed}")
    run_svm(seed,x_v, y_v, x_train, y_train, x_test, y_test)
    print("\n")

    print(f"XGBoost Model - Seed {seed}")
    # Capture trained model and returned DataFrames from run_xgb so we can reuse for counterfactuals
    model_trained, X_tr_used, X_te_used, y_tr_used, y_te_used = run_xgb(seed, x_v, y_v, x_train, y_train, x_test, y_test)
    print("\n")

### Counterfactual Script Running
import pandas as pd
import counterfactual

# Use the model and data returned above for counterfactuals (no retrain)
model_cf = model_trained
X_tr_used_df = X_tr_used
X_te_used_df = X_te_used

# Default feature mapping (adjust if your real column names differ)
n_features = X_tr_used_df.shape[1]
bill_cols = [f"Feature_{i}" for i in range(12, 18) if i < n_features]
util_col_names = [f"Feature_{i}" for i in range(6, 12) if i < n_features]
limit_col = "Feature_0"

print("Running counterfactual A vs B (paired Wilcoxon tests)...")
cf_results = counterfactual.run_counterfactual_and_tests(model_cf, X_te_used_df, X_tr_used_df, bill_cols, util_col_names, limit_col)

print("\nWilcoxon p-values by level:")
for lvl, res in cf_results.items():
    print(f"{int(lvl*100)}% -> p = {res['wilcoxon_p']}")

# Collect results rows for summary
results_rows = []

# If we ran MLP CF, add its rows
if model_mlp is not None:
    # compute basic test metrics for MLP if possible
    try:
        from sklearn.metrics import f1_score, roc_auc_score
        # get MLP test probs
        X_test_mlp_arr = X_test_mlp_df
        probs_mlp = mlp_wrapper.predict_proba(X_test_mlp_arr)[:, 1]
        y_test_mlp_arr = y_test_mlp.numpy().ravel() if hasattr(y_test_mlp, 'numpy') else y_test_mlp.ravel()
        f1_mlp = f1_score(y_test_mlp_arr, (probs_mlp >= 0.5).astype(int))
        auc_mlp = roc_auc_score(y_test_mlp_arr, probs_mlp)
    except Exception:
        f1_mlp = None
        auc_mlp = None

    for lvl, r in cf_results_mlp.items():
        results_rows.append({
            'model': 'MLP',
            'seed': seed,
            'intervention_level': int(lvl*100),
            'mean_A': float(r['mean_A']),
            'mean_B': float(r['mean_B']),
            'mean_abs_A': float(r['mean_abs_A']),
            'mean_abs_B': float(r['mean_abs_B']),
            'wilcoxon_p': float(r['wilcoxon_p']) if r['wilcoxon_p'] is not None else None,
            'f1': f1_mlp,
            'auc': auc_mlp,
            'notes': None
        })

# Add XGBoost CF rows
for lvl, r in cf_results.items():
    # XGBoost test metrics may be available from run_xgb printed output; if not, leave None
    try:
        from sklearn.metrics import f1_score, roc_auc_score
        # y_te_used, y_test_prob variables exist from XGBoost context if returned; otherwise compute None
        # Here we attempt to compute using y_te_used and predict_proba
        y_te_arr = y_te_used
        probs_xgb = model_cf.predict_proba(X_te_used_df)[:, 1]
        f1_xgb = None
        auc_xgb = None
        try:
            f1_xgb = f1_score(y_te_arr, (probs_xgb >= 0.5).astype(int))
            auc_xgb = roc_auc_score(y_te_arr, probs_xgb)
        except Exception:
            f1_xgb = None
            auc_xgb = None
    except Exception:
        f1_xgb = None
        auc_xgb = None

    results_rows.append({
        'model': 'XGBoost',
        'seed': seed,
        'intervention_level': int(lvl*100),
        'mean_A': float(r['mean_A']),
        'mean_B': float(r['mean_B']),
        'mean_abs_A': float(r['mean_abs_A']),
        'mean_abs_B': float(r['mean_abs_B']),
        'wilcoxon_p': float(r['wilcoxon_p']) if r['wilcoxon_p'] is not None else None,
        'f1': f1_xgb,
        'auc': auc_xgb,
        'notes': None
    })

# Save results
import os
import pandas as pd
import sqlite3

os.makedirs('results', exist_ok=True)
df_results = pd.DataFrame(results_rows)
df_results.to_csv('results/counterfactual_summary.csv', index=False)

conn = sqlite3.connect('results/counterfactuals.db')
df_results.to_sql('cf_summary', conn, if_exists='replace', index=False)
conn.close()

print('\nSaved counterfactual summary to results/counterfactual_summary.csv and results/counterfactuals.db')

### Print out results