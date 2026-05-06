# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             precision_recall_curve, brier_score_loss)
from sklearn.calibration import calibration_curve
from scipy.stats import wilcoxon
import shap
import random


# %%
# Hyperparameter Tuning with Grid Search
# 5-fold cross validation
def run_Model(seed, x_v, y_v, x_train, y_train, x_test, y_test):
    
    random.seed(seed)
    np.random.seed(seed)

    X_val, y_val, X_train, y_train, X_test, y_test = x_v, y_v, x_train, y_train, x_test, y_test

    # Coerce y to numpy 1D arrays and X to pandas DataFrames (handle torch tensors)
    import numpy as _np
    import pandas as _pd
    try:
        y_train = _np.array(y_train).ravel()
        y_test  = _np.array(y_test).ravel()
        y_val   = _np.array(y_val).ravel()
    except Exception:
        y_train = _np.asarray(y_train).ravel()
        y_test  = _np.asarray(y_test).ravel()
        y_val   = _np.asarray(y_val).ravel()

    def _to_df(X):
        if hasattr(X, 'columns'):
            return X
        try:
            X_arr = _np.array(X)
        except Exception:
            X_arr = X
        cols = [f"Feature_{i}" for i in range(X_arr.shape[1])]
        return _pd.DataFrame(X_arr, columns=cols)

    X_train = _to_df(X_train)
    X_test  = _to_df(X_test)
    X_val   = _to_df(X_val)

    # Heuristic feature mapping for counterfactuals
    # Try to detect real column names (e.g., 'BILL_AMT1'..'BILL_AMT6', 'LIMIT_BAL', 'UTIL_avg')
    cols = X_train.columns.tolist()
    n_features = len(cols)

    # Attempt to detect by keyword matches
    detected_bill = [c for c in cols if any(k in c.upper() for k in ['BILL', 'BILL_AMT', 'BILLAMT', 'AMT'])]
    detected_limit = next((c for c in cols if any(k in c.upper() for k in ['LIMIT', 'LIMIT_BAL', 'CREDIT_LIMIT'])), None)
    detected_util = [c for c in cols if 'UTIL' in c.upper() or 'UTILI' in c.upper()]

    if detected_bill and detected_limit:
        bill_cols = detected_bill[:6]  # take up to 6 if more detected
        limit_col = detected_limit
        util_col_names = detected_util if detected_util else []
    else:
        # Fall back to positional assumptions used elsewhere in the project
        # Typical ordering (after scaling, before one-hot): limit_col at idx 0,
        # utilization cols at 6-11, bill amt cols at 12-17. Guard for small feature sets.
        if n_features >= 18:
            limit_col = cols[0]
            util_col_names = cols[6:12]
            bill_cols = cols[12:18]
        elif n_features >= 7:
            limit_col = cols[0]
            # assume last 6 are bill amounts
            bill_cols = cols[-6:]
            # util columns: the 6 columns prior to the bills if available, else a best-effort slice
            if n_features >= 12:
                util_col_names = cols[-12:-6]
            else:
                util_col_names = cols[1:1+min(6, n_features-1)]
        else:
            # extremely small feature set: treat everything as bill_cols fallback
            limit_col = cols[0] if cols else None
            bill_cols = cols.copy()
            util_col_names = []

    # Ensure unique, existing columns and consistent ordering
    bill_cols = [c for c in bill_cols if c in cols]
    util_col_names = [c for c in util_col_names if c in cols]
    if limit_col not in cols:
        limit_col = cols[0] if cols else None

    param_grid = {
        'n_estimators'    : [200, 300],
        'max_depth'       : [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features'    : ['sqrt', 'log2'],
        'class_weight'    : ['balanced', {0: 1, 1: 3}]
    }

    rf_base = RandomForestClassifier(random_state=seed, n_jobs=-1)

    # Use exhaustive GridSearchCV for hyperparameter tuning (original behavior)
    grid_search = GridSearchCV(
        estimator = rf_base,
        param_grid = param_grid,
        scoring = 'f1',
        cv = 5,
        verbose = 2,
        n_jobs = -1
    )

    grid_search.fit(X_train, y_train.ravel())

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1: {grid_search.best_score_:.4f}")

    # %%
    # Train Final Model with Best Parameters

    rf_model = grid_search.best_estimator_

    y_val_pred = rf_model.predict(X_val)
    y_val_prob = rf_model.predict_proba(X_val)[:, 1]

    print("=== Validation Set Performance ===")
    print(f"Accuracy : {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
    print(f"Recall   : {recall_score(y_val, y_val_pred):.4f}")
    print(f"F1 Score : {f1_score(y_val, y_val_pred):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_val, y_val_prob):.4f}")

    # %%
    # Threshold Tuning on Validation Set

    y_val_prob_thresh = rf_model.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob_thresh)
    f1_scores_curve = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_thresh_idx = f1_scores_curve.argmax()
    best_threshold  = thresholds[best_thresh_idx]

    print(f"Best threshold (val F1): {best_threshold:.4f}")
    print(f"Val F1 at best threshold: {f1_scores_curve[best_thresh_idx]:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, precisions[:-1], label='Precision')
    plt.plot(thresholds, recalls[:-1],    label='Recall')
    plt.plot(thresholds, f1_scores_curve[:-1], label='F1')
    plt.axvline(best_threshold, color='red', linestyle='--',
                label=f'Best threshold = {best_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision / Recall / F1 vs Decision Threshold (Validation Set)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # %%
    # Final Evaluation on Test Set

    y_test_prob      = rf_model.predict_proba(X_test)[:, 1]
    y_test_pred_tuned = (y_test_prob >= best_threshold).astype(int)

    print("=== RF Test Set Results (Tuned Threshold) ===")
    print(f"Threshold  : {best_threshold:.4f}")
    print(f"Accuracy   : {accuracy_score(y_test, y_test_pred_tuned):.4f}")
    print(f"Precision  : {precision_score(y_test, y_test_pred_tuned):.4f}")
    print(f"Recall     : {recall_score(y_test, y_test_pred_tuned):.4f}")
    print(f"F1 Score   : {f1_score(y_test, y_test_pred_tuned):.4f}")
    print(f"ROC AUC    : {roc_auc_score(y_test, y_test_prob):.4f}")

    # CHANGE: Brier score added per reviewer feedback (calibration check)
    brier = brier_score_loss(y_test, y_test_prob)
    print(f"Brier Score: {brier:.4f}")

    print("\n=== LR Baseline Results ===")
    print("Accuracy : 0.8082")
    print("Precision: 0.6913")
    print("Recall   : 0.2396")
    print("F1 Score : 0.3559")
    print("ROC AUC  : 0.7095")

    # %%
    # Calibration Curve

    prob_true, prob_pred = calibration_curve(y_test, y_test_prob, n_bins=10)

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Random Forest')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve -- Random Forest')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('RF_calibration_curve.png', dpi=150)
    plt.show()

    print(f"Brier Score: {brier_score_loss(y_test, y_test_prob):.4f}")
    print("(Lower is better; perfectly calibrated model = 0)")

    # %%
    # SHAP Feature Importance

    explainer = shap.TreeExplainer(rf_model)
    X_sample  = X_test.sample(500, random_state=seed)

    shap_explanation = explainer(X_sample)

    if len(shap_explanation.values.shape) == 3:
        shap_vals_class1 = shap_explanation.values[:, :, 1]
        shap_explanation_class1 = shap.Explanation(
            values        = shap_vals_class1,
            base_values   = shap_explanation.base_values[:, 1],
            data          = shap_explanation.data,
            feature_names = X_sample.columns.tolist()
        )
    else:
        shap_explanation_class1 = shap_explanation

    shap.plots.bar(shap_explanation_class1,      max_display=20, show=True)
    shap.plots.beeswarm(shap_explanation_class1, max_display=20, show=True)

    # %%
    # Counterfactual Helper Function

    def run_intervention(model, X_original, intervention_fn, label):
        X_modified    = intervention_fn(X_original.copy())
        prob_original = model.predict_proba(X_original)[:, 1]
        prob_modified = model.predict_proba(X_modified)[:, 1]
        delta_p       = prob_modified - prob_original

        print(f"\n--- {label} ---")
        print(f"Mean ΔP(default)       : {delta_p.mean():.6f}")
        print(f"Mean |ΔP(default)|     : {np.abs(delta_p).mean():.6f}")
        print(f"Clients with reduction : {(delta_p < 0).sum()} / {len(delta_p)}")

        return delta_p

    # Helper: safe clipping bounds (removed special-case for 'UTIL_avg')
    def _get_clip_bounds(col, X_train, util_col_names, X_mod=None):
        # If the column exists in training data, use its range
        if col in X_train.columns:
            return X_train[col].min(), X_train[col].max()
        # Fallback: use the modified data's range if present
        if X_mod is not None and col in X_mod.columns:
            return X_mod[col].min(), X_mod[col].max()
        # Final fallback: generic 0-1 range
        return 0.0, 1.0

    # %%
    # Intervention A: Reduce Bill Amounts

    results_A = {}

    for pct in [0.10, 0.25, 0.50]:
        def intervention_A(X, reduction=pct):
            X_mod = X.copy()
            for col in bill_cols:
                X_mod[col] = X_mod[col] * (1 - reduction)
            # Recompute util columns only if we detected them
            if util_col_names:
                for bill, util in zip(bill_cols, util_col_names):
                    # avoid division by zero by replacing 0 with NaN on denominator
                    X_mod[util] = X_mod[bill] / X_mod[limit_col].replace(0, np.nan)

            # Clip each column using safe bounds (no UTIL_avg)
            for col in bill_cols + util_col_names:
                minv, maxv = _get_clip_bounds(col, X_train, util_col_names, X_mod)
                X_mod[col] = X_mod[col].clip(minv, maxv)
            return X_mod

        delta = run_intervention(
            rf_model, X_test, intervention_A,
            f"Intervention A - Reduce Bill Amounts by {int(pct*100)}%"
        )
        results_A[pct] = delta

    # %%
    # Intervention B: Increase Credit Limit

    results_B = {}

    for pct in [0.10, 0.25, 0.50]:
        def intervention_B(X, increase=pct):
            X_mod = X.copy()
            X_mod[limit_col] = X_mod[limit_col] * (1 + increase)
            if util_col_names:
                for bill, util in zip(bill_cols, util_col_names):
                    X_mod[util] = X_mod[bill] / X_mod[limit_col].replace(0, np.nan)

            for col in [limit_col] + util_col_names:
                minv, maxv = _get_clip_bounds(col, X_train, util_col_names, X_mod)
                X_mod[col] = X_mod[col].clip(minv, maxv)
            return X_mod

        delta = run_intervention(
            rf_model, X_test, intervention_B,
            f"Intervention B - Increase Credit Limit by {int(pct*100)}%"
        )
        results_B[pct] = delta

    # %%
    # Intervention C: Increase Limit, Hold Utilization Constant

    results_C = {}

    for pct in [0.10, 0.25, 0.50]:
        def intervention_C(X, increase=pct):
            X_mod = X.copy()
            X_mod[limit_col] = X_mod[limit_col] * (1 + increase)
            for col in bill_cols:
                X_mod[col] = X_mod[col] * (1 + increase)
            if util_col_names:
                for bill, util in zip(bill_cols, util_col_names):
                    X_mod[util] = X_mod[bill] / X_mod[limit_col].replace(0, np.nan)

            for col in [limit_col] + bill_cols + util_col_names:
                minv, maxv = _get_clip_bounds(col, X_train, util_col_names, X_mod)
                X_mod[col] = X_mod[col].clip(minv, maxv)
            return X_mod

        delta = run_intervention(
            rf_model, X_test, intervention_C,
            f"Intervention C - Increase Limit+Bills by {int(pct*100)}% (Utilization Held Constant)"
        )
        results_C[pct] = delta

    # %%
    # Summary Comparison Plot

    levels = [0.10, 0.25, 0.50]
    labels = ['10%', '25%', '50%']

    mean_A = [np.abs(results_A[p]).mean() for p in levels]
    mean_B = [np.abs(results_B[p]).mean() for p in levels]
    mean_C = [np.abs(results_C[p]).mean() for p in levels]

    plt.figure(figsize=(8, 5))
    plt.plot(labels, mean_A, marker='o', label='A: Reduce Bill Amounts')
    plt.plot(labels, mean_B, marker='s', label='B: Increase Credit Limit')
    plt.plot(labels, mean_C, marker='^', label='C: Increase Limit (Util Constant)')
    plt.xlabel('Intervention Level')
    plt.ylabel('Mean |ΔP(default)|')
    plt.title('Random Forest: Mean Change in Predicted Default Probability\n'
            'by Intervention Type and Level')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('RF_counterfactual_comparison.png', dpi=150)
    plt.show()

    # %%
    # Client Segmentation Analysis
    # Added grouped bar chart per reviewer feedback so the risk-group reversal is visually prominent

    baseline_probs = rf_model.predict_proba(X_test)[:, 1]

    high_risk_mask = baseline_probs > 0.5
    low_risk_mask  = baseline_probs < 0.4

    print(f"High-risk clients (P > 0.5): {high_risk_mask.sum()}")
    print(f"Low-risk clients  (P < 0.4): {low_risk_mask.sum()}")

    seg_results = {}

    for mask, group_label in [(high_risk_mask, 'High-Risk'),
                            (low_risk_mask,  'Low-Risk')]:
        X_group = X_test[mask]
        if len(X_group) == 0:
            continue
        print(f"\n=== {group_label} Group (n={len(X_group)}) ===")

        def int_A_seg(X):
            X_mod = X.copy()
            for col in bill_cols:
                X_mod[col] = X_mod[col] * 0.75
            if util_col_names:
                for bill, util in zip(bill_cols, util_col_names):
                    X_mod[util] = X_mod[bill] / X_mod[limit_col].replace(0, np.nan)
            for col in bill_cols + util_col_names:
                minv, maxv = _get_clip_bounds(col, X_train, util_col_names, X_mod)
                X_mod[col] = X_mod[col].clip(minv, maxv)
            return X_mod

        def int_B_seg(X):
            X_mod = X.copy()
            X_mod[limit_col] = X_mod[limit_col] * 1.25
            if util_col_names:
                for bill, util in zip(bill_cols, util_col_names):
                    X_mod[util] = X_mod[bill] / X_mod[limit_col].replace(0, np.nan)
            for col in [limit_col] + util_col_names:
                minv, maxv = _get_clip_bounds(col, X_train, util_col_names, X_mod)
                X_mod[col] = X_mod[col].clip(minv, maxv)
            return X_mod

        dA = run_intervention(rf_model, X_group, int_A_seg,
                            f"{group_label} - Intervention A (25% bill reduction)")
        dB = run_intervention(rf_model, X_group, int_B_seg,
                            f"{group_label} - Intervention B (25% limit increase)")

        seg_results[group_label] = {'A': dA.mean(), 'B': dB.mean()}

        # Wilcoxon test per group
        stat, p = wilcoxon(dA, dB, alternative='two-sided')
        print(f"  Wilcoxon A vs B: W={stat:.1f}, p={p:.4f}")

    groups = list(seg_results.keys())
    x      = np.arange(len(groups))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_A = ax.bar(x - width/2,
                    [seg_results[g]['A'] for g in groups],
                    width, label='Int A: Reduce Bills')
    bars_B = ax.bar(x + width/2,
                    [seg_results[g]['B'] for g in groups],
                    width, label='Int B: Increase Limit')

    ax.set_xlabel('Risk Group')
    ax.set_ylabel('Mean ΔP(default)')
    ax.set_title('Segmentation: Mean ΔP by Risk Group and Intervention (25% level)')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('RF_segmentation_bar.png', dpi=150)
    plt.show()

    # Save trained RF model so counterfactual figures can be regenerated without retraining
    from joblib import dump
    dump(rf_model, f'rf_model_seed{seed}.joblib')
    print(f"RF model saved to rf_model_seed{seed}.joblib")

    # Return the trained model and DataFrames for downstream analysis
    return rf_model, X_train, X_test, y_train, y_test
