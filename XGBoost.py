# XGBoost Model
## Humaid Billoo
### V3 (Added A/B/C Counterfactuals)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import wilcoxon

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    classification_report, confusion_matrix,
    precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
import shap


def run_Model(seed, x_v, y_v, x_train, y_train, x_test, y_test):

    random.seed(seed)
    np.random.seed(seed)

    X_val, y_val, X_train, y_train, X_test, y_test = x_v, y_v, x_train, y_train, x_test, y_test

    import numpy as _np
    try:
        y_train = _np.array(y_train).ravel()
        y_test  = _np.array(y_test).ravel()
        y_val   = _np.array(y_val).ravel()
    except Exception:
        y_train = _np.asarray(y_train).ravel()
        y_test  = _np.asarray(y_test).ravel()
        y_val   = _np.asarray(y_val).ravel()

    import pandas as _pd
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

    # Feature mapping for counterfactuals 
    cols = X_train.columns.tolist()
    detected_bill  = [c for c in cols if any(k in c.upper() for k in ['BILL', 'BILL_AMT', 'AMT'])]
    detected_limit = next((c for c in cols if any(k in c.upper() for k in ['LIMIT', 'LIMIT_BAL'])), None)
    detected_util  = [c for c in cols if 'UTIL' in c.upper()]

    if detected_bill and detected_limit:
        bill_cols      = detected_bill[:6]
        limit_col      = detected_limit
        util_col_names = detected_util if detected_util else []
    else:
        n = len(cols)
        if n >= 18:
            limit_col      = cols[0]
            util_col_names = cols[6:12]
            bill_cols      = cols[12:18]
        else:
            limit_col      = cols[0]
            bill_cols      = cols[-6:]
            util_col_names = cols[1:7] if n >= 7 else []

    bill_cols      = [c for c in bill_cols      if c in cols]
    util_col_names = [c for c in util_col_names if c in cols]
    if limit_col not in cols:
        limit_col = cols[0]

    # Class imbalance weight 
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(1, pos)
    print("scale_pos_weight:", round(scale_pos_weight, 3))

    xgb = XGBClassifier(
        objective        = 'binary:logistic',
        eval_metric      = 'auc',
        random_state     = seed,
        n_jobs           = -1,
        tree_method      = 'hist',
        scale_pos_weight = scale_pos_weight
    )

    # Hyperparameter tuning
    param_dist = {
        'n_estimators'    : [200, 300, 400, 500],
        'max_depth'       : [3, 4, 5, 6],
        'learning_rate'   : [0.01, 0.03, 0.05, 0.1],
        'subsample'       : [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma'           : [0, 0.1, 0.3],
        'reg_alpha'       : [0, 0.01, 0.1, 1],
        'reg_lambda'      : [1, 2, 5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator           = xgb,
        param_distributions = param_dist,
        n_iter              = 25,
        scoring             = 'f1',
        cv                  = cv,
        verbose             = 1,
        random_state        = seed,
        n_jobs              = -1
    )
    search.fit(X_train, y_train)

    print("\nBest Params:")
    print(search.best_params_)
    print("Best CV F1:", round(search.best_score_, 4))

    best_model = search.best_estimator_

    # Validation results
    y_val_prob = best_model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_prob >= 0.50).astype(int)

    print("\nVALIDATION RESULTS")
    print("Accuracy :", round(accuracy_score(y_val, y_val_pred), 4))
    print("AUC      :", round(roc_auc_score(y_val, y_val_prob), 4))
    print("F1       :", round(f1_score(y_val, y_val_pred), 4))
    print("Precision:", round(precision_score(y_val, y_val_pred), 4))
    print("Recall   :", round(recall_score(y_val, y_val_pred), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
    print("\nClassification Report - Validation")
    print(classification_report(y_val, y_val_pred, digits=4))

    # Threshold tuning
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)
    f1_scores_curve = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_thresh_idx = f1_scores_curve.argmax()
    best_thresh     = thresholds[best_thresh_idx]

    print("Best validation threshold:", round(best_thresh, 4))
    print("Best validation F1       :", round(f1_scores_curve[best_thresh_idx], 4))

    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, precisions[:-1], label='Precision')
    plt.plot(thresholds, recalls[:-1],    label='Recall')
    plt.plot(thresholds, f1_scores_curve[:-1], label='F1')
    plt.axvline(best_thresh, color='red', linestyle='--',
                label=f'Best threshold = {best_thresh:.3f}')
    plt.xlabel('Threshold'); plt.ylabel('Score')
    plt.title('Precision / Recall / F1 vs Threshold (Validation Set)')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # Test evaluation
    y_test_prob     = best_model.predict_proba(X_test)[:, 1]
    y_test_pred     = (y_test_prob >= 0.50).astype(int)
    y_test_pred_opt = (y_test_prob >= best_thresh).astype(int)

    print("\nTEST RESULTS (default threshold = 0.50)")
    print("Accuracy :", round(accuracy_score(y_test, y_test_pred), 4))
    print("AUC      :", round(roc_auc_score(y_test, y_test_prob), 4))
    print("F1       :", round(f1_score(y_test, y_test_pred), 4))
    print("Precision:", round(precision_score(y_test, y_test_pred), 4))
    print("Recall   :", round(recall_score(y_test, y_test_pred), 4))

    print(f"\nTEST RESULTS (optimal threshold = {best_thresh:.4f})")
    print("Accuracy :", round(accuracy_score(y_test, y_test_pred_opt), 4))
    print("AUC      :", round(roc_auc_score(y_test, y_test_prob), 4))
    print("F1       :", round(f1_score(y_test, y_test_pred_opt), 4))
    print("Precision:", round(precision_score(y_test, y_test_pred_opt), 4))
    print("Recall   :", round(recall_score(y_test, y_test_pred_opt), 4))
    print("Brier Score:", round(brier_score_loss(y_test, y_test_prob), 4))
    print("\nClassification Report - Test (Optimal Threshold)")
    print(classification_report(y_test, y_test_pred_opt, digits=4))

    # Calibration curve 
    prob_true, prob_pred = calibration_curve(y_test, y_test_prob, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='XGBoost')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability'); plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve -- XGBoost')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('XGB_calibration_curve.png', dpi=150); plt.show()
    print(f"Brier Score: {brier_score_loss(y_test, y_test_prob):.4f}")

    # Feature importance 
    importance_df = pd.DataFrame({
        'Feature'   : X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nTop 20 Features:")
    print(importance_df.head(20))

    top20 = importance_df.head(20).sort_values('Importance', ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(top20['Feature'], top20['Importance'])
    plt.title('Top 20 XGBoost Feature Importances (Gain)')
    plt.xlabel('Importance'); plt.tight_layout()
    plt.savefig('XGB_feature_importance.png', dpi=150); plt.show()

    # SHAP
    explainer   = shap.TreeExplainer(best_model)
    X_sample    = X_test.sample(500, random_state=seed)
    shap_values = explainer.shap_values(X_sample)
    shap_explanation = shap.Explanation(
        values        = shap_values,
        base_values   = np.full(len(X_sample), explainer.expected_value),
        data          = X_sample.values,
        feature_names = X_sample.columns.tolist()
    )
    shap.plots.bar(shap_explanation,      max_display=20, show=True)
    shap.plots.beeswarm(shap_explanation, max_display=20, show=True)

    # Summary
    print("SUMMARY OF RESULTS\n")
    print(f"Best cross-validated F1: {search.best_score_:.4f}")
    print(f"Best validation threshold: {best_thresh:.4f}\n")
    print("Before threshold tuning (0.50):")
    print(f"  AUC={roc_auc_score(y_test,y_test_prob):.4f}  Acc={accuracy_score(y_test,y_test_pred):.4f}  "
          f"P={precision_score(y_test,y_test_pred):.4f}  R={recall_score(y_test,y_test_pred):.4f}  "
          f"F1={f1_score(y_test,y_test_pred):.4f}")
    print(f"After threshold tuning ({best_thresh:.4f}):")
    print(f"  AUC={roc_auc_score(y_test,y_test_prob):.4f}  Acc={accuracy_score(y_test,y_test_pred_opt):.4f}  "
          f"P={precision_score(y_test,y_test_pred_opt):.4f}  R={recall_score(y_test,y_test_pred_opt):.4f}  "
          f"F1={f1_score(y_test,y_test_pred_opt):.4f}  Brier={brier_score_loss(y_test,y_test_prob):.4f}")

    # Counterfactual Helper 
    def _get_clip_bounds(col, X_tr):
        if col in X_tr.columns:
            return X_tr[col].min(), X_tr[col].max()
        return 0.0, 1.0

    def run_intervention(model, X_original, intervention_fn, label):
        X_modified    = intervention_fn(X_original.copy())
        prob_orig     = model.predict_proba(X_original)[:, 1]
        prob_mod      = model.predict_proba(X_modified)[:, 1]
        delta_p       = prob_mod - prob_orig
        print(f"\n--- {label} ---")
        print(f"Mean ΔP(default)       : {delta_p.mean():.6f}")
        print(f"Mean |ΔP(default)|     : {np.abs(delta_p).mean():.6f}")
        print(f"Clients with reduction : {(delta_p < 0).sum()} / {len(delta_p)}")
        return delta_p

    # Intervention A: reduce bill amounts 
    results_A = {}
    for pct in [0.10, 0.25, 0.50]:
        def intervention_A(X, reduction=pct):
            X_mod = X.copy()
            for col in bill_cols:
                X_mod[col] = X_mod[col] * (1 - reduction)
            if util_col_names:
                for bill, util in zip(bill_cols, util_col_names):
                    X_mod[util] = X_mod[bill] / X_mod[limit_col].replace(0, np.nan)
            for col in bill_cols + util_col_names:
                mn, mx = _get_clip_bounds(col, X_train)
                X_mod[col] = X_mod[col].clip(mn, mx)
            return X_mod
        delta = run_intervention(best_model, X_test, intervention_A,
                                 f"Intervention A - Reduce Bill Amounts by {int(pct*100)}%")
        results_A[pct] = delta

    # Intervention B: increase credit limit 
    results_B = {}
    for pct in [0.10, 0.25, 0.50]:
        def intervention_B(X, increase=pct):
            X_mod = X.copy()
            X_mod[limit_col] = X_mod[limit_col] * (1 + increase)
            if util_col_names:
                for bill, util in zip(bill_cols, util_col_names):
                    X_mod[util] = X_mod[bill] / X_mod[limit_col].replace(0, np.nan)
            for col in [limit_col] + util_col_names:
                mn, mx = _get_clip_bounds(col, X_train)
                X_mod[col] = X_mod[col].clip(mn, mx)
            return X_mod
        delta = run_intervention(best_model, X_test, intervention_B,
                                 f"Intervention B - Increase Credit Limit by {int(pct*100)}%")
        results_B[pct] = delta

    # Intervention C: increase limit + bills (util constant) 
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
                mn, mx = _get_clip_bounds(col, X_train)
                X_mod[col] = X_mod[col].clip(mn, mx)
            return X_mod
        delta = run_intervention(best_model, X_test, intervention_C,
                                 f"Intervention C - Increase Limit+Bills by {int(pct*100)}% (Util Constant)")
        results_C[pct] = delta

    # Summary plot 
    levels = [0.10, 0.25, 0.50]
    labels = ['10%', '25%', '50%']
    mean_A = [np.abs(results_A[p]).mean() for p in levels]
    mean_B = [np.abs(results_B[p]).mean() for p in levels]
    mean_C = [np.abs(results_C[p]).mean() for p in levels]

    plt.figure(figsize=(8, 5))
    plt.plot(labels, mean_A, marker='o', label='A: Reduce Bill Amounts')
    plt.plot(labels, mean_B, marker='s', label='B: Increase Credit Limit')
    plt.plot(labels, mean_C, marker='^', label='C: Increase Limit (Util Constant)')
    plt.xlabel('Intervention Level'); plt.ylabel('Mean |ΔP(default)|')
    plt.title('XGBoost: Mean Change in Predicted Default Probability\nby Intervention Type and Level')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('XGB_counterfactual_comparison.png', dpi=150); plt.show()

    # Pairwise Wilcoxon tests
    def safe_wilcoxon(d1, d2, label):
        try:
            stat, p = wilcoxon(d1, d2, alternative='two-sided')
            print(f"  Wilcoxon {label}: W={stat:.1f}, p={p:.4e}")
            return float(p)
        except ValueError as e:
            print(f"  Wilcoxon {label}: failed ({e})")
            return np.nan

    print("\n=== Pairwise Wilcoxon Tests (XGBoost) ===")
    for pct in levels:
        print(f"\n-- {int(pct*100)}% level --")
        safe_wilcoxon(results_A[pct], results_B[pct], "A vs B")
        safe_wilcoxon(results_A[pct], results_C[pct], "A vs C")
        safe_wilcoxon(results_B[pct], results_C[pct], "B vs C")

    # Client segmentation 
    baseline_probs = best_model.predict_proba(X_test)[:, 1]
    high_risk_mask = baseline_probs > 0.5
    low_risk_mask  = baseline_probs < 0.4

    print(f"\nHigh-risk clients (P > 0.5): {high_risk_mask.sum()}")
    print(f"Low-risk clients  (P < 0.4): {low_risk_mask.sum()}")

    for mask, group_label in [(high_risk_mask, 'High-Risk'), (low_risk_mask, 'Low-Risk')]:
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
                mn, mx = _get_clip_bounds(col, X_train)
                X_mod[col] = X_mod[col].clip(mn, mx)
            return X_mod

        def int_B_seg(X):
            X_mod = X.copy()
            X_mod[limit_col] = X_mod[limit_col] * 1.25
            if util_col_names:
                for bill, util in zip(bill_cols, util_col_names):
                    X_mod[util] = X_mod[bill] / X_mod[limit_col].replace(0, np.nan)
            for col in [limit_col] + util_col_names:
                mn, mx = _get_clip_bounds(col, X_train)
                X_mod[col] = X_mod[col].clip(mn, mx)
            return X_mod

        dA = run_intervention(best_model, X_group, int_A_seg,
                              f"{group_label} - Intervention A (25% bill reduction)")
        dB = run_intervention(best_model, X_group, int_B_seg,
                              f"{group_label} - Intervention B (25% limit increase)")
        safe_wilcoxon(dA, dB, "A vs B")

    # Save model 
    from joblib import dump
    dump(best_model, f'xgb_model_seed{seed}.joblib')
    print(f"\nXGBoost model saved to xgb_model_seed{seed}.joblib")

    return best_model, X_train, X_test, y_train, y_test
