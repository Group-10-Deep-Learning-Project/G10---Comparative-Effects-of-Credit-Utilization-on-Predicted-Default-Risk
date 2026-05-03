# %%
# XGBoost Model

## Humaid Billoo

### V2 (Fixed)

# %%
### Setup

# %%
# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    X_val, y_val, X_train, y_train, X_test, y_test = x_v, y_v, x_train, y_train, x_test, y_test

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    print("scale_pos_weight:", round(scale_pos_weight, 3))



    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        random_state=99,
        n_jobs=-1,
        tree_method='hist',
        scale_pos_weight=scale_pos_weight
    )

    ### Hyperparameter Tuning

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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=28)

    search = RandomizedSearchCV(
        estimator           = xgb,
        param_distributions = param_dist,
        n_iter              = 25,
        scoring             = 'f1',
        cv                  = cv,
        verbose             = 1,
        random_state        = 99,
        n_jobs              = -1
    )

    ### Fit Best Model

    search.fit(X_train, y_train)

    print("\nBest Params:")
    print(search.best_params_)
    print("Best CV F1:", round(search.best_score_, 4))

    best_model = search.best_estimator_

    ### Validation Predictions

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


    ### Threshold Tuning on Validation Set

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
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision / Recall / F1 vs Threshold (Validation Set)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    ### Test Evaluation -- Tuned Threshold

    y_test_prob     = best_model.predict_proba(X_test)[:, 1]
    y_test_pred     = (y_test_prob >= 0.50).astype(int)
    y_test_pred_opt = (y_test_prob >= best_thresh).astype(int)

    print("\nTEST RESULTS (default threshold = 0.50)")
    print("Accuracy :", round(accuracy_score(y_test, y_test_pred), 4))
    print("AUC      :", round(roc_auc_score(y_test, y_test_prob), 4))
    print("F1       :", round(f1_score(y_test, y_test_pred), 4))
    print("Precision:", round(precision_score(y_test, y_test_pred), 4))
    print("Recall   :", round(recall_score(y_test, y_test_pred), 4))

    print("\nTEST RESULTS (optimal threshold = {:.4f})".format(best_thresh))
    print("Accuracy :", round(accuracy_score(y_test, y_test_pred_opt), 4))
    print("AUC      :", round(roc_auc_score(y_test, y_test_prob), 4))
    print("F1       :", round(f1_score(y_test, y_test_pred_opt), 4))
    print("Precision:", round(precision_score(y_test, y_test_pred_opt), 4))
    print("Recall   :", round(recall_score(y_test, y_test_pred_opt), 4))
    print("Brier Score:", round(brier_score_loss(y_test, y_test_prob), 4))
    print("\nClassification Report - Test (Optimal Threshold)")
    print(classification_report(y_test, y_test_pred_opt, digits=4))

    ### Calibration Curve

    prob_true, prob_pred = calibration_curve(y_test, y_test_prob, n_bins=10)

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='XGBoost')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve -- XGBoost')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('XGB_calibration_curve.png', dpi=150)
    plt.show()

    print(f"Brier Score: {brier_score_loss(y_test, y_test_prob):.4f}")

    # %%
    ### Feature Importance (Gain-Based)

    # %%

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
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('XGB_feature_importance.png', dpi=150)
    plt.show()

    # %%
    ### SHAP Feature Importance

    # %%

    explainer        = shap.TreeExplainer(best_model)
    X_sample         = X_test.sample(500, random_state=42)
    shap_values      = explainer.shap_values(X_sample)

    shap_explanation = shap.Explanation(
        values        = shap_values,
        base_values   = np.full(len(X_sample), explainer.expected_value),
        data          = X_sample.values,
        feature_names = X_sample.columns.tolist()
    )

    shap.plots.bar(shap_explanation,      max_display=20, show=True)
    shap.plots.beeswarm(shap_explanation, max_display=20, show=True)

    # %%
    ### Results Summary

    # %%

    print("SUMMARY OF RESULTS\n")

    print(f"Best cross-validated F1 from randomized search: {search.best_score_:.4f}")
    print(f"Best validation threshold based on F1        : {best_thresh:.4f}\n")

    print("Before threshold tuning (default threshold = 0.50):")
    print(f"  Test AUC       : {roc_auc_score(y_test, y_test_prob):.4f}")
    print(f"  Test Accuracy  : {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"  Test Precision : {precision_score(y_test, y_test_pred):.4f}")
    print(f"  Test Recall    : {recall_score(y_test, y_test_pred):.4f}")
    print(f"  Test F1        : {f1_score(y_test, y_test_pred):.4f}\n")

    print(f"After threshold tuning (threshold = {best_thresh:.4f}):")
    print(f"  Test AUC       : {roc_auc_score(y_test, y_test_prob):.4f}")
    print(f"  Test Accuracy  : {accuracy_score(y_test, y_test_pred_opt):.4f}")
    print(f"  Test Precision : {precision_score(y_test, y_test_pred_opt):.4f}")
    print(f"  Test Recall    : {recall_score(y_test, y_test_pred_opt):.4f}")
    print(f"  Test F1        : {f1_score(y_test, y_test_pred_opt):.4f}")
    print(f"  Brier Score    : {brier_score_loss(y_test, y_test_prob):.4f}")


