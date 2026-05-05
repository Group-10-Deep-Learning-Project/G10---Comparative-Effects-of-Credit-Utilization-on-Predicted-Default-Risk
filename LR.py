# %%
# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve,
    confusion_matrix, classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve
import shap
import random


def run_Model(seed, x_v, y_v, x_train, y_train, x_test, y_test):
    
    random.seed(seed)
    np.random.seed(seed)

    X_val, y_val, X_train, y_train, X_test, y_test = x_v, y_v, x_train, y_train, x_test, y_test

    y_train = np.array(y_train).ravel()

    y_test  = np.array(y_test).ravel()


    # Logistic Regression Baseline
    # Kept simple intentionally, this is the baseline, not a tuned model

    log_model = LogisticRegression(
        penalty      = 'l2',
        C            = 1.0,
        class_weight = 'balanced',
        solver       = 'lbfgs',
        max_iter     = 1000,
        random_state = seed
    )

    log_model.fit(X_train, y_train)
    print("Model fitted.")

    # %%
    # Validation Set Evaluation

    y_val_prob = log_model.predict_proba(X_val)[:, 1]
    y_val_pred = log_model.predict(X_val)

    print("=== LR Baseline -- Validation Set ===")
    print(f"Accuracy : {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
    print(f"Recall   : {recall_score(y_val, y_val_pred):.4f}")
    print(f"F1 Score : {f1_score(y_val, y_val_pred):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_val, y_val_prob):.4f}")

    # %%
    # Threshold Tuning on Validation Set

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)
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
    plt.title('Precision / Recall / F1 vs Threshold (Validation Set)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # %%
    # Final Evaluation on Test Set
    # Brier score added to match other models

    y_test_prob = log_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    print("=== Logistic Regression Baseline -- Test Set ===")
    print(f"Threshold  : {best_threshold:.4f}")
    print(f"Accuracy   : {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Precision  : {precision_score(y_test, y_test_pred):.4f}")
    print(f"Recall     : {recall_score(y_test, y_test_pred):.4f}")
    print(f"F1 Score   : {f1_score(y_test, y_test_pred):.4f}")
    print(f"ROC AUC    : {roc_auc_score(y_test, y_test_prob):.4f}")
    print(f"Brier Score: {brier_score_loss(y_test, y_test_prob):.4f}")
    print()
    print(classification_report(y_test, y_test_pred, digits=4))

    # %%
    # Calibration Curve

    prob_true, prob_pred = calibration_curve(y_test, y_test_prob, n_bins=10)

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Logistic Regression')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve -- Logistic Regression')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('LR_calibration_curve.png', dpi=150)
    plt.show()

    print(f"Brier Score: {brier_score_loss(y_test, y_test_prob):.4f}")

    # %%
    # SHAP Feature Importance
    if hasattr(X_train, "columns"):
        feature_names = X_train.columns.tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    explainer   = shap.LinearExplainer(log_model, X_train_df)
    X_sample    = X_test_df.sample(500, random_state=seed)
    shap_values = explainer.shap_values(X_sample)

    shap_explanation = shap.Explanation(
        values        = shap_values,
        base_values   = np.full(len(X_sample), explainer.expected_value),
        data          = X_sample.values,
        feature_names = X_sample.columns.tolist()
    )

    shap.plots.bar(shap_explanation,      max_display=20, show=True)
    shap.plots.beeswarm(shap_explanation, max_display=20, show=True)

    # Return the trained model and DataFrames for downstream analysis
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df  = pd.DataFrame(X_test,  columns=feature_names)
    return log_model, X_train_df, X_test_df, y_train, y_test


