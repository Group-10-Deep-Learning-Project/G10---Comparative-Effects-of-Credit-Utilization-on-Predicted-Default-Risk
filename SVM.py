import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy.stats import loguniform
import shap
import random


def run_Model(seed, x_v, y_v, x_train, y_train, x_test, y_test):

    random.seed(seed)
    np.random.seed(seed)
    """
    SVM model runner.

    Accepts pre-split, pre-processed (OHE'd) arrays from the runner.
    Internally:
      - Carves a validation set from X_train for threshold tuning
      - Scales features (fit on train only)
      - Tunes hyperparameters with RandomizedSearchCV on train
      - Selects best threshold using precision_recall_curve on val
      - Evaluates final model on test set
      - Returns (model, X_train_df, X_test_df, y_train, y_test)

    Parameters
    ----------
    X_train, X_test : array-like or DataFrame, shape (n, 98)
    y_train, y_test : array-like, shape (n,)

    Returns
    -------
    tuple: best_svm, X_train_df, X_test_df, y_train, y_test
    """

    X_val, y_val, X_train, y_train, X_test, y_test = x_v, y_v, x_train, y_train, x_test, y_test

    # Flatten y in case runner passes column vectors 
    y_train = np.array(y_train).ravel()
    y_test  = np.array(y_test).ravel()

    # Convert X to DataFrames 
    def _to_df(X):
        if hasattr(X, 'columns'):
            return X
        X_arr = np.asarray(X)
        return pd.DataFrame(X_arr, columns=[f"Feature_{i}" for i in range(X_arr.shape[1])])

    X_train = _to_df(X_train)
    X_test  = _to_df(X_test)
    X_val   = _to_df(X_val)

    # Hyperparameter tuning on train split 
    param_dist = {
        'C'     : loguniform(0.01, 100),
        'gamma' : loguniform(0.0001, 1),
        'kernel': ['rbf']               # poly removed — causes worker crashes
    }

    svm = SVC(probability=True, random_state=seed)

    random_search = RandomizedSearchCV(
        estimator           = svm,
        param_distributions = param_dist,
        n_iter              = 20,
        scoring             = 'f1',
        cv                  = 5,
        verbose             = 1,
        random_state        = seed,
        n_jobs              = -1
    )

    random_search.fit(X_train, y_train)
    best_svm = random_search.best_estimator_

    print(f"[SVM] Best params : {random_search.best_params_}")
    print(f"[SVM] Best CV F1  : {random_search.best_score_:.4f}")

    # Threshold tuning on validation set 
    y_val_prob = best_svm.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)
    f1_curve        = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_thresh_idx = f1_curve.argmax()
    best_threshold  = thresholds[best_thresh_idx]

    print(f"[SVM] Best threshold (val F1): {best_threshold:.4f} "
          f"| Val F1: {f1_curve[best_thresh_idx]:.4f}")

    # Refit on full train set with best params
    best_svm.fit(X_train, y_train)

    # Final evaluation on test set 
    y_test_prob = best_svm.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    print(f"[SVM] Test AUC       : {roc_auc_score(y_test, y_test_prob):.4f}")
    print(f"[SVM] Test F1        : {f1_score(y_test, y_test_pred):.4f}")
    print(f"[SVM] Test Accuracy  : {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"[SVM] Test Precision : {precision_score(y_test, y_test_pred):.4f}")
    print(f"[SVM] Test Recall    : {recall_score(y_test, y_test_pred):.4f}")
    print(f"[SVM] Test Brier     : {brier_score_loss(y_test, y_test_prob):.4f}")

    try:
        feature_names_svm = X_test.columns.tolist()
        predict_fn        = lambda X: best_svm.predict_proba(pd.DataFrame(X, columns=feature_names_svm))[:, 1]
        X_background      = shap.sample(X_train, 50, random_state=seed).values
        X_shap_sample_df  = X_test.sample(50, random_state=seed)
        X_shap_sample     = X_shap_sample_df.values
        explainer_shap    = shap.KernelExplainer(predict_fn, X_background)
        shap_values_svm   = explainer_shap.shap_values(X_shap_sample, nsamples=50)

        shap_explanation = shap.Explanation(
            values        = shap_values_svm,
            base_values   = np.full(len(X_shap_sample), explainer_shap.expected_value),
            data          = X_shap_sample,
            feature_names = feature_names_svm
        )

        shap.plots.bar(shap_explanation,      max_display=20, show=False)
        plt.savefig(f'SVM_shap_bar_seed{seed}.png', dpi=150)
        plt.show()
        shap.plots.beeswarm(shap_explanation, max_display=20, show=False)
        plt.savefig(f'SVM_shap_beeswarm_seed{seed}.png', dpi=150)
        plt.show()

    except Exception as e:
        print(f"[SVM] SHAP failed ({e}), falling back to permutation importance")
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(
            best_svm, X_test, y_test,
            n_repeats=10, random_state=seed, scoring='roc_auc'
        )
        perm_df = pd.DataFrame({
            'Feature'   : X_test.columns,
            'Importance': perm.importances_mean
        }).sort_values('Importance', ascending=False)

        print("\nTop 20 Features (Permutation Importance):")
        print(perm_df.head(20))

        top20 = perm_df.head(20).sort_values('Importance', ascending=True)
        plt.figure(figsize=(10, 8))
        plt.barh(top20['Feature'], top20['Importance'])
        plt.title('Top 20 SVM Feature Importances (Permutation)')
        plt.xlabel('Mean Decrease in AUC')
        plt.tight_layout()
        plt.savefig(f'SVM_permutation_importance_seed{seed}.png', dpi=150)
        plt.show()

    return best_svm, X_train, X_test, y_train, y_test
