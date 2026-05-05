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
      - Returns metrics dict

    Parameters
    ----------
    X_train, X_test : array-like or DataFrame, shape (n, 98)
    y_train, y_test : array-like, shape (n,)

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc,
                    brier, best_threshold
    """

    X_val, y_val, X_train, y_train, X_test, y_test = x_v, y_v, x_train, y_train, x_test, y_test

    # ── Flatten y in case runner passes column vectors ───────
    y_train = np.array(y_train).ravel()
    y_test  = np.array(y_test).ravel()

    # ── Hyperparameter tuning on train split ─────────────────
    param_dist = {
        'C'     : loguniform(0.01, 100),
        'gamma' : loguniform(0.0001, 1),
        'kernel': ['rbf', 'poly']
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

    # ── Threshold tuning on validation set ───────────────────
    y_val_prob = best_svm.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)
    f1_curve        = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_thresh_idx = f1_curve.argmax()
    best_threshold  = thresholds[best_thresh_idx]

    print(f"[SVM] Best threshold (val F1): {best_threshold:.4f} "
          f"| Val F1: {f1_curve[best_thresh_idx]:.4f}")

    # ── Refit on full train set with best params ─────────────
    best_svm.fit(X_train, y_train)

    # ── Final evaluation on test set ─────────────────────────
    y_test_prob = best_svm.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    print(f"[SVM] Test AUC : {roc_auc_score(y_test, y_test_prob):.4f}")
    print(f"[SVM] Test F1  : {f1_score(y_test, y_test_pred):.4f}")

    return {
        "accuracy"       : accuracy_score(y_test, y_test_pred),
        "precision"      : precision_score(y_test, y_test_pred),
        "recall"         : recall_score(y_test, y_test_pred),
        "f1"             : f1_score(y_test, y_test_pred),
        "roc_auc"        : roc_auc_score(y_test, y_test_prob),
        "brier"          : brier_score_loss(y_test, y_test_prob),
        "best_threshold" : best_threshold
    }

    # Add return of estimator and dataframes for counterfactual usage
    import pandas as _pd
    try:
        X_train_df = _pd.DataFrame(X_train)
        X_test_df = _pd.DataFrame(X_test)
    except Exception:
        X_train_df = X_train
        X_test_df = X_test

    # Return best_svm for reuse (estimator, X_train_df, X_test_df, y_train, y_test)
    return best_svm, X_train_df, X_test_df, y_train, y_test
