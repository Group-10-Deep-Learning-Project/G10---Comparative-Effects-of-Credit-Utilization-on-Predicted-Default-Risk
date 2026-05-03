"""
counterfactual.py

Utility to run counterfactual interventions A/B/C on a fitted model and
perform paired Wilcoxon signed-rank tests comparing per-client ΔP for
Intervention A vs Intervention B at multiple levels (10%, 25%, 50%).

Expectations:
- model: sklearn-like estimator with predict_proba(X) -> ndarray (n,2)
- X_test, X_train: pandas.DataFrame (columns must include bill_cols, util_col_names, limit_col)
- bill_cols: list of bill column names (e.g. ['BILL_AMT1', ...])
- util_col_names: list of utilization column names (e.g. ['UTIL1', ...])
- limit_col: name of credit limit column

Returns dict with per-level results including mean deltas and Wilcoxon p-values.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import warnings


def run_counterfactual_and_tests(model, X_test, X_train, bill_cols, util_col_names, limit_col,
                                 levels=(0.10, 0.25, 0.50)):
    """Run interventions A (reduce bills) and B (increase limit) and
    perform paired Wilcoxon signed-rank tests comparing per-client delta P.

    Returns a dict keyed by level with entries:
      {
        'delta_A': ndarray,   # per-client delta (modified - original)
        'delta_B': ndarray,
        'mean_abs_A': float,
        'mean_abs_B': float,
        'mean_A': float,
        'mean_B': float,
        'wilcoxon_p': float or np.nan
      }
    """

    # Input validation
    if not isinstance(X_test, pd.DataFrame) or not isinstance(X_train, pd.DataFrame):
        raise ValueError("X_test and X_train must be pandas DataFrame objects with column names.")

    results = {}

    # Helper: safely compute utilizations (do NOT add new 'UTIL_avg' column to avoid mismatched model features)
    def recompute_utils(df):
        df = df.copy()
        # avoid division by zero
        denom = df[limit_col].replace(0, np.nan)
        for bill, util in zip(bill_cols, util_col_names):
            df[util] = df[bill] / denom
        # do not assign UTIL_avg as a permanent column (keep feature set identical to training)
        return df

    # Ensure X_train has util-derived columns for clipping reference
    try:
        X_train = recompute_utils(X_train)
    except Exception:
        # If recompute fails, continue and clipping will be skipped for missing columns
        pass

    # Original probabilities
    prob_original = model.predict_proba(X_test)[:, 1]

    for pct in levels:
        # Intervention A: reduce bill amounts
        X_A = X_test.copy()
        for col in bill_cols:
            X_A[col] = X_A[col] * (1 - pct)
        X_A = recompute_utils(X_A)
        # Clip to training min/max to keep values in-distribution
        for col in bill_cols + util_col_names:
            if col in X_train.columns and col in X_A.columns:
                minv = X_train[col].min()
                maxv = X_train[col].max()
                X_A[col] = X_A[col].clip(minv, maxv)

        prob_A = model.predict_proba(X_A)[:, 1]
        delta_A = prob_A - prob_original

        # Intervention B: increase credit limit
        X_B = X_test.copy()
        X_B[limit_col] = X_B[limit_col] * (1 + pct)
        X_B = recompute_utils(X_B)
        for col in [limit_col] + util_col_names:
            if col in X_train.columns and col in X_B.columns:
                minv = X_train[col].min()
                maxv = X_train[col].max()
                X_B[col] = X_B[col].clip(minv, maxv)

        prob_B = model.predict_proba(X_B)[:, 1]
        delta_B = prob_B - prob_original

        # Paired Wilcoxon signed-rank test: delta_A vs delta_B per client
        p_value = np.nan
        try:
            # Need differences delta = delta_A - delta_B
            stat, p_value = wilcoxon(delta_A, delta_B, alternative='two-sided')
        except ValueError as e:
            # This can happen if all differences are zero or sample too small
            warnings.warn(f"Wilcoxon test failed at level {pct}: {e}")
            p_value = np.nan

        results[pct] = {
            'delta_A': delta_A,
            'delta_B': delta_B,
            'mean_abs_A': np.abs(delta_A).mean(),
            'mean_abs_B': np.abs(delta_B).mean(),
            'mean_A': delta_A.mean(),
            'mean_B': delta_B.mean(),
            'wilcoxon_p': p_value
        }

    # Print summary table
    print("\nCounterfactual A vs B (paired Wilcoxon signed-rank test)")
    print("Level | mean_A | mean_B | mean|A| | mean|B| | p-value")
    for pct in results:
        r = results[pct]
        print(f"{int(pct*100):>5}% | {r['mean_A']:+.6f} | {r['mean_B']:+.6f} | {r['mean_abs_A']:.6f} | {r['mean_abs_B']:.6f} | {r['wilcoxon_p']}")

    return results


if __name__ == '__main__':
    # Example usage snippet (won't run as-is; provided for reference):
    # from joblib import load
    # model = load('rf_model.joblib')
    # X_train = pd.read_csv('X_train.csv')
    # X_test  = pd.read_csv('X_test.csv')
    # bill_cols = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
    # util_col_names = ['UTIL1','UTIL2','UTIL3','UTIL4','UTIL5','UTIL6']
    # limit_col = 'LIMIT_BAL'
    # run_counterfactual_and_tests(model, X_test, X_train, bill_cols, util_col_names, limit_col)
    pass
