"""
counterfactual.py

Runs counterfactual interventions A, B, C on a fitted model and
performs all three pairwise Wilcoxon signed-rank tests at each level.
Also runs client segmentation analysis (high-risk vs low-risk).

Interventions:
  A: Reduce BILL_AMT1-6 by X%, hold LIMIT_BAL constant. Recompute utilization.
  B: Increase LIMIT_BAL by X%, hold bill amounts constant. Recompute utilization.
  C: Increase LIMIT_BAL and BILL_AMT1-6 proportionally so utilization is unchanged.
     Isolates the pure credit limit effect.

Wilcoxon tests (paired, two-sided) at each level:
  A vs B, A vs C, B vs C

Segmentation: at 25% level, compare A vs B for:
  High-risk clients (baseline P > 0.5)
  Low-risk clients  (baseline P < 0.4)
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import warnings


def run_counterfactual_and_tests(model, X_test, X_train, bill_cols, util_col_names, limit_col,
                                 levels=(0.10, 0.25, 0.50)):
    """
    Run interventions A, B, C with all pairwise Wilcoxon tests and segmentation.

    Returns:
        cf_results  : dict keyed by level, each containing deltas, means, and p-values
        seg_results : dict with segmentation results at 25% level
    """

    if not isinstance(X_test, pd.DataFrame) or not isinstance(X_train, pd.DataFrame):
        raise ValueError("X_test and X_train must be pandas DataFrames.")

    # Helper: recompute utilization columns 
    def recompute_utils(df):
        df = df.copy()
        denom = df[limit_col].replace(0, np.nan)
        for bill, util in zip(bill_cols, util_col_names):
            df[util] = df[bill] / denom
        return df

    # Helper: safe Wilcoxon 
    def safe_wilcoxon(d1, d2):
        try:
            _, p = wilcoxon(d1, d2, alternative='two-sided')
            return float(p)
        except ValueError as e:
            warnings.warn(f"Wilcoxon test failed: {e}")
            return np.nan

    # Helper: clip to training range 
    def clip_cols(df, cols_to_clip):
        df = df.copy()
        for col in cols_to_clip:
            if col in X_train.columns and col in df.columns:
                df[col] = df[col].clip(X_train[col].min(), X_train[col].max())
        return df

    # Ensure X_train util cols are populated for clipping reference
    try:
        X_train = recompute_utils(X_train)
    except Exception:
        pass

    prob_original = model.predict_proba(X_test)[:, 1]

    cf_results = {}

    for pct in levels:

        # Intervention A
        X_A = X_test.copy()
        for col in bill_cols:
            X_A[col] = X_A[col] * (1 - pct)
        X_A = recompute_utils(X_A)
        X_A = clip_cols(X_A, bill_cols + util_col_names)
        prob_A  = model.predict_proba(X_A)[:, 1]
        delta_A = prob_A - prob_original

        # Intervention B
        X_B = X_test.copy()
        X_B[limit_col] = X_B[limit_col] * (1 + pct)
        X_B = recompute_utils(X_B)
        X_B = clip_cols(X_B, [limit_col] + util_col_names)
        prob_B  = model.predict_proba(X_B)[:, 1]
        delta_B = prob_B - prob_original

        # Intervention C 
        X_C = X_test.copy()
        X_C[limit_col] = X_C[limit_col] * (1 + pct)
        for col in bill_cols:
            X_C[col] = X_C[col] * (1 + pct)
        X_C = recompute_utils(X_C)
        X_C = clip_cols(X_C, [limit_col] + bill_cols + util_col_names)
        prob_C  = model.predict_proba(X_C)[:, 1]
        delta_C = prob_C - prob_original

        # Pairwise Wilcoxon tests
        p_AB = safe_wilcoxon(delta_A, delta_B)
        p_AC = safe_wilcoxon(delta_A, delta_C)
        p_BC = safe_wilcoxon(delta_B, delta_C)

        cf_results[pct] = {
            'delta_A'       : delta_A,
            'delta_B'       : delta_B,
            'delta_C'       : delta_C,
            'mean_A'        : float(delta_A.mean()),
            'mean_B'        : float(delta_B.mean()),
            'mean_C'        : float(delta_C.mean()),
            'mean_abs_A'    : float(np.abs(delta_A).mean()),
            'mean_abs_B'    : float(np.abs(delta_B).mean()),
            'mean_abs_C'    : float(np.abs(delta_C).mean()),
            'n_reduced_A'   : int((delta_A < 0).sum()),
            'n_reduced_B'   : int((delta_B < 0).sum()),
            'n_reduced_C'   : int((delta_C < 0).sum()),
            'n_total'       : len(delta_A),
            'wilcoxon_p_AB' : p_AB,
            'wilcoxon_p_AC' : p_AC,
            'wilcoxon_p_BC' : p_BC,
        }

    # Print summary 
    print("\nCounterfactual A / B / C — Summary")
    print(f"{'Lvl':>4} | {'mean_A':>9} | {'mean_B':>9} | {'mean_C':>9} | "
          f"{'|A|':>8} | {'|B|':>8} | {'|C|':>8} | "
          f"{'p(A-B)':>10} | {'p(A-C)':>10} | {'p(B-C)':>10}")
    for pct in levels:
        r = cf_results[pct]
        print(f"{int(pct*100):>4}% | {r['mean_A']:+.6f} | {r['mean_B']:+.6f} | {r['mean_C']:+.6f} | "
              f"{r['mean_abs_A']:.6f} | {r['mean_abs_B']:.6f} | {r['mean_abs_C']:.6f} | "
              f"{r['wilcoxon_p_AB']:>10.4e} | {r['wilcoxon_p_AC']:>10.4e} | {r['wilcoxon_p_BC']:>10.4e}")

    # Segmentation at 25% level 
    seg_results = {}
    seg_pct = 0.25

    high_mask = prob_original > 0.5
    low_mask  = prob_original < 0.4

    print(f"\nSegmentation Analysis (at {int(seg_pct*100)}% intervention level)")
    print(f"High-risk clients (P > 0.5): {high_mask.sum()}")
    print(f"Low-risk  clients (P < 0.4): {low_mask.sum()}")

    # Recompute A and B at 25% for segmentation
    X_A25 = X_test.copy()
    for col in bill_cols:
        X_A25[col] = X_A25[col] * (1 - seg_pct)
    X_A25 = recompute_utils(X_A25)
    X_A25 = clip_cols(X_A25, bill_cols + util_col_names)

    X_B25 = X_test.copy()
    X_B25[limit_col] = X_B25[limit_col] * (1 + seg_pct)
    X_B25 = recompute_utils(X_B25)
    X_B25 = clip_cols(X_B25, [limit_col] + util_col_names)

    prob_A25 = model.predict_proba(X_A25)[:, 1]
    prob_B25 = model.predict_proba(X_B25)[:, 1]

    for mask, group in [(high_mask, 'high_risk'), (low_mask, 'low_risk')]:
        if mask.sum() == 0:
            continue

        mask_arr = np.array(mask)
        dA_group = (prob_A25 - prob_original)[mask_arr]
        dB_group = (prob_B25 - prob_original)[mask_arr]
        p_seg    = safe_wilcoxon(dA_group, dB_group)

        label = 'High-Risk' if group == 'high_risk' else 'Low-Risk'
        print(f"\n  {label} (n={mask.sum()}):")
        print(f"    Int A mean ΔP : {dA_group.mean():+.6f}  |ΔP|: {np.abs(dA_group).mean():.6f}")
        print(f"    Int B mean ΔP : {dB_group.mean():+.6f}  |ΔP|: {np.abs(dB_group).mean():.6f}")
        print(f"    Wilcoxon A vs B p-value: {p_seg:.4e}")

        seg_results[group] = {
            'n'             : int(mask.sum()),
            'mean_A'        : float(dA_group.mean()),
            'mean_B'        : float(dB_group.mean()),
            'mean_abs_A'    : float(np.abs(dA_group).mean()),
            'mean_abs_B'    : float(np.abs(dB_group).mean()),
            'wilcoxon_p_AB' : p_seg,
        }

    return cf_results, seg_results


if __name__ == '__main__':
    pass
