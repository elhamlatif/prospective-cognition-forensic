"""
Main Analyses
Project: Quantitative Analysis of Prospective Cognition, Intrusive Imagery,
         and Affective Symptoms in a Forensic Sample

Author: Elham Latif
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def run_correlations(data):
    print("=" * 80)
    print("CORRELATIONS WITH DEPRESSION SCORE")
    print("=" * 80)

    vars_to_test = [
        'PIT_Pos_Vivid', 'PIT_Neg_Vivid',
        'PIT_Pos_Likelihood', 'PIT_Neg_Likelihood',
        'IFES_Negative', 'Intrusive_Visual', 'Intrusive_Verbal',
        'HIT_Negative', 'PSIQ_Total'
    ]

    results = []
    for var in vars_to_test:
        temp = data[['Depression_Score', var]].dropna()
        if len(temp) <= 15:
            continue
        r, p = stats.pearsonr(temp['Depression_Score'], temp[var])
        results.append({'outcome': 'Depression_Score', 'var': var, 'r': r, 'p': p, 'n': len(temp)})

    for var in vars_to_test:
        temp = data[['Anxiety_Score', var]].dropna()
        if len(temp) <= 15:
            continue
        r, p = stats.pearsonr(temp['Anxiety_Score'], temp[var])
        results.append({'outcome': 'Anxiety_Score', 'var': var, 'r': r, 'p': p, 'n': len(temp)})

    # apply FDR (Benjamini-Hochberg) correction across all correlation tests run
    raw_p = [row['p'] for row in results]
    reject, p_adj, _, _ = multipletests(raw_p, alpha=0.05, method='fdr_bh')
    for row, p_c, rej in zip(results, p_adj, reject):
        row['p_fdr'] = p_c
        row['sig_fdr'] = sig_stars(p_c)

    dep_rows = [r for r in results if r['outcome'] == 'Depression_Score']
    anx_rows = [r for r in results if r['outcome'] == 'Anxiety_Score']

    print(f"{'variable':25s} {'r':>7s} {'p':>8s} {'p_fdr':>8s} {'n':>5s}")
    for row in dep_rows:
        print(f"{row['var']:25s} {row['r']:7.3f} {row['p']:8.4f} {row['p_fdr']:8.4f} {row['n']:5d}  {row['sig_fdr']}")

    print("\n" + "=" * 80)
    print("CORRELATIONS WITH ANXIETY SCORE")
    print("=" * 80)
    print(f"{'variable':25s} {'r':>7s} {'p':>8s} {'p_fdr':>8s} {'n':>5s}")
    for row in anx_rows:
        print(f"{row['var']:25s} {row['r']:7.3f} {row['p']:8.4f} {row['p_fdr']:8.4f} {row['n']:5d}  {row['sig_fdr']}")

    print("\nNote: p_fdr = Benjamini-Hochberg corrected p-value across all correlation")
    print("tests reported above (14 tests total). Use p_fdr, not raw p, to judge significance.")

    return results


def f_change_test(model_small, model_big, n):
    """Test whether adding predictors in model_big significantly improves fit over model_small."""
    df1 = model_big.df_model - model_small.df_model
    df2 = n - model_big.df_model - 1
    r2_small, r2_big = model_small.rsquared, model_big.rsquared
    f_stat = ((r2_big - r2_small) / df1) / ((1 - r2_big) / df2)
    p_val = stats.f.sf(f_stat, df1, df2)
    return f_stat, df1, df2, p_val


def hierarchical_regression(data):
    print("\n\n" + "=" * 80)
    print("HIERARCHICAL REGRESSION: Predicting Depression_Score")
    print("=" * 80)

    reg_data = data[['Depression_Score', 'Age', 'PSIQ_Total',
                     'PIT_Pos_Vivid', 'PIT_Neg_Vivid', 'IFES_Negative']].dropna()
    n = len(reg_data)
    print(f"N for regression: {n}")

    y = reg_data['Depression_Score']

    # step 1 - age only, baseline
    X1 = sm.add_constant(reg_data[['Age']])
    m1 = sm.OLS(y, X1).fit()
    print("\nStep 1 (Age only):")
    print(f"  R2 = {m1.rsquared:.3f}, Adj.R2 = {m1.rsquared_adj:.3f}")
    print(f"  F({int(m1.df_model)},{int(m1.df_resid)}) = {m1.fvalue:.2f}, p = {m1.f_pvalue:.4f}")

    # step 2 - add sensory imagery
    X2 = sm.add_constant(reg_data[['Age', 'PSIQ_Total']])
    m2 = sm.OLS(y, X2).fit()
    f2, df1_2, df2_2, p2 = f_change_test(m1, m2, n)
    print("\nStep 2 (+ PSIQ_Total):")
    print(f"  R2 = {m2.rsquared:.3f}, Adj.R2 = {m2.rsquared_adj:.3f}, deltaR2 = {m2.rsquared - m1.rsquared:.3f}")
    print(f"  F-change({int(df1_2)},{int(df2_2)}) = {f2:.2f}, p = {p2:.4f}")

    # step 3 - add prospective imagery + intrusion measures
    X3 = sm.add_constant(reg_data[['Age', 'PSIQ_Total', 'PIT_Pos_Vivid',
                                   'PIT_Neg_Vivid', 'IFES_Negative']])
    m3 = sm.OLS(y, X3).fit()
    f3, df1_3, df2_3, p3 = f_change_test(m2, m3, n)
    print("\nStep 3 (+ PIT & IFES):")
    print(f"  R2 = {m3.rsquared:.3f}, Adj.R2 = {m3.rsquared_adj:.3f}, deltaR2 = {m3.rsquared - m2.rsquared:.3f}")
    print(f"  F-change({int(df1_3)},{int(df2_3)}) = {f3:.2f}, p = {p3:.4f}")

    # standardized betas for the final model (z-score predictors and outcome)
    z_data = reg_data.apply(stats.zscore)
    Xz = sm.add_constant(z_data[['Age', 'PSIQ_Total', 'PIT_Pos_Vivid', 'PIT_Neg_Vivid', 'IFES_Negative']])
    mz = sm.OLS(z_data['Depression_Score'], Xz).fit()

    print("\nFinal model - unstandardized coefficients:")
    print(m3.summary().tables[1])

    print("\nFinal model - standardized betas (for reporting effect sizes):")
    for var in ['Age', 'PSIQ_Total', 'PIT_Pos_Vivid', 'PIT_Neg_Vivid', 'IFES_Negative']:
        beta = mz.params[var]
        p = mz.pvalues[var]
        print(f"  {var:20s} beta = {beta:7.3f}  p = {p:.4f}  {sig_stars(p)}")

    return m3


def group_comparison(data):
    print("\n\n" + "=" * 80)
    print("GROUP COMPARISON: Depressed vs Non-depressed")
    print("=" * 80)

    vars_compare = ['PIT_Pos_Vivid', 'PIT_Neg_Vivid', 'PIT_Pos_Likelihood',
                    'PIT_Neg_Likelihood', 'IFES_Negative', 'PSIQ_Total']

    raw_p = []
    rows = []
    for var in vars_compare:
        g0 = data.loc[data['Depression_Group'] == 0, var].dropna()
        g1 = data.loc[data['Depression_Group'] == 1, var].dropna()
        if len(g0) <= 5 or len(g1) <= 5:
            continue
        t, p = stats.ttest_ind(g0, g1, equal_var=False)

        # Welch-Satterthwaite df
        v0, v1 = g0.var(ddof=1), g1.var(ddof=1)
        n0, n1 = len(g0), len(g1)
        df = ((v0 / n0 + v1 / n1) ** 2) / (
            (v0 / n0) ** 2 / (n0 - 1) + (v1 / n1) ** 2 / (n1 - 1)
        )

        pooled_sd = np.sqrt((g0.std()**2 + g1.std()**2) / 2)
        d = (g1.mean() - g0.mean()) / pooled_sd

        raw_p.append(p)
        rows.append({'var': var, 'm0': g0.mean(), 'm1': g1.mean(), 't': t, 'df': df, 'p': p, 'd': d})

    reject, p_adj, _, _ = multipletests(raw_p, alpha=0.05, method='fdr_bh')
    for row, p_c in zip(rows, p_adj):
        row['p_fdr'] = p_c

    print(f"{'variable':22s} {'Non-dep':>8s} {'Dep':>8s} {'t':>7s} {'df':>7s} {'p':>8s} {'p_fdr':>8s} {'d':>6s}")
    for row in rows:
        print(f"{row['var']:22s} {row['m0']:8.2f} {row['m1']:8.2f} {row['t']:7.2f} "
              f"{row['df']:7.1f} {row['p']:8.4f} {row['p_fdr']:8.4f} {row['d']:6.2f}  {sig_stars(row['p_fdr'])}")

    print("\nNote: df uses Welch-Satterthwaite correction (unequal variances assumed).")
    print("p_fdr = Benjamini-Hochberg corrected p-value across the 6 comparisons above.")


def main():
    data = pd.read_csv("cleaned_data.csv")
    print("Data loaded:", data.shape)

    run_correlations(data)
    hierarchical_regression(data)
    group_comparison(data)


if __name__ == "__main__":
    main()
