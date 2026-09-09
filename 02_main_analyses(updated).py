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


def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def run_correlations(data):
    print("=" * 65)
    print("CORRELATIONS WITH DEPRESSION SCORE")
    print("=" * 65)

    vars_to_test = [
        'PIT_Pos_Vivid', 'PIT_Neg_Vivid',
        'PIT_Pos_Likelihood', 'PIT_Neg_Likelihood',
        'IFES_Negative', 'Intrusive_Visual', 'Intrusive_Verbal',
        'HIT_Negative', 'PSIQ_Total'
    ]

    for var in vars_to_test:
        temp = data[['Depression_Score', var]].dropna()
        if len(temp) <= 15:
            continue
        r, p = stats.pearsonr(temp['Depression_Score'], temp[var])
        print(f"{var:25s}  r = {r:7.3f}   p = {p:.4f}  {sig_stars(p)}   (n={len(temp)})")

    print("\n" + "=" * 65)
    print("CORRELATIONS WITH ANXIETY SCORE")
    print("=" * 65)

    for var in vars_to_test:
        temp = data[['Anxiety_Score', var]].dropna()
        if len(temp) <= 15:
            continue
        r, p = stats.pearsonr(temp['Anxiety_Score'], temp[var])
        print(f"{var:25s}  r = {r:7.3f}   p = {p:.4f}  {sig_stars(p)}   (n={len(temp)})")


def hierarchical_regression(data):
    print("\n\n" + "=" * 65)
    print("HIERARCHICAL REGRESSION: Predicting Depression_Score")
    print("=" * 65)

    reg_data = data[['Depression_Score', 'Age', 'PSIQ_Total',
                     'PIT_Pos_Vivid', 'PIT_Neg_Vivid', 'IFES_Negative']].dropna()
    print(f"N for regression: {len(reg_data)}")

    y = reg_data['Depression_Score']

    # step 1 - age only, baseline
    X1 = sm.add_constant(reg_data[['Age']])
    m1 = sm.OLS(y, X1).fit()
    print("\nStep 1 (Age only):")
    print(f"  R2 = {m1.rsquared:.3f}, Adj.R2 = {m1.rsquared_adj:.3f}")

    # step 2 - add sensory imagery
    X2 = sm.add_constant(reg_data[['Age', 'PSIQ_Total']])
    m2 = sm.OLS(y, X2).fit()
    print("\nStep 2 (+ PSIQ_Total):")
    print(f"  R2 = {m2.rsquared:.3f}, Adj.R2 = {m2.rsquared_adj:.3f}, "
          f"deltaR2 = {m2.rsquared - m1.rsquared:.3f}")

    # step 3 - add prospective imagery + intrusion measures
    X3 = sm.add_constant(reg_data[['Age', 'PSIQ_Total', 'PIT_Pos_Vivid',
                                   'PIT_Neg_Vivid', 'IFES_Negative']])
    m3 = sm.OLS(y, X3).fit()
    print("\nStep 3 (+ PIT & IFES):")
    print(f"  R2 = {m3.rsquared:.3f}, Adj.R2 = {m3.rsquared_adj:.3f}, "
          f"deltaR2 = {m3.rsquared - m2.rsquared:.3f}")

    print("\nFinal model coefficients:")
    print(m3.summary().tables[1])

    return m3


def group_comparison(data):
    print("\n\n" + "=" * 65)
    print("GROUP COMPARISON: Depressed vs Non-depressed")
    print("=" * 65)

    vars_compare = ['PIT_Pos_Vivid', 'PIT_Neg_Vivid', 'PIT_Pos_Likelihood',
                    'PIT_Neg_Likelihood', 'IFES_Negative', 'PSIQ_Total']

    for var in vars_compare:
        g0 = data.loc[data['Depression_Group'] == 0, var].dropna()
        g1 = data.loc[data['Depression_Group'] == 1, var].dropna()
        if len(g0) <= 5 or len(g1) <= 5:
            continue
        t, p = stats.ttest_ind(g0, g1, equal_var=False)
        pooled_sd = np.sqrt((g0.std()**2 + g1.std()**2) / 2)
        d = (g1.mean() - g0.mean()) / pooled_sd
        print(f"{var:25s}  Non-dep={g0.mean():.2f}  Dep={g1.mean():.2f}  "
              f"t={t:.2f}  p={p:.4f} {sig_stars(p)}  d={d:.2f}")


def main():
    data = pd.read_csv("cleaned_data.csv")
    print("Data loaded:", data.shape)

    run_correlations(data)
    hierarchical_regression(data)
    group_comparison(data)


if __name__ == "__main__":
    main()
