"""
stats_utils.py
==============

Statistical helpers shared across the analysis scripts.

Everything in this module is deliberately generic — nothing here knows
about "depression" or "IFES". That separation means the helpers can be
unit-tested on toy data, and reused if the project is extended to a
different outcome.

The functions are organised into four groups:

    1. Multiple-comparison correction
       fdr_correct, sig_stars

    2. Effect sizes and their uncertainty
       cohens_d, hedges_g, bootstrap_effect_ci, fisher_ci

    3. Group tests and nested-model comparison
       welch_ttest, f_change_test

    4. Regression diagnostics
       regression_diagnostics, standardized_regression

A small convention runs through the whole file: for any two-group
function, group_a is the reference group (in this project: the
non-depressed participants) and group_b is the comparison group
(the depressed participants). Positive effect sizes therefore mean
"higher in the depressed group".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import (
    OLSInfluence,
    variance_inflation_factor,
)
from statsmodels.stats.diagnostic import het_breuschpagan


# ---------------------------------------------------------------------
# 1. Multiple-comparison correction
# ---------------------------------------------------------------------

def sig_stars(p):
    """Return the conventional significance marker for a p-value.

    *** p < .001, ** p < .01, * p < .05, otherwise an empty string.
    Used only for display — never for decision-making.
    """
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fdr_correct(p_values, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction to a family of p-values.

    Returns
    -------
    reject : ndarray of bool
        True where the null is rejected at `alpha` after correction.
    adjusted : ndarray of float
        FDR-adjusted p-values (q-values).

    The correction is applied *within* each family of tests rather than
    across the whole project — depression correlations are corrected as
    one family, anxiety correlations as another. Pooling them would
    assume the two outcomes answer the same question, which they don't.
    """
    if not p_values:
        return np.array([], dtype=bool), np.array([])

    reject, adjusted = multipletests(
        p_values,
        alpha=alpha,
        method="fdr_bh",
    )[:2]

    return reject, adjusted


# ---------------------------------------------------------------------
# 2. Effect sizes and their uncertainty
# ---------------------------------------------------------------------

def cohens_d(group_a, group_b):
    """Cohen's d using the pooled standard deviation.

    Sign convention: positive means group_b scored higher than group_a.
    With group_a = non-depressed and group_b = depressed, a positive d
    means the depressed group scored higher on that variable.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    n_a = len(a)
    n_b = len(b)

    pooled_var = (
        (n_a - 1) * np.var(a, ddof=1)
        + (n_b - 1) * np.var(b, ddof=1)
    ) / (n_a + n_b - 2)

    if pooled_var <= 0:
        return np.nan

    return (np.mean(b) - np.mean(a)) / np.sqrt(pooled_var)


def hedges_g(group_a, group_b):
    """Hedges' g — Cohen's d with a small-sample bias correction.

    At n ≈ 120 the correction is tiny, but reporting g alongside d is
    standard practice in psychology and costs nothing.
    """
    d = cohens_d(group_a, group_b)

    n_a = len(group_a)
    n_b = len(group_b)
    df = n_a + n_b - 2

    if not np.isfinite(d) or df <= 0:
        return np.nan

    correction = 1 - 3 / (4 * df - 1)

    return correction * d


def bootstrap_effect_ci(
    group_a,
    group_b,
    n_boot=5000,
    confidence=0.95,
    seed=42,
):
    """Bootstrap percentile confidence interval for Hedges' g.

    The seed is fixed so the interval is reproducible across runs.
    5000 resamples is more than enough for a stable 95% interval at
    this sample size; going higher would only slow the script down.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)

    for i in range(n_boot):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)

        estimates[i] = hedges_g(
            pd.Series(sample_a),
            pd.Series(sample_b),
        )

    alpha = 1 - confidence

    lower = np.nanpercentile(estimates, 100 * alpha / 2)
    upper = np.nanpercentile(estimates, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def fisher_ci(r, n, confidence=0.95):
    """Confidence interval for Pearson's r via Fisher's z transform.

    Works well for |r| < .9 and n > 10. For small samples the interval
    is wide and may be visibly asymmetric — that asymmetry is real, not
    a bug.
    """
    if n <= 3 or not np.isfinite(r):
        return np.nan, np.nan

    # Clip away from ±1 so arctanh stays finite.
    r = np.clip(r, -0.999999, 0.999999)

    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)

    critical = stats.norm.ppf(1 - (1 - confidence) / 2)

    lower_z = z - critical * se
    upper_z = z + critical * se

    return (
        float(np.tanh(lower_z)),
        float(np.tanh(upper_z)),
    )


# ---------------------------------------------------------------------
# 3. Group tests and nested-model comparison
# ---------------------------------------------------------------------

def welch_ttest(group_a, group_b):
    """Welch's independent-samples t-test (equal variances not assumed).

    I use Welch rather than Student's t-test because the two groups in
    this project differ substantially in size (~40 vs ~83), which makes
    the equal-variance assumption hard to justify. Welch costs nothing
    when the assumption does hold and protects against it when it
    doesn't.
    """
    result = stats.ttest_ind(
        group_a,
        group_b,
        equal_var=False,
        nan_policy="omit",
    )

    return {
        "t": float(result.statistic),
        "p": float(result.pvalue),
    }


def f_change_test(old_model, new_model):
    """F-change test for two nested OLS models.

    Tests whether adding the predictors in `new_model` significantly
    improves fit over `old_model`. The two models must be nested — the
    predictors in `old_model` must be a subset of those in `new_model`.

    Used here for the hierarchical regression: Step 2 vs Step 1, and
    Step 3 vs Step 2.
    """
    delta_r2 = new_model.rsquared - old_model.rsquared

    df1 = new_model.df_model - old_model.df_model
    df2 = new_model.df_resid

    if df1 <= 0 or df2 <= 0:
        return {
            "delta_r2": float(delta_r2),
            "F": np.nan,
            "p": np.nan,
            "df1": df1,
            "df2": df2,
        }

    # Compare residual sums of squares. The better-fitting model always
    # has the smaller SSR, so (old - new) is guaranteed positive.
    f_value = (
        ((old_model.ssr - new_model.ssr) / df1)
        / (new_model.ssr / df2)
    )

    p_value = stats.f.sf(f_value, df1, df2)

    return {
        "delta_r2": float(delta_r2),
        "F": float(f_value),
        "p": float(p_value),
        "df1": df1,
        "df2": df2,
    }


# ---------------------------------------------------------------------
# 4. Regression diagnostics
# ---------------------------------------------------------------------

def regression_diagnostics(model, predictors):
    """Return observation-level and model-level regression diagnostics.

    Observation-level (one row per participant):
        studentized_resid          internally studentized residuals
        studentized_resid_external externally studentized residuals
        cooks_d                    Cook's distance
        leverage                   hat-matrix diagonal

    Model-level (one row per predictor):
        vif                        variance inflation factor
        breusch_pagan_*            Breusch-Pagan heteroskedasticity test

    The two frames are returned separately so the model-level statistics
    aren't duplicated once per participant. The Breusch-Pagan columns
    are repeated across predictor rows on purpose — BP is a single test
    for the whole model, and keeping it in the same table makes the
    output easy to export.
    """
    influence = OLSInfluence(model)

    observation = pd.DataFrame(
        {
            "studentized_resid": influence.resid_studentized_internal,
            "studentized_resid_external": (
                influence.resid_studentized_external
            ),
            "cooks_d": influence.cooks_distance[0],
            "leverage": influence.hat_matrix_diag,
        },
        index=model.model.data.row_labels,
    )

    X = model.model.exog

    vif_values = []

    # Start at 1 to skip the constant term in column 0.
    for i, name in enumerate(predictors, start=1):
        vif_values.append(
            {
                "variable": name,
                "vif": variance_inflation_factor(X, i),
            }
        )

    bp = het_breuschpagan(
        model.resid,
        model.model.exog,
    )

    vif = pd.DataFrame(vif_values)

    vif["breusch_pagan_lm"] = bp[0]
    vif["breusch_pagan_p"] = bp[1]
    vif["breusch_pagan_f"] = bp[2]
    vif["breusch_pagan_f_p"] = bp[3]

    return observation, vif


def standardized_regression(data, outcome, predictors):
    """Fit an OLS model on z-scored variables, so coefficients are betas.

    Useful when the predictors are measured on different scales
    (0–50 questionnaires, counts, etc.) and you want their effects to
    be directly comparable. The z-scoring is done inside the function
    so the original data frame is left untouched.
    """
    variables = [outcome] + predictors
    z_data = data[variables].apply(stats.zscore)

    X = sm_add_constant(z_data[predictors])

    return statsmodels_ols(z_data[outcome], X)


# ---------------------------------------------------------------------
# Small wrappers around statsmodels, kept as functions so the import
# stays local and only happens when these helpers are actually used.
# ---------------------------------------------------------------------

def sm_add_constant(data):
    """Add an intercept column to a predictor matrix."""
    import statsmodels.api as sm

    return sm.add_constant(data)


def statsmodels_ols(y, X):
    """Fit an ordinary least squares model."""
    import statsmodels.api as sm

    return sm.OLS(y, X).fit()