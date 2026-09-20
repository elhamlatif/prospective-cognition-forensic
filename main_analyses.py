"""
main_analyses.py

Runs the main analyses for the prospective cognition project:
correlations with FDR correction, hierarchical regression predicting
depression, independent-samples comparisons between depressed and
non-depressed participants, and saves all result tables to outputs/.

Reads cleaned_data.csv (produced by 01_data_preparation.py).

Elham Latif
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Make `src` importable when the script is run from the project root.
# Using `parent` (not `parent.parent`) because src/ lives next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.io_utils import setup_logging
from src.stats_utils import (
    bootstrap_effect_ci,
    cohens_d,
    f_change_test,
    fdr_correct,
    fisher_ci,
    hedges_g,
    regression_diagnostics,
    sig_stars,
    welch_ttest,
)


# Predictors entered into the correlation analysis, separately for
# depression and anxiety. The same set is used for both outcomes so
# the two correlation families are directly comparable.
CORRELATION_PREDICTORS = [
    "PIT_Pos_Vivid",
    "PIT_Neg_Vivid",
    "PIT_Pos_Likelihood",
    "PIT_Neg_Likelihood",
    "IFES_Negative",
    "Intrusive_Visual",
    "Intrusive_Verbal",
    "HIT_Negative",
    "PSIQ_Total",
]

# Variables compared between depressed and non-depressed groups.
# PIT Positive Experiencing is excluded because it correlates almost
# perfectly with PIT Positive Vividness in this sample.
GROUP_VARIABLES = [
    "PIT_Pos_Vivid",
    "PIT_Neg_Vivid",
    "PIT_Pos_Likelihood",
    "PIT_Neg_Likelihood",
    "IFES_Negative",
    "PSIQ_Total",
]


def compute_correlations(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
) -> pd.DataFrame:
    """Calculate Pearson correlations between one outcome and all predictors.

    For each predictor, listwise-deletes on the (outcome, predictor) pair
    only, so the sample size can vary slightly across rows depending on
    missingness in the predictor.
    """
    results = []

    for predictor in predictors:
        subset = data[[outcome, predictor]].dropna()

        # Skip variables with almost no data; correlations based on
        # fewer than ~15 observations are not meaningful here.
        if len(subset) <= 15:
            continue

        r, p = stats.pearsonr(
            subset[outcome],
            subset[predictor],
        )

        ci_low, ci_high = fisher_ci(r, len(subset))

        results.append(
            {
                "outcome": outcome,
                "var": predictor,
                "r": r,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p": p,
                "n": len(subset),
            }
        )

    results = pd.DataFrame(results)

    if results.empty:
        return results

    # Benjamini-Hochberg FDR within each outcome family.
    # I run this separately for depression and anxiety rather than
    # pooling all 18 tests, because the two outcomes answer different
    # questions and I want each family to stand on its own.
    reject, adjusted_p = fdr_correct(results["p"].tolist())

    results["p_fdr"] = adjusted_p
    results["sig_fdr"] = [sig_stars(p) for p in adjusted_p]
    results["reject_fdr"] = reject

    return results


def run_correlations(
    data: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    """Run correlations for depression and anxiety, and log both tables."""

    depression = compute_correlations(
        data,
        "Depression_Score",
        CORRELATION_PREDICTORS,
    )

    anxiety = compute_correlations(
        data,
        "Anxiety_Score",
        CORRELATION_PREDICTORS,
    )

    for result, label in [
        (depression, "Depression"),
        (anxiety, "Anxiety"),
    ]:
        if result.empty:
            continue

        logger.info(
            "%s correlation family: %d tests",
            label,
            len(result),
        )

        logger.info(
            "\n%s",
            result[
                [
                    "var",
                    "r",
                    "ci_low",
                    "ci_high",
                    "p",
                    "p_fdr",
                    "n",
                    "sig_fdr",
                ]
            ].round(3).to_string(index=False),
        )

    return pd.concat([depression, anxiety], ignore_index=True)


def run_hierarchical_regression(
    data: pd.DataFrame,
    logger,
) -> dict:
    """Fit the three-step hierarchical regression predicting depression.

    Step 1: Age (control)
    Step 2: + PSIQ Total (general sensory imagery ability)
    Step 3: + PIT positive/negative vividness and IFES Negative

    The idea is to see whether prospective cognition still predicts
    depression after accounting for general imagery ability.
    """

    columns = [
        "Depression_Score",
        "Age",
        "PSIQ_Total",
        "PIT_Pos_Vivid",
        "PIT_Neg_Vivid",
        "IFES_Negative",
    ]

    regression_data = data[columns].dropna()

    logger.info(
        "Hierarchical regression N = %d",
        len(regression_data),
    )

    y = regression_data["Depression_Score"]

    steps = [
        ("Step 1 (Age)", ["Age"]),
        ("Step 2 (+ PSIQ_Total)", ["Age", "PSIQ_Total"]),
        (
            "Step 3 (+ PIT & IFES)",
            [
                "Age",
                "PSIQ_Total",
                "PIT_Pos_Vivid",
                "PIT_Neg_Vivid",
                "IFES_Negative",
            ],
        ),
    ]

    models = []

    for label, predictors in steps:
        X = sm.add_constant(regression_data[predictors])
        model = sm.OLS(y, X).fit()
        models.append(
            {
                "label": label,
                "model": model,
                "predictors": predictors,
            }
        )

    # Assemble the model-comparison table (R², ΔR², F-change per step).
    model_rows = []

    for i, current in enumerate(models):
        model = current["model"]

        row = {
            "step": i + 1,
            "label": current["label"],
            "n": int(model.nobs),
            "r2": model.rsquared,
            "adjusted_r2": model.rsquared_adj,
            "f": model.fvalue,
            "f_p": model.f_pvalue,
        }

        if i > 0:
            change = f_change_test(models[i - 1]["model"], model)
            row.update(
                {
                    "delta_r2": change["delta_r2"],
                    "f_change": change["F"],
                    "f_change_p": change["p"],
                    "df_change": change["df1"],
                }
            )
        else:
            row.update(
                {
                    "delta_r2": np.nan,
                    "f_change": np.nan,
                    "f_change_p": np.nan,
                    "df_change": np.nan,
                }
            )

        model_rows.append(row)

        logger.info(
            "%s: R²=%.3f, adjusted R²=%.3f, F=%.2f, p=%.4f",
            current["label"],
            model.rsquared,
            model.rsquared_adj,
            model.fvalue,
            model.f_pvalue,
        )

    # Coefficient table for the final (Step 3) model.
    final = models[-1]
    final_model = final["model"]
    final_predictors = final["predictors"]

    X_final = sm.add_constant(regression_data[final_predictors])

    # Standardized coefficients (betas) so effect sizes are comparable
    # across predictors measured on different scales.
    standardized = regression_data[
        ["Depression_Score"] + final_predictors
    ].apply(stats.zscore)

    standardized_model = sm.OLS(
        standardized["Depression_Score"],
        sm.add_constant(standardized[final_predictors]),
    ).fit()

    # Heteroskedasticity-robust standard errors (HC3) as a sensitivity
    # check. If the HC3 p-values diverge from the classical ones, the
    # classical inference is worth treating with caution.
    robust_model = sm.OLS(y, X_final).fit(cov_type="HC3")

    coefficients = pd.DataFrame(
        {
            "b": final_model.params,
            "se": final_model.bse,
            "t": final_model.tvalues,
            "p": final_model.pvalues,
            "p_hc3": robust_model.pvalues,
            "beta": standardized_model.params,
        }
    ).drop(index="const")

    coefficients["sig"] = coefficients["p"].apply(sig_stars)

    # Regression diagnostics (studentized residuals, Cook's D, leverage,
    # VIF, Breusch-Pagan) for the final model.
    diagnostic_rows, diagnostic_summary = regression_diagnostics(
        final_model,
        final_predictors,
    )

    vif = diagnostic_summary[["variable", "vif"]].dropna()

    coefficients["vif"] = coefficients.index.map(
        vif.set_index("variable")["vif"]
    )

    logger.info(
        "Final model coefficients:\n%s",
        coefficients.round(3).to_string(),
    )

    logger.info(
        "Regression diagnostics:\n%s",
        diagnostic_summary.round(3).to_string(index=False),
    )

    return {
        "models": models,
        "model_table": pd.DataFrame(model_rows),
        "final_coefs": coefficients,
        "diagnostics": diagnostic_rows,
        "diagnostic_summary": diagnostic_summary,
        "n": len(regression_data),
    }


def run_group_comparison(
    data: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    """Compare depressed and non-depressed participants on key variables.

    Uses Welch's t-test (equal variances not assumed) because the two
    groups have very different sizes (40 vs. ~83) and variances may
    differ. Effect sizes are reported as Cohen's d and Hedges' g
    (with a bootstrap 95% CI), and FDR-corrected across variables.
    """

    results = []

    for variable in GROUP_VARIABLES:
        non_depressed = data.loc[
            data["Depression_Group"] == 0,
            variable,
        ].dropna()

        depressed = data.loc[
            data["Depression_Group"] == 1,
            variable,
        ].dropna()

        # Skip variables with too few observations in either group.
        if len(non_depressed) <= 5:
            continue
        if len(depressed) <= 5:
            continue

        test = welch_ttest(non_depressed, depressed)
        g = hedges_g(non_depressed, depressed)

        ci_low, ci_high = bootstrap_effect_ci(
            non_depressed,
            depressed,
            n_boot=5000,
            confidence=0.95,
            seed=42,
        )

        results.append(
            {
                "var": variable,
                "n_non_dep": len(non_depressed),
                "n_dep": len(depressed),
                "m_non_dep": non_depressed.mean(),
                "m_dep": depressed.mean(),
                "sd_non_dep": non_depressed.std(ddof=1),
                "sd_dep": depressed.std(ddof=1),
                "t": test["t"],
                "p": test["p"],
                "cohens_d": cohens_d(non_depressed, depressed),
                "hedges_g": g,
                "g_ci_low": ci_low,
                "g_ci_high": ci_high,
            }
        )

    results = pd.DataFrame(results)

    if results.empty:
        return results

    reject, adjusted_p = fdr_correct(results["p"].tolist())

    results["p_fdr"] = adjusted_p
    results["sig_fdr"] = [sig_stars(p) for p in adjusted_p]
    results["reject_fdr"] = reject

    logger.info(
        "Group comparisons:\n%s",
        results.round(3).to_string(index=False),
    )

    return results


def main():
    logger = setup_logging(config.LOG_FILE)

    logger.info("Loading cleaned data from %s", config.CLEANED_FILE)

    data = pd.read_csv(config.CLEANED_FILE)

    logger.info("Loaded data shape: %s", data.shape)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Correlations (Table 2) ---
    correlations = run_correlations(data, logger)
    correlations.to_csv(
        config.OUTPUT_DIR / "Table2_correlations.csv",
        index=False,
    )

    # --- Hierarchical regression (Tables 3 & 4) ---
    regression = run_hierarchical_regression(data, logger)

    regression["model_table"].to_csv(
        config.OUTPUT_DIR / "Table3_hierarchical_regression.csv",
        index=False,
    )

    regression["final_coefs"].to_csv(
        config.OUTPUT_DIR / "Table4_final_model_coefficients.csv"
    )

    # --- Regression diagnostics (Appendix) ---
    regression["diagnostics"].to_csv(
        config.OUTPUT_DIR / "Appendix_A_diagnostics.csv"
    )

    regression["diagnostic_summary"].to_csv(
        config.OUTPUT_DIR / "Appendix_B_vif_summary.csv",
        index=False,
    )

    # --- Group comparison (Table 5) ---
    group_results = run_group_comparison(data, logger)
    group_results.to_csv(
        config.OUTPUT_DIR / "Table5_group_comparisons.csv",
        index=False,
    )

    logger.info(
        "Analysis finished. Results saved to %s",
        config.OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()