"""
prepare_data.py

Build cleaned_data.csv from the public PLOS ONE dataset
(pone.0191551.s001.sav).

Project: prospective cognition, intrusive imagery, and affective
symptoms in a forensic sample.

Elham Latif
"""

import pandas as pd
import pyreadstat

RAW_FILE = "data/pone.0191551.s001.sav"
OUT_FILE = "cleaned_data.csv"

# Map raw SPSS column names to the shorter names I use throughout
# the analysis. Keeping this mapping explicit means if I ever go back
# to the original .sav file, I can see exactly which variable is which.
#
# A note on spelling: the raw file uses "Negativecenarios" (missing the
# "s" in "scenarios"). I left the misspelling intact on the left side
# of the mapping, because that is what pyreadstat actually reads.
COLUMN_MAP = {
    "Age": "Age",

    # Group variables (0 = no, 1 = yes) — used for the t-tests
    "Depression": "Depression_Group",
    "GeneralAnxiety": "Anxiety_Group",

    # Continuous symptom scores
    "CESDtotal": "Depression_Score",
    "GADtotal": "Anxiety_Score",

    # Prospective Imagery Task (PIT)
    # Vividness and likelihood are the two dimensions I care about most;
    # experiencing is included for completeness.
    "Vividness_Positivescenarios": "PIT_Pos_Vivid",
    "Vividness_Negativecenarios": "PIT_Neg_Vivid",
    "Likelihood_Positivecenarios": "PIT_Pos_Likelihood",
    "Likelihood_Negativecenarios": "PIT_Neg_Likelihood",
    "Experiencing_Positivecenarios": "PIT_Pos_Experiencing",
    "Experiencing_Negativecenarios": "PIT_Neg_Experiencing",

    # Intrusive future-related cognition
    "IFES_N_total": "IFES_Negative",
    "Intrusive_visual_imagery": "Intrusive_Visual",
    "Intrusive_verbal_thought": "Intrusive_Verbal",

    # Interpretation bias (HIT)
    "HIT_N_NEG": "HIT_Negative",
    "HIT_N_POS": "HIT_Positive",

    # PSIQ sensory subscales — averaged into PSIQ_Total below
    "Appearance": "PSIQ_Appearance",
    "Sound": "PSIQ_Sound",
    "Smell": "PSIQ_Smell",
    "Taste": "PSIQ_Taste",
    "Touching": "PSIQ_Touch",
    "Feeling": "PSIQ_Feeling",
    "Bodysensations": "PSIQ_Body",
}

PSIQ_COLS = [
    "PSIQ_Appearance",
    "PSIQ_Sound",
    "PSIQ_Smell",
    "PSIQ_Taste",
    "PSIQ_Touch",
    "PSIQ_Feeling",
    "PSIQ_Body",
]


def main():
    raw, _meta = pyreadstat.read_sav(RAW_FILE)
    print(f"Raw data: {raw.shape[0]} participants, {raw.shape[1]} variables")

    # Keep only the variables I actually use.
    # The raw file has 207 columns; most are not relevant to this project.
    data = raw[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)

    # PSIQ Total is the *mean* of the 7 subscales, not their sum.
    # Averaging keeps the scale interpretable on the original 0–50 metric
    # and avoids artificially inflating scores for participants who
    # happened to answer more items.
    data["PSIQ_Total"] = data[PSIQ_COLS].mean(axis=1)

    print(f"\nCleaned data: {data.shape[0]} participants, {data.shape[1]} variables")

    print("\nMissing values per column:")
    missing = data.isnull().sum().sort_values(ascending=False)
    print(missing[missing > 0] if (missing > 0).any() else "  (none)")

    print("\nDescriptives:")
    print(data.describe().round(2).T)

    data.to_csv(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()