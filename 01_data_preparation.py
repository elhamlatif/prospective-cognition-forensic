"""
Data Preparation Script
Project: Quantitative Analysis of Prospective Cognition, Intrusive Imagery, 
         and Affective Symptoms in a Forensic Sample

Author: Elham Latif
"""

import pyreadstat
import pandas as pd
import numpy as np


def main():
    # --------------------------------------------
    # Load the original SPSS file
    # --------------------------------------------
    df, meta = pyreadstat.read_sav("pone.0191551.s001.sav")
    print("Raw data shape:", df.shape)

    # --------------------------------------------
    # Create a clean dataframe with selected variables
    # --------------------------------------------
    data = pd.DataFrame()

    # Demographics & clinical groups
    data["Age"] = df["Age"]
    data["Depression_Group"] = df["Depression"]       # 0 = No, 1 = Yes
    data["Anxiety_Group"] = df["GeneralAnxiety"]      # 0 = No, 1 = Yes

    # Symptom scores
    data["Depression_Score"] = df["CESDtotal"]
    data["Anxiety_Score"] = df["GADtotal"]

    # Prospective Imagery Task (PIT)
    data["PIT_Pos_Vivid"] = df["Vividness_Positivescenarios"]
    data["PIT_Neg_Vivid"] = df["Vividness_Negativecenarios"]
    data["PIT_Pos_Likelihood"] = df["Likelihood_Positivecenarios"]
    data["PIT_Neg_Likelihood"] = df["Likelihood_Negativecenarios"]
    data["PIT_Pos_Experiencing"] = df["Experiencing_Positivecenarios"]
    data["PIT_Neg_Experiencing"] = df["Experiencing_Negativecenarios"]

    # Intrusive cognition
    data["IFES_Negative"] = df["IFES_N_total"]
    data["Intrusive_Visual"] = df["Intrusive_visual_imagery"]
    data["Intrusive_Verbal"] = df["Intrusive_verbal_thought"]

    # Interpretation bias (HIT)
    data["HIT_Negative"] = df["HIT_N_NEG"]
    data["HIT_Positive"] = df["HIT_N_POS"]

    # Sensory imagery ability (PSIQ subscales)
    data["PSIQ_Appearance"] = df["Appearance"]
    data["PSIQ_Sound"] = df["Sound"]
    data["PSIQ_Smell"] = df["Smell"]
    data["PSIQ_Taste"] = df["Taste"]
    data["PSIQ_Touch"] = df["Touching"]
    data["PSIQ_Feeling"] = df["Feeling"]
    data["PSIQ_Body"] = df["Bodysensations"]

    # Total sensory imagery score
    psiq_items = [
        "PSIQ_Appearance", "PSIQ_Sound", "PSIQ_Smell", "PSIQ_Taste",
        "PSIQ_Touch", "PSIQ_Feeling", "PSIQ_Body"
    ]
    data["PSIQ_Total"] = data[psiq_items].mean(axis=1)

    # --------------------------------------------
    # Quick check
    # --------------------------------------------
    print("\nCleaned data shape:", data.shape)
    print("\nMissing values:")
    print(data.isnull().sum().sort_values(ascending=False))

    print("\nDescriptive statistics:")
    print(data.describe().round(2).T)

    # --------------------------------------------
    # Save cleaned data
    # --------------------------------------------
    data.to_csv("cleaned_data.csv", index=False)
    print("\nSaved: cleaned_data.csv")


if __name__ == "__main__":
    main()
