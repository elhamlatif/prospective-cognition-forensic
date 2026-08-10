# Prospective Cognition and Affective Symptoms in a Forensic Sample

**Quantitative Analysis of Prospective Cognition, Intrusive Imagery, and Affective Symptoms in a Forensic Sample**

Author: Elham Latif

---

## Project Overview

This project examines the relationship between future-oriented mental imagery (prospective cognition), intrusive imagery, sensory imagery ability, and affective symptoms (depression and anxiety) in a forensic sample.

The analysis focuses on:

- Vividness and likelihood of positive vs. negative future scenarios (Prospective Imagery Task)
- Impact of intrusive future-related imagery (IFES)
- Sensory imagery ability (PSIQ)
- Interpretation bias (HIT)
- Depressive and anxiety symptoms (CES-D and GAD-7)

---

## Data

- Original file: `pone.0191551.s001.sav` (SPSS format)
- Sample size: 123 participants
- Cleaned version: `cleaned_data.csv`

---

## Repository Structure

```
├── 01_data_preparation.py     # Script to clean and prepare the data
├── cleaned_data.csv           # Cleaned dataset ready for analysis
├── README.md                  # Project description
└── (future scripts for analysis will be added here)
```

---

## How to Run

1. Place the original `.sav` file in the same folder.
2. Install required packages:

```bash
pip install pyreadstat pandas numpy
```

3. Run the preparation script:

```bash
python 01_data_preparation.py
```

---

## Main Variables in Cleaned Data

| Variable | Description |
|----------|-------------|
| Depression_Score | CES-D total score |
| Anxiety_Score | GAD-7 total score |
| PIT_Pos_Vivid / PIT_Neg_Vivid | Vividness of positive / negative future scenarios |
| PIT_Pos_Likelihood / PIT_Neg_Likelihood | Perceived likelihood of positive / negative scenarios |
| IFES_Negative | Impact of negative future events (intrusive imagery) |
| Intrusive_Visual / Intrusive_Verbal | Intrusive visual and verbal cognition |
| HIT_Negative / HIT_Positive | Number of negative / positive interpretations |
| PSIQ_Total | Overall sensory imagery ability |
| PSIQ_* | Subscales of sensory imagery ability |

---

## Planned Analyses

- Descriptive statistics and correlations
- Hierarchical regression
- Moderation analysis (role of sensory imagery ability)
- Mediation analysis (role of interpretation bias)
- Group comparisons (depressed vs. non-depressed)

---

## Notes

This is a secondary data analysis project. The original dataset comes from a published study (PLOS ONE related supplementary material).
