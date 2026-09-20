Prospective Cognition, Intrusive Imagery, and Affective Symptoms in a Forensic Sample

Quantitative secondary analysis of future-oriented mental imagery and its associations with depression and anxiety symptoms.

Elham Latif

Overview

This project examines whether individual differences in future-oriented cognition and sensory imagery are associated with depression and anxiety symptoms in a forensic sample.

The analysis uses a publicly available dataset from the supplementary material of a PLOS ONE study. The original dataset includes measures of prospective imagery, intrusive future-related cognition, interpretation bias, sensory imagery, depression, and anxiety.

The main focus of this analysis is the relationship between negative future imagery, intrusive future-related cognition, and affective symptoms. I also examine whether general sensory imagery ability contributes to these associations.

This is a secondary analysis. The data were not collected as part of this project.

Research questions

The analysis addresses four main questions:

1. Are future-oriented imagery and intrusive future-related cognition associated with depression and anxiety symptoms?
2. Do depressed and non-depressed participants differ in their patterns of positive and negative future imagery?
3. Does general sensory imagery ability explain additional variance in depression symptoms or modify the association between negative future imagery and depression?
4. Does interpretation bias account for part of the association between intrusive future cognition and depression?

Dataset

The dataset contains 123 participants and 207 variables. Only variables relevant to the present analyses are retained.

The measures used in this project include:

Prospective Imagery Task (PIT):vividness, likelihood, and experiencing ratings for positive and negative future scenarios.
Impact of Future Events Scale (IFES): intrusive and distressing future-related cognition.
Intrusive visual and verbal cognition: measures of unwanted future-related mental content.
Plymouth Sensory Imagery Questionnaire (PSIQ): sensory imagery across seven modalities.
CES-D:depressive symptom severity.
GAD-7:anxiety symptom severity.
Homograph Interpretation Task (HIT):interpretation bias for ambiguous information.

The original .sav  file is retained as the source dataset, while the analysis pipeline selects and renames the variables required for the current project.

 Analysis

The analysis pipeline includes:

descriptive statistics and missing-data inspection
Pearson correlation analyses
Benjamini–Hochberg false discovery rate (FDR) correction
 independent-samples comparisons using Welch's t-test
Cohen's d and Hedges' g
bootstrap confidence intervals for effect sizes
hierarchical linear regression
standardized regression coefficients
moderation analysis
mediation analysis
regression diagnostics, including VIF, Cook's distance, leverage, and Breusch–Pagan testing

Multiple-comparison correction is applied within predefined families of related tests rather than across all analyses combined.

Effect sizes and confidence intervals are reported alongside p-values where appropriate.

Main findings

The strongest association in the current analysis was between intrusive negative future-related cognition and depression symptoms.

IFES Negative was positively correlated with depression (r = .65, p < .001) and anxiety (r = .52). In the final regression model, IFES Negative remained a significant predictor of depression symptoms (standardized β = .54, p < .001).

Group comparisons also showed differences between depressed and non-depressed participants in several measures of future-oriented cognition. The largest reported group difference was observed for IFES Negative (Cohen's d = 1.19). Positive future likelihood showed a negative effect size (d = −0.88), while general sensory imagery measured by PSIQ Total also differed between groups (d = −0.65).

PSIQ Total was negatively correlated with depression (r = −.30) and contributed additional explanatory value in the regression analyses. The interaction between PSIQ Total and negative future vividness was statistically significant (p = .025), suggesting that the association between negative future vividness and depression differed according to general sensory imagery ability.

The final regression model accounted for approximately 49% of the variance in depression symptoms (R*² = .49).

The mediation analyses did not provide evidence that negative interpretation bias, measured by HIT Negative, mediated the associations tested between future-related cognition and depression.

Interpretation

The findings are consistent with an association between intrusive negative future-related cognition and greater depressive symptom severity in this sample. The results also indicate that positive future expectations, negative future imagery, and general sensory imagery ability may show different relationships with depression.

The moderation analysis provides a possible indication that general imagery ability may influence the relationship between negative future imagery and depression. However, this interaction should be interpreted cautiously and requires replication.

The absence of evidence for mediation by interpretation bias does not establish that the constructs are unrelated. It indicates that the specific mediation pathways tested in this cross-sectional dataset were not supported.

Limitations

Several limitations are important when interpreting these results.

First, this is a "secondary analysis" of an existing dataset. The sampling strategy and measurement instruments were determined by the original study.

Second, the data are "cross-sectional". Regression coefficients describe statistical associations and should not be interpreted as evidence that intrusive future imagery causes depression.

Third, the sample size is relatively modest (N = 123), particularly for interaction and mediation analyses.

Fourth, the analyses involve multiple related hypotheses. FDR correction was therefore incorporated into the analysis pipeline, but the possibility of false-negative findings and limited statistical power remains.

Finally, the results come from a single forensic sample. Replication in independent forensic and clinical samples would be necessary to determine how well these associations generalize.

Project structure

prospective-cognition-forensic/
│
├── README.md
├── .gitignore
│
├── 01_data_preparation.py
├── main_analyses.py
│
├── data/
│   └── pone.0191551.s001.sav
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── io_utils.py
│   └── stats_utils.py
│
├── outputs/
│   ├── Table2_correlations.csv
│   ├── Table3_hierarchical_regression.csv
│   ├── Table4_final_model_coefficients.csv
│   ├── Table5_group_comparisons.csv
│   ├── Appendix_A_diagnostics.csv
│   └── Appendix_B_vif_summary.csv
│
└── cleaned_data.csv


Reproducibility

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install pandas numpy scipy statsmodels pyreadstat
```

Run the data-preparation script:

```bash
python 01_data_preparation.py
```

Then run the main analysis:

```bash
python main_analyses.py
```

The analysis scripts generate the tables and diagnostic outputs in the `outputs/` directory.

Selected results

 Analysis                                                             Result       

IFES Negative × depression                        r = .65, p < .001
IFES Negative × anxiety                                    r = .52          
PIT Positive Likelihood × depression         r = −.44, p < .001
Final regression model                                     R*² = .49
IFES Negative in final model                    β = .54, *p* < .001
IFES group comparison                           Cohen's *d* = 1.19  
PSIQ × PIT Negative Vividness                       p = .025          
HIT Negative mediation                               Not supported    

Future work

A longitudinal dataset would allow testing whether changes in intrusive future imagery are associated with subsequent changes in depression symptoms.

Future analyses could also examine whether the observed moderation effect replicates in an independent sample and whether the relationships differ between depression and anxiety. Replication would be particularly important for the interaction and mediation findings because these analyses are more sensitive to sample size and measurement characteristics.

Data source

The data are derived from publicly available supplementary material associated with the original PLOS ONE publication. The present repository contains a secondary analysis and is not affiliated with the original data collection.

The original investigators designed the study, collected the data, and developed the measures. My contribution is limited to the secondary data-processing and statistical analysis presented in this repository.

Contact

Elham Latif
