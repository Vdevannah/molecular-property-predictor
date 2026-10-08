# Phase 4 — Model Evaluation and Validation

## Goal

Phase 4 checks whether the trained Random Forest model is useful beyond the initial random split. The evaluation moves from model fitting to model validation, diagnostic interpretation, and an assessment of chemical similarity between training and testing molecules.

## Evaluation metrics

The model is evaluated with three regression metrics:

- **MAE (mean absolute error):** the average absolute difference between experimental and predicted LogS values.
- **RMSE (root mean squared error):** the square root of the average squared error, giving more weight to large mistakes.
- **R² (coefficient of determination):** the proportion of variation in the target explained by the model.

The scripts use the same metric definitions as Phase 3 so that results remain directly comparable.

## Diagnostic plots

The evaluator generates three plots:

1. **Actual versus predicted LogS:** shows whether points fall near a perfect-prediction line.
2. **Residual plot:** shows residuals across predicted values and helps reveal systematic bias.
3. **Feature importance:** shows which molecular descriptors contribute most strongly to the Random Forest predictions.

These plots are saved in the local `results/` directory and are not committed to Git.

## Random-split validation

The saved model is loaded from `models/solubility_model.joblib` using the stored feature order from `models/feature_columns.joblib`.

The random split uses:

- `test_size=0.20`
- `random_state=42`

This is useful for a baseline check, but it may contain molecules from the same chemical series in both sets. A model can appear accurate when chemically similar molecules are placed in both training and testing sets.

## Scaffold-aware validation

A scaffold-aware split groups molecules by RDKit Murcko scaffold. Acyclic molecules receive a unique canonical SMILES-based group, so chemically distinct acyclic molecules are not incorrectly combined into one scaffold family.

The split then assigns complete scaffold groups to either training or testing data. This reduces overlap between the two sets and gives a more realistic estimate of whether the model can generalize to unseen chemical series.

## Final validation statistics

For the complete ESOL descriptor dataset:

- 1,128 molecules
- 584 total scaffold groups
- 317 acyclic molecules
- 316 unique acyclic groups
- 902 training molecules
- 226 testing molecules
- zero training/test scaffold-group overlap
- zero training/test index overlap

The corrected scaffold split produced:

- MAE: 0.615514
- RMSE: 0.880497
- R²: 0.795425

## Why scaffold separation matters

The main risk is **data leakage**. If molecules with the same scaffold appear in both training and testing data, the model can learn a chemical pattern that is repeated in the evaluation set.

Scaffold-aware validation helps answer a more challenging question:

> Can the model predict solubility for molecules from chemical series that were not seen during training?

A scaffold split is more conservative than a random split, so it may produce worse metrics. Lower performance does not necessarily mean the model is poor; it may reflect a more difficult and realistic evaluation setting.

## Applicability domain

The model is trained on ESOL molecules and should not be assumed to generalize to all possible chemicals. Its reliability is limited by:

- the molecules present in the training data,
- the chosen molecular descriptors,
- the range of experimental values,
- the absence of explicit uncertainty estimates,
- the potential for scaffold or chemical-series bias.

## Model interpretation

Feature importance is an explanatory, not causal, view of the model. The Random Forest ranks descriptors according to how often their splits improve impurity reduction.

The descriptor ranking can help explain predictions, but it is not proof that a single chemical property causes the model's behavior.

## Practical interview discussion

A strong explanation of this phase should include:

- why metrics must be calculated on a held-out set,
- why random splitting can leak chemical similarity,
- why the Murcko scaffold is a useful chemical grouping,
- why whole-group assignment is preferable to splitting individual rows,
- why plots help identify systematic errors,
- why performance on a scaffold split may differ from performance on a random split,
- why model evaluation is only one part of a scientific validation process.

## Phase 4 output files

The evaluator writes local files to `results/`:

- `actual_vs_predicted.png`
- `residual_plot.png`
- `feature_importance.png`
- `phase4_metrics.json`

These files are generated locally and excluded from Git.
