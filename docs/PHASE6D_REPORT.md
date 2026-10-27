# Phase 6D Report — Scaffold Generalization and Robustness

**Run:** `20261027T154454Z_d02c301c` · **Configuration:** `configs/phase6d.json` (MD5 `d02c301c5b840ecb9874417a2423c833`, unchanged from the committed, prespecified version) · **Plan:** `docs/PHASE6D_PLAN.md`

This report presents the prespecified Phase 6D analyses. Labels for the primary and secondary contrasts come from the prespecified decision rule. Everything else is exploratory and carries no significance claim.

All equivalence statements are **conditional on δ = 0.05 LogS and on the validity of the scaffold-cluster bootstrap** (assumptions in Section 9).

## 1. Execution

| Item | Value |
|---|---|
| Benchmark estimate | 15.2 min (threshold 60 min) |
| Full run | 781 s (13.0 min), 8-core Apple Silicon, `n_jobs=-1` |
| Outer folds fitted | 104 tuned outer folds (4 scenarios × 26), plus model-seed refits |
| Protected files | Production model, Phase 3/4 results and all Phase 6C outputs byte-identical before and after (checked automatically by the run, and again manually) |
| Reproducibility | A second independent full run (758 s) gave identical results (Section 10) |
| Environment | Python 3.12.15, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.18.1, XGBoost 3.4.1, RDKit 2026.3.6; `pip check` clean |

**Pre-run verification on the real data:** zero Murcko or generic group overlaps across all 52 outer and 260 inner folds. The benzene and generic blocks never appear in any inner validation fold.

## 2. Primary protocol: non-benzene scaffold CV (S0)

874 molecules in 583 Murcko groups; 5 folds × 5 repeats; benzene always in training.

| Model | Repeat-averaged RMSE | MAE | Mean signed error | Per-repeat RMSE sd | Fold-averaged RMSE (sd) | R² (mean over repeats) |
|---|---|---|---|---|---|---|
| Baseline | 2.258 | 1.813 | 0.067 | 0.003 | 2.241 (0.285) | −0.011 |
| Linear regression | 1.057 | 0.829 | 0.084 | 0.006 | 1.052 (0.101) | 0.779 |
| Lasso | 1.057 | 0.829 | 0.084 | 0.005 | 1.052 (0.101) | 0.778 |
| RF reference (production config) | 0.837 | 0.597 | 0.026 | 0.011 | 0.831 (0.102) | 0.861 |
| RF tuned | 0.832 | 0.594 | 0.028 | 0.010 | 0.826 (0.098) | 0.863 |
| XGBoost | 0.828 | 0.601 | 0.033 | 0.009 | 0.823 (0.093) | 0.864 |

**Molecule-weighted vs fold-averaged.** The fold-averaged RMSE is 0.005–0.017 lower than the molecule-weighted value for every model. The two differ because averaging per-fold RMSEs is not the same as computing RMSE over all molecules: folds with fewer, harder molecules get the same weight as larger folds. The balanced splitter keeps fold sizes close (no benzene fold here), so the gap is small.

## 3. Secondary protocol: benzene holdout (S0)

254 benzene molecules; trained on all 874 non-benzene molecules; single deterministic split.

| Model | RMSE | MAE | Mean signed error | R² |
|---|---|---|---|---|
| Baseline | 1.471 | 1.169 | −0.253 | −0.030 |
| Linear regression | 0.892 | 0.689 | −0.292 | 0.621 |
| Lasso | 0.887 | 0.687 | −0.280 | 0.625 |
| RF reference | 0.783 | 0.615 | −0.020 | 0.708 |
| RF tuned | 0.785 | 0.617 | −0.027 | 0.707 |
| XGBoost | 0.745 | 0.589 | −0.057 | 0.736 |

The benzene targets have a smaller spread (sd 1.45) than the non-benzene molecules, so RMSE and R² are not directly comparable between the two protocols.

**Composite out-of-fold result (benzene holdout + non-benzene scaffold CV; not a single cross-validation estimate)**, 1,128 molecules, descriptive only:

| Model | Baseline | Linear | Lasso | RF reference | RF tuned | XGBoost |
|---|---|---|---|---|---|---|
| RMSE | 2.107 | 1.022 | 1.021 | 0.825 | 0.822 | 0.810 |

## 4. Prespecified contrasts (non-benzene scaffold CV, S0)

Δ = RMSE(A) − RMSE(B). Intervals are 95% scaffold-cluster percentile bootstrap intervals (583 clusters, largest 39 molecules, 10,000 replicates).

| Role | A − B | Δ | 95% CI | p | Holm p | Decision label |
|---|---|---|---|---|---|---|
| **Primary** | RF reference − XGBoost | **0.009** | **−0.016 to 0.035** | 0.54 | — | **Practically equivalent at δ** |
| Secondary | RF reference − RF tuned | 0.005 | −0.008 to 0.018 | 0.47 | 0.94 | Practically equivalent at δ |
| Secondary | RF tuned − XGBoost | 0.004 | −0.019 to 0.026 | 0.76 | 0.94 | Practically equivalent at δ |
| Secondary | Lasso − RF reference | 0.220 | 0.131 to 0.305 | 0.0002 | 0.0006 | Difference larger than δ |

**Primary result.** Under the prespecified rule, the production-configuration Random Forest and XGBoost are practically equivalent at δ = 0.05 LogS on unseen non-benzene Murcko scaffolds. This holds conditional on δ = 0.05 LogS and on the validity of the scaffold-cluster bootstrap. The whole interval lies inside ±0.05. It does not establish that the models are identical, and the conclusion would change with a smaller δ (for example 0.03).

**Supporting statistics for the primary pair**
- **Correlated errors:** the per-molecule signed errors of the two models correlate at 0.94.
- **Win rate:** XGBoost has the lower repeat-averaged squared error for 48.3% of molecules.
- **Per-repeat Δ:** 0.006, 0.004, 0.014, 0.025, −0.004. This is split-to-split variability that the bootstrap does not include.
- **Model seeds (no re-tuning):** Δ = 0.008 to 0.011 across random states 42–45, so seed variability is much smaller than the bootstrap interval.
- **Phase 6C continuity test** (Nadeau–Bengio on 25 distinct folds; continuity only): RF reference − XGBoost 0.008 (−0.022 to 0.038); Lasso − RF reference 0.221 (0.158 to 0.285).

## 5. Duplicate sensitivity analysis (prespecified)

Primary contrast (RF reference − XGBoost) in non-benzene scaffold CV:

| Variant | Rows | Δ | 95% CI | Label | S0 predictions rescored on this variant's molecules |
|---|---|---|---|---|---|
| S0 | 1,128 | 0.009 | −0.016 to 0.035 | Practically equivalent at δ | 0.009 (−0.016 to 0.035) |
| S1 (hexitol pair removed) | 1,126 | −0.008 | −0.034 to 0.018 | Practically equivalent at δ | 0.009 (−0.016 to 0.035) |
| S2 (duplicates collapsed, conflicts dropped) | 1,116 | −0.008 | −0.046 to 0.029 | Practically equivalent at δ | 0.009 (−0.016 to 0.035) |

**The prespecified robustness criterion is met:** the primary contrast gets the same label in S0, S1 and S2.

The point estimate changes sign between S0 and S1, even though S1 removes only 2 molecules. Scoring the S0 predictions on S1's molecules gives 0.009, so the shift is **not** caused by the removed molecules. Removing them changes the group composition, which changes the seeded fold partition and therefore the tuning. The magnitude of the shift (≈ 0.017) matches the per-repeat variability above. Partition variability is a real source of uncertainty that the cluster bootstrap does not capture.

**Benzene holdout** (exploratory): Δ = 0.038 (−0.008 to 0.087) in S0; 0.019 in S1; 0.037 in S2.

## 6. Scaffold families (exploratory)

Benzene comes from the benzene holdout; all other families come from non-benzene CV. Intervals are shown only for n ≥ 20. R² is not reported within families.

| Family | n | RF reference RMSE | XGBoost RMSE | Lasso RMSE | XGBoost mean signed error | Target sd | RF − XGBoost Δ (95% CI) |
|---|---|---|---|---|---|---|---|
| Benzene | 254 | 0.783 | 0.745 | 0.887 | −0.057 | 1.45 | 0.038 (−0.008 to 0.087) |
| Biphenyl | 39 | 1.197 | 1.023 | 0.676 | 0.117 | 1.74 | 0.174 (0.112 to 0.254) |
| Naphthalene | 22 | 0.426 | 0.355 | 0.718 | −0.094 | 0.82 | 0.071 (−0.015 to 0.151) |
| Barbiturate | 21 | 0.388 | 0.484 | 0.507 | −0.078 | 0.46 | −0.095 (−0.234 to 0.029) |
| Steroid enone | 17 | 0.669 | 0.687 | 0.825 | −0.168 | 0.80 | −0.018 |
| 1,3,5-Triazine | 16 | 1.725 | 1.730 | 1.575 | 1.521 | 0.75 | −0.005 |
| Pyridine | 14 | 1.326 | 1.317 | 1.773 | −1.071 | 1.95 | 0.009 |
| Diphenylmethane | 12 | 0.541 | 0.572 | 0.754 | −0.279 | 1.70 | −0.030 |
| Cyclohexane | 10 | 0.338 | 0.511 | 1.152 | 0.148 | 1.73 | −0.173 |
| **Acyclic** | 317 | 0.679 | 0.694 | 0.988 | −0.109 | 1.83 | −0.014 (−0.042 to 0.013) |
| Cyclic, group size 2–9 | 212 | 0.869 | 0.848 | 1.002 | 0.134 | 2.05 | 0.021 (−0.020 to 0.061) |
| Cyclic singletons | 194 | 0.904 | 0.906 | 1.252 | 0.154 | 2.08 | −0.002 (−0.047 to 0.044) |

These are descriptive observations, not tested hypotheses:

- **Systematic bias in two families.** Triazines are over-predicted by about 1.5 LogS and pyridines under-predicted by about 1.1 LogS by every model.
- **Biphenyl.** Lasso has a lower RMSE than both tree models. Named-family intervals come from a molecule bootstrap within a single scaffold, so they ignore within-family dependence and are descriptive only.

## 7. Acyclic molecules

**Limitation of identity grouping.** An acyclic molecule has no Murcko scaffold, so it is grouped only by its own canonical SMILES (316 groups for 317 molecules). For these 28% of the dataset, the scaffold protocol behaves like an identity-grouped random split: close analogues, such as neighbouring members of a homologous series, can appear on opposite sides of a split.

**Evidence**

| | Acyclic (n = 317) | Cyclic (n = 811) |
|---|---|---|
| Mean max Tanimoto similarity to training set | 0.588 | 0.453 |
| Median | 0.538 | 0.443 |
| 90th percentile | **1.000** | 0.640 |

At least 10% of acyclic test molecules have a training neighbour with an **identical** Morgan fingerprint (Tanimoto = 1.0). The fingerprint cannot distinguish some close isomers and homologues.

Acyclic RMSE (RF reference 0.679, XGBoost 0.694) should therefore be read as optimistic relative to genuinely novel chemistry. It is not evidence of scaffold generalization.

## 8. Other prespecified analyses

**Generic-scaffold robustness (exploratory; generic block `C1CCCCC1`, 311 molecules)**

| Protocol | RF reference | RF tuned | XGBoost | Lasso | RF − XGBoost Δ (95% CI) |
|---|---|---|---|---|---|
| Generic non-block CV (817 molecules, 481 groups) | 0.801 | 0.797 | 0.792 | 1.028 | 0.009 (−0.017 to 0.032), cluster bootstrap |
| Generic block holdout (311) | 0.861 | 0.859 | 0.859 | 0.995 | 0.003 (−0.044 to 0.049), molecule bootstrap |

No decision label is applied to generic contrasts. Full intervals are in `reports/phase6d/contrasts.json`.

**Grid-boundary check** (flag when more than 50% of outer folds select a grid minimum or maximum). Rows marked † are flagged mechanically and carry no information (see the notes below the table).

| Scenario / model | Parameter | Selection (25 outer folds) | Flagged |
|---|---|---|---|
| Murcko S0 / XGBoost | `max_depth` | 2: 18, 3: 5, 4: 2 | Yes, lower edge again |
| Murcko S0 / XGBoost | `learning_rate` † | 0.03: 14, 0.1: 11 | Yes |
| Murcko S0 / XGBoost | `min_child_weight` † | 1: 17, 5: 8 | Yes |
| Murcko S0 / XGBoost | `n_estimators` | 200: 6, 500: 13, 1000: 6 | No |
| Murcko S0 / RF tuned | `min_samples_leaf` | 1: 16, 3: 7, 5: 2 | Yes; 1 is the smallest possible value |
| Murcko S0 / RF tuned | `max_features` | 0.5: 13, 1.0: 12 | No |
| Murcko S0 / Lasso | `alpha` | 1e-5: 6, 0.001: 6, 0.003: 12, 0.01: 1 | No (resolved since Phase 6C) |
| Generic S0 / Lasso | `alpha` | 1e-5: 25 | Yes; Lasso is effectively unregularized least squares (RMSE identical to linear regression) |

Notes:
- † For two-value dimensions, every choice is a boundary by definition, so these flags are mechanical.
- Single-fold holdouts are also flagged on n = 1.
- Per the plan, grids were not extended. The full table is in `reports/phase6d/boundary_report.json`.

**Exploratory analyses**
- **Error floor:** 304 rows (115 groups) share an identical six-descriptor vector. The within-group target variance implies a floor RMSE of 0.144 for any model using these features.
- **Descriptor distance:** the Spearman correlation between nearest-training distance and absolute error is 0.34 for RF reference, 0.32 for RF tuned and 0.29 for XGBoost, but −0.08 for the linear models. Tree-model error grows with distance from the training data in descriptor space.

## 9. Assumptions and limitations

1. **The cluster bootstrap treats Murcko groups as independent, exchangeable units.** Related scaffolds (for example benzene and pyridine, or steroid variants) are not independent, so intervals may be too narrow.
2. **The bootstrap is conditional on the fitted models.** It does not capture partition and retraining variability. Section 5 shows that this variability is the same order as the primary Δ: the sign flips between S0 and S1 through re-partitioning alone. Repeats were averaged before resampling and were never treated as independent observations, but five repeats sharing most of their training data underestimate training-set variability.
3. **The decision labels depend on δ.** δ = 0.05 LogS is a provisional judgement (about 10% of commonly cited experimental uncertainty and about 6% of the scaffold RMSE). The primary interval reaches 0.035, so δ = 0.03 would make the result *inconclusive*.
4. **Single-split holdouts.** Benzene and generic block holdout intervals use a molecule bootstrap within one scaffold family, conditional on one training set. They are descriptive only.
5. **Sensitivity variants mix two effects.** S1 and S2 change both the data and the fold partition. Scoring S0 predictions on each variant's molecules separates the two.
6. **The boundary rule flags two-value dimensions mechanically**, and single-fold holdouts on n = 1.
7. **Acyclic molecules (28%) are only identity-grouped** (Section 7).
8. **Features.** Six global descriptors, an error floor of 0.144, and many unresolved family-specific biases (triazine, pyridine) limit what any of these models can achieve.
9. **Single dataset.** All results are specific to ESOL and this descriptor set. No external validation was performed.

## 10. Reproducibility and outputs

**Implementation clarifications** (no configuration change; documented here):
- **S2 descriptor equality** uses an absolute tolerance of 10⁻⁹. Two barbiturate duplicate pairs differ by at most 5.7 × 10⁻¹⁴ because RDKit's floating-point summation depends on input atom order; exact equality would have aborted S2 on rounding noise.
- **Scope of variants:** the generic analysis uses S0 data. S1, S2 and the generic analysis use all six models.
- **Seeds:** family and contrast bootstraps reuse the configured seed (20261027), so all contrasts within a protocol share the same resamples.
- **Signed error** is predicted − observed.
- **Averaging:** similarity and descriptor distance are averaged per molecule over repeats.
- **S0 rescoring:** S0 predictions are rescored on S0 rows whose identity group is retained in the variant.
- **R² for repeated protocols** is the mean over repeats of the per-repeat pooled R².

**Outputs**

| Location | Contents |
|---|---|
| `results/phase6d/<run_id>/` (Git ignored) | `oof_predictions.csv.gz` (every out-of-fold prediction, including seed refits), `fold_assignments.csv.gz`, `fold_metrics.csv`, `inner_cv_results.csv.gz` (every candidate's per-fold RMSE, fold sizes and pooled RMSE), `protocol_metrics.csv`, `contrasts.csv`, `family_metrics.csv`, `boundary_report.csv`, `neighbour_statistics.csv`, `bootstrap_distributions.npz`, `config_snapshot.json`, `run_info.json`, `pip_freeze.txt` |
| `reports/phase6d/` (tracked, JSON) | `summary.json`, `contrasts.json`, `families.json`, `sensitivity.json`, `boundary_report.json`, `model_seed_check.json`, `exploratory.json`, `benchmark.json`, `feasibility_estimate.json` |

**Reproduce**
```bash
python src/scaffold_evaluation.py --benchmark
python src/scaffold_evaluation.py
```

**Verification**
- 114 backend tests (73 existing + 41 new in `tests/test_scaffold_evaluation.py`) and 12 frontend tests pass.
- The new tests include:
  - the hand-calculated pooled RMSE (1.3229);
  - equality with RMSE over raw pooled errors, and refit selection matched against raw inner-fold predictions;
  - group disjointness on the real data for every outer and inner fold;
  - block placement and once-per-repeat testing;
  - whole-cluster resampling;
  - repeats not inflating bootstrap observations;
  - every decision-rule branch and Holm values;
  - variant construction, refusal to overwrite, and checksum abort;
  - determinism.
- A second full run is compared in the reproducibility check below.

**Reproducibility check:** a second independent full run (`20261027T155812Z_d02c301c`, 758 s) produced byte-identical results apart from timing columns. That covers every out-of-fold prediction, fold assignment, inner-CV result, fold metric, protocol metric, contrast, family metric, boundary row, neighbour statistic, all 120 bootstrap distributions, and every tracked JSON report.
