# Phase 6C — Nested Cross-Validation and Hyperparameter Optimization

## Scope

Phase 6C implements the evaluation protocol defined in `docs/PHASE6A_REPORT.md`. It compares a mean baseline, Lasso, Random Forest and XGBoost using grouped nested cross-validation. This report records the procedure, configuration, runtime and raw results only. Scientific interpretation and any change to the production model are deferred to later phases.

The production model, FastAPI application, React frontend and Phase 3/4 results are unchanged. MD5 hashes of `models/*.joblib` and `results/phase{3,4}_metrics.json` were identical before and after every run.

## How to run

```bash
python src/cross_validation.py --benchmark   # time one outer fold per protocol, estimate total
python src/cross_validation.py               # full experiment (~3 minutes on 8 cores)
```

Outputs are written to `results/phase6c/` (Git ignored):

| File | Content |
|---|---|
| `config.json` | Dataset hash, grouping summary, CV settings, seeds, model parameters, grids, library versions, runtime |
| `fold_metrics.csv` | One row per protocol × repeat × fold × model: RMSE, MAE, R², train/test sample and group counts, inner-CV RMSE, selected hyperparameters, fit time |
| `fold_assignments.csv` | The outer test fold and group label of every row, for every protocol and repeat |
| `summary.csv` | Mean, sd, min and max of each metric per protocol and model |
| `paired_comparisons.csv` | Every model pair on RMSE and MAE: mean and sd of the per-fold differences, corrected standard error, 95% CI, t statistic, p-value |
| `benchmark.json` | Pre-run runtime benchmark |

## Procedure

**Data.** All 1,128 rows of `data/esol_descriptors.csv`, including duplicate measurements, are used without filtering. The features are the six RDKit descriptors; the target is measured log solubility.

**Grouping**

| Protocol | Group label | Groups | Largest group |
|---|---|---|---|
| `random_grouped` | Stereo-aware canonical SMILES, whitespace stripped | 1,117 | 2 rows |
| `scaffold` | Bemis–Murcko scaffold; acyclic molecules use canonical SMILES (the Phase 4 rule) | 584 | benzene, 254 rows |

**Splitter.** `BalancedGroupKFold` in `src/cross_validation.py` assigns whole groups to folds. It places the largest group first and always fills the lightest fold, breaking ties at random with a fixed seed. Every group, including benzene, is tested exactly once per repeat.

scikit-learn's `GroupKFold(shuffle=True)` was not used. It balances the *number of groups* per fold, not the number of samples, which would give very uneven scaffold folds.

**Nested cross-validation**
- Outer loop: 5 folds × 3 repeats per protocol (outer seed = 42 + repeat).
- All models use the identical outer folds.
- Inner loop: `GridSearchCV` with 5 grouped folds, using the same grouping as the outer protocol (inner seed = 42 + 1000·(repeat+1) + fold).
- The search runs only on the outer-training rows, with scoring `neg_root_mean_squared_error`. The best candidate is refit on the full outer-training fold and then scored once on the outer test fold.

**Models and grids** (fixed before the run)

| Model | Grid |
|---|---|
| `baseline` (DummyRegressor, mean) | none |
| `random_forest_reference` (production config: 200 trees) | none |
| `lasso` (StandardScaler → Lasso, `max_iter=10000`) | `alpha` ∈ {0.001, 0.01, 0.1, 1.0} |
| `random_forest_tuned` | `n_estimators` ∈ {200, 500}; `max_depth` ∈ {None, 10, 20}; `min_samples_leaf` ∈ {1, 3} |
| `xgboost` (`subsample=0.8`, `tree_method="hist"`) | `n_estimators` ∈ {200, 500}; `max_depth` ∈ {3, 6}; `learning_rate` ∈ {0.05, 0.1} |

All estimators use `random_state=42` and `n_jobs=1`. Parallelism comes only from `GridSearchCV(n_jobs=-1)`, so threads are never nested.

**Paired comparison.** For each pair of models, the per-fold metric differences are tested with the Nadeau–Bengio corrected resampled t-test (variance multiplied by 1/J + n_test/n_train, df = J − 1, 95% CI). A difference counts as detected only if its interval excludes zero. A small numerical difference on its own is not reported as superiority.

## Runtime

| Step | Time |
|---|---|
| Benchmark estimate (one outer fold per protocol × 15) | 200 s |
| Full run 1 | 188 s |
| Full run 2 | 177 s |

Measured on an 8-core Apple Silicon Mac with `n_jobs=-1`. No reduction in folds, repeats or grids was needed.

## Results

Mean ± sd over 15 outer folds per protocol:

| Protocol | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| random_grouped | baseline | 2.095 ± 0.079 | 1.660 | −0.001 |
| random_grouped | random_forest_reference | 0.720 ± 0.052 | 0.518 | 0.880 |
| random_grouped | lasso | 1.000 ± 0.037 | 0.781 | 0.770 |
| random_grouped | random_forest_tuned | 0.722 ± 0.052 | 0.520 | 0.879 |
| random_grouped | xgboost | 0.710 ± 0.045 | 0.520 | 0.884 |
| scaffold | baseline | 2.097 ± 0.409 | 1.690 | −0.050 |
| scaffold | random_forest_reference | 0.821 ± 0.054 | 0.600 | 0.824 |
| scaffold | lasso | 1.025 ± 0.092 | 0.802 | 0.730 |
| scaffold | random_forest_tuned | 0.816 ± 0.053 | 0.597 | 0.826 |
| scaffold | xgboost | 0.802 ± 0.064 | 0.593 | 0.833 |

**Paired RMSE differences** (model A − model B; 95% CI from the corrected test):

| Protocol | A − B | Mean | 95% CI | Interval excludes 0 |
|---|---|---|---|---|
| random_grouped | RF reference − XGBoost | 0.010 | −0.013 to 0.033 | No |
| random_grouped | RF reference − RF tuned | −0.002 | −0.009 to 0.005 | No |
| random_grouped | RF tuned − XGBoost | 0.012 | −0.010 to 0.034 | No |
| random_grouped | Lasso − RF reference | 0.280 | 0.237 to 0.324 | Yes |
| random_grouped | Lasso − XGBoost | 0.290 | 0.243 to 0.337 | Yes |
| scaffold | RF reference − XGBoost | 0.019 | −0.009 to 0.047 | No |
| scaffold | RF reference − RF tuned | 0.005 | −0.003 to 0.013 | No |
| scaffold | RF tuned − XGBoost | 0.014 | −0.014 to 0.042 | No |
| scaffold | Lasso − RF reference | 0.204 | 0.115 to 0.294 | Yes |
| scaffold | Lasso − XGBoost | 0.224 | 0.150 to 0.298 | Yes |

Every model's interval against the baseline excludes zero. The full table, including MAE, is in `paired_comparisons.csv`.

**Selected hyperparameters (across 30 outer folds)**
- Lasso: `alpha=0.001` in 25 folds and `alpha=0.01` in 5.
- XGBoost: `max_depth=3` in 29 folds.
- Random Forest tuned: seven different combinations were selected, none in more than 5 of 15 folds per protocol.

**Fold sizes**
- random_grouped: 225–226 test rows per fold.
- scaffold: 218–219 test rows per fold, except the benzene fold with 254.

## Verification

- 73 backend tests pass: 43 existing, 11 from Phase 6B and 19 new in `tests/test_cross_validation.py`. The 12 frontend tests pass.
- The new tests check:
  - groups stay intact in outer and inner folds, and every row is tested exactly once per repeat;
  - the largest group appears in exactly one test fold;
  - fold sizes are balanced, and the splitter is reproducible for a fixed seed;
  - an instrumented estimator shows that no inner fit and no refit ever sees outer test rows;
  - all models share the same outer folds;
  - the full procedure is reproducible;
  - the corrected t-test matches a hand calculation, and mismatched folds are rejected.
- Two complete runs produced identical `fold_metrics.csv`, `fold_assignments.csv` and `paired_comparisons.csv`, apart from timing columns.

## Limitations and unresolved issues

1. **Repeated benzene fold.** Benzene (254 rows) is larger than the target fold size (≈ 226), so in every repeat it forms a test fold on its own. Repeats 0, 1 and 2 therefore contain an identical benzene-only fold with identical training data. That gives 13 distinct scaffold folds rather than 15, and the corrected test treats them as 15.
   - Sensitivity check with the duplicates removed (13 folds): RF reference − XGBoost 0.016 (95% CI −0.013 to 0.045), RF reference − RF tuned 0.004 (−0.004 to 0.011), Lasso − RF reference 0.219 (0.135 to 0.303).
   - The same intervals exclude zero as in the main analysis. This check is not yet part of the saved outputs.
2. **Grid boundaries.** Lasso selects the smallest `alpha` and XGBoost the smallest `max_depth` in almost every fold, so their optimum may lie outside the pre-specified grids. Widening the grids after seeing results would be post-hoc; any extension should be pre-specified in the next phase.
3. **Unstable Random Forest tuning.** The tuned Random Forest selects different hyperparameters across folds, and its paired difference from the untuned reference is within ±0.01 RMSE. The grid may not span settings that matter for this dataset.
4. **Scaffold-fold heterogeneity.** The benzene-only fold contains a single chemical series. The scaffold baseline RMSE sd (0.409) is about five times the random-protocol sd, which reflects differences in target distribution between folds.
5. **Statistical test assumptions.** The Nadeau–Bengio correction is an approximation designed for random resampling. It has not been validated for grouped or scaffold folds. 10 pairs × 2 metrics × 2 protocols are tested with no multiplicity adjustment.
6. **Unchanged data caveats from Phase 6A.** The six descriptors cannot distinguish 304 rows. The hexitol duplicate pair with conflicting values (1.03 log units) is retained, in line with the no-filtering rule.
7. **Environment.** XGBoost on macOS needs `libomp` (`brew install libomp`). Results were produced with scikit-learn 1.9.1, XGBoost 3.4.1 and RDKit 2026.3.6; other versions may change results slightly.
