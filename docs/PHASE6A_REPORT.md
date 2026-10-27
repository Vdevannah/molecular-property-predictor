# Phase 6A — Baseline Audit and Evaluation Protocol

## Scope

Phase 6A audits the existing ESOL data, the Random Forest training workflow, and the random-split and Murcko scaffold-split evaluations before Lasso and XGBoost are introduced. No new algorithms were implemented. No source files, saved models, or result files were changed; MD5 hashes of `models/*.joblib` and `results/phase{3,4}_metrics.json` were identical before and after the audit.

## 1. Baseline reproduction

All existing results reproduce exactly from `data/esol_descriptors.csv` (differences are at floating-point precision, ~1e-16).

| Evaluation | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| Phase 3 random split (80/20, seed 42) | Mean baseline | 1.732 | 2.174 | −0.000 |
| Phase 3 random split (80/20, seed 42) | Random Forest | 0.574 | 0.835 | 0.853 |
| Phase 4 scaffold split (80/20, seed 42) | Random Forest | 0.616 | 0.880 | 0.795 |

- The saved `models/solubility_model.joblib` produces predictions identical to a fresh retrain (`n_estimators=200`, `max_depth=None`, `max_features=1.0`, `random_state=42`).
- Backend tests: 43 passed. Frontend tests: 12 passed.

## 2. Data quality findings

| Check | Result |
|---|---|
| Rows (raw / descriptor file) | 1,128 / 1,128 |
| Missing or non-finite values | 0 |
| Invalid SMILES | 0 |
| SMILES with trailing whitespace | 212 (harmless to RDKit, but should be stripped before any string matching) |
| Descriptors recomputed from SMILES | 0 mismatches against the CSV |
| Duplicate structures (canonical, stereo-aware) | 11 groups, 22 rows |
| Duplicates with conflicting targets | 1 large conflict: `OCC(O)C(O)C(O)C(O)CO` differs by 1.03 log units (hexitol stereoisomers whose stereochemistry is not encoded); 5 small (≤ 0.24) |
| Rows sharing an identical 6-descriptor vector with another molecule | 304 rows in 115 groups (mostly positional isomers) |
| Target (LogS) | mean −3.05, sd 2.10, range −11.60 to 1.58 |
| Strongest feature–target correlations | LogP −0.83, MolWt −0.64 |
| Strongest feature–feature correlations | HBA–TPSA 0.90, HBD–TPSA 0.76 |

The 304 rows that share a descriptor vector cannot be told apart by any model trained on these six features, which sets an irreducible error floor for this feature set.

## 3. Leakage assessment

| Risk | Finding | Severity |
|---|---|---|
| Target-derived feature | The raw file contains `ESOL predicted log solubility`. It is **not** used as a feature (verified). | None today; it must remain excluded |
| Duplicate structures across random split | 4 test molecules also appear in training (e.g. `COc1ccccc1O`, identical LogS). | Low (4/226), but optimistic |
| Duplicates across scaffold split | 0. Identical structures share a scaffold, so they are always kept together. | None |
| Preprocessing fit on full data | None at present. Random Forest needs no scaling. | Will matter for Lasso |
| Hyperparameters chosen on test set | None at present (fixed defaults). | Will matter for GridSearchCV |

## 4. Weaknesses in the current evaluation

1. **Results from a single split are dominated by split noise.** Repeating the identical Random Forest setup with split seeds 0–4 gives:
   - Random split RMSE: 0.615–0.792
   - Scaffold split RMSE: 0.640–0.831

   The two ranges overlap, and seed 42 is a pessimistic draw for the random split. A single holdout cannot reliably separate models whose RMSEs differ by less than about 0.15.
2. **The greedy scaffold split is biased.** The benzene scaffold (254 molecules, 22% of the data) is larger than the 226-molecule test budget, so it can never enter the test set. The largest scaffold that reached the test set has 16 molecules. The scaffold test set is enriched in singleton and acyclic molecules, and its mean LogS (−2.78) differs from the training mean (−3.12). It measures generalisation to rare scaffolds only.
3. **There is no internal validation set.** Any hyperparameter tuning that reuses the seed-42 test split would leak test information into model selection.
4. **Minor documentation drift.** `phase4_metrics.json` says acyclic groups are formed "based on atom count", but `get_scaffold_group` actually uses canonical SMILES.

## 5. Proposed evaluation protocol for Lasso, Random Forest and XGBoost

**Data preparation (in memory, evaluation only)**
- Strip whitespace from SMILES and assign each row a canonical-SMILES group ID.
- Keep duplicates in the data but use the group IDs so that a structure never appears on both sides of any split.
- Exclude the hexitol group from the conflicting-duplicate check (spread > 0.5 log units) and report it separately.
- Features: the existing six descriptors only. Never use `ESOL predicted log solubility`.

**Outer evaluation: nested cross-validation under two protocols**
- **Random protocol:** `GroupKFold(n_splits=5, shuffle=True)` on canonical-SMILES groups, repeated with 3 seeds (15 outer folds).
- **Scaffold protocol:** `GroupKFold(n_splits=5, shuffle=True)` on Murcko scaffold groups, repeated with 3 seeds. Every scaffold, including benzene, is tested exactly once per repeat, which removes the greedy-split bias.
- **Paired design:** within each protocol, all models use exactly the same outer folds.

**Inner model selection**
- `GridSearchCV` with 5 grouped folds on the outer-training data only, using the same grouping as the outer protocol.
- Scoring: `neg_root_mean_squared_error`.
- All preprocessing goes inside a `Pipeline`; Lasso is `StandardScaler → Lasso`.

**Candidate models and grids (small and fixed in advance)**

| Model | Grid |
|---|---|
| Mean baseline | none |
| Random Forest, current config | none (reference) |
| Lasso | `alpha` ∈ {1e-3, 1e-2, 1e-1, 1} |
| Random Forest, tuned | `n_estimators` ∈ {200, 500}; `max_depth` ∈ {None, 10, 20}; `min_samples_leaf` ∈ {1, 3} |
| XGBoost | `n_estimators` ∈ {200, 500}; `max_depth` ∈ {3, 6}; `learning_rate` ∈ {0.05, 0.1}; `subsample` = 0.8; `tree_method="hist"` |

**Reporting and decision rule**
- **Primary metric:** RMSE under the scaffold protocol. **Secondary:** MAE and R², plus all metrics under the random protocol.
- Report the mean ± sd across outer folds, per-fold values, and each fold's selected hyperparameters (to show how stable the selection is).
- A model is called "better" only if its paired per-fold RMSE difference against the reference Random Forest has a 95% interval that excludes zero, using the Nadeau–Bengio corrected resampled t-test. Otherwise the models are reported as statistically indistinguishable.
- The Phase 3 seed-42 holdout is reported once per model for continuity only. It is not used for model selection.
- Outputs go to a new `results/phase6_model_comparison.json`. The existing files are not overwritten.

## 6. Preserving application behaviour

- `src/api.py`, `frontend/`, `models/solubility_model.joblib`, `models/feature_columns.joblib`, and the Phase 3 split logic used for the API's training/test labels stay unchanged.
- `train_models()` keeps returning `baseline` and `random_forest` with their current configuration.
- Any change to which model is served is a separate decision, made after Phase 6B results are reviewed.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Small dataset (1,128 molecules) gives high-variance estimates | Repeated nested cross-validation; report variance; corrected paired test |
| Scaffold folds are unequal in size because the benzene group is large | Report per-fold sizes; weight metrics by fold size |
| HBA/TPSA collinearity makes Lasso coefficients unstable | Interpret Lasso coefficients cautiously; predictions are not affected |
| Six descriptors cap achievable accuracy (304 indistinguishable rows) | State this limit explicitly; richer features are out of scope |
| `xgboost` is not installed; on macOS it may need `libomp` | Pin the version in `requirements.txt`; verify import in the test suite |
| Nested CV runtime (~1,500–2,500 fits) | Use `n_jobs=-1`; use reduced grids in unit tests |
| Nondeterminism in tree models | Fix `random_state` everywhere; use `tree_method="hist"` for XGBoost |

## 8. Acceptance criteria

**Phase 6A (this phase)**
- [x] Phase 3 and Phase 4 metrics reproduced exactly
- [x] Data quality and leakage audit documented
- [x] Existing tests pass (43 backend, 12 frontend); saved artifacts unchanged
- [x] Evaluation protocol, risks and acceptance criteria defined

**Phase 6B (implementation)**
- All existing tests pass unchanged, and API responses are identical for the same input.
- New tests confirm:
  - no group (canonical SMILES or scaffold) appears in both training and test of any outer or inner fold;
  - scaling is fit only inside the pipeline;
  - grids are searched only on outer-training data;
  - results are reproducible with fixed seeds.
- The reference Random Forest, run through the new pipeline on the seed-42 holdout, reproduces RMSE 0.835.
- `phase6_model_comparison.json` contains per-fold metrics, selected hyperparameters and paired-test results for all five candidates under both protocols.
- Conclusions follow the decision rule in Section 5.
