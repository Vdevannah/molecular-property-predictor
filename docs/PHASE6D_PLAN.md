# Phase 6D Plan — Scaffold Generalization and Robustness

**Status:** prespecified, not yet run. **Configuration:** [`configs/phase6d.json`](../configs/phase6d.json) is the authoritative, machine-readable version of this plan. If the two ever disagree, the configuration wins and the discrepancy is recorded under *Amendments*.

## 1. Purpose

Phase 6C compared the models with grouped nested cross-validation, but left four problems open:

- the benzene-only test fold was repeated in all three repeats;
- fold sizes were unequal, and only fold-averaged metrics were reported;
- several hyperparameters were selected at the edge of their grids;
- the paired t-test assumes resampling of independent samples, which grouped folds violate.

Phase 6D asks how well the models generalize to Bemis–Murcko scaffolds they have not seen. It addresses each problem above, prespecifies one confirmatory comparison, and treats everything else as exploratory.

Phase 6C stays the original prespecified result. Phase 6D is a follow-up designed after seeing those results, and is labelled as such.

## 2. Decisions approved on 2026-10-27

| Topic | Decision |
|---|---|
| Practical-equivalence margin | δ = 0.05 LogS, provisional (justification and limitations in Section 6.3) |
| Generic scaffolds | Included as an exploratory robustness analysis |
| Repeats | 5 repeats of the non-benzene scaffold CV |
| Tracked artifacts | Small JSON in `reports/phase6d/`; large outputs in `results/phase6d/<run_id>/` (Git ignored) |
| Benzene holdout vs non-benzene CV | Separate protocols, reported separately; any pooled number is labelled a *composite out-of-fold result* |
| Inner-CV score | Molecule-weighted pooled RMSE, verified by a hand-calculated unit test |
| Repeated folds | Never treated as independent observations |
| Equivalence wording | Always conditional on δ and on the validity of the uncertainty method |
| Acyclic molecules | Reported explicitly, with the limitations of identity grouping |
| Preservation | The production model, API, frontend, and Phase 3/4/6C results stay unchanged (checksum-verified) |

## 3. Data

- `data/esol_descriptors.csv`: 1,128 rows, MD5 `1a17c759…`. The run aborts if the checksum doesn't match.
- Features: the six RDKit descriptors. Target: measured log solubility.
- The `ESOL predicted log solubility` column is forbidden as a feature.
- No filtering in the primary analysis (variant S0). Data variants appear only in the prespecified sensitivity analysis (Section 9).

## 4. Evaluation protocols

| Protocol | Role | Test population | Training population | Repeats |
|---|---|---|---|---|
| **Non-benzene scaffold CV** | **Primary** | 874 non-benzene molecules in 583 Murcko groups, 5 balanced group folds | Benzene (254 molecules) plus the other four folds | 5 (outer seed 42 + repeat) |
| **Benzene holdout** | Secondary, reported separately | All 254 molecules with Murcko scaffold `c1ccccc1` | All 874 non-benzene molecules | 1 (deterministic) |
| Generic block holdout | Exploratory | Generic group `C1CCCCC1` (311 molecules) | The other 817 molecules | 1 |
| Generic non-block scaffold CV | Exploratory | 817 molecules in 481 generic groups | The block plus the other four folds | 5 |

**Why separate the two Murcko protocols.** Benzene (254 molecules) is larger than any balanced fold. In Phase 6C it therefore always formed the same benzene-only fold. The two protocols ask different questions:

- *Non-benzene CV:* given the dominant scaffold and most others, how well does a model predict unseen minor scaffolds?
- *Benzene holdout:* how well does a model predict an entire dominant family it has never seen?

**Composite out-of-fold result.** This is reported only as a descriptive point estimate, labelled *"Composite out-of-fold result (benzene holdout + non-benzene scaffold CV); not a single cross-validation estimate"*. It is computed as:

√[(Σ benzene-holdout squared errors + Σ non-benzene repeat-averaged squared errors) / 1,128]

No inference is attached to it.

**Guarantees, each covered by a test.**
- No Murcko group appears in both training and test, in outer or inner folds.
- Every non-benzene molecule is tested exactly once per repeat.
- Benzene is in training for every non-benzene fold and is tested exactly once overall.

## 5. Nested tuning

**Inner splits.** For each outer fold, 5 balanced group folds are formed over the outer-training groups, excluding the protocol's block group. The block group (benzene or generic `C1CCCCC1`) is placed in every inner training fold and never in an inner validation fold, so tuning mirrors the outer design. The splits are passed to `GridSearchCV` as an explicit list, which makes the validation-fold sizes nₖ known.

**Selection criterion.** Candidates are ranked by molecule-weighted pooled RMSE:

RMSE_pooled = √( Σₖ nₖ · RMSEₖ² / Σₖ nₖ ) = √( total squared error / total validation molecules )

This is implemented as a `refit` callable on `cv_results_`. Ties go to the lowest candidate index. Phase 6C instead used the unweighted mean of fold RMSEs, which gave each small fold the same weight as a large one.

**Required hand-calculated test.** With validation-fold RMSEs of 1.0 and 2.0 and fold sizes 3 and 1:

- pooled RMSE = √((3·1² + 1·2²)/4) = √1.75 = **1.3229**
- the unweighted mean would be 1.5, and √(mean MSE) would be 1.5811

The test must reproduce 1.3229. It must also show that the value equals √(SSE/N) computed directly from raw predictions, and that the refit callable selects the candidate with the lowest pooled score even when the unweighted mean would pick a different one.

## 6. Statistical analysis

### 6.1 Metrics
- **Per-repeat pooled RMSE:** √(mean over test molecules of e²) for each repeat.
- **Repeat-averaged RMSE (primary statistic):** √(mean over molecules of [mean over repeats of e²]). Each molecule contributes exactly once.
- **Fold-averaged RMSE:** the unweighted mean of outer-fold RMSEs, for continuity with Phase 6C. Reported beside the molecule-weighted value, with the difference explained.
- **Also reported:** MAE, mean signed error, R² (whole protocols only), and molecule and group counts.

### 6.2 Contrasts
- **Primary (confirmatory):** in non-benzene scaffold CV with S0 data, Δ = RMSE(production-config Random Forest) − RMSE(XGBoost). Positive means XGBoost has lower error.
- **Secondary:** RF reference vs RF tuned, RF tuned vs XGBoost, and Lasso vs RF reference, with Holm adjustment across the three.
- **Exploratory:** everything else, including other protocols, families, sensitivity variants and the remaining model pairs. These get intervals but no significance language.
- **Supporting statistics for the primary pair:** the correlation between per-molecule errors, the per-molecule win rate, per-repeat Δ (split variability), and Δ under model seeds 42–45.

### 6.3 Practical-equivalence margin δ = 0.05 LogS (provisional)

**Justification**
- Commonly cited estimates of experimental uncertainty in aqueous log solubility are about 0.5 LogS or more. δ is roughly one-tenth of that.
- δ is also about 6% of the Phase 6C scaffold RMSE (≈ 0.82).
- A difference smaller than δ is unlikely to change any practical use of a solubility prediction.

**Limitations**
- δ is a judgement, not derived from a specific decision context.
- It is applied to a difference in RMSE, an aggregate, not to individual predictions.
- Different values would be defensible.
- Given the variability seen in Phase 6C, the 95% interval may be wider than 2δ, in which case the result can only be *inconclusive*.
- A 95% two-sided interval is more conservative than the usual 90% interval of two one-sided equivalence tests.

**Decision rule** (primary and secondary contrasts only):

| 95% interval for Δ | Label |
|---|---|
| Within (−δ, δ) and includes 0 | Practically equivalent at δ = 0.05, conditional on the bootstrap assumptions |
| Within (−δ, δ) and excludes 0 | Detectable difference smaller than δ |
| Excludes 0 and lies entirely beyond ±δ | Difference larger than δ |
| Excludes 0 and crosses ±δ | Difference detected; magnitude relative to δ uncertain |
| Otherwise | Inconclusive |

Every label is reported with the phrase *"conditional on δ = 0.05 LogS and on the validity of the scaffold-cluster bootstrap"*.

### 6.4 Uncertainty: scaffold-cluster bootstrap

**Procedure**
1. For each molecule, average its squared error over the 5 repeats. Repeats are combined *before* resampling and are never counted as separate observations.
2. Resample Murcko groups (583 in the primary protocol, the largest with 39 molecules) with replacement, 10,000 times (seed 20261027).
3. Recompute Δ for each replicate and take the 95% percentile interval. Two-sided p = min(1, 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0))).

**Assumptions and limitations**
- Scaffold groups are treated as independent, exchangeable units. Related scaffolds (for example pyridine and benzene, or steroid variants) are not independent, so intervals may be too narrow.
- The bootstrap is conditional on the fitted models. It reflects variation from *which molecules and scaffolds were sampled*, not from retraining on a different dataset. Repeat averaging captures some split variability, but the five repeats share most of their training data, so their spread underestimates training-set variability.
- Percentile intervals can under-cover when the distribution is skewed or a few large clusters dominate. Cluster-size heterogeneity is reported.
- **Benzene holdout and generic block holdout** use a molecule-level bootstrap within the single held-out block, conditional on one training set. Molecules within one scaffold family are not independent, so these intervals are descriptive only.
- **The Phase 6C Nadeau–Bengio test** is recomputed on distinct folds for continuity only. It was derived for random resampling of independent samples and is not used for conclusions.

## 7. Scaffold families (S0, Murcko)

| Family | Molecules |
|---|---|
| 9 named scaffolds with ≥ 10 molecules: benzene 254, biphenyl 39, naphthalene 22, barbiturate 21, steroid enone 17, 1,3,5-triazine 16, pyridine 14, diphenylmethane 12, cyclohexane 10 | 405 |
| Acyclic | 317 |
| Cyclic, Murcko group size 2–9 | 212 |
| Cyclic singletons | 194 |

- **Reported per family:** n, RMSE, MAE, mean signed error, and target mean, sd and range.
- **R² is not reported within families**, because it is misleading when a family's target variance is small.
- **Intervals** only for families with n ≥ 20: molecule bootstrap within named families, cluster bootstrap within aggregate families.
- Per-family RF vs XGBoost differences are exploratory.
- Benzene's family result comes from the benzene-holdout protocol. All other families' results come from non-benzene CV.

### 7.1 Acyclic molecules

**Limitation.** An acyclic molecule has no Murcko scaffold, so it is grouped only by its own canonical SMILES (316 groups for 317 molecules). For these 28% of the data, "scaffold" CV is therefore an identity-grouped random split. Close analogues, such as neighbouring members of a homologous alcohol or alkane series, can sit on opposite sides of a split. Acyclic performance is likely to be optimistic relative to genuinely unseen chemistry. Generic scaffolds do not change this.

**What is reported**
- Acyclic RMSE, MAE and bias in non-benzene CV.
- The maximum Tanimoto similarity of each acyclic test molecule to its training set, compared with cyclic families. This quantifies how "unseen" these molecules really are.

No alternative acyclic grouping (for example by functional class) is introduced in Phase 6D; that would be a new, unvalidated design.

## 8. Generic-scaffold robustness (exploratory)

- The same two-protocol design is used with generic scaffolds (atom and bond types removed): block `C1CCCCC1`, 311 molecules covering all single six-membered-ring compounds, plus non-block CV over 481 groups.
- Outputs: the same metrics and an exploratory Δ for RF reference vs XGBoost. No decision-rule label is applied.

## 9. Duplicate sensitivity analysis (prespecified)

| Variant | Data | Rows |
|---|---|---|
| S0 (primary) | All rows | 1,128 |
| S1 | Mannitol (LogS 0.06) and sorbitol (1.09) removed. They are distinct stereoisomers whose SMILES lack stereochemistry, so they are indistinguishable to the model. | 1,126 |
| S2 | Each of the 11 identity-duplicate groups collapsed to its mean target; groups with a range > 0.5 LogS dropped (only the hexitol pair) | 1,116 |

- S1 and S2 are run through both Murcko protocols with the same seeds and design.
- S0 predictions are also scored on the S1 and S2 molecule subsets, separating the effect of the data change from the effect of a different molecule set.
- **Robustness criterion (prespecified):** the primary contrast gets the same decision-rule label in S0, S1 and S2.

## 10. Hyperparameter grids

| Model | Phase 6C grid → selection | Phase 6D grid |
|---|---|---|
| Lasso | α ∈ {0.001 … 1}; α = 0.001 in 25/30 folds | α ∈ {1e-5, 3e-5, …, 0.3, 1} (11 half-decade values), `max_iter=100000`, plus an unregularized `LinearRegression` reference |
| XGBoost | `max_depth` ∈ {3, 6}; depth 3 in 29/30 folds | `max_depth` {2, 3, 4} × `learning_rate` {0.03, 0.1} × `n_estimators` {200, 500, 1000} × `min_child_weight` {1, 5} (36 candidates) |
| RF tuned | `max_depth` and `min_samples_leaf`; seven different selections | `max_features` {1.0, 0.5, 0.33} × `min_samples_leaf` {1, 3, 5}, 500 trees, `max_depth=None` (9 candidates) |
| RF reference | Production config | Unchanged |
| Baseline | Mean | Unchanged |

**Rationale**
- The Lasso and XGBoost grids are extended past the boundaries Phase 6C hit.
- If Lasso's optimum approaches α → 0, the model is effectively ordinary least squares. The `LinearRegression` reference makes that visible.
- `max_features` was never searched before; 1.0 means plain bagging, while decorrelating the trees is the main lever for random forests.
- `max_depth` showed no consistent signal, so it is fixed.

**Boundary check.** For each numeric grid dimension, report the fraction of outer folds that select its minimum or maximum value. Anything above 0.5 is flagged and reported. Grids are not extended again within Phase 6D.

## 11. Exploratory analyses

- **Error floor:** 304 rows share an identical six-descriptor vector with another molecule. Their within-group target variance gives a lower bound on RMSE achievable with these features.
- **Descriptor distance:** each test molecule's distance to its nearest training molecule (descriptors standardized on outer-training data), with the Spearman correlation against absolute error. A scaffold the model hasn't seen is not necessarily far away in descriptor space.
- **Structural similarity:** maximum Morgan-fingerprint Tanimoto similarity to the training set, summarized by family.
- **Model-seed check:** RF reference and XGBoost refit with `random_state` 43–45 (no re-tuning; XGBoost keeps its per-fold S0 hyperparameters). Δ is reported for seeds 42–45.

## 12. Reproducibility

| Item | Location |
|---|---|
| Prespecified configuration | `configs/phase6d.json` (tracked). Its MD5 is stored in every run. |
| Feasibility estimate | `reports/phase6d/feasibility_estimate.json` (tracked) |
| Code (planned) | `src/scaffold_evaluation.py`; `tests/test_scaffold_evaluation.py`. Reuses `BalancedGroupKFold` and the model builders. `src/cross_validation.py` is not modified, so Phase 6C stays reproducible. |
| Large outputs | `results/phase6d/<run_id>/`: out-of-fold predictions, fold metrics, inner-CV results, fold assignments, bootstrap results, families, sensitivity, boundary report, similarity, `pip freeze`. The run refuses to overwrite an existing directory. |
| Tracked summaries | `reports/phase6d/*.json`: summary, contrasts, families, sensitivity, boundary report, run ID, config MD5, versions. JSON only, because `.gitignore` excludes `*.csv`. |
| Report | `docs/PHASE6D_REPORT.md` |
| Protection | Checksums of the production model, Phase 3/4 results and all Phase 6C outputs are verified before and after each run. The run aborts on any mismatch. |
| Environment | Python 3.12.15, scikit-learn 1.9.1, XGBoost 3.4.1, SciPy 1.18.1, RDKit 2026.3.6, `libomp` on macOS. `scipy==1.18.1` will be pinned in `requirements.txt` during implementation. |

## 13. Computational feasibility

Every grid candidate was timed single-threaded on real ESOL rows. The estimate was calibrated against Phase 6C's measured wall time (an effective speed-up of 4.2× on 8 cores).

| Quantity | Estimate |
|---|---|
| CPU time per outer fold | 27.6 s (Phase 6C: 26.3 s). The smaller RF grid offsets the larger XGBoost grid. |
| Outer folds | 104 (26 per scenario × S0, S1, S2, generic) |
| Model fitting, wall time | ≈ 12 min (primary S0 alone ≈ 3 min) |
| Planning range including bootstrap and similarity | 12–20 min |
| Approval threshold | 60 min; stop and report if the benchmark exceeds it |

The full run will be preceded by a benchmark of one outer fold per protocol.

## 14. Planned tests

- **Protocol construction:**
  - no Murcko group appears in both training and test (outer and inner folds);
  - benzene is always in training for non-benzene CV and in no inner validation fold;
  - each non-benzene molecule is tested exactly once per repeat;
  - benzene is tested exactly once overall;
  - the block-group logic works for the generic protocol too.
- **Pooled inner RMSE:** the hand-calculated 1.3229 case; equality with √(SSE/N) from raw predictions; the refit callable choosing the pooled-best candidate where the unweighted mean disagrees.
- **Repeat averaging:** a molecule with repeated predictions contributes one averaged squared error.
- **Bootstrap:**
  - resamples whole groups;
  - reproducible for a fixed seed;
  - Δ = 0 for identical models;
  - known distributions recover known intervals.
- **Decision rule:** every branch of the δ table.
- **Sensitivity variants:** S1 removes exactly the two hexitol rows; S2 yields 1,116 rows and aborts if descriptors differ within a duplicate group.
- **Families:** the counts add up to 1,128; no R² within families.
- **Safety:** the run refuses to overwrite an existing directory and aborts on any checksum mismatch.
- **Determinism:** a reduced end-to-end run is repeated and produces identical outputs.

## 15. Acceptance criteria

1. All existing tests (73 backend, 12 frontend) and all new tests pass.
2. The production artifacts, Phase 3/4 results and Phase 6C outputs are byte-identical before and after the run.
3. Two full runs produce identical outputs apart from timings.
4. The configuration's MD5 at the final run matches the one recorded at the benchmark (or an amendment is documented).
5. The report presents the two Murcko protocols separately. Any pooled number is labelled as the composite out-of-fold result.
6. The primary contrast is reported with its 95% scaffold-cluster bootstrap interval, its decision-rule label, and the conditional wording from Section 6.3.
7. Molecule-weighted and fold-averaged RMSE are both reported, with the difference explained.
8. Acyclic performance is reported with the identity-grouping limitation and similarity evidence.
9. The S0–S2 sensitivity results, the boundary report, the model-seed range and the generic-scaffold results are all reported.
10. No claim of superiority or equivalence goes beyond what the prespecified rule supports.
11. Nothing is committed or pushed without explicit approval.

## Amendments

None yet.
