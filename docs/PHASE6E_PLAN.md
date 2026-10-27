# Phase 6E Plan — Scientific Interpretation and Model Diagnostics

**Status:** prespecified, not yet run. **Configuration:** [`configs/phase6e.json`](../configs/phase6e.json) is the authoritative, machine-readable version of this plan; if the two ever disagree, the configuration wins and the discrepancy is recorded under *Amendments*.

## 1. Purpose and role

Phase 6E explains *how* the Phase 6D models behave: where they err, which descriptors their predictions rely on, and where they are unreliable. It relates those patterns to chemistry as **hypotheses**.

It is **diagnostic and descriptive only**:
- no model selection;
- no confirmatory tests;
- no revision of the Phase 6D decision labels. These are quoted, never strengthened. For example, the primary contrast stays *"practically equivalent at δ = 0.05 LogS, conditional on δ and on the validity of the scaffold-cluster bootstrap"*.

## 2. Decisions approved on 2026-10-27

| Topic | Decision |
|---|---|
| Deployed production Random Forest | Included as read-only diagnostics on its Phase 3 random split, labelled separately |
| XGBoost TreeSHAP | Included as an exploratory interpretation method |
| Per-prediction intervals | RF tree spread and cross-fold residual intervals, reported as **exploratory empirical diagnostics** with observed (not guaranteed) coverage |
| ESOL-equation predictions | Included as an explicitly **in-sample, descriptive comparator** |
| Tracked figures | About 5 MB of PNGs in `docs/images/phase6e/` |

**Required refinements and where they are addressed**

| Refinement | Section |
|---|---|
| Cross-fold intervals are exploratory diagnostics; observed coverage only | 11 |
| Pooled permutation and drop-column importance from prediction-level squared errors; fold variability reported separately | 6 |
| Representation floor defined as a dataset-specific limitation of the six descriptors | 10 |
| ESOL equation is an in-sample comparator; shared biases prompt investigation, not proof | 8, 9 |
| Refits use the exact Phase 6D training sets, hyperparameters, preprocessing and seeds; any mismatch stops the analysis | 3 |
| Benzene holdout and non-benzene scaffold CV kept separate in all statistics | 4 |
| Small families never treated as statistically established effects | 8 |

## 3. Inputs, integrity and the refit gate

**Inputs (read-only)**
- Phase 6D run `results/phase6d/20261027T154454Z_d02c301c` (the run that the tracked `reports/phase6d/` files refer to) and `configs/phase6d.json`.
- `data/esol_descriptors.csv`, plus `data/esol.csv` for the ESOL-equation column.
- The two data files are aligned by row position. The run stops unless every `Compound ID` and `smiles` matches; the feasibility check confirmed they do.

**Protected files.** 52 files are checksum-verified before and after the run. They are:
- the production model and feature list;
- the Phase 3/4 results, all Phase 6C outputs, the Phase 6D config, run files and reports;
- both data files and `requirements.txt`;
- every `src/*.py` file and the frontend sources.

**Refit gate.** Feature importance, ALE, TreeSHAP and tree spread need fitted models, which Phase 6D did not save. Each fold model (25 non-benzene CV folds plus the benzene holdout) is refitted using:
- the exact Phase 6D fold indices, via `scenario_inputs(…, "murcko_S0", …)`;
- the exact estimators and preprocessing, via `build_model_specs(configs/phase6d.json)`;
- the hyperparameters recorded for that fold;
- the configured seeds.

The refit predictions must match the stored out-of-fold predictions within **10⁻⁹**; otherwise the run stops before computing any diagnostic. The feasibility check measured a maximum difference of **1.8 × 10⁻¹⁵** for all four models.

**Production model.** It is loaded read-only and must reproduce its Phase 3 test RMSE of 0.8349452807 within 10⁻⁹.

## 4. Populations and per-molecule quantities

**Non-benzene scaffold CV** (874 molecules, 25 outer folds) is the primary diagnostic population. **The benzene holdout** (254 molecules, one fold) is reported separately.

- **No pooling across protocols.** Every statistic is computed per protocol. A figure showing both uses distinct markers (circle for non-benzene CV, triangle for benzene) and states each protocol's statistics separately. Any combined display is labelled *composite out-of-fold (descriptive)*.
- **Error metrics** use repeat-averaged squared errors per molecule, as in Phase 6D.
- **Residual and calibration plots** use the per-molecule mean prediction over the 5 repeats; the residual is that mean prediction minus the observed value.

**Uncertainty**
- Non-benzene CV uses the scaffold-cluster bootstrap over Murcko groups.
- The benzene holdout uses a molecule bootstrap, conditional on its single training set and descriptive only.
- All bootstraps use 10,000 replicates (seed 20261027) and 95% percentile intervals.

## 5. Observed vs predicted, calibration, residuals and extreme errors

- **Observed vs predicted:** RF reference, XGBoost and Lasso panels, each with an identity line, ±1 LogS guides and per-protocol RMSE.
- **Calibration:** ordinary least-squares fit of observed on predicted (slope and intercept with intervals), plus binned calibration (deciles for non-benzene CV, quintiles for benzene).
- **Shrinkage** is reported separately from calibration: the slope of the residual against the *observed* value. Tree models pull extremes towards the mean, so residuals plotted against observed values show a trend even for a well-calibrated model.
- **Residual distributions:** 1–99% quantiles, skewness and excess kurtosis, and Q–Q plots against a normal distribution.
- **Binned bias:** mean signed residual by decile of predicted value, observed value and each descriptor. Integer descriptors use deduplicated quantile edges.
- **Extreme errors:** the top 25 molecules by RF-reference |residual|, plus any molecule with |residual| > 2 LogS under RF reference, RF tuned or XGBoost. Each row lists the compound, family, observed value, every model's prediction, the ESOL-equation prediction, the nearest training neighbour and its similarity, the descriptor distance, range flags, and any duplicate or identical-descriptor group. **Flag only; nothing is corrected or removed.**

## 6. Feature importance (RF reference and XGBoost)

**Pooled metric (refinement 2).** For each permutation repeat *p*:
1. collect the prediction-level squared error of every test molecule, each predicted by its own fold model;
2. average each molecule's squared error over the 5 CV repeats;
3. RMSEₚ = √(mean over molecules);
4. importanceₚ = RMSEₚ − RMSE(unpermuted), computed the same way.

Report the mean and sd over *p*, plus a cluster-bootstrap interval from per-molecule squared errors averaged over permutations. **Fold variability is reported separately:** the median, IQR and range of per-fold ΔRMSE across the 25 folds, and how often each feature or group ranks first. A weighted average of fold-level RMSE differences is *not* used. Drop-column importance uses the same pooled construction.

**Methods**

| Method | Settings | Limitation stated next to every result |
|---|---|---|
| Permutation, single feature | 30 repeats, permuted within each outer test fold; seeds from `SeedSequence([6105, protocol, repeat, fold, unit, p])` | Correlated features share credit and each looks less important; permuting creates unrealistic combinations (e.g. high TPSA with HBA = 0) |
| **Permutation, grouped** | Primary groups: polarity/H-bonding {TPSA, HBA, HBD}, size/flexibility {MolWt, RotBonds}, lipophilicity {LogP}. Sensitivity groups: {MolWt, LogP}, {TPSA, HBA, HBD}, {RotBonds} | The result depends on the grouping, so both groupings are shown |
| Drop-column (single and grouped) | Refit on the same rows with Phase 6D hyperparameters, not re-tuned | Hyperparameters were tuned with all six features |
| Impurity (RF) / total gain (XGBoost) | Mean and sd across fold models | Biased towards continuous features; shown only for continuity with Phase 4 |
| XGBoost TreeSHAP | `pred_contribs=True`; additivity tolerance 10⁻⁴ (float32) | Path-dependent attribution under correlated features; not causal |
| Lasso standardized coefficients | Median, IQR and sign consistency across folds | Unstable under collinearity |
| **ALE** (RF reference, XGBoost, Lasso) | 20 quantile bins (unique-value edges for integer features); fold model evaluated on its own training rows; median and 10–90% band across 25 folds; benzene model separately | Unreliable at sparse range ends; describes the model, not chemistry |

The grouping is based on the observed correlations: TPSA–HBA 0.90/0.93, TPSA–HBD 0.76/0.66, MolWt–HBA 0.56, MolWt–LogP 0.48/0.54 (Pearson/Spearman).

## 7. Chemical interpretation of the descriptors

A reference table (T4) lists each descriptor's RDKit function, matched against `src/descriptors.py` at run time, and its general physical-chemistry expectation, stated as *background context, not a finding*. Beside these it shows the observed ALE direction and importance rank.

- **LogP is calculated, not measured.** `Descriptors.MolLogP` is Crippen's atom-contribution estimate, and its accuracy varies by chemotype.
- **What the six descriptors cannot represent:**
  - crystal-lattice energy or melting point;
  - ionization and pH;
  - stereochemistry and molecular symmetry;
  - substitution pattern (positional isomers produce identical descriptors).

## 8. Scaffold families

Families are as defined in Phase 6D; benzene results come from the benzene holdout and are labelled.

**For every family**
- forest plot of RMSE and mean signed error (intervals only for n ≥ 20);
- residual strip plots;
- a **shared-bias check**: the mean signed error of every model and of the ESOL equation;
- **extrapolation indicators**, all relative to the outer-training fold:
  - observed value below the training minimum;
  - observed value below the minimum of its 10 nearest training neighbours (standardized descriptors);
  - LogP above the training maximum;
  - any descriptor out of range.

**Small-family policy.** Triazines (n = 16) and pyridines (n = 14) get molecule-level tables and descriptive statistics only, with no intervals and no claims of established effects.

**Focus families.** Each hypothesis is paired with the diagnostic that could contradict it.

| Family | Phase 6D observation | Hypothesis | Contradicting diagnostic |
|---|---|---|---|
| 1,3,5-Triazine (16) | All models over-predict by about 1.5 LogS | Missing crystal-lattice/melting-point contribution; Crippen LogP error for this chemotype | The bias is *not* shared across model classes and the ESOL equation; the bias is explained by out-of-range descriptors; or it varies strongly with LogP within the family. Melting points are not in ESOL, so the lattice hypothesis is **untestable here** and is stated as such. |
| Pyridine (14) | All models under-predict by about 1.1 LogS | The basic, H-bond-accepting nitrogen is more solubilizing than the descriptors capture | The bias is not shared; the nearest training neighbours are not mostly carbocyclic analogues; the residual is unrelated to HBA/TPSA within the family |
| Biphenyl (39) | Lasso RMSE (0.68) below RF (1.20) and XGBoost (1.02) | Many members have target or LogP values beyond their training neighbourhood; tree models cannot extrapolate, while a linear model can | The extrapolation indicators are rare in this family; tree residuals do not grow with LogP; the Lasso advantage persists among non-extrapolating members |

A bias shared across models and the ESOL equation is **evidence for further investigation, not proof of a mechanism**. The ESOL equation was fitted on this dataset, so its comparison is in-sample.

## 9. Acyclic molecules, similarity and distance

- **Acyclic vs cyclic** (non-benzene CV): RMSE by Tanimoto bin (Morgan radius 2, 2048 bits). Bins are [0, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0), and exactly 1.0. For every acyclic test molecule, the nearest training neighbour is recorded (compound, SMILES, target; ties go to the lowest `row_id`).
- **Homologue heuristic:** flags pairs whose molecular formulas differ by exactly one CH₂. This is a descriptive count only.
- **Descriptor-space reliability:** the distance definition is identical to Phase 6D, and recomputed values must match the stored ones within 10⁻⁹.
  - RMSE and bias by distance quintile, per protocol;
  - selective-prediction curves (RMSE of the 10%…100% closest molecules);
  - a range flag (any descriptor outside the training range), evaluated only; the API is not changed.

## 10. Representation floor (refinement 3)

For rows grouped by identical six-descriptor vectors (rounded to 6 decimals; 115 groups, 304 rows):

floor RMSE = √(Σ within-group squared deviation from the group mean / 1,128) = **0.144**

This is the lowest RMSE *any* function of these six descriptors could reach on *this* dataset. It is **a dataset-specific limitation of the six-descriptor representation of ESOL**, not a universal irreducible error of aqueous solubility and not an estimate of measurement noise.

Measurement noise is **not estimated**. Identity duplicates are not treated as replicates, because 6 of the 10 non-conflicting pairs have exactly identical values, which suggests repeated entries. The 15 groups with the largest spread are listed (mostly positional isomers).

## 11. Interval diagnostics (refinement 1)

These are **exploratory empirical diagnostics. Observed coverage is reported, and no coverage guarantee is claimed.** A formally justified conformal procedure is not implemented.

**RF tree spread**
- Interval: ŷ ± z·sd at nominal 80/90/95%, where sd is the spread of individual tree predictions.
- Also reported: the Spearman correlation of sd with |error|.

**Cross-fold residual intervals** (RF reference and XGBoost)
- For repeat *r* and outer fold *k*, the half-width is the empirical quantile (`method="higher"`) of absolute residuals from the *other* folds of repeat *r*.
- Coverage is evaluated on fold *k* and averaged over repeats per molecule.

**Shift check:** calibrate on the non-benzene residuals, evaluate on the benzene holdout.

**Breakdown:** per protocol, by family and by distance quintile.

**Assumptions stated with every result:**
- exchangeability of scaffold groups, which is not established;
- dependence among molecules within a scaffold;
- residuals come from different fold models.

## 12. Production model diagnostics

The deployed `solubility_model.joblib` is loaded read-only and evaluated on its Phase 3 random split (test RMSE must reproduce exactly). Diagnostics:
- permutation importance, single and grouped;
- drop-column importance, refitting the production configuration in memory on the Phase 3 training rows (the file is never written);
- impurity importance;
- tree-spread interval diagnostics.

Every result is labelled *random-split diagnostics of the deployed model; not evidence of scaffold generalization*.

## 13. Figures and tables

**Style**
- Figures are PNG at 150 dpi on the Agg backend, with PNG metadata stripped so reruns are byte-identical. Total budget is about 5 MB.
- Model colors are constant across all figures: RF reference `#2a78d6`, XGBoost `#eb6834`, Lasso `#1baf7a`. They were validated with the data-visualization palette checker (all pairs pass the colorblind and normal-vision checks). Aqua is below 3:1 contrast, so every figure carries a legend and direct labels.
- Protocol is shown by marker shape. Focus-family colors appear only in F13, which has no model series. The correlation heatmap uses a blue–gray–red diverging scale.
- Every figure states its protocol(s), n, and whether it is out-of-fold or in-sample.

| # | Figure |
|---|---|
| F1 | Observed vs predicted LogS (RF reference, XGBoost, Lasso) |
| F2 | Binned calibration, with slope and intercept |
| F3 | Residual histograms and Q–Q plots |
| F4 | Residual vs predicted (calibration) and vs observed (shrinkage) |
| F5 | Binned bias vs each descriptor |
| F6 | Descriptor correlation matrices |
| F7 | Pooled permutation importance (single and grouped) with fold dispersion |
| F8 | Pooled drop-column importance |
| F9 | Importance-method rank comparison |
| F10 | ALE curves |
| F11 | Family forest plot |
| F12 | Family residual strip plots |
| F13 | LogS vs LogP, focus families highlighted |
| F14 | Biphenyl residual vs LogP by model |
| F15 | RMSE by similarity bin, acyclic vs cyclic |
| F16 | Distance reliability and selective-prediction curves |
| F17 | Representation-floor groups |
| F18 | Observed interval coverage |
| F19 | Production-model importance |

**Tables (`reports/phase6e/*.json`; Markdown versions in the report)**

| # | Table |
|---|---|
| T1 | Calibration |
| T2 | Extreme errors |
| T3 | Feature importance |
| T4 | Descriptor reference |
| T5 | Family diagnostics and ESOL comparator |
| T6 | Focus-family molecules |
| T7 | Similarity and distance |
| T8 | Representation floor |
| T9 | Interval coverage |
| T10 | Production model |
| `summary.json` | Run summary |

Tables are JSON because `.gitignore` excludes CSV.

## 14. Implementation, outputs and runtime

- **Code (not yet written):**
  - `src/model_diagnostics.py`, which imports from `src/scaffold_evaluation.py` and `src/train_model.py` without modifying them;
  - `tests/test_model_diagnostics.py`.
- **Large outputs:** `results/phase6e/<run_id>/`, never overwritten. A reproducibility re-run writes its figures and tables to an alternative directory.
- **Tracked outputs:** `reports/phase6e/`, `docs/images/phase6e/` and `docs/PHASE6E_REPORT.md`.
- **Dependencies:** no new ones (matplotlib 3.11.2 is already pinned).
- **Feasibility** (`reports/phase6e/feasibility_estimate.json`): about **9.2 minutes serial upper bound**, planning range 6–12 minutes, against a 60-minute approval threshold. The benchmark runs again before the full run.

## 15. Planned tests

- **Integrity:**
  - the refit gate passes on the real data, and a deliberately perturbed prediction stops the run;
  - the production RMSE check, the protected-file check and the raw/descriptor alignment check each stop the run on mismatch;
  - the run directory is never overwritten.
- **Pooled importance:**
  - a hand-calculated example where the pooled ΔRMSE differs from the weighted mean of fold ΔRMSEs;
  - an irrelevant feature gives ≈ 0;
  - a duplicated feature splits importance when permuted alone but recovers it as a group;
  - permutation stays within the test fold;
  - seeds are independent of execution order.
- **Drop-column:** keeps the stored hyperparameters, and an unrelated feature gives ≈ 0.
- **ALE:** for y = βx, the slope equals β and the curve is centred; integer-feature edges are correct.
- **TreeSHAP:** additivity within 10⁻⁴.
- **Calibration:** recovers a known slope and intercept; shrinkage is computed against observed values, calibration against predicted.
- **Binning:** configured edges, each molecule in exactly one bin, and Tanimoto = 1.0 detected.
- **Homologue heuristic:** a CH₂ difference is flagged; other formula changes are not.
- **Intervals:** half-widths never use the evaluated fold's residuals, and observed coverage matches a hand calculation.
- **Representation floor:** equals the hand calculation and 0.14399468 on ESOL.
- **Extreme errors:** selection follows the rule.
- **Protocol separation:** no statistic mixes molecules from both protocols.
- **Figures:** every configured figure exists, and two runs produce identical PNG hashes.

## 16. Acceptance criteria

1. **Nothing existing changes:** all existing tests (114 backend, 12 frontend) and all new tests pass, and all 52 protected files are byte-identical before and after.
2. **Refit gate:** passes for every model and fold (≤ 10⁻⁹), and the production model reproduces RMSE 0.8349452807.
3. **Reproducible:** two full runs produce identical outputs and identical PNG hashes.
4. **Complete:** all of F1–F19 and T1–T10 are produced with their configured settings, each labelled with protocol, n, and out-of-fold or in-sample.
5. **Pooled importance as specified:** importance comes from prediction-level squared errors, with fold variability reported separately and the correlation caveat beside every importance result.
6. **Protocol separation:** benzene holdout and non-benzene CV are never pooled in a statistic.
7. **Families:** under 20 molecules are described only; chemistry explanations are hypotheses with contradicting diagnostics; the ESOL comparison is labelled in-sample.
8. **Representation floor:** worded as a dataset-specific representation limitation; measurement noise not estimated.
9. **Interval coverage:** reported as observed, with stated assumptions, including where it fails.
10. **Phase 6D conclusions:** quoted, not strengthened; no new model ranking.
11. **No commit or push** without approval.

## Amendments

**Erratum (recorded 2026-10-27, Phase 6F).** Section 10, and the descriptive text of `representation_floor.measurement_noise` in `configs/phase6e.json`, state that "6 of the 10 non-conflicting pairs have exactly identical values". The correct count is **5 of 10**: of the 11 identity-duplicate groups, 5 have identical values and 6 differ, one of them the excluded hexitol pair. The conclusion is unchanged: identity duplicates are not treated as independent replicates, and measurement noise is not estimated. The frozen configuration was deliberately left unmodified because its MD5 (`e5e48d78…`) is recorded in every Phase 6E run output; this erratum supersedes its descriptive text.
