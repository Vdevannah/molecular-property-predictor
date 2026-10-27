# Phase 6B — Experimental Lasso and XGBoost Models

## Scope

Phase 6B added Lasso and XGBoost regressors alongside the existing mean baseline and Random Forest. It deliberately kept them **separate from the production workflow**. No cross-validation, hyperparameter search or scaffold-based comparison was performed in this phase; those followed in Phases 6C and 6D.

This report records results produced on 2026-10-27. It was written retrospectively in Phase 6F from the saved output `results/phase6b_holdout_metrics.json`. A byte-for-byte copy of that file is tracked at `reports/evidence/phase6b/phase6b_holdout_metrics.json`.

## Implementation

| Item | Detail |
|---|---|
| Module | `src/model_experiments.py`; the production `src/train_model.py` is unchanged |
| Models | `DummyRegressor(strategy="mean")`; `RandomForestRegressor(n_estimators=200)` (same settings as production); `Pipeline(StandardScaler, Lasso(alpha=0.01, max_iter=10000))`; `XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, tree_method="hist")` |
| Reproducibility | `random_state=42`. Every estimator uses `n_jobs=1`, so later cross-validation can parallelise across folds without nested threading |
| Data | ESOL, the six RDKit descriptors, all 1,128 rows (duplicates retained) |
| Dependency | `xgboost==3.4.1` added to `requirements.txt`; macOS additionally needs the OpenMP runtime (`brew install libomp`) |
| Tests | `tests/test_model_experiments.py` (11 tests): construction, fitting, finite predictions, feature-name compatibility, column-order handling, reproducibility, and equality of the experimental Random Forest with the production configuration |

## Sanity check on the Phase 3 split

Untuned hyperparameters on a single random split (902 training / 226 test molecules, `random_state=42`). This is **not** a model comparison.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Mean baseline | 1.732 | 2.174 | −0.000 |
| Random Forest | 0.574 | 0.835 | 0.853 |
| Lasso | 0.837 | 1.108 | 0.740 |
| XGBoost | 0.562 | 0.822 | 0.857 |

The Random Forest reproduces the Phase 3 result exactly, which shows that single-threaded fitting does not change its predictions.

The 0.013 RMSE difference between XGBoost and the Random Forest is far smaller than the split-to-split variability seen in Phase 6A (RMSE 0.615–0.792 across split seeds), so no ranking can be drawn from it.

## What happened next

- **Phase 6C** tuned and compared the models with grouped nested cross-validation (`docs/PHASE6C_REPORT.md`).
- **Phase 6D** compared them on unseen Bemis–Murcko scaffolds under a prespecified experimental design (`docs/PHASE6D_REPORT.md`).
- None of the experimental models replaced the deployed Phase 3 Random Forest.
