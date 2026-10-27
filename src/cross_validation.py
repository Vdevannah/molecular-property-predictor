"""Run grouped nested cross-validation for the experimental solubility models."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import rdkit
import sklearn
import xgboost
from rdkit import Chem, RDLogger
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from src.evaluate_model import get_scaffold_group
from src.model_experiments import build_experimental_models
from src.train_model import FEATURE_COLUMNS, TARGET_COLUMN, evaluate_model, validate_training_data

RDLogger.DisableLog("rdApp.*")

CV_RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 5
REPEATS = 3
SCORING = "neg_root_mean_squared_error"
PROTOCOLS = ("random_grouped", "scaffold")
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "lasso": {"lasso__alpha": [0.001, 0.01, 0.1, 1.0]},
    "random_forest_tuned": {
        "n_estimators": [200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 3],
    },
    "xgboost": {
        "n_estimators": [200, 500],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
    },
}
RESULTS_DIR = PROJECT_ROOT / "results" / "phase6c"


class BalancedGroupKFold:
    """K-fold splitter that keeps groups intact and balances fold sample counts.

    Groups are assigned largest first to the currently lightest fold. Ties in group
    size and fold weight are broken with a seeded random generator, so different
    seeds give different partitions while every group, however large, is tested
    exactly once per split.
    """

    def __init__(self, n_splits: int = 5, random_state: int | None = None) -> None:
        self.n_splits = n_splits
        self.random_state = random_state

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def assign_folds(self, groups: Any) -> np.ndarray:
        """Return the fold number assigned to each sample."""
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        group_array = np.asarray(groups)
        unique_groups, inverse, counts = np.unique(group_array, return_inverse=True, return_counts=True)
        if self.n_splits > len(unique_groups):
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} greater than the number of groups: {len(unique_groups)}."
            )

        rng = np.random.default_rng(self.random_state)
        order = np.lexsort((rng.random(len(unique_groups)), -counts))
        fold_sizes = np.zeros(self.n_splits)
        group_folds = np.empty(len(unique_groups), dtype=int)
        for group_index in order:
            lightest_folds = np.flatnonzero(fold_sizes == fold_sizes.min())
            fold = int(rng.choice(lightest_folds))
            group_folds[group_index] = fold
            fold_sizes[fold] += counts[group_index]
        return group_folds[inverse]

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        sample_folds = self.assign_folds(groups)
        indices = np.arange(len(sample_folds))
        for fold in range(self.n_splits):
            yield indices[sample_folds != fold], indices[sample_folds == fold]


def canonical_identity_group(smiles: str) -> str:
    """Return a stereo-aware canonical SMILES used to keep identical structures together."""
    cleaned = str(smiles).strip()
    molecule = Chem.MolFromSmiles(cleaned)
    if molecule is None:
        return f"INVALID:{cleaned}"
    return Chem.MolToSmiles(molecule, canonical=True)


def scaffold_group(smiles: str) -> str:
    """Return the Phase 4 Bemis–Murcko scaffold group for a molecule."""
    return get_scaffold_group(str(smiles).strip())


def build_groups(dataset: pd.DataFrame, protocol: str) -> np.ndarray:
    """Return one group label per row for the requested evaluation protocol."""
    if "smiles" not in dataset.columns:
        raise ValueError("Dataset is missing the SMILES column required for grouping.")
    if protocol == "random_grouped":
        return dataset["smiles"].map(canonical_identity_group).to_numpy()
    if protocol == "scaffold":
        return dataset["smiles"].map(scaffold_group).to_numpy()
    raise ValueError(f"Unknown evaluation protocol: {protocol}")


def build_model_specs(random_state: int = CV_RANDOM_STATE) -> dict[str, tuple[Any, dict[str, list[Any]] | None]]:
    """Return each model with its hyperparameter grid, or None when it is not tuned."""
    models = build_experimental_models(random_state=random_state, n_jobs=1)
    return {
        "baseline": (models["baseline"], None),
        "random_forest_reference": (models["random_forest"], None),
        "lasso": (models["lasso"], PARAM_GRIDS["lasso"]),
        "random_forest_tuned": (clone(models["random_forest"]), PARAM_GRIDS["random_forest_tuned"]),
        "xgboost": (models["xgboost"], PARAM_GRIDS["xgboost"]),
    }


def nested_cross_validate(
    dataset: pd.DataFrame,
    groups: np.ndarray,
    model_specs: dict[str, tuple[Any, dict[str, list[Any]] | None]],
    protocol: str,
    outer_splits: int = OUTER_SPLITS,
    inner_splits: int = INNER_SPLITS,
    repeats: int = REPEATS,
    random_state: int = CV_RANDOM_STATE,
    n_jobs: int = -1,
    fold_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every model on identical grouped outer folds, tuning only on outer-training data.

    Returns per-fold metrics and the outer test-fold assignment of every row.
    """
    validate_training_data(dataset)
    X = dataset[FEATURE_COLUMNS].reset_index(drop=True)
    y = dataset[TARGET_COLUMN].reset_index(drop=True)
    group_array = np.asarray(groups)
    if len(group_array) != len(X):
        raise ValueError("Group labels must have one entry per dataset row.")

    fold_rows: list[dict[str, Any]] = []
    assignment_rows: list[pd.DataFrame] = []
    for repeat in range(repeats):
        outer = BalancedGroupKFold(n_splits=outer_splits, random_state=random_state + repeat)
        for fold, (train_index, test_index) in enumerate(outer.split(X, y, group_array)):
            if fold_limit is not None and fold >= fold_limit:
                break
            assignment_rows.append(
                pd.DataFrame({"protocol": protocol, "repeat": repeat, "fold": fold, "row": test_index})
            )
            X_train, y_train, groups_train = X.iloc[train_index], y.iloc[train_index], group_array[train_index]
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]
            inner_seed = random_state + 1000 * (repeat + 1) + fold

            for model_name, (estimator, param_grid) in model_specs.items():
                start = perf_counter()
                if param_grid is None:
                    model = clone(estimator).fit(X_train, y_train)
                    best_params: dict[str, Any] = {}
                    inner_rmse = None
                else:
                    search = GridSearchCV(
                        clone(estimator),
                        param_grid,
                        scoring=SCORING,
                        cv=BalancedGroupKFold(n_splits=inner_splits, random_state=inner_seed),
                        n_jobs=n_jobs,
                        refit=True,
                        error_score="raise",
                    )
                    search.fit(X_train, y_train, groups=groups_train)
                    model = search.best_estimator_
                    best_params = search.best_params_
                    inner_rmse = float(-search.best_score_)
                predictions = model.predict(X_test)
                fold_rows.append(
                    {
                        "protocol": protocol,
                        "repeat": repeat,
                        "fold": fold,
                        "model": model_name,
                        "n_train": int(len(train_index)),
                        "n_test": int(len(test_index)),
                        "n_train_groups": int(len(np.unique(groups_train))),
                        "n_test_groups": int(len(np.unique(group_array[test_index]))),
                        **evaluate_model(y_test, predictions),
                        "inner_cv_rmse": inner_rmse,
                        "best_params": json.dumps(best_params, sort_keys=True),
                        "fit_seconds": perf_counter() - start,
                    }
                )

    return pd.DataFrame(fold_rows), pd.concat(assignment_rows, ignore_index=True)


def corrected_paired_ttest(
    differences: np.ndarray,
    n_train: np.ndarray,
    n_test: np.ndarray,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Nadeau–Bengio corrected resampled t-test for paired cross-validation differences."""
    values = np.asarray(differences, dtype=float)
    n_folds = len(values)
    if n_folds < 2:
        raise ValueError("At least two paired folds are required.")

    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    test_train_ratio = float(np.mean(n_test) / np.mean(n_train))
    standard_error = float(np.sqrt((1.0 / n_folds + test_train_ratio) * sd**2))
    degrees_of_freedom = n_folds - 1

    if standard_error == 0.0:
        t_statistic = 0.0 if mean == 0.0 else float(np.sign(mean) * np.inf)
        p_value = 1.0 if mean == 0.0 else 0.0
        half_width = 0.0
    else:
        t_statistic = mean / standard_error
        p_value = float(2 * stats.t.sf(abs(t_statistic), degrees_of_freedom))
        half_width = float(stats.t.ppf(0.5 + confidence / 2, degrees_of_freedom) * standard_error)

    return {
        "n_folds": n_folds,
        "mean_difference": mean,
        "sd_difference": sd,
        "corrected_standard_error": standard_error,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
        "confidence": confidence,
        "t_statistic": float(t_statistic),
        "p_value": p_value,
        "ci_excludes_zero": bool(mean - half_width > 0 or mean + half_width < 0),
    }


def paired_comparisons(fold_metrics: pd.DataFrame, metrics: tuple[str, ...] = ("RMSE", "MAE")) -> pd.DataFrame:
    """Compare every model pair on identical outer folds (difference = model_a − model_b)."""
    rows: list[dict[str, Any]] = []
    for protocol, protocol_rows in fold_metrics.groupby("protocol", sort=False):
        models = list(dict.fromkeys(protocol_rows["model"]))
        for model_a, model_b in itertools.combinations(models, 2):
            a = protocol_rows[protocol_rows["model"] == model_a].set_index(["repeat", "fold"]).sort_index()
            b = protocol_rows[protocol_rows["model"] == model_b].set_index(["repeat", "fold"]).sort_index()
            if not a.index.equals(b.index) or not (a["n_test"] == b["n_test"]).all():
                raise ValueError(f"Models {model_a} and {model_b} were not evaluated on identical folds.")
            for metric in metrics:
                result = corrected_paired_ttest(
                    (a[metric] - b[metric]).to_numpy(),
                    a["n_train"].to_numpy(),
                    a["n_test"].to_numpy(),
                )
                rows.append({"protocol": protocol, "metric": metric, "model_a": model_a, "model_b": model_b, **result})
    return pd.DataFrame(rows)


def summarize_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return mean and standard deviation of each metric per protocol and model."""
    summary = fold_metrics.groupby(["protocol", "model"], sort=False)[["RMSE", "MAE", "R2", "n_test"]].agg(
        ["mean", "std", "min", "max"]
    )
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    return summary.reset_index()


def experiment_config(dataset_path: Path, dataset: pd.DataFrame, groups: dict[str, np.ndarray]) -> dict[str, Any]:
    """Describe everything required to reproduce the experiment."""
    model_specs = build_model_specs()
    group_summary = {}
    for protocol, labels in groups.items():
        sizes = pd.Series(labels).value_counts()
        group_summary[protocol] = {
            "n_groups": int(len(sizes)),
            "largest_groups": {str(name): int(size) for name, size in sizes.head(5).items()},
            "singleton_groups": int((sizes == 1).sum()),
        }
    return {
        "dataset": {
            "path": str(dataset_path.relative_to(PROJECT_ROOT)),
            "md5": hashlib.md5(dataset_path.read_bytes()).hexdigest(),
            "rows": int(len(dataset)),
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "filtering": "none; all rows including duplicate measurements are retained",
        },
        "groups": group_summary,
        "cross_validation": {
            "protocols": list(PROTOCOLS),
            "outer_splits": OUTER_SPLITS,
            "inner_splits": INNER_SPLITS,
            "repeats": REPEATS,
            "splitter": "BalancedGroupKFold (largest group first to lightest fold, seeded tie-breaks)",
            "outer_seed": f"{CV_RANDOM_STATE} + repeat",
            "inner_seed": f"{CV_RANDOM_STATE} + 1000 * (repeat + 1) + fold",
            "scoring": SCORING,
            "paired_test": "Nadeau–Bengio corrected resampled t-test, 95% interval",
        },
        "models": {
            name: {
                "estimator": type(estimator).__name__,
                "base_params": {key: repr(value) for key, value in estimator.get_params(deep=False).items()},
                "param_grid": grid,
            }
            for name, (estimator, grid) in model_specs.items()
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "rdkit": rdkit.__version__,
            "platform": platform.platform(),
        },
    }


def run_benchmark(dataset: pd.DataFrame, groups: dict[str, np.ndarray], n_jobs: int) -> dict[str, Any]:
    """Time one outer fold per protocol and extrapolate to the full experiment."""
    model_specs = build_model_specs()
    fold_seconds: dict[str, dict[str, float]] = {}
    for protocol in PROTOCOLS:
        folds, _ = nested_cross_validate(
            dataset, groups[protocol], model_specs, protocol, repeats=1, n_jobs=n_jobs, fold_limit=1
        )
        fold_seconds[protocol] = dict(zip(folds["model"], folds["fit_seconds"].round(3)))
    per_fold_total = sum(sum(seconds.values()) for seconds in fold_seconds.values())
    return {
        "n_jobs": n_jobs,
        "seconds_for_one_outer_fold": fold_seconds,
        "outer_folds_per_protocol": OUTER_SPLITS * REPEATS,
        "estimated_total_seconds": round(per_fold_total * OUTER_SPLITS * REPEATS, 1),
    }


def main() -> None:
    """Run the runtime benchmark or the full nested cross-validation experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="Time one outer fold and estimate total runtime.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for GridSearchCV.")
    args = parser.parse_args()

    dataset_path = PROJECT_ROOT / "data" / "esol_descriptors.csv"
    dataset = pd.read_csv(dataset_path)
    groups = {protocol: build_groups(dataset, protocol) for protocol in PROTOCOLS}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.benchmark:
        benchmark = run_benchmark(dataset, groups, args.n_jobs)
        (RESULTS_DIR / "benchmark.json").write_text(json.dumps(benchmark, indent=2))
        print(json.dumps(benchmark, indent=2))
        return

    start = perf_counter()
    model_specs = build_model_specs()
    fold_frames, assignment_frames = [], []
    for protocol in PROTOCOLS:
        folds, assignments = nested_cross_validate(dataset, groups[protocol], model_specs, protocol, n_jobs=args.n_jobs)
        fold_frames.append(folds)
        assignment_frames.append(assignments)
        print(f"Completed {protocol} protocol after {perf_counter() - start:.1f} s")

    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    assignments = pd.concat(assignment_frames, ignore_index=True)
    assignments["group"] = [groups[protocol][row] for protocol, row in zip(assignments["protocol"], assignments["row"])]
    summary = summarize_folds(fold_metrics)
    comparisons = paired_comparisons(fold_metrics)

    config = experiment_config(dataset_path, dataset, groups)
    config["runtime_seconds"] = round(perf_counter() - start, 1)
    config["n_jobs"] = args.n_jobs
    (RESULTS_DIR / "config.json").write_text(json.dumps(config, indent=2))
    fold_metrics.to_csv(RESULTS_DIR / "fold_metrics.csv", index=False)
    assignments.to_csv(RESULTS_DIR / "fold_assignments.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    comparisons.to_csv(RESULTS_DIR / "paired_comparisons.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[["protocol", "model", "RMSE_mean", "RMSE_std", "MAE_mean", "R2_mean"]].round(3).to_string(index=False))
    print(f"Total runtime: {config['runtime_seconds']} s")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
