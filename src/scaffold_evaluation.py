"""Run the prespecified Phase 6D scaffold-generalization experiment (configs/phase6d.json)."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import rdkit
import scipy
import sklearn
import xgboost
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.cross_validation import BalancedGroupKFold, canonical_identity_group, corrected_paired_ttest, scaffold_group
from src.train_model import FEATURE_COLUMNS, TARGET_COLUMN

RDLogger.DisableLog("rdApp.*")

CONFIG_PATH = PROJECT_ROOT / "configs" / "phase6d.json"
SCORING = "neg_root_mean_squared_error"
OUTER_SEED_BASE = 42
DESCRIPTOR_EQUALITY_TOLERANCE = 1e-9
EXPECTED_FORMULAS = {
    ("protocols", "nonbenzene_scaffold_cv", "outer_seed"): "42 + repeat",
    ("protocols", "generic_nonblock_scaffold_cv", "outer_seed"): "42 + repeat",
    ("inner_cv", "seed"): "42 + 1000 * (repeat + 1) + outer_fold",
    ("inner_cv", "tie_breaking"): "lowest candidate index in ParameterGrid order",
    ("statistics", "bootstrap", "method"): "scaffold-cluster percentile bootstrap",
    ("statistics", "bootstrap", "interval"): "95% percentile",
    ("statistics", "primary_contrast", "statistic"): (
        "repeat_averaged_rmse(random_forest_reference) - repeat_averaged_rmse(xgboost)"
    ),
}
# Scenario name -> (grouping, data variant, CV protocol, holdout protocol, block group)
SCENARIOS = {
    "murcko_S0": ("murcko", "S0", "nonbenzene_scaffold_cv", "benzene_holdout"),
    "murcko_S1": ("murcko", "S1", "nonbenzene_scaffold_cv", "benzene_holdout"),
    "murcko_S2": ("murcko", "S2", "nonbenzene_scaffold_cv", "benzene_holdout"),
    "generic_S0": ("generic", "S0", "generic_nonblock_scaffold_cv", "generic_block_holdout"),
}
ESTIMATOR_FACTORIES: dict[str, tuple[str, Callable[[], Any]]] = {
    "baseline": ("DummyRegressor", DummyRegressor),
    "linear_regression_reference": (
        "Pipeline(StandardScaler, LinearRegression)",
        lambda: Pipeline([("scaler", StandardScaler()), ("linear", LinearRegression())]),
    ),
    "random_forest_reference": ("RandomForestRegressor", RandomForestRegressor),
    "lasso": ("Pipeline(StandardScaler, Lasso)", lambda: Pipeline([("scaler", StandardScaler()), ("lasso", Lasso())])),
    "random_forest_tuned": ("RandomForestRegressor", RandomForestRegressor),
    "xgboost": ("XGBRegressor", XGBRegressor),
}


# Configuration and integrity -------------------------------------------------


def file_md5(path: str | Path) -> str:
    """Return the MD5 hex digest of a file."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load_config(path: str | Path = CONFIG_PATH) -> tuple[dict[str, Any], str]:
    """Load the experiment configuration and its MD5."""
    config_path = Path(path)
    return json.loads(config_path.read_text()), file_md5(config_path)


def validate_config(config: dict[str, Any]) -> None:
    """Abort if the configuration specifies anything this implementation does not follow."""
    for keys, expected in EXPECTED_FORMULAS.items():
        value: Any = config
        for key in keys:
            value = value[key]
        if value != expected:
            raise ValueError(f"Configuration {'.'.join(keys)} is {value!r}; implementation expects {expected!r}.")
    data = config["data"]
    if data["feature_columns"] != FEATURE_COLUMNS or data["target_column"] != TARGET_COLUMN:
        raise ValueError("Configured features or target differ from the production feature set.")
    if set(data["forbidden_columns"]) & set(FEATURE_COLUMNS):
        raise ValueError("A forbidden column is listed as a feature.")
    for name, entry in config["models"].items():
        if name not in ESTIMATOR_FACTORIES or ESTIMATOR_FACTORIES[name][0] != entry["estimator"]:
            raise ValueError(f"Unsupported model configuration: {name}")
    if config["inner_cv"]["splits"] < 2:
        raise ValueError("Inner cross-validation needs at least two splits.")


def verify_checksums(expected: dict[str, str], root: Path = PROJECT_ROOT) -> None:
    """Raise if any file differs from its recorded MD5."""
    mismatches = [
        path
        for path, digest in expected.items()
        if path != "check" and (not (root / path).exists() or file_md5(root / path) != digest)
    ]
    if mismatches:
        raise RuntimeError(f"Checksum mismatch for: {', '.join(mismatches)}")


def create_run_directory(results_root: Path, run_id: str) -> Path:
    """Create a new run directory, refusing to overwrite an existing one."""
    run_dir = results_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


# Grouping, families and data variants ----------------------------------------


def generic_scaffold_group(smiles: str) -> str:
    """Return the generic Murcko scaffold; acyclic molecules use canonical SMILES."""
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        return canonical_identity_group(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms() == 0:
        return canonical_identity_group(smiles)
    return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold))


def is_acyclic(smiles: str) -> bool:
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    return molecule is not None and molecule.GetRingInfo().NumRings() == 0


def annotate_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add row identifiers and identity, Murcko and generic group labels."""
    annotated = dataset.reset_index(drop=True).copy()
    annotated["row_id"] = np.arange(len(annotated))
    annotated["identity"] = annotated["smiles"].map(canonical_identity_group)
    annotated["murcko"] = annotated["smiles"].map(scaffold_group)
    annotated["generic"] = annotated["smiles"].map(generic_scaffold_group)
    annotated["acyclic"] = annotated["smiles"].map(is_acyclic)
    return annotated


def assign_families(dataset: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """Assign each molecule to a prespecified Murcko family."""
    families = config["families"]
    named = families["named"]
    sizes = dataset["murcko"].map(dataset["murcko"].value_counts())
    large_cyclic = set(dataset.loc[~dataset["acyclic"] & (sizes >= families["named_min_size"]), "murcko"])
    if large_cyclic != set(named):
        raise ValueError("Named families do not match the cyclic Murcko groups above the size threshold.")

    labels = np.where(
        dataset["murcko"].isin(named),
        dataset["murcko"],
        np.where(dataset["acyclic"], "acyclic", np.where(sizes >= 2, "cyclic_small_groups", "cyclic_singletons")),
    )
    return pd.Series(labels, index=dataset.index, name="family")


def make_variant(dataset: pd.DataFrame, variant: str, config: dict[str, Any]) -> pd.DataFrame:
    """Return the S0, S1 or S2 version of an annotated dataset."""
    spec = config["sensitivity"]["variants"][variant]
    if variant == "S0":
        return dataset.copy()
    if variant == "S1":
        removed = dataset["Compound ID"].isin(spec["remove_compound_ids"])
        if set(dataset.loc[removed, "identity"]) != {spec["canonical_smiles"]}:
            raise ValueError("S1 rows do not match the configured canonical SMILES.")
        return dataset.loc[~removed].copy()
    if variant == "S2":
        rows = []
        for _, group in dataset.groupby("identity", sort=False):
            if len(group) == 1:
                rows.append(group)
                continue
            # Identical structures can differ by ~1e-14 from atom-order-dependent float summation.
            descriptors = group[FEATURE_COLUMNS].to_numpy(dtype=float)
            if not np.allclose(descriptors, descriptors[0], rtol=0.0, atol=DESCRIPTOR_EQUALITY_TOLERANCE):
                raise ValueError(f"Descriptors differ within duplicate group {group['identity'].iloc[0]}.")
            target = group[TARGET_COLUMN]
            if target.max() - target.min() > spec["conflict_threshold_logS"]:
                continue
            collapsed = group.iloc[[0]].copy()
            collapsed[TARGET_COLUMN] = target.mean()
            rows.append(collapsed)
        return pd.concat(rows).sort_values("row_id").copy()
    raise ValueError(f"Unknown data variant: {variant}")


def validate_expected_counts(dataset: pd.DataFrame, config: dict[str, Any]) -> None:
    """Abort if the data differ from the counts recorded in the configuration."""
    checks = {"rows": (len(dataset), config["data"]["expected_rows"])}
    for grouping in ("murcko", "generic"):
        spec = config["grouping"][grouping]
        counts = dataset[grouping].value_counts()
        checks[f"{grouping} groups"] = (len(counts), spec["expected_groups"])
        checks[f"{grouping} largest group"] = ((counts.index[0], int(counts.iloc[0])), (spec["largest_group"]["label"], spec["largest_group"]["rows"]))
    families = assign_families(dataset, config).value_counts()
    named_rows = int(families[config["families"]["named"]].sum())
    checks["named family rows"] = (named_rows, config["families"]["expected_named_rows"])
    for name, spec in config["families"]["aggregate"].items():
        checks[f"{name} rows"] = (int(families.get(name, 0)), spec["expected_rows"])
    for variant in ("S1", "S2"):
        checks[f"{variant} rows"] = (len(make_variant(dataset, variant, config)), config["sensitivity"]["variants"][variant]["expected_rows"])
    failed = {name: values for name, values in checks.items() if values[0] != values[1]}
    if failed:
        raise ValueError(f"Data do not match configuration: {failed}")


# Splits ----------------------------------------------------------------------


@dataclass(frozen=True)
class OuterFold:
    repeat: int
    fold: int
    train_index: np.ndarray
    test_index: np.ndarray


def assert_group_disjoint(groups: np.ndarray, train_index: np.ndarray, test_index: np.ndarray) -> None:
    """Raise if any group label appears on both sides of a split."""
    overlap = set(groups[train_index]) & set(groups[test_index])
    if overlap:
        raise AssertionError(f"Groups present in both training and test: {sorted(overlap)[:5]}")


def block_cv_folds(groups: np.ndarray, block_label: str, n_splits: int, repeats: int) -> list[OuterFold]:
    """Grouped K-fold over non-block groups, with the block group always in training."""
    group_array = np.asarray(groups)
    block_index = np.flatnonzero(group_array == block_label)
    rest_index = np.flatnonzero(group_array != block_label)
    folds = []
    for repeat in range(repeats):
        splitter = BalancedGroupKFold(n_splits=n_splits, random_state=OUTER_SEED_BASE + repeat)
        for fold, (train, test) in enumerate(splitter.split(None, None, group_array[rest_index])):
            train_index = np.sort(np.concatenate([block_index, rest_index[train]]))
            folds.append(OuterFold(repeat, fold, train_index, rest_index[test]))
    return folds


def block_holdout_fold(groups: np.ndarray, block_label: str) -> list[OuterFold]:
    """Single split testing the block group after training on everything else."""
    group_array = np.asarray(groups)
    return [OuterFold(0, 0, np.flatnonzero(group_array != block_label), np.flatnonzero(group_array == block_label))]


def inner_seed(repeat: int, fold: int) -> int:
    return OUTER_SEED_BASE + 1000 * (repeat + 1) + fold


def make_inner_splits(
    train_groups: np.ndarray, block_label: str, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inner grouped splits in which the block group is always in the inner training set."""
    group_array = np.asarray(train_groups)
    block_index = np.flatnonzero(group_array == block_label)
    rest_index = np.flatnonzero(group_array != block_label)
    splitter = BalancedGroupKFold(n_splits=n_splits, random_state=seed)
    return [
        (np.sort(np.concatenate([block_index, rest_index[train]])), rest_index[test])
        for train, test in splitter.split(None, None, group_array[rest_index])
    ]


# Molecule-weighted inner selection -------------------------------------------


def pooled_rmse(fold_rmse: Any, fold_sizes: Any) -> float:
    """Return sqrt(sum n_k * RMSE_k^2 / sum n_k), the RMSE over all pooled validation molecules."""
    rmse = np.asarray(fold_rmse, dtype=float)
    sizes = np.asarray(fold_sizes, dtype=float)
    return float(np.sqrt(np.sum(sizes * rmse**2) / np.sum(sizes)))


def pooled_scores(cv_results: dict[str, Any], fold_sizes: list[int]) -> np.ndarray:
    """Pooled validation RMSE of every candidate from GridSearchCV split scores."""
    split_rmse = np.column_stack([-np.asarray(cv_results[f"split{k}_test_score"]) for k in range(len(fold_sizes))])
    return np.array([pooled_rmse(row, fold_sizes) for row in split_rmse])


def make_pooled_refit(fold_sizes: list[int]) -> Callable[[dict[str, Any]], int]:
    """GridSearchCV refit callable selecting the lowest pooled RMSE (first index on ties)."""

    def select(cv_results: dict[str, Any]) -> int:
        return int(np.argmin(pooled_scores(cv_results, fold_sizes)))

    return select


# Models ----------------------------------------------------------------------


def build_model_specs(config: dict[str, Any]) -> dict[str, tuple[Any, dict[str, list[Any]] | None]]:
    """Build every configured estimator with its fixed parameters and grid."""
    specs = {}
    for name, entry in config["models"].items():
        estimator = ESTIMATOR_FACTORIES[name][1]()
        estimator.set_params(**entry["params"])
        specs[name] = (estimator, entry["grid"])
    return specs


# Running a protocol ------------------------------------------------------------


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "mean_signed_error": float(np.mean(y_pred - y_true)),
    }


def run_protocol(
    dataset: pd.DataFrame,
    groups: np.ndarray,
    folds: list[OuterFold],
    model_specs: dict[str, tuple[Any, dict[str, list[Any]] | None]],
    *,
    scenario: str,
    protocol: str,
    block_label: str,
    inner_splits: int,
    n_jobs: int = -1,
) -> dict[str, pd.DataFrame]:
    """Fit every model on every outer fold with molecule-weighted nested tuning."""
    X = dataset[FEATURE_COLUMNS].reset_index(drop=True)
    y = dataset[TARGET_COLUMN].to_numpy(dtype=float)
    group_array = np.asarray(groups)
    row_ids = dataset["row_id"].to_numpy()
    predictions, fold_rows, inner_rows, assignments = [], [], [], []

    for outer in folds:
        assert_group_disjoint(group_array, outer.train_index, outer.test_index)
        X_train, y_train = X.iloc[outer.train_index], y[outer.train_index]
        X_test, y_test = X.iloc[outer.test_index], y[outer.test_index]
        splits = make_inner_splits(group_array[outer.train_index], block_label, inner_splits, inner_seed(outer.repeat, outer.fold))
        for inner_train, inner_test in splits:
            assert_group_disjoint(group_array[outer.train_index], inner_train, inner_test)
        fold_sizes = [len(test) for _, test in splits]
        key = {"scenario": scenario, "protocol": protocol, "repeat": outer.repeat, "fold": outer.fold}
        assignments.append(pd.DataFrame({**key, "row_id": row_ids[outer.test_index], "group": group_array[outer.test_index]}))

        for model_name, (estimator, grid) in model_specs.items():
            start = perf_counter()
            selected: dict[str, Any] = {}
            inner_pooled = inner_mean = None
            unweighted_differs = False
            if grid is None:
                model = clone(estimator).fit(X_train, y_train)
            else:
                search = GridSearchCV(
                    clone(estimator),
                    grid,
                    scoring=SCORING,
                    cv=splits,
                    refit=make_pooled_refit(fold_sizes),
                    n_jobs=n_jobs,
                    error_score="raise",
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                selected = search.cv_results_["params"][search.best_index_]
                pooled = pooled_scores(search.cv_results_, fold_sizes)
                unweighted = -np.asarray(search.cv_results_["mean_test_score"])
                inner_pooled = float(pooled[search.best_index_])
                inner_mean = float(unweighted[search.best_index_])
                unweighted_differs = bool(np.argmin(unweighted) != search.best_index_)
                for candidate, params in enumerate(search.cv_results_["params"]):
                    inner_rows.append(
                        {
                            **key,
                            "model": model_name,
                            "candidate": candidate,
                            "params": json.dumps(params, sort_keys=True),
                            "split_rmse": json.dumps([float(-search.cv_results_[f"split{k}_test_score"][candidate]) for k in range(len(fold_sizes))]),
                            "fold_sizes": json.dumps(fold_sizes),
                            "pooled_rmse": float(pooled[candidate]),
                            "unweighted_mean_rmse": float(unweighted[candidate]),
                            "selected": candidate == search.best_index_,
                        }
                    )
            y_pred = np.asarray(model.predict(X_test), dtype=float)
            predictions.append(
                pd.DataFrame(
                    {
                        **key,
                        "model": model_name,
                        "model_seed": 42,
                        "row_id": row_ids[outer.test_index],
                        "group": group_array[outer.test_index],
                        "y_true": y_test,
                        "y_pred": y_pred,
                    }
                )
            )
            fold_rows.append(
                {
                    **key,
                    "model": model_name,
                    "n_train": int(len(outer.train_index)),
                    "n_test": int(len(outer.test_index)),
                    "n_train_groups": int(len(np.unique(group_array[outer.train_index]))),
                    "n_test_groups": int(len(np.unique(group_array[outer.test_index]))),
                    **_fold_metrics(y_test, y_pred),
                    "selected_params": json.dumps(selected, sort_keys=True),
                    "inner_pooled_rmse": inner_pooled,
                    "inner_unweighted_mean_rmse": inner_mean,
                    "unweighted_selection_differs": unweighted_differs,
                    "fit_seconds": perf_counter() - start,
                }
            )

    return {
        "predictions": pd.concat(predictions, ignore_index=True),
        "fold_metrics": pd.DataFrame(fold_rows),
        "inner_results": pd.DataFrame(inner_rows),
        "assignments": pd.concat(assignments, ignore_index=True),
    }


def run_seed_check(
    dataset: pd.DataFrame,
    folds: list[OuterFold],
    fold_metrics: pd.DataFrame,
    model_specs: dict[str, tuple[Any, dict[str, list[Any]] | None]],
    seeds: list[int],
    *,
    scenario: str,
    protocol: str,
    groups: np.ndarray,
) -> pd.DataFrame:
    """Refit RF reference and XGBoost with other random states, reusing selected XGBoost settings."""
    X = dataset[FEATURE_COLUMNS].reset_index(drop=True)
    y = dataset[TARGET_COLUMN].to_numpy(dtype=float)
    row_ids = dataset["row_id"].to_numpy()
    group_array = np.asarray(groups)
    rows = []
    for outer in folds:
        selected = fold_metrics[
            (fold_metrics["protocol"] == protocol)
            & (fold_metrics["repeat"] == outer.repeat)
            & (fold_metrics["fold"] == outer.fold)
            & (fold_metrics["model"] == "xgboost")
        ]["selected_params"].iloc[0]
        for seed in seeds:
            for model_name, extra in (("random_forest_reference", {}), ("xgboost", json.loads(selected))):
                model = clone(model_specs[model_name][0]).set_params(**extra, random_state=seed)
                model.fit(X.iloc[outer.train_index], y[outer.train_index])
                rows.append(
                    pd.DataFrame(
                        {
                            "scenario": scenario,
                            "protocol": protocol,
                            "repeat": outer.repeat,
                            "fold": outer.fold,
                            "model": model_name,
                            "model_seed": seed,
                            "row_id": row_ids[outer.test_index],
                            "group": group_array[outer.test_index],
                            "y_true": y[outer.test_index],
                            "y_pred": model.predict(X.iloc[outer.test_index]),
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


# Metrics -----------------------------------------------------------------------


def molecule_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average each molecule's errors over repeats so that it contributes exactly once."""
    frame = predictions.assign(
        sq_error=(predictions["y_pred"] - predictions["y_true"]) ** 2,
        abs_error=(predictions["y_pred"] - predictions["y_true"]).abs(),
        signed_error=predictions["y_pred"] - predictions["y_true"],
    )
    return (
        frame.groupby(["model", "row_id"], sort=True)
        .agg(
            group=("group", "first"),
            y_true=("y_true", "first"),
            sq_error=("sq_error", "mean"),
            abs_error=("abs_error", "mean"),
            signed_error=("signed_error", "mean"),
            n_repeats=("repeat", "nunique"),
        )
        .reset_index()
    )


def protocol_metrics(predictions: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Repeat-averaged, per-repeat pooled and fold-averaged metrics for each model."""
    rows = []
    molecules = molecule_errors(predictions)
    for model, model_molecules in molecules.groupby("model", sort=False):
        model_predictions = predictions[predictions["model"] == model]
        per_repeat = model_predictions.groupby("repeat").apply(
            lambda frame: pd.Series(
                {
                    "rmse": np.sqrt(mean_squared_error(frame["y_true"], frame["y_pred"])),
                    "r2": r2_score(frame["y_true"], frame["y_pred"]),
                }
            ),
            include_groups=False,
        )
        folds = fold_metrics[fold_metrics["model"] == model]["RMSE"]
        rows.append(
            {
                "model": model,
                "n_molecules": int(len(model_molecules)),
                "n_groups": int(model_molecules["group"].nunique()),
                "n_repeats": int(model_predictions["repeat"].nunique()),
                "repeat_averaged_rmse": float(np.sqrt(model_molecules["sq_error"].mean())),
                "repeat_averaged_mae": float(model_molecules["abs_error"].mean()),
                "mean_signed_error": float(model_molecules["signed_error"].mean()),
                "per_repeat_rmse_mean": float(per_repeat["rmse"].mean()),
                "per_repeat_rmse_sd": float(per_repeat["rmse"].std(ddof=1)) if len(per_repeat) > 1 else 0.0,
                "per_repeat_rmse_min": float(per_repeat["rmse"].min()),
                "per_repeat_rmse_max": float(per_repeat["rmse"].max()),
                "fold_averaged_rmse": float(folds.mean()),
                "fold_rmse_sd": float(folds.std(ddof=1)) if len(folds) > 1 else 0.0,
                "r2_mean_over_repeats": float(per_repeat["r2"].mean()),
            }
        )
    return pd.DataFrame(rows)


# Bootstrap and decision rule ---------------------------------------------------


def cluster_bootstrap_weights(clusters: Any, replicates: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (replicate x cluster) resampling counts and each observation's cluster code."""
    codes, uniques = pd.factorize(pd.Series(clusters), sort=True)
    n_clusters = len(uniques)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=replicates)
    return weights, codes


def bootstrap_rmse(weights: np.ndarray, codes: np.ndarray, sq_errors: Any) -> np.ndarray:
    """RMSE of every bootstrap replicate from per-molecule (repeat-averaged) squared errors."""
    n_clusters = weights.shape[1]
    sums = np.bincount(codes, weights=np.asarray(sq_errors, dtype=float), minlength=n_clusters)
    counts = np.bincount(codes, minlength=n_clusters).astype(float)
    return np.sqrt((weights @ sums) / (weights @ counts))


def bootstrap_difference(weights: np.ndarray, codes: np.ndarray, sq_a: Any, sq_b: Any) -> dict[str, Any]:
    """Point estimate, 95% percentile interval and two-sided p for RMSE(a) - RMSE(b)."""
    point = float(np.sqrt(np.mean(sq_a)) - np.sqrt(np.mean(sq_b)))
    distribution = bootstrap_rmse(weights, codes, sq_a) - bootstrap_rmse(weights, codes, sq_b)
    lower, upper = np.percentile(distribution, [2.5, 97.5])
    p_value = min(1.0, 2 * min(np.mean(distribution <= 0), np.mean(distribution >= 0)))
    return {
        "delta": point,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value": float(p_value),
        "bootstrap_sd": float(distribution.std(ddof=1)),
        "distribution": distribution,
    }


def decision_label(ci_lower: float, ci_upper: float, delta: float) -> str:
    """Apply the prespecified practical-equivalence decision rule."""
    within = ci_lower > -delta and ci_upper < delta
    excludes_zero = ci_lower > 0 or ci_upper < 0
    if within and not excludes_zero:
        return "practically equivalent at delta"
    if within and excludes_zero:
        return "detectable difference smaller than delta"
    if excludes_zero and (ci_lower > delta or ci_upper < -delta):
        return "difference larger than delta"
    if excludes_zero:
        return "difference detected; magnitude relative to delta uncertain"
    return "inconclusive"


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values in the original order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[index]))
        adjusted[index] = running
    return adjusted.tolist()


# Analyses ----------------------------------------------------------------------


def contrast_table(
    predictions: pd.DataFrame,
    *,
    cluster_column: str,
    replicates: int,
    seed: int,
    pairs: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    """Bootstrap every model pair on identical molecules and identical resamples."""
    molecules = molecule_errors(predictions)
    wide = molecules.pivot(index="row_id", columns="model", values="sq_error").sort_index()
    clusters = molecules.drop_duplicates("row_id").set_index("row_id").loc[wide.index]
    cluster_labels = clusters["group"] if cluster_column == "group" else clusters.index.to_series()
    weights, codes = cluster_bootstrap_weights(cluster_labels.to_numpy(), replicates, seed)
    models = list(dict.fromkeys(predictions["model"]))
    rows, distributions = [], {}
    for model_a, model_b in pairs or list(itertools.combinations(models, 2)):
        result = bootstrap_difference(weights, codes, wide[model_a].to_numpy(), wide[model_b].to_numpy())
        distributions[(model_a, model_b)] = result.pop("distribution")
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                **result,
                "n_molecules": int(len(wide)),
                "n_clusters": int(weights.shape[1]),
                "largest_cluster": int(np.bincount(codes).max()),
            }
        )
    return pd.DataFrame(rows), distributions


def boundary_report(fold_metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Fraction of outer folds selecting the minimum or maximum of each numeric grid dimension."""
    threshold = config["grid_boundary_check"]["flag_threshold_fraction_of_outer_folds"]
    rows = []
    for (scenario, protocol, model), frame in fold_metrics.groupby(["scenario", "protocol", "model"], sort=False):
        grid = config["models"][model]["grid"]
        if not grid:
            continue
        selections = frame["selected_params"].map(json.loads)
        for param, values in grid.items():
            numeric = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
            if len(numeric) < 2 or len(numeric) != len(values):
                continue
            chosen = selections.map(lambda params: params[param])
            at_min, at_max = float((chosen == min(numeric)).mean()), float((chosen == max(numeric)).mean())
            rows.append(
                {
                    "scenario": scenario,
                    "protocol": protocol,
                    "model": model,
                    "parameter": param,
                    "n_outer_folds": int(len(chosen)),
                    "fraction_at_min": at_min,
                    "fraction_at_max": at_max,
                    "selection_counts": json.dumps({str(k): int(v) for k, v in chosen.value_counts().items()}),
                    "flagged": bool(at_min + at_max > threshold),
                }
            )
    return pd.DataFrame(rows)


def family_metrics(
    cv_predictions: pd.DataFrame,
    holdout_predictions: pd.DataFrame,
    families: pd.Series,
    murcko: pd.Series,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Per-family errors: benzene from its holdout, all other families from non-benzene CV."""
    spec = config["families"]
    boot = config["statistics"]["bootstrap"]
    molecules = pd.concat([molecule_errors(cv_predictions), molecule_errors(holdout_predictions)], ignore_index=True)
    molecules["family"] = molecules["row_id"].map(families)
    molecules["murcko"] = molecules["row_id"].map(murcko)
    rows = []
    for family in [*spec["named"], *spec["aggregate"]]:
        subset = molecules[molecules["family"] == family]
        wide = subset.pivot(index="row_id", columns="model", values="sq_error").sort_index()
        targets = subset.drop_duplicates("row_id").set_index("row_id").loc[wide.index, "y_true"]
        n = len(wide)
        cluster_labels = (
            wide.index.to_numpy() if family in spec["named"] else subset.drop_duplicates("row_id").set_index("row_id").loc[wide.index, "murcko"].to_numpy()
        )
        weights, codes = (cluster_bootstrap_weights(cluster_labels, boot["replicates"], boot["seed"]) if n >= spec["min_n_for_intervals"] else (None, None))
        for model in wide.columns:
            model_rows = subset[subset["model"] == model]
            row = {
                "family": family,
                "model": model,
                "n": n,
                "RMSE": float(np.sqrt(wide[model].mean())),
                "MAE": float(model_rows["abs_error"].mean()),
                "mean_signed_error": float(model_rows["signed_error"].mean()),
                "target_mean": float(targets.mean()),
                "target_sd": float(targets.std(ddof=1)) if n > 1 else 0.0,
                "target_min": float(targets.min()),
                "target_max": float(targets.max()),
                "rmse_ci_lower": None,
                "rmse_ci_upper": None,
            }
            if weights is not None:
                replicate = bootstrap_rmse(weights, codes, wide[model].to_numpy())
                row["rmse_ci_lower"], row["rmse_ci_upper"] = (float(v) for v in np.percentile(replicate, [2.5, 97.5]))
            rows.append(row)
        if {"random_forest_reference", "xgboost"} <= set(wide.columns):
            difference = {
                "family": family,
                "model": "delta_random_forest_reference_minus_xgboost",
                "n": n,
                "RMSE": float(np.sqrt(wide["random_forest_reference"].mean()) - np.sqrt(wide["xgboost"].mean())),
                "rmse_ci_lower": None,
                "rmse_ci_upper": None,
            }
            if weights is not None:
                result = bootstrap_difference(weights, codes, wide["random_forest_reference"].to_numpy(), wide["xgboost"].to_numpy())
                difference["rmse_ci_lower"], difference["rmse_ci_upper"] = result["ci_lower"], result["ci_upper"]
            rows.append(difference)
    return pd.DataFrame(rows)


def error_floor(dataset: pd.DataFrame) -> dict[str, float]:
    """Lower bound on RMSE from targets of rows with identical descriptor vectors."""
    keys = dataset[FEATURE_COLUMNS].round(6).astype(str).agg("|".join, axis=1)
    grouped = dataset.groupby(keys)[TARGET_COLUMN]
    sizes = grouped.transform("size")
    within = (dataset[TARGET_COLUMN] - grouped.transform("mean")) ** 2
    return {
        "rows_sharing_descriptor_vector": int((sizes > 1).sum()),
        "groups_sharing_descriptor_vector": int((grouped.size() > 1).sum()),
        "floor_rmse": float(np.sqrt(within.sum() / len(dataset))),
    }


def neighbour_statistics(dataset: pd.DataFrame, folds: list[OuterFold]) -> pd.DataFrame:
    """Per-molecule nearest-training distance (standardized descriptors) and max Tanimoto, averaged over repeats."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(s.strip())) for s in dataset["smiles"]]
    X = dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    row_ids = dataset["row_id"].to_numpy()
    rows = []
    for outer in folds:
        scaler = StandardScaler().fit(X[outer.train_index])
        distances, _ = (
            NearestNeighbors(n_neighbors=1).fit(scaler.transform(X[outer.train_index])).kneighbors(scaler.transform(X[outer.test_index]))
        )
        train_fps = [fingerprints[i] for i in outer.train_index]
        similarity = [max(DataStructs.BulkTanimotoSimilarity(fingerprints[i], train_fps)) for i in outer.test_index]
        rows.append(pd.DataFrame({"row_id": row_ids[outer.test_index], "descriptor_distance": distances[:, 0], "max_tanimoto": similarity}))
    return pd.concat(rows).groupby("row_id").mean().reset_index()


# Orchestration -----------------------------------------------------------------


def scenario_inputs(dataset: pd.DataFrame, scenario: str, config: dict[str, Any]) -> dict[str, Any]:
    """Data, groups, block label and folds for one scenario."""
    grouping, variant, cv_protocol, holdout_protocol = SCENARIOS[scenario]
    data = make_variant(dataset, variant, config).reset_index(drop=True)
    groups = data[grouping].to_numpy()
    block = config["grouping"][grouping]["largest_group"]["label"]
    cv_spec = config["protocols"][cv_protocol]
    return {
        "data": data,
        "groups": groups,
        "block": block,
        "cv_protocol": cv_protocol,
        "holdout_protocol": holdout_protocol,
        "cv_folds": block_cv_folds(groups, block, cv_spec["outer_splits"], cv_spec["repeats"]),
        "holdout_folds": block_holdout_fold(groups, block),
    }


def run_scenario(dataset: pd.DataFrame, scenario: str, config: dict[str, Any], n_jobs: int, fold_limit: int | None = None) -> dict[str, pd.DataFrame]:
    """Run the CV and holdout protocols of one scenario."""
    inputs = scenario_inputs(dataset, scenario, config)
    specs = build_model_specs(config)
    outputs = []
    for protocol, folds in ((inputs["cv_protocol"], inputs["cv_folds"]), (inputs["holdout_protocol"], inputs["holdout_folds"])):
        outputs.append(
            run_protocol(
                inputs["data"],
                inputs["groups"],
                folds[:fold_limit] if fold_limit else folds,
                specs,
                scenario=scenario,
                protocol=protocol,
                block_label=inputs["block"],
                inner_splits=config["inner_cv"]["splits"],
                n_jobs=n_jobs,
            )
        )
    return {key: pd.concat([output[key] for output in outputs], ignore_index=True) for key in outputs[0]}


def analyse(results: dict[str, pd.DataFrame], dataset: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Compute every prespecified summary from the raw run outputs."""
    stats_cfg = config["statistics"]
    boot = stats_cfg["bootstrap"]
    delta = stats_cfg["practical_equivalence"]["delta_logS"]
    predictions = results["predictions"]
    main_predictions = predictions[predictions["model_seed"] == 42]
    fold_metrics = results["fold_metrics"]
    primary = stats_cfg["primary_contrast"]
    primary_pair = ("random_forest_reference", "xgboost")
    secondary_pairs = [tuple(pair) for pair in stats_cfg["secondary_contrasts"]["pairs"]]
    holdouts = {"benzene_holdout", "generic_block_holdout"}

    metrics_rows, contrast_rows, distributions = [], [], {}
    for (scenario, protocol), frame in main_predictions.groupby(["scenario", "protocol"], sort=False):
        folds = fold_metrics[(fold_metrics["scenario"] == scenario) & (fold_metrics["protocol"] == protocol)]
        metrics = protocol_metrics(frame, folds)
        metrics_rows.append(metrics.assign(scenario=scenario, protocol=protocol))
        pairs = list(itertools.combinations(list(dict.fromkeys(frame["model"])), 2))
        for pair in secondary_pairs:
            if pair not in pairs and pair[::-1] in pairs:
                pairs[pairs.index(pair[::-1])] = pair
        table, dists = contrast_table(
            frame, cluster_column="row_id" if protocol in holdouts else "group", replicates=boot["replicates"], seed=boot["seed"], pairs=pairs
        )
        table["scenario"], table["protocol"] = scenario, protocol
        table["uncertainty_method"] = "molecule bootstrap (conditional, descriptive)" if protocol in holdouts else "scaffold-cluster bootstrap"
        contrast_rows.append(table)
        distributions.update({(scenario, protocol, *pair): dist for pair, dist in dists.items()})

    protocol_table = pd.concat(metrics_rows, ignore_index=True)
    contrasts = pd.concat(contrast_rows, ignore_index=True)
    contrasts["role"] = "exploratory"
    contrasts["decision_label"] = None
    contrasts["holm_p_value"] = None
    is_primary_cv = (contrasts["protocol"] == primary["protocol"]) & (contrasts["model_a"] == primary_pair[0]) & (contrasts["model_b"] == primary_pair[1])
    contrasts.loc[is_primary_cv & (contrasts["scenario"] == "murcko_S0"), "role"] = "primary"
    contrasts.loc[is_primary_cv & contrasts["scenario"].isin(["murcko_S1", "murcko_S2"]), "role"] = "primary (sensitivity)"
    secondary_mask = (
        (contrasts["scenario"] == "murcko_S0")
        & (contrasts["protocol"] == primary["protocol"])
        & contrasts.apply(lambda row: (row["model_a"], row["model_b"]) in secondary_pairs, axis=1)
    )
    contrasts.loc[secondary_mask, "role"] = "secondary"
    contrasts.loc[secondary_mask, "holm_p_value"] = holm_adjust(contrasts.loc[secondary_mask, "p_value"].tolist())
    labelled = contrasts["role"] != "exploratory"
    contrasts.loc[labelled, "decision_label"] = [
        decision_label(lo, hi, delta) for lo, hi in zip(contrasts.loc[labelled, "ci_lower"], contrasts.loc[labelled, "ci_upper"])
    ]

    # Supporting statistics for the primary pair.
    s0_cv = main_predictions[(main_predictions["scenario"] == "murcko_S0") & (main_predictions["protocol"] == primary["protocol"])]
    molecules = molecule_errors(s0_cv)
    signed = molecules.pivot(index="row_id", columns="model", values="signed_error")
    squared = molecules.pivot(index="row_id", columns="model", values="sq_error")
    per_repeat = s0_cv[s0_cv["model"].isin(primary_pair)].groupby(["repeat", "model"]).apply(
        lambda f: np.sqrt(mean_squared_error(f["y_true"], f["y_pred"])), include_groups=False
    ).unstack()
    s0_folds = fold_metrics[(fold_metrics["scenario"] == "murcko_S0") & (fold_metrics["protocol"] == primary["protocol"])]
    continuity = {}
    for pair in [primary_pair, *secondary_pairs]:
        a = s0_folds[s0_folds["model"] == pair[0]].sort_values(["repeat", "fold"])
        b = s0_folds[s0_folds["model"] == pair[1]].sort_values(["repeat", "fold"])
        continuity[f"{pair[0]} - {pair[1]}"] = corrected_paired_ttest(
            a["RMSE"].to_numpy() - b["RMSE"].to_numpy(), a["n_train"].to_numpy(), a["n_test"].to_numpy()
        )
    supporting = {
        "signed_error_correlation": float(np.corrcoef(signed[primary_pair[0]], signed[primary_pair[1]])[0, 1]),
        "fraction_molecules_xgboost_lower_error": float((squared["xgboost"] < squared["random_forest_reference"]).mean()),
        "fraction_molecules_tied": float((squared["xgboost"] == squared["random_forest_reference"]).mean()),
        "per_repeat_delta": (per_repeat[primary_pair[0]] - per_repeat[primary_pair[1]]).round(6).tolist(),
        "phase6c_continuity_nadeau_bengio": continuity,
    }

    # Composite out-of-fold result (descriptive only).
    s0 = main_predictions[main_predictions["scenario"] == "murcko_S0"]
    composite_molecules = pd.concat([molecule_errors(s0[s0["protocol"] == protocol]) for protocol in ("nonbenzene_scaffold_cv", "benzene_holdout")])
    composite = {
        "label": config["protocols"]["composite_out_of_fold"]["label_in_reports"],
        "n_molecules": int(composite_molecules["row_id"].nunique()),
        "rmse": {model: float(np.sqrt(frame["sq_error"].mean())) for model, frame in composite_molecules.groupby("model", sort=False)},
    }

    # Sensitivity: primary contrast per variant and S0 rescored on each variant's molecules.
    sensitivity_rows = []
    s0_variants = {variant: make_variant(dataset, variant, config) for variant in ("S0", "S1", "S2")}
    for variant, variant_data in s0_variants.items():
        for protocol in config["sensitivity"]["applies_to_protocols"]:
            row = contrasts[(contrasts["scenario"] == f"murcko_{variant}") & (contrasts["protocol"] == protocol) & (contrasts["model_a"] == primary_pair[0]) & (contrasts["model_b"] == primary_pair[1])].iloc[0]
            s0_subset = s0[(s0["protocol"] == protocol) & s0["row_id"].isin(dataset.loc[dataset["identity"].isin(variant_data["identity"]), "row_id"])]
            subset_table, _ = contrast_table(
                s0_subset, cluster_column="row_id" if protocol in holdouts else "group", replicates=boot["replicates"], seed=boot["seed"], pairs=[primary_pair]
            )
            variant_metrics = protocol_table[(protocol_table["scenario"] == f"murcko_{variant}") & (protocol_table["protocol"] == protocol)]
            sensitivity_rows.append(
                {
                    "variant": variant,
                    "protocol": protocol,
                    "rows": int(len(variant_data)),
                    "rmse_by_model": dict(zip(variant_metrics["model"], variant_metrics["repeat_averaged_rmse"].round(6))),
                    "primary_delta": float(row["delta"]),
                    "ci_lower": float(row["ci_lower"]),
                    "ci_upper": float(row["ci_upper"]),
                    "decision_label": row["decision_label"] if protocol == primary["protocol"] else None,
                    "s0_rescored_on_variant_molecules": {k: float(subset_table.iloc[0][k]) for k in ("delta", "ci_lower", "ci_upper")},
                }
            )
    labels = {r["variant"]: r["decision_label"] for r in sensitivity_rows if r["protocol"] == primary["protocol"]}
    sensitivity = {"rows": sensitivity_rows, "robustness_criterion_met": len(set(labels.values())) == 1, "labels": labels}

    # Model-seed check.
    seed_rows = []
    for protocol in config["model_seed_check"]["protocols"]:
        frame = predictions[(predictions["scenario"] == "murcko_S0") & (predictions["protocol"] == protocol) & predictions["model"].isin(primary_pair)]
        for seed, seed_frame in frame.groupby("model_seed"):
            errors = molecule_errors(seed_frame).pivot(index="row_id", columns="model", values="sq_error")
            seed_rows.append(
                {
                    "protocol": protocol,
                    "model_seed": int(seed),
                    "rmse_random_forest_reference": float(np.sqrt(errors[primary_pair[0]].mean())),
                    "rmse_xgboost": float(np.sqrt(errors[primary_pair[1]].mean())),
                    "delta": float(np.sqrt(errors[primary_pair[0]].mean()) - np.sqrt(errors[primary_pair[1]].mean())),
                }
            )
    seed_table = pd.DataFrame(seed_rows)
    seed_check = {
        "rows": seed_rows,
        "delta_range": {p: [float(f["delta"].min()), float(f["delta"].max())] for p, f in seed_table.groupby("protocol")},
    }

    return {
        "protocol_metrics": protocol_table,
        "contrasts": contrasts,
        "distributions": distributions,
        "supporting": supporting,
        "composite": composite,
        "sensitivity": sensitivity,
        "seed_check": seed_check,
        "boundary": boundary_report(fold_metrics, config),
    }


def exploratory_analyses(dataset: pd.DataFrame, results: dict[str, pd.DataFrame], config: dict[str, Any]) -> dict[str, Any]:
    """Error floor, family metrics, descriptor distance and structural similarity (S0, Murcko)."""
    inputs = scenario_inputs(dataset, "murcko_S0", config)
    families = assign_families(dataset, config).set_axis(dataset["row_id"])
    predictions = results["predictions"]
    main = predictions[(predictions["scenario"] == "murcko_S0") & (predictions["model_seed"] == 42)]
    cv = main[main["protocol"] == "nonbenzene_scaffold_cv"]
    holdout = main[main["protocol"] == "benzene_holdout"]
    family_table = family_metrics(cv, holdout, families, dataset.set_index("row_id")["murcko"], config)

    neighbours = pd.concat([neighbour_statistics(inputs["data"], inputs["cv_folds"]), neighbour_statistics(inputs["data"], inputs["holdout_folds"])])
    neighbours["family"] = neighbours["row_id"].map(families)
    neighbours["acyclic"] = neighbours["row_id"].map(dataset.set_index("row_id")["acyclic"])
    errors = pd.concat([molecule_errors(cv), molecule_errors(holdout)])
    correlations = {}
    for model, frame in errors.groupby("model", sort=False):
        merged = frame.merge(neighbours, on="row_id")
        correlations[model] = float(stats.spearmanr(merged["descriptor_distance"], merged["abs_error"]).statistic)
    similarity = (
        neighbours.groupby("family")["max_tanimoto"]
        .describe(percentiles=[0.1, 0.5, 0.9])
        .rename(columns={"10%": "q10", "50%": "median", "90%": "q90"})
        .reset_index()
    )
    similarity_by_ring = neighbours.groupby("acyclic")["max_tanimoto"].describe(percentiles=[0.1, 0.5, 0.9]).reset_index()
    return {
        "family_metrics": family_table,
        "error_floor": error_floor(dataset),
        "descriptor_distance_spearman_with_abs_error": correlations,
        "similarity_by_family": similarity,
        "similarity_acyclic_vs_cyclic": similarity_by_ring,
        "neighbours": neighbours,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict(orient="records"))
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2))


def environment_info() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "xgboost": xgboost.__version__,
        "rdkit": rdkit.__version__,
        "platform": platform.platform(),
    }


def run_benchmark(dataset: pd.DataFrame, config: dict[str, Any], n_jobs: int) -> dict[str, Any]:
    """Time one CV fold and the holdout per grouping, then extrapolate to the full run."""
    timings = {}
    for scenario in ("murcko_S0", "generic_S0"):
        start = perf_counter()
        inputs = scenario_inputs(dataset, scenario, config)
        specs = build_model_specs(config)
        common = dict(block_label=inputs["block"], inner_splits=config["inner_cv"]["splits"], n_jobs=n_jobs, scenario=scenario)
        t0 = perf_counter()
        cv = run_protocol(inputs["data"], inputs["groups"], inputs["cv_folds"][:1], specs, protocol=inputs["cv_protocol"], **common)
        cv_seconds = perf_counter() - t0
        t0 = perf_counter()
        run_protocol(inputs["data"], inputs["groups"], inputs["holdout_folds"], specs, protocol=inputs["holdout_protocol"], **common)
        holdout_seconds = perf_counter() - t0
        t0 = perf_counter()
        run_seed_check(
            inputs["data"], inputs["cv_folds"][:1], cv["fold_metrics"], specs, [43], scenario=scenario, protocol=inputs["cv_protocol"], groups=inputs["groups"]
        )
        seed_seconds = perf_counter() - t0
        timings[scenario] = {"cv_fold_seconds": cv_seconds, "holdout_seconds": holdout_seconds, "seed_refit_seconds_per_fold_per_seed": seed_seconds, "elapsed": perf_counter() - start}

    repeats = config["protocols"]["nonbenzene_scaffold_cv"]["repeats"] * config["protocols"]["nonbenzene_scaffold_cv"]["outer_splits"]
    murcko = timings["murcko_S0"]
    per_murcko_scenario = murcko["cv_fold_seconds"] * repeats + murcko["holdout_seconds"]
    generic = timings["generic_S0"]["cv_fold_seconds"] * repeats + timings["generic_S0"]["holdout_seconds"]
    seeds = len(config["model_seed_check"]["additional_random_states"])
    seed_total = murcko["seed_refit_seconds_per_fold_per_seed"] * seeds * (repeats + 1)
    total = 3 * per_murcko_scenario + generic + seed_total
    return {
        "n_jobs": n_jobs,
        "measured_seconds": timings,
        "estimated_seconds": {
            "per_murcko_scenario": per_murcko_scenario,
            "generic_scenario": generic,
            "model_seed_check": seed_total,
            "total_model_fitting": total,
        },
        "estimated_total_minutes": total / 60,
        "approval_threshold_minutes": config["runtime"]["approval_threshold_wall_minutes"],
        "within_threshold": total / 60 <= config["runtime"]["approval_threshold_wall_minutes"],
    }


def main() -> None:
    """Run the benchmark or the full prespecified Phase 6D experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="Time one outer fold per protocol and estimate total runtime.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results" / "phase6d")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports" / "phase6d")
    args = parser.parse_args()

    config, config_md5 = load_config(args.config)
    validate_config(config)
    verify_checksums(config["protected_files_md5"])
    data_spec = config["data"]
    verify_checksums({data_spec["descriptor_path"]: data_spec["descriptor_md5"], data_spec["raw_path"]: data_spec["raw_md5"]})
    dataset = annotate_dataset(pd.read_csv(PROJECT_ROOT / data_spec["descriptor_path"]))
    validate_expected_counts(dataset, config)

    if args.benchmark:
        benchmark = run_benchmark(dataset, config, args.n_jobs)
        benchmark.update({"config_md5": config_md5, "environment": environment_info(), "run_at_utc": datetime.now(timezone.utc).isoformat()})
        _write_json(args.reports_dir / "benchmark.json", benchmark)
        print(json.dumps(_jsonable(benchmark), indent=2))
        return

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{config_md5[:8]}"
    run_dir = create_run_directory(args.results_root, run_id)
    start = perf_counter()

    scenario_results = []
    for scenario in SCENARIOS:
        scenario_results.append(run_scenario(dataset, scenario, config, args.n_jobs))
        print(f"Completed {scenario} after {perf_counter() - start:.1f} s", flush=True)
    results = {key: pd.concat([r[key] for r in scenario_results], ignore_index=True) for key in scenario_results[0]}

    s0 = scenario_inputs(dataset, "murcko_S0", config)
    specs = build_model_specs(config)
    seeds = config["model_seed_check"]["additional_random_states"]
    for protocol, folds in ((s0["cv_protocol"], s0["cv_folds"]), (s0["holdout_protocol"], s0["holdout_folds"])):
        seed_predictions = run_seed_check(
            s0["data"], folds, results["fold_metrics"][results["fold_metrics"]["scenario"] == "murcko_S0"], specs, seeds,
            scenario="murcko_S0", protocol=protocol, groups=s0["groups"],
        )
        results["predictions"] = pd.concat([results["predictions"], seed_predictions], ignore_index=True)
    print(f"Completed model-seed check after {perf_counter() - start:.1f} s", flush=True)

    analysis = analyse(results, dataset, config)
    exploratory = exploratory_analyses(dataset, results, config)
    runtime_seconds = perf_counter() - start
    verify_checksums(config["protected_files_md5"])

    run_info = {
        "run_id": run_id,
        "config_md5": config_md5,
        "runtime_seconds": runtime_seconds,
        "n_jobs": args.n_jobs,
        "environment": environment_info(),
        "data_md5": {data_spec["descriptor_path"]: data_spec["descriptor_md5"], data_spec["raw_path"]: data_spec["raw_md5"]},
    }
    (run_dir / "config_snapshot.json").write_text(args.config.read_text())
    _write_json(run_dir / "run_info.json", run_info)
    (run_dir / "pip_freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    results["predictions"].to_csv(run_dir / "oof_predictions.csv.gz", index=False)
    results["fold_metrics"].to_csv(run_dir / "fold_metrics.csv", index=False)
    results["inner_results"].to_csv(run_dir / "inner_cv_results.csv.gz", index=False)
    results["assignments"].to_csv(run_dir / "fold_assignments.csv.gz", index=False)
    analysis["protocol_metrics"].to_csv(run_dir / "protocol_metrics.csv", index=False)
    analysis["contrasts"].to_csv(run_dir / "contrasts.csv", index=False)
    analysis["boundary"].to_csv(run_dir / "boundary_report.csv", index=False)
    exploratory["family_metrics"].to_csv(run_dir / "family_metrics.csv", index=False)
    exploratory["neighbours"].to_csv(run_dir / "neighbour_statistics.csv", index=False)
    np.savez_compressed(run_dir / "bootstrap_distributions.npz", **{"|".join(key): value for key, value in analysis["distributions"].items()})

    reports = args.reports_dir
    _write_json(
        reports / "summary.json",
        {
            **run_info,
            "protocol_metrics": analysis["protocol_metrics"],
            "composite_out_of_fold": analysis["composite"],
            "primary_contrast": analysis["contrasts"][analysis["contrasts"]["role"] == "primary"],
            "primary_supporting_statistics": analysis["supporting"],
            "conditional_statement": f"Conditional on delta = {config['statistics']['practical_equivalence']['delta_logS']} LogS and on the validity of the scaffold-cluster bootstrap.",
        },
    )
    _write_json(reports / "contrasts.json", analysis["contrasts"])
    _write_json(reports / "families.json", {"family_metrics": exploratory["family_metrics"], "similarity_by_family": exploratory["similarity_by_family"]})
    _write_json(reports / "sensitivity.json", analysis["sensitivity"])
    _write_json(reports / "boundary_report.json", analysis["boundary"])
    _write_json(reports / "model_seed_check.json", analysis["seed_check"])
    _write_json(
        reports / "exploratory.json",
        {
            "error_floor": exploratory["error_floor"],
            "descriptor_distance_spearman_with_abs_error": exploratory["descriptor_distance_spearman_with_abs_error"],
            "similarity_acyclic_vs_cyclic": exploratory["similarity_acyclic_vs_cyclic"],
        },
    )
    print(f"Run {run_id} finished in {runtime_seconds:.1f} s; outputs in {run_dir}")


if __name__ == "__main__":
    main()
