"""Run the prespecified Phase 6E model diagnostics (configs/phase6e.json)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from scipy import stats
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.scaffold_evaluation import (
    annotate_dataset,
    assign_families,
    bootstrap_difference,
    bootstrap_rmse,
    build_model_specs,
    cluster_bootstrap_weights,
    create_run_directory,
    environment_info,
    error_floor,
    file_md5,
    load_config,
    molecule_errors,
    scenario_inputs,
    verify_checksums,
)
from src.train_model import FEATURE_COLUMNS, TARGET_COLUMN, split_dataset

RDLogger.DisableLog("rdApp.*")

CONFIG_PATH = PROJECT_ROOT / "configs" / "phase6e.json"
CV = "nonbenzene_scaffold_cv"
HOLDOUT = "benzene_holdout"
PROTOCOLS = (CV, HOLDOUT)
PROTOCOL_INDEX = {CV: 0, HOLDOUT: 1, "production": 2}
PROTOCOL_LABEL = {CV: "Non-benzene scaffold CV", HOLDOUT: "Benzene holdout"}
MODEL_LABEL = {
    "random_forest_reference": "RF reference",
    "random_forest_tuned": "RF tuned",
    "xgboost": "XGBoost",
    "lasso": "Lasso",
    "baseline": "Baseline",
    "linear_regression_reference": "Linear regression",
}
IMPORTANCE_MODELS = ("random_forest_reference", "xgboost")
FIGURE_MODELS = ("random_forest_reference", "xgboost", "lasso")


class DiagnosticGateError(RuntimeError):
    """Raised when a prespecified reproduction or integrity check fails."""


# Integrity checks -------------------------------------------------------------------


def check_reproduction(new: np.ndarray, stored: np.ndarray, tolerance: float, label: str) -> float:
    """Return the maximum absolute difference, stopping if it exceeds the tolerance."""
    difference = float(np.max(np.abs(np.asarray(new, dtype=float) - np.asarray(stored, dtype=float)))) if len(new) else 0.0
    if not difference <= tolerance:
        raise DiagnosticGateError(f"{label}: prediction difference {difference:.3e} exceeds {tolerance:.0e}")
    return difference


def verify_raw_alignment(descriptors: pd.DataFrame, raw: pd.DataFrame) -> None:
    """Stop unless the raw ESOL rows align with the descriptor rows by position."""
    if len(descriptors) != len(raw) or not (
        (descriptors["Compound ID"].to_numpy() == raw["Compound ID"].to_numpy()).all()
        and (descriptors["smiles"].to_numpy() == raw["smiles"].to_numpy()).all()
    ):
        raise DiagnosticGateError("Raw ESOL rows do not align with the descriptor rows.")


def verify_descriptor_functions(config: dict[str, Any], source: str) -> None:
    """Stop unless every configured RDKit function is the one used in src/descriptors.py."""
    for feature, entry in config["descriptor_reference"].items():
        if feature == "not_represented" or not isinstance(entry, dict):
            continue
        function = re.match(r"(Descriptors\.\w+)", entry["rdkit"]).group(1)
        if f"{function}(" not in source:
            raise DiagnosticGateError(f"{function} for {feature} is not used in src/descriptors.py")


def require_single_protocol(frame: pd.DataFrame) -> str:
    """Return the protocol of a frame, refusing frames that mix protocols."""
    protocols = frame["protocol"].unique()
    if len(protocols) != 1:
        raise ValueError(f"Statistics must not pool protocols: {sorted(protocols)}")
    return str(protocols[0])


# Importance building blocks ---------------------------------------------------------


def importance_units(config: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Single features, then primary groups, then sensitivity groups (fixed order = unit index)."""
    units = [(f"feature:{feature}", [feature]) for feature in FEATURE_COLUMNS]
    for kind in ("primary", "sensitivity"):
        for name, columns in config["feature_importance"]["groups"][kind].items():
            units.append((f"{kind}:{name}", list(columns)))
    return units


def permutation_indices(n: int, base_seed: int, protocol_index: int, repeat: int, fold: int, unit_index: int, permutation: int) -> np.ndarray:
    """Order-independent permutation drawn from a SeedSequence keyed on its position."""
    sequence = np.random.SeedSequence([base_seed, protocol_index, repeat, fold, unit_index, permutation])
    return np.random.default_rng(sequence).permutation(n)


def permuted_squared_errors(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    units: list[tuple[str, list[str]]],
    repeats: int,
    seed_key: tuple[int, int, int, int],
) -> dict[str, np.ndarray]:
    """Squared errors (repeats x molecules) after jointly permuting each unit within the test fold."""
    base_seed, protocol_index, repeat, fold = seed_key
    results = {}
    for unit_index, (name, columns) in enumerate(units):
        errors = np.empty((repeats, len(y_test)))
        for permutation in range(repeats):
            order = permutation_indices(len(y_test), base_seed, protocol_index, repeat, fold, unit_index, permutation)
            permuted = X_test.copy()
            permuted[columns] = X_test[columns].to_numpy()[order]
            errors[permutation] = (model.predict(permuted) - y_test) ** 2
        results[name] = errors
    return results


def drop_column_squared_errors(
    estimator: Any,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    units: list[tuple[str, list[str]]],
) -> dict[str, np.ndarray]:
    """Squared test errors after refitting without each unit, keeping the stored hyperparameters."""
    results = {}
    for name, columns in units:
        model = clone(estimator).set_params(**params)
        model.fit(X_train.drop(columns=columns), y_train)
        results[name] = (model.predict(X_test.drop(columns=columns)) - y_test) ** 2
    return results


def pooled_importance(perturbed_sq: np.ndarray, base_sq: np.ndarray) -> dict[str, Any]:
    """Pooled RMSE increase from prediction-level squared errors.

    perturbed_sq has shape (permutations, repeats, molecules); base_sq has shape (repeats, molecules).
    Each molecule's squared error is averaged over repeats before RMSE is taken over molecules.
    """
    base_rmse = float(np.sqrt(base_sq.mean(axis=0).mean()))
    per_permutation = np.sqrt(perturbed_sq.mean(axis=1).mean(axis=1)) - base_rmse
    return {
        "base_rmse": base_rmse,
        "importance_mean": float(per_permutation.mean()),
        "importance_sd": float(per_permutation.std(ddof=1)) if len(per_permutation) > 1 else 0.0,
        "per_permutation": per_permutation,
    }


def fold_rmse_increase(perturbed_sq: np.ndarray, base_sq: np.ndarray) -> float:
    """Fold-level RMSE increase, averaged over permutations (reported only as fold variability)."""
    return float(np.mean(np.sqrt(np.atleast_2d(perturbed_sq).mean(axis=1))) - np.sqrt(base_sq.mean()))


# ALE ------------------------------------------------------------------------------------


def ale_edges(values: np.ndarray, bins: int) -> np.ndarray:
    """Quantile edges, or unique values when there are fewer distinct values than bins."""
    unique = np.unique(values)
    if len(unique) < bins:
        return unique.astype(float)
    return np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))


def ale_curve(model: Any, X: pd.DataFrame, feature: str, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First-order accumulated local effects, centred to zero mean over the rows of X."""
    x = X[feature].to_numpy(dtype=float)
    edges = ale_edges(x, bins)
    if len(edges) < 2:
        return edges, np.zeros_like(edges), np.array([len(x)])
    interval = np.clip(np.searchsorted(edges, x, side="left"), 1, len(edges) - 1)
    lower, upper = X.copy(), X.copy()
    lower[feature] = edges[interval - 1]
    upper[feature] = edges[interval]
    difference = np.asarray(model.predict(upper), dtype=float) - np.asarray(model.predict(lower), dtype=float)
    n_intervals = len(edges) - 1
    counts = np.bincount(interval - 1, minlength=n_intervals)
    sums = np.bincount(interval - 1, weights=difference, minlength=n_intervals)
    effects = np.divide(sums, counts, out=np.zeros(n_intervals), where=counts > 0)
    accumulated = np.concatenate([[0.0], np.cumsum(effects)])
    midpoints = (accumulated[:-1] + accumulated[1:]) / 2
    return edges, accumulated - np.sum(midpoints * counts) / counts.sum(), counts


def interpolate_curve(grid: np.ndarray, edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interpolate a fold curve on a common grid without extrapolating beyond its edges."""
    result = np.interp(grid, edges, values)
    result[(grid < edges[0]) | (grid > edges[-1])] = np.nan
    return result


# Other per-molecule diagnostics -----------------------------------------------------------


def treeshap_contributions(model: Any, X: pd.DataFrame, tolerance: float) -> np.ndarray:
    """XGBoost path-dependent TreeSHAP; the last column is the bias term."""
    contributions = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    error = float(np.max(np.abs(contributions.sum(axis=1) - model.predict(X))))
    if error > tolerance:
        raise DiagnosticGateError(f"TreeSHAP additivity error {error:.2e} exceeds {tolerance}")
    return contributions


def tree_spread(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Standard deviation of individual Random Forest tree predictions."""
    values = X.to_numpy(dtype=float)
    return np.std([tree.predict(values) for tree in model.estimators_], axis=0)


def calibration_fit(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """OLS of observed on predicted: returns (slope, intercept)."""
    slope, intercept = np.polyfit(predicted, observed, 1)
    return float(slope), float(intercept)


def shrinkage_slope(observed: np.ndarray, predicted: np.ndarray) -> float:
    """OLS slope of the signed residual (predicted - observed) on observed."""
    return float(np.polyfit(observed, predicted - observed, 1)[0])


def bootstrap_ols(weights: np.ndarray, codes: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slope and intercept of y on x for every cluster-bootstrap replicate."""
    n_clusters = weights.shape[1]
    s, sx, sy, sxx, sxy = (weights @ np.bincount(codes, weights=v, minlength=n_clusters) for v in (np.ones_like(x), x, y, x * x, x * y))
    slope = (sxy - sx * sy / s) / (sxx - sx**2 / s)
    return slope, (sy - slope * sx) / s


def bootstrap_mean(weights: np.ndarray, codes: np.ndarray, values: np.ndarray) -> np.ndarray:
    n_clusters = weights.shape[1]
    return (weights @ np.bincount(codes, weights=values, minlength=n_clusters)) / (weights @ np.bincount(codes, minlength=n_clusters))


def similarity_bin(similarity: float, bins: list[Any]) -> str:
    """Half-open bins; exactly 1.0 forms its own bin."""
    if similarity == 1.0:
        return "exactly 1.0"
    for lower, upper in [b for b in bins if isinstance(b, list)]:
        if lower <= similarity < upper:
            return f"[{lower}, {upper})"
    raise ValueError(f"Similarity {similarity} outside configured bins")


def formula_counts(smiles: str) -> Counter:
    formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(str(smiles).strip()))
    counts: Counter = Counter()
    for element, number in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        counts[element] += int(number) if number else 1
    charge = re.findall(r"[+-]\d*$", formula)
    counts["charge"] = charge[0] if charge else ""
    return counts


def is_ch2_homologue(smiles_a: str, smiles_b: str) -> bool:
    """True when two molecular formulas differ by exactly one CH2 and nothing else."""
    a, b = formula_counts(smiles_a), formula_counts(smiles_b)
    if a["charge"] != b["charge"]:
        return False
    elements = (set(a) | set(b)) - {"charge"}
    difference = {element: a[element] - b[element] for element in elements if a[element] != b[element]}
    return difference in ({"C": 1, "H": 2}, {"C": -1, "H": -2})


def cross_fold_half_widths(residuals: pd.DataFrame, level: float) -> pd.Series:
    """Half-width for each (repeat, fold) from absolute residuals of the other folds of that repeat."""
    widths = {}
    for (repeat, fold), _ in residuals.groupby(["repeat", "fold"]):
        others = residuals[(residuals["repeat"] == repeat) & (residuals["fold"] != fold)]["residual"].abs()
        widths[(repeat, fold)] = float(np.quantile(others, level, method="higher"))
    return pd.Series(widths)


def select_extreme_errors(residuals: pd.DataFrame, reference: str, top_k: int, threshold: float, threshold_models: list[str]) -> list[int]:
    """Top-k molecules by reference |residual| plus any molecule above the threshold for listed models."""
    ranked = residuals[reference].abs().sort_values(ascending=False, kind="stable")
    selected = set(ranked.index[:top_k])
    for model in threshold_models:
        selected |= set(residuals.index[residuals[model].abs() > threshold])
    return sorted(selected, key=lambda row: (-abs(residuals.loc[row, reference]), row))


# Per-fold work ------------------------------------------------------------------------------


def diagnose_fold(task: dict[str, Any]) -> dict[str, Any]:
    """All model-dependent diagnostics for one outer fold (runs in a worker process)."""
    data, config = task["data"], task["config"]
    train, test = task["train_index"], task["test_index"]
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN].to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y[train], y[test]
    units = importance_units(config)
    importance = config["feature_importance"]
    seed_key = (importance["permutation"]["seed"], PROTOCOL_INDEX[task["protocol"]], task["repeat"], task["fold"])
    out: dict[str, Any] = {key: task[key] for key in ("protocol", "repeat", "fold")}
    out["row_id"] = data["row_id"].to_numpy()[test]

    out["permutation"], out["drop_column"], out["base_sq"] = {}, {}, {}
    for name in IMPORTANCE_MODELS:
        model = task["models"][name]
        out["base_sq"][name] = (model.predict(X_test) - y_test) ** 2
        out["permutation"][name] = permuted_squared_errors(model, X_test, y_test, units, importance["permutation"]["repeats"], seed_key)
        out["drop_column"][name] = drop_column_squared_errors(task["estimators"][name], task["params"][name], X_train, y_train, X_test, y_test, units)

    rf, booster_model = task["models"]["random_forest_reference"], task["models"]["xgboost"]
    gain = booster_model.get_booster().get_score(importance_type="total_gain")
    out["impurity"] = {
        "random_forest_reference": dict(zip(FEATURE_COLUMNS, rf.feature_importances_.tolist())),
        "xgboost_total_gain": {feature: float(gain.get(feature, 0.0)) for feature in FEATURE_COLUMNS},
    }
    out["shap"] = treeshap_contributions(booster_model, X_test, importance["treeshap"]["additivity_tolerance"])
    out["lasso_coefficients"] = dict(zip(FEATURE_COLUMNS, task["models"]["lasso"].named_steps["lasso"].coef_.tolist()))
    out["ale"] = {
        (name, feature): ale_curve(task["models"][name], X_train, feature, importance["ale"]["bins"])
        for name in importance["ale"]["models"]
        for feature in FEATURE_COLUMNS
    }
    out["tree_sd"] = tree_spread(rf, X_test)
    out["predictions"] = {name: task["models"][name].predict(X_test) for name in IMPORTANCE_MODELS}

    # Neighbours in descriptor space and fingerprint space.
    scaler = StandardScaler().fit(X_train)
    neighbours = NearestNeighbors(n_neighbors=config["families"]["extrapolation_indicators"]["neighbours_k"]).fit(scaler.transform(X_train))
    distances, indices = neighbours.kneighbors(scaler.transform(X_test))
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=config["acyclic"]["fingerprint"]["radius"], fpSize=config["acyclic"]["fingerprint"]["bits"])
    smiles = data["smiles"].to_numpy()
    train_fps = [generator.GetFingerprint(Chem.MolFromSmiles(s.strip())) for s in smiles[train]]
    similarity, nearest = [], []
    for s in smiles[test]:
        values = np.asarray(DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(Chem.MolFromSmiles(s.strip())), train_fps))
        best = int(np.argmax(values))  # first maximum = lowest row_id, because train indices are sorted
        similarity.append(float(values[best]))
        nearest.append(int(data["row_id"].to_numpy()[train][best]))
    lows, highs = X_train.min().to_numpy(), X_train.max().to_numpy()
    logp = FEATURE_COLUMNS.index("LogP")
    out["neighbours"] = pd.DataFrame(
        {
            "row_id": out["row_id"],
            "descriptor_distance": distances[:, 0],
            "max_tanimoto": similarity,
            "nearest_row_id": nearest,
            "below_training_min_target": y_test < y_train.min(),
            "below_knn_min_target": y_test < y_train[indices].min(axis=1),
            "logp_above_training_max": X_test["LogP"].to_numpy() > highs[logp],
            "any_descriptor_out_of_range": ((X_test.to_numpy() < lows) | (X_test.to_numpy() > highs)).any(axis=1),
        }
    )
    return out


# Main analysis ---------------------------------------------------------------------------------


class Diagnostics:
    """Loads inputs, runs the refit gate and fold work, and computes every table and figure."""

    def __init__(self, config: dict[str, Any], config_md5: str, n_jobs: int) -> None:
        self.config, self.config_md5, self.n_jobs = config, config_md5, n_jobs
        self.cfg6d, self.cfg6d_md5 = load_config(PROJECT_ROOT / config["inputs"]["phase6d_config"])
        boot = config["uncertainty"]
        self.replicates, self.seed = boot["bootstrap_replicates"], boot["bootstrap_seed"]
        self.run6d = PROJECT_ROOT / config["inputs"]["phase6d_run_dir"]
        self.checks: dict[str, Any] = {}

    # Loading and gates ------------------------------------------------------------------

    def load(self) -> None:
        raw = pd.read_csv(PROJECT_ROOT / self.config["inputs"]["raw_data"])
        descriptors = pd.read_csv(PROJECT_ROOT / self.config["inputs"]["descriptor_data"])
        verify_raw_alignment(descriptors, raw)
        verify_descriptor_functions(self.config, (PROJECT_ROOT / "src" / "descriptors.py").read_text())
        self.dataset = annotate_dataset(descriptors)
        self.dataset["esol_equation"] = raw[self.config["esol_equation_comparator"]["column"]].to_numpy()
        self.dataset["family"] = assign_families(self.dataset, self.cfg6d).to_numpy()
        self.inputs = scenario_inputs(self.dataset, "murcko_S0", self.cfg6d)
        self.data = self.inputs["data"]
        if not (self.data["row_id"].to_numpy() == np.arange(len(self.data))).all():
            raise DiagnosticGateError("S0 data rows are not in row_id order.")
        self.folds = [(CV, fold) for fold in self.inputs["cv_folds"]] + [(HOLDOUT, fold) for fold in self.inputs["holdout_folds"]]
        oof = pd.read_csv(self.run6d / "oof_predictions.csv.gz")
        self.oof = oof[(oof["scenario"] == "murcko_S0") & (oof["model_seed"] == 42)].copy()
        fold_metrics = pd.read_csv(self.run6d / "fold_metrics.csv")
        self.fold_metrics = fold_metrics[fold_metrics["scenario"] == "murcko_S0"].copy()
        assignments = pd.read_csv(self.run6d / "fold_assignments.csv.gz")
        assignments = assignments[assignments["scenario"] == "murcko_S0"]
        for protocol, fold in self.folds:
            stored = assignments[(assignments["protocol"] == protocol) & (assignments["repeat"] == fold.repeat) & (assignments["fold"] == fold.fold)]["row_id"]
            if sorted(stored) != sorted(self.data["row_id"].to_numpy()[fold.test_index]):
                raise DiagnosticGateError(f"Fold assignment mismatch for {protocol} repeat {fold.repeat} fold {fold.fold}")
        self.checks["fold_assignments_match_phase6d"] = True

    def refit(self) -> None:
        """Refit every fold model and stop if any prediction differs from Phase 6D."""
        specs = build_model_specs(self.cfg6d)
        X = self.data[FEATURE_COLUMNS]
        y = self.data[TARGET_COLUMN].to_numpy(dtype=float)
        tolerance = self.config["refit"]["reproduction_check"]["max_abs_difference"]
        self.models, self.params, max_difference = {}, {}, {}
        for protocol, fold in self.folds:
            for name in self.config["scope"]["models"]["refit_and_reproduce"]:
                row = self.fold_metrics[(self.fold_metrics["protocol"] == protocol) & (self.fold_metrics["repeat"] == fold.repeat) & (self.fold_metrics["fold"] == fold.fold) & (self.fold_metrics["model"] == name)]
                params = json.loads(row["selected_params"].iloc[0])
                model = clone(specs[name][0]).set_params(**params).fit(X.iloc[fold.train_index], y[fold.train_index])
                stored = self.oof[(self.oof["protocol"] == protocol) & (self.oof["repeat"] == fold.repeat) & (self.oof["fold"] == fold.fold) & (self.oof["model"] == name)].set_index("row_id")
                stored = stored.loc[self.data["row_id"].to_numpy()[fold.test_index], "y_pred"].to_numpy()
                difference = check_reproduction(model.predict(X.iloc[fold.test_index]), stored, tolerance, f"{name} {protocol} r{fold.repeat} f{fold.fold}")
                max_difference[name] = max(max_difference.get(name, 0.0), difference)
                self.models[(protocol, fold.repeat, fold.fold, name)] = model
                self.params[(protocol, fold.repeat, fold.fold, name)] = params
        self.specs = specs
        self.checks["refit_max_abs_difference"] = max_difference

    def fold_tasks(self, limit: int | None = None) -> list[dict[str, Any]]:
        folds = self.folds if limit is None else self.folds[:limit]
        return [
            {
                "protocol": protocol,
                "repeat": fold.repeat,
                "fold": fold.fold,
                "train_index": fold.train_index,
                "test_index": fold.test_index,
                "data": self.data[["row_id", "smiles", TARGET_COLUMN, *FEATURE_COLUMNS]],
                "config": self.config,
                "models": {name: self.models[(protocol, fold.repeat, fold.fold, name)] for name in ("random_forest_reference", "xgboost", "lasso")},
                "estimators": {name: self.specs[name][0] for name in IMPORTANCE_MODELS},
                "params": {name: self.params[(protocol, fold.repeat, fold.fold, name)] for name in IMPORTANCE_MODELS},
            }
            for protocol, fold in folds
        ]

    def run_folds(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return Parallel(n_jobs=self.n_jobs)(delayed(diagnose_fold)(task) for task in tasks)

    # Helpers ----------------------------------------------------------------------------------------

    def population(self, protocol: str) -> np.ndarray:
        rows = self.oof[self.oof["protocol"] == protocol]["row_id"].unique()
        return np.sort(rows)

    def clusters(self, protocol: str, rows: np.ndarray) -> np.ndarray:
        return self.data.loc[rows, "murcko"].to_numpy() if protocol == CV else np.asarray(rows)

    def weights(self, protocol: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return cluster_bootstrap_weights(self.clusters(protocol, rows), self.replicates, self.seed)

    def molecules(self, protocol: str) -> pd.DataFrame:
        """Per-molecule table: observed, mean prediction, repeat-averaged squared and absolute errors."""
        frame = self.oof[self.oof["protocol"] == protocol]
        require_single_protocol(frame)
        errors = molecule_errors(frame)
        table = errors.pivot(index="row_id", columns="model", values="sq_error").add_prefix("sq:")
        table = table.join(errors.pivot(index="row_id", columns="model", values="signed_error").add_prefix("res:"))
        table = table.join(errors.pivot(index="row_id", columns="model", values="abs_error").add_prefix("abs:"))
        table["observed"] = self.data.loc[table.index, TARGET_COLUMN].to_numpy()
        for model in errors["model"].unique():
            table[f"pred:{model}"] = table["observed"] + table[f"res:{model}"]
        table["esol_equation"] = self.data.loc[table.index, "esol_equation"].to_numpy()
        table["family"] = self.data.loc[table.index, "family"].to_numpy()
        table["murcko"] = self.data.loc[table.index, "murcko"].to_numpy()
        table["acyclic"] = self.data.loc[table.index, "acyclic"].to_numpy()
        return table.sort_index()

    def ci(self, distribution: np.ndarray) -> list[float]:
        return [float(v) for v in np.percentile(distribution, [2.5, 97.5])]

    # Analyses ----------------------------------------------------------------------------------------

    def analyse(self, fold_results: list[dict[str, Any]]) -> dict[str, Any]:
        self.fold_results = fold_results
        self.tables: dict[str, Any] = {}
        self.mol = {protocol: self.molecules(protocol) for protocol in PROTOCOLS}
        self.neighbours = self.neighbour_summary()
        self.tables["T1_calibration.json"] = self.calibration()
        self.tables["T3_feature_importance.json"] = self.feature_importance()
        self.tables["T4_descriptor_reference.json"] = self.descriptor_reference()
        self.tables["T5_family_diagnostics.json"] = self.family_diagnostics()
        self.tables["T6_focus_family_molecules.json"] = self.focus_families()
        self.tables["T2_extreme_errors.json"] = self.extreme_errors()
        self.tables["T7_similarity_distance.json"] = self.similarity_distance()
        self.tables["T8_representation_floor.json"] = self.representation_floor()
        self.tables["T9_interval_coverage.json"] = self.intervals()
        self.tables["T10_production_model.json"] = self.production_model()
        return self.tables

    def neighbour_summary(self) -> dict[str, pd.DataFrame]:
        frames = []
        for result in self.fold_results:
            frames.append(result["neighbours"].assign(protocol=result["protocol"], repeat=result["repeat"]))
        neighbours = pd.concat(frames, ignore_index=True)
        stored = pd.read_csv(self.run6d / "neighbour_statistics.csv").set_index("row_id").sort_index()
        recomputed = neighbours.groupby("row_id")[["descriptor_distance", "max_tanimoto"]].mean().sort_index()
        tolerance = 1e-9
        difference = float(np.max(np.abs(recomputed.to_numpy() - stored.loc[recomputed.index, ["descriptor_distance", "max_tanimoto"]].to_numpy())))
        if difference > tolerance:
            raise DiagnosticGateError(f"Recomputed neighbour statistics differ from Phase 6D by {difference:.2e}")
        self.checks["neighbour_statistics_max_abs_difference"] = difference
        summary = {}
        flags = ["below_training_min_target", "below_knn_min_target", "logp_above_training_max", "any_descriptor_out_of_range"]
        for protocol in PROTOCOLS:
            frame = neighbours[neighbours["protocol"] == protocol]
            grouped = frame.groupby("row_id")
            table = grouped[["descriptor_distance", "max_tanimoto"]].mean()
            table[flags] = grouped[flags].mean()
            table["nearest_row_id"] = grouped["nearest_row_id"].agg(lambda rows: sorted(Counter(rows).items(), key=lambda item: (-item[1], item[0]))[0][0])
            summary[protocol] = table.sort_index()
        self.neighbour_rows = neighbours
        return summary

    def calibration(self) -> dict[str, Any]:
        cal_cfg, res_cfg = self.config["calibration"], self.config["residuals"]
        output: dict[str, Any] = {}
        for protocol in PROTOCOLS:
            mol = self.mol[protocol]
            weights, codes = self.weights(protocol, mol.index.to_numpy())
            n_bins = 10 if protocol == CV else 5
            entry = {}
            for model in self.config["scope"]["models"]["residual_and_calibration"]:
                observed, predicted = mol["observed"].to_numpy(), mol[f"pred:{model}"].to_numpy()
                residual = predicted - observed
                slope, intercept = calibration_fit(observed, predicted)
                boot_slope, boot_intercept = bootstrap_ols(weights, codes, predicted, observed)
                shrink_boot, _ = bootstrap_ols(weights, codes, observed, residual)
                bins = pd.qcut(predicted, n_bins, labels=False, duplicates="drop")
                binned = pd.DataFrame({"bin": bins, "predicted": predicted, "observed": observed}).groupby("bin").agg(mean_predicted=("predicted", "mean"), mean_observed=("observed", "mean"), n=("observed", "size"))
                bias_bins = {}
                for by in res_cfg["bias_bins"]["by"]:
                    source = predicted if by == "predicted" else observed if by == "observed" else self.data.loc[mol.index, by].to_numpy()
                    labels = pd.qcut(source, res_cfg["bias_bins"]["bins"], labels=False, duplicates="drop")
                    rows = []
                    for label in np.unique(labels):
                        mask = labels == label
                        sub_w, sub_c = self.weights(protocol, mol.index.to_numpy()[mask])
                        rows.append({"bin": int(label), "center": float(np.median(source[mask])), "lower": float(source[mask].min()), "upper": float(source[mask].max()), "n": int(mask.sum()), "mean_residual": float(residual[mask].mean()), "ci": self.ci(bootstrap_mean(sub_w, sub_c, residual[mask]))})
                    bias_bins[by] = rows
                entry[model] = {
                    "n": int(len(mol)),
                    "calibration_slope": slope,
                    "calibration_slope_ci": self.ci(boot_slope),
                    "calibration_intercept": intercept,
                    "calibration_intercept_ci": self.ci(boot_intercept),
                    "shrinkage_slope": shrinkage_slope(observed, predicted),
                    "shrinkage_slope_ci": self.ci(shrink_boot),
                    "binned_calibration": binned.reset_index().to_dict(orient="records"),
                    "residual_distribution": {
                        "mean": float(residual.mean()),
                        "sd": float(residual.std(ddof=1)),
                        "quantiles": {str(q): float(np.quantile(residual, q)) for q in res_cfg["quantiles"]},
                        "skewness": float(stats.skew(residual, bias=False)),
                        "excess_kurtosis": float(stats.kurtosis(residual, fisher=True, bias=False)),
                    },
                    "bias_bins": bias_bins,
                }
            output[protocol] = {"label": PROTOCOL_LABEL[protocol], "uncertainty": "scaffold-cluster bootstrap" if protocol == CV else "molecule bootstrap (conditional, descriptive)", "models": entry}
        return output

    def feature_importance(self) -> dict[str, Any]:
        units = importance_units(self.config)
        repeats = self.config["feature_importance"]["permutation"]["repeats"]
        output: dict[str, Any] = {"units": [name for name, _ in units], "caveat": "Correlated descriptors share credit; single-feature importance understates their joint role and permutation creates unrealistic descriptor combinations. Importance describes these models on ESOL, not chemical causation."}
        self.importance_arrays: dict[Any, Any] = {}
        for protocol in PROTOCOLS:
            rows = self.population(protocol)
            position = pd.Series(np.arange(len(rows)), index=rows)
            n_repeats = 5 if protocol == CV else 1
            weights, codes = self.weights(protocol, rows)
            results = [r for r in self.fold_results if r["protocol"] == protocol]
            protocol_out: dict[str, Any] = {}
            for model in IMPORTANCE_MODELS:
                base = np.full((n_repeats, len(rows)), np.nan)
                perm = {name: np.full((repeats, n_repeats, len(rows)), np.nan) for name, _ in units}
                drop = {name: np.full((1, n_repeats, len(rows)), np.nan) for name, _ in units}
                fold_perm = {name: [] for name, _ in units}
                fold_drop = {name: [] for name, _ in units}
                for result in results:
                    idx = position.loc[result["row_id"]].to_numpy()
                    base[result["repeat"], idx] = result["base_sq"][model]
                    for name, _ in units:
                        perm[name][:, result["repeat"], idx] = result["permutation"][model][name]
                        drop[name][0, result["repeat"], idx] = result["drop_column"][model][name]
                        fold_perm[name].append(fold_rmse_increase(result["permutation"][model][name], result["base_sq"][model]))
                        fold_drop[name].append(fold_rmse_increase(result["drop_column"][model][name], result["base_sq"][model]))
                if np.isnan(base).any():
                    raise DiagnosticGateError("Every molecule must be predicted exactly once per repeat.")
                stored_rmse = float(np.sqrt(self.mol[protocol][f"sq:{model}"].mean()))
                check_reproduction(np.array([np.sqrt(base.mean(axis=0).mean())]), np.array([stored_rmse]), 1e-9, f"{model} {protocol} base RMSE")
                model_out = {}
                for kind, arrays, fold_values in (("permutation", perm, fold_perm), ("drop_column", drop, fold_drop)):
                    unit_out = {}
                    for name, _ in units:
                        pooled = pooled_importance(arrays[name], base)
                        averaged = bootstrap_difference(weights, codes, arrays[name].mean(axis=(0, 1)), base.mean(axis=0))
                        values = np.asarray(fold_values[name])
                        unit_out[name] = {
                            "pooled_importance": pooled["importance_mean"],
                            "pooled_importance_sd_over_permutations": pooled["importance_sd"] if kind == "permutation" else None,
                            "importance_from_permutation_averaged_errors": averaged["delta"],
                            "ci": [averaged["ci_lower"], averaged["ci_upper"]],
                            "fold_median": float(np.median(values)),
                            "fold_iqr": [float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))],
                            "fold_range": [float(values.min()), float(values.max())],
                            "n_folds": int(len(values)),
                            "fold_values": values.tolist(),
                        }
                    for family in ("feature:", "primary:", "sensitivity:"):
                        names = [n for n, _ in units if n.startswith(family)]
                        matrix = np.array([unit_out[n]["fold_values"] for n in names])
                        firsts = Counter(np.argmax(matrix, axis=0))
                        for i, n in enumerate(names):
                            unit_out[n]["fraction_folds_ranked_first_within_set"] = firsts.get(i, 0) / matrix.shape[1]
                    model_out[kind] = unit_out
                self.importance_arrays[(protocol, model)] = {"base": base, "perm": perm, "drop": drop}
                model_out["base_rmse"] = stored_rmse
                protocol_out[model] = model_out
            impurity = pd.DataFrame([r["impurity"]["random_forest_reference"] for r in results])
            gain = pd.DataFrame([r["impurity"]["xgboost_total_gain"] for r in results])
            share = gain.div(gain.sum(axis=1), axis=0)
            shap = np.full((n_repeats, len(rows), len(FEATURE_COLUMNS) + 1), np.nan)
            for result in results:
                shap[result["repeat"], position.loc[result["row_id"]].to_numpy()] = result["shap"]
            mean_shap = shap.mean(axis=0)
            self.shap_values = getattr(self, "shap_values", {})
            self.shap_values[protocol] = (rows, mean_shap)
            coefficients = pd.DataFrame([r["lasso_coefficients"] for r in results])
            median_sign = np.sign(coefficients.median())
            protocol_out["impurity_random_forest_reference"] = {f: {"mean": float(impurity[f].mean()), "sd": float(impurity[f].std(ddof=1)) if len(impurity) > 1 else 0.0} for f in FEATURE_COLUMNS}
            protocol_out["xgboost_total_gain"] = {f: {"mean_raw": float(gain[f].mean()), "mean_share": float(share[f].mean()), "sd_share": float(share[f].std(ddof=1)) if len(share) > 1 else 0.0} for f in FEATURE_COLUMNS}
            protocol_out["xgboost_mean_abs_shap"] = {f: float(np.abs(mean_shap[:, i]).mean()) for i, f in enumerate(FEATURE_COLUMNS)}
            protocol_out["lasso_standardized_coefficients"] = {
                f: {"median": float(coefficients[f].median()), "iqr": [float(coefficients[f].quantile(0.25)), float(coefficients[f].quantile(0.75))], "sign_consistency": float((np.sign(coefficients[f]) == median_sign[f]).mean()), "fraction_zero": float((coefficients[f] == 0).mean())}
                for f in FEATURE_COLUMNS
            }
            protocol_out["method_ranks"] = self.method_ranks(protocol_out)
            output[protocol] = protocol_out
        output["ale"] = self.ale_summary()
        return output

    def method_ranks(self, protocol_out: dict[str, Any]) -> dict[str, Any]:
        def ranks(values: dict[str, float]) -> dict[str, int]:
            ordered = sorted(values, key=lambda f: (-values[f], FEATURE_COLUMNS.index(f)))
            return {f: ordered.index(f) + 1 for f in FEATURE_COLUMNS}

        result = {}
        for model in IMPORTANCE_MODELS:
            methods = {
                "permutation": {f: protocol_out[model]["permutation"][f"feature:{f}"]["pooled_importance"] for f in FEATURE_COLUMNS},
                "drop_column": {f: protocol_out[model]["drop_column"][f"feature:{f}"]["pooled_importance"] for f in FEATURE_COLUMNS},
            }
            if model == "random_forest_reference":
                methods["impurity"] = {f: protocol_out["impurity_random_forest_reference"][f]["mean"] for f in FEATURE_COLUMNS}
            else:
                methods["total_gain"] = {f: protocol_out["xgboost_total_gain"][f]["mean_share"] for f in FEATURE_COLUMNS}
                methods["mean_abs_treeshap"] = protocol_out["xgboost_mean_abs_shap"]
            result[model] = {method: ranks(values) for method, values in methods.items()}
        return result

    def ale_summary(self) -> dict[str, Any]:
        bins = self.config["feature_importance"]["ale"]["bins"]
        summary: dict[str, Any] = {"grid_rule": "fold curves interpolated onto ale_edges of the full S0 descriptor values; no extrapolation beyond each fold's edges", "sparse_region_rule": "outside the 5th-95th percentile of S0 values (shaded in F10)"}
        self.ale_curves = {}
        for feature in FEATURE_COLUMNS:
            values = self.data[feature].to_numpy(dtype=float)
            grid = ale_edges(values, bins)
            feature_out = {"grid": grid.tolist(), "p05": float(np.quantile(values, 0.05)), "p95": float(np.quantile(values, 0.95))}
            for model in self.config["feature_importance"]["ale"]["models"]:
                cv_curves = np.array([interpolate_curve(grid, *r["ale"][(model, feature)][:2]) for r in self.fold_results if r["protocol"] == CV])
                holdout = [interpolate_curve(grid, *r["ale"][(model, feature)][:2]) for r in self.fold_results if r["protocol"] == HOLDOUT][0]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN grid points outside every fold's range
                    median = np.nanmedian(cv_curves, axis=0)
                    low, high = np.nanquantile(cv_curves, 0.1, axis=0), np.nanquantile(cv_curves, 0.9, axis=0)
                valid = ~np.isnan(median)
                direction = float(stats.spearmanr(grid[valid], median[valid]).statistic) if valid.sum() > 2 and np.nanstd(median) > 0 else 0.0
                feature_out[model] = {"median": median.tolist(), "p10": low.tolist(), "p90": high.tolist(), "benzene_holdout": holdout.tolist(), "direction_spearman": direction, "n_fold_curves": int(len(cv_curves))}
                self.ale_curves[(feature, model)] = (grid, median, low, high, holdout)
            summary[feature] = feature_out
        return summary

    def descriptor_reference(self) -> dict[str, Any]:
        reference = self.config["descriptor_reference"]
        ranks = self.tables["T3_feature_importance.json"][CV]["method_ranks"]
        output = {"note": reference["language"], "not_represented": reference["not_represented"], "descriptors": {}}
        for feature in FEATURE_COLUMNS:
            entry = dict(reference[feature])
            entry["observed_ale_direction_spearman_cv"] = {m: self.tables["T3_feature_importance.json"]["ale"][feature][m]["direction_spearman"] for m in self.config["feature_importance"]["ale"]["models"]}
            entry["permutation_rank_cv"] = {m: ranks[m]["permutation"][feature] for m in IMPORTANCE_MODELS}
            entry["lasso_median_standardized_coefficient_cv"] = self.tables["T3_feature_importance.json"][CV]["lasso_standardized_coefficients"][feature]["median"]
            output["descriptors"][feature] = entry
        return output

    def family_diagnostics(self) -> dict[str, Any]:
        families_cfg = self.cfg6d["families"]
        threshold = families_cfg["min_n_for_intervals"]
        models = sorted(self.oof["model"].unique(), key=lambda m: list(MODEL_LABEL).index(m))
        output = {"policy": self.config["families"]["small_family_policy"], "uncertainty_note": self.config["uncertainty"]["small_families"], "esol_equation_role": self.config["esol_equation_comparator"]["role"], "families": {}}
        flags = ["below_training_min_target", "below_knn_min_target", "logp_above_training_max", "any_descriptor_out_of_range"]
        for family in [*families_cfg["named"], *families_cfg["aggregate"]]:
            protocol = HOLDOUT if family == "c1ccccc1" else CV
            mol = self.mol[protocol]
            subset = mol[mol["family"] == family]
            n = int(len(subset))
            entry: dict[str, Any] = {"protocol": protocol, "n": n, "descriptive_only": n < threshold, "models": {}}
            if n >= threshold:
                clusters = subset.index.to_numpy() if family in families_cfg["named"] else subset["murcko"].to_numpy()
                weights, codes = cluster_bootstrap_weights(clusters, self.replicates, self.seed)
            for model in models:
                model_entry = {"rmse": float(np.sqrt(subset[f"sq:{model}"].mean())), "mean_signed_error": float(subset[f"res:{model}"].mean())}
                if n >= threshold:
                    model_entry["rmse_ci"] = self.ci(bootstrap_rmse(weights, codes, subset[f"sq:{model}"].to_numpy()))
                    model_entry["mean_signed_error_ci"] = self.ci(bootstrap_mean(weights, codes, subset[f"res:{model}"].to_numpy()))
                entry["models"][model] = model_entry
            esol_residual = subset["esol_equation"] - subset["observed"]
            entry["esol_equation_in_sample"] = {"rmse": float(np.sqrt((esol_residual**2).mean())), "mean_signed_error": float(esol_residual.mean())}
            biases = [entry["models"][m]["mean_signed_error"] for m in models if m != "baseline"] + [entry["esol_equation_in_sample"]["mean_signed_error"]]
            entry["shared_bias_sign"] = "all positive" if all(b > 0 for b in biases) else "all negative" if all(b < 0 for b in biases) else "mixed"
            neighbours = self.neighbours[protocol].loc[subset.index]
            entry["extrapolation_fraction_of_predictions"] = {flag: float(neighbours[flag].mean()) for flag in flags}
            entry["target"] = {"mean": float(subset["observed"].mean()), "sd": float(subset["observed"].std(ddof=1)) if n > 1 else 0.0, "min": float(subset["observed"].min()), "max": float(subset["observed"].max())}
            output["families"][family] = entry
        output["esol_equation_overall_in_sample"] = {
            protocol: {"rmse": float(np.sqrt(((self.mol[protocol]["esol_equation"] - self.mol[protocol]["observed"]) ** 2).mean())), "mean_signed_error": float((self.mol[protocol]["esol_equation"] - self.mol[protocol]["observed"]).mean()), "n": int(len(self.mol[protocol]))}
            for protocol in PROTOCOLS
        }
        return output

    def molecule_rows(self, protocol: str, rows: list[int]) -> list[dict[str, Any]]:
        mol, neighbours = self.mol[protocol], self.neighbours[protocol]
        records = []
        for row in rows:
            nearest = int(neighbours.loc[row, "nearest_row_id"])
            record = {
                "protocol": protocol,
                "row_id": int(row),
                "compound_id": self.data.loc[row, "Compound ID"],
                "smiles": self.data.loc[row, "smiles"].strip(),
                "family": mol.loc[row, "family"],
                "observed": float(mol.loc[row, "observed"]),
                "predictions": {m: float(mol.loc[row, f"pred:{m}"]) for m in ("random_forest_reference", "random_forest_tuned", "xgboost", "lasso")},
                "esol_equation": float(mol.loc[row, "esol_equation"]),
                **{f: float(self.data.loc[row, f]) for f in FEATURE_COLUMNS},
                "nearest_training_tanimoto": float(neighbours.loc[row, "max_tanimoto"]),
                "nearest_training_compound": self.data.loc[nearest, "Compound ID"],
                "nearest_training_smiles": self.data.loc[nearest, "smiles"].strip(),
                "nearest_training_observed": float(self.data.loc[nearest, TARGET_COLUMN]),
                "descriptor_distance": float(neighbours.loc[row, "descriptor_distance"]),
                "fraction_repeats_target_below_training_min": float(neighbours.loc[row, "below_training_min_target"]),
                "fraction_repeats_target_below_knn_min": float(neighbours.loc[row, "below_knn_min_target"]),
                "fraction_repeats_logp_above_training_max": float(neighbours.loc[row, "logp_above_training_max"]),
                "fraction_repeats_any_descriptor_out_of_range": float(neighbours.loc[row, "any_descriptor_out_of_range"]),
            }
            records.append(record)
        return records

    def focus_families(self) -> dict[str, Any]:
        output = {"policy": self.config["families"]["hypothesis_policy"], "families": {}}
        aromatic_n = Chem.MolFromSmarts("n")
        for name, spec in self.config["families"]["focus"].items():
            subset = self.mol[CV][self.mol[CV]["family"] == spec["murcko"]]
            rows = subset.index.tolist()
            neighbours = self.neighbours[CV].loc[rows]
            diagnostics: dict[str, Any] = {"n": len(rows), "descriptive_only": len(rows) < 20, "phase6d_observation": spec["phase6d_observation"]}
            diagnostics["mean_signed_error"] = {m: float(subset[f"res:{m}"].mean()) for m in ("random_forest_reference", "random_forest_tuned", "xgboost", "lasso", "linear_regression_reference", "baseline")}
            diagnostics["esol_equation_mean_signed_error_in_sample"] = float((subset["esol_equation"] - subset["observed"]).mean())
            diagnostics["spearman_residual_vs_logp"] = {m: float(stats.spearmanr(self.data.loc[rows, "LogP"], subset[f"res:{m}"]).statistic) for m in FIGURE_MODELS}
            diagnostics["fraction_any_descriptor_out_of_range"] = float((neighbours["any_descriptor_out_of_range"] > 0).mean())
            if name == "pyridine":
                nearest = neighbours["nearest_row_id"].astype(int)
                diagnostics["fraction_nearest_neighbours_without_aromatic_nitrogen"] = float(np.mean([not Chem.MolFromSmiles(self.data.loc[r, "smiles"].strip()).HasSubstructMatch(aromatic_n) for r in nearest]))
                diagnostics["spearman_residual_vs_hba"] = {m: float(stats.spearmanr(self.data.loc[rows, "HBA"], subset[f"res:{m}"]).statistic) for m in FIGURE_MODELS}
                diagnostics["spearman_residual_vs_tpsa"] = {m: float(stats.spearmanr(self.data.loc[rows, "TPSA"], subset[f"res:{m}"]).statistic) for m in FIGURE_MODELS}
            if name == "biphenyl":
                extrapolating = (neighbours[["below_training_min_target", "below_knn_min_target", "logp_above_training_max"]] > 0).any(axis=1)
                diagnostics["fraction_with_any_extrapolation_indicator"] = float(extrapolating.mean())
                diagnostics["indicator_fractions"] = {flag: float((neighbours[flag] > 0).mean()) for flag in ["below_training_min_target", "below_knn_min_target", "logp_above_training_max"]}
                for label, mask in (("extrapolating", extrapolating), ("non_extrapolating", ~extrapolating)):
                    diagnostics[f"rmse_{label}"] = {"n": int(mask.sum()), **{m: float(np.sqrt(subset.loc[mask.to_numpy(), f"sq:{m}"].mean())) if mask.any() else None for m in FIGURE_MODELS}}
            output["families"][name] = {"murcko": spec["murcko"], "diagnostics": diagnostics, "molecules": self.molecule_rows(CV, rows)}
        return output

    def extreme_errors(self) -> dict[str, Any]:
        cfg = self.config["extreme_errors"]
        residuals = pd.concat([self.mol[p][[f"res:{m}" for m in cfg["threshold_models"]]].rename(columns=lambda c: c[4:]).assign(protocol=p) for p in PROTOCOLS])
        selected = select_extreme_errors(residuals, cfg["reference_model"], cfg["top_k_by_abs_residual"], cfg["also_include_abs_residual_above"], cfg["threshold_models"])
        identity_counts = self.data["identity"].value_counts()
        descriptor_key = self.data[FEATURE_COLUMNS].round(6).astype(str).agg("|".join, axis=1)
        descriptor_counts = descriptor_key.value_counts()
        records = []
        for row in selected:
            record = self.molecule_rows(residuals.loc[row, "protocol"], [row])[0]
            record["residual_random_forest_reference"] = float(residuals.loc[row, "random_forest_reference"])
            record["identity_duplicate_group"] = bool(identity_counts[self.data.loc[row, "identity"]] > 1)
            record["identical_descriptor_group_size"] = int(descriptor_counts[descriptor_key[row]])
            records.append(record)
        return {"rule": cfg, "n_selected": len(records), "policy": cfg["policy"], "molecules": records}

    def similarity_distance(self) -> dict[str, Any]:
        bins = self.config["acyclic"]["similarity_bins"]
        mol = self.mol[CV].join(self.neighbours[CV])
        mol["similarity_bin"] = mol["max_tanimoto"].map(lambda s: similarity_bin(s, bins))
        order = [f"[{lo}, {hi})" for lo, hi in [b for b in bins if isinstance(b, list)]] + ["exactly 1.0"]
        acyclic_rows = []
        for ring_label, ring_mask in (("acyclic", mol["acyclic"]), ("cyclic", ~mol["acyclic"])):
            for label in order:
                subset = mol[ring_mask & (mol["similarity_bin"] == label)]
                entry = {"ring_status": ring_label, "similarity_bin": label, "n": int(len(subset))}
                if len(subset) >= 2:
                    weights, codes = cluster_bootstrap_weights(subset["murcko"].to_numpy(), self.replicates, self.seed)
                for model in FIGURE_MODELS:
                    entry[model] = {"rmse": float(np.sqrt(subset[f"sq:{model}"].mean())) if len(subset) else None, "ci": self.ci(bootstrap_rmse(weights, codes, subset[f"sq:{model}"].to_numpy())) if len(subset) >= 2 else None}
                acyclic_rows.append(entry)
        collisions = mol[mol["acyclic"] & (mol["max_tanimoto"] == 1.0)]
        collision_rows = []
        homologues = 0
        for row in mol[mol["acyclic"]].index:
            nearest = int(mol.loc[row, "nearest_row_id"])
            if is_ch2_homologue(self.data.loc[row, "smiles"], self.data.loc[nearest, "smiles"]):
                homologues += 1
        for row in collisions.index:
            nearest = int(mol.loc[row, "nearest_row_id"])
            collision_rows.append({"compound_id": self.data.loc[row, "Compound ID"], "smiles": self.data.loc[row, "smiles"].strip(), "observed": float(mol.loc[row, "observed"]), "nearest_compound": self.data.loc[nearest, "Compound ID"], "nearest_smiles": self.data.loc[nearest, "smiles"].strip(), "nearest_observed": float(self.data.loc[nearest, TARGET_COLUMN]), "abs_target_difference": float(abs(mol.loc[row, "observed"] - self.data.loc[nearest, TARGET_COLUMN]))})
        distance: dict[str, Any] = {}
        fractions = self.config["distance_reliability"]["selective_prediction"]["coverage_fractions"]
        for protocol in PROTOCOLS:
            table = self.mol[protocol].join(self.neighbours[protocol])
            table["quintile"] = pd.qcut(table["descriptor_distance"], 5, labels=False)
            quintiles = []
            for q in range(5):
                subset = table[table["quintile"] == q]
                weights, codes = self.weights(protocol, subset.index.to_numpy())
                entry = {"quintile": q + 1, "distance_range": [float(subset["descriptor_distance"].min()), float(subset["descriptor_distance"].max())], "n": int(len(subset))}
                for model in FIGURE_MODELS:
                    entry[model] = {"rmse": float(np.sqrt(subset[f"sq:{model}"].mean())), "rmse_ci": self.ci(bootstrap_rmse(weights, codes, subset[f"sq:{model}"].to_numpy())), "mean_signed_error": float(subset[f"res:{model}"].mean())}
                quintiles.append(entry)
            ordered = table.assign(row=table.index).sort_values(["descriptor_distance", "row"])
            selective = {model: [{"fraction": f, "n": int(np.ceil(f * len(ordered))), "rmse": float(np.sqrt(ordered[f"sq:{model}"].iloc[: int(np.ceil(f * len(ordered)))].mean()))} for f in fractions] for model in FIGURE_MODELS}
            preds = self.oof[(self.oof["protocol"] == protocol) & self.oof["model"].isin(FIGURE_MODELS)].merge(self.neighbour_rows[self.neighbour_rows["protocol"] == protocol][["row_id", "repeat", "any_descriptor_out_of_range"]], on=["row_id", "repeat"])
            range_flag = {}
            for model in FIGURE_MODELS:
                frame = preds[preds["model"] == model]
                sq = (frame["y_pred"] - frame["y_true"]) ** 2
                flagged = frame["any_descriptor_out_of_range"].astype(bool)
                range_flag[model] = {"flagged": {"n_predictions": int(flagged.sum()), "n_molecules": int(frame.loc[flagged, "row_id"].nunique()), "rmse": float(np.sqrt(sq[flagged].mean())) if flagged.any() else None}, "unflagged": {"n_predictions": int((~flagged).sum()), "rmse": float(np.sqrt(sq[~flagged].mean()))}}
            spearman = {model: float(stats.spearmanr(table["descriptor_distance"], table[f"abs:{model}"]).statistic) for model in FIGURE_MODELS}
            distance[protocol] = {"quintiles": quintiles, "selective_prediction": selective, "range_flag_prediction_level": range_flag, "spearman_distance_vs_abs_error": spearman}
        self.similarity_table = mol
        return {
            "acyclic_vs_cyclic_similarity_bins": acyclic_rows,
            "acyclic_fingerprint_collisions": collision_rows,
            "acyclic_n": int(mol["acyclic"].sum()),
            "acyclic_nearest_neighbour_ch2_homologues": homologues,
            "similarity_summary": {label: {"mean": float(mol.loc[mask, "max_tanimoto"].mean()), "median": float(mol.loc[mask, "max_tanimoto"].median()), "fraction_exactly_1": float((mol.loc[mask, "max_tanimoto"] == 1.0).mean())} for label, mask in (("acyclic", mol["acyclic"]), ("cyclic", ~mol["acyclic"]))},
            "distance_reliability": distance,
            "phase6d_neighbour_statistics_max_abs_difference": self.checks["neighbour_statistics_max_abs_difference"],
        }

    def representation_floor(self) -> dict[str, Any]:
        spec = self.config["representation_floor"]
        floor = error_floor(self.dataset)
        if abs(floor["floor_rmse"] - spec["expected"]["floor_rmse"]) > spec["expected"]["tolerance"] or floor["groups_sharing_descriptor_vector"] != spec["expected"]["groups"] or floor["rows_sharing_descriptor_vector"] != spec["expected"]["rows"]:
            raise DiagnosticGateError(f"Representation floor {floor} differs from the configured expectation.")
        key = self.dataset[FEATURE_COLUMNS].round(6).astype(str).agg("|".join, axis=1)
        groups = []
        for _, group in self.dataset.groupby(key):
            if len(group) > 1:
                groups.append({"n": int(len(group)), "target_sd": float(group[TARGET_COLUMN].std(ddof=1)), "target_range": float(np.ptp(group[TARGET_COLUMN])), "compounds": group["Compound ID"].tolist(), "smiles": [s.strip() for s in group["smiles"]], "targets": group[TARGET_COLUMN].tolist()})
        groups.sort(key=lambda g: (-g["target_sd"], g["compounds"][0]))
        self.floor_groups = groups
        return {"definition": spec["definition"], "scope_statement": spec["scope_statement"], **floor, "measurement_noise": spec["measurement_noise"], "group_target_sd": [g["target_sd"] for g in groups], "top_groups": groups[: int(re.search(r"top (\d+)", spec["listing"]).group(1))]}

    def intervals(self) -> dict[str, Any]:
        cfg = self.config["interval_diagnostics"]
        output: dict[str, Any] = {"label": cfg["label"], "assumptions": cfg["assumptions_to_state"], "levels": cfg["levels"]}
        tree_frames = []
        for result in self.fold_results:
            y = self.data.loc[result["row_id"], TARGET_COLUMN].to_numpy()
            tree_frames.append(pd.DataFrame({"protocol": result["protocol"], "repeat": result["repeat"], "row_id": result["row_id"], "sd": result["tree_sd"], "abs_error": np.abs(result["predictions"]["random_forest_reference"] - y)}))
        tree = pd.concat(tree_frames, ignore_index=True)
        coverage_rows = []
        for protocol in PROTOCOLS:
            frame = tree[tree["protocol"] == protocol]
            mol = self.mol[protocol].join(self.neighbours[protocol])
            mol["quintile"] = pd.qcut(mol["descriptor_distance"], 5, labels=False) + 1
            per_molecule = {}
            for level in cfg["levels"]:
                z = stats.norm.ppf(0.5 + level / 2)
                covered = (frame["abs_error"] <= z * frame["sd"]).groupby(frame["row_id"]).mean()
                per_molecule[("tree_spread", "random_forest_reference", level)] = covered
            if protocol == CV:
                for model in ("random_forest_reference", "xgboost"):
                    residuals = self.oof[(self.oof["protocol"] == CV) & (self.oof["model"] == model)].assign(residual=lambda f: f["y_pred"] - f["y_true"])
                    for level in cfg["levels"]:
                        widths = cross_fold_half_widths(residuals, level)
                        half = residuals.apply(lambda r: widths[(r["repeat"], r["fold"])], axis=1)
                        per_molecule[("cross_fold_residual", model, level)] = (residuals["residual"].abs() <= half).groupby(residuals["row_id"]).mean()
            else:
                for model in ("random_forest_reference", "xgboost"):
                    calibration = self.mol[CV][f"res:{model}"].abs().to_numpy()
                    for level in cfg["levels"]:
                        width = float(np.quantile(calibration, level, method="higher"))
                        per_molecule[("shift_check_cross_fold", model, level)] = (self.mol[HOLDOUT][f"res:{model}"].abs() <= width).astype(float)
            for (method, model, level), covered in per_molecule.items():
                covered = covered.loc[mol.index]
                base = {"protocol": protocol, "method": method, "model": model, "level": level}
                coverage_rows.append({**base, "group": "overall", "n": int(len(covered)), "observed_coverage": float(covered.mean())})
                for family, values in covered.groupby(mol["family"]):
                    coverage_rows.append({**base, "group": f"family:{family}", "n": int(len(values)), "observed_coverage": float(values.mean()), "descriptive_only": len(values) < 20})
                for quintile, values in covered.groupby(mol["quintile"]):
                    coverage_rows.append({**base, "group": f"distance_quintile:{int(quintile)}", "n": int(len(values)), "observed_coverage": float(values.mean())})
            sd_mean = frame.groupby("row_id")["sd"].mean().loc[mol.index]
            output[f"{protocol}_spearman_tree_sd_vs_abs_error"] = float(stats.spearmanr(sd_mean, mol["abs:random_forest_reference"]).statistic)
        output["coverage"] = coverage_rows
        self.coverage = pd.DataFrame(coverage_rows)
        return output

    def production_model(self) -> dict[str, Any]:
        cfg = self.config["production_model"]
        model = joblib.load(PROJECT_ROOT / cfg["path"])
        # In-memory only: the saved model predicts with n_jobs=-1, whose threaded summation order
        # changes the last floating-point bit between runs. The file itself is never written.
        model.set_params(n_jobs=1)
        descriptors = pd.read_csv(PROJECT_ROOT / self.config["inputs"]["descriptor_data"])
        train, test = split_dataset(descriptors, test_size=0.20, random_state=42)
        X_train, X_test = train[FEATURE_COLUMNS], test[FEATURE_COLUMNS]
        y_train, y_test = train[TARGET_COLUMN].to_numpy(), test[TARGET_COLUMN].to_numpy()
        predictions = model.predict(X_test)
        rmse = float(np.sqrt(np.mean((predictions - y_test) ** 2)))
        check_reproduction(np.array([rmse]), np.array([cfg["reproduction_check"]["expected_test_rmse"]]), cfg["reproduction_check"]["tolerance"], "production test RMSE")
        estimator = self.specs["random_forest_reference"][0]
        refit = clone(estimator).fit(X_train, y_train)
        check_reproduction(refit.predict(X_test), predictions, 1e-9, "production configuration refit")
        units = importance_units(self.config)
        repeats = self.config["feature_importance"]["permutation"]["repeats"]
        base = (predictions - y_test) ** 2
        perm = permuted_squared_errors(model, X_test, y_test, units, repeats, (self.config["feature_importance"]["permutation"]["seed"], PROTOCOL_INDEX["production"], 0, 0))
        drop = drop_column_squared_errors(estimator, {}, X_train, y_train, X_test, y_test, units)
        weights, codes = cluster_bootstrap_weights(np.arange(len(y_test)), self.replicates, self.seed)
        result: dict[str, Any] = {"label": cfg["label"], "test_rmse": rmse, "n_test": int(len(y_test)), "permutation": {}, "drop_column": {}}
        for name, _ in units:
            pooled = pooled_importance(perm[name][:, None, :], base[None, :])
            averaged = bootstrap_difference(weights, codes, perm[name].mean(axis=0), base)
            result["permutation"][name] = {"pooled_importance": pooled["importance_mean"], "sd_over_permutations": pooled["importance_sd"], "ci_molecule_bootstrap": [averaged["ci_lower"], averaged["ci_upper"]]}
            dropped = bootstrap_difference(weights, codes, drop[name], base)
            result["drop_column"][name] = {"pooled_importance": dropped["delta"], "ci_molecule_bootstrap": [dropped["ci_lower"], dropped["ci_upper"]]}
        result["impurity"] = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
        sd = tree_spread(model, X_test)
        result["tree_spread_observed_coverage"] = {str(level): float(np.mean(np.abs(predictions - y_test) <= stats.norm.ppf(0.5 + level / 2) * sd)) for level in self.config["interval_diagnostics"]["levels"]}
        result["spearman_tree_sd_vs_abs_error"] = float(stats.spearmanr(sd, np.abs(predictions - y_test)).statistic)
        return result


# Figures --------------------------------------------------------------------------------------------


def apply_style(style: dict[str, Any]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": style["font_size"],
            "axes.facecolor": style["surface"],
            "figure.facecolor": style["surface"],
            "savefig.facecolor": style["surface"],
            "axes.edgecolor": style["axis"],
            "axes.labelcolor": style["primary_ink"],
            "text.color": style["primary_ink"],
            "xtick.color": style["secondary_ink"],
            "ytick.color": style["secondary_ink"],
            "axes.grid": True,
            "grid.color": style["gridline"],
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": style["line_width"],
            "legend.frameon": False,
            "svg.hashsalt": "phase6e",
        }
    )


def save_figure(fig: plt.Figure, path: Path, dpi: int, footnote: str) -> None:
    """Save with a wrapped footnote in a reserved bottom band so it never overlaps the axes."""
    width, height = fig.get_size_inches()
    lines = textwrap.wrap(footnote, width=int(width * 17))
    band = (len(lines) * 0.14 + 0.08) / height
    if fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(rect=(0, band, 1, 1 - band))
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=7, color="#52514e", ha="left", va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, metadata={"Software": None})
    plt.close(fig)


class FigureBuilder:
    def __init__(self, diagnostics: Diagnostics, directory: Path) -> None:
        self.d, self.directory = diagnostics, directory
        self.cfg = diagnostics.config["figures"]
        self.style = self.cfg["style"]
        self.colors = self.style["model_colors"]
        self.items = {item["id"]: item for item in self.cfg["items"]}
        apply_style(self.style)

    def new(self, fid: str, nrows: int = 1, ncols: int = 1, **kwargs: Any) -> tuple[plt.Figure, Any]:
        width, height = self.items[fid]["size_inches"]
        return plt.subplots(nrows, ncols, figsize=(width, height), layout="constrained", **kwargs)

    def save(self, fig: plt.Figure, fid: str, footnote: str) -> None:
        directory = self.directory if self.directory.is_absolute() else PROJECT_ROOT / self.directory
        save_figure(fig, directory / Path(self.items[fid]["file"]).name, self.cfg["format"]["dpi"], footnote)

    def build_all(self) -> list[str]:
        for builder in (self.f1, self.f2, self.f3, self.f4, self.f5, self.f6, self.f7, self.f8, self.f9, self.f10, self.f11, self.f12, self.f13, self.f14, self.f15, self.f16, self.f17, self.f18, self.f19):
            builder()
        return [Path(item["file"]).name for item in self.cfg["items"]]

    def _model_legend(self, ax: Any, models: tuple[str, ...] = FIGURE_MODELS, **kwargs: Any) -> None:
        handles = [plt.Line2D([], [], color=self.colors[m], marker="o", linestyle="-", label=MODEL_LABEL[m]) for m in models]
        ax.legend(handles=handles, **kwargs)

    def f1(self) -> None:
        fig, axes = self.new("F1", 1, 3, sharex=True, sharey=True)
        lims = [-12.5, 2.5]
        for ax, model in zip(axes, FIGURE_MODELS):
            text = []
            for protocol, marker in self.style["protocol_markers"].items():
                mol = self.d.mol[protocol]
                ax.scatter(mol[f"pred:{model}"], mol["observed"], s=14, marker=marker, color=self.colors[model] if protocol == CV else "none", edgecolors=self.colors[model], linewidths=0.8, alpha=0.6 if protocol == CV else 0.9, label=f"{PROTOCOL_LABEL[protocol]} (n={len(mol)})")
                text.append(f"{PROTOCOL_LABEL[protocol]}: RMSE {np.sqrt(mol[f'sq:{model}'].mean()):.3f}")
            ax.plot(lims, lims, color="#0b0b0b", linewidth=1)
            for offset in (-1, 1):
                ax.plot(lims, [lims[0] + offset, lims[1] + offset], color="#898781", linewidth=0.8, linestyle="--")
            ax.set(xlim=lims, ylim=lims, xlabel="Predicted LogS (out-of-fold)", title=MODEL_LABEL[model])
            ax.text(0.03, 0.97, "\n".join(text), transform=ax.transAxes, va="top", fontsize=7)
        axes[0].set_ylabel("Observed LogS")
        axes[0].legend(loc="lower right", fontsize=7)
        self.save(fig, "F1", "Out-of-fold predictions; non-benzene CV uses the mean over 5 repeats (circles), benzene holdout single predictions (open triangles). Protocol statistics reported separately. Dashed: ±1 LogS.")

    def f2(self) -> None:
        fig, axes = self.new("F2", 1, 2)
        table = self.d.tables["T1_calibration.json"]
        for ax, protocol in zip(axes, PROTOCOLS):
            lims = [np.inf, -np.inf]
            for model in FIGURE_MODELS:
                entry = table[protocol]["models"][model]
                binned = pd.DataFrame(entry["binned_calibration"])
                ax.plot(binned["mean_predicted"], binned["mean_observed"], marker=self.style["protocol_markers"][protocol], color=self.colors[model], label=f"{MODEL_LABEL[model]}: slope {entry['calibration_slope']:.2f} [{entry['calibration_slope_ci'][0]:.2f}, {entry['calibration_slope_ci'][1]:.2f}]")
                values = binned[["mean_predicted", "mean_observed"]].to_numpy()
                lims = [min(lims[0], values.min()), max(lims[1], values.max())]
            ax.plot(lims, lims, color="#0b0b0b", linewidth=1)
            ax.set(title=f"{PROTOCOL_LABEL[protocol]} (n={table[protocol]['models'][FIGURE_MODELS[0]]['n']})", xlabel="Mean predicted LogS (bin)", ylabel="Mean observed LogS (bin)")
            ax.legend(fontsize=7, loc="upper left")
        self.save(fig, "F2", "Binned calibration (deciles CV, quintiles holdout). Slope of observed on predicted with 95% bootstrap interval (scaffold-cluster for CV; molecule, descriptive, for benzene).")

    def f3(self) -> None:
        fig, axes = self.new("F3", 2, 2)
        for column, protocol in enumerate(PROTOCOLS):
            mol = self.d.mol[protocol]
            for model in FIGURE_MODELS:
                residual = mol[f"pred:{model}"] - mol["observed"]
                axes[0, column].hist(residual, bins=np.arange(-5, 5.25, 0.25), histtype="step", color=self.colors[model], linewidth=1.5, label=MODEL_LABEL[model])
                (theoretical, ordered), _ = stats.probplot(residual, dist="norm")
                axes[1, column].plot(theoretical, ordered, marker="o", markersize=2.5, linestyle="none", color=self.colors[model], label=MODEL_LABEL[model])
            axes[0, column].set(title=f"{PROTOCOL_LABEL[protocol]} (n={len(mol)})", xlabel="Residual (predicted − observed)", ylabel="Molecules")
            axes[1, column].axline((0, 0), slope=float(np.std(mol[f"pred:random_forest_reference"] - mol["observed"], ddof=1)), color="#898781", linestyle="--", linewidth=1)
            axes[1, column].set(xlabel="Normal quantile", ylabel="Residual quantile")
            axes[0, column].legend(fontsize=7)
        self.save(fig, "F3", "Residual distributions of out-of-fold mean predictions; Q–Q dashed reference uses the RF-reference residual sd. Protocols shown separately.")

    def f4(self) -> None:
        fig, axes = self.new("F4", 2, 2)
        table = self.d.tables["T1_calibration.json"]
        for row, protocol in enumerate(PROTOCOLS):
            mol = self.d.mol[protocol]
            for column, by in enumerate(("predicted", "observed")):
                ax = axes[row, column]
                for model in FIGURE_MODELS:
                    residual = mol[f"pred:{model}"] - mol["observed"]
                    x = mol[f"pred:{model}"] if by == "predicted" else mol["observed"]
                    ax.scatter(x, residual, s=6, color=self.colors[model], alpha=0.25, linewidths=0)
                    bins = pd.DataFrame(table[protocol]["models"][model]["bias_bins"][by])
                    ax.errorbar(bins["center"], bins["mean_residual"], yerr=[bins["mean_residual"] - bins["ci"].str[0], bins["ci"].str[1] - bins["mean_residual"]], color=self.colors[model], marker="o", markersize=4, capsize=2, linewidth=1.5, label=MODEL_LABEL[model])
                ax.axhline(0, color="#0b0b0b", linewidth=1)
                ax.set(xlabel=f"{by.capitalize()} LogS", ylabel="Residual", title=f"{PROTOCOL_LABEL[protocol]}: residual vs {by}" + (" (calibration)" if by == "predicted" else " (shrinkage)"))
        axes[0, 0].legend(fontsize=7)
        self.save(fig, "F4", "Binned mean residual (decile bins) with 95% bootstrap intervals; points are individual molecules. A trend against observed values reflects shrinkage toward the mean, not miscalibration.")

    def f5(self) -> None:
        fig, axes = self.new("F5", 2, 3)
        table = self.d.tables["T1_calibration.json"][CV]["models"]
        for ax, feature in zip(axes.flat, FEATURE_COLUMNS):
            for model in FIGURE_MODELS:
                bins = pd.DataFrame(table[model]["bias_bins"][feature])
                ax.errorbar(bins["center"], bins["mean_residual"], yerr=[bins["mean_residual"] - bins["ci"].str[0], bins["ci"].str[1] - bins["mean_residual"]], color=self.colors[model], marker="o", markersize=4, capsize=2, linewidth=1.5, label=MODEL_LABEL[model])
            ax.axhline(0, color="#0b0b0b", linewidth=1)
            ax.set(title=feature, xlabel=f"{feature} (bin median)", ylabel="Mean residual")
        axes.flat[0].legend(fontsize=7)
        self.save(fig, "F5", f"Non-benzene scaffold CV (n={len(self.d.mol[CV])}); mean residual by descriptor decile (duplicate edges dropped for integer descriptors) with scaffold-cluster 95% intervals.")

    def f6(self) -> None:
        fig, axes = self.new("F6", 1, 2)
        div = self.style["diverging"]
        cmap = LinearSegmentedColormap.from_list("div", [div["negative"], div["midpoint"], div["positive"]])
        for ax, method in zip(axes, ("pearson", "spearman")):
            corr = self.d.data[FEATURE_COLUMNS].corr(method=method)
            image = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
            ax.set_xticks(range(6), FEATURE_COLUMNS, rotation=45)
            ax.set_yticks(range(6), FEATURE_COLUMNS)
            ax.grid(False)
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
            ax.set_title(method.capitalize())
        fig.colorbar(image, ax=axes, shrink=0.8, label="Correlation")
        self.save(fig, "F6", "Descriptor correlations over all 1,128 ESOL molecules (S0).")

    def _importance_plot(self, fid: str, kind: str, title: str) -> None:
        fig, axes = self.new(fid, 1, 2, sharey=True)
        table = self.d.tables["T3_feature_importance.json"]
        units = table["units"]
        y = np.arange(len(units))
        for ax, protocol in zip(axes, PROTOCOLS):
            for offset, model in zip((-0.15, 0.15), IMPORTANCE_MODELS):
                entries = [table[protocol][model][kind][u] for u in units]
                values = np.array([e["pooled_importance"] for e in entries])
                ci = np.array([e["ci"] for e in entries])
                if protocol == CV:
                    for i, e in enumerate(entries):
                        ax.scatter(e["fold_values"], np.full(len(e["fold_values"]), i + offset), s=4, color="#898781", alpha=0.5, linewidths=0)
                ax.errorbar(values, y + offset, xerr=[values - ci[:, 0], ci[:, 1] - values], fmt="o", color=self.colors[model], markersize=5, capsize=2, label=MODEL_LABEL[model])
            ax.axvline(0, color="#0b0b0b", linewidth=1)
            ax.set(title=f"{PROTOCOL_LABEL[protocol]} (n={len(self.d.mol[protocol])})", xlabel=f"{title}: pooled RMSE increase (LogS)")
        axes[0].set_yticks(y, [u.replace("feature:", "").replace("primary:", "group: ").replace("sensitivity:", "sensitivity: ") for u in units])
        axes[0].invert_yaxis()
        axes[0].legend(fontsize=7, loc="lower right")
        self.save(fig, fid, "Pooled from prediction-level squared errors; 95% interval: scaffold-cluster (CV) or molecule (benzene, descriptive) bootstrap; gray dots: per-fold values. Correlated descriptors share credit; not causal.")

    def f7(self) -> None:
        self._importance_plot("F7", "permutation", "Permutation")

    def f8(self) -> None:
        self._importance_plot("F8", "drop_column", "Drop-column")

    def f9(self) -> None:
        fig, axes = self.new("F9", 1, 2, sharey=True)
        ranks = self.d.tables["T3_feature_importance.json"][CV]["method_ranks"]
        markers = {"permutation": "o", "drop_column": "s", "impurity": "^", "total_gain": "^", "mean_abs_treeshap": "D"}
        for ax, model in zip(axes, IMPORTANCE_MODELS):
            for offset, (method, values) in zip(np.linspace(-0.2, 0.2, len(ranks[model])), ranks[model].items()):
                ax.scatter([values[f] for f in FEATURE_COLUMNS], np.arange(6) + offset, marker=markers[method], s=40, color=self.colors[model], edgecolors="#0b0b0b", linewidths=0.4, label=method.replace("_", " "))
            ax.set(title=MODEL_LABEL[model], xlabel="Rank (1 = most important)", xticks=range(1, 7))
            ax.legend(fontsize=7, loc="lower right")
        axes[0].set_yticks(range(6), FEATURE_COLUMNS)
        axes[0].invert_yaxis()
        self.save(fig, "F9", f"Non-benzene scaffold CV (n={len(self.d.mol[CV])}). Impurity/gain are training-data measures (continuity only); permutation, drop-column and TreeSHAP use out-of-fold test molecules.")

    def f10(self) -> None:
        fig, axes = self.new("F10", 2, 3)
        for ax, feature in zip(axes.flat, FEATURE_COLUMNS):
            values = self.d.data[feature].to_numpy()
            p05, p95 = np.quantile(values, [0.05, 0.95])
            for model in self.d.config["feature_importance"]["ale"]["models"]:
                grid, median, low, high, holdout = self.d.ale_curves[(feature, model)]
                ax.fill_between(grid, low, high, color=self.colors[model], alpha=0.18, linewidth=0)
                ax.plot(grid, median, color=self.colors[model], label=MODEL_LABEL[model])
                ax.plot(grid, holdout, color=self.colors[model], linestyle=":", linewidth=1.2)
            ax.axvspan(values.min(), p05, color="#898781", alpha=0.12, linewidth=0)
            ax.axvspan(p95, values.max(), color="#898781", alpha=0.12, linewidth=0)
            ax.plot(values, np.full(len(values), ax.get_ylim()[0]), "|", color="#52514e", alpha=0.25, markersize=6)
            ax.set(title=feature, xlabel=feature, ylabel="ALE (LogS)")
        axes.flat[0].legend(fontsize=7)
        self.save(fig, "F10", "ALE: median (line) and 10–90% band across 25 non-benzene CV fold models; dotted: benzene-holdout model. Gray shading: sparse regions outside the 5th–95th percentile; rug: data. Model behaviour, not chemistry.")

    def f11(self) -> None:
        fig, axes = self.new("F11", 1, 2, sharey=True)
        families = self.d.tables["T5_family_diagnostics.json"]["families"]
        names = list(families)
        y = np.arange(len(names))
        for offset, model in zip((-0.2, 0, 0.2), FIGURE_MODELS):
            for ax, metric in zip(axes, ("rmse", "mean_signed_error")):
                values = np.array([families[n]["models"][model][metric] for n in names])
                ci = [families[n]["models"][model].get(f"{metric}_ci" if metric != "rmse" else "rmse_ci") for n in names]
                has = np.array([c is not None for c in ci])
                ax.scatter(values[~has], y[~has] + offset, marker="o", facecolors="none", edgecolors=self.colors[model], s=24)
                low = np.array([c[0] if c else v for c, v in zip(ci, values)])
                high = np.array([c[1] if c else v for c, v in zip(ci, values)])
                ax.errorbar(values[has], y[has] + offset, xerr=[values[has] - low[has], high[has] - values[has]], fmt="o", color=self.colors[model], markersize=4, capsize=2, label=MODEL_LABEL[model] if metric == "rmse" else None)
        labels = [f"{n} (n={families[n]['n']}{', benzene holdout' if n == 'c1ccccc1' else ''}{', descriptive' if families[n]['descriptive_only'] else ''})" for n in names]
        axes[0].set_yticks(y, labels, fontsize=7)
        axes[0].invert_yaxis()
        axes[0].set(xlabel="RMSE (LogS)")
        axes[1].axvline(0, color="#0b0b0b", linewidth=1)
        axes[1].set(xlabel="Mean signed error (pred. − obs.)")
        axes[0].legend(fontsize=7)
        self.save(fig, "F11", "Filled markers with 95% intervals: n ≥ 20 (molecule bootstrap within named families, cluster bootstrap within aggregate families). Open markers: n < 20, descriptive only.")

    def f12(self) -> None:
        fig, ax = self.new("F12")
        families = list(self.d.tables["T5_family_diagnostics.json"]["families"])
        rng = np.random.default_rng(0)
        for i, family in enumerate(families):
            protocol = HOLDOUT if family == "c1ccccc1" else CV
            mol = self.d.mol[protocol]
            subset = mol[mol["family"] == family]
            for offset, model in zip((-0.25, 0, 0.25), FIGURE_MODELS):
                ax.scatter(i + offset + rng.uniform(-0.08, 0.08, len(subset)), subset[f"res:{model}"], s=6, color=self.colors[model], alpha=0.5, linewidths=0, label=MODEL_LABEL[model] if i == 0 else None)
        ax.axhline(0, color="#0b0b0b", linewidth=1)
        names = {"c1ncncn1": "triazine", "c1ccncc1": "pyridine", "c1ccc(-c2ccccc2)cc1": "biphenyl"}
        ax.set_xticks(range(len(families)), [f"{names.get(f, f)}{' *' if f in names else ''}" for f in families], rotation=40, ha="right", fontsize=7)
        ax.set(ylabel="Residual (predicted − observed)")
        ax.legend(fontsize=7, markerscale=2)
        self.save(fig, "F12", "Out-of-fold residuals by family (benzene from its holdout; others from non-benzene CV). * focus families. Families with n < 20 are descriptive only. Jitter seed 0.")

    def f13(self) -> None:
        fig, ax = self.new("F13")
        encoding = self.style["focus_family_encoding"]
        murcko = {spec["murcko"]: name for name, spec in self.d.config["families"]["focus"].items()}
        data = self.d.data
        other = ~data["murcko"].isin(list(murcko))
        ax.scatter(data.loc[other, "LogP"], data.loc[other, TARGET_COLUMN], s=8, color=encoding["other"]["color"], alpha=0.35, linewidths=0, label=f"Other (n={int(other.sum())})")
        for scaffold, name in murcko.items():
            mask = data["murcko"] == scaffold
            ax.scatter(data.loc[mask, "LogP"], data.loc[mask, TARGET_COLUMN], s=28, marker=encoding[name]["marker"], color=encoding[name]["color"], edgecolors="#0b0b0b", linewidths=0.4, label=f"{name} (n={int(mask.sum())})")
        ax.set(xlabel="Crippen LogP (calculated)", ylabel="Observed LogS")
        ax.legend(fontsize=7)
        self.save(fig, "F13", "All 1,128 ESOL molecules (observed data, no model series). Colors here denote families only.")

    def f14(self) -> None:
        fig, ax = self.new("F14")
        spec = self.d.config["families"]["focus"]["biphenyl"]
        mol = self.d.mol[CV]
        subset = mol[mol["family"] == spec["murcko"]]
        neighbours = self.d.neighbours[CV].loc[subset.index]
        extrapolating = (neighbours[["below_training_min_target", "below_knn_min_target", "logp_above_training_max"]] > 0).any(axis=1).to_numpy()
        logp = self.d.data.loc[subset.index, "LogP"]
        for model in FIGURE_MODELS:
            ax.scatter(logp, subset[f"res:{model}"], s=22, color=self.colors[model], alpha=0.8, linewidths=0, label=MODEL_LABEL[model])
            ax.scatter(logp[extrapolating], subset.loc[extrapolating, f"res:{model}"], s=60, facecolors="none", edgecolors="#0b0b0b", linewidths=0.6)
        ax.axhline(0, color="#0b0b0b", linewidth=1)
        ax.scatter([], [], s=60, facecolors="none", edgecolors="#0b0b0b", label="any extrapolation indicator")
        ax.set(xlabel="Crippen LogP", ylabel="Residual (predicted − observed)", title=f"Biphenyl family (n={len(subset)})")
        ax.legend(fontsize=7)
        self.save(fig, "F14", "Non-benzene scaffold CV, out-of-fold mean predictions. Descriptive (single scaffold family); indicators: target below training or 10-NN minimum, or LogP above training maximum, in any repeat.")

    def f15(self) -> None:
        fig, ax = self.new("F15")
        rows = pd.DataFrame(self.d.tables["T7_similarity_distance.json"]["acyclic_vs_cyclic_similarity_bins"])
        order = list(dict.fromkeys(rows["similarity_bin"]))
        for model_offset, model in zip((-0.2, 0, 0.2), FIGURE_MODELS):
            for ring, marker, shift in (("cyclic", "o", -0.05), ("acyclic", "s", 0.05)):
                sub = rows[rows["ring_status"] == ring].set_index("similarity_bin").loc[order]
                values = np.array([v["rmse"] if v["rmse"] is not None else np.nan for v in sub[model]])
                ci = np.array([v["ci"] if v["ci"] is not None else [np.nan, np.nan] for v in sub[model]])
                x = np.arange(len(order)) + model_offset + shift
                ax.errorbar(x, values, yerr=[values - ci[:, 0], ci[:, 1] - values], fmt=marker, color=self.colors[model], markersize=5, capsize=2, label=f"{MODEL_LABEL[model]}, {ring}")
        counts = rows.pivot(index="similarity_bin", columns="ring_status", values="n").loc[order]
        ax.set_xticks(range(len(order)), [f"{b}\ncyclic n={counts.loc[b, 'cyclic']}\nacyclic n={counts.loc[b, 'acyclic']}" for b in order], fontsize=7)
        ax.set(xlabel="Max Tanimoto similarity to training set (repeat-averaged)", ylabel="RMSE (LogS)")
        ax.legend(fontsize=6, ncol=2)
        self.save(fig, "F15", "Non-benzene scaffold CV; 95% scaffold-cluster bootstrap intervals (none when n < 2). Circles cyclic, squares acyclic.")

    def f16(self) -> None:
        fig, axes = self.new("F16", 2, 2)
        table = self.d.tables["T7_similarity_distance.json"]["distance_reliability"]
        for row, protocol in enumerate(PROTOCOLS):
            entry = table[protocol]
            for model in FIGURE_MODELS:
                values = np.array([q[model]["rmse"] for q in entry["quintiles"]])
                ci = np.array([q[model]["rmse_ci"] for q in entry["quintiles"]])
                axes[row, 0].errorbar(np.arange(1, 6), values, yerr=[values - ci[:, 0], ci[:, 1] - values], marker="o", color=self.colors[model], capsize=2, label=MODEL_LABEL[model])
                curve = pd.DataFrame(entry["selective_prediction"][model])
                axes[row, 1].plot(curve["fraction"], curve["rmse"], marker="o", markersize=3, color=self.colors[model], label=MODEL_LABEL[model])
            axes[row, 0].set(title=f"{PROTOCOL_LABEL[protocol]} (n={len(self.d.mol[protocol])})", xlabel="Descriptor-distance quintile (1 = closest)", ylabel="RMSE (LogS)")
            axes[row, 1].set(title="Selective prediction", xlabel="Fraction of molecules kept (closest first)", ylabel="RMSE (LogS)")
        axes[0, 0].legend(fontsize=7)
        self.save(fig, "F16", "Distance: nearest outer-training molecule in standardized descriptor space (repeat-averaged). Intervals: scaffold-cluster (CV) or molecule (benzene, descriptive) bootstrap.")

    def f17(self) -> None:
        fig, ax = self.new("F17")
        sds = self.d.tables["T8_representation_floor.json"]["group_target_sd"]
        ax.hist(sds, bins=30, color=self.colors["random_forest_reference"], edgecolor=self.style["surface"], linewidth=0.5)
        floor = self.d.tables["T8_representation_floor.json"]["floor_rmse"]
        ax.axvline(floor, color="#0b0b0b", linestyle="--", linewidth=1.2)
        ax.text(floor, ax.get_ylim()[1] * 0.95, f"  dataset floor RMSE {floor:.3f}\n  (over all 1,128 molecules)", va="top", fontsize=7)
        ax.set(xlabel="Within-group target sd (LogS)", ylabel="Identical-descriptor groups", title=f"{len(sds)} groups sharing identical six-descriptor vectors")
        self.save(fig, "F17", "Dataset-specific limitation of the six-descriptor representation of ESOL; not universal irreducible solubility error and not measurement noise.")

    def f18(self) -> None:
        fig, axes = self.new("F18", 2, 3, sharey=True)
        coverage = self.d.coverage
        markers = {"tree_spread": "D", "cross_fold_residual": "o", "shift_check_cross_fold": "s"}
        for column, level in enumerate(self.d.config["interval_diagnostics"]["levels"]):
            for row, prefix in enumerate(("family:", "distance_quintile:")):
                ax = axes[row, column]
                subset = coverage[(coverage["level"] == level) & coverage["group"].str.startswith(prefix)]
                groups = list(dict.fromkeys(subset[subset["protocol"] == CV]["group"]))
                for offset, ((method, model), frame) in zip(np.linspace(-0.25, 0.25, 3), subset[subset["protocol"] == CV].groupby(["method", "model"], sort=True)):
                    frame = frame.set_index("group").loc[groups]
                    ax.scatter(np.arange(len(groups)) + offset, frame["observed_coverage"], marker=markers[method], color=self.colors[model], s=18, label=f"{method.replace('_', ' ')}, {MODEL_LABEL[model]}")
                ax.axhline(level, color="#0b0b0b", linestyle="--", linewidth=1)
                ax.set_xticks(range(len(groups)), [g.split(":", 1)[1][:14] for g in groups], rotation=60, ha="right", fontsize=6)
                ax.set(title=f"Nominal {int(level * 100)}%", ylim=(0, 1.02), xlabel="Family" if row == 0 else "Descriptor-distance quintile (1 = closest)")
            axes[0, column].set_ylabel("Observed coverage") if column == 0 else None
        axes[1, 0].set_ylabel("Observed coverage")
        axes[0, 0].legend(fontsize=6, loc="lower left")
        self.save(fig, "F18", "Exploratory empirical diagnostics, non-benzene scaffold CV: observed coverage only, no guarantee. Benzene shift-check coverage is in T9. Families with n < 20 descriptive.")

    def f19(self) -> None:
        fig, ax = self.new("F19")
        table = self.d.tables["T10_production_model.json"]
        units = list(table["permutation"])
        y = np.arange(len(units))
        for offset, kind, marker in ((-0.15, "permutation", "o"), (0.15, "drop_column", "s")):
            values = np.array([table[kind][u]["pooled_importance"] for u in units])
            ci = np.array([table[kind][u]["ci_molecule_bootstrap"] for u in units])
            ax.errorbar(values, y + offset, xerr=[values - ci[:, 0], ci[:, 1] - values], fmt=marker, color=self.colors["random_forest_reference"], capsize=2, label=kind.replace("_", "-"))
        ax.axvline(0, color="#0b0b0b", linewidth=1)
        ax.set_yticks(y, [u.split(":", 1)[1] if u.startswith("feature:") else u.replace(":", ": ") for u in units], fontsize=7)
        ax.invert_yaxis()
        ax.set(xlabel="RMSE increase on Phase 3 test set (LogS)", title=f"Deployed RF, Phase 3 random split (n={table['n_test']})")
        ax.legend(fontsize=7)
        self.save(fig, "F19", "Random-split diagnostics of the deployed model; not evidence of scaffold generalization. Molecule bootstrap 95% intervals (descriptive).")


# Orchestration -------------------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict(orient="records"))
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False))


def run_benchmark(diagnostics: Diagnostics) -> dict[str, Any]:
    start = perf_counter()
    diagnostics.load()
    diagnostics.refit()
    refit_seconds = perf_counter() - start
    tasks = diagnostics.fold_tasks()
    t0 = perf_counter()
    diagnose_fold(tasks[0])
    cv_fold_seconds = perf_counter() - t0
    t0 = perf_counter()
    diagnose_fold(tasks[-1])
    holdout_seconds = perf_counter() - t0
    serial = refit_seconds + 25 * cv_fold_seconds + holdout_seconds + 60.0
    return {
        "refit_and_gate_seconds": refit_seconds,
        "one_cv_fold_seconds": cv_fold_seconds,
        "holdout_fold_seconds": holdout_seconds,
        "analysis_and_figures_allowance_seconds": 60.0,
        "estimated_serial_minutes": serial / 60,
        "approval_threshold_minutes": diagnostics.config["runtime"]["approval_threshold_wall_minutes"],
        "within_threshold": serial / 60 <= diagnostics.config["runtime"]["approval_threshold_wall_minutes"],
        "refit_max_abs_difference": diagnostics.checks["refit_max_abs_difference"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results" / "phase6e")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports" / "phase6e")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "docs" / "images" / "phase6e")
    args = parser.parse_args()

    config, config_md5 = load_config(args.config)
    verify_checksums(config["protected_files_md5"])
    diagnostics = Diagnostics(config, config_md5, args.n_jobs)
    if diagnostics.cfg6d_md5 != config["protected_files_md5"]["configs/phase6d.json"]:
        raise DiagnosticGateError("Phase 6D configuration checksum mismatch.")

    if args.benchmark:
        benchmark = run_benchmark(diagnostics)
        benchmark.update({"config_md5": config_md5, "environment": environment_info(), "run_at_utc": datetime.now(timezone.utc).isoformat()})
        write_json(args.reports_dir / "benchmark.json", benchmark)
        print(json.dumps(_jsonable(benchmark), indent=2))
        verify_checksums(config["protected_files_md5"])
        return

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{config_md5[:8]}"
    run_dir = create_run_directory(args.results_root, run_id)
    start = perf_counter()
    diagnostics.load()
    diagnostics.refit()
    print(f"Refit gate passed after {perf_counter() - start:.1f} s: {diagnostics.checks['refit_max_abs_difference']}", flush=True)
    fold_results = diagnostics.run_folds(diagnostics.fold_tasks())
    print(f"Fold diagnostics finished after {perf_counter() - start:.1f} s", flush=True)
    tables = diagnostics.analyse(fold_results)
    print(f"Analysis finished after {perf_counter() - start:.1f} s", flush=True)
    verify_checksums(config["protected_files_md5"])

    figures = FigureBuilder(diagnostics, args.figures_dir).build_all()
    runtime = perf_counter() - start
    verify_checksums(config["protected_files_md5"])

    summary = {
        "run_id": run_id,
        "config_md5": config_md5,
        "phase6d_run": config["inputs"]["phase6d_run_dir"],
        "runtime_seconds": runtime,
        "n_jobs": args.n_jobs,
        "checks": diagnostics.checks,
        "protected_files_verified": len(config["protected_files_md5"]),
        "figures": figures,
        "environment": environment_info(),
        "role": config["role"],
    }
    for name, payload in tables.items():
        write_json(args.reports_dir / name, payload)
    write_json(args.reports_dir / "summary.json", summary)

    write_json(run_dir / "run_info.json", summary)
    (run_dir / "config_snapshot.json").write_text(args.config.read_text())
    (run_dir / "pip_freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    neighbours = diagnostics.neighbour_rows
    neighbours.to_csv(run_dir / "neighbour_records.csv.gz", index=False)
    for (protocol, model), arrays in diagnostics.importance_arrays.items():
        np.savez_compressed(run_dir / f"importance_{protocol}_{model}.npz", base=arrays["base"], **{f"perm|{k}": v for k, v in arrays["perm"].items()}, **{f"drop|{k}": v for k, v in arrays["drop"].items()})
    for protocol, (rows, values) in diagnostics.shap_values.items():
        np.savez_compressed(run_dir / f"treeshap_{protocol}.npz", row_id=rows, contributions=values)
    diagnostics.coverage.to_csv(run_dir / "interval_coverage.csv", index=False)
    pd.concat(
        [pd.DataFrame({"protocol": r["protocol"], "repeat": r["repeat"], "fold": r["fold"], "row_id": r["row_id"], "tree_sd": r["tree_sd"], **{f"pred:{m}": r["predictions"][m] for m in IMPORTANCE_MODELS}}) for r in fold_results]
    ).to_csv(run_dir / "refit_predictions_and_tree_spread.csv.gz", index=False)
    print(f"Run {run_id} finished in {runtime:.1f} s; outputs in {run_dir}")


if __name__ == "__main__":
    main()
