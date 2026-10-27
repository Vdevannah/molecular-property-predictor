import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.model_diagnostics import (
    CONFIG_PATH,
    DiagnosticGateError,
    ale_curve,
    ale_edges,
    bootstrap_ols,
    calibration_fit,
    check_reproduction,
    cross_fold_half_widths,
    drop_column_squared_errors,
    fold_rmse_increase,
    importance_units,
    interpolate_curve,
    is_ch2_homologue,
    permutation_indices,
    permuted_squared_errors,
    pooled_importance,
    require_single_protocol,
    save_figure,
    select_extreme_errors,
    shrinkage_slope,
    similarity_bin,
    tree_spread,
    treeshap_contributions,
    verify_descriptor_functions,
    verify_raw_alignment,
)
from src.scaffold_evaluation import cluster_bootstrap_weights, error_floor, load_config
from src.train_model import FEATURE_COLUMNS, TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "esol_descriptors.csv"
RUN_6D = PROJECT_ROOT / "results" / "phase6d" / "20261027T154454Z_d02c301c"


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config(CONFIG_PATH)[0]


def _frame(rng, n=60):
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    return X


# Integrity -----------------------------------------------------------------------------------


def test_reproduction_gate_passes_identical_and_stops_on_mismatch():
    values = np.array([1.0, 2.0, 3.0])
    assert check_reproduction(values, values.copy(), 1e-9, "same") == 0.0
    with pytest.raises(DiagnosticGateError, match="exceeds"):
        check_reproduction(values + np.array([0, 2e-9, 0]), values, 1e-9, "perturbed")


def test_raw_alignment_gate():
    descriptors = pd.DataFrame({"Compound ID": ["a", "b"], "smiles": ["C", "CC"]})
    verify_raw_alignment(descriptors, descriptors.copy())
    with pytest.raises(DiagnosticGateError):
        verify_raw_alignment(descriptors, descriptors.iloc[::-1].reset_index(drop=True))


def test_descriptor_functions_match_source(config):
    verify_descriptor_functions(config, (PROJECT_ROOT / "src" / "descriptors.py").read_text())
    with pytest.raises(DiagnosticGateError):
        verify_descriptor_functions(config, "Descriptors.MolWt(m)")


def test_statistics_refuse_to_pool_protocols():
    assert require_single_protocol(pd.DataFrame({"protocol": ["a", "a"]})) == "a"
    with pytest.raises(ValueError, match="pool"):
        require_single_protocol(pd.DataFrame({"protocol": ["a", "b"]}))


# Importance ----------------------------------------------------------------------------------


def test_units_follow_configured_order(config):
    units = importance_units(config)
    assert [name for name, _ in units[:6]] == [f"feature:{f}" for f in FEATURE_COLUMNS]
    assert units[6] == ("primary:polarity_hbonding", ["TPSA", "HBA", "HBD"])
    assert len(units) == 12


def test_permutation_seeds_do_not_depend_on_call_order():
    first = permutation_indices(50, 6105, 0, 2, 3, 4, 5)
    permutation_indices(50, 6105, 0, 0, 0, 0, 0)
    again = permutation_indices(50, 6105, 0, 2, 3, 4, 5)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, permutation_indices(50, 6105, 0, 2, 3, 4, 6))


def test_pooled_importance_differs_from_weighted_fold_average():
    # Two folds of unequal size: pooled RMSE from molecule-level errors versus weighted fold differences.
    base = np.array([[1.0, 1.0, 1.0, 4.0]])  # (repeats, molecules)
    perturbed = np.array([[[4.0, 4.0, 4.0, 4.0]]])  # (permutations, repeats, molecules)
    pooled = pooled_importance(perturbed, base)
    assert pooled["base_rmse"] == pytest.approx(np.sqrt(7 / 4))
    assert pooled["importance_mean"] == pytest.approx(2.0 - np.sqrt(7 / 4))
    fold_a = fold_rmse_increase(perturbed[:, 0, :3], base[0, :3])  # 3 molecules: 2 - 1
    fold_b = fold_rmse_increase(perturbed[:, 0, 3:], base[0, 3:])  # 1 molecule: 2 - 2
    weighted = (3 * fold_a + 1 * fold_b) / 4
    assert weighted == pytest.approx(0.75)
    assert pooled["importance_mean"] != pytest.approx(weighted)


def test_pooled_importance_averages_repeats_before_rmse():
    base = np.array([[1.0, 0.0], [1.0, 4.0]])
    perturbed = np.array([[[1.0, 0.0], [1.0, 4.0]], [[4.0, 4.0], [4.0, 4.0]]])
    pooled = pooled_importance(perturbed, base)
    expected_base = np.sqrt(np.mean([1.0, 2.0]))
    assert pooled["base_rmse"] == pytest.approx(expected_base)
    np.testing.assert_allclose(pooled["per_permutation"], [0.0, 2.0 - expected_base])


def test_irrelevant_feature_has_no_permutation_importance(config):
    rng = np.random.default_rng(0)
    X = _frame(rng)
    y = 3 * X["LogP"].to_numpy()
    model = LinearRegression().fit(X, y)
    model.coef_[[i for i, f in enumerate(FEATURE_COLUMNS) if f != "LogP"]] = 0.0
    errors = permuted_squared_errors(model, X, y, importance_units(config), 5, (6105, 0, 0, 0))
    assert errors["feature:MolWt"].max() == pytest.approx(0.0)
    assert errors["feature:LogP"].mean() > 1.0


def test_grouped_permutation_recovers_importance_split_between_duplicates():
    rng = np.random.default_rng(1)
    X = _frame(rng, 200)
    X["HBA"] = X["TPSA"]  # perfectly correlated pair
    y = 2 * X["TPSA"].to_numpy()
    model = LinearRegression().fit(X, y)
    units = [("feature:TPSA", ["TPSA"]), ("feature:HBA", ["HBA"]), ("group", ["TPSA", "HBA"]), ("feature:RotBonds", ["RotBonds"])]
    errors = permuted_squared_errors(model, X, y, units, 10, (6105, 0, 0, 0))
    single = max(errors["feature:TPSA"].mean(), errors["feature:HBA"].mean())
    assert errors["group"].mean() > 2 * single
    assert errors["feature:RotBonds"].mean() == pytest.approx(0.0, abs=1e-12)


def test_permutation_stays_within_test_fold():
    rng = np.random.default_rng(2)
    X = _frame(rng, 30)
    seen = []

    class Recorder:
        def predict(self, frame):
            seen.append(frame["LogP"].to_numpy().copy())
            return np.zeros(len(frame))

    permuted_squared_errors(Recorder(), X, np.zeros(30), [("feature:LogP", ["LogP"])], 3, (6105, 0, 0, 0))
    for values in seen:
        np.testing.assert_array_equal(np.sort(values), np.sort(X["LogP"].to_numpy()))


def test_drop_column_keeps_stored_hyperparameters_and_ignores_unrelated_feature():
    rng = np.random.default_rng(3)
    X_train, X_test = _frame(rng, 120), _frame(rng, 40)
    y_train, y_test = 2 * X_train["LogP"].to_numpy(), 2 * X_test["LogP"].to_numpy()
    errors = drop_column_squared_errors(LinearRegression(), {"fit_intercept": False}, X_train, y_train, X_test, y_test, [("feature:MolWt", ["MolWt"]), ("feature:LogP", ["LogP"])])
    assert errors["feature:MolWt"].mean() == pytest.approx(0.0, abs=1e-20)
    assert errors["feature:LogP"].mean() > 1.0
    forest = RandomForestRegressor(n_estimators=5, random_state=0)
    drop_column_squared_errors(forest, {"max_depth": 2}, X_train, y_train, X_test, y_test, [("feature:MolWt", ["MolWt"])])
    assert forest.get_params()["max_depth"] is None  # the template is cloned, never mutated


# ALE, TreeSHAP, tree spread ------------------------------------------------------------------


def test_ale_of_linear_model_has_slope_beta_and_zero_weighted_mean():
    rng = np.random.default_rng(4)
    X = _frame(rng, 400)
    model = LinearRegression().fit(X, 1.5 * X["MolWt"] - 0.5 * X["LogP"])
    edges, values, counts = ale_curve(model, X, "MolWt", 20)
    np.testing.assert_allclose(np.diff(values) / np.diff(edges), 1.5, rtol=1e-9)
    midpoints = (values[:-1] + values[1:]) / 2
    assert np.sum(midpoints * counts) / counts.sum() == pytest.approx(0.0, abs=1e-12)
    assert counts.sum() == 400


def test_ale_uses_unique_values_for_sparse_integer_features():
    values = np.array([0, 0, 1, 1, 1, 2, 5, 5])
    np.testing.assert_array_equal(ale_edges(values, 20), [0, 1, 2, 5])
    assert len(ale_edges(np.linspace(0, 1, 100), 20)) == 21


def test_ale_interpolation_never_extrapolates():
    result = interpolate_curve(np.array([-1.0, 0.5, 2.0]), np.array([0.0, 1.0]), np.array([0.0, 2.0]))
    assert np.isnan(result[0]) and np.isnan(result[2]) and result[1] == pytest.approx(1.0)


def test_treeshap_contributions_reconstruct_predictions(config):
    rng = np.random.default_rng(5)
    X = _frame(rng, 80)
    model = XGBRegressor(n_estimators=20, max_depth=3, n_jobs=1, random_state=0).fit(X, X["LogP"] * 2)
    contributions = treeshap_contributions(model, X, config["feature_importance"]["treeshap"]["additivity_tolerance"])
    assert contributions.shape == (80, len(FEATURE_COLUMNS) + 1)
    with pytest.raises(DiagnosticGateError):
        treeshap_contributions(model, X, -1.0)


def test_tree_spread_is_zero_for_identical_trees():
    rng = np.random.default_rng(6)
    X = _frame(rng, 50)
    forest = RandomForestRegressor(n_estimators=4, bootstrap=False, max_features=None, random_state=0).fit(X, X["LogP"])
    assert tree_spread(forest, X).max() == pytest.approx(0.0)


# Calibration, bins, homologues, intervals --------------------------------------------------------


def test_calibration_recovers_known_slope_and_shrinkage_uses_observed():
    predicted = np.linspace(-5, 1, 50)
    observed = 0.5 + 1.2 * predicted
    slope, intercept = calibration_fit(observed, predicted)
    assert slope == pytest.approx(1.2) and intercept == pytest.approx(0.5)
    assert shrinkage_slope(observed, predicted) == pytest.approx(np.polyfit(observed, predicted - observed, 1)[0])
    weights, codes = cluster_bootstrap_weights(np.arange(50), 200, 0)
    boot_slope, boot_intercept = bootstrap_ols(weights, codes, predicted, observed)
    np.testing.assert_allclose(boot_slope, 1.2)
    np.testing.assert_allclose(boot_intercept, 0.5)


def test_similarity_bins_follow_config(config):
    bins = config["acyclic"]["similarity_bins"]
    assert similarity_bin(0.0, bins) == "[0.0, 0.4)"
    assert similarity_bin(0.4, bins) == "[0.4, 0.6)"
    assert similarity_bin(0.9999, bins) == "[0.8, 1.0)"
    assert similarity_bin(1.0, bins) == "exactly 1.0"


def test_ch2_homologue_heuristic():
    assert is_ch2_homologue("CCO", "CCCO")
    assert is_ch2_homologue("CCCCCl", "CCCCl")
    assert not is_ch2_homologue("CCO", "CCN")
    assert not is_ch2_homologue("CCO", "CCCCO")


def test_cross_fold_half_width_excludes_evaluated_fold():
    residuals = pd.DataFrame({"repeat": 0, "fold": [0, 0, 1, 1, 2, 2], "residual": [100.0, -100.0, 1.0, 2.0, 3.0, 4.0]})
    widths = cross_fold_half_widths(residuals, 0.5)
    assert widths[(0, 0)] == pytest.approx(np.quantile([1, 2, 3, 4], 0.5, method="higher"))
    assert widths[(0, 1)] == pytest.approx(np.quantile([100, 100, 3, 4], 0.5, method="higher"))
    covered = residuals["residual"].abs() <= residuals.apply(lambda r: widths[(r["repeat"], r["fold"])], axis=1)
    assert covered.tolist() == [False, False, True, True, True, True]


def test_extreme_error_selection_rule():
    residuals = pd.DataFrame({"a": [0.1, -3.0, 0.5, 2.5, 0.2], "b": [0.0, 0.0, 2.1, 0.0, 0.0]}, index=[10, 11, 12, 13, 14])
    assert select_extreme_errors(residuals, "a", 2, 2.0, ["a", "b"]) == [11, 13, 12]


def test_representation_floor_hand_calculation():
    dataset = pd.DataFrame({c: [1.0, 1.0, 1.0, 2.0] for c in FEATURE_COLUMNS})
    dataset[TARGET_COLUMN] = [0.0, 1.0, 2.0, 5.0]
    assert error_floor(dataset)["floor_rmse"] == pytest.approx(np.sqrt(2.0 / 4))


@pytest.mark.skipif(not DATA_PATH.exists(), reason="ESOL data not available")
def test_representation_floor_matches_configured_value(config):
    floor = error_floor(pd.read_csv(DATA_PATH))
    assert floor["floor_rmse"] == pytest.approx(config["representation_floor"]["expected"]["floor_rmse"], abs=1e-12)


# Figures and real-data gates -----------------------------------------------------------------------


def test_figures_are_byte_identical_across_saves(tmp_path):
    hashes = []
    for name in ("a.png", "b.png"):
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot([0, 1], [1, 0])
        save_figure(fig, tmp_path / name, 150, "footnote")
        hashes.append(hashlib.md5((tmp_path / name).read_bytes()).hexdigest())
    assert hashes[0] == hashes[1]


@pytest.mark.skipif(not (DATA_PATH.exists() and RUN_6D.exists()), reason="Phase 6D outputs not available")
def test_refit_gate_reproduces_phase6d_predictions_on_real_data(config):
    from src.model_diagnostics import Diagnostics

    diagnostics = Diagnostics(config, "test", n_jobs=1)
    diagnostics.load()
    diagnostics.folds = diagnostics.folds[:1] + diagnostics.folds[-1:]
    diagnostics.refit()
    assert max(diagnostics.checks["refit_max_abs_difference"].values()) <= 1e-9
    assert diagnostics.checks["fold_assignments_match_phase6d"]
