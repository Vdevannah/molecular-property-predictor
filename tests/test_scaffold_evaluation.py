import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.scaffold_evaluation import (
    CONFIG_PATH,
    OuterFold,
    annotate_dataset,
    assert_group_disjoint,
    assign_families,
    block_cv_folds,
    block_holdout_fold,
    bootstrap_difference,
    bootstrap_rmse,
    build_model_specs,
    cluster_bootstrap_weights,
    contrast_table,
    create_run_directory,
    decision_label,
    error_floor,
    generic_scaffold_group,
    holm_adjust,
    inner_seed,
    load_config,
    make_inner_splits,
    make_pooled_refit,
    make_variant,
    molecule_errors,
    pooled_rmse,
    pooled_scores,
    protocol_metrics,
    run_protocol,
    run_seed_check,
    validate_config,
    validate_expected_counts,
    verify_checksums,
)
from src.train_model import FEATURE_COLUMNS, TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "esol_descriptors.csv"
requires_data = pytest.mark.skipif(not DATA_PATH.exists(), reason="ESOL descriptor data not available")


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()[0]


@pytest.fixture(scope="module")
def esol() -> pd.DataFrame:
    if not DATA_PATH.exists():
        pytest.skip("ESOL descriptor data not available")
    return annotate_dataset(pd.read_csv(DATA_PATH))


def _synthetic_groups() -> np.ndarray:
    labels = ["block"] * 20
    for index in range(8):
        labels += [f"medium{index}"] * 3
    labels += [f"single{index}" for index in range(30)]
    return np.array(labels)


@pytest.fixture
def synthetic() -> tuple[pd.DataFrame, np.ndarray]:
    groups = _synthetic_groups()
    rng = np.random.default_rng(0)
    features = rng.random((len(groups), len(FEATURE_COLUMNS)))
    dataset = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    dataset[TARGET_COLUMN] = 3 * features[:, 1] - 1 + rng.normal(0, 0.05, len(groups))
    dataset["row_id"] = np.arange(len(groups))
    return dataset, groups


def _small_specs() -> dict:
    return {
        "baseline": (DummyRegressor(strategy="mean"), None),
        "random_forest_reference": (Pipeline([("scaler", StandardScaler()), ("linear", LinearRegression())]), None),
        "xgboost": (Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(max_iter=10000))]), {"lasso__alpha": [0.001, 0.1]}),
    }


def _run_small(dataset, groups, repeats=2):
    folds = block_cv_folds(groups, "block", n_splits=3, repeats=repeats)
    return folds, run_protocol(
        dataset, groups, folds, _small_specs(), scenario="test", protocol="cv", block_label="block", inner_splits=3, n_jobs=1
    )


# Configuration and integrity ---------------------------------------------------


def test_committed_config_is_followed_by_implementation(config):
    validate_config(config)
    specs = build_model_specs(config)
    assert list(specs) == list(config["models"])
    assert specs["random_forest_reference"][0].get_params()["n_estimators"] == 200
    assert specs["lasso"][0].get_params()["lasso__max_iter"] == 100000
    assert specs["xgboost"][1] == config["models"]["xgboost"]["grid"]
    for name in ("random_forest_reference", "random_forest_tuned", "xgboost"):
        assert specs[name][0].get_params()["n_jobs"] == 1
        assert specs[name][0].get_params()["random_state"] == 42


def test_config_change_to_seed_formula_is_rejected(config):
    changed = copy.deepcopy(config)
    changed["protocols"]["nonbenzene_scaffold_cv"]["outer_seed"] = "7 + repeat"
    with pytest.raises(ValueError, match="outer_seed"):
        validate_config(changed)


def test_checksum_mismatch_aborts(tmp_path):
    protected = tmp_path / "artifact.bin"
    protected.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="artifact.bin"):
        verify_checksums({"artifact.bin": "0" * 32}, root=tmp_path)


# Protected files that the documented setup regenerates (`python src/train_model.py`); every
# other protected entry is a historical experiment artifact that a fresh clone does not contain.
SETUP_ARTIFACTS = frozenset({"models/solubility_model.joblib", "models/feature_columns.joblib", "results/phase3_metrics.json"})


def check_protected_artifacts(expected: dict, root: Path) -> None:
    """Skip when no historical artifact exists, fail when only some exist, else verify every checksum."""
    historical = [path for path in expected if path != "check" and path not in SETUP_ARTIFACTS]
    present = [path for path in historical if (root / path).exists()]
    if not present:
        pytest.skip(
            "Original Phase 4 and Phase 6C experiment artifacts are Git-ignored and not included in a "
            "fresh clone; see docs/REPRODUCIBILITY.md."
        )
    missing = sorted(set(historical) - set(present))
    if missing:
        pytest.fail(f"Incomplete historical experiment artifacts; missing: {', '.join(missing)}")
    verify_checksums(expected, root=root)


def test_protected_files_match_committed_checksums(config):
    check_protected_artifacts(config["protected_files_md5"], PROJECT_ROOT)


def test_setup_artifacts_are_listed_in_protected_files(config):
    assert SETUP_ARTIFACTS <= set(config["protected_files_md5"])


@pytest.fixture
def protected_tree(tmp_path) -> tuple[Path, dict, dict]:
    files = {"models/model.bin": b"model", "results/phase4.json": b"phase4", "results/phase6c/summary.csv": b"summary"}
    expected = {path: hashlib.md5(content).hexdigest() for path, content in files.items()}
    expected["check"] = "verified before and after every run"
    return tmp_path, expected, files


def _write(root: Path, files: dict, paths: list[str]) -> None:
    for path in paths:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(files[path])


def test_protected_check_skips_when_no_historical_artifact_exists(protected_tree, monkeypatch):
    root, expected, files = protected_tree
    monkeypatch.setattr(f"{__name__}.SETUP_ARTIFACTS", frozenset({"models/model.bin"}))
    with pytest.raises(pytest.skip.Exception, match="fresh clone"):
        check_protected_artifacts(expected, root)
    _write(root, files, ["models/model.bin"])  # setup artifact alone (fresh clone after setup) still skips
    with pytest.raises(pytest.skip.Exception, match="fresh clone"):
        check_protected_artifacts(expected, root)


def test_protected_check_fails_when_historical_artifacts_are_partial(protected_tree, monkeypatch):
    root, expected, files = protected_tree
    monkeypatch.setattr(f"{__name__}.SETUP_ARTIFACTS", frozenset({"models/model.bin"}))
    _write(root, files, ["models/model.bin", "results/phase4.json"])
    with pytest.raises(pytest.fail.Exception, match="results/phase6c/summary.csv"):
        check_protected_artifacts(expected, root)


def test_protected_check_fails_on_checksum_mismatch(protected_tree, monkeypatch):
    root, expected, files = protected_tree
    monkeypatch.setattr(f"{__name__}.SETUP_ARTIFACTS", frozenset({"models/model.bin"}))
    _write(root, files, list(files))
    (root / "models/model.bin").write_bytes(b"retrained")
    with pytest.raises(RuntimeError, match="models/model.bin"):
        check_protected_artifacts(expected, root)


def test_protected_check_passes_when_all_checksums_match(protected_tree, monkeypatch):
    root, expected, files = protected_tree
    monkeypatch.setattr(f"{__name__}.SETUP_ARTIFACTS", frozenset({"models/model.bin"}))
    _write(root, files, list(files))
    check_protected_artifacts(expected, root)


def test_run_directory_is_never_overwritten(tmp_path):
    create_run_directory(tmp_path, "run1")
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, "run1")


# Grouping, families and variants ---------------------------------------------------


def test_generic_scaffold_merges_atom_types_and_keeps_acyclic_identity():
    assert generic_scaffold_group("c1ccncc1") == generic_scaffold_group("Oc1ccccc1") == "C1CCCCC1"
    assert generic_scaffold_group("CCO") == generic_scaffold_group("OCC ") == "CCO"


@requires_data
def test_esol_matches_configured_counts(esol, config):
    validate_expected_counts(esol, config)
    families = assign_families(esol, config)
    assert families.value_counts().sum() == 1128
    assert set(esol.loc[families == "acyclic", "acyclic"]) == {True}


def test_s1_removes_exactly_the_hexitol_pair(config):
    dataset = pd.DataFrame(
        {
            "Compound ID": ["mannitol", "Sorbitol", "ethanol"],
            "identity": ["OCC(O)C(O)C(O)C(O)CO", "OCC(O)C(O)C(O)C(O)CO", "CCO"],
        }
    )
    assert make_variant(dataset, "S1", config)["Compound ID"].tolist() == ["ethanol"]


def test_s2_collapses_agreeing_duplicates_and_drops_conflicts(config):
    dataset = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3, 4],
            "identity": ["A", "A", "B", "B", "C"],
            **{column: [1.0, 1.0, 2.0, 2.0, 3.0] for column in FEATURE_COLUMNS},
            TARGET_COLUMN: [-1.0, -1.2, 0.0, 1.0, 2.0],
        }
    )
    variant = make_variant(dataset, "S2", config)
    assert variant["identity"].tolist() == ["A", "C"]
    assert variant.loc[variant["identity"] == "A", TARGET_COLUMN].item() == pytest.approx(-1.1)


def test_s2_tolerates_floating_point_rounding_in_duplicate_descriptors(config):
    dataset = pd.DataFrame(
        {
            "row_id": [0, 1],
            "identity": ["A", "A"],
            **{column: [1.0, 1.0] for column in FEATURE_COLUMNS},
            TARGET_COLUMN: [0.0, 0.1],
        }
    )
    dataset.loc[1, "MolWt"] = 1.0 + 5.7e-14  # magnitude observed for ESOL atom-order duplicates
    assert len(make_variant(dataset, "S2", config)) == 1


def test_s2_aborts_when_duplicate_descriptors_differ(config):
    dataset = pd.DataFrame(
        {
            "row_id": [0, 1],
            "identity": ["A", "A"],
            **{column: [1.0, 1.0] for column in FEATURE_COLUMNS},
            TARGET_COLUMN: [0.0, 0.1],
        }
    )
    dataset.loc[1, "LogP"] = 2.0
    with pytest.raises(ValueError, match="Descriptors differ"):
        make_variant(dataset, "S2", config)


# Splits -------------------------------------------------------------------------


def test_block_cv_keeps_block_in_training_and_tests_each_row_once():
    groups = _synthetic_groups()
    block = set(np.flatnonzero(groups == "block"))
    folds = block_cv_folds(groups, "block", n_splits=5, repeats=3)
    for repeat in range(3):
        tested = np.concatenate([f.test_index for f in folds if f.repeat == repeat])
        assert sorted(tested) == sorted(set(range(len(groups))) - block)
    for fold in folds:
        assert block <= set(fold.train_index)
        assert not block & set(fold.test_index)
        assert_group_disjoint(groups, fold.train_index, fold.test_index)


def test_block_holdout_tests_block_exactly_once():
    groups = _synthetic_groups()
    (fold,) = block_holdout_fold(groups, "block")
    assert set(groups[fold.test_index]) == {"block"}
    assert "block" not in set(groups[fold.train_index])
    assert len(fold.train_index) + len(fold.test_index) == len(groups)


def test_inner_splits_never_validate_on_block_and_respect_groups():
    groups = _synthetic_groups()
    splits = make_inner_splits(groups, "block", n_splits=4, seed=inner_seed(0, 0))
    block = set(np.flatnonzero(groups == "block"))
    validated = np.concatenate([test for _, test in splits])
    assert sorted(validated) == sorted(set(range(len(groups))) - block)
    for train, test in splits:
        assert block <= set(train)
        assert_group_disjoint(groups, train, test)


def test_group_overlap_is_detected():
    groups = np.array(["a", "a", "b"])
    with pytest.raises(AssertionError, match="both training and test"):
        assert_group_disjoint(groups, np.array([0]), np.array([1, 2]))


@requires_data
@pytest.mark.parametrize("grouping", ["murcko", "generic"])
def test_no_heldout_scaffold_reaches_training_on_esol(esol, config, grouping):
    groups = esol[grouping].to_numpy()
    block = config["grouping"][grouping]["largest_group"]["label"]
    folds = block_cv_folds(groups, block, 5, 5) + block_holdout_fold(groups, block)
    assert len(folds) == 26
    for fold in folds:
        assert_group_disjoint(groups, fold.train_index, fold.test_index)
        for train, test in make_inner_splits(groups[fold.train_index], block, 5, inner_seed(fold.repeat, fold.fold)):
            assert_group_disjoint(groups[fold.train_index], train, test)
            assert block not in set(groups[fold.train_index][test])


# Molecule-weighted pooled RMSE ---------------------------------------------------


def test_pooled_rmse_matches_hand_calculation():
    assert pooled_rmse([1.0, 2.0], [3, 1]) == pytest.approx(np.sqrt(7 / 4)) == pytest.approx(1.3228756555)
    assert np.mean([1.0, 2.0]) == 1.5
    assert np.sqrt(np.mean([1.0, 4.0])) == pytest.approx(1.5811388301)


def test_pooled_rmse_equals_rmse_of_raw_pooled_errors():
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=10)
    y_pred = y_true + rng.normal(scale=0.5, size=10)
    folds = [np.arange(0, 6), np.arange(6, 8), np.arange(8, 10)]
    fold_rmse = [np.sqrt(np.mean((y_pred[f] - y_true[f]) ** 2)) for f in folds]
    assert pooled_rmse(fold_rmse, [len(f) for f in folds]) == pytest.approx(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def test_refit_selects_pooled_best_when_unweighted_mean_disagrees():
    cv_results = {"split0_test_score": np.array([-1.0, -1.4]), "split1_test_score": np.array([-2.0, -1.4])}
    sizes = [3, 1]
    np.testing.assert_allclose(pooled_scores(cv_results, sizes), [np.sqrt(1.75), 1.4])
    assert make_pooled_refit(sizes)(cv_results) == 0  # unweighted means 1.5 vs 1.4 would choose candidate 1


def test_grid_search_selection_matches_raw_inner_predictions(synthetic):
    dataset, groups = synthetic
    X, y = dataset[FEATURE_COLUMNS], dataset[TARGET_COLUMN].to_numpy()
    splits = make_inner_splits(groups, "block", 3, seed=5)
    sizes = [len(test) for _, test in splits]
    grid = {"lasso__alpha": [0.001, 0.05, 0.5]}
    estimator = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(max_iter=10000))])
    search = GridSearchCV(estimator, grid, scoring="neg_root_mean_squared_error", cv=splits, refit=make_pooled_refit(sizes)).fit(X, y)

    raw_pooled = []
    for alpha in grid["lasso__alpha"]:
        squared = []
        for train, test in splits:
            model = estimator.set_params(lasso__alpha=alpha).fit(X.iloc[train], y[train])
            squared.append((model.predict(X.iloc[test]) - y[test]) ** 2)
        raw_pooled.append(np.sqrt(np.concatenate(squared).mean()))
    np.testing.assert_allclose(pooled_scores(search.cv_results_, sizes), raw_pooled)
    assert search.best_index_ == int(np.argmin(raw_pooled))


# Protocol execution ----------------------------------------------------------------


def test_run_protocol_predicts_each_nonblock_molecule_once_per_repeat(synthetic):
    dataset, groups = synthetic
    folds, results = _run_small(dataset, groups)
    predictions = results["predictions"]
    nonblock = set(np.flatnonzero(groups != "block"))
    for (model, repeat), frame in predictions.groupby(["model", "repeat"]):
        assert sorted(frame["row_id"]) == sorted(nonblock)
    fold_metrics = results["fold_metrics"]
    assert (fold_metrics.groupby(["repeat", "fold"])["n_test"].nunique() == 1).all()
    tuned = fold_metrics[fold_metrics["model"] == "xgboost"]
    assert tuned["selected_params"].str.contains("lasso__alpha").all()
    selected = results["inner_results"][results["inner_results"]["selected"]]
    assert len(selected) == len(folds)
    best = results["inner_results"].groupby(["repeat", "fold"])["pooled_rmse"].min().to_numpy()
    np.testing.assert_allclose(selected.sort_values(["repeat", "fold"])["pooled_rmse"].to_numpy(), best)


def test_run_protocol_is_reproducible(synthetic):
    dataset, groups = synthetic
    _, first = _run_small(dataset, groups)
    _, second = _run_small(dataset, groups)
    for key in ("predictions", "inner_results", "assignments"):
        pd.testing.assert_frame_equal(first[key], second[key])
    pd.testing.assert_frame_equal(first["fold_metrics"].drop(columns="fit_seconds"), second["fold_metrics"].drop(columns="fit_seconds"))


def test_seed_check_reuses_selected_parameters(synthetic):
    dataset, groups = synthetic
    folds, results = _run_small(dataset, groups, repeats=1)
    specs = {
        "random_forest_reference": (Lasso(alpha=0.01), None),
        "xgboost": (Lasso(), {"alpha": [0.001, 0.1]}),
    }
    fold_metrics = results["fold_metrics"].assign(selected_params=json.dumps({"alpha": 0.001}))
    seeds = run_seed_check(dataset, folds, fold_metrics, specs, [43, 44], scenario="test", protocol="cv", groups=groups)
    assert set(seeds["model_seed"]) == {43, 44}
    assert len(seeds) == 2 * 2 * (groups != "block").sum()


# Repeat averaging and metrics --------------------------------------------------------


def test_molecule_errors_average_repeats_into_one_observation():
    predictions = pd.DataFrame(
        {"model": "m", "row_id": [7, 7, 7], "repeat": [0, 1, 2], "group": "g", "y_true": 1.0, "y_pred": [1.0, 2.0, 4.0]}
    )
    molecules = molecule_errors(predictions)
    assert len(molecules) == 1
    assert molecules["sq_error"].item() == pytest.approx((0 + 1 + 9) / 3)
    assert molecules["n_repeats"].item() == 3


def test_protocol_metrics_distinguish_molecule_weighted_and_fold_averaged():
    predictions = pd.DataFrame(
        {
            "model": "m",
            "repeat": 0,
            "row_id": [0, 1, 2, 3],
            "group": ["a", "a", "a", "b"],
            "y_true": [0.0, 0.0, 0.0, 0.0],
            "y_pred": [1.0, 1.0, 1.0, 2.0],
        }
    )
    fold_metrics = pd.DataFrame({"model": ["m", "m"], "RMSE": [1.0, 2.0]})
    metrics = protocol_metrics(predictions, fold_metrics).iloc[0]
    assert metrics["repeat_averaged_rmse"] == pytest.approx(np.sqrt(7 / 4))
    assert metrics["fold_averaged_rmse"] == pytest.approx(1.5)


def test_error_floor_uses_identical_descriptor_vectors():
    dataset = pd.DataFrame({column: [1.0, 1.0, 2.0] for column in FEATURE_COLUMNS})
    dataset[TARGET_COLUMN] = [0.0, 2.0, 5.0]
    floor = error_floor(dataset)
    assert floor["rows_sharing_descriptor_vector"] == 2
    assert floor["floor_rmse"] == pytest.approx(np.sqrt(2 / 3))


# Bootstrap, decision rule and multiplicity ---------------------------------------------


def test_bootstrap_resamples_whole_clusters():
    clusters = np.array(["a", "a", "b", "c", "c", "c"])
    weights, codes = cluster_bootstrap_weights(clusters, replicates=200, seed=1)
    assert weights.shape == (200, 3)
    assert (weights.sum(axis=1) == 3).all()
    row_weights = weights[:, codes]
    np.testing.assert_array_equal(row_weights[:, 0], row_weights[:, 1])
    np.testing.assert_array_equal(row_weights[:, 3], row_weights[:, 5])


def test_bootstrap_is_reproducible_for_fixed_seed():
    clusters = np.repeat(np.arange(20), 3)
    first, _ = cluster_bootstrap_weights(clusters, 100, seed=20261027)
    second, _ = cluster_bootstrap_weights(clusters, 100, seed=20261027)
    np.testing.assert_array_equal(first, second)


def test_bootstrap_identical_models_give_zero_difference():
    clusters = np.repeat(np.arange(10), 2)
    weights, codes = cluster_bootstrap_weights(clusters, 500, seed=0)
    errors = np.random.default_rng(0).random(20)
    result = bootstrap_difference(weights, codes, errors, errors)
    assert result["delta"] == 0.0
    assert result["ci_lower"] == result["ci_upper"] == 0.0
    assert result["p_value"] == 1.0


def test_bootstrap_recovers_known_constant_difference():
    weights, codes = cluster_bootstrap_weights(np.arange(15), 500, seed=0)
    result = bootstrap_difference(weights, codes, np.full(15, 4.0), np.full(15, 1.0))
    assert result["delta"] == pytest.approx(1.0)
    assert result["ci_lower"] == pytest.approx(1.0) and result["ci_upper"] == pytest.approx(1.0)
    assert result["p_value"] == 0.0


def test_bootstrap_rmse_matches_direct_calculation_for_unit_weights():
    codes = np.array([0, 0, 1, 2])
    errors = np.array([1.0, 3.0, 4.0, 0.0])
    replicate = bootstrap_rmse(np.ones((1, 3), dtype=int), codes, errors)
    assert replicate[0] == pytest.approx(np.sqrt(errors.mean()))


def test_repeated_predictions_do_not_add_bootstrap_observations():
    base = pd.DataFrame(
        {
            "model": np.repeat(["a", "b"], 6),
            "row_id": np.tile(np.arange(6), 2),
            "group": np.tile(["g1", "g1", "g2", "g3", "g3", "g4"], 2),
            "y_true": 0.0,
            "y_pred": np.r_[np.linspace(0.1, 0.6, 6), np.linspace(0.2, 0.4, 6)],
            "repeat": 0,
        }
    )
    tripled = pd.concat([base.assign(repeat=r) for r in range(3)], ignore_index=True)
    single, _ = contrast_table(base, cluster_column="group", replicates=300, seed=3)
    repeated, _ = contrast_table(tripled, cluster_column="group", replicates=300, seed=3)
    pd.testing.assert_frame_equal(single, repeated)
    assert single["n_molecules"].item() == 6 and single["n_clusters"].item() == 4


@pytest.mark.parametrize(
    ("lower", "upper", "label"),
    [
        (-0.02, 0.03, "practically equivalent at delta"),
        (0.01, 0.04, "detectable difference smaller than delta"),
        (0.06, 0.10, "difference larger than delta"),
        (-0.10, -0.06, "difference larger than delta"),
        (0.02, 0.08, "difference detected; magnitude relative to delta uncertain"),
        (-0.04, 0.09, "inconclusive"),
    ],
)
def test_decision_rule_covers_every_branch(lower, upper, label):
    assert decision_label(lower, upper, 0.05) == label


def test_holm_adjustment_matches_known_values():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


@requires_data
def test_family_output_never_reports_r2(esol, config):
    from src.scaffold_evaluation import family_metrics

    families = assign_families(esol, config).set_axis(esol["row_id"])
    predictions = pd.DataFrame(
        {
            "model": np.repeat(["random_forest_reference", "xgboost"], len(esol)),
            "row_id": np.tile(esol["row_id"], 2),
            "group": np.tile(esol["murcko"], 2),
            "repeat": 0,
            "y_true": np.tile(esol[TARGET_COLUMN], 2),
            "y_pred": np.tile(esol[TARGET_COLUMN], 2) + 0.1,
        }
    )
    benzene = predictions["row_id"].map(families) == "c1ccccc1"
    small = copy.deepcopy(config)
    small["statistics"]["bootstrap"]["replicates"] = 50
    table = family_metrics(predictions[~benzene], predictions[benzene], families, esol.set_index("row_id")["murcko"], small)
    assert "R2" not in table.columns
    assert set(table["family"]) == set(config["families"]["named"]) | set(config["families"]["aggregate"])
    assert table.loc[table["model"] == "xgboost", "n"].sum() == 1128
