import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.cross_validation import (
    PARAM_GRIDS,
    BalancedGroupKFold,
    build_groups,
    build_model_specs,
    canonical_identity_group,
    corrected_paired_ttest,
    nested_cross_validate,
    paired_comparisons,
    scaffold_group,
    summarize_folds,
)
from src.train_model import FEATURE_COLUMNS, TARGET_COLUMN


def _grouped_labels() -> np.ndarray:
    """One dominant group (like benzene), several medium groups, and singletons."""
    labels = ["large"] * 18
    for index in range(6):
        labels += [f"medium{index}"] * 3
    labels += [f"single{index}" for index in range(24)]
    return np.array(labels)


@pytest.fixture
def grouped_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    groups = _grouped_labels()
    rng = np.random.default_rng(0)
    features = rng.random((len(groups), len(FEATURE_COLUMNS)))
    features[:, 0] = np.arange(len(groups))  # unique row identifier carried in MolWt
    dataset = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    dataset[TARGET_COLUMN] = features[:, 1] * 3.0 - 1.0 + rng.normal(0, 0.05, len(groups))
    return dataset, groups


def _small_specs() -> dict:
    return {
        "baseline": (DummyRegressor(strategy="mean"), None),
        "lasso": (
            Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(max_iter=10000))]),
            {"lasso__alpha": [0.01, 0.1]},
        ),
    }


class RecordingRegressor(RegressorMixin, BaseEstimator):
    """Records the MolWt row identifiers seen in every fit call."""

    fitted_rows: list[set[int]] = []

    def __init__(self, offset: float = 0.0):
        self.offset = offset

    def fit(self, X, y):
        RecordingRegressor.fitted_rows.append(set(X["MolWt"].astype(int)))
        self.mean_ = float(np.mean(y)) + self.offset
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


# Grouping -----------------------------------------------------------------


def test_canonical_identity_groups_identical_structures_together():
    assert canonical_identity_group("OCC") == canonical_identity_group("CCO ")
    assert canonical_identity_group("CCO") != canonical_identity_group("CCCO")


def test_scaffold_group_combines_benzene_derivatives():
    assert scaffold_group("Cc1ccccc1") == scaffold_group("Oc1ccccc1 ") == "c1ccccc1"
    assert scaffold_group("CCO") != scaffold_group("CCCO")


def test_build_groups_supports_both_protocols():
    dataset = pd.DataFrame({"smiles": ["Cc1ccccc1", "Oc1ccccc1", "CCO", "OCC"]})

    random_groups = build_groups(dataset, "random_grouped")
    scaffold_groups = build_groups(dataset, "scaffold")

    assert random_groups[2] == random_groups[3] and random_groups[0] != random_groups[1]
    assert scaffold_groups[0] == scaffold_groups[1]
    with pytest.raises(ValueError, match="Unknown"):
        build_groups(dataset, "unknown")


# Splitter -----------------------------------------------------------------


def test_splitter_keeps_groups_intact_and_tests_every_row_once():
    groups = _grouped_labels()
    tested = np.zeros(len(groups), dtype=int)

    for train_index, test_index in BalancedGroupKFold(5, random_state=1).split(None, None, groups):
        assert not set(groups[train_index]) & set(groups[test_index])
        assert not set(train_index) & set(test_index)
        tested[test_index] += 1

    assert (tested == 1).all()


def test_largest_group_participates_in_exactly_one_test_fold():
    groups = _grouped_labels()
    folds_containing_large = [
        fold
        for fold, (_, test_index) in enumerate(BalancedGroupKFold(5, random_state=3).split(None, None, groups))
        if "large" in set(groups[test_index])
    ]
    assert len(folds_containing_large) == 1


def test_splitter_balances_fold_sizes_around_large_group():
    groups = _grouped_labels()
    sizes = [len(test) for _, test in BalancedGroupKFold(5, random_state=0).split(None, None, groups)]

    assert max(sizes) == 18  # the dominant group alone sets the largest fold
    remaining = sorted(sizes)[:-1]
    assert max(remaining) - min(remaining) <= 3


def test_splitter_is_reproducible_and_seed_dependent():
    groups = _grouped_labels()

    first = BalancedGroupKFold(5, random_state=7).assign_folds(groups)
    second = BalancedGroupKFold(5, random_state=7).assign_folds(groups)
    other = BalancedGroupKFold(5, random_state=8).assign_folds(groups)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_splitter_rejects_missing_or_too_few_groups():
    with pytest.raises(ValueError, match="groups"):
        list(BalancedGroupKFold(5).split(None, None, None))
    with pytest.raises(ValueError, match="n_splits"):
        list(BalancedGroupKFold(5).split(None, None, np.array(["a", "a", "b"])))


# Nested cross-validation --------------------------------------------------


def test_tuning_never_sees_outer_test_rows(grouped_dataset):
    dataset, groups = grouped_dataset
    RecordingRegressor.fitted_rows = []
    specs = {"recorder": (RecordingRegressor(), {"offset": [0.0, 1.0]})}

    _, assignments = nested_cross_validate(
        dataset, groups, specs, "test", outer_splits=3, inner_splits=3, repeats=1, n_jobs=1
    )

    fits_per_outer_fold = 2 * 3 + 1  # two candidates × three inner folds + refit
    assert len(RecordingRegressor.fitted_rows) == 3 * fits_per_outer_fold
    for fold in range(3):
        outer_test = set(assignments.loc[assignments["fold"] == fold, "row"])
        outer_train = set(range(len(dataset))) - outer_test
        fold_fits = RecordingRegressor.fitted_rows[fold * fits_per_outer_fold : (fold + 1) * fits_per_outer_fold]
        for inner_fit in fold_fits[:-1]:
            assert inner_fit < outer_train  # strict subset: inner validation rows held out
            assert not inner_fit & outer_test
        assert fold_fits[-1] == outer_train  # refit uses the full outer-training fold only


def test_inner_folds_respect_groups(grouped_dataset):
    dataset, groups = grouped_dataset
    RecordingRegressor.fitted_rows = []
    specs = {"recorder": (RecordingRegressor(), {"offset": [0.0]})}

    nested_cross_validate(dataset, groups, specs, "test", outer_splits=3, inner_splits=3, repeats=1, n_jobs=1)

    for fitted in RecordingRegressor.fitted_rows:
        fitted_groups = set(groups[list(fitted)])
        held_out_groups = set(groups[list(set(range(len(groups))) - fitted)])
        assert not fitted_groups & held_out_groups


def test_all_models_share_identical_outer_folds_and_record_counts(grouped_dataset):
    dataset, groups = grouped_dataset
    folds, assignments = nested_cross_validate(
        dataset, groups, _small_specs(), "test", outer_splits=3, inner_splits=3, repeats=2, n_jobs=1
    )

    assert len(folds) == 2 * 3 * 2
    per_fold = folds.groupby(["repeat", "fold"])
    assert (per_fold["n_test"].nunique() == 1).all()
    assert (per_fold["n_train"].nunique() == 1).all()
    assert ((folds["n_train"] + folds["n_test"]) == len(dataset)).all()
    for repeat in range(2):
        rows = assignments.loc[assignments["repeat"] == repeat, "row"]
        assert sorted(rows) == list(range(len(dataset)))
    assert {"RMSE", "MAE", "R2", "n_train_groups", "n_test_groups", "best_params"} <= set(folds.columns)


def test_nested_cross_validation_is_reproducible(grouped_dataset):
    dataset, groups = grouped_dataset
    kwargs = dict(outer_splits=3, inner_splits=3, repeats=2, n_jobs=1)

    first, first_assignments = nested_cross_validate(dataset, groups, _small_specs(), "test", **kwargs)
    second, second_assignments = nested_cross_validate(dataset, groups, _small_specs(), "test", **kwargs)

    pd.testing.assert_frame_equal(first.drop(columns="fit_seconds"), second.drop(columns="fit_seconds"))
    pd.testing.assert_frame_equal(first_assignments, second_assignments)


def test_tuned_models_report_selected_parameters(grouped_dataset):
    dataset, groups = grouped_dataset
    folds, _ = nested_cross_validate(
        dataset, groups, _small_specs(), "test", outer_splits=3, inner_splits=3, repeats=1, n_jobs=1
    )

    lasso = folds[folds["model"] == "lasso"]
    assert lasso["best_params"].str.contains("lasso__alpha").all()
    assert lasso["inner_cv_rmse"].notna().all()
    assert folds.loc[folds["model"] == "baseline", "best_params"].eq("{}").all()


def test_production_model_specs_cover_required_models():
    specs = build_model_specs()

    assert list(specs) == ["baseline", "random_forest_reference", "lasso", "random_forest_tuned", "xgboost"]
    assert specs["baseline"][1] is None and specs["random_forest_reference"][1] is None
    lasso, grid = specs["lasso"]
    assert isinstance(lasso.named_steps["scaler"], StandardScaler)
    assert grid == PARAM_GRIDS["lasso"]
    for name in ("random_forest_reference", "random_forest_tuned", "xgboost"):
        assert specs[name][0].get_params()["n_jobs"] == 1
        assert specs[name][0].get_params()["random_state"] == 42


# Paired statistics --------------------------------------------------------


def test_corrected_ttest_matches_hand_calculation():
    differences = np.array([0.1, 0.2, 0.0, 0.1, 0.1])
    result = corrected_paired_ttest(differences, np.full(5, 80), np.full(5, 20))

    expected_se = np.sqrt((1 / 5 + 20 / 80) * np.var(differences, ddof=1))
    assert result["mean_difference"] == pytest.approx(0.1)
    assert result["corrected_standard_error"] == pytest.approx(expected_se)
    assert result["t_statistic"] == pytest.approx(0.1 / expected_se)
    assert result["ci_lower"] < result["mean_difference"] < result["ci_upper"]


def test_correction_widens_interval_relative_to_naive_test():
    differences = np.array([0.05, 0.02, 0.04, 0.03, 0.06, 0.01])
    result = corrected_paired_ttest(differences, np.full(6, 80), np.full(6, 20))
    naive_se = np.std(differences, ddof=1) / np.sqrt(len(differences))

    assert result["corrected_standard_error"] > naive_se


def test_identical_models_have_zero_difference_and_no_claimed_effect():
    result = corrected_paired_ttest(np.zeros(5), np.full(5, 80), np.full(5, 20))

    assert result["mean_difference"] == 0.0
    assert result["p_value"] == 1.0
    assert result["ci_excludes_zero"] is False


def test_paired_comparisons_and_summary_cover_every_model_pair(grouped_dataset):
    dataset, groups = grouped_dataset
    folds, _ = nested_cross_validate(
        dataset, groups, _small_specs(), "test", outer_splits=3, inner_splits=3, repeats=1, n_jobs=1
    )

    comparisons = paired_comparisons(folds)
    summary = summarize_folds(folds)

    assert set(comparisons["metric"]) == {"RMSE", "MAE"}
    assert len(comparisons) == 2  # one model pair × two metrics
    assert (comparisons["n_folds"] == 3).all()
    assert set(summary["model"]) == {"baseline", "lasso"}


def test_paired_comparisons_reject_mismatched_folds(grouped_dataset):
    dataset, groups = grouped_dataset
    folds, _ = nested_cross_validate(
        dataset, groups, _small_specs(), "test", outer_splits=3, inner_splits=3, repeats=1, n_jobs=1
    )
    folds.loc[folds.index[0], "n_test"] += 1

    with pytest.raises(ValueError, match="identical folds"):
        paired_comparisons(folds)
