import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.model_experiments import (
    EXPERIMENT_MODEL_NAMES,
    build_experimental_models,
    fit_experimental_models,
    predict_experimental_models,
)
from src.train_model import FEATURE_COLUMNS, TARGET_COLUMN, split_dataset, train_models


@pytest.fixture
def synthetic_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    points = 40
    feature_values = rng.random((points, 6), dtype=np.float64)
    target = (feature_values[:, 0] * 4.0) - 2.0 + rng.normal(0, 0.1, points)
    dataset = pd.DataFrame(feature_values, columns=FEATURE_COLUMNS)
    dataset[TARGET_COLUMN] = target
    return dataset


def test_all_experimental_models_are_constructed():
    models = build_experimental_models()

    assert tuple(models) == EXPERIMENT_MODEL_NAMES
    assert isinstance(models["baseline"], DummyRegressor)
    assert isinstance(models["random_forest"], RandomForestRegressor)
    assert isinstance(models["lasso"], Pipeline)
    assert isinstance(models["xgboost"], XGBRegressor)


def test_lasso_pipeline_scales_before_regression():
    lasso = build_experimental_models()["lasso"]

    assert [name for name, _ in lasso.steps] == ["scaler", "lasso"]
    assert isinstance(lasso.named_steps["scaler"], StandardScaler)
    assert isinstance(lasso.named_steps["lasso"], Lasso)


def test_models_use_fixed_seed_and_single_worker():
    models = build_experimental_models(random_state=7)

    assert models["random_forest"].random_state == 7
    assert models["xgboost"].random_state == 7
    assert models["random_forest"].n_jobs == 1
    assert models["xgboost"].n_jobs == 1


def test_all_models_fit_and_predict_finite_values(synthetic_dataset):
    train_df, test_df = split_dataset(synthetic_dataset)
    models = fit_experimental_models(train_df)
    predictions = predict_experimental_models(models, test_df)

    assert set(predictions) == set(EXPERIMENT_MODEL_NAMES)
    for prediction in predictions.values():
        assert prediction.shape == (len(test_df),)
        assert np.isfinite(prediction).all()


def test_fitted_models_record_descriptor_feature_names(synthetic_dataset):
    models = fit_experimental_models(synthetic_dataset)

    for model in models.values():
        assert model.n_features_in_ == len(FEATURE_COLUMNS)
        assert list(model.feature_names_in_) == FEATURE_COLUMNS


def test_predictions_ignore_column_order_and_extra_columns(synthetic_dataset):
    models = fit_experimental_models(synthetic_dataset)
    shuffled = synthetic_dataset[list(reversed(FEATURE_COLUMNS))].copy()
    shuffled["smiles"] = "C"

    original = predict_experimental_models(models, synthetic_dataset)
    reordered = predict_experimental_models(models, shuffled)

    for name in EXPERIMENT_MODEL_NAMES:
        np.testing.assert_array_equal(original[name], reordered[name])


def test_missing_feature_column_raises_clear_error(synthetic_dataset):
    models = fit_experimental_models(synthetic_dataset)

    with pytest.raises(ValueError, match="LogP"):
        predict_experimental_models(models, synthetic_dataset.drop(columns=["LogP"]))


def test_repeated_fits_are_reproducible(synthetic_dataset):
    first = predict_experimental_models(fit_experimental_models(synthetic_dataset), synthetic_dataset)
    second = predict_experimental_models(fit_experimental_models(synthetic_dataset), synthetic_dataset)

    for name in EXPERIMENT_MODEL_NAMES:
        np.testing.assert_array_equal(first[name], second[name])


def test_experimental_random_forest_matches_production_configuration(synthetic_dataset):
    train_df, test_df = split_dataset(synthetic_dataset)
    production = train_models(train_df)
    experimental = fit_experimental_models(train_df)

    np.testing.assert_allclose(
        experimental["random_forest"].predict(test_df[FEATURE_COLUMNS]),
        production["random_forest"].predict(test_df[FEATURE_COLUMNS]),
    )
    np.testing.assert_allclose(
        experimental["baseline"].predict(test_df[FEATURE_COLUMNS]),
        production["baseline"].predict(test_df[FEATURE_COLUMNS]),
    )


def test_production_training_still_returns_only_existing_models(synthetic_dataset):
    models = train_models(synthetic_dataset)

    assert set(models) == {"baseline", "random_forest"}


def test_invalid_training_data_is_rejected():
    dataset = pd.DataFrame({column: [1.0, 2.0] for column in FEATURE_COLUMNS})
    dataset[TARGET_COLUMN] = [1.0, np.nan]

    with pytest.raises(ValueError, match="finite"):
        fit_experimental_models(dataset)
