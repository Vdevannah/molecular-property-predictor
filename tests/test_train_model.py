from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.train_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    evaluate_model,
    save_model_artifacts,
    split_dataset,
    train_models,
    validate_training_data,
)


@pytest.fixture
def synthetic_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    points = 30
    feature_values = rng.random((points, 6), dtype=np.float64)
    target = (feature_values[:, 0] * 4.0) - 2.0 + rng.normal(0, 0.1, points)
    dataset = pd.DataFrame(feature_values, columns=FEATURE_COLUMNS)
    dataset[TARGET_COLUMN] = target
    return dataset


def test_required_feature_columns_are_selected_correctly(synthetic_dataset):
    features = synthetic_dataset[FEATURE_COLUMNS]
    target = synthetic_dataset[TARGET_COLUMN]

    assert list(features.columns) == FEATURE_COLUMNS
    assert target.name == TARGET_COLUMN


def test_experimental_log_s_is_the_target(synthetic_dataset):
    assert TARGET_COLUMN in synthetic_dataset.columns
    assert "ESOL predicted log solubility in mols per litre" not in FEATURE_COLUMNS
    assert pd.api.types.is_numeric_dtype(synthetic_dataset[TARGET_COLUMN])


def test_training_and_testing_sets_do_not_overlap(synthetic_dataset):
    train_df, test_df = split_dataset(synthetic_dataset, test_size=0.20, random_state=42)

    train_indices = train_df.index
    test_indices = test_df.index
    assert set(train_indices).isdisjoint(test_indices)
    assert len(train_df) + len(test_df) == len(synthetic_dataset)


def test_both_models_can_train_and_generate_predictions(synthetic_dataset):
    train_df, test_df = split_dataset(synthetic_dataset, test_size=0.20, random_state=42)
    models = train_models(train_df)

    assert isinstance(models["baseline"], DummyRegressor)
    assert isinstance(models["random_forest"], RandomForestRegressor)

    baseline_pred = models["baseline"].predict(test_df[FEATURE_COLUMNS])
    forest_pred = models["random_forest"].predict(test_df[FEATURE_COLUMNS])

    assert baseline_pred.shape == (len(test_df),)
    assert forest_pred.shape == (len(test_df),)
    assert np.all(np.isfinite(baseline_pred))
    assert np.all(np.isfinite(forest_pred))


def test_metrics_are_calculated_correctly(synthetic_dataset):
    train_df, test_df = split_dataset(synthetic_dataset, test_size=0.20, random_state=42)
    models = train_models(train_df)
    predictions = {
        "baseline": models["baseline"].predict(test_df[FEATURE_COLUMNS]),
        "random_forest": models["random_forest"].predict(test_df[FEATURE_COLUMNS]),
    }
    actual = test_df[TARGET_COLUMN].to_numpy()

    metrics = {
        name: evaluate_model(actual, predictions[name])
        for name in predictions
    }

    assert set(metrics["baseline"]) == {"MAE", "RMSE", "R2"}
    assert metrics["baseline"]["MAE"] == pytest.approx(
        mean_absolute_error(actual, predictions["baseline"])
    )
    assert metrics["baseline"]["RMSE"] == pytest.approx(
        np.sqrt(mean_squared_error(actual, predictions["baseline"]))
    )
    assert metrics["baseline"]["R2"] == pytest.approx(r2_score(actual, predictions["baseline"]))
    assert metrics["random_forest"]["MAE"] >= 0
    assert metrics["random_forest"]["RMSE"] >= 0
    assert metrics["random_forest"]["R2"] <= 1


def test_saved_model_can_be_loaded_and_used_for_prediction(synthetic_dataset, tmp_path):
    train_df, test_df = split_dataset(synthetic_dataset, test_size=0.20, random_state=42)
    models = train_models(train_df)
    model_path = tmp_path / "model.joblib"
    feature_path = tmp_path / "features.joblib"

    save_model_artifacts(models["random_forest"], FEATURE_COLUMNS, model_path, feature_path)

    loaded_model = joblib.load(model_path)
    loaded_features = joblib.load(feature_path)
    predictions = loaded_model.predict(test_df[loaded_features])

    assert loaded_features == FEATURE_COLUMNS
    assert predictions.shape == (len(test_df),)


def test_missing_or_nonfinite_data_produces_validation_error():
    dataset = pd.DataFrame(
        {
            "MolWt": [50.0, 60.0],
            "LogP": [1.0, np.inf],
            "TPSA": [20.0, 25.0],
            "HBD": [1, 1],
            "HBA": [2, 2],
            "RotBonds": [0, 1],
            TARGET_COLUMN: [0.0, -1.0],
        }
    )

    with pytest.raises(ValueError, match="finite"):
        validate_training_data(dataset)

    dataset_missing = dataset.drop(columns=TARGET_COLUMN)
    with pytest.raises(ValueError, match="required columns"):
        validate_training_data(dataset_missing)
