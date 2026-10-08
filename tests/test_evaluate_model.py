import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.evaluate_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    create_scaffold_split,
    evaluate_model_predictions,
    get_scaffold_group,
    load_saved_model,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residuals,
    prepare_random_split,
    validate_model_inputs,
)
from src.train_model import evaluate_model, train_models

matplotlib.use("Agg")


@pytest.fixture
def synthetic_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 30
    features = rng.random((rows, 6), dtype=np.float64)
    target = features[:, 0] * 4.0 - 2.0 + rng.normal(0, 0.1, rows)
    dataset = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    dataset["smiles"] = ["CCO", "CCN", "CCC", "CCCl", "CCBr"] * 6
    dataset[TARGET_COLUMN] = target
    return dataset


def test_metric_calculation_matches_phase3_equivalent(synthetic_dataset):
    train_data, test_data = prepare_random_split(synthetic_dataset, test_size=0.20, random_state=42)
    models = train_models(train_data)
    predictions = models["random_forest"].predict(test_data[FEATURE_COLUMNS])
    actual = test_data[TARGET_COLUMN].to_numpy()

    metrics = evaluate_model_predictions(actual, predictions)

    assert set(metrics) == {"MAE", "RMSE", "R2"}
    assert metrics["MAE"] == pytest.approx(np.mean(np.abs(actual - predictions)))
    assert metrics["RMSE"] == pytest.approx(np.sqrt(np.mean((actual - predictions) ** 2)))
    assert metrics["R2"] == pytest.approx(1.0 - np.sum((actual - predictions) ** 2) / np.sum((actual - actual.mean()) ** 2))


def test_plot_generation_creates_expected_files(tmp_path, synthetic_dataset):
    train_data, test_data = prepare_random_split(synthetic_dataset, test_size=0.20, random_state=42)
    model = train_models(train_data)["random_forest"]
    actual = test_data[TARGET_COLUMN].to_numpy()
    predictions = model.predict(test_data[FEATURE_COLUMNS])

    actual_path = tmp_path / "actual_vs_predicted.png"
    residual_path = tmp_path / "residual_plot.png"
    feature_path = tmp_path / "feature_importance.png"

    plot_actual_vs_predicted(actual, predictions, actual_path)
    plot_residuals(actual, predictions, residual_path)
    plot_feature_importance(model, feature_path)

    assert actual_path.exists()
    assert residual_path.exists()
    assert feature_path.exists()
    assert actual_path.stat().st_size > 0
    assert residual_path.stat().st_size > 0
    assert feature_path.stat().st_size > 0


def test_scaffold_split_has_no_overlap_and_reproducible(synthetic_dataset):
    train_data, test_data = create_scaffold_split(
        synthetic_dataset,
        test_size=0.20,
        random_state=42,
    )

    train_scaffolds = {get_scaffold_group(value) for value in train_data["smiles"]}
    test_scaffolds = {get_scaffold_group(value) for value in test_data["smiles"]}
    assert train_scaffolds.isdisjoint(test_scaffolds)
    assert set(train_data.index).isdisjoint(test_data.index)
    assert len(train_data) + len(test_data) == len(synthetic_dataset)


def test_acyclic_molecules_are_kept_separate_from_other_groups(synthetic_dataset):
    dataset = synthetic_dataset.copy()
    dataset.loc[0, "smiles"] = "CCO"
    dataset.loc[1, "smiles"] = "CCO"
    dataset.loc[2, "smiles"] = "CCN"
    dataset.loc[3, "smiles"] = "CCN"

    train_data, test_data = create_scaffold_split(dataset, test_size=0.50, random_state=42)
    train_scaffolds = {get_scaffold_group(value) for value in train_data["smiles"]}
    test_scaffolds = {get_scaffold_group(value) for value in test_data["smiles"]}

    assert train_scaffolds.isdisjoint(test_scaffolds)


def test_acyclic_groups_are_unique_per_molecule(synthetic_dataset):
    dataset = synthetic_dataset.copy()
    dataset.loc[0, "smiles"] = "CCO"
    dataset.loc[1, "smiles"] = "CCN"

    groups = [get_scaffold_group(smiles) for smiles in dataset["smiles"]]
    assert groups[0] != groups[1]
    assert groups[0] == get_scaffold_group("CCO")
    assert groups[1] == get_scaffold_group("CCN")
    assert len(set(groups)) == 5


def test_feature_order_is_preserved_when_loading_model(tmp_path, synthetic_dataset):
    model_path = tmp_path / "model.joblib"
    feature_path = tmp_path / "features.joblib"
    model = train_models(prepare_random_split(synthetic_dataset)[0])["random_forest"]
    joblib.dump(model, model_path)
    joblib.dump(FEATURE_COLUMNS, feature_path)

    loaded_model, loaded_features = load_saved_model(model_path, feature_path)
    assert loaded_features == FEATURE_COLUMNS
    assert loaded_model.predict(synthetic_dataset[loaded_features]).shape == (len(synthetic_dataset),)


def test_missing_model_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_saved_model(tmp_path / "missing.joblib", tmp_path / "missing_features.joblib")


def test_validation_rejects_missing_or_nonfinite_data():
    dataset = pd.DataFrame(
        {
            "MolWt": [50.0, 60.0],
            "LogP": [1.0, np.inf],
            "TPSA": [20.0, 25.0],
            "HBD": [1, 1],
            "HBA": [2, 2],
            "RotBonds": [0, 1],
            "smiles": ["CCO", "CCN"],
            TARGET_COLUMN: [0.0, -1.0],
        }
    )
    with pytest.raises(ValueError, match="finite"):
        validate_model_inputs(dataset)

    dataset_missing = dataset.drop(columns=TARGET_COLUMN)
    with pytest.raises(ValueError, match="required columns"):
        validate_model_inputs(dataset_missing)


def test_small_synthetic_dataset_can_be_trained_and_evaluated(synthetic_dataset):
    train_data, test_data = prepare_random_split(synthetic_dataset, test_size=0.20, random_state=42)
    model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
    model.fit(train_data[FEATURE_COLUMNS], train_data[TARGET_COLUMN])
    predictions = model.predict(test_data[FEATURE_COLUMNS])
    metrics = evaluate_model_predictions(test_data[TARGET_COLUMN].to_numpy(), predictions)

    assert set(metrics) == {"MAE", "RMSE", "R2"}
    assert np.all(np.isfinite(list(metrics.values())))
