"""Train and evaluate a baseline and random-forest solubility model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "MolWt",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotBonds",
]
TARGET_COLUMN = "measured log solubility in mols per litre"


def validate_training_data(dataset: pd.DataFrame) -> None:
    """Validate that a dataset contains usable feature and target columns."""
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    features = dataset[FEATURE_COLUMNS]
    target = dataset[TARGET_COLUMN]

    if not features.apply(pd.api.types.is_numeric_dtype).all():
        raise ValueError("Feature columns must contain numeric values.")

    if not pd.api.types.is_numeric_dtype(target):
        raise ValueError("Target column must contain numeric values.")

    numeric_data = features.copy()
    numeric_data[TARGET_COLUMN] = target
    if not np.isfinite(numeric_data.to_numpy(dtype=float)).all():
        raise ValueError("Feature and target columns must contain finite values only.")


def split_dataset(
    dataset: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a validated dataset into training and testing sets."""
    validate_training_data(dataset)
    train_data, test_data = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
    )
    return train_data, test_data


def train_models(train_data: pd.DataFrame) -> dict[str, Any]:
    """Train the baseline and Random Forest regression models."""
    validate_training_data(train_data)
    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]

    baseline_model = DummyRegressor(strategy="mean")
    baseline_model.fit(X_train, y_train)

    random_forest_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    random_forest_model.fit(X_train, y_train)

    return {
        "baseline": baseline_model,
        "random_forest": random_forest_model,
    }


def evaluate_model(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, and R² for a model's predictions."""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    if y_true_array.shape != y_pred_array.shape:
        raise ValueError("True and predicted values must have the same shape.")
    if y_true_array.size == 0:
        raise ValueError("No values are available for evaluation.")

    return {
        "MAE": float(mean_absolute_error(y_true_array, y_pred_array)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "R2": float(r2_score(y_true_array, y_pred_array)),
    }


def save_model_artifacts(
    model: Any,
    feature_columns: list[str],
    model_path: str | Path,
    feature_path: str | Path,
) -> None:
    """Persist a trained model and its feature-column order."""
    model_file = Path(model_path)
    features_file = Path(feature_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    features_file.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_file)
    joblib.dump(feature_columns, features_file)


def load_and_predict(
    model_path: str | Path,
    feature_path: str | Path,
    dataset: pd.DataFrame,
) -> np.ndarray:
    """Reload a saved model and predict using the stored feature order."""
    loaded_model = joblib.load(model_path)
    saved_features = joblib.load(feature_path)
    missing_features = [column for column in saved_features if column not in dataset.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing required prediction columns: {', '.join(missing_features)}")

    prediction_data = dataset[saved_features]
    return loaded_model.predict(prediction_data)


def _build_metrics(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> dict[str, Any]:
    """Train both models and calculate their evaluation metrics."""
    models = train_models(train_data)
    predictions = {
        "baseline": models["baseline"].predict(test_data[FEATURE_COLUMNS]),
        "random_forest": models["random_forest"].predict(test_data[FEATURE_COLUMNS]),
    }
    target = test_data[TARGET_COLUMN].to_numpy()

    return {
        "dataset": {
            "training_molecules": int(len(train_data)),
            "testing_molecules": int(len(test_data)),
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "test_size": 0.20,
            "random_state": 42,
        },
        "metrics": {
            name: evaluate_model(target, prediction)
            for name, prediction in predictions.items()
        },
    }


def main() -> None:
    """Load the descriptor dataset, train models, and save evaluation artifacts."""
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "esol_descriptors.csv"
    metrics_path = project_root / "results" / "phase3_metrics.json"
    model_path = project_root / "models" / "solubility_model.joblib"
    feature_path = project_root / "models" / "feature_columns.joblib"

    dataset = pd.read_csv(dataset_path)
    train_data, test_data = split_dataset(dataset, test_size=0.20, random_state=42)
    metrics = _build_metrics(train_data, test_data)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    models = train_models(train_data)
    save_model_artifacts(models["random_forest"], FEATURE_COLUMNS, model_path, feature_path)

    print(f"Training molecules: {len(train_data)}")
    print(f"Testing molecules: {len(test_data)}")
    print("Model metrics:")
    for model_name, metric_values in metrics["metrics"].items():
        print(f"  {model_name}: MAE={metric_values['MAE']:.6f}, RMSE={metric_values['RMSE']:.6f}, R²={metric_values['R2']:.6f}")

    reloaded_predictions = load_and_predict(model_path, feature_path, test_data)
    print(f"Reloaded model predictions: {len(reloaded_predictions)}")


if __name__ == "__main__":
    main()
