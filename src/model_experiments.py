"""Build and fit experimental solubility models separately from the production workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.train_model import FEATURE_COLUMNS, TARGET_COLUMN, evaluate_model, split_dataset, validate_training_data

EXPERIMENT_RANDOM_STATE = 42
EXPERIMENT_N_JOBS = 1
EXPERIMENT_MODEL_NAMES = ("baseline", "random_forest", "lasso", "xgboost")


def build_experimental_models(
    random_state: int = EXPERIMENT_RANDOM_STATE,
    n_jobs: int = EXPERIMENT_N_JOBS,
) -> dict[str, Any]:
    """Return unfitted baseline, Random Forest, Lasso, and XGBoost regressors.

    Each estimator uses a single worker by default so that later cross-validation
    can parallelise across folds without nested thread oversubscription.
    """
    return {
        "baseline": DummyRegressor(strategy="mean"),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "lasso": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lasso", Lasso(alpha=0.01, max_iter=10000)),
            ]
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=n_jobs,
        ),
    }


def fit_experimental_models(
    train_data: pd.DataFrame,
    random_state: int = EXPERIMENT_RANDOM_STATE,
    n_jobs: int = EXPERIMENT_N_JOBS,
) -> dict[str, Any]:
    """Fit every experimental model on the six descriptor features."""
    validate_training_data(train_data)
    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]

    models = build_experimental_models(random_state=random_state, n_jobs=n_jobs)
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def predict_experimental_models(
    models: dict[str, Any],
    dataset: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Predict with each fitted model using the canonical feature order."""
    missing_features = [column for column in FEATURE_COLUMNS if column not in dataset.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing required prediction columns: {', '.join(missing_features)}")

    prediction_data = dataset[FEATURE_COLUMNS]
    return {name: np.asarray(model.predict(prediction_data), dtype=float) for name, model in models.items()}


def main() -> None:
    """Fit experimental models on the Phase 3 split and report holdout metrics.

    These metrics are a sanity check only; they are not used for model selection,
    and no model artifacts are saved.
    """
    dataset_path = PROJECT_ROOT / "data" / "esol_descriptors.csv"
    metrics_path = PROJECT_ROOT / "results" / "phase6b_holdout_metrics.json"

    dataset = pd.read_csv(dataset_path)
    train_data, test_data = split_dataset(dataset, test_size=0.20, random_state=42)
    models = fit_experimental_models(train_data)
    predictions = predict_experimental_models(models, test_data)
    target = test_data[TARGET_COLUMN].to_numpy()

    metrics = {
        "dataset": {
            "training_molecules": int(len(train_data)),
            "testing_molecules": int(len(test_data)),
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "split": "Phase 3 random split",
            "test_size": 0.20,
            "random_state": 42,
        },
        "note": "Single-split sanity check with untuned hyperparameters; not a model comparison.",
        "metrics": {name: evaluate_model(target, prediction) for name, prediction in predictions.items()},
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Training molecules: {len(train_data)}")
    print(f"Testing molecules: {len(test_data)}")
    print("Experimental model metrics (Phase 3 split, untuned):")
    for model_name, metric_values in metrics["metrics"].items():
        print(f"  {model_name}: MAE={metric_values['MAE']:.6f}, RMSE={metric_values['RMSE']:.6f}, R²={metric_values['R2']:.6f}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
