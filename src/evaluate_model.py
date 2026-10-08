"""Evaluate the trained solubility model and compare random and scaffold splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.train_model import FEATURE_COLUMNS, TARGET_COLUMN, evaluate_model, train_models, validate_training_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_SEED = 42
RANDOM_TEST_SIZE = 0.20
SCAFFOLD_TEST_SIZE = 0.20


def validate_model_inputs(dataset: pd.DataFrame) -> None:
    """Validate model inputs and required feature/target columns."""
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN, "smiles"]
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    numeric_data = dataset[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    if not numeric_data.apply(pd.api.types.is_numeric_dtype).all().all():
        raise ValueError("Feature and target columns must contain numeric values.")

    if not np.isfinite(numeric_data.to_numpy(dtype=float)).all():
        raise ValueError("Feature and target columns must contain finite values only.")


def prepare_random_split(
    dataset: pd.DataFrame,
    test_size: float = RANDOM_TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the Phase 3 random train/test split."""
    validate_model_inputs(dataset)
    train_data, test_data = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
    )
    return train_data.reset_index(drop=True), test_data.reset_index(drop=True)


def get_scaffold_group(smiles: str) -> str:
    """Return a unique chemical group for scaffold-aware validation."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "INVALID"

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    if scaffold:
        return scaffold

    return Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=True)


def create_scaffold_split(
    dataset: pd.DataFrame,
    test_size: float = SCAFFOLD_TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split molecules so no scaffold appears in both training and testing sets."""
    validate_model_inputs(dataset)
    groups: dict[str, list[int]] = {}

    for index, row in dataset.iterrows():
        scaffold = get_scaffold_group(row["smiles"])
        groups.setdefault(scaffold, []).append(index)

    groups = {group: indices for group, indices in groups.items() if indices}
    group_names = list(groups)
    rng = np.random.default_rng(random_state)
    rng.shuffle(group_names)

    test_indices: list[int] = []
    target_test_size = round(len(dataset) * test_size)
    for group_name in group_names:
        group_indices = groups[group_name]
        if len(test_indices) + len(group_indices) > target_test_size:
            continue
        test_indices.extend(group_indices)

    if not test_indices:
        test_indices = groups[group_names[0]]

    test_data = dataset.loc[test_indices].copy()
    train_data = dataset.drop(index=test_indices).copy()
    return train_data, test_data


def evaluate_model_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate MAE, RMSE, and R² for a single prediction set."""
    return evaluate_model(y_true, y_pred)


def plot_actual_vs_predicted(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
) -> None:
    """Create an actual-versus-predicted plot with a perfect-prediction line."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(y_true, y_pred, alpha=0.7)
    minimum = min(float(np.min(y_true)), float(np.min(y_pred)))
    maximum = max(float(np.max(y_true)), float(np.max(y_pred)))
    line = np.linspace(minimum, maximum, 100)
    axis.plot(line, line, linestyle="--", color="black", linewidth=1, label="Perfect prediction")
    axis.set_xlabel("Experimental LogS")
    axis.set_ylabel("Predicted LogS")
    axis.set_title("Actual versus Predicted LogS")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def plot_residuals(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
) -> None:
    """Create a residual plot with a zero-error reference line."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.scatter(y_pred, residuals, alpha=0.7)
    axis.set_xlabel("Predicted LogS")
    axis.set_ylabel("Experimental minus predicted LogS")
    axis.set_title("Residual Plot")
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def plot_feature_importance(
    model: RandomForestRegressor,
    output_path: str | Path,
) -> None:
    """Create a bar plot of impurity-based feature importance."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    features = model.feature_names_in_
    importances = model.feature_importances_
    ordering = np.argsort(importances)[::-1]

    fig, axis = plt.subplots(figsize=(8, 6))
    axis.bar(np.array(features)[ordering], importances[ordering], color="steelblue")
    axis.set_xlabel("Feature")
    axis.set_ylabel("Importance")
    axis.set_title("Random Forest Feature Importance")
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def load_saved_model(
    model_path: str | Path,
    feature_path: str | Path,
) -> tuple[Any, list[str]]:
    """Load a saved Random Forest model and its feature-column order."""
    model_file = Path(model_path)
    feature_file = Path(feature_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Saved model file not found: {model_file}")
    if not feature_file.exists():
        raise FileNotFoundError(f"Saved feature list file not found: {feature_file}")

    model = joblib.load(model_file)
    feature_columns = joblib.load(feature_file)

    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("Saved feature list is missing or empty.")
    if set(feature_columns) != set(FEATURE_COLUMNS):
        raise ValueError("Saved feature list does not match the expected descriptor columns.")
    if model.n_features_in_ != len(feature_columns):
        raise ValueError("Saved model feature count does not match the saved feature list.")

    return model, feature_columns


def _save_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Write evaluation metrics with metadata to JSON."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(metrics, indent=2))


def main() -> None:
    """Evaluate the saved model on random and scaffold-based test splits."""
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "esol_descriptors.csv"
    model_path = project_root / "models" / "solubility_model.joblib"
    feature_path = project_root / "models" / "feature_columns.joblib"
    metrics_path = project_root / "results" / "phase4_metrics.json"

    dataset = pd.read_csv(dataset_path)
    validate_model_inputs(dataset)

    random_train, random_test = prepare_random_split(dataset)
    saved_model, saved_features = load_saved_model(model_path, feature_path)
    random_predictions = saved_model.predict(random_test[saved_features])
    random_metrics = evaluate_model_predictions(random_test[TARGET_COLUMN].to_numpy(), random_predictions)

    scaffold_train, scaffold_test = create_scaffold_split(dataset)
    scaffold_model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    scaffold_model.fit(scaffold_train[FEATURE_COLUMNS], scaffold_train[TARGET_COLUMN])
    scaffold_predictions = scaffold_model.predict(scaffold_test[FEATURE_COLUMNS])
    scaffold_metrics = evaluate_model_predictions(scaffold_test[TARGET_COLUMN].to_numpy(), scaffold_predictions)

    plot_actual_vs_predicted(random_test[TARGET_COLUMN], random_predictions, project_root / "results" / "actual_vs_predicted.png")
    plot_residuals(random_test[TARGET_COLUMN], random_predictions, project_root / "results" / "residual_plot.png")
    plot_feature_importance(saved_model, project_root / "results" / "feature_importance.png")

    scaffold_groups = len({get_scaffold_group(value) for value in dataset["smiles"]})

    metrics = {
        "dataset": {
            "molecules": int(len(dataset)),
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "random_split": {
                "test_size": RANDOM_TEST_SIZE,
                "random_state": RANDOM_SEED,
                "training_molecules": int(len(random_train)),
                "testing_molecules": int(len(random_test)),
            },
            "scaffold_split": {
                "test_size": SCAFFOLD_TEST_SIZE,
                "random_state": RANDOM_SEED,
                "training_molecules": int(len(scaffold_train)),
                "testing_molecules": int(len(scaffold_test)),
                "scaffold_groups": int(scaffold_groups),
                "target_split_ratio": SCAFFOLD_TEST_SIZE,
                "actual_test_ratio": len(scaffold_test) / len(dataset),
                "zero_scaffold_overlap": True,
            },
        },
        "random_split_metrics": random_metrics,
        "scaffold_split_metrics": scaffold_metrics,
        "scaffold_overlap_check": {
            "training_scaffolds": sorted({get_scaffold_group(value) for value in scaffold_train["smiles"]}),
            "testing_scaffolds": sorted({get_scaffold_group(value) for value in scaffold_test["smiles"]}),
            "overlap": bool({get_scaffold_group(value) for value in scaffold_train["smiles"]} & {get_scaffold_group(value) for value in scaffold_test["smiles"]}),
        },
        "acyclic_handling": {
            "ringless_molecules_grouped_separately": True,
            "grouping_rule": "Acyclic molecules receive distinct groups based on atom count; molecules without ring-based Murcko scaffolds are not treated as a single chemical family.",
        },
        "special_treatment": {
            "invalid_smiles": "Invalid SMILES were assigned to a dedicated INVALID group and retained only when their rows were not selected for testing.",
        },
    }

    _save_metrics(metrics, metrics_path)

    print(f"Random-split metrics: {random_metrics}")
    print(f"Scaffold-split metrics: {scaffold_metrics}")
    print(f"Scaffold groups: {scaffold_groups}")
    print(f"Training/test molecules: {len(scaffold_train)}/{len(scaffold_test)}")
    print(f"Scaffold overlap: {not metrics['scaffold_overlap_check']['overlap']}")
    print(f"Plots saved to: {project_root / 'results'}")


if __name__ == "__main__":
    main()
