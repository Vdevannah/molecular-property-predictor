"""FastAPI application for molecular solubility prediction."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.model_selection import train_test_split

from src.descriptors import DESCRIPTOR_FIELDS, calculate_descriptors
from src.train_model import FEATURE_COLUMNS, TARGET_COLUMN, load_and_predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "solubility_model.joblib"
FEATURE_PATH = PROJECT_ROOT / "models" / "feature_columns.joblib"
ESOL_PATH = PROJECT_ROOT / "data" / "esol.csv"
DESCRIPTOR_DATA_PATH = PROJECT_ROOT / "data" / "esol_descriptors.csv"
PHASE3_RANDOM_STATE = 42
PHASE3_TEST_SIZE = 0.20


def _load_model_assets() -> tuple[Any, list[str]]:
    """Load the saved Random Forest model and its feature order."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model file not found: {MODEL_PATH}")
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Saved feature list file not found: {FEATURE_PATH}")

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    if not isinstance(features, list) or not features:
        raise ValueError("Saved feature list is missing or empty.")
    if set(features) != set(FEATURE_COLUMNS):
        raise ValueError("Saved feature column order does not match the trained model.")
    if getattr(model, "n_features_in_", None) != len(features):
        raise ValueError("Saved model feature count does not match its feature list.")
    return model, features


def _molecule_svg(smiles: str) -> str:
    """Render a molecule as a publication-safe SVG string."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Input did not produce a valid RDKit molecule.")

    drawer = rdMolDraw2D.MolDraw2DSVG(500, 350)
    options = drawer.drawOptions()
    options.padding = 0.05
    options.fixedBondLength = 35
    options.addAtomIndices = False
    options.legendFontSize = 12
    options.clearBackground = True
    options.setBackgroundColour((0, 0, 0, 0))
    options.setSymbolColour((0.9, 0.9, 0.9))
    options.setAtomPalette({
        6: (0.96, 0.96, 0.94),
        7: (0.32, 0.65, 1.0),
        8: (1.0, 0.38, 0.38),
        9: (0.65, 0.90, 1.0),
        15: (1.0, 0.65, 0.20),
        16: (1.0, 0.78, 0.28),
        17: (0.40, 0.85, 0.40),
        35: (0.75, 0.35, 0.10),
        53: (0.55, 0.25, 0.82),
    })
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


@lru_cache(maxsize=1)
def _load_esol_dataset() -> pd.DataFrame:
    """Load the canonical ESOL dataset and derive the exact Phase 3 split."""
    dataset = pd.read_csv(ESOL_PATH)
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"ESOL dataset is missing the target column: {TARGET_COLUMN}")
    if "smiles" not in dataset.columns:
        raise ValueError("ESOL dataset is missing the SMILES column.")

    dataset = dataset.copy()
    dataset["canonical_smiles"] = dataset["smiles"].map(_canonical_smiles)
    return dataset


@lru_cache(maxsize=1)
def _load_descriptor_dataset() -> pd.DataFrame:
    """Load the processed ESOL descriptor dataset used by the trained model."""
    dataset = pd.read_csv(DESCRIPTOR_DATA_PATH)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN, "smiles"]
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Processed dataset is missing required columns: {', '.join(missing_columns)}")
    return dataset


def _canonical_smiles(smiles: str) -> str:
    """Canonicalize a SMILES string while preserving stereochemistry."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True)


def _phase3_split_membership() -> tuple[set[int], set[int]]:
    """Return the exact row indices assigned to Phase 3 training and testing sets."""
    dataset = _load_descriptor_dataset()
    train_index, test_index = train_test_split(
        dataset.index,
        test_size=PHASE3_TEST_SIZE,
        random_state=PHASE3_RANDOM_STATE,
    )
    return set(train_index), set(test_index)


def _experimental_reference(smiles: str) -> dict[str, Any]:
    """Find an ESOL experimental measurement without silently resolving conflicts."""
    dataset = _load_esol_dataset()
    canonical = _canonical_smiles(smiles)
    matches = dataset[dataset["canonical_smiles"] == canonical].copy()

    if matches.empty:
        return {
            "status": "not_available",
            "experimental_log_s": None,
            "absolute_error": None,
            "esol_source": None,
            "compound_id": None,
            "phase3_split": "not_in_phase3_split",
            "conflicting_measurements": [],
            "notes": "No exact experimental reference is available in the ESOL dataset.",
        }

    unique_values = matches[TARGET_COLUMN].dropna().unique().tolist()
    training_rows, testing_rows = _phase3_split_membership()
    split_labels = {index: "training" for index in training_rows}
    split_labels.update({index: "testing" for index in testing_rows})
    actual_split = "not_in_phase3_split"
    for record_index in matches.index:
        split_value = split_labels.get(record_index)
        if split_value is not None:
            actual_split = split_value
            break

    if len(unique_values) == 1:
        experimental_value = float(unique_values[0])
        records = matches.iloc[0]
        return {
            "status": "available",
            "experimental_log_s": experimental_value,
            "absolute_error": abs(float(records[TARGET_COLUMN]) - float(experimental_value)),
            "esol_source": "ESOL (Delaney) dataset",
            "compound_id": records["Compound ID"],
            "phase3_split": actual_split,
            "conflicting_measurements": [],
            "notes": "Exact canonical SMILES match found in the ESOL dataset.",
        }

    conflicting_values = [float(value) for value in sorted(set(unique_values))]
    return {
        "status": "conflicting_measurements",
        "experimental_log_s": None,
        "absolute_error": None,
        "esol_source": "ESOL (Delaney) dataset",
        "compound_id": None,
        "phase3_split": actual_split,
        "conflicting_measurements": conflicting_values,
        "notes": "Multiple distinct experimental values are associated with this canonical SMILES. The measurements are not silently combined or arbitrarily selected.",
    }


def _molecular_formula(smiles: str) -> str:
    """Calculate the molecular formula of a valid RDKit structure."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Input did not produce a valid RDKit molecule.")
    return rdMolDescriptors.CalcMolFormula(molecule)


def _applicability_domain(smiles: str, descriptors: dict[str, float]) -> dict[str, Any]:
    """Assess descriptor ranges using the Phase 3 training set only."""
    training_data = _load_descriptor_dataset().iloc[list(_phase3_split_membership()[0])]
    descriptor_ranges = {}
    warnings: list[str] = []

    for descriptor in FEATURE_COLUMNS:
        minimum = float(training_data[descriptor].min())
        maximum = float(training_data[descriptor].max())
        descriptor_ranges[descriptor] = {"minimum": minimum, "maximum": maximum}
        value = float(descriptors[descriptor])
        if value < minimum or value > maximum:
            warnings.append(descriptor)

    return {
        "status": "Within descriptor ranges" if not warnings else "Outside descriptor ranges",
        "warnings": warnings,
        "descriptor_ranges": [
            {"descriptor": descriptor, **limits}
            for descriptor, limits in descriptor_ranges.items()
        ],
        "training_molecules": int(len(training_data)),
        "source": "Phase 3 random train split",
        "notes": "This assessment uses training-set descriptor ranges only. It is not calibrated confidence, a probability of correctness, or prediction uncertainty.",
    }


class PredictRequest(BaseModel):
    """Request schema for a molecular property prediction."""

    smiles: str = Field(..., min_length=1, description="SMILES string for the molecule")

    @field_validator("smiles")
    @classmethod
    def validate_smiles(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("SMILES is required.")
        molecule = Chem.MolFromSmiles(cleaned)
        if molecule is None:
            raise ValueError("SMILES is not valid for an RDKit molecule.")
        return Chem.MolToSmiles(molecule, canonical=True)


class ExperimentalReference(BaseModel):
    """Observed ESOL measurement and its evaluation status."""

    status: str
    experimental_log_s: float | None
    absolute_error: float | None
    esol_source: str | None
    compound_id: str | None
    phase3_split: str
    conflicting_measurements: list[float]
    notes: str


class ApplicabilityDomain(BaseModel):
    """Descriptor-range assessment derived from the Phase 3 training set."""

    status: str
    warnings: list[str]
    descriptor_ranges: list[dict[str, Any]]
    training_molecules: int
    source: str
    notes: str


class PredictionResponse(BaseModel):
    """Predicted property and supporting molecular information."""

    canonical_smiles: str
    predicted_log_s: float
    descriptors: dict[str, float]
    molecular_formula: str
    experimental_reference: ExperimentalReference
    applicability_domain: ApplicabilityDomain
    svg: str
    disclaimer: str


def create_app(model: Any | None = None) -> FastAPI:
    """Create the FastAPI application and attach a single model instance."""
    app = FastAPI(title="Molecular Property Predictor", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5175",
            "http://127.0.0.1:5175",
            "http://localhost:5176",
            "http://127.0.0.1:5176",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    model_state = model
    if model_state is None:
        try:
            model_state = _load_model_assets()[0]
        except (FileNotFoundError, ValueError):
            model_state = None

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": model_state is not None,
        }

    @app.post("/api/predict", response_model=PredictionResponse)
    def predict(request: PredictRequest) -> dict[str, Any]:
        if model_state is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction model is not available on this server.",
            )

        descriptor_result = calculate_descriptors(request.smiles)
        if descriptor_result is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="SMILES could not be converted into a valid molecule.",
            )

        feature_values = pd.DataFrame([descriptor_result], columns=FEATURE_COLUMNS)
        prediction = model_state.predict(feature_values[FEATURE_COLUMNS])[0]

        try:
            molecular_formula = _molecular_formula(request.smiles)
            svg = _molecule_svg(request.smiles)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        experimental_reference = _experimental_reference(request.smiles)
        if experimental_reference["status"] == "available":
            experimental_reference["absolute_error"] = abs(float(prediction) - float(experimental_reference["experimental_log_s"]))
        applicability_domain = _applicability_domain(request.smiles, descriptor_result)

        return {
            "canonical_smiles": request.smiles,
            "predicted_log_s": float(prediction),
            "descriptors": descriptor_result,
            "molecular_formula": molecular_formula,
            "experimental_reference": experimental_reference,
            "applicability_domain": applicability_domain,
            "svg": svg,
            "disclaimer": "This prediction is an estimate based on molecular descriptors and is not an experimental measurement.",
        }

    return app


app = create_app()
