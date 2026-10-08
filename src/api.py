"""FastAPI application for molecular solubility prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from src.descriptors import DESCRIPTOR_FIELDS, calculate_descriptors
from src.train_model import FEATURE_COLUMNS, load_and_predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "solubility_model.joblib"
FEATURE_PATH = PROJECT_ROOT / "models" / "feature_columns.joblib"


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
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


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


class PredictionResponse(BaseModel):
    """Predicted property and supporting molecular information."""

    canonical_smiles: str
    predicted_log_s: float
    descriptors: dict[str, float]
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
            svg = _molecule_svg(request.smiles)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        return {
            "canonical_smiles": request.smiles,
            "predicted_log_s": float(prediction),
            "descriptors": descriptor_result,
            "svg": svg,
            "disclaimer": "This prediction is an estimate based on molecular descriptors and is not an experimental measurement.",
        }

    return app


app = create_app()
