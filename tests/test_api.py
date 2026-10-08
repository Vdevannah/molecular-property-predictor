from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from rdkit import Chem

from src.api import app, create_app
from src.train_model import FEATURE_COLUMNS, TARGET_COLUMN


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint_returns_loaded_status(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


@pytest.mark.parametrize(
    "origin",
    ["http://localhost:5175", "http://127.0.0.1:5175"],
)
def test_cors_allows_local_vite_frontend(origin: str) -> None:
    client = TestClient(app, headers={"Origin": origin})
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_predict_valid_smiles_returns_descriptors_and_svg() -> None:
    client = TestClient(app)
    response = client.post("/api/predict", json={"smiles": "CCO"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["canonical_smiles"] == "CCO"
    assert payload["predicted_log_s"] == pytest.approx(0.9725, rel=1e-6)
    assert list(payload["descriptors"]) == FEATURE_COLUMNS
    assert payload["descriptors"]["MolWt"] > 0.0
    assert payload["molecular_formula"] == "C2H6O"
    assert payload["experimental_reference"]["experimental_log_s"] == pytest.approx(1.1)
    assert payload["applicability_domain"]["training_molecules"] == 902
    assert payload["svg"]
    assert "<svg" in payload["svg"]
    assert "estimate" in payload["disclaimer"].lower()


def test_predict_rejects_missing_smiles(client: TestClient) -> None:
    response = client.post("/api/predict", json={})

    assert response.status_code == 422


def test_predict_rejects_empty_smiles(client: TestClient) -> None:
    response = client.post("/api/predict", json={"smiles": "   "})

    assert response.status_code == 422


def test_predict_rejects_invalid_smiles(client: TestClient) -> None:
    response = client.post("/api/predict", json={"smiles": "not-a-molecule"})

    assert response.status_code == 422


def test_exact_experimental_reference_is_returned_for_training_molecule() -> None:
    client = TestClient(app)
    response = client.post("/api/predict", json={"smiles": "CCO"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["experimental_reference"]["status"] == "available"
    assert payload["experimental_reference"]["experimental_log_s"] == pytest.approx(1.1)
    assert payload["experimental_reference"]["absolute_error"] == pytest.approx(abs(0.9725 - 1.1))
    assert payload["experimental_reference"]["esol_source"] == "ESOL (Delaney) dataset"
    assert payload["experimental_reference"]["phase3_split"] == "training"
    assert payload["experimental_reference"]["compound_id"] == "Ethanol"


def test_conflicting_duplicate_measurements_are_not_silently_resolved() -> None:
    client = TestClient(app)
    response = client.post("/api/predict", json={"smiles": "CC12CCC(CC1)C(C)(C)O2"})

    assert response.status_code == 200
    ref = response.json()["experimental_reference"]
    assert ref["status"] == "conflicting_measurements"
    assert ref["experimental_log_s"] is None
    assert ref["absolute_error"] is None
    assert ref["conflicting_measurements"] == pytest.approx([-1.74, -1.64])
    assert ref["phase3_split"] == "training"


def test_missing_experimental_reference_is_explicit() -> None:
    client = TestClient(app)
    response = client.post("/api/predict", json={"smiles": "CCN"})

    assert response.status_code == 200
    ref = response.json()["experimental_reference"]
    assert ref["status"] == "not_available"
    assert ref["experimental_log_s"] is None
    assert ref["absolute_error"] is None
    assert ref["esol_source"] is None
    assert ref["phase3_split"] == "not_in_phase3_split"


def test_predict_reports_molecular_formula() -> None:
    client = TestClient(app)
    for smiles, expected in {
        "CCO": "C2H6O",
        "c1ccccc1": "C6H6",
        "CC(=O)Oc1ccccc1C(=O)O": "C9H8O4",
        "C1CN2CC3=CCOC4CC(=O)N5C6C4C3CC2C61C7=CC=CC=C75": "C21H22N2O2",
    }.items():
        response = client.post("/api/predict", json={"smiles": smiles})
        assert response.status_code == 200
        assert response.json()["molecular_formula"] == expected


def test_applicability_domain_reports_training_ranges() -> None:
    client = TestClient(app)
    response = client.post("/api/predict", json={"smiles": "CCO"})

    assert response.status_code == 200
    domain = response.json()["applicability_domain"]
    assert domain["status"] == "Within descriptor ranges"
    assert "MolWt" not in domain["warnings"]
    assert all("descriptor" in entry for entry in domain["descriptor_ranges"])
    assert domain["training_molecules"] == 902
    assert domain["source"] == "Phase 3 random train split"


def test_predict_uses_existing_descriptor_values() -> None:
    client = TestClient(app)
    response = client.post("/api/predict", json={"smiles": "CCO"})
    payload = response.json()["descriptors"]

    molecule = Chem.MolFromSmiles("CCO")
    assert payload["MolWt"] == pytest.approx(Chem.Descriptors.MolWt(molecule))
    assert payload["LogP"] == pytest.approx(Chem.Descriptors.MolLogP(molecule))
    assert payload["TPSA"] == pytest.approx(Chem.Descriptors.TPSA(molecule))
    assert payload["HBD"] == pytest.approx(Chem.Descriptors.NumHDonors(molecule))
    assert payload["HBA"] == pytest.approx(Chem.Descriptors.NumHAcceptors(molecule))
    assert payload["RotBonds"] == pytest.approx(Chem.Descriptors.NumRotatableBonds(molecule))


def test_model_unavailable_returns_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.api._load_model_assets", lambda: (_ for _ in ()).throw(FileNotFoundError("missing model")))
    app_without_model = create_app(model=None)
    client = TestClient(app_without_model)

    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False

    response = client.post("/api/predict", json={"smiles": "CCO"})
    assert response.status_code == 503
    assert "not available" in response.json()["detail"].lower()


def test_svg_generation_is_valid_svg() -> None:
    client = TestClient(app)
    response = client.post("/api/predict", json={"smiles": "CCO"})
    svg = response.json()["svg"]

    assert "<svg" in svg
    assert "http://www.w3.org/2000/svg" in svg
    assert "<path" in svg
    assert svg.rstrip().endswith("</svg>")
