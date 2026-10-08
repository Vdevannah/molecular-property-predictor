from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from rdkit import Chem

from src.api import app, create_app
from src.train_model import FEATURE_COLUMNS


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
