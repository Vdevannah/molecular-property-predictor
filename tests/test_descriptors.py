from pathlib import Path

import pandas as pd
import pytest

from src.descriptors import calculate_descriptors, process_dataset


@pytest.fixture
def esol_dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "esol.csv"


def test_ethanol_descriptors_are_valid():
    descriptors = calculate_descriptors("CCO")

    assert descriptors is not None
    assert descriptors["MolWt"] > 0
    assert descriptors["LogP"] >= -1.0
    assert descriptors["TPSA"] > 0
    assert descriptors["HBD"] == 1
    assert descriptors["HBA"] == 1
    assert descriptors["RotBonds"] == 0


def test_invalid_smiles_returns_none():
    assert calculate_descriptors("not_a_valid_smiles") is None
    assert calculate_descriptors("") is None
    assert calculate_descriptors(None) is None


def test_expected_descriptor_keys_and_numeric_values():
    descriptors = calculate_descriptors("CCO")

    assert list(descriptors) == [
        "MolWt",
        "LogP",
        "TPSA",
        "HBD",
        "HBA",
        "RotBonds",
    ]
    assert all(pd.notna(value) for value in descriptors.values())
    assert all(value == value for value in descriptors.values())


def test_dataset_processing_success():
    dataset = pd.read_csv("data/esol.csv")
    processed = process_dataset(dataset)

    assert len(processed) == len(dataset)
    assert set(processed.columns) >= {
        "Compound ID",
        "smiles",
        "measured log solubility in mols per litre",
        "MolWt",
        "LogP",
        "TPSA",
        "HBD",
        "HBA",
        "RotBonds",
    }
    assert processed[["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds"]].notna().all().all()
    assert not processed[["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds"]].isin([float("inf"), float("-inf")]).any().any()
    assert pd.to_numeric(processed["measured log solubility in mols per litre"], errors="coerce").notna().all()
