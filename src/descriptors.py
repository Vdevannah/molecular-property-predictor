"""Utilities for generating molecular descriptors with RDKit."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

DESCRIPTOR_FIELDS = [
    "MolWt",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotBonds",
]


def calculate_descriptors(smiles: str | None) -> dict[str, float] | None:
    """Return six RDKit descriptors for a valid SMILES string.

    Returns ``None`` for missing or invalid SMILES. The function is intentionally
    tolerant: downstream code can skip invalid molecules without interrupting the
    entire dataset processing pipeline.
    """
    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        return None

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None

    descriptors = {
        "MolWt": Descriptors.MolWt(molecule),
        "LogP": Descriptors.MolLogP(molecule),
        "TPSA": Descriptors.TPSA(molecule),
        "HBD": Descriptors.NumHDonors(molecule),
        "HBA": Descriptors.NumHAcceptors(molecule),
        "RotBonds": Descriptors.NumRotatableBonds(molecule),
    }

    if not all(pd.notna(value) for value in descriptors.values()):
        return None

    if any(not pd.notna(value) or not float("-inf") < float(value) < float("inf") for value in descriptors.values()):
        return None

    return descriptors


def process_dataset(
    dataset: pd.DataFrame,
    output_path: str | Path | None = None,
    report: bool = True,
) -> pd.DataFrame:
    """Filter, describe, and save a valid ESOL dataset with RDKit descriptors."""
    required_columns = {
        "Compound ID",
        "smiles",
        "measured log solubility in mols per litre",
    }
    missing_columns = sorted(required_columns - set(dataset.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    processed_rows: list[dict[str, float | str]] = []
    exclusion_counts: Counter[str] = Counter()

    for _, row in dataset.iterrows():
        compound_id = row["Compound ID"]
        smiles = row["smiles"]
        target = row["measured log solubility in mols per litre"]

        if pd.isna(compound_id):
            exclusion_counts["missing Compound ID"] += 1
            continue

        if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
            exclusion_counts["missing or empty SMILES"] += 1
            continue

        numeric_target = pd.to_numeric(target, errors="coerce")
        if pd.isna(numeric_target):
            reason = "missing target LogS" if pd.isna(target) else "non-numeric target LogS"
            exclusion_counts[reason] += 1
            continue

        descriptors = calculate_descriptors(smiles)
        if descriptors is None:
            exclusion_counts["invalid SMILES structure"] += 1
            continue

        processed_rows.append(
            {
                "Compound ID": compound_id,
                "smiles": smiles,
                "measured log solubility in mols per litre": float(numeric_target),
                **descriptors,
            }
        )

    processed = pd.DataFrame(
        processed_rows,
        columns=[
            "Compound ID",
            "smiles",
            "measured log solubility in mols per litre",
            *DESCRIPTOR_FIELDS,
        ],
    )

    if not processed.empty:
        numeric_columns = [
            "measured log solubility in mols per litre",
            *DESCRIPTOR_FIELDS,
        ]
        processed[numeric_columns] = processed[numeric_columns].apply(pd.to_numeric, errors="raise")
        finite_mask = processed[numeric_columns].map(lambda value: float(value) == value and float("-inf") < float(value) < float("inf"))
        if not finite_mask.all().all():
            raise ValueError("Processed dataset contains non-finite descriptor or target values.")

    if report:
        total_rows = len(dataset)
        retained_rows = len(processed)
        excluded_rows = total_rows - retained_rows
        print(f"Input molecules: {total_rows}")
        print(f"Successfully processed: {retained_rows}")
        print(f"Excluded: {excluded_rows}")
        for reason, count in sorted(exclusion_counts.items()):
            print(f"  {reason}: {count}")

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(output_file, index=False)

    return processed


def main() -> None:
    """Generate descriptor features for the packaged ESOL dataset."""
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "esol.csv"
    output_path = project_root / "data" / "esol_descriptors.csv"

    dataset = pd.read_csv(input_path)
    processed = process_dataset(dataset, output_path=output_path, report=True)

    print("Processed dataset summary:")
    print(processed.head().to_string(index=False))
    print(f"Columns: {processed.columns.tolist()}")
    print(f"Rows: {len(processed)}")


if __name__ == "__main__":
    main()
