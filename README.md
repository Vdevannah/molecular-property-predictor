# Molecular Property Predictor

**Cheminformatics | RDKit | Python | Molecular Descriptors | Machine Learning**

A Python-based cheminformatics project that converts chemical structures into numerical molecular descriptors to support the development of machine-learning models for predicting aqueous solubility.

The project uses the **ESOL (Delaney) dataset**, containing experimental aqueous solubility measurements for 1,128 molecules.

The dataset contains 1,117 unique canonical structures after stereochemistry-preserving canonicalization. Eleven canonical groups contain duplicates, and six of those groups contain conflicting measured values; those cases are explicitly reported as ambiguous rather than resolved automatically.

## Project Status

| Phase | Description | Status |
|---|---|---|
| 1 | Project setup and dataset acquisition | Complete |
| 2 | RDKit molecular descriptor generation | Complete |
| 3 | Machine-learning model development | Complete |
| 4 | Model evaluation and validation | Complete |
| 5 | FastAPI backend, React/Vite frontend, and scientific validation | Complete |
| 6 | Documentation, testing, and deployment | Planned |

## Project Objective

Aqueous solubility is an important physicochemical property in pharmaceutical discovery and chemical development.

The goal is to develop a machine-learning pipeline that predicts experimental aqueous solubility (**LogS**) from molecular structure.

The completed Phase 2 workflow is:

**SMILES → RDKit molecule → Molecular descriptors → Processed dataset**

The numerical descriptors will serve as input features for regression models developed in Phase 3.

## Dataset

**Source:** ESOL (Delaney) dataset, distributed through DeepChem.

Dataset URL:

[https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv](https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv)

The dataset contains:

- 1,128 chemical compounds
- SMILES molecular representations
- Experimental aqueous solubility measurements
- Additional chemical descriptors and existing ESOL predictions

**Prediction target:** `measured log solubility in mols per litre`

LogS represents the base-10 logarithm of molar aqueous solubility.

The existing ESOL-predicted solubility column is excluded from model inputs to avoid data leakage.

## Phase 2 — Molecular Descriptor Generation

RDKit converts each SMILES string into a molecular object and calculates six descriptors.

| Descriptor | Meaning | Chemical relevance |
|---|---|---|
| MolWt | Molecular weight | Molecular size |
| LogP | Calculated octanol/water partition coefficient | Lipophilicity |
| TPSA | Topological polar surface area | Molecular polarity |
| HBD | Hydrogen-bond donor count | Hydrogen-bonding capacity |
| HBA | Hydrogen-bond acceptor count | Hydrogen-bonding capacity |
| RotBonds | Rotatable bond count | Molecular flexibility |

These descriptors provide a compact numerical representation of molecular structure.

### Implementation

The `src/descriptors.py` module:

1. Loads the ESOL dataset using Pandas.
2. Converts SMILES strings into RDKit molecular objects.
3. Calculates six molecular descriptors.
4. Validates structures and required values.
5. Excludes invalid or incomplete records.
6. Exports the processed dataset to `data/esol_descriptors.csv`.

### Phase 2 Results

| Metric | Result |
|---|---:|
| Input molecules | 1,128 |
| Successfully processed | 1,128 |
| Excluded molecules | 0 |
| Molecular descriptors | 6 |
| Automated tests | 4 passed |
| Missing required values | 0 |

All 1,128 ESOL molecules were successfully processed during Phase 2 validation.

These results describe descriptor generation and data quality, **not machine-learning predictive accuracy**.

## Phase 3 — Machine Learning Model Development

Phase 3 introduces supervised regression using scikit-learn.

The workflow is:

**Molecular descriptors → training/testing split → baseline model → Random Forest → initial evaluation**

### Features and target

The model uses the following ordered feature columns:

- MolWt
- LogP
- TPSA
- HBD
- HBA
- RotBonds

The target is:

- `measured log solubility in mols per litre`

The ESOL-predicted LogS values are not used as model inputs or targets.

### Training and testing split

The dataset is split using `train_test_split`:

- `test_size=0.20`
- `random_state=42`

The training set trains the models, while the held-out test set provides an initial estimate of generalization.

### Models

Two models are trained:

1. **Baseline:** `DummyRegressor(strategy="mean")`
2. **Main model:** `RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)`

The model configuration is intentionally simple. Hyperparameter tuning is not included in this phase.

### Initial evaluation

Predictions for the held-out test set are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

The results are saved to `results/phase3_metrics.json`.

These metrics provide an initial model comparison. They do not constitute comprehensive validation or scientific approval; Phase 4 will address broader evaluation.

### Model artifacts

The trained Random Forest and feature-column order are saved using Joblib:

- `models/solubility_model.joblib`
- `models/feature_columns.joblib`

The feature order is preserved to ensure that later predictions use the correct descriptor sequence.

### Run Phase 3

```bash
python src/train_model.py
```

The script will:

1. Load `data/esol_descriptors.csv`.
2. Validate required columns and finite numeric values.
3. Split the dataset.
4. Train the baseline and Random Forest models.
5. Calculate MAE, RMSE, and R².
6. Save metrics to `results/phase3_metrics.json`.
7. Save model artifacts in `models/`.

### Run automated tests

```bash
python -m pytest -q
```

The tests cover feature selection, target validation, non-overlapping training and testing sets, model predictions, metric calculation, model persistence, invalid input handling, evaluation plots, and scaffold-aware validation.

## Phase 4 — Model Evaluation and Validation

Phase 4 evaluates the saved Random Forest model using both random and scaffold-aware validation strategies.

### Evaluation workflow

```bash
python src/evaluate_model.py
```

The script:

1. Loads the processed ESOL dataset.
2. Validates required feature and target columns.
3. Reproduces the Phase 3 random split.
4. Loads the saved Random Forest and feature-order artifacts.
5. Calculates MAE, RMSE, and R² for the random split.
6. Creates actual-versus-predicted, residual, and feature-importance plots.
7. Builds a scaffold-aware split using RDKit Murcko scaffolds.
8. Trains a Random Forest on the scaffold-separated training set.
9. Evaluates the scaffold split and writes `results/phase4_metrics.json`.

### Evaluation results

The current model produces Phase 3-style metrics on the random split, while the scaffold-aware split provides a more conservative estimate of chemical generalization.

| Validation strategy | MAE | RMSE | R² | Molecules |
|---|---:|---:|---:|---:|
| Random split | 0.573728 | 0.834945 | 0.852515 | 226 |
| Scaffold-disjoint split | 0.615514 | 0.880497 | 0.795425 | 226 |

The corrected scaffold-aware validation uses unique canonical groups for acyclic molecules, so unrelated acyclic structures are not treated as one chemical family.

### Generated outputs

Local evaluation artifacts are written to `results/`:

- `actual_vs_predicted.png`
- `residual_plot.png`
- `feature_importance.png`
- `phase4_metrics.json`

The files are generated locally and excluded from Git.

### Scientific limitations

The evaluation has important limitations:

- It uses a single random split and a single scaffold split.
- ESOL molecules may not represent the full universe of chemical structures.
- Feature importance is not causal evidence.
- Model uncertainty is not explicitly quantified.
- Descriptor-range membership does not provide calibrated confidence or prediction uncertainty.
- The evaluation does not prove suitability for a production or regulatory setting.

The Phase 4 teach-back document provides a more accessible explanation of the evaluation methodology and interpretation. The Phase 5 scientific card is intended to support interpretation, not to transform the model into a validated experimental measurement or uncertainty estimator.

## Phase 5 — Local Development

### Start the FastAPI backend

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8003 --reload
```

The backend is available at http://127.0.0.1:8003 and provides the health check at http://127.0.0.1:8003/api/health.

### Start the React/Vite frontend

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5175
```

The frontend is available at http://localhost:5175.

The React application reads the API base URL from `VITE_API_BASE_URL`. The default is `http://127.0.0.1:8003`.

To use a different backend, start the frontend with:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8004 npm run dev -- --host 127.0.0.1 --port 5175
```

The backend CORS configuration allows requests from `http://localhost:5175` and `http://127.0.0.1:5175`.

### Example molecules and interface

The React interface includes four selectable example molecules:

- Ethanol — `CCO`
- Benzene — `c1ccccc1`
- Aspirin — `CC(=O)Oc1ccccc1C(=O)OC`
- Caffeine — `Cn1c(=O)c2c(ncn2C)n(C)c1=O`

Selecting an example populates the SMILES input. The user can edit the value before submitting it for prediction. The response is always displayed for the submitted SMILES, rather than merely the selected example.

The prediction panel displays the value with the `LogS (log₁₀ mol/L)` label, a short interpretation, and the machine-learning estimate disclaimer. The molecule rendering is responsive and preserves the aspect ratio of the RDKit SVG for small and large structures.

The application preserves the trained Random Forest model, the exact six descriptor order, RDKit validation and canonicalization, the FastAPI endpoint, and the existing scientific disclaimer. The model was trained on the ESOL dataset and predictions may be less reliable outside its training domain.

### Phase 5B — Scientific validation and interpretability

The prediction response extends the model output with scientific metadata that is read only at inference time:

- **Exact ESOL identity matching:** The API matches canonical SMILES exactly and preserves stereochemistry.
- **Experimental reference:** Returns the selected ESOL value, its source, compound ID, absolute error, and phase-split label when available.
- **Ambiguity handling:** Duplicate canonical records with conflicting measurements are reported as ambiguous rather than silently selecting one value.
- **Molecular formula:** RDKit calculates the molecular formula for the canonicalized structure.
- **Applicability domain:** Descriptor values are compared with the Phase 3 training-set minima and maxima. The result is descriptive and does not quantify prediction confidence.
- **Split labeling:** Each result states whether the compound is in the Phase 3 training set, held-out test set, or outside that split.

The UI presents this metadata in a scientific validation card immediately below the prediction summary. It still does not mutate, retrain, or replace the saved Random Forest model.

### Run automated tests

```bash
python -m pytest -q
cd frontend
npm test -- --run
npm run build
```

The model artifacts and generated local outputs remain unchanged by the web application.

## Technology Stack

- **Python 3.12** — Programming language
- **RDKit** — Molecular structure parsing and descriptor calculation
- **Pandas** — Dataset processing and validation
- **NumPy** — Numerical operations
- **scikit-learn** — Supervised regression and model evaluation
- **FastAPI** — Prediction API
- **React and Vite** — Interactive frontend
- **pytest** — Automated testing
- **Matplotlib** — Model evaluation plots
- **Joblib** — Model and feature serialization

## Project Structure

```text
molecular-property-predictor/
├── data/
│   ├── esol.csv                 # Local only; Git ignored
│   └── esol_descriptors.csv     # Generated; Git ignored
├── src/
│   ├── descriptors.py
│   └── train_model.py
├── tests/
│   ├── test_descriptors.py
│   └── test_train_model.py
├── docs/
│   ├── PHASE2_TEACHBACK.md
│   └── PHASE3_TEACHBACK.md
├── models/
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Vdevannah/molecular-property-predictor.git
cd molecular-property-predictor
```

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Download the Dataset

CSV files are intentionally excluded from GitHub.

Create the local data directory and download the ESOL dataset:

```bash
mkdir -p data

curl -L \
  https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv \
  -o data/esol.csv
```

Verify the dataset:

```bash
python -c "import pandas as pd; print(pd.read_csv('data/esol.csv').shape)"
```

Expected shape:

```text
(1128, 10)
```

## Generate Molecular Descriptors

Run:

```bash
python src/descriptors.py
```

The script generates:

`data/esol_descriptors.csv`

The output contains the compound ID, SMILES, measured LogS, and six calculated descriptors.

## Data and Git Policy

CSV files are not tracked in this repository.

The `.gitignore` should include:

```gitignore
*.csv
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
.DS_Store
```

Datasets are downloaded or regenerated locally using the documented commands.

## Scientific Considerations

Molecular descriptors summarize selected structural and physicochemical characteristics, but they do not fully represent molecular behavior.

Aqueous solubility is influenced by factors including lipophilicity, polarity, hydrogen bonding, crystal packing, and ionization.

The planned machine-learning model will learn statistical relationships between the selected descriptors and measured LogS.

Future evaluation will examine predictive accuracy and limitations, including how well the model generalizes to molecules outside the training set.

## Next Steps — Phase 4

Phase 4 will introduce comprehensive model evaluation and validation.

Planned tasks include:

- Cross-validation
- Multiple regression models
- Model comparison
- Hyperparameter tuning
- Feature importance analysis
- Training and testing diagnostics
- External validation
- Uncertainty estimation

## Author

**Vijayarajan Devannah, Ph.D.**

Organic Chemistry | Cheminformatics | Python | Data Engineering & Analytics

[GitHub Repository](https://github.com/Vdevannah/molecular-property-predictor)

---

**Current milestone:** Phase 3 development is implemented; Phase 4 validation is the next development phase.
